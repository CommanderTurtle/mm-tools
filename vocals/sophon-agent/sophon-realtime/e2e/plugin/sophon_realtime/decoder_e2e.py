"""Decoder stage (LLM_GENERATION): chunks of talker frames -> adapter -> 2-step DiT (batched across requests) -> HiFT -> 24 kHz audio.
Frames arrive as input_ids (float32 bit-cast, D values per frame); the first row of a request is a header [MAGIC, lang id]."""
import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .bridge220 import _f32
from . import clone

logger = init_logger("vllm_omni.sophon_realtime.decoder")
SR = 24000
PROF = os.environ.get("SOPHON_E2E_PROF") == "1"


class _Rq:
    __slots__ = ("R", "fin", "n", "sec")

    def __init__(self):
        self.R = None; self.fin = False; self.n = 0; self.sec = 0.0


class SophonE2EDecoder(nn.Module):
    is_attention_free = True
    input_modalities = "audio"
    requires_request_ids = True
    supports_native_payload_input = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True
        self.dummy = nn.Parameter(torch.zeros(1))
        self._rq: dict[str, _Rq] = {}
        from .bridge220 import ROW, DecoderCore, engine_module, unpack_row
        self.T = engine_module(); self.core = DecoderCore(logger); self.ROW = ROW; self._unpack = unpack_row

    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def get_dummy_runtime_additional_information(self, num_reqs: int):
        return [{"meta": {"request_id": f"__e2e_dummy__{i}", "finished": True}} for i in range(max(0, int(num_reqs)))]

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        return None

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None, inputs_embeds=None,
                runtime_additional_information=None, model_intermediate_buffer=None, **kwargs: Any) -> OmniOutput:
        T = self.T; C = self.core
        if PROF:
            import time as _t
            _t0 = _t.perf_counter()
            if not hasattr(self, "_prof"): self._prof = [0, 0, 0.0, 0.0, 0.0, 0]
        if not getattr(self, "_flags_logged", False):
            self._flags_logged = True; B = torch.backends
            logger.info("E2E decoder forward flags: autocast=%s(%s) inference=%s grad=%s default=%s cudnn_sdp=%s bench=%s matmul_tf32=%s", torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda"),
                        torch.is_inference_mode_enabled(), torch.is_grad_enabled(), torch.get_default_dtype(), B.cuda.cudnn_sdp_enabled(), B.cudnn.benchmark, B.cuda.matmul.allow_tf32)
        infos = runtime_additional_information or model_intermediate_buffer or []
        rids = kwargs.get("request_ids") or []
        ids = input_ids.reshape(-1) if input_ids is not None else torch.zeros(0, dtype=torch.long)
        counts = kwargs.get("seq_token_counts")
        counts = [int(c) for c in counts] if counts is not None else [int(ids.numel())]
        n = max(len(infos), len(rids), len(counts), 1)
        empty = torch.zeros((0,), dtype=torch.float32); sr = torch.tensor(SR, dtype=torch.int32)
        audios = [empty] * n; items = []; ended = []; off = 0; outs: dict[int, list] = {}
        with _f32(), torch.inference_mode(False), torch.no_grad():
            for i, c in enumerate(counts):
                seg = ids[off: off + c]; off += c
                info = infos[i] if i < len(infos) and isinstance(infos[i], dict) else {}
                meta = info.get("meta") or {}
                rid = str(rids[i]) if i < len(rids) else str(meta.get("request_id"))
                if rid.startswith("__e2e_dummy__"):
                    continue
                fin = meta.get("finished", False)
                fin = bool(fin.reshape(-1)[0].item()) if isinstance(fin, torch.Tensor) and fin.numel() else bool(fin)
                x = self._rq.get(rid)
                if x is None: x = self._rq[rid] = _Rq()
                if c >= self.ROW and c % self.ROW == 0:            # one or several chunks of this request, in order
                    rows = seg.to(self.dummy.device).reshape(-1, self.ROW); q = []
                    for k in range(rows.shape[0]):
                        mu, last, lang = self._unpack(rows[k])
                        if x.R is None:
                            x.R = T.Render(C, lang); x.R.pend = torch.zeros(1, 80, 0, device=self.dummy.device)
                            vp = clone.prompt(C, clone.take_voice(meta.get("request_id"), rid), lang, self.dummy.device, log=logger)
                            if vp is not None:                     # cloned voice: same decoder, the clip as its reference prompt
                                old = x.R.v2
                                x.R.v2 = type(old)(C.dec, vp[0], vp[1], vp[2], ch=old.ch, left=old.left, nt=old.nt, cfg=old.cfg)
                        if mu.shape[2] > 0: q.append((mu, last)); x.n += 1
                    if q: items.append((i, x, q))
                if fin:
                    ended.append((rid, x)); clone.drop_voice(meta.get("request_id"), rid)
            k = 0
            while True:                                            # k-th chunk of every request in one batch
                reqs = [(i, x, q[k][0], q[k][1]) for i, x, q in items if k < len(q)]
                if not reqs: break
                k += 1
                try:
                    mels = T.e2b_all(C.eb, [(x.R.v2, mu) for _, x, mu, _ in reqs]); hr = []
                    for (_, x, mu, last), mel in zip(reqs, mels):
                        ws, dn = x.R.after_mel(mel, last); hr.append((x.R, ws, dn))
                    wavs = T.EN._hift_batch(C.hift, hr)
                    for (i, x, mu, last), w, (_, ws, dn) in zip(reqs, wavs, hr):
                        wn = x.R.after_wav(w, ws)
                        if len(wn):
                            x.sec += len(wn) / SR; outs.setdefault(i, []).append(np.clip(wn, -1, 1).astype(np.float32))
                except Exception as e:  # noqa: BLE001  -- keep the stage alive; these requests lose this chunk
                    logger.warning("decoder render error (%d requests): %r", len(reqs), e)
            for i, ws_ in outs.items():
                audios[i] = torch.from_numpy(np.concatenate(ws_))
            reqs = [None] * sum(len(q) for _, _, q in items)
            for rid, x in ended:
                self._rq.pop(rid, None)
                logger.info("decoder done %s chunks=%d audio=%.2fs", rid[-10:], x.n, x.sec)
        if PROF:
            import time as _t
            now = _t.perf_counter(); P = self._prof; P[0] += 1; P[1] += len(reqs); P[2] += now - _t0; P[3] += (_t0 - P[4]) if P[4] else 0.0; P[4] = now; P[5] = max(P[5], len(self._rq))
            if P[0] % 100 == 0:
                logger.info("decoder prof: %d calls, %.2f chunks/call, %.1f ms/call inside, %.1f ms between calls, max open requests %d, vocoder graphs %d hit %d miss %d", P[0], P[1] / P[0], 1000 * P[2] / P[0], 1000 * P[3] / P[0], P[5], len(C.hg.g), C.hg.hit, C.hg.miss); self._prof = [0, 0, 0.0, 0.0, now, 0]
        return OmniOutput(text_hidden_states=None, multimodal_outputs={"model_outputs": audios, "sr": [sr] * n})

    def make_omni_output(self, model_outputs, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        raise TypeError(type(model_outputs))

    def on_requests_finished(self, finished_req_ids) -> None:
        for r in finished_req_ids:
            self._rq.pop(str(r), None)

    def load_weights(self, weights) -> set[str]:
        for _ in weights:
            pass
        return {"dummy"}
