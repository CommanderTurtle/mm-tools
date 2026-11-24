"""Talker stage (LLM_AR). vLLM only keeps the beat: the model is an attention-free shell, and every decode step runs one
CUDA-Graph step of the Sophon E2E talker engine for all sessions. One vector frame per request per step goes to the decoder
stage (float32 bit-cast into codes.audio)."""
import os
import time
from typing import Any

import torch
import torch.nn as nn
from vllm.logger import init_logger
from vllm_omni.model_executor.models.output_templates import OmniOutput

from .bridge220 import _f32
from . import segreg
from . import clone

logger = init_logger("vllm_omni.sophon_realtime.talker")
FRAME, WAIT, STOP = 1000, 1001, 151645
LANGS = ("zh", "en", "ja")
RENDER_EVERY = int(os.environ.get("S220_RENDER_EVERY", "2"))
SLOTS = int(os.environ.get("SOPHON_E2E_SLOTS", "16"))
os.environ.setdefault("S220_REFRESH_MIN", "3")       # reload the text memory after 3 new characters (the thinker and the talker start at the same time here)
PROF = os.environ.get("SOPHON_E2E_PROF") in ("1", "2"); PROF2 = os.environ.get("SOPHON_E2E_PROF") == "2"   # 2: also time the parts of a step (adds GPU syncs)
_PT = {"refresh": 0.0, "step": 0.0, "adapter": 0.0, "n_refresh": 0, "n_step": 0, "n_adp": 0, "n_adp_items": 0}


def _pt(key, t0, n=None, cnt=1):
    if PROF2:
        torch.cuda.synchronize(); _PT[key] += time.perf_counter() - t0
        if n: _PT[n] += cnt
AHEAD = float(os.environ.get("SOPHON_E2E_AHEAD", "2.0"))      # a reply is handed to the decoder at most this many seconds ahead of real-time playback
URGENT = float(os.environ.get("SOPHON_E2E_URGENT", "1.0"))    # a reply with less than this much audio ahead of playback is stepped without waiting
STEP_GAP = float(os.environ.get("SOPHON_E2E_STEP_GAP", "0"))   # otherwise the engine steps at most this often (one step = 0.154 s of audio for every reply that wants it); 0 = off (default: measured no gain)


class _St:
    __slots__ = ("sess", "slot", "frames", "R", "chunks", "flushed", "nchunk", "tok", "t0", "text", "t_f1", "t_c1", "t_e0", "sec",
                 "fin", "fed", "seg_start", "seg_end", "seg_f0")

    def __init__(self, sess, R):
        self.sess = sess; self.slot = None; self.frames = []; self.R = R; self.chunks = []; self.flushed = False; self.nchunk = 0
        self.tok = WAIT; self.t0 = time.perf_counter(); self.text = ""; self.t_f1 = None; self.t_c1 = None; self.t_e0 = None; self.sec = 0.0
        # long replies in segments (segreg.find_cut): each segment is a fresh engine session on the rest of the text; the frames,
        # the adapter / render state and the decoder stream continue, so the audio stays one stream
        self.fin = False; self.fed = 0; self.seg_start = 0; self.seg_end = None; self.seg_f0 = 0


class SophonE2ETalker(nn.Module):
    is_attention_free = True
    prefer_model_sampler = True
    requires_raw_input_tokens = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.H = int(getattr(self.config, "hidden_size", 16)); self.V = int(getattr(self.config, "vocab_size", 151936))
        self.have_multimodal_outputs = True
        self.has_preprocess = True
        self.has_postprocess = False
        self.omni_pooler_payload_include_hidden = False
        self.dummy = nn.Parameter(torch.zeros(1))
        self._st: dict[str, _St] = {}
        self._order: list[tuple[str | None, int]] = []
        self._batch_req_ids: list[str] = []
        self._last_mm = None
        from .bridge220 import ROW, TalkerCore, engine_module, pack_row
        self.T = engine_module(); self.core = TalkerCore(SLOTS, logger); self.ROW = ROW; self._pack = pack_row; self._tk = 0
        self.cap = min(self.core.eng.Tmax - 4, self.T.FRAME_CAP); self.LA = int(self.T.S.LA)

    # ---- vLLM model interface
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        return torch.zeros((input_ids.shape[0], self.H), device=input_ids.device, dtype=self.dummy.dtype)

    def _sync_text(self, st: _St, info: dict, input_ids: torch.Tensor, is_prefill: bool):
        ids = info.get("ids"); ids = ids.get("output") if isinstance(ids, dict) else None
        meta = info.get("meta") or {}
        if ids is not None:                                # thinker -> talker: the reply text so far, as code points
            if isinstance(ids, torch.Tensor): ids = ids.reshape(-1).tolist()
            text = "".join(chr(int(c)) for c in ids)
            fin = meta.get("finished", False) if isinstance(meta, dict) else False
            fin = bool(fin.reshape(-1)[0].item()) if isinstance(fin, torch.Tensor) and fin.numel() else bool(fin)
        elif is_prefill and not st.text:                   # standalone (no thinker): the prompt is the text
            text = self.core.tok.decode(input_ids.tolist(), skip_special_tokens=True); fin = True
            if text[:4] in ("<zh>", "<en>", "<ja>"):       # standalone test convention: language prefix
                st.sess.lang = text[1:3]; text = text[4:]
            st.sess.add_text(text); st.text = "\0"; st.sess.finish(); st.text = text
            return
        else:
            return
        if len(text) > len(st.text):
            st.text = text
        st.fin = st.fin or fin
        self._feed(st)

    def _feed(self, st: _St):
        """Give the engine session the text of the current segment that has arrived; end the session at the cut / reply end."""
        if segreg.SEEK_S > 0 and st.seg_end is None:
            st.seg_end = segreg.find_cut(st.text, st.seg_start, st.fed, lambda c: c, st.fin)
        lim = len(st.text) if st.seg_end is None else st.seg_end
        if lim > st.fed:
            st.sess.add_text(st.text[st.fed:lim]); st.fed = lim
        if not st.sess.final and ((st.seg_end is not None and st.fed >= st.seg_end) or (st.seg_end is None and st.fin)):
            st.sess.finish()

    def _next_segment(self, st: _St) -> bool:
        """The engine session ended: start the next segment of a long reply (True), or the reply is over (False)."""
        if st.seg_end is None or (st.fin and st.seg_end >= len(st.text)):
            return False
        st.sess = self.T.Session(self.core, st.sess.lang, "", None)
        st.seg_start = st.fed = st.seg_end; st.seg_end = None; st.seg_f0 = len(st.frames)
        self._feed(st)
        logger.info("talker: new segment at char %d of %d (%d frames so far)", st.seg_start, len(st.text), len(st.frames))
        return True

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any):
        add = info.get("additional_information")
        if isinstance(add, dict):
            info = {**add, **{k: v for k, v in info.items() if k != "additional_information"}}
        rid = info.get("request_id"); span = int(input_ids.shape[0])
        emb = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
        if rid is not None:
            st = self._st.get(rid)
            is_prefill = bool(info.get("_omni_is_prefill", span > 1))
            if st is None:
                lang = info.get("language"); lang = lang[0] if isinstance(lang, (list, tuple)) and lang else lang
                lang = str(lang or "zh").lower()[:2]
                R = self.T.Render.__new__(self.T.Render); R.C = self.core; R.u = 0; R.pend = None     # adapter bookkeeping only
                st = self._st[rid] = _St(self.T.Session(self.core, lang if lang in LANGS else "zh", "", None), R)
                spk = info.get("speaker"); clone.put_voice(rid, spk[0] if isinstance(spk, (list, tuple)) and spk else spk)   # for the decoder stage
            try:
                self._sync_text(st, info, input_ids, is_prefill)
            except Exception as e:  # noqa: BLE001
                logger.warning("talker text sync failed for %s: %r", rid, e)
        self._order.append((rid, span))
        return input_ids, emb, {}

    def _tick(self):
        """One engine step for every live slot (if any session needs more audio), then the adapter for the sessions that got enough new frames.
        vLLM may schedule only some of the requests in a given decode step, so this works on all open sessions, not on the step's batch."""
        T = self.T; C = self.core; E = C.eng; now = time.perf_counter()
        if not hasattr(self, "_t_step"): self._t_step = 0.0
        sts = list(self._st.items())
        for r, st in sts:                                   # admit
            if st.slot is None and not st.sess.talk_done and st.sess.ready():
                free = [j for j, x in enumerate(C.slots) if x is None]
                if not free: break
                j = free[0]; C.slots[j] = r; st.slot = j; E.reset_slot(j); st.sess.loaded = -1
        act = [(st.slot, st) for r, st in sts if st.slot is not None and not st.sess.talk_done]
        grew = []
        if act:
            cur = E.cur.tolist()
            for j, st in act:
                try:
                    _v = st.sess.loaded; _tr = time.perf_counter(); st.sess.refresh(C, j, cur[j])
                    if st.sess.loaded != _v: _pt("refresh", _tr, "n_refresh")
                except Exception as e:  # noqa: BLE001  -- one bad session must not take the stage down
                    logger.warning("talker session error: %r", e); st.sess.talk_done = True; E.done[j] = True; C.slots[j] = None; st.slot = None
            act = [(j, st) for j, st in act if st.slot is not None]
            # A session is held for this step (its engine state is put back afterwards) when
            #  - the talker has caught up with the text the thinker has written so far (it must not read past it), or
            #  - it already has AHEAD + 1 s more audio than real-time playback needs.
            go, hold = [], []; urgent = False
            for j, st in act:
                starving = (not st.sess.final) and (st.sess.ntok - cur[j] <= self.LA)      # its lookahead would reach past the loaded text
                lead = (st.sec + 0.02 * sum(int(c[0].shape[2]) for c in st.chunks) - (now - st.t_e0)) if st.t_e0 is not None else -1.0   # audio already rendered (sent + queued) vs playback time
                ahead = AHEAD > 0 and st.t_e0 is not None and lead >= AHEAD + 1.0
                (hold if (starving or ahead) else go).append((j, st))
                if not (starving or ahead) and lead < URGENT: urgent = True
            # One engine step costs the same for 1 or 16 sessions. A reply that is just starting (or close to running dry) steps at once; replies that
            # are comfortably ahead of playback wait and step together at most every STEP_GAP seconds, instead of one step per reply.
            if go and not urgent and now - self._t_step < STEP_GAP:
                go = []
            if go:
                self._t_step = now
                if hold:
                    idx = torch.tensor([j for j, _ in hold], device=E.pos.device); state = (E.pos, E.fprev, E.first, E.cur, E.cf, E.stall, E.lastc, E.tend, E.done)
                    snap = [x[idx].clone() for x in state]
                _ts = time.perf_counter(); E.step(); f = E.fout.clone(); al = (~E.done).tolist(); _pt("step", _ts, "n_step")
                if hold:
                    for x, v in zip(state, snap): x[idx] = v
                for j, st in go:
                    if al[j]:
                        st.frames.append(f[j]); grew.append(st)
                        if st.t_f1 is None: st.t_f1 = time.perf_counter() - st.t0
                        if len(st.frames) - st.seg_f0 >= self.cap:
                            st.sess.talk_done = True; E.done[j] = True; C.slots[j] = None; st.slot = None
                    else:
                        st.sess.talk_done = True; C.slots[j] = None; st.slot = None
                    if st.sess.talk_done:
                        self._next_segment(st)
        adp = []
        for r, st in sts:
            K = len(st.frames)
            if st.sess.talk_done:
                if st.flushed: continue
                st.flushed = True
                while True:
                    pl = st.R.ad_need(K, True)
                    if pl is None: break
                    st.R.ad_apply(T.ad_batch(C, [torch.stack(st.frames[pl[0]:pl[1]])[None].float()])[0], pl)
                while True:
                    b = st.R.take(True, K)
                    if b is None: break
                    st.chunks.append(b)
            elif st in grew and (K % RENDER_EVERY == 0 or st.nchunk + len(st.chunks) == 0):
                pl = st.R.ad_need(K, False)
                if pl is not None: adp.append((st, pl))
        if adp:
            _ta = time.perf_counter(); outs = T.ad_batch(C, [torch.stack(st.frames[pl[0]:pl[1]])[None].float() for st, pl in adp]); _pt("adapter", _ta, "n_adp"); _PT["n_adp_items"] += len(adp)
            for (st, pl), mu in zip(adp, outs):
                st.R.ad_apply(mu, pl); b = st.R.take(False, len(st.frames))
                if b is not None: st.chunks.append(b)
        return bool(grew)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs: Any):
        n = int(inputs_embeds.shape[0]) if inputs_embeds is not None else int(input_ids.shape[0])
        dev = self.dummy.device
        order, self._order = self._order, []
        _t0 = time.perf_counter()
        with _f32(), torch.inference_mode(False), torch.no_grad():
            rows = torch.zeros((n, self.ROW), device=dev, dtype=torch.long); valid = torch.zeros((n,), device=dev, dtype=torch.int8)
            if not getattr(self, "_flags_logged", False):
                self._flags_logged = True
                B = torch.backends
                logger.info("E2E forward flags: autocast=%s(%s) inference=%s grad=%s matmul_tf32=%s cudnn_tf32=%s cudnn.enabled=%s det=%s bench=%s f32prec=%s flash_sdp=%s mem_sdp=%s math_sdp=%s cudnn_sdp=%s mha_fast=%s default=%s",
                            torch.is_autocast_enabled(), torch.get_autocast_dtype("cuda"), torch.is_inference_mode_enabled(), torch.is_grad_enabled(), B.cuda.matmul.allow_tf32, B.cudnn.allow_tf32, B.cudnn.enabled,
                            B.cudnn.deterministic, B.cudnn.benchmark, torch.get_float32_matmul_precision(), B.cuda.flash_sdp_enabled(), B.cuda.mem_efficient_sdp_enabled(), B.cuda.math_sdp_enabled(),
                            B.cuda.cudnn_sdp_enabled(), B.mha.get_fastpath_enabled(), torch.get_default_dtype())
            _tk = time.perf_counter(); stepped = self._tick()
            if PROF2: torch.cuda.synchronize(); _PT["tick"] = _PT.get("tick", 0.0) + time.perf_counter() - _tk
            row = 0; now = time.perf_counter(); due = 1.0
            for rid, span in order:
                st = self._st.get(rid) if rid is not None else None
                last = row + span - 1; row += span
                if st is None or last >= n:
                    continue
                lead = (st.sec - (now - st.t_e0)) if st.t_e0 is not None else -1.0
                if st.chunks and (AHEAD <= 0 or lead < AHEAD):   # one decoder chunk per step, paced to real time
                    mu, fin = st.chunks.pop(0)
                    rows[last] = self._pack(mu, fin, st.sess.lang); valid[last] = 1; st.tok = FRAME; st.nchunk += 1
                    if st.t_c1 is None: st.t_c1 = now - st.t0; st.t_e0 = now
                    st.sec += 0.02 * int(mu.shape[2])
                elif st.chunks:
                    st.tok = WAIT; due = min(due, lead - AHEAD)
                elif st.sess.talk_done and st.flushed:
                    st.tok = STOP
                else:
                    st.tok = WAIT
            if order and not stepped and not valid.any() and not any(self._st[r].tok == STOP for r, _ in order if r in self._st):
                time.sleep(min(0.01, max(0.002, due)))       # nobody needs the engine now (waiting for text, a free slot, or for playback to catch up): do not spin
            if PROF:
                now = time.perf_counter(); P = self._prof = getattr(self, "_prof", [0, 0, 0.0, 0.0, 0.0]); P[0] += 1; P[1] += len(order); P[2] += now - _t0; P[3] += (_t0 - P[4]) if P[4] else 0.0; P[4] = now
                if P[0] % 300 == 0:
                    logger.info("talker prof: %d steps, %.1f requests/step, %.1f ms/step inside, %.1f ms between steps", P[0], P[1] / P[0], 1000 * P[2] / P[0], 1000 * P[3] / P[0]); self._prof = [0, 0, 0.0, 0.0, now]
                    if PROF2:
                        logger.info("talker prof parts (ms total over 300 steps): tick %.0f | text load %.0f (%d loads) | engine step %.0f (%d steps) | adapter %.0f (%d calls, %d items)", 1000 * _PT.get("tick", 0.0),
                                    1000 * _PT["refresh"], _PT["n_refresh"], 1000 * _PT["step"], _PT["n_step"], 1000 * _PT["adapter"], _PT["n_adp"], _PT["n_adp_items"])
                        for k in list(_PT): _PT[k] = 0 if k.startswith("n_") else 0.0
            self._last_mm = {"codes": {"audio": rows}, "meta": {"codec_frame_valid": valid}}
        return torch.zeros((n, self.H), device=dev, dtype=self.dummy.dtype)

    def make_omni_output(self, model_outputs, **kwargs: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        return OmniOutput(text_hidden_states=model_outputs, multimodal_outputs=self._last_mm or {})

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        return torch.zeros((hidden_states.shape[0], self.V), device=hidden_states.device, dtype=torch.float32)

    def set_batch_req_ids(self, req_ids) -> None:
        self._batch_req_ids = list(req_ids)

    def sample(self, logits, sampling_metadata):
        from vllm.v1.outputs import SamplerOutput
        rows = int(logits.shape[0])
        rids = self._batch_req_ids if len(self._batch_req_ids) == rows else [None] * rows
        out = [(self._st[r].tok if r in self._st else STOP) for r in rids]
        return SamplerOutput(sampled_token_ids=torch.tensor(out, device=logits.device, dtype=torch.int32).unsqueeze(-1), logprobs_tensors=None)

    def on_requests_finished(self, finished_req_ids) -> None:
        for r in finished_req_ids:
            st = self._st.pop(r, None)
            if st is None:
                continue
            if st.slot is not None:
                with _f32(), torch.inference_mode(False), torch.no_grad():
                    self.core.eng.done[st.slot] = True
                self.core.slots[st.slot] = None
            if os.environ.get("SOPHON_E2E_DUMP") and st.frames: torch.save(torch.stack(st.frames).float().cpu(), os.path.join(os.environ["SOPHON_E2E_DUMP"], "fr_%s.pt" % abs(hash(st.text))))
            logger.info("talker done %s frames=%d chunks=%d first_frame=%.3fs first_chunk=%.3fs total=%.2fs text=%r", str(r)[-10:], len(st.frames), st.nchunk, st.t_f1 or -1, st.t_c1 or -1, time.perf_counter() - st.t0, st.text[:60])

    def load_weights(self, weights) -> set[str]:
        for _ in weights:
            pass
        return {"dummy"}
