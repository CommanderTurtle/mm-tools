"""tag-tts (CosyVoice3) talker for Sophon-Realtime-32B-Tag, with *bistream* text input inside vLLM-Omni.

vLLM-Omni's native CosyVoice3 talker only accepts the whole text up front (unistream prompt). Here the thinker's text arrives
step by step (cumulative CosyVoice token ids in ids.output, via sophon_realtime.proc) and we replay the official
CosyVoice3LM.inference_bistream schedule (the one tag-tts C3 was trained / evaluated with in cosy_server.py):

  prompt   = [sos] + "You are a helpful assistant.<|endofprompt|>"           (no prompt speech tokens in the LM)
  loop     : model emits fill_token -> feed the next 5 text tokens -> decode 15 speech tokens -> forced fill_token -> ...
  final    : thinker finished -> feed the remaining text + task_id -> decode until eos
  sampling : RAS (top_p 0.8, top_k 25, win 10, tau 0.1); in the bistream phase every special token except fill is masked
             (the tag-line fix for "should not get token 6562"), in the final phase only eos is allowed.

vLLM decodes one position per step, so a multi-token input (5 text tokens, or text + task_id) is fed one embedding per
step; the samples of the intermediate positions are discarded (emitted as fill_token, which code2wav ignores). This is
mathematically identical to feeding them in one forward (causal attention, same KV).

Flow / HiFT stages are vLLM-Omni's native code2wav; the voice prompt (flow prompt tokens trimmed to a hop multiple, mel,
speaker embedding, precomputed by tools/mk_voices.py with the official frontend) is emitted from the talker so the native
async-chunk processor picks it up. Flow steps: SOPHON_FLOW_STEPS (default 10); CFG rate: SOPHON_FLOW_CFG (default: model)."""
from __future__ import annotations

import os
from typing import Any

import torch
from vllm.logger import init_logger
from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer

logger = init_logger("vllm_omni.sophon_realtime.talker_tag")

LM_PROMPT = "You are a helpful assistant.<|endofprompt|>"
FLOW_STEPS = int(os.environ.get("SOPHON_FLOW_STEPS", "10"))
FLOW_CFG = os.environ.get("SOPHON_FLOW_CFG")            # None = keep the model's inference_cfg_rate (0.7)
MAX_STATES = 512


def _scalar_true(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.reshape(-1)[0].item()) if x.numel() else False
    if isinstance(x, (list, tuple)):
        return bool(x and _scalar_true(x[0]))
    return bool(x)


def _first(x, default=None):
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    return default if x is None else x


DEBUG = os.environ.get("SOPHON_TAG_DEBUG", "0") == "1"
# Long replies are spoken in segments with a fresh talker context each (see segreg), so a reply is spoken in full however
# long the text is (the thinker's max_tokens bounds it). SOPHON_MAX_SPEECH_S > 0 ends a reply after that many seconds.
from . import segreg
MAX_SPEECH_TOK = int(float(os.environ.get("SOPHON_MAX_SPEECH_S", "0")) * 25)
SEEK_TOK = int(segreg.SEEK_S * 25)
# Safety net: a segment never speaks longer than this (the talker degrades after ~55 s of context).
SEG_MAX_TOK = int(float(os.environ.get("SOPHON_SEG_MAX_S", "50")) * 25)


class _St:
    __slots__ = ("t0", "nspeech", "queue", "out", "next_fill", "text", "pos", "fin", "phase", "last_fed", "voice", "steps", "dummy", "starved", "done_eos", "last_tok", "seg_speech", "seg_end", "restart", "seg_start")

    def __init__(self, voice):
        self.queue: list[torch.Tensor] = []   # pending input embeddings (fed one per step)
        self.out: list[int] = []              # official out_tokens (real samples incl. fill)
        self.next_fill = 15                   # (0 // 15 + 1) * 15 - 0 with no prompt speech tokens
        self.text: list[int] = []             # all CosyVoice text ids received so far
        self.pos = 0                          # text ids consumed
        self.fin = False                      # thinker finished
        self.phase = "bi"                     # "bi" | "final"
        self.last_fed: torch.Tensor | None = None
        self.voice = voice
        self.steps = 0
        self.dummy = False                    # this step's sample is an intermediate position -> discard
        self.starved = 0
        self.done_eos = False
        self.last_tok = None
        self.t0 = __import__("time").monotonic()
        self.nspeech = 0
        self.seg_speech = 0                   # speech tokens in the current segment
        self.seg_end = None                   # text index where the current segment ends (None: not decided yet)
        self.restart = False                  # segment spoken, waiting for the scheduler to restart the request
        self.seg_start = 0                    # text index where the current segment starts

    def new_segment(self):
        """Fresh LM state for the next segment (text, voice and totals are kept)."""
        self.queue = []; self.out = []; self.next_fill = 15; self.phase = "bi"; self.last_fed = None
        self.dummy = False; self.starved = 0; self.done_eos = False; self.last_tok = None
        self.pos = self.seg_end if self.seg_end is not None else self.pos
        self.seg_speech = 0; self.seg_end = None; self.restart = False; self.seg_start = self.pos


class SophonTagTalker(CosyVoice3Model):
    has_preprocess = True

    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._st: dict[str, _St] = {}
        self._order: list[str] = []
        self._prefill_rows: set[str] = set()
        self._batch_req_ids: list[str] = []
        self._voices: dict[str, dict] = {}
        root = os.path.dirname(os.path.abspath(str(self.model_dir).rstrip("/")))
        self._voice_dir = os.environ.get("SOPHON_VOICES") or os.path.join(root, "voices")
        llm = self.config.llm
        self.S = int(llm["speech_token_size"])           # 6561
        self.SOS, self.EOS, self.TASK, self.FILL = self.S, self.S + 1, self.S + 2, self.S + 3
        self.MIX = list(llm.get("mix_ratio", [5, 15]))
        if self.model_stage == "cosyvoice3_talker":
            tok = get_qwen_tokenizer(os.path.join(self.model_dir, self.config.qwen_pretrain_path), skip_special_tokens=True, version="cosyvoice3")
            self._prompt_ids = tok.encode(LM_PROMPT, allowed_special="all")
            self._tok = tok; self._pieces: dict[int, str] = {}
            if SEEK_TOK > 0:
                segreg.install(logger)
            logger.info("SophonTagTalker: LM prompt %d tokens (prefill span %d); voices dir %s", len(self._prompt_ids), len(self._prompt_ids) + 1, self._voice_dir)
            if os.environ.get("SOPHON_VOICE_SCAN", "1") == "1":     # clip voices (voices/<name>.wav) are prepared in the background
                from . import voices
                voices.start_background(self._voice_dir, str(self.model_dir), log=logger)
        if self.model_stage == "cosyvoice3_code2wav":
            self.has_preprocess = False          # only the talker stage rewrites its inputs
            self._patch_flow()

    # ------------------------------------------------------------------ code2wav: flow steps / CFG
    def _patch_flow(self):
        c2w = self.code2wav
        dec = c2w.flow_model.decoder
        if FLOW_CFG is not None:
            try:
                dec.inference_cfg_rate = float(FLOW_CFG)
            except Exception as e:  # noqa: BLE001
                logger.warning("SOPHON_FLOW_CFG not applied: %s", e)
        if FLOW_CFG is not None and float(FLOW_CFG) == 0.0:
            # CFG off (tag-tts v1.3 distilled flow): run only the conditional row -- half the estimator work.
            # TensorRT: the stock ONNX has a fixed batch of 2, so point SOPHON_FLOW_ONNX at a batch-1 export
            # (tools/export_flow_onnx.py --batch 1); the optimization profile is switched to batch 1 here.
            def solve_euler_nocfg(x, t_span, mu, mask, spks, cond):
                t, dt = t_span[0], t_span[1] - t_span[0]
                B = int(x.size(0)); dt_ = spks.dtype if spks is not None else x.dtype
                tt = torch.zeros([B], device=x.device, dtype=dt_)
                for step in range(1, len(t_span)):
                    tt[:] = t
                    d = dec.forward_estimator(x.to(dt_), mask.to(dt_), mu.to(dt_), tt, spks.to(dt_) if spks is not None else torch.zeros([B, 80], device=x.device, dtype=dt_),
                                              cond.to(dt_) if cond is not None else torch.zeros_like(x, dtype=dt_))
                    x = x + dt * d.to(x.dtype); t = t + dt
                    if step < len(t_span) - 1:
                        dt = t_span[step + 1] - t
                return x.float()
            dec.solve_euler = solve_euler_nocfg
        onnx_path = os.environ.get("SOPHON_FLOW_ONNX")
        if onnx_path:
            # SOPHON_FLOW_ONNX: an estimator ONNX with a dynamic batch dim (v1.3 export) or a fixed batch-1 export.
            # Dynamic: TensorRT profile batch MIN..MAX rows, so several requests' flow chunks run in one call
            # (set COSYVOICE3_BATCH_FLOW=1). With CFG every request takes 2 rows.
            os.environ["COSYVOICE3_ESTIMATOR_ONNX"] = onnx_path
            import vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt as fe
            rows = 1 if (FLOW_CFG is not None and float(FLOW_CFG) == 0.0) else 2
            maxb = int(os.environ.get("SOPHON_FLOW_MAXB", "8")) * rows if os.environ.get("SOPHON_FLOW_DYNB", "1") == "1" else rows
            optb = min(maxb, 4 * rows)
            dyn_t = False
            try:   # a dynamic-batch export also has dynamic t / spks, which then need profile entries too
                import onnx as _onnx
                g = _onnx.load(onnx_path, load_external_data=False).graph
                dims = {i.name: [d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in g.input}
                dyn_t = isinstance(dims.get("t", [2])[0], str)
            except Exception as e:  # noqa: BLE001
                logger.warning("could not inspect %s: %s", onnx_path, e)
            base = ("x", "mask", "mu", "cond")
            mn = [(rows,) + s[1:] for s in fe._MIN_SHAPES[:4]]; op = [(optb,) + s[1:] for s in fe._OPT_SHAPES[:4]]; mx = [(maxb,) + s[1:] for s in fe._MAX_SHAPES[:4]]
            if dyn_t:
                base += ("t", "spks"); mn += [(rows,), (rows, 80)]; op += [(optb,), (optb, 80)]; mx += [(maxb,), (maxb, 80)]
            elif rows > 1 or maxb > 1:
                pass
            fe._DYNAMIC_INPUTS = base
            fe._convert_onnx_to_trt = _convert_onnx_to_trt_big
            fe._MIN_SHAPES, fe._OPT_SHAPES, fe._MAX_SHAPES = tuple(mn), tuple(op), tuple(mx)
            if not dyn_t and maxb != rows:
                logger.warning("SOPHON_FLOW_ONNX has a fixed batch: no cross-request flow batching (set COSYVOICE3_BATCH_FLOW=0)")
                fe._MAX_SHAPES = tuple((rows,) + s[1:] for s in fe._MAX_SHAPES); fe._OPT_SHAPES = tuple((rows,) + s[1:] for s in fe._OPT_SHAPES); maxb = optb = rows
            # the stock plan cache key ignores the profile -> keep one cache dir per profile
            os.environ.setdefault("COSYVOICE3_TRT_CACHE", os.path.join(os.path.expanduser("~"), ".cache", "vllm_omni", "cosyvoice3_trt", f"b{rows}-{maxb}" + ("t" if dyn_t else "")))
            logger.info("SophonTagTalker: TensorRT flow profile batch %d..%d (opt %d), cache %s", rows, maxb, optb, os.environ["COSYVOICE3_TRT_CACHE"])
        # Isolation: if the flow fails for one request, lose that one chunk of that request instead of the whole stage.
        # (The chunk length itself stays bounded: proc._bound_left_context keeps codec_left_context_frames of context.)
        fs = getattr(c2w, "forward_streaming", None)
        fb = getattr(c2w, "forward_streaming_batch", None)
        if fs is not None:
            def fs_safe(token, prompt_token, prompt_feat, embedding, *, cache_state=None, n_timesteps=10, token_offset_tokens=0, finalize=False):
                off = int(token_offset_tokens)
                try:
                    return fs(token, prompt_token, prompt_feat, embedding, cache_state=cache_state, n_timesteps=n_timesteps,
                              token_offset_tokens=off, finalize=finalize)
                except Exception as e:  # noqa: BLE001
                    logger.error("SophonTagTalker: flow failed for one request (%r, %d tokens); that audio chunk is dropped", e, int(token.shape[1]))
                    return torch.zeros((1, 1, 0), device=prompt_feat.device), (None if finalize else cache_state)
            c2w.forward_streaming = fs_safe
            if fb is not None:
                def batch_safe(items, **kw):
                    try:
                        return fb(items, **kw)
                    except Exception as e:  # noqa: BLE001
                        logger.error("SophonTagTalker: batched flow failed (%r), running %d requests one by one", e, len(items))
                        return [fs_safe(it["token"], it["prompt_token"], it["prompt_feat"], it["embedding"], cache_state=it.get("cache_state"),
                                        n_timesteps=kw.get("n_timesteps", 10), token_offset_tokens=int(it.get("token_offset_tokens", 0)),
                                        finalize=bool(it.get("finalize", False))) for it in items]
                c2w.forward_streaming_batch = batch_safe
        for name in ("forward", "forward_streaming", "forward_streaming_batch"):
            fn = getattr(c2w, name, None)
            if fn is None:
                continue
            def wrap(f):
                def g(*a, **k):
                    if "n_timesteps" in k:
                        k["n_timesteps"] = FLOW_STEPS
                    return f(*a, **k)
                return g
            setattr(c2w, name, wrap(fn))
        logger.info("SophonTagTalker code2wav: flow steps %d, cfg %s, onnx %s", FLOW_STEPS, FLOW_CFG or "model default", os.environ.get("SOPHON_FLOW_ONNX", "stock"))

    # ------------------------------------------------------------------ helpers
    def _voice(self, name: str | None, lang: str) -> str:
        """Name of the voice this request will use (the conditioning itself is attached by proc.talker2code2wav_async_chunk)."""
        from . import voices
        for n in voices.candidates(str(name or ""), lang):
            if any(os.path.exists(os.path.join(self._voice_dir, n + e)) for e in voices.EXTS):
                return n
        raise ValueError(f"sophon tag talker: no voice found in {self._voice_dir}")

    def _semb(self, ids) -> torch.Tensor:
        w = self.model.speech_embedding.weight
        return w[torch.as_tensor(list(ids), device=w.device, dtype=torch.long)]

    def _temb(self, ids) -> torch.Tensor:
        emb = self.model.llm.model.embed_tokens
        return emb(torch.as_tensor(list(ids), device=emb.weight.device, dtype=torch.long))

    @staticmethod
    def _flat(info: dict) -> dict:
        add = info.get("additional_information")
        if isinstance(add, dict):
            merged = {k: v for k, v in info.items() if k != "additional_information"}
            for k, v in add.items():
                merged.setdefault(k, v)
            return merged
        return info

    @staticmethod
    def _ids(info) -> list[int] | None:
        ids = info.get("ids")
        ids = ids.get("output") if isinstance(ids, dict) else None
        if ids is None:
            return None
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        return [int(x) for x in ids]

    def _sync_text(self, st: _St, info: dict):
        ids = self._ids(info)
        if ids is not None and len(ids) >= len(st.text):
            st.text = ids
        meta = info.get("meta") or {}
        if isinstance(meta, dict) and _scalar_true(meta.get("finished", False)):
            st.fin = True

    # ------------------------------------------------------------------ preprocess: choose this step's input embedding
    def _touch(self, rid):
        """The state of rid, moved to the newest end: states are dropped oldest first (MAX_STATES), and a long reply must
        not lose its state to the many short requests that start while it is speaking."""
        st = self._st.pop(rid, None)
        if st is not None:
            self._st[rid] = st
        return st

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info: Any):
        info = self._flat(info)
        rid = info.get("request_id")
        span = int(input_ids.shape[0])
        is_prefill = info.get("_omni_is_prefill")
        if not isinstance(is_prefill, bool):
            is_prefill = span > 1
        dev = self.model.speech_embedding.weight.device
        if rid is None:
            emb = self._semb(input_ids.tolist()) if input_embeds is None else input_embeds
            return input_ids, emb, {}
        if is_prefill:
            st = self._touch(rid)
            if st is not None and st.restart:          # next segment of a long reply: same request, fresh LM context
                st.new_segment(); segreg.RESTARTED.discard(rid)
            else:
                lang = str(_first(info.get("language"), "zh") or "zh").lower()[:2]
                voice = self._voice(str(_first(info.get("speaker"), "") or ""), lang)
                st = _St(voice)
                if len(self._st) > MAX_STATES:
                    for k in list(self._st)[: len(self._st) - MAX_STATES]:
                        self._st.pop(k, None)
                self._st[rid] = st
            self._sync_text(st, info)
            pre = torch.cat([self._semb([self.SOS]), self._temb(self._prompt_ids)], 0)
            if pre.shape[0] != span:
                logger.warning_once("sophon tag talker prefill span %d != prompt %d; set hf_overrides.async_chunk_prewarm_prompt_len=%d",
                                    span, pre.shape[0], pre.shape[0])
                pad = self._semb([self.SOS]).expand(max(0, span - pre.shape[0]), -1)
                pre = torch.cat([pad, pre], 0)[-span:]
            st.last_fed = pre[-1:]
            st.dummy = False
            self._order.append(rid); self._prefill_rows.add(rid)
            return input_ids, pre.to(input_embeds.dtype if input_embeds is not None else pre.dtype), {}
        st = self._touch(rid)
        if st is None:      # unknown (should not happen): behave like the native decode step
            self._order.append(rid)
            emb = self._semb(input_ids.tolist())
            return input_ids, emb, {}
        self._sync_text(st, info)
        segreg.AHEAD[rid] = len(st.text) - st.pos
        if st.queue:
            emb = st.queue.pop(0)
            st.dummy = bool(st.queue)
        else:
            last = st.last_tok if st.last_tok is not None else int(input_ids.reshape(-1)[-1].item())
            emb = self._semb([last])
            st.dummy = False
        st.last_fed = emb
        self._order.append(rid)
        return input_ids, emb.to(input_embeds.dtype if input_embeds is not None else emb.dtype), {}

    # ------------------------------------------------------------------ forward: emit the voice prompt for code2wav
    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, additional_information=None, **kwargs):
        self._order = []; self._prefill_rows = set()
        return super().forward(input_ids, positions, intermediate_tensors, inputs_embeds, additional_information, **kwargs)

    # ------------------------------------------------------------------ logits: keep the raw fill logit
    def compute_logits(self, hidden_states):
        if self.model_stage != "cosyvoice3_talker":
            return super().compute_logits(hidden_states)
        from vllm_omni.model_executor.models.output_templates import OmniOutput
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        raw = self.model.llm_decoder(hidden_states)
        fill = raw[..., self.FILL].clone()
        logits = super().compute_logits(hidden_states)
        logits[..., self.FILL] = fill.to(logits.dtype)
        return logits

    def set_batch_req_ids(self, req_ids) -> None:
        self._batch_req_ids = list(req_ids)

    # ------------------------------------------------------------------ sampling: the bistream state machine
    def _ras(self, row: torch.Tensor, st: _St, temperature: float, top_p: float, top_k: int, generator) -> int:
        scfg = dict(self.config.llm.get("sampling", {}))
        if temperature < self._sampling_eps:
            return int(torch.argmax(row).item())
        ws = torch.log_softmax(row / max(temperature, self._sampling_eps), dim=0)
        return self._ras_sample_one(ws, st.out, top_p=top_p, top_k=top_k, win_size=int(scfg.get("win_size", 10)),
                                    tau_r=float(scfg.get("tau_r", 0.1)), generator=generator)

    def _need_text(self, st: _St):
        """After a fill token: queue the next 5 text tokens, or go to the final phase, or (rarely) starve."""
        if SEEK_TOK > 0 and st.seg_end is None and st.pos > st.seg_start:
            self._plan_segment_end(st)
        lim = len(st.text) if st.seg_end is None else st.seg_end
        fin = st.fin or st.seg_end is not None
        avail = lim - st.pos
        n = self.MIX[0]
        if avail >= n:
            st.queue = list(self._temb(st.text[st.pos: st.pos + n]).split(1, 0)); st.pos += n
            return
        if fin:
            st.phase = "final"
            rest = list(self._temb(st.text[st.pos:lim]).split(1, 0)) if avail > 0 else []
            st.pos = lim
            # official: lm_input = concat([lm_input, text_cache, task_id]) where lm_input is the (already consumed) input
            # that produced the fill token -> it is fed once more.
            dup = [st.last_fed] if st.last_fed is not None else []
            st.queue = dup + rest + [self._semb([self.TASK])]
            return
        # thinker still writing but not 5 tokens ahead: feed what there is (rare; logged)
        st.starved += 1
        if st.starved in (1, 10, 100):
            logger.warning("sophon tag talker: text starved (%d tokens available, thinker not finished) x%d", avail, st.starved)
        if avail > 0:
            st.queue = list(self._temb(st.text[st.pos:]).split(1, 0)); st.pos = len(st.text)
        else:
            st.queue = [st.last_fed] if st.last_fed is not None else [self._semb([self.FILL])]

    def _piece(self, tid: int) -> str:
        p = self._pieces.get(tid)
        if p is None:
            p = self._pieces[tid] = self._tok.decode([tid])
        return p

    def _plan_segment_end(self, st: _St):
        """Where the current segment ends (segreg.find_cut: the talker is fed text well ahead of its speech, so the cut is
        planned on the text, never before the text already fed)."""
        end = segreg.find_cut(st.text, st.seg_start, st.pos, self._piece, st.fin)
        if end is not None:
            st.seg_end = max(end, st.pos)

    def sample(self, logits, sampling_metadata):
        if logits is None or logits.numel() == 0 or self.model_stage != "cosyvoice3_talker":
            return super().sample(logits, sampling_metadata)
        from vllm.v1.outputs import SamplerOutput
        from vllm.v1.sample.ops.topk_topp_sampler import random_sample
        rows = int(logits.shape[0])
        rids = self._batch_req_ids if len(self._batch_req_ids) == rows else [None] * rows
        sts = [self._st.get(r) for r in rids]
        if not any(st is not None for st in sts):
            return super().sample(logits, sampling_metadata)
        dev = logits.device
        scfg = dict(self.config.llm.get("sampling", {}))
        win = int(scfg.get("win_size", 10)); tau = float(scfg.get("tau_r", 0.1))
        # per-row plan (CPU, no GPU sync): forced token or sampling mode
        forced = [None] * rows; final = [False] * rows
        for i, st in enumerate(sts):
            if st is None:
                continue
            st.steps += 1
            if st.restart:
                forced[i] = self.FILL                  # waiting for the restart: nothing to say
            elif MAX_SPEECH_TOK > 0 and st.nspeech >= MAX_SPEECH_TOK and not st.done_eos:
                forced[i] = self.EOS
                if st.nspeech == MAX_SPEECH_TOK:
                    logger.info("SophonTagTalker: reply reached SOPHON_MAX_SPEECH_S (%d speech tokens), ending it", MAX_SPEECH_TOK)
            elif SEEK_TOK > 0 and st.seg_speech >= SEG_MAX_TOK and not st.done_eos:
                forced[i] = self.EOS
                if st.seg_end is None:                 # cut here (the rest of the text goes to the next segment)
                    st.seg_end = st.pos; st.phase = "final"
                logger.warning("SophonTagTalker: segment reached SOPHON_SEG_MAX_S, cutting it")
            elif st.dummy:
                forced[i] = self.FILL
            elif st.phase == "bi" and st.next_fill != -1 and len(st.out) == st.next_fill:
                forced[i] = self.FILL; st.next_fill += self.MIX[1] + 1
            elif st.phase == "final":
                final[i] = True
        x = logits.to(torch.float32)
        keep_fill = x[:, self.FILL].clone(); keep_eos = x[:, self.EOS].clone()
        x[:, self.S:] = float("-inf")
        fin_t = torch.tensor(final, device=dev)
        x[:, self.FILL] = torch.where(fin_t, torch.full_like(keep_fill, float("-inf")), keep_fill)
        x[:, self.EOS] = torch.where(fin_t, keep_eos, torch.full_like(keep_eos, float("-inf")))
        temp = sampling_metadata.temperature
        temp = temp.to(dev, torch.float32).reshape(-1)[:rows] if isinstance(temp, torch.Tensor) and temp.numel() >= rows else torch.ones(rows, device=dev)
        greedy = temp < self._sampling_eps
        ws = torch.log_softmax(x / temp.clamp(min=self._sampling_eps).unsqueeze(1), dim=1)
        tp = sampling_metadata.top_p; tk = sampling_metadata.top_k
        top_p = tp.to(dev, torch.float32).reshape(-1)[:rows] if isinstance(tp, torch.Tensor) and tp.numel() >= rows else torch.full((rows,), float(scfg.get("top_p", 0.8)), device=dev)
        top_k = tk.to(dev).reshape(-1)[:rows] if isinstance(tk, torch.Tensor) and tk.numel() >= rows else torch.full((rows,), int(scfg.get("top_k", 25)), device=dev)
        probs = ws.softmax(dim=1)
        sp, si = probs.sort(dim=1, descending=True, stable=True)
        cum_before = sp.cumsum(dim=1) - sp
        ar = torch.arange(sp.shape[1], device=dev).unsqueeze(0)
        keep = (cum_before < top_p.unsqueeze(1)) & ((top_k.unsqueeze(1) <= 0) | (ar < top_k.unsqueeze(1)))
        gens = sampling_metadata.generators or {}
        pick = random_sample(sp * keep.to(sp.dtype), gens)                 # index into sorted
        top_id = si.gather(1, pick.long().unsqueeze(1)).squeeze(1)
        # RAS: resample from the full distribution (minus top_id) when top_id is in the recent window often enough
        recent = torch.full((rows, win), -1, dtype=torch.long)
        for i, st in enumerate(sts):
            if st is not None and st.out:
                r = st.out[-win:]; recent[i, win - len(r):] = torch.as_tensor(r)
        recent = recent.to(dev, non_blocking=True)
        rep = (recent == top_id.unsqueeze(1)).sum(1)
        ws2 = ws.scatter(1, top_id.unsqueeze(1), float("-inf"))
        alt = random_sample(ws2.softmax(dim=1), gens)
        tok = torch.where(rep >= win * tau, alt.long(), top_id)
        tok = torch.where(greedy, ws.argmax(dim=1), tok)
        out = tok.tolist()                                                  # the one GPU sync of the step
        for i, st in enumerate(sts):
            if forced[i] is not None:
                out[i] = forced[i]
            if st is None or st.dummy or st.restart:
                continue
            t = int(out[i])
            capped = MAX_SPEECH_TOK > 0 and st.nspeech >= MAX_SPEECH_TOK
            if t == self.EOS and st.phase == "final" and st.seg_end is not None and not capped:
                # end of a segment, not of the reply: restart the request with a fresh LM context
                out[i] = self.FILL; st.restart = True
                if rids[i] is not None:
                    segreg.PENDING.add(rids[i])
                continue
            if st.phase == "bi":
                if t == self.FILL and forced[i] is None:
                    st.next_fill = len(st.out) + self.MIX[1] + 1
                if t == self.FILL and forced[i] is not None:
                    st.next_fill = len(st.out) + self.MIX[1] + 1
                st.out.append(t)
                if t == self.FILL:
                    self._need_text(st)
                else:
                    st.last_tok = t
            else:
                st.out.append(t); st.last_tok = t
                if t == self.EOS:
                    st.done_eos = True
            if t == self.EOS:
                st.done_eos = True
            if t < self.S:
                st.nspeech += 1; st.seg_speech += 1
                if DEBUG and st.nspeech in (1, 13, 33, 73):
                    logger.info("tagdbg %s speech#%d at %.3fs steps=%d text=%d/%d fin=%s", str(rids[i])[-12:], st.nspeech, __import__("time").monotonic() - st.t0, st.steps, st.pos, len(st.text), st.fin)
        sampled = torch.tensor(out, device=dev, dtype=torch.int32)
        return SamplerOutput(sampled_token_ids=sampled.unsqueeze(-1), logprobs_tensors=None)


def _convert_onnx_to_trt_big(onnx_path: str, plan_path: str, strongly_typed: bool) -> None:
    """Same as vLLM-Omni's builder, with a bigger workspace (a multi-row profile needs it) and our profile."""
    import tensorrt as trt
    import vllm_omni.model_executor.models.cosyvoice3.flow_estimator_trt as fe
    logger.info("Building flow-estimator TensorRT engine from %s, profile %s..%s", onnx_path, fe._MIN_SHAPES[0], fe._MAX_SHAPES[0])
    tl = fe._trt_logger(); builder = trt.Builder(tl)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)) if strongly_typed else builder.create_network(0)
    parser = trt.OnnxParser(network, tl)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise ValueError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(os.environ.get("SOPHON_TRT_WORKSPACE_GB", "16")) << 30)
    if not strongly_typed:
        for name in ("FP16", "TF32"):
            flag = getattr(trt.BuilderFlag, name, None)
            if flag is not None:
                config.set_flag(flag); break
    profile = builder.create_optimization_profile()
    for name, mn, op, mx in zip(fe._DYNAMIC_INPUTS, fe._MIN_SHAPES, fe._OPT_SHAPES, fe._MAX_SHAPES):
        profile.set_shape(name, mn, op, mx)
    config.add_optimization_profile(profile)
    engine_bytes = builder.build_serialized_network(network, config)
    if engine_bytes is None:
        raise RuntimeError(f"TensorRT failed to build flow-estimator engine from {onnx_path}")
    fe._write_plan_atomically(engine_bytes, plan_path)
    logger.info("Wrote flow-estimator TensorRT engine to %s", plan_path)


def _to_dict(x):
    from vllm_omni.data_entry_keys import to_dict
    return to_dict(x)
