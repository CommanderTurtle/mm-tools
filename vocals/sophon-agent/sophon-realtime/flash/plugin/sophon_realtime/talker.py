"""Qwen3-TTS talker (vLLM-Omni implementation) + streaming text input.
Text is not given all at once: each token the thinker writes is sent over (cumulative list ids.output); newly arrived tokens are projected and appended to the tail of trailing_text,
and a tts_eos is appended once the thinker finishes -- after that the original frame-by-frame text-reading decode logic is reused as-is (official streaming layout: the prefix contains only the first text token).
Voices: voices/<name>.pt (x-vector computed by the Base model's speaker encoder) or voices/<name>.wav / .flac (zero-shot clone: a 3-10 s
reference clip; its x-vector is computed by the same speaker encoder on first use), looked up in the order "<voice>_<lang>" -> "<voice>".
Files are re-read when they change, so voices can be added, replaced or deleted while the server runs."""
import os
import re
from typing import Any

import numpy as np

import torch
from vllm.logger import init_logger
from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSTalkerForConditionalGeneration

from . import segreg

logger = init_logger("vllm_omni.sophon_realtime.talker")
LANG = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french", "es": "spanish",
        "it": "italian", "pt": "portuguese", "ru": "russian"}
ROLE = [151644, 77091, 198]          # "<|im_start|>assistant\n"
PROMPT_LEN = 10                      # 3 role + 6 codec prefix + 1 (first text token)
_VOICE_NAME = re.compile(r"[a-z0-9][a-z0-9_.-]{0,63}")
MAX_REF_S = float(os.environ.get("SOPHON_MAX_REF_SECONDS", "15"))   # longer reference clips are cut
FRAMES_PER_TOK = int(os.environ.get("SOPHON_FRAMES_PER_TOKEN", "12"))  # runaway guard: total frame cap ~= 12 frames/text token + 60


def _first(x, default=None):
    if isinstance(x, (list, tuple)):
        return x[0] if x else default
    return default if x is None else x


def _scalar_true(x) -> bool:
    if isinstance(x, torch.Tensor):
        return bool(x.reshape(-1)[0].item()) if x.numel() else False
    if isinstance(x, (list, tuple)):
        return bool(x and _scalar_true(x[0]))
    return bool(x)


class SophonTTSTalker(Qwen3TTSTalkerForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        root = os.path.dirname(os.path.abspath(str(self.model_path).rstrip("/")))
        self._voice_dir = os.environ.get("SOPHON_VOICES") or os.path.join(root, "voices")
        self._voices: dict[str, tuple[torch.Tensor, tuple[str, float]]] = {}
        self._missing: set[str] = set()
        self._bad: set[tuple[str, float]] = set()
        self._force_eos: set[str] = set()
        self._text_open: set[str] = set()     # the end of their text (tts_eos) is not fed yet: no codec EOS (it would end the reply)
        # Long replies in segments (segreg): per request {"start": first text id of the segment, "end": cut or None}
        self._seg: dict[str, dict] = {}
        self._pieces: dict[int, str] = {}
        if segreg.SEEK_S > 0:
            segreg.install(logger)
            try:                                  # load the thinker tokenizer now, not on the first reply (~10 s)
                self._piece(0)
            except Exception as e:  # noqa: BLE001
                logger.warning("sophon talker: tokenizer not loaded at start (%r); it loads on the first long reply", e)
        logger.info("SophonTTSTalker voices dir: %s (%s)", self._voice_dir,
                    sorted(os.listdir(self._voice_dir)) if os.path.isdir(self._voice_dir) else "missing")

    # ---------- helpers ----------
    def _te(self, ids, device) -> torch.Tensor:
        t = torch.tensor([list(ids)], dtype=torch.long, device=device)
        return self.text_projection(self.text_embedding(t)).reshape(len(ids), -1).to(self._embedding_dtype)

    def _ce(self, ids, device) -> torch.Tensor:
        t = torch.tensor(list(ids), dtype=torch.long, device=device).reshape(-1, 1)
        return self.embed_input_ids(t).reshape(len(ids), -1).to(self._embedding_dtype)

    def _voice(self, voice: str, lang: str, device) -> torch.Tensor:
        voice = (voice or "chelsie").lower().strip()
        if not _VOICE_NAME.fullmatch(voice):      # voice names come from clients: no paths
            voice = "chelsie"
        names = (f"{voice}_{lang}", voice, f"chelsie_{lang}", "chelsie_zh")
        for i, name in enumerate(names):
            for ext in (".pt", ".wav", ".flac"):
                p = os.path.join(self._voice_dir, name + ext)
                try:
                    mt = os.path.getmtime(p)
                except OSError:
                    continue
                hit = self._voices.get(name)
                if hit is None or hit[1] != (p, mt):
                    if (p, mt) in self._bad:
                        continue
                    try:
                        hit = self._voices[name] = (self._load_voice(p, ext), (p, mt))
                    except Exception as e:  # noqa: BLE001  -- a bad voice file must not take the stage down
                        self._bad.add((p, mt))
                        logger.error("sophon_realtime: cannot load voice %s: %s", p, e)
                        continue
                if i >= 2 and voice not in ("chelsie", f"chelsie_{lang}") and voice not in self._missing:
                    self._missing.add(voice)
                    logger.warning("sophon_realtime: voice %r not found in %s, using %s", voice, self._voice_dir, name)
                return hit[0].to(device)
        raise ValueError(f"sophon_realtime: voice {voice!r} not found in {self._voice_dir}")

    def _load_voice(self, path: str, ext: str) -> torch.Tensor:
        if ext == ".pt":
            v = torch.load(path, map_location="cpu", weights_only=True)
        else:   # zero-shot clone from a reference clip: same speaker encoder + mel as the official Qwen3-TTS x-vector mode
            import soundfile as sf
            wav, sr = sf.read(path, dtype="float32", always_2d=True)
            wav = np.ascontiguousarray(wav.mean(1)[: int(sr * MAX_REF_S)])
            if len(wav) < sr * 1.0:
                raise ValueError(f"sophon_realtime: reference clip {path} is shorter than 1 s")
            with torch.no_grad():
                v = self._prompt_builder.extract_speaker_embedding(wav, int(sr))
            logger.info("sophon_realtime: cloned voice from %s (%.1f s)", path, len(wav) / sr)
        return v.detach().reshape(1, -1).to(self._embedding_dtype).cpu()

    def _piece(self, tid: int) -> str:
        p = self._pieces.get(tid)
        if p is None:
            from .proc import _tokenizer
            p = self._pieces[tid] = _tokenizer(None).decode([tid])
        return p

    def _segment(self, rid, ids: list[int], fed: int, fin: bool) -> tuple[int, int | None]:
        """(start, end) of the current segment; plans the cut once there is enough text (segreg.find_cut)."""
        if rid is None:
            return 0, None
        seg = self._seg.setdefault(rid, {"start": 0, "end": None, "ids": ids, "fin": fin})
        if len(ids) >= len(seg["ids"]):
            seg["ids"] = ids; seg["fin"] = fin
        if segreg.SEEK_S > 0 and seg["end"] is None:
            seg["end"] = segreg.find_cut(ids, seg["start"], fed, self._piece, fin)
        return seg["start"], seg["end"]

    @staticmethod
    def _ids(info) -> list[int]:
        ids = (info.get("ids") or {}).get("output") if isinstance(info.get("ids"), dict) else None
        if isinstance(ids, torch.Tensor):
            ids = ids.reshape(-1).tolist()
        return [int(x) for x in (ids or [])]

    @staticmethod
    def _fin(info) -> bool:
        meta = info.get("meta") or {}
        return _scalar_true(meta.get("finished", False))

    # ---------- batched prompt builder: not needed (the prefix is built in preprocess) ----------
    def preprocess_batch(self, **_: Any) -> None:
        return None

    def preprocess_batch_mrv2(self, **_: Any) -> None:
        return None

    # ---------- prefill ----------
    def _prefill(self, input_ids: torch.Tensor, info: dict) -> tuple[torch.Tensor, torch.Tensor, dict]:
        dev = input_ids.device; span = int(input_ids.shape[0])
        cfg = self.config; tc = self.talker_config
        ids = self._ids(info); fin = self._fin(info)
        rid = info.get("request_id")
        seg = self._seg.pop(rid, None)     # re-inserted below at the newest end: oldest entries are dropped first
        if seg is not None and len(seg.get("ids", ())) > len(ids):   # restarted after the thinker finished: the request no longer carries the text
            ids, fin = seg["ids"], seg["fin"]
        if seg is not None and seg.get("end") is not None and rid in segreg.RESTARTED:   # the next segment of a long reply
            segreg.RESTARTED.discard(rid)
            self._seg[rid] = {"start": seg["end"], "end": None, "ids": ids, "fin": fin}
        elif rid is not None:
            self._seg[rid] = {"start": 0, "end": None, "ids": ids, "fin": fin}
            if len(self._seg) > 512:
                for k in list(self._seg)[:len(self._seg) - 512]:
                    self._seg.pop(k, None); self._text_open.discard(k)
        start, end = self._segment(rid, ids, 0, fin)
        lim = len(ids) if end is None else end
        seg_ids = ids[start:lim]
        lang = str(_first(info.get("language"), "zh") or "zh").lower()[:2]
        voice = str(_first(info.get("speaker"), "chelsie") or "chelsie")
        lang_id = tc.codec_language_id.get(LANG.get(lang, "english"), tc.codec_language_id["english"])
        bos, eos, pad = self._te([cfg.tts_bos_token_id, cfg.tts_eos_token_id, cfg.tts_pad_token_id], dev).split(1, 0)
        codec = torch.cat([self._ce([tc.codec_think_id, tc.codec_think_bos_id, lang_id, tc.codec_think_eos_id], dev),
                           self._voice(voice, lang, dev), self._ce([tc.codec_pad_id, tc.codec_bos_id], dev)], 0)     # (7, H)
        first = self._te(seg_ids[:1], dev) if seg_ids else pad
        pre = torch.cat([self._te(ROLE, dev), torch.cat([pad.expand(codec.shape[0] - 2, -1), bos], 0) + codec[:-1], first + codec[-1:]], 0)
        tail = [self._te(seg_ids[1:], dev)] if len(seg_ids) > 1 else []
        seg_eos = end is not None
        if rid is not None:
            (self._text_open.discard if fin or seg_eos else self._text_open.add)(rid)
        if fin or seg_eos:
            tail.append(eos)
            if seg_eos:
                segreg.AT_EOS.add(rid)                    # its end token restarts the request for the next segment
        trailing = torch.cat(tail, 0) if tail else pre.new_zeros((0, pre.shape[1]))
        if pre.shape[0] != span:
            logger.warning_once("sophon talker prefill span %d != prompt %d; set hf_overrides.async_chunk_prewarm_prompt_len=%d",
                                span, pre.shape[0], PROMPT_LEN)
            pre = torch.cat([pre, pad.expand(max(0, span - pre.shape[0]), -1)], 0)[:span]
        ids_out = input_ids.clone(); ids_out[:] = int(tc.codec_pad_id)
        if rid is not None:
            self._force_eos.discard(rid)                  # a new segment starts with a clean runaway guard
            if fin and not ids:
                self._force_eos.add(rid)                  # no text (full-duplex turn-check request): finish immediately
            else:
                self._req_x_vector_only[rid] = True       # forbid silence in the first frames (original guard for x-vector cloning)
        upd = {
            "hidden_states": {"trailing_text": trailing.detach()},
            "meta": {"talker_text_offset": 0, "codec_streaming": True, "talker_prefill_offset": span,
                     "codec_frame_valid": torch.zeros((), dtype=torch.bool, device=dev),
                     "sophon_n": lim, "sophon_eos": fin or seg_eos, "sophon_steps": 0, "sophon_post": 0},
            "codes": {"audio": torch.zeros((span, int(tc.num_code_groups)), dtype=torch.long, device=dev)},
        }
        return ids_out, pre, upd

    # ---------- decode: append newly arrived text to the tail of trailing_text ----------
    def _extend(self, info: dict, device) -> dict | None:
        meta = info.get("meta")
        if not isinstance(meta, dict):
            meta = {}; info["meta"] = meta
        ids = self._ids(info); fin = self._fin(info)
        n = int(meta.get("sophon_n", 0) or 0); eos_done = bool(meta.get("sophon_eos", False))
        steps = int(meta.get("sophon_steps", 0) or 0) + 1; post = int(meta.get("sophon_post", 0) or 0) + (1 if eos_done else 0)
        meta["sophon_steps"] = steps; meta["sophon_post"] = post
        rid = info.get("request_id")
        seg = self._seg.pop(rid, None)
        if seg is not None:
            self._seg[rid] = seg                          # newest end: a long reply keeps its entry while short requests come and go
            if len(seg.get("ids", ())) > len(ids):
                ids, fin = seg["ids"], seg["fin"]
        start, end = self._segment(rid, ids, n, fin)
        lim = len(ids) if end is None else end
        if rid is not None and eos_done and steps > FRAMES_PER_TOK * max(1, lim - start) + 60:
            if rid not in self._force_eos:
                logger.warning("sophon talker: force codec EOS for %s (steps=%d post_eos=%d text_tokens=%d)", rid, steps, post, lim - start)
            self._force_eos.add(rid)
        add = []; fed = n
        if lim > n:
            add.append(self._te(ids[n:lim], device)); n = lim
        if (fin or end is not None) and n >= lim and not eos_done:
            add.append(self._te([self.config.tts_eos_token_id], device)); eos_done = True
            self._text_open.discard(rid)
            if end is not None:
                segreg.AT_EOS.add(rid)                    # end of a segment: its end token restarts the request
        hs0 = info.get("hidden_states"); tail0 = hs0.get("trailing_text") if isinstance(hs0, dict) else None
        in_hand = (tail0.shape[0] - int(meta.get("talker_text_offset", 0) or 0)) if isinstance(tail0, torch.Tensor) and tail0.ndim == 2 else 0
        if rid is not None:     # segreg: keep stepping without new thinker chunks while there is text in hand / all of it is here
            segreg.AHEAD[rid] = 1 << 20 if eos_done else in_hand + (n - fed)
        if not add:
            return {"sophon_steps": steps, "sophon_post": post}
        hs = info.get("hidden_states")
        if not isinstance(hs, dict):
            hs = {}; info["hidden_states"] = hs
        tail = hs.get("trailing_text"); off = int(meta.get("talker_text_offset", 0) or 0)
        rest = tail[off:].to(device=device, dtype=self._embedding_dtype) if isinstance(tail, torch.Tensor) and tail.ndim == 2 else None
        new_tail = torch.cat(([rest] if rest is not None and rest.shape[0] else []) + add, 0)
        hs["trailing_text"] = new_tail; meta["talker_text_offset"] = 0; meta["sophon_n"] = n; meta["sophon_eos"] = eos_done
        return {"trailing_text": new_tail, "sophon_n": n, "sophon_eos": eos_done, "sophon_steps": steps, "sophon_post": post}

    @staticmethod
    def _merge(upd: dict, ext: dict | None) -> dict:
        if ext is None:
            return upd
        upd.setdefault("meta", {}).update({k: v for k, v in ext.items() if k != "trailing_text"})
        if "trailing_text" in ext:
            hs = upd.setdefault("hidden_states", {})
            if "trailing_text" not in hs:
                hs["trailing_text"] = ext["trailing_text"].detach()
        return upd

    # ---------- runaway guard: force codec EOS when the cap is reached ----------
    def set_batch_req_ids(self, req_ids) -> None:
        super().set_batch_req_ids(req_ids)
        if self._force_eos or self._text_open:
            live = set(self._batch_req_ids)
            self._force_eos &= live

    def compute_logits(self, hidden_states, sampling_metadata: Any = None):
        logits = super().compute_logits(hidden_states, sampling_metadata)
        if logits is None or not (self._force_eos or self._text_open) or len(self._batch_req_ids) != logits.shape[0]:
            return logits
        hold = [i for i, r in enumerate(self._batch_req_ids) if r in self._text_open and r not in self._force_eos]
        if hold:      # the thinker is still writing: waiting for text is not the end of the reply
            logits[torch.tensor(hold, device=logits.device), self._codec_eos_token_id] = -1e4
        rows = [i for i, r in enumerate(self._batch_req_ids) if r in self._force_eos] if len(self._batch_req_ids) == logits.shape[0] else []
        if rows:
            idx = torch.tensor(rows, device=logits.device)
            logits[idx] = logits[idx] - 1e4; logits[idx, self._codec_eos_token_id] = 0.0   # not -inf: avoids an all -inf row when min_tokens masks EOS
        return logits

    def preprocess(self, input_ids: torch.Tensor, input_embeds: torch.Tensor | None, **info_dict: Any):
        add = info_dict.get("additional_information")
        if isinstance(add, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in add.items():
                merged.setdefault(k, v)
            info_dict = merged
        span = int(input_ids.shape[0])
        is_prefill = info_dict.get("_omni_is_prefill")
        if not isinstance(is_prefill, bool):
            is_prefill = span > 1
        if is_prefill and span > 0:
            return self._prefill(input_ids, info_dict)
        ext = self._extend(info_dict, input_ids.device)
        ids_o, emb_o, upd = super().preprocess(input_ids, input_embeds, **info_dict)
        return ids_o, emb_o, self._merge(upd, ext)

    def preprocess_decode_batch(self, *, input_ids: torch.Tensor, req_infos: list[dict[str, Any]], input_embeds: torch.Tensor | None = None):
        exts = [self._extend(info, input_ids.device) for info in req_infos]
        out = super().preprocess_decode_batch(input_ids=input_ids, req_infos=req_infos, input_embeds=input_embeds)
        updates = out[-1]
        for i, ext in enumerate(exts):
            updates[i] = self._merge(updates[i], ext)
        return out
