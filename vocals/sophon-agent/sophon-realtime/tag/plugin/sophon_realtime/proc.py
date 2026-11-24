"""Step-by-step thinker -> tag-tts talker forwarding (Sophon-Realtime-32B-Tag).
Every thinker step sends the talker the *cumulative* CosyVoice token ids of the reply written so far:
- sound tags are kept whole and passed through (whitelist: [laughter] [breath] [sigh] [cough] [quick_breath],
  <laughter>..</laughter>, <strong>..</strong>); anything else in brackets and markdown symbols is dropped; an unclosed
  bracket at the end is held back; <strong> is auto-closed at punctuation; open spans are closed when the thinker finishes
  (same rules as the Tag one-box server sophon_tag_server.TagStream).
- the text is re-tokenized with the tag-tts (CosyVoice3) tokenizer, so every tag is one special token.
- the last token is held back until the next step (it may still merge); everything is released when the thinker finishes.
- a chunk is sent on *every* thinker step once LEAD tokens are ready (heartbeat): in async-chunk mode the talker only steps
  when a chunk arrives, and it needs ~4 steps per text token (bistream 5 text : 15 speech + the text positions)."""
import os
import re
import threading

import torch
from vllm.logger import init_logger
from .tools import cut_tool
from vllm_omni.data_entry_keys import IdsStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.model_executor.stage_input_processors.tts_utils import extract_language_from_request, extract_speaker_from_request

logger = init_logger("vllm_omni.sophon_realtime.proc")

LEAD = int(os.environ.get("SOPHON_LEAD", "5"))
POINT = {"[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]"}
SPAN = {"laughter", "strong"}
CLOSE = {"[": "]", "<": ">", "(": ")", "（": "）", "【": "】"}
_TOK = None
_TTOK = None
_LOCK = threading.Lock()
_ST: dict[str, dict] = {}


def tag_clean(s: str, final: bool) -> str:
    """Whole-string version of sophon_tag_server.TagStream (returns the text with whitelisted tags kept)."""
    out: list[str] = []; open_: list[str] = []; buf = s

    def emit_tag(t):
        m = re.fullmatch(r"<(/?)([a-z_\-]+)>", t)
        if t in POINT:
            out.append(t); return
        if m and m.group(2) in SPAN:
            name = m.group(2)
            if not m.group(1):
                if name not in open_:
                    open_.append(name); out.append(t)
            elif name in open_:
                open_.remove(name); out.append(t)

    def text(v):
        v = v.replace("→", ", so ").replace("⇒", ", so ").replace("≥", " at least ").replace("≤", " at most ")
        if "strong" in open_:
            m = re.search(r"[，。！？、,.!?;；:：]", v)
            if m:
                if m.start(): out.append(v[:m.start()])
                out.append("</strong>"); open_.remove("strong"); v = v[m.start():]
        if v: out.append(v)

    while buf:
        idx = [k for k in (buf.find(c) for c in list(CLOSE) + ["*", "#", "`"]) if k >= 0]
        i = min(idx) if idx else -1
        if i < 0:
            text(buf); buf = ""; break
        if i > 0:
            text(buf[:i]); buf = buf[i:]
        c = buf[0]
        if c in "*#`":
            buf = buf[1:]; continue
        j = buf.find(CLOSE[c], 1)
        if j < 0 or j > 40:             # a bracket that does not close within 40 characters is not a tag / aside: speak its text
            if j > 40 or len(buf) > 40:  # (decided the same way before and after it closes, so the text never shrinks)
                buf = buf[1:]; continue
            break                       # may still close: hold it back
        tok, buf = buf[:j + 1], buf[j + 1:]
        if c in "[<":
            emit_tag(tok)
    if final:
        if buf:
            t = re.sub(r"[\[<(（【].*$", "", buf)
            if t: text(t)
        for name in reversed(open_): out.append(f"</{name}>")
    return "".join(out)


TAG_RE = re.compile(r"\[(?:laughter|breath|sigh|cough|quick_breath)\]|</?(?:laughter|strong)>")


def strip_tags(s: str) -> str:
    return TAG_RE.sub("", s)


def _tokenizer(tm):
    global _TOK
    if _TOK is None:
        with _LOCK:
            if _TOK is None:
                from transformers import AutoTokenizer
                path = os.environ.get("SOPHON_TOKENIZER")
                if not path:
                    from .env import model_dir
                    d = model_dir()
                    if d and os.path.isdir(os.path.join(d, "thinker")):
                        path = os.path.join(d, "thinker")
                if not path:
                    raise RuntimeError("sophon_realtime: cannot locate thinker tokenizer; set SOPHON_TOKENIZER")
                _TOK = AutoTokenizer.from_pretrained(path)
    return _TOK


def _talker_tokenizer():
    global _TTOK
    if _TTOK is None:
        with _LOCK:
            if _TTOK is None:
                from vllm_omni.model_executor.models.cosyvoice3.tokenizer import get_qwen_tokenizer
                from .env import model_dir
                path = os.environ.get("SOPHON_TALKER_TOKENIZER") or os.path.join(model_dir(), "talker", "CosyVoice-BlankEN")
                _TTOK = get_qwen_tokenizer(path, skip_special_tokens=True, version="cosyvoice3")
    return _TTOK


def _entry(request, key):
    info = getattr(request, "additional_information", None)
    entries = getattr(info, "entries", None)
    if not isinstance(entries, dict) or key not in entries:
        return None
    data = getattr(entries[key], "list_data", None)
    return data[0] if isinstance(data, list) and data else None


def thinker2talker_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    rid = request.external_req_id
    if _entry(request, "sophon_kind") == "check":       # full-duplex turn-check query: the talker gets no text
        if not is_finished:
            return None
        return OmniPayloadStruct(ids=IdsStruct(output=[]), meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)))
    tok = _tokenizer(transfer_manager)
    out = [int(t) for t in (request.output_token_ids or [])]
    raw = cut_tool(tok.decode(out, skip_special_tokens=True), is_finished)
    st = _ST.setdefault(rid, {"sent": 0, "raw_len": 0})
    final = is_finished
    text = tag_clean(raw, final).lstrip()
    ids = _talker_tokenizer().encode(text, allowed_special="all") if text else []
    stable = ids if final else ids[:-1]
    if not final:
        if st["sent"] == 0 and len(stable) < LEAD:
            return None
        if len(stable) < st["sent"]:
            return None
    if is_finished:
        _ST.pop(rid, None)
    else:
        st["raw_len"] = max(st["raw_len"], len(raw))
    st["sent"] = max(st["sent"], len(stable), 1)
    is_finished = final
    return OmniPayloadStruct(
        ids=IdsStruct(output=list(stable)),
        meta=MetaStruct(finished=torch.tensor(bool(is_finished), dtype=torch.bool)),
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
    )


# ---------------------------------------------------------------- talker -> code2wav: add the voice prompt once


def _voice_payload(request):
    from vllm_omni.engine.serialization import deserialize_additional_information
    from . import voices
    from .env import model_dir
    info = deserialize_additional_information(getattr(request, "additional_information", None))
    info = info if isinstance(info, dict) else {}
    def first(x):
        return x[0] if isinstance(x, (list, tuple)) and x else x
    name = str(first(info.get("speaker")) or "")
    lang = str(first(info.get("language")) or "zh").lower()[:2]
    vdir = os.environ.get("SOPHON_VOICES") or os.path.join(model_dir(), "voices")
    got, v = voices.load(vdir, name, lang, os.path.join(model_dir(), "talker"), log=logger)
    if name and got not in (name.lower().strip(), f"{name.lower().strip()}_{lang}") and name.lower().strip() not in _MISSING:
        _MISSING.add(name.lower().strip())
        logger.warning("sophon tag: voice %r not found in %s, using %s", name, vdir, got)
    return {"embed": {"speech_token": v["speech_token"], "speech_feat": v["speech_feat"], "embedding": v["embedding"],
                      "speech_token_len": torch.tensor([[v["speech_token"].shape[1]]], dtype=torch.int32)}}


_MISSING: set[str] = set()


def talker2code2wav_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    """Talker tokens -> code2wav chunks: vLLM-Omni's CosyVoice3 chunk schedule (first hop + prompt pad, pre-lookahead,
    growing hop), with three differences:
    - the voice prompt (precomputed flow conditioning) is supplied on the first call;
    - incremental: each call reads only the new talker tokens and sends codec_left_context_frames of left context plus
      the new hop (the stock chunker re-reads and re-sends every token spoken so far, so its cost per call grows with
      the reply and a long reply falls behind real time);
    - long replies are spoken in segments (segreg): the request's output restarts at every segment, the segments spoken
      before are on request._sophon_prev_out, so the token stream read here stays cumulative."""
    from vllm_omni.model_executor.stage_input_processors import cosyvoice3 as _cv
    from vllm_omni.data_entry_keys import CodesStruct
    rid = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    cfg = getattr(getattr(transfer_manager, "connector", None), "config", {}) or {}
    cfg = cfg.get("extra", cfg) if isinstance(cfg, dict) else {}
    hop0 = int(cfg.get("codec_chunk_frames", 25)); vocab = int(cfg.get("codec_vocab_size", 6561))
    look = int(cfg.get("codec_pre_lookahead_frames", 3)); max_hop = int(cfg.get("codec_max_chunk_frames", 4 * hop0))
    scale = int(cfg.get("codec_stream_scale_factor", 2)); left = int(cfg.get("codec_left_context_frames", 25))
    hold = int(cfg.get("codec_stream_hold_chunks", 1))    # this many chunks of hop0 before the hop starts growing
    rs = transfer_manager.request_payload.get(rid)
    if not isinstance(rs, dict) or "_sophon_chunk" not in rs:
        mm = _voice_payload(request)["embed"]
        pp = {k: _cv._to_cpu_tensor(mm.get(k)) for k in ("speech_token", "speech_feat", "embedding")}
        st_, sf_ = _cv.unpad_prompt_conditioning(pp["speech_token"], pp["speech_feat"], mm.get("speech_token_len"))
        pp["speech_token"] = st_
        if sf_ is not None:
            pp["speech_feat"] = sf_
        plen = int(pp["speech_token"].shape[1]) if pp["speech_token"] is not None and pp["speech_token"].ndim >= 2 else 0
        pad = ((plen + hop0 - 1) // hop0) * hop0 - plen if plen > 0 else 0
        rs = {"_sophon_chunk": {"seen": 0, "frames": [], "emitted": 0, "hop": hop0, "pad": pad, "sent_prompt": False,
                                "done": False, "prompt": pp}}
        transfer_manager.request_payload[rid] = rs
    s = rs["_sophon_chunk"]
    if s["done"]:
        return None
    prev = getattr(request, "_sophon_prev_out", None) or ()
    out = request.output_token_ids
    total = len(prev) + len(out)
    if total > s["seen"]:
        k = s["seen"]
        new = (list(prev[k:]) + list(out)) if k < len(prev) else list(out[k - len(prev):])
        s["seen"] = total
        s["frames"].extend(int(t) for t in new if 0 <= int(t) < vocab)
    frames = s["frames"]; n = len(frames); em = s["emitted"]

    def embed_once():
        if s["sent_prompt"]:
            return None
        s["sent_prompt"] = True
        return _cv._build_prompt_embed_struct(s["prompt"])

    if finished and n <= em:
        s["done"] = True
        return OmniPayloadStruct(codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                                 meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)), embed=embed_once())
    hop = s["hop"] + (s["pad"] if em == 0 else 0)
    if not finished:
        if n - em < hop + look:
            return None
        end = em + hop + look
    else:
        end = n
    start = max(0, em - left)
    payload = OmniPayloadStruct(
        codes=CodesStruct(audio=torch.tensor(frames[start:end], dtype=torch.long)),
        meta=MetaStruct(finished=torch.tensor(finished, dtype=torch.bool), stream_finished=torch.tensor(finished, dtype=torch.bool),
                        req_id=[rid], left_context_size=em - start),
        embed=embed_once())
    if not finished:
        s["emitted"] = em + hop
        s["n"] = s.get("n", 0) + 1
        if s["n"] >= hold:
            s["hop"] = min(max(hop0, max_hop), max(hop0, s["hop"] * scale))
    else:
        s["done"] = True
    return payload
