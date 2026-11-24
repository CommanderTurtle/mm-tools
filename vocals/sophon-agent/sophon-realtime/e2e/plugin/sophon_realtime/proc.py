"""Step-by-step thinker -> E2E talker forwarding (Sophon-Realtime-32B-E2E).
Every thinker step sends the talker the *cumulative* reply text written so far (as code points):
- sound tags are kept whole and passed through (whitelist: [laughter] [breath] [sigh] [cough] [quick_breath],
  <laughter>..</laughter>, <strong>..</strong>); anything else in brackets and markdown symbols is dropped; an unclosed
  bracket at the end is held back; <strong> is auto-closed at punctuation; open spans are closed when the thinker finishes
  (same rules as the Tag one-box server sophon_tag_server.TagStream).
- the last token is held back until the next step (it may still merge); everything is released when the thinker finishes.
- a chunk is sent on *every* thinker step once LEAD tokens are ready (heartbeat): in async-chunk mode the talker only steps
  when a chunk arrives, and it needs ~4 steps per text token (bistream 5 text : 15 speech + the text positions)."""
import os
import re
import threading
import time

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
    ids = [ord(c) for c in text]              # the E2E talker tokenizes and reads the tags itself: send the text as code points
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


# ---------------------------------------------------------------- talker -> decoder
# Every valid talker row is one decoder chunk (0.32 s of audio). The first chunks go out one by one (first audio as early as
# possible); after that BUNDLE chunks travel together, so the decoder stage runs fewer, larger steps under load.
BUNDLE = int(os.environ.get("SOPHON_E2E_BUNDLE", "2")); SINGLE_FIRST = int(os.environ.get("SOPHON_E2E_SINGLE_FIRST", "4"))
# With TICK > 0 the later chunks of ALL replies are handed over on a shared clock (every TICK seconds, whatever has accumulated), so they
# reach the decoder together and it renders them as one large batch instead of many small ones. 0 (default) = send every BUNDLE chunks per reply.
# Measured 2026-11-12 with 0.64: no gain on H100 or RTX PRO 6000 (the GPU, shared by the three stages, is the limit, not the batch size), so it is off.
TICK = float(os.environ.get("SOPHON_E2E_TICK", "0"))


def talker2decoder_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    from collections.abc import Mapping
    from vllm_omni.data_entry_keys import CodesStruct
    rid = request.external_req_id
    finished = bool(is_finished or request.is_finished())
    if not hasattr(transfer_manager, "request_payload"):
        transfer_manager.request_payload = {}
    st = transfer_manager.request_payload.get(rid)
    if not isinstance(st, dict) or "_e2e" not in st:
        st = {"_e2e": {"terminal": False, "rows": [], "sent": 0}}
        transfer_manager.request_payload[rid] = st
    s = st["_e2e"]
    if s["terminal"]:
        return None
    if isinstance(multimodal_output, Mapping):
        codes = multimodal_output.get("codes", {})
        ac = codes.get("audio") if isinstance(codes, Mapping) else None
        meta = multimodal_output.get("meta", {})
        fv = meta.get("codec_frame_valid") if isinstance(meta, Mapping) else None
        if isinstance(ac, torch.Tensor) and ac.numel() > 0:
            ok = bool(fv.reshape(-1)[-1].item()) if isinstance(fv, torch.Tensor) and fv.numel() else True
            if ok:
                s["rows"].append((ac[-1] if ac.ndim == 2 else ac).detach().to("cpu", torch.long).reshape(-1).contiguous())
    if s["sent"] < SINGLE_FIRST or TICK <= 0:
        need = 1 if s["sent"] < SINGLE_FIRST else max(1, BUNDLE)
        if not finished and len(s["rows"]) < need:
            return None
    elif not finished:
        tick = int(time.time() / TICK)
        if not s["rows"] or s.get("tick") == tick:
            return None
        s["tick"] = tick
    if not s["rows"]:
        s["terminal"] = True
        return OmniPayloadStruct(codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)), meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)))
    out = torch.cat(s["rows"]); s["sent"] += len(s["rows"]); s["rows"] = []
    if finished:
        s["terminal"] = True
    return OmniPayloadStruct(codes=CodesStruct(audio=out), meta=MetaStruct(request_id=rid, left_context_size=0, finished=torch.tensor(finished, dtype=torch.bool)))
