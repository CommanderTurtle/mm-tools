"""Step-by-step thinker -> talker forwarding: each step sends the talker the cleaned reply-text tokens the thinker has written so far (cumulative list).
- Only "stable" tokens are sent (the last one may still change with following characters; it is released only at the end); no new tokens means nothing is sent (so the talker does not advance either -- better to wait than to pad)
- At the start, buffer LEAD tokens before sending the first chunk, to give the talker some headroom
- Strip parenthesized stage directions / markdown symbols (same as the testbench gateway's CLEAN)"""
import os
import re
import threading

import torch
from .tools import cut_tool
from vllm_omni.data_entry_keys import IdsStruct, MetaStruct, OmniPayloadStruct
from vllm_omni.model_executor.stage_input_processors.tts_utils import extract_language_from_request, extract_speaker_from_request

CLEAN = re.compile(r"[（(\[【][^）)\]】]*[）)\]】]|[*#_`~>]")
OPEN = re.compile(r"[（(\[【][^）)\]】]*$")
LEAD = int(os.environ.get("SOPHON_LEAD", "3"))
_TOK = None
_LOCK = threading.Lock()
_ST: dict[str, dict] = {}


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
                    for obj in (tm, getattr(tm, "vllm_config", None), getattr(tm, "model_config", None)):
                        mc = getattr(obj, "model_config", obj)
                        path = getattr(mc, "tokenizer", None) or getattr(mc, "model", None)
                        if isinstance(path, str) and os.path.isdir(path):
                            break
                        path = None
                if not path:
                    raise RuntimeError("sophon_realtime: cannot locate thinker tokenizer; set SOPHON_TOKENIZER")
                _TOK = AutoTokenizer.from_pretrained(path)
    return _TOK


def _cut_repetition(t: str, run: int = 4) -> str:
    """When the model repeats itself (e.g. "mmmmmm..."), keep only two and don't pass the rest to the talker."""
    out = []; n = 0
    for ch in t:
        n = n + 1 if out and ch == out[-1] else 1
        if n > run: return "".join(out[: len(out) - (run - 2)])
        out.append(ch)
    return t


POINT = {"[laughter]", "[breath]", "[sigh]", "[cough]", "[quick_breath]"}
SPAN = {"laughter", "strong"}
CLOSE = {"[": "]", "<": ">", "(": ")", "（": "）", "【": "】"}


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




def clean_text_old(s: str, final: bool) -> str:
    s = CLEAN.sub("", s)
    return OPEN.sub("", s)


def _entry(request, key):
    info = getattr(request, "additional_information", None)
    entries = getattr(info, "entries", None)
    if not isinstance(entries, dict) or key not in entries:
        return None
    data = getattr(entries[key], "list_data", None)
    return data[0] if isinstance(data, list) and data else None


def thinker2talker_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    rid = request.external_req_id
    if _entry(request, "sophon_kind") == "check":       # full-duplex turn-check query: the thinker only emits one letter; give the talker no text
        if not is_finished:
            return None
        return OmniPayloadStruct(ids=IdsStruct(output=[]), meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)))
    tok = _tokenizer(transfer_manager)
    out = [int(t) for t in (request.output_token_ids or [])]
    # tool calls are never spoken: the talker only gets the text before <tool_call> (a partial "<too" at the end is held back)
    text = tag_clean(cut_tool(tok.decode(out, skip_special_tokens=True), is_finished), is_finished).lstrip()
    if os.environ.get("SOPHON_REP_GUARD", "0") == "1": text = _cut_repetition(text)
    ids = tok(text, add_special_tokens=False).input_ids if text else []
    stable = ids if is_finished else ids[:-1]
    st = _ST.setdefault(rid, {"sent": 0})
    if not is_finished:
        if st["sent"] == 0 and len(stable) < LEAD:
            return None
        if len(stable) <= st["sent"]:
            return None
    else:
        _ST.pop(rid, None)
    st["sent"] = len(stable)
    return OmniPayloadStruct(
        ids=IdsStruct(output=list(stable)),
        meta=MetaStruct(finished=torch.tensor(bool(is_finished), dtype=torch.bool)),
        speaker=extract_speaker_from_request(request),
        language=extract_language_from_request(request),
    )
