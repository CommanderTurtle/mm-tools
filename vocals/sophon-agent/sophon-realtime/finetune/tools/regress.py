"""Regression check for a fine-tuned thinker LoRA, served by vLLM next to the shipped tag LoRA.

    vllm serve Sophon-Realtime-v1/base --served-model-name base --enable-lora --max-lora-rank 64 \
        --lora-modules tag=Sophon-Realtime-v1/adapters/tag mine=my_lora_vllm --max-loras 2
    python regress.py --url http://127.0.0.1:8000 --data my_eval.jsonl --base tag --new mine

`my_eval.jsonl` uses the ms-swift format (messages + audios); the last assistant message, if present, is
ignored. Reports, for both adapters: sound-tag rate, malformed markup, wrong-language replies, reply length,
and how often the two give the same greedy reply. Turn-taking questions go to the plain base model in
Sophon-Realtime, so a reply LoRA cannot change them.
"""
import argparse
import base64
import json
import os
import re
import statistics
import urllib.request

TAG_OK = re.compile(r"\[(laughter|cough|sigh|breath)\]|</?(laughter|strong)>")
TAG_ANY = re.compile(r"\[[a-z_ ]+\]|</?[a-z_]+>")
CJK = re.compile(r"[一-鿿]")


def load(path, n):
    rows = []
    base = os.path.dirname(os.path.abspath(path))
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        msgs = r["messages"]
        if msgs and msgs[-1]["role"] == "assistant":
            msgs = msgs[:-1]
        auds = [a if os.path.isabs(a) or os.path.exists(a) else os.path.join(base, a) for a in r.get("audios", [])]
        rows.append((msgs, auds))
    return rows[:n] if n else rows


def to_openai(msgs, audios):
    out, k = [], 0
    for m in msgs:
        parts = m["content"].split("<audio>")
        if len(parts) == 1:
            out.append({"role": m["role"], "content": m["content"]})
            continue
        content = []
        for i, p in enumerate(parts):
            if p:
                content.append({"type": "text", "text": p})
            if i < len(parts) - 1:
                b64 = base64.b64encode(open(audios[k], "rb").read()).decode()
                fmt = os.path.splitext(audios[k])[1].lstrip(".") or "wav"
                content.append({"type": "text", "text": "<|audio|>"})
                content.append({"type": "input_audio", "input_audio": {"data": b64, "format": fmt}})
                k += 1
        out.append({"role": m["role"], "content": content})
    return out


def ask(url, model, msgs, max_tokens):
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens, "messages": msgs}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"] or ""


def stats(replies, systems):
    n = len(replies)
    tag = sum(bool(TAG_OK.search(r)) for r in replies)
    bad = sum(any(not TAG_OK.fullmatch(t) for t in TAG_ANY.findall(r)) for r in replies)
    # expected language = language of the system prompt / user text (brand names ignored)
    exp = [bool(CJK.search(s.replace("AI降临派", "").replace("智子", ""))) for s in systems]
    wrong_lang = sum(e != bool(CJK.search(TAG_ANY.sub("", r).replace("AI降临派", "")))
                     for r, e in zip(replies, exp))
    lens = [len(TAG_ANY.sub("", r)) for r in replies]
    return {"n": n, "tag_rate": round(tag / n, 3), "malformed": bad, "wrong_language": wrong_lang,
            "median_chars": statistics.median(lens) if lens else 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="tag")
    ap.add_argument("--new", default="mine")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max_tokens", type=int, default=160)
    ap.add_argument("--out", default="regress_out.jsonl")
    a = ap.parse_args()
    rows = load(a.data, a.n)
    R = {a.base: [], a.new: []}
    systems = []
    with open(a.out, "w", encoding="utf-8") as f:
        for msgs, auds in rows:
            om = to_openai(msgs, auds)
            systems.append(next((m["content"] for m in msgs if m["role"] == "system"), "") +
                           " ".join(m["content"] for m in msgs if m["role"] == "user"))
            for name in (a.base, a.new):
                R[name].append(ask(a.url, name, om, a.max_tokens))
            f.write(json.dumps({a.base: R[a.base][-1], a.new: R[a.new][-1]}, ensure_ascii=False) + "\n")
    same = sum(x == y for x, y in zip(R[a.base], R[a.new]))
    print(json.dumps({a.base: stats(R[a.base], systems), a.new: stats(R[a.new], systems),
                      "identical": f"{same}/{len(rows)}"}, ensure_ascii=False, indent=1))
    print(f"replies: {a.out}")


if __name__ == "__main__":
    main()
