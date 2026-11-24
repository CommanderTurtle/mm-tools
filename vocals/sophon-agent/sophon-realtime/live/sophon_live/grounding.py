"""Argument grounding for tool calls (Realtime face): a short argument value (name, email, zip, id) must come from somewhere
the agent actually saw -- the caller's words (ASR), an earlier tool result, or the instructions/tool schema. Values the model
made up (placeholders like John Doe / example.com / 12345, or a name it misheard) are sent back to the model instead of executed."""
from __future__ import annotations
import os

import difflib
import json
import re

PLACEHOLDER_NAMES = {("john", "doe"), ("jane", "doe"), ("jane", "smith"), ("john", "smith")}
PLACEHOLDER_VALUES = {"12345", "00000", "11111", "90210", "123456", "1234", "000000", "5551234567", "5555555555", "1234567890", "5550123"}
PLACEHOLDER_DOMAINS = ("example.com", "example.org", "test.com", "email.com", "domain.com")
FREE_TEXT_KEYS = {"summary", "reason", "query", "note", "notes", "message", "description", "comment", "content"}
NUM_WORDS = {"zero": "0", "oh": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
             "eight": "8", "nine": "9"}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9@._]", "", s.lower())


_UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
_SCALE = {"hundred": 100, "thousand": 1000, "million": 1000000}


def word_numbers(text: str) -> list[str]:
    """'four thousand' -> '4000', 'two hundred fifty' -> '250', '1.5 thousand' ignored; '$4,000' -> '4000'."""
    out = [re.sub(r"[,$]", "", m) for m in re.findall(r"\$?\d[\d,]*(?:\.\d+)?", text)]
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    cur = tot = 0; on = False
    for t in toks + ["<end>"]:
        if t in _UNITS: cur += _UNITS[t]; on = True
        elif t in _TENS: cur += _TENS[t]; on = True
        elif t in _SCALE and on:
            if _SCALE[t] == 100: cur *= 100
            else: tot += cur * _SCALE[t]; cur = 0
        elif t in _SCALE:                       # "a hundred", "thousand dollars"
            if _SCALE[t] == 100: cur = 100
            else: tot = _SCALE[t]
            on = True
        elif t == "and" and on: continue
        else:
            if on: out.append(str(tot + cur))
            cur = tot = 0; on = False
    return out


_MONTHS = {m: i + 1 for i, m in enumerate("january february march april may june july august september october november december".split())}
_ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17,
        "eighteenth": 18, "nineteenth": 19, "twentieth": 20, "thirtieth": 30}


_CURRENCY = {"dollar": "usd", "buck": "usd", "euro": "eur", "pound": "gbp", "sterling": "gbp", "quid": "gbp", "yen": "jpy",
             "yuan": "cny", "renminbi": "cny", "rmb": "cny", "canadian": "cad", "australian": "aud", "aussie": "aud",
             "swiss": "chf", "franc": "chf", "rupee": "inr", "peso": "mxn", "won": "krw", "kiwi": "nzd", "zealand": "nzd",
             "hong kong": "hkd", "singapore": "sgd", "krona": "sek", "krone": "nok", "real": "brl", "rand": "zar", "ruble": "rub", "lira": "try"}


def spoken_forms(text: str) -> list[str]:
    """Extra normalised candidates from conversational speech: dates -> YYYY-MM-DD / MM-DD, spoken emails, grouped digits."""
    t = text.lower().replace("-", " ")
    out = [c for w, c in _CURRENCY.items() if re.search(rf"\b{w}", t)]
    # dates: "december 14th, 2026" / "december fourteenth twenty twenty six" / "the 14th of december"
    t2 = re.sub(r"twenty twenty (one|two|three|four|five|six|seven|eight|nine)", lambda m: "202" + str(_UNITS[m.group(1)]), t)
    for m in re.finditer(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(?:the\s+)?(\d{1,2}|[a-z]+(?:\s[a-z]+)?)(?:st|nd|rd|th)?(?:,?\s*(\d{4}))?", t2):
        mo = _MONTHS[m.group(1)]; dtok = m.group(2)
        d = int(dtok) if dtok.isdigit() else None
        if d is None:
            parts = dtok.split()
            if parts[0] in _ORD: d = _ORD[parts[0]]
            elif parts[0] in _TENS and len(parts) > 1 and parts[1] in _ORD: d = _TENS[parts[0]] + _ORD[parts[1]]
        if d:
            out.append(f"{mo:02d}-{d:02d}")
            if m.group(3): out.append(f"{m.group(3)}-{mo:02d}-{d:02d}")
    for m in re.finditer(r"(?:the\s+)?(\d{1,2}|[a-z]+)(?:st|nd|rd|th)?\s+of\s+(january|february|march|april|may|june|july|august|september|october|november|december)", t2):
        dtok = m.group(1); d = int(dtok) if dtok.isdigit() else _ORD.get(dtok)
        if d: out.append(f"{_MONTHS[m.group(2)]:02d}-{d:02d}")
    # seats like "fourteen C" / "14 C"
    for m in re.finditer(r"\b(?:(twenty|thirty)\s+)?(\d{1,2}|[a-z]+)\s+([a-f])\b", t):
        n = m.group(2); v = int(n) if n.isdigit() else _UNITS.get(n, _TENS.get(n))
        if v is not None and m.group(1) and v < 10: v += _TENS[m.group(1)]
        if v: out.append(f"{v}{m.group(3)}")
    # spoken email: "name dot surname thirty one at gmail dot com"
    if " at " in t and " dot " in t:
        e = t
        for w, v in list(_UNITS.items()) + list(_TENS.items()): e = re.sub(rf"\b{w}\b", str(v), e)
        e = re.sub(r"(\d+)\s+(\d)\b", lambda m: str(int(m.group(1)) + int(m.group(2))) if int(m.group(1)) % 10 == 0 and int(m.group(1)) >= 20 else m.group(0), e)
        e = e.replace(" underscore ", "_").replace(" dash ", "-").replace(" at ", "@").replace(" dot ", ".")
        for m in re.finditer(r"[a-z0-9._ \-]+@[a-z0-9. ]+", e):
            out.append(re.sub(r"\s+", "", m.group(0)).strip("."))
    # grouped digits: "eight sixteen" -> 816, "nineteen one twenty two" -> 19122
    toks = re.findall(r"[a-z]+|\d+", t); run = []
    for tok in toks + ["<end>"]:
        v = _UNITS.get(tok, _TENS.get(tok)) if tok != "oh" else 0
        if tok.isdigit(): v = int(tok)
        if v is not None and tok != "<end>":
            if run and tok in _UNITS and run[-1] in _TENS and _UNITS[tok] < 10:
                run[-1] = str(int(_TENS[run[-1]]) + _UNITS[tok]) if False else run[-1]
            run.append(tok)
        else:
            if len(run) >= 2:
                vals, i = [], 0
                while i < len(run):
                    a_ = run[i]
                    if a_ in _TENS and i + 1 < len(run) and run[i + 1] in _UNITS and 0 < _UNITS[run[i + 1]] < 10:
                        vals.append(str(_TENS[a_] + _UNITS[run[i + 1]])); i += 2; continue
                    vals.append(str(_UNITS[a_] if a_ in _UNITS else _TENS[a_] if a_ in _TENS else 0 if a_ == "oh" else int(a_))); i += 1
                out.append("".join(vals))
            run = []
    return out


def spelled_joins(text: str) -> list[str]:
    """'Y, U, S, U, F' -> 'yusuf'; '1 9 1 2 2' / 'one nine one two two' -> '19122'; 'W 2 3 7 8' -> 'w2378'."""
    text = re.sub(r"(\w)['’](\w)", r"\1\2", text)              # it's -> its (not a spelled "s")
    toks = re.findall(r"[A-Za-z]+|\d+", text)
    out, run = [], []
    for t in toks + ["<end>"]:
        tl = t.lower()
        unit = (len(t) == 1 and t.isalpha()) or t.isdigit() and len(t) <= 3 or tl in NUM_WORDS
        if unit and t != "<end>":
            run.append(NUM_WORDS.get(tl, tl))
        else:
            if len(run) >= 3: out.append("".join(run))
            run = []
    return out


def _values(args, key=""):
    if isinstance(args, dict):
        for k, v in args.items(): yield from _values(v, k)
    elif isinstance(args, list):
        for v in args: yield from _values(v, key)
    elif isinstance(args, (str, int)) and not isinstance(args, bool):
        yield key, str(args)


def check(name: str, args: dict, heard: list[str], seen: list[str]) -> list[tuple[str, str, str]]:
    """Returns [(key, value, why)] for ungrounded values; [] = OK."""
    heard_n = [_norm(h) for h in heard]
    corpus = " ".join(heard_n + [_norm(s) for s in seen])
    joins = ([j for h in heard for j in spelled_joins(h)] + [n for h in heard for n in word_numbers(h)]
             + [_norm(x) for h in heard for x in spoken_forms(h)])
    words = set(re.findall(r"[a-z0-9@._]+", " ".join(heard + seen).lower())) | set(joins)
    bad = []
    a = args if isinstance(args, dict) else {}
    fn, ln = str(a.get("first_name", "")).lower(), str(a.get("last_name", "")).lower()
    if (fn, ln) in PLACEHOLDER_NAMES and not any(fn in h and ln in h for h in heard_n):
        bad.append(("first_name/last_name", f"{fn} {ln}", "placeholder name"))
    for k, v in _values(args):
        if k.lower() in FREE_TEXT_KEYS or len(v.split()) > 3:
            continue
        nv = _norm(v)
        if len(nv) < 3:
            continue
        if nv in PLACEHOLDER_VALUES or any(nv.endswith(d) for d in PLACEHOLDER_DOMAINS):
            if nv not in corpus: bad.append((k, v, "placeholder value")); continue
        if os.environ.get("SOPHON_LIVE_GROUND_MODE", "full") == "placeholder":   # only block obvious made-up values
            continue
        if nv in corpus or nv.lstrip("#") in corpus or nv in joins:
            continue
        dm = re.fullmatch(r"\d{4}-(\d{2})-(\d{2})", v.strip())
        if dm and (dm.group(1) + dm.group(2)) in joins:                  # year usually not said
            continue
        if "@" in nv:                                            # emails are often said as "name at domain dot com"
            loc, _, dom = nv.partition("@")
            if loc and loc in corpus and dom.split(".")[0] in corpus: continue
        best = max((difflib.SequenceMatcher(None, nv, w).ratio() for w in words), default=0.0)
        if best >= 0.85:
            continue
        parts = [_norm(p) for p in re.split(r"[\s,]+", v) if _norm(p)]
        if len(parts) > 1 and all(p in corpus or p in joins or max((difflib.SequenceMatcher(None, p, w).ratio() for w in words), default=0) >= 0.85 for p in parts):
            continue
        bad.append((k, v, f"not said by the caller (closest match {best:.2f})"))
    return bad


def guard_message(bad, heard: list[str]) -> str:
    vals = "; ".join(f"{k}={v!r} ({why})" for k, v, why in bad)
    recent = " | ".join(h.strip() for h in heard[-3:] if h.strip())
    joins = [j for h in heard[-3:] for j in spelled_joins(h)]
    s = (f"NOT EXECUTED. These argument values were not provided by the caller: {vals}. Do not guess or invent values. "
         f"Ask the caller for the missing information (for names and emails, ask them to spell it letter by letter), "
         f"read it back, then call the tool again with exactly what they said.")
    if recent:
        s += f" Caller's recent words (speech transcript): \"{recent}\"."
    if joins:
        s += f" Spelled-out sequences heard: {', '.join(joins)}."
    return s


def failure_hint(output: str, heard: list[str]) -> str:
    if not re.search(r"not found|no user|could not find|cannot find|does not exist|invalid|error", output, re.I):
        return ""
    recent = " | ".join(h.strip() for h in heard[-2:] if h.strip())
    joins = [j for h in heard[-3:] for j in spelled_joins(h)]
    if not recent:
        return ""
    s = f"\n[Caller's recent words (speech transcript): \"{recent}\"."
    if joins: s += f" Spelled-out sequences heard: {', '.join(joins)}."
    return s + " If a name or id may have been misheard, ask the caller to spell it letter by letter and retry with the spelled version.]"
