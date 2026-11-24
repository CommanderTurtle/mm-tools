# Copyright 2026 Adventists AI. Apache-2.0.
"""Sound tags written by the thinker -> per-character event marks for the talker.
[breath]=1, [laughter]=2, [cough]=6, [sigh]=8 attach to the previous non-punctuation character (start of reply -> lead);
<laughter>...</laughter> -> span 0, <strong>...</strong> -> span 1. Tags themselves are never spoken."""
import re
PT = {"[breath]": 1, "[laughter]": 2, "[cough]": 6, "[sigh]": 8}; SP = {"laughter": 0, "strong": 1}


def walk(t):
    it, open_ = [], set(); lead = 0
    for m in re.finditer(r"\[[a-z_]+\]|</?[a-z_]+>|.", t, re.S):
        s = m.group(0)
        if s in PT:
            j = len(it) - 1
            while j >= 0 and (it[j]["c"].isspace() or re.match(r"[\W_]", it[j]["c"])): j -= 1
            if j < 0: lead = lead or PT[s]
            elif not it[j]["a"]: it[j]["a"] = PT[s]
            it.append(dict(c=" ", a=0, sp=[0, 0])); continue
        mm = re.fullmatch(r"<(/?)([a-z_]+)>", s)
        if mm:
            (open_.discard if mm.group(1) else open_.add)(mm.group(2)); it.append(dict(c=" ", a=0, sp=[0, 0])); continue
        if s.startswith("[") and len(s) > 1: it.append(dict(c=" ", a=0, sp=[0, 0])); continue
        sp = [0, 0]
        for k in open_:
            if k in SP: sp[SP[k]] = 1
        it.append(dict(c=s, a=0, sp=sp))
    out = []
    for x in it:
        if x["c"].isspace():
            if out and out[-1]["c"] == " ": out[-1]["a"] = out[-1]["a"] or x["a"]; continue
            x = dict(x, c=" ")
        out.append(x)
    while out and out[0]["c"] == " ": out.pop(0)
    while out and out[-1]["c"] == " ": out.pop()
    o2 = []
    for x in out:
        if x["c"] in "，。,." and o2 and o2[-1]["c"] == " ":
            p = o2.pop()
            if p["a"] and o2 and not o2[-1]["a"]: o2[-1]["a"] = p["a"]
        o2.append(x)
    return o2, lead
