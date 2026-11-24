"""Front + back-office mode for the Realtime face (SOPHON_LIVE_BACKEND=<model>, e.g. deepseek-v4-pro).

Sophon (front) keeps the voice: it listens, takes turns, speaks, and calls one tool, `delegate(task)`, whenever the caller's
request needs a lookup or an action. A stronger text model (back office) then sees the whole call (caller turns from the
gateway's ASR + what Sophon said + earlier back-office actions), the client's real tools and instructions, and runs the tool
loop; its tool calls go to the client as normal function calls. Its final answer goes back to Sophon as the delegate result,
and Sophon says it. Same division of labour as GPT-Live delegation.
"""
from __future__ import annotations

import json
import logging
import os
import time

import aiohttp

log = logging.getLogger("sophon_live.backend")

MODEL = os.environ.get("SOPHON_LIVE_BACKEND", "")
URL = os.environ.get("SOPHON_LIVE_BACKEND_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
KEY_FILE = os.environ.get("SOPHON_LIVE_BACKEND_KEY_FILE", "")
MAX_STEPS = int(os.environ.get("SOPHON_LIVE_BACKEND_STEPS", "8"))
ALL = os.environ.get("SOPHON_LIVE_BACKEND_SCOPE", "tools") == "all"      # also delegate reasoning / knowledge questions (no client tools)
THINK = os.environ.get("SOPHON_LIVE_BACKEND_THINK", "0") == "1"
THINK_BUDGET = int(os.environ.get("SOPHON_LIVE_BACKEND_THINK_BUDGET", "0"))   # cap thinking tokens (0 = no cap)
MAX_FAILS = int(os.environ.get("SOPHON_LIVE_BACKEND_MAX_FAILS", "2"))   # failed lookups per tool per call before we stop guessing
RELAY = os.environ.get("SOPHON_LIVE_BACKEND_INPUT", "relay") == "relay"   # relay: back office sees only what the front (who hears the audio) hands over; asr: also the side ASR transcript           # let the back office think before answering


def _key() -> str:
    k = os.environ.get("SOPHON_LIVE_BACKEND_KEY", "")
    if not k and KEY_FILE and os.path.exists(KEY_FILE):
        for line in open(KEY_FILE):
            line = line.strip()
            if line and not line.startswith("#"):
                k = line.split("=", 1)[-1].strip().strip('"').strip("'")
                break
    return k


DELEGATE_TOOL = {"type": "function", "name": "delegate",
                 "description": "Hand the caller's request to the back-office colleague, who can look things up and make changes in the "
                                "company systems, and is good at careful reasoning. The colleague cannot hear the call: everything they "
                                "know comes from what you write in task.",
                 "parameters": {"type": "object", "properties": {"task": {"type": "string", "description":
                                "Everything the colleague needs, in the caller's own words: what they want now, and every detail they "
                                "said, written out exactly (full names with the spelling if they spelled it, every digit of numbers, zip "
                                "codes, order or reservation IDs, emails letter by letter, dates, amounts, item names, choices), plus "
                                "whether they clearly said yes to a read-back. For a question, write the whole question as asked, "
                                "including every item and every answer option."}}, "required": ["task"]}}

FRONT_RULES_EN = (
    "\n\nHow you work: you are the voice on the phone. A back-office colleague does every lookup and every change through the "
    "company systems; you cannot see those systems yourself."
    "\n- When the caller's request needs account data, a lookup, or an action, first say a very short lead-in (\"Let me check that\"), "
    "then call delegate. The colleague cannot hear the call, so the task must contain every detail the caller has given so far, "
    "written out exactly: names and spellings, all digits, emails, IDs, dates, items, choices, and whether they confirmed. "
    "If the caller spells or repeats something, pass the spelled version. Collect what is obviously needed (for example the "
    "name and zip code, or the order number) before delegating, but don't interrogate."
    "\n- Phone lines are noisy. Names, emails and IDs are easy to mishear: unless you are sure you heard one exactly, ask the "
    "caller to spell it letter by letter, then read it back letter by letter and wait for a yes before you pass it on. Read "
    "numbers back digit by digit. Do this once per detail, not again for details already confirmed."
    "\n- When the delegate result comes back, tell the caller in one or two natural spoken sentences. If it says something is needed "
    "from the caller (a detail, a spelling, or a yes to confirm), ask exactly that."
    "\n- Never invent account details, prices or outcomes yourself."
    "\n- Say one thing at a time and keep replies short.")

BACK_SYS = (
    "You are the back-office agent behind a voice assistant on a live phone call. The voice assistant (front) talks to the "
    "caller and hears them; you cannot hear the call. You see what the front has relayed to you in this call (its requests "
    "carry the caller's details as the caller said them; spelled letters like 'Y, U, S, U, F' mean 'Yusuf'; digits said one "
    "by one form a number), what the front said to the caller, and your earlier actions in this call.\n"
    "Rules:\n"
    "1. Follow the company policy below exactly.\n"
    "2. Only use argument values the caller said or that came from tool results. Never guess or use placeholders. If a lookup "
    "fails, you may try ONE obvious variant of what was heard (spelling or digit grouping). Never make more than two failed "
    "calls in one request: every failed call counts against the call. If still not found, ask the caller to spell it letter "
    "by letter or repeat the digits.\n"
    "3. Before any action that changes something, the caller must have explicitly confirmed the exact details in the transcript. "
    "If not, do not act: reply with the exact details to read back and ask for a yes. If the caller's latest turns contain a "
    "clear yes after those details were read back (or the front's request says the caller confirmed), act now without asking "
    "again.\n"
    "4. When you need something from the caller, or when you are done, reply WITHOUT a tool call, in exactly this form:\n"
    "SAY: <the words the front will read aloud to the caller, verbatim>\n"
    "Those words go straight to the caller: one or two short, natural spoken sentences addressed to the caller ('you'). "
    "Never mention the front, yourself, the policy, tools or your reasoning; no lists or markdown. Example: "
    "'SAY: I found your order. Before I change anything, can you confirm your name and zip code for me?'\n"
    "5. Do as much as you can per turn: chain lookups (user, then order, then product) before replying.\n"
    "6. Fill arguments in the form the caller used, unless the tool description asks for a specific format: keep dates, names "
    "and search words as said (for example a date said as 'August 20' stays 'August 20'; 'mechanical keyboards' stays plural). "
    "Use JSON types: true / false for yes-no values, numbers for amounts. Include every detail the caller gave that fits a "
    "parameter, and don't add parameters the caller didn't ask for.\n\n"
    "Company policy and instructions:\n")


FRONT_RULES_ALL_EN = (
    "\n\nHow you work: a colleague who is very good at reasoning, maths and facts works with you; you are the voice."
    "\n- For small talk or a simple reply, answer yourself."
    "\n- For anything that needs careful reasoning, calculation, counting, logic, a puzzle or a specific fact, first say a very "
    "short lead-in (\"Let me think about that\"), then call delegate with the user's full question exactly as asked, every item and number in it, and all "
    "answer options; the colleague cannot hear the user."
    "\n- When the delegate result comes back, say it to the user naturally, keeping the final answer exactly as given.")

ANSWER_SYS = (
    "You are the expert behind a voice assistant in a live spoken conversation. The voice assistant heard the user and passed "
    "you the question as the user asked it. Work out the correct answer "
    "carefully. Then reply in exactly this form:\n"
    "SAY: <one or two short spoken sentences for the user that state the final answer clearly; if answer options were given, "
    "say 'The answer is' followed by the option, then a very short reason>\n"
    "No lists or markdown in the SAY part.\n")


def _say(text: str) -> str:
    """Keep only the words meant for the caller (after 'SAY:'); fall back to the last paragraph."""
    t = (text or "").strip()
    i = t.upper().rfind("SAY:")
    if i >= 0:
        t = t[i + 4:].strip()
    else:
        paras = [p.strip() for p in t.split("\n\n") if p.strip()]
        t = paras[-1] if paras else ""
    return t.strip().strip('"').strip() or "Okay."


class BackOffice:
    def __init__(self, sess_id: str):
        self.sid = sess_id
        self.dialog: list = []            # (role, text, t) — 'caller' / 'front'
        self.actions: list = []           # "name(args) -> result"
        self.fails: dict = {}             # tool name -> failed calls in this call

    def add(self, role: str, text: str) -> None:
        if text and text.strip():
            self.dialog.append((role, text.strip(), time.monotonic()))

    def transcript(self) -> str:
        if RELAY:
            lines = [f"{'Front relayed' if r == 'relay' else 'Front said to the caller'}: {t}" for r, t, _ in self.dialog[-60:] if r != "caller"]
        else:
            lines = [f"{'Caller' if r == 'caller' else ('Front relayed' if r == 'relay' else 'Front')}: {t}" for r, t, _ in self.dialog[-60:]]
        return "\n".join(lines) or "(nothing yet)"

    async def run(self, instructions: str, tools: list, task: str, call_tool) -> str:
        """call_tool(name, args) -> awaitable str (the client's function_call_output)."""
        ftools = []
        for t in tools:
            if not isinstance(t, dict):
                continue
            f = t.get("function") or t
            ftools.append({"type": "function", "function": {"name": f.get("name"), "description": f.get("description", ""),
                                                             "parameters": f.get("parameters") or {"type": "object", "properties": {}}}})
        prior = self.transcript()
        self.add("relay", task)
        user = (("Earlier in this call:\n" if RELAY else "Call transcript so far:\n") + prior +
                ("\n\nYour earlier actions in this call:\n" + "\n".join(self.actions[-30:]) if self.actions else "") +
                f"\n\nThe front asks you now: {task}")
        msgs = [{"role": "system", "content": (BACK_SYS + (instructions or "")) if ftools else ANSWER_SYS}, {"role": "user", "content": user}]
        key = _key()
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600 if THINK else 120)) as cs:
            for step in range(MAX_STEPS):
                body = {"model": MODEL, "messages": msgs, "temperature": 0.2, "enable_thinking": THINK}
                if ftools: body["tools"] = ftools
                if THINK and THINK_BUDGET: body["thinking_budget"] = THINK_BUDGET
                t0 = time.monotonic()
                for k in range(3):
                    try:
                        async with cs.post(URL, json=body, headers={"Authorization": "Bearer " + key}) as r:
                            j = await r.json(content_type=None)
                        m = j["choices"][0]["message"]; break
                    except Exception as e:  # noqa: BLE001
                        log.warning("backend %s: call failed (%s)", self.sid, repr(e)[:160])
                        if k == 2:
                            return "Sorry, our system is slow right now. Could you give me a moment and say that again?"
                log.info("backend %s: step %d %.1fs tool_calls=%d", self.sid, step, time.monotonic() - t0, len(m.get("tool_calls") or []))
                tcs = m.get("tool_calls") or []
                if not tcs:
                    return _say(m.get("content") or "")
                msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": tcs})
                for tc in tcs:
                    name = tc["function"]["name"]
                    try:
                        args = json.loads(tc["function"].get("arguments") or "{}")
                    except Exception:  # noqa: BLE001
                        args = {}
                    if self.fails.get(name, 0) >= MAX_FAILS and name.startswith(("get_", "find_", "search_")):
                        out = ("Not called: this lookup already failed %d times in this call. Do not guess other variants. Ask the caller "
                               "to spell the value letter by letter (or read the digits one by one) and wait for it." % self.fails[name])
                    else:
                        out = await call_tool(name, args)
                        if out.lower().startswith("error") or "not found" in out.lower()[:200]:
                            self.fails[name] = self.fails.get(name, 0) + 1
                    self.actions.append(f"{name}({json.dumps(args, ensure_ascii=False)}) -> {out[:600]}")
                    msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": out})
        return "Let me look into that a bit more. Could you hold on a moment?"
