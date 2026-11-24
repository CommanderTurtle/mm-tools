"""Long replies: the talker speaks them in segments, each with a fresh LM context.

The tag-tts talker LM degrades after ~55-60 s of continuous speech context. So after about SEEK_S seconds of text the
talker ends the current segment at the next sentence end (FORCE_S: at a comma / where it is), and the *same* request
starts again from its prompt on the rest of the text. The thinker, the request and the client stream do not change; the
flow / HiFT stage gets one continuous token stream with the same voice prompt.

The talker decides the cut (find_cut) and either puts the request id in PENDING itself (Tag) or in AT_EOS, so that the
end token it samples after the segment's text restarts the request instead of finishing it (Flash, E2E). This module
patches the talker stage (all of it runs in that stage's EngineCore process):
1. scheduler: restart PENDING requests with vLLM's own preemption-and-recompute (Scheduler._preempt_request), after
   cutting the token history back to the prompt -- the request mutation vLLM-Omni uses for a replaced streaming prompt;
2. model runner: drop the cached output tokens of a restarted request;
3. chunk receiver: in async-chunk mode the talker steps only when a new chunk from the thinker has arrived, so with many
   long replies at once the thinker's pace stalls the speech. While the talker has AHEAD_MIN text tokens in hand it keeps
   stepping; when it runs low it waits for chunks as before (those that piled up are already there).
The talker -> code2wav chunker (proc.py) prepends request._sophon_prev_out, so it still reads one continuous stream."""
from __future__ import annotations

import os
import time

SEEK_S = float(os.environ.get("SOPHON_SEG_SEEK_S", "25"))      # from here, end the segment at the next sentence end
FORCE_S = float(os.environ.get("SOPHON_SEG_FORCE_S", "40"))    # no sentence end by here: end it at a comma / now
AHEAD_MIN = int(os.environ.get("SOPHON_TALKER_AHEAD", "10"))   # text tokens in hand to keep stepping without a chunk

END = "。！？!?；;…\n"           # sentence ends
SOFT = "，,、：:"               # clause ends

PENDING: set[str] = set()      # talker -> scheduler: restart these requests
AT_EOS: set[str] = set()       # talker -> scheduler: restart these requests when they sample their end token
RESTARTED: set[str] = set()    # scheduler -> talker: these requests were restarted (their next prefill is a new segment)
AHEAD: dict[str, int] = {}     # talker -> chunk receiver: text tokens received but not yet fed
_CLEAR: set[str] = set()       # scheduler -> model runner: drop the cached outputs of these requests
_installed = False


def _cjk(c: str) -> bool:
    return "\u3040" <= c <= "\u9fff" or "\uac00" <= c <= "\ud7af"


class _Clock:
    """speech_seconds() of a text that grows piece by piece, in O(new characters)."""
    def __init__(self):
        self.cjk = 0; self.words = 0; self.run = False; self.alnum = False

    def add(self, piece: str) -> float:
        for c in piece:
            if _cjk(c) or c.isspace():
                self.cjk += _cjk(c)
                if self.run and self.alnum:
                    self.words += 1
                self.run = self.alnum = False
            else:
                self.run = True; self.alnum = self.alnum or c.isalnum()
        return self.cjk * 0.22 + (self.words + (self.run and self.alnum)) * 0.4


def speech_seconds(text: str) -> float:
    """Rough spoken length: CJK characters ~0.22 s, other words ~0.4 s."""
    return _Clock().add(text)


def find_cut(ids: list[int], start: int, lo: int, piece, fin: bool) -> int | None:
    """Where the segment that starts at ids[start] ends (an index > lo, the text fed so far), or None (not decidable yet,
    or the reply ends first). From SEEK_S seconds of estimated speech: after the next sentence end; from FORCE_S: after
    a clause end within the next 12 tokens, else right there. piece(id) -> the token's text."""
    clock = _Clock()
    for j in range(start, len(ids)):
        sec = clock.add(piece(ids[j]))
        if j < lo or sec < SEEK_S:
            continue
        p = piece(ids[j]).rstrip(" ")
        nxt = piece(ids[j + 1]) if j + 1 < len(ids) else ""
        if (p and p[-1] in END) or (p.endswith(".") and nxt[:1].isspace()):        # "3.5" is no sentence end
            return None if fin and j + 1 >= len(ids) else j + 1
        if sec >= FORCE_S:
            soft = [k for k in range(j, min(len(ids), j + 12)) if piece(ids[k]).rstrip(" ")[-1:] in tuple(SOFT + ".")]
            return soft[0] + 1 if soft else j + 1
    return None


def install(log) -> None:
    global _installed
    if _installed:
        return
    _installed = True
    from vllm.v1.request import RequestStatus
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
    from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import OmniChunkTransferAdapter
    from vllm_omni.worker.gpu_model_runner import GPUModelRunner

    schedule = OmniARScheduler.schedule

    def restart_then_schedule(self, *a, **k):
        for rid in list(PENDING):
            req = self.requests.get(rid)
            if req is None:
                PENDING.discard(rid)
            elif req.status == RequestStatus.RUNNING and req in self.running:     # else (parked for a chunk): next step
                PENDING.discard(rid)
                self.running.remove(req)
                self._preempt_request(req, time.monotonic(), drop_stale_output=True)
                req._sophon_prev_out = tuple(getattr(req, "_sophon_prev_out", ())) + tuple(req._output_token_ids)
                del req._all_token_ids[req.num_prompt_tokens:]
                req._output_token_ids.clear()
                # the chunk sender's "already sent" mark counts this request's tokens: reset it with them
                self.chunk_transfer_adapter.requests_num_chunks_sent.pop(req.external_req_id, None)
                _CLEAR.add(rid); RESTARTED.add(rid)
                log.info("sophon talker: new segment for %s", rid)
        return schedule(self, *a, **k)

    OmniARScheduler.schedule = restart_then_schedule

    import vllm.v1.core.sched.scheduler as vsched
    check_stop = vsched.check_stop

    def restart_at_eos(request, *a, **k):
        rid = request.request_id
        if rid in AT_EOS and request.output_token_ids:
            sp = request.sampling_params
            if request.output_token_ids[-1] == sp.eos_token_id or request.output_token_ids[-1] in (sp.stop_token_ids or ()):
                AT_EOS.discard(rid); PENDING.add(rid)          # the end of a segment, not of the reply
                return False
        return check_stop(request, *a, **k)

    vsched.check_stop = restart_at_eos

    update_states = GPUModelRunner._update_states

    def clear_then_update(self, scheduler_output):
        for rid in list(_CLEAR):
            _CLEAR.discard(rid)
            if rid in self.requests:
                self.requests[rid].output_token_ids = []
        return update_states(self, scheduler_output)

    GPUModelRunner._update_states = clear_then_update

    process_queue = OmniChunkTransferAdapter._process_chunk_queue_legacy

    def process_queue_unless_ahead(self, queue, waiting_list, target_status, finished_load_reqs):
        ahead = []
        if target_status == RequestStatus.RUNNING:
            ahead = [r for r in queue if AHEAD_MIN > 0 and AHEAD.get(r.request_id, 0) >= AHEAD_MIN and r.status == RequestStatus.RUNNING
                     and r.request_id not in self.requests_with_ready_chunks]
            for r in ahead:
                queue.remove(r)
        process_queue(self, queue, waiting_list, target_status, finished_load_reqs)
        queue.extend(ahead)

    OmniChunkTransferAdapter._process_chunk_queue_legacy = process_queue_unless_ahead
    log.info("sophon talker: long replies in segments (seek %.0f s, force %.0f s), steps ahead with %d tokens in hand",
             SEEK_S, FORCE_S, AHEAD_MIN)
