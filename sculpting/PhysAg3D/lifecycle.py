"""Own detached OMP descendants as well as ordinary subprocess groups on Linux."""

import ctypes
import os
from pathlib import Path
import signal
import threading
import time

_cleanup = threading.Lock()


def process_info(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[1]), fields[19], fields[0]
    except (OSError, ValueError, IndexError):
        return None


def descendants():
    rows = {}
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and (info := process_info(entry.name)):
            rows[int(entry.name)] = info
    owned, parents = {}, {os.getpid()}
    while parents:
        children = {pid for pid, row in rows.items() if row[0] in parents}
        for pid in children:
            owned[pid] = rows[pid]
        parents = children
    return owned


def release():
    _cleanup.acquire()
    try:
        # Subreaping keeps double-forked/setsid children attributable to this
        # worker. Re-scan during termination to catch late forks and adoption.
        for signum, seconds in ((signal.SIGTERM, 1.0), (signal.SIGKILL, 1.5)):
            deadline = time.monotonic() + seconds
            while time.monotonic() < deadline:
                living = {
                    pid: info for pid, info in descendants().items() if info[2] != "Z"
                }
                if not living:
                    return
                for pid, original in living.items():
                    current = process_info(pid)
                    if current and current[1] == original[1]:
                        try:
                            os.kill(pid, signum)
                        except ProcessLookupError:
                            pass
                time.sleep(0.05)
    finally:
        _cleanup.release()


def own(parent):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Could not own detached stage processes")

    def interrupted(signum, frame):
        if _cleanup.locked():
            return
        release()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def watch():
        while True:
            if os.getppid() != parent:
                release()
                os._exit(130)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()
