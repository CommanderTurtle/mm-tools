"""Keep the native worker's process group tied to the serving process."""

import os
import select
import signal
import subprocess
import sys


class Stopped(BaseException):
    pass


def main():
    parent_pid = int(os.environ["AUDIO8_PARENT_PID"])
    if os.getppid() != parent_pid:
        raise SystemExit("Studio closed")
    try:
        parent = os.pidfd_open(parent_pid)
    except ProcessLookupError:
        raise SystemExit("Studio closed") from None
    child = worker = None
    launching, interrupted, force, code = True, False, False, 143

    def stop(*_):
        nonlocal interrupted
        interrupted = True
        if not launching:
            raise Stopped

    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        child = subprocess.Popen(sys.argv[1:], start_new_session=True)
        launching = False
        if interrupted:
            raise Stopped
        try:
            worker = os.pidfd_open(child.pid)
        except ProcessLookupError:
            code = child.wait()
        else:
            ready, _, _ = select.select([parent, worker], [], [])
            force = parent in ready
            if not force:
                code = child.wait()
    except Stopped:
        pass
    finally:
        for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
            signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            try:
                os.killpg(child.pid, signal.SIGKILL if force else signal.SIGTERM)
                child.wait(timeout=2)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                pass
            finally:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                child.wait()
        if worker is not None:
            os.close(worker)
        os.close(parent)
    raise SystemExit(code if code >= 0 else 128 - code)


if __name__ == "__main__":
    main()
