# SPDX-License-Identifier: AGPL-3.0-only
"""Own a studio's native process group until the studio closes its control pipe."""
import os
import select
import signal
import subprocess
import sys

def main():
    child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL, start_new_session=True)
    closing = False
    def stop(*_):
        nonlocal closing
        closing = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while child.poll() is None and not closing:
            if select.select([0], [], [], 0.5)[0]:
                break
    finally:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try: os.killpg(child.pid, sig)
            except ProcessLookupError: break
            if sig == signal.SIGTERM:
                try: child.wait(timeout=12)
                except subprocess.TimeoutExpired: pass
        child.wait()
    return child.returncode

if __name__ == "__main__":
    sys.exit(main())
