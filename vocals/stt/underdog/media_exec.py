"""Tie the decoder to its Linux/WSL server before replacing this process."""

import ctypes
import os
import signal
import sys

parent = int(sys.argv[1])
libc = ctypes.CDLL(None, use_errno=True)
if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
    raise OSError(ctypes.get_errno(), "Could not bind decoder lifetime to server")
if os.getppid() != parent:
    raise SystemExit("Studio closed")
os.execvp(sys.argv[2], sys.argv[2:])
