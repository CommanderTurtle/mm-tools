"""Console entry points that run the repository's pipeline scripts (install from a clone: pip install -e .)."""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(cmd):
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p])}
    sys.exit(subprocess.call(cmd, env=env, cwd=os.getcwd()))


def world():
    """fab-world <video | frame dir> [--out DIR] ...: world body, cameras, contacts and scene (scripts/fab_world.sh)."""
    _run(["bash", str(ROOT / "scripts" / "fab_world.sh")] + sys.argv[1:])


def retarget():
    """fab-retarget <fab-world output .npz> [--out DIR]: the Unitree G1 reference motion (scripts/fab_retarget.py)."""
    _run([sys.executable, str(ROOT / "scripts" / "fab_retarget.py")] + sys.argv[1:])


def retarget_ee():
    """fab-retarget-ee <fab-world output .npz> [--out DIR]: the G1 reference with the end-effector positions first
    (scripts/fab_retarget_ee.py; docs/retarget_ee.md)."""
    _run([sys.executable, str(ROOT / "scripts" / "fab_retarget_ee.py")] + sys.argv[1:])


def build_trt():
    """fab-build-trt [--model fab-l] ...: TensorRT engines for this GPU (scripts/build_engines.sh)."""
    _run(["bash", str(ROOT / "scripts" / "build_engines.sh")] + sys.argv[1:])
