"""The studio runs the source committed beside it; the stamp records which."""

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parent.parent / "models"
SOURCE = ROOT / 'wemm-video'


def tree_stamp(tree: Path) -> str:
    """Identity of the source this lane runs, derived from the files on disk."""
    generated = {"__pycache__", ".venv", ".state", ".runtime", ".hands",
                  ".cache", "node_modules", "workspace", ".work", "runs", ".git-archives"}
    scratch = {".env", "MUJOCO_LOG.TXT"}
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*")
                       if p.is_file() and p.suffix not in (".pyc", ".lst")
                       and p.name not in scratch
                       and not generated & set(p.relative_to(tree).parts)):
        digest.update(path.relative_to(tree).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:40]


LANE_STAMP = tree_stamp(ROOT)
