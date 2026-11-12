import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[2] / "models/underdog"


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "gpu"
    if mode not in {"gpu", "cpu"}:
        raise SystemExit("Usage: setupwithuv.sh [gpu|cpu]")
    if not MODELS.is_dir():
        raise SystemExit(
            "Select at least one underdog bundle in models/download_models.py first"
        )
    link = ROOT / "models"
    if not (link.is_symlink() and link.resolve() == MODELS.resolve()):
        if link.exists() or link.is_symlink():
            raise SystemExit(
                f"Preserving existing models path; expected a symlink to {MODELS}"
            )
        link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)
    from core import VARIANTS, missing_files

    available = [key for key in VARIANTS if not missing_files(key)]
    if not available:
        raise SystemExit(
            "Complete at least one underdog downloader bundle, including its processor"
        )
    env = ROOT / ".env"
    if not env.exists():
        default = "pro" if "pro" in available else available[0]
        template = (
            (ROOT / ".env.example")
            .read_text()
            .replace("BARK_MODEL=pro", f"BARK_MODEL={default}")
        )
        if mode == "cpu":
            template = template.replace("BARK_DEVICE=cuda", "BARK_DEVICE=cpu").replace(
                "BARK_PRO_DTYPE=bfloat16", "BARK_PRO_DTYPE=float32"
            )
        env.write_text(template)
    print(f"Available bundles: {', '.join(available)}")


if __name__ == "__main__":
    main()
