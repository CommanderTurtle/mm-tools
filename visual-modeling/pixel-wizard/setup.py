import os

from state import ROOT, MODELS, NATIVE, PIN, require_models


def main():
    require_models()
    entry = NATIVE / "wevisone_vl" / "engine.py"
    if not entry.is_file():
        raise SystemExit("Missing committed native package: " + str(entry))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Preserving models path; expected link to {MODELS}")
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
