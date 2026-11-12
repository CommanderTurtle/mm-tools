import os

from state import ROOT, MODELS, NATIVE, PIN, require_models


def main():
    require_models()
    processor = NATIVE / "utils" / "process_unlimitedOCR2.py"
    if not processor.is_file():
        raise SystemExit("Missing committed native postprocessor: " + str(processor))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Preserving models path; expected link to {MODELS}")
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
