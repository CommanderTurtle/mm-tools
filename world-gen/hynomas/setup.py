import os

from paths import MODELS, NATIVE, PIN, ROOT, require_models


def main():
    require_models()
    entry = NATIVE / "apps" / "seg_ar_web_server.py"
    if not entry.is_file():
        raise SystemExit("Missing committed native app: " + str(entry))
    for path in (ROOT / "models", NATIVE / "models"):
        if path.is_symlink() and path.resolve() == MODELS.resolve():
            continue
        if path.exists() or path.is_symlink():
            raise SystemExit(f"Preserving existing models path: {path}")
        path.symlink_to(os.path.relpath(MODELS, path.parent), target_is_directory=True)


if __name__ == "__main__":
    main()
