import os

from state import MODELS, ROOT, VOICES, require_models


def main():
    require_models()
    for voice in VOICES.values():
        path = MODELS / "examples" / voice["file"]
        if not path.is_file() or path.stat().st_size < 256:
            raise SystemExit(
                "Complete the audio8 bundle, including reference recordings"
            )
        with path.open("rb") as stream:
            if stream.read(45).startswith(b"version https://git-lfs.github.com/spec"):
                raise SystemExit("Complete the audio8 reference-recording download")
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Preserving existing models path; expected link to {MODELS}")
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
