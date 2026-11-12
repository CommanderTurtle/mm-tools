import os

from state import ROOT, MODELS, require_models, source_paths


def main():
    require_models()
    entries = {SOURCES / "hi3d": "sparc3d/__init__.py",
               SOURCES / "hunyuan": "hy3dgen/__init__.py"}
    for path, revision in source_paths():
        if not (path / entries[path]).is_file():
            raise SystemExit("Missing committed native source: " + str(path / entries[path]))
        if revision and not revision.startswith(tuple("0123456789abcdef")):
            raise SystemExit("Malformed native revision for " + str(path))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Preserving existing models path; expected link to {MODELS}")
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
