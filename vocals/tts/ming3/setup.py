import os
import subprocess

from state import MODELS, NATIVE, PIN, PRESETS, ROOT, require_models


def main():
    selected = [
        key for key in PIN["models"] if (MODELS / key / "config.json").is_file()
    ]
    if not selected:
        raise SystemExit(
            "Select ming3-0.5b and/or ming3-tiny in models/download_models.py first"
        )
    for key in selected:
        require_models(key)
    for name in ["voices.json", *(p["file"] for p in PRESETS.values())]:
        file = MODELS / "references" / name
        if not file.is_file() or file.stat().st_size < 20:
            raise SystemExit("Complete the Ming3 reference assets in the model picker")
    entries = ("ming_tts.py", "ming_tiny/__init__.py")
    for name in entries:
        if not (NATIVE / name).is_file():
            raise SystemExit("Missing committed native module: " + str(NATIVE / name))
    link = ROOT / "models"
    if not (link.is_symlink() and link.resolve() == MODELS.resolve()):
        if link.exists() or link.is_symlink():
            raise SystemExit(
                f"Preserving existing models path; expected link to {MODELS}"
            )
        link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)
    for key in selected:
        venv = ROOT / f".venv-{key}"
        subprocess.run(
            ["uv", "venv", "--python", "3.11", "--seed", "--allow-existing", str(venv)],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv / "bin/python"),
                "--index-strategy",
                "unsafe-best-match",
                "-r",
                str(ROOT / f"requirements-{key}.txt"),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
