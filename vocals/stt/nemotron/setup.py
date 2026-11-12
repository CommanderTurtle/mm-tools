import os

from state import MODELS, NATIVE, PIN, ROOT, require_models


def main():
    require_models()
    module = NATIVE / "nemo/collections/speechlm2/vllm/salm/ctc_serving.py"
    if not module.is_file() or not (NATIVE / "pyproject.toml").is_file():
        raise SystemExit("Missing committed NeMo Speech source: " + str(module))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(f"Preserving models path; expected link to {MODELS}")
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
