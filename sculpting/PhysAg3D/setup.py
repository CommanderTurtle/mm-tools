"""Source and model paths are anchored to this checkout, never the caller's cwd."""

import json
import os
from pathlib import Path
import sys

from state import ROOT, SOURCE, TRELLIS, PIN, atomic


def main():
    models = TRELLIS / "pretrained"
    required = [
        "TRELLIS.2-4B/pipeline.json",
        "TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
        "deps/camenduru--dinov3-vitl16-pretrain-lvd1689m/model.safetensors",
        "deps/ZhengPeng7--BiRefNet/model.safetensors",
    ]
    for name in required:
        if not (models / name).is_file():
            raise SystemExit(
                "Choose physag3d in ../../models/download_models.py first; missing "
                + str(models / name)
            )
    config = json.loads((models / required[0]).read_text())
    for key, value in config["args"]["models"].items():
        prefix = (
            "ckpts/ss_dec_conv3d_16l8_fp16"
            if key == "sparse_structure_decoder"
            else value
        )
        for suffix in (".json", ".safetensors"):
            if not (models / "TRELLIS.2-4B" / (prefix + suffix)).is_file():
                raise SystemExit("Incomplete physag3d model bundle: " + prefix + suffix)
    runner = SOURCE / "method/trellis2-native-refine/scripts/run_trellis.py"
    if not runner.is_file():
        raise SystemExit("Missing committed native runner: " + str(runner))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == models.resolve():
        pass
    elif link.exists() or link.is_symlink():
        raise SystemExit(
            "Preserving existing models path; expected link to " + str(models)
        )
    else:
        link.symlink_to(os.path.relpath(models, ROOT), target_is_directory=True)
    if len(sys.argv) == 3 and sys.argv[1] == "--native-python":
        executable = Path(sys.argv[2]).expanduser().absolute()
        if not executable.is_file():
            raise SystemExit("Native interpreter does not exist")
        atomic(
            ROOT / ".native-python.json",
            {"path": str(executable.parent.resolve() / executable.name)},
        )


if __name__ == "__main__":
    main()
