import os
import sys
import time
import traceback

from lifecycle import own, release
from state import MODELS, atomic, job_path, read, require_models


def main(job):
    require_models()
    from PIL import Image
    from idol import IdolPipeline

    config = read(job / "run.json")["settings"]
    started = time.monotonic()
    atomic(job / "progress.json", {"phase": "Loading model"})
    pipeline = IdolPipeline.from_pretrained(
        str(MODELS / "weights"),
        device=os.environ.get("IDOL_DEVICE") or None,
        image_encoders={
            "dinov3": str(MODELS / "dinov3"),
            "flux2_vae": str(MODELS / "flux2"),
        },
    )
    atomic(job / "progress.json", {"phase": "Generating splats"})
    with Image.open(job / "input.png") as image:
        result = pipeline(
            image,
            mode=config["mode"],
            seed=config["seed"],
            steps=config["steps"],
            cfg_strength=config["cfg"],
            rescale_t=config["rescale_t"],
            num_gaussians=config["num_gaussians"],
            verbose=True,
        )
    atomic(job / "progress.json", {"phase": "Saving PLY"})
    result.gaussian.save_ply(str(job / "object.partial.ply"))
    (job / "object.partial.ply").replace(job / "object.ply")
    result.preprocessed_image.save(job / "prepared.png")
    atomic(job / "camera.json", result.camera)
    atomic(
        job / "result.json",
        {
            "seconds": time.monotonic() - started,
            "count": int(result.gaussian.xyz.shape[0]),
        },
    )


if __name__ == "__main__":
    own(int(sys.argv[2]))
    job = job_path(sys.argv[1])
    try:
        main(job)
    except Exception as error:
        atomic(job / "error.json", {"error": str(error)})
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        release()
