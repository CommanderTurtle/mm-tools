import hashlib
import sys
import time
import traceback

from lifecycle import own, release
from state import MODELS, SOURCES, COARSE, atomic, job_path, read, require_models


def progress(job, phase, **values):
    atomic(job / "progress.json", {"phase": phase, **values})


def save_mesh(job, mesh, stem):
    from sparc3d.sparcubes.postprocess import validate_mesh

    mesh = validate_mesh(mesh)
    for extension in ("glb", "ply", "obj"):
        target = job / f"{stem}.{extension}"
        temporary = job / f"{stem}.partial.{extension}"
        mesh.export(str(temporary), file_type=extension)
        temporary.replace(target)
    atomic(
        job / f"{stem}.json", {"vertices": len(mesh.vertices), "faces": len(mesh.faces)}
    )


def prepare(job, config):
    from PIL import Image
    from sparc3d.utils.image import prepare_image, remove_background

    progress(job, "Preparing reference")
    with Image.open(job / "input.png") as source:
        image = source.convert("RGBA")
    if config["remove_background"]:
        import onnxruntime as ort
        from rembg.sessions.u2net import U2netSession

        path = MODELS / "rembg/u2net.onnx"
        checksum = hashlib.md5()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                checksum.update(block)
        if checksum.hexdigest() != "60024c5c889badc19c04ad937298a77b":
            raise ValueError("U2Net checksum differs from the pinned release")

        class LocalSession(U2netSession):
            @classmethod
            def download_models(cls, *args, **kwargs):
                return str(path)

        session = LocalSession(
            "u2net", ort.SessionOptions(), providers=["CPUExecutionProvider"]
        )
        image = remove_background(image, session=session)
    image.save(job / "condition.png")
    prepare_image(image).save(job / "prepared.png")
    if config["route"] == "mesh":
        import trimesh

        name = read(job / "run.json")["mesh_input"]
        # Empty resolver keeps uploaded OBJ/GLB references inside the upload.
        with (job / name).open("rb") as handle:
            mesh = trimesh.load(
                handle,
                file_type=name.rsplit(".", 1)[1],
                process=False,
                resolver={},
                skip_materials=True,
            )
        save_mesh(job, mesh, "coarse")
    atomic(job / "prepare.json", {"complete": True})


def coarse(job, config):
    import torch
    from PIL import Image
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    progress(job, "Loading coarse generator")
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        str(MODELS / "coarse"), subfolder=COARSE, use_safetensors=True, device="cuda"
    )

    def report(step, timestep, outputs):
        progress(
            job,
            "Coarse sampling" if step < 29 else "Extracting coarse mesh",
            step=step + 1,
            steps=30,
        )

    with Image.open(job / "condition.png") as image:
        mesh = pipeline(
            image=image,
            num_inference_steps=30,
            octree_resolution=256,
            generator=torch.manual_seed(config["seed"]),
            output_type="trimesh",
            callback=report,
            callback_steps=1,
        )[0]
    progress(job, "Saving coarse mesh")
    save_mesh(job, mesh, "coarse")


def refine(job, config):
    import trimesh
    from PIL import Image
    from sparc3d import Sparc3DPipeline

    started = time.monotonic()

    def report(event):
        progress(
            job,
            "Refining geometry"
            if event.step < event.num_inference_steps
            else "Decoding / extracting geometry",
            step=event.step,
            steps=event.num_inference_steps,
            pass_index=event.pass_index,
            passes=event.num_passes,
            structure=event.structure_resolution,
        )

    progress(job, "Loading refinement model")
    pipeline = Sparc3DPipeline.from_pretrained(
        str(MODELS / "weights"),
        local_files_only=True,
        dino_model_id=str(MODELS / "dinov2"),
        clip_model_id=str(MODELS / "clip"),
        progress_callback=report,
    )
    if config["offload"]:
        pipeline.enable_sequential_cpu_offload("cuda")
    else:
        pipeline.to("cuda")
    mesh = trimesh.load(job / "coarse.ply", process=False)
    with Image.open(job / "condition.png") as image:
        result = pipeline(
            image,
            mesh,
            resolution=config["resolution"],
            num_inference_steps=config["steps"],
            seed=config["seed"],
        )
    progress(job, "Saving geometry")
    save_mesh(job, result.mesh, "refined")
    atomic(
        job / "result.json",
        {
            **read(job / "refined.json"),
            "resolution": result.resolution,
            "refinement_seconds": time.monotonic() - started,
        },
    )


if __name__ == "__main__":
    own(int(sys.argv[3]))
    job = job_path(sys.argv[1])
    sys.path.insert(0, str(SOURCES / "hunyuan"))
    try:
        config = read(job / "run.json")["settings"]
        require_models(config)
        {"prepare": prepare, "coarse": coarse, "refine": refine}[sys.argv[2]](
            job, config
        )
    except Exception as error:
        atomic(job / "error.json", {"error": str(error)})
        traceback.print_exc()
        raise SystemExit(1)
    finally:
        release()
