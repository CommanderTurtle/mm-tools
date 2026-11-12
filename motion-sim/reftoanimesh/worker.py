"""One native job per process; exiting releases its model and simulator memory."""

import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

from state import ROOT, SOURCE, atomic, read


def guard(parent):
    def watch():
        while True:
            if os.getppid() != parent:
                os.killpg(os.getpgrp(), signal.SIGKILL)
            time.sleep(0.5)

    threading.Thread(target=watch, daemon=True).start()


def main():
    path = Path(sys.argv[1]).resolve()
    phase = sys.argv[2]
    guard(int(sys.argv[3]))
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    meta = read(path / "job.json")
    config = meta["settings"]
    os.environ["FLOWHMR_ATTENTION_BACKEND"] = config["attention"]
    os.environ["FLOWHMR_ATTENTION_QUERY_CHUNK_SIZE"] = str(config["query_chunk"])
    os.environ["FLOWHMR_SAM_COMPILE"] = "1" if config["sam_compile"] else "0"
    # Released checkpoints rely on the native pre-2.6 torch.load default.
    # Explicit weights_only=True feature-cache loads remain unchanged.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

    def cancel():
        if (path / "cancel").exists():
            raise InterruptedError("Cancelled")

    def progress(stage, percent, message):
        cancel()
        atomic(
            path / "progress.json",
            dict(
                stage=getattr(stage, "value", stage),
                percent=int(percent),
                message=message,
            ),
        )

    if phase == "physics":
        from physics import reconstruct_physics

        reconstruct_physics(path, config, progress, cancel)
        return

    import torch
    from native import export_viewer, select_person
    from flowhmr.pipeline.v2m_inference import V2MInferenceEngine

    progress("loading", 0, "Loading reconstruction models")
    native_hub = torch.hub.load

    def local_hub(repository, model, *args, **kwargs):
        if repository.split(":")[0] == "facebookresearch/dinov3":
            if kwargs.get("pretrained", True):
                raise RuntimeError("SAM must construct DINOv3 with pretrained=False")
            kwargs["source"] = "local"
            return native_hub(str(ROOT / "dinov3"), model, *args, **kwargs)
        raise RuntimeError("Unregistered torch.hub source: " + str(repository))

    torch.hub.load = local_hub
    try:
        engine = V2MInferenceEngine.build(
            str(SOURCE / "checkpoints/flowhmr_latest/flowhmr_latest.ckpt"),
            str(SOURCE / "checkpoints/flowhmr_latest/config.yml"),
            device="cuda:" + str(config["device"]),
            validation_steps=config["steps"],
            postprocess=config["postprocess"],
        )
    finally:
        torch.hub.load = native_hub
    selector = None
    if config["manual_person"]:

        def selector(result, candidates, default):
            return select_person(path, candidates, default, cancel, progress)

    destination = engine.process_video(
        str(path / meta["input"]),
        str(path / "native"),
        seed=config["seed"],
        cfg_scale=config["cfg"],
        max_frames=config["max_frames"] or None,
        vggt_frame_interval=config["camera_stride"],
        static_camera=config["static_camera"],
        sam_chunk_size=config["sam_chunk"],
        person_selector=selector,
        progress_callback=progress,
        cancel_check=cancel,
    )
    suffix = "" if config["postprocess"] else "_origin"
    motion = Path(destination) / f"input_seed{config['seed']}{suffix}.npz"
    export_viewer(path, motion, engine, cancel, progress)
    progress("ready", 100, "Reconstruction saved")


if __name__ == "__main__":
    try:
        main()
    except BaseException as exc:
        if len(sys.argv) > 1:
            atomic(
                Path(sys.argv[1]) / "error.json",
                {"error": str(exc) or type(exc).__name__},
            )
        traceback.print_exc()
        raise SystemExit(1)
