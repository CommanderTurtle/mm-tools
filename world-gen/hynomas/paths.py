import json
import os
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
MODELS = REPO / "models/hynomas"
NATIVE = ROOT / "Hy-Nomas"
WORK = ROOT / ".work"
OUTPUT = WORK / "outputs"
SNAPSHOTS = WORK / "snapshots"
DB = WORK / "jobs.json"
WORKER_ID = "mm-tools-hynomas"
PIN = {
    'models_revision': '621df1570098aa1ebf916dac3ac72ae3cbe25caa',
    'text_encoder_revision': 'cc594898137f460bfe9f0759e9844b3ce807cfb5',
}

def tree_stamp(tree: Path) -> str:
    """Identity of the source this lane runs, derived from the files on disk."""
    generated = {"__pycache__", ".venv", "node_modules", "workspace", ".work",
                 "runs", ".git-archives"}
    scratch = {".env", ".physics-ready", ".snapshot-members", "MUJOCO_LOG.TXT"}
    digest = hashlib.sha256()
    for path in sorted(p for p in tree.rglob("*")
                       if p.is_file() and p.suffix not in (".pyc", ".lst")
                       and p.name not in scratch
                       and not generated & set(p.relative_to(tree).parts)):
        digest.update(path.relative_to(tree).as_posix().encode())
        digest.update(str(path.stat().st_size).encode())
    return digest.hexdigest()[:40]


LANE_STAMP = tree_stamp(ROOT)

DIT = "DiT/distilled_int8/diffusion_pytorch_model_ema_0_995_int8.safetensors"
VAE = "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors"
ENCODER = "VAE/DCAE-V/dc-ae-v-1.0-f32t4c32-bf16.pt"
ADAPTERS = ("base", "cinematic", "dual_arm_robotics")


def require_models():
    missing = [
        name
        for name in (DIT, VAE, ENCODER, "text-encoder/config.json", "PE/config.json")
        if not (MODELS / name).is_file()
    ]
    if missing:
        raise SystemExit("Run models/download_models.py hynomas: " + ", ".join(missing))


def dit_path(adapter):
    if adapter not in ADAPTERS:
        raise ValueError("Unknown adapter")
    path = MODELS / DIT if adapter == "base" else WORK / "adapters" / adapter / "int8"
    if not path.exists() or (
        adapter != "base" and not (path.parent / "ready.json").is_file()
    ):
        raise ValueError("Prepare this adapter before loading it")
    if adapter != "base" and json.loads(
        (path.parent / "ready.json").read_text()
    ) != model_identity(adapter):
        raise ValueError("Cached adapter belongs to a different source/model pin")
    return path


def model_identity(adapter):
    return {
        "source": LANE_STAMP,
        "models": PIN["models_revision"],
        "adapter": adapter,
        "profile": "int8-dit-bf16-pixelumm-v1",
    }


def overrides(adapter="base"):
    # Native INT8 preset, with the SM89-only FP8 decoder explicitly replaced by
    # the native BF16 decoder. Every path is derived from this relocated checkout.
    values = {
        "dit_weight": str(dit_path(adapter)),
        "model.text_encoder.model_name": str(MODELS / "text-encoder"),
        "model.vae.checkpoint": str(MODELS / VAE),
        "model.vae.encoder_checkpoint": str(MODELS / ENCODER),
        "model.vae.cutlass_fp8": False,
        "model.vae.stable_prefix_streaming": True,
        "parallelism.tp_size": 1,
        "parallelism.sp_size": 1,
        "inference.seg_ar_segment_vae_decode": True,
        "inference.save_path": str(OUTPUT),
    }
    return ["+experiment=single_gpu_low_mem_inference"] + [
        f"{key}={json.dumps(value)}" for key, value in values.items()
    ]


def offline_env():
    return dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        DO_NOT_TRACK="1",
        TOKENIZERS_PARALLELISM="false",
        NOMAS_WORKER_ID=WORKER_ID,
        PYTHONPATH=str(NATIVE),
        NOMAS_PARENT_PID=str(os.getpid()),
    )


def parent_guard():
    import ctypes
    import signal
    import sys

    if sys.platform == "linux":
        if ctypes.CDLL(None).prctl(1, signal.SIGTERM) != 0:
            raise RuntimeError("Could not bind child lifetime to studio")
        if os.getppid() != int(os.environ["NOMAS_PARENT_PID"]):
            raise SystemExit("Studio closed")
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
