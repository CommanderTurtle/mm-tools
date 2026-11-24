"""Where FAB looks for its files.

Everything lives under one root, ``$FAB_HOME`` (default ``~/.cache/fab``):

    $FAB_HOME/weights/              FAB weights (``$FAB_WEIGHTS`` overrides): fab-l/, fab-s/, fab-h/, shared/
    $FAB_HOME/sam-3d-body-dinov3/   SAM 3D Body config + MHR model (``$FAB_SAM3D`` overrides; ``fab.download_sam3d()``)
    $FAB_HOME/body_models/smpl/     SMPL model files you download yourself (``$FAB_SMPL_DIR`` overrides the smpl/ folder)
    $FAB_HOME/engines/<gpu>/        TensorRT engines built on this machine (``fab-build-trt``; ``$FAB_ENGINES`` overrides)
    $FAB_HOME/g1/                   Unitree G1 robot description (``scripts/download_g1.sh``; ``$FAB_G1`` overrides)
    $FAB_HOME/envs/                 world-mode tool environments (built by ``fab-world`` on first use; ``$FAB_ENVS`` overrides)
"""
import os
import re
from pathlib import Path

MODELS = {"fab-l": "fab-l", "l": "fab-l", "fab-s": "fab-s", "s": "fab-s", "fab-h": "fab-h", "h": "fab-h"}


def home() -> Path:
    return Path(os.environ.get("FAB_HOME", Path.home() / ".cache" / "fab")).expanduser()


def weights_dir() -> Path:
    return Path(os.environ.get("FAB_WEIGHTS", home() / "weights")).expanduser()


def sam3d_dir() -> Path:
    return Path(os.environ.get("FAB_SAM3D", home() / "sam-3d-body-dinov3")).expanduser()


def smpl_dir() -> Path:
    """The folder holding SMPL_NEUTRAL.pkl and J_regressor_h36m.npy."""
    return Path(os.environ.get("FAB_SMPL_DIR", home() / "body_models" / "smpl")).expanduser()


def body_models_dir() -> Path:
    """The parent of smpl/ (the layout the body-model loaders expect)."""
    return smpl_dir().parent


def g1_dir() -> Path:
    return Path(os.environ.get("FAB_G1", home() / "g1")).expanduser()


def envs_dir() -> Path:
    return Path(os.environ.get("FAB_ENVS", home() / "envs")).expanduser()


def gpu_slug(device=None) -> str:
    """A file-name-safe name of the CUDA device (engines are specific to the GPU they were built on)."""
    import torch

    name = torch.cuda.get_device_name(torch.device(device) if device is not None else None)
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def engines_dir(device=None) -> Path:
    return Path(os.environ.get("FAB_ENGINES", home() / "engines")).expanduser() / gpu_slug(device)


def model_file(name: str) -> Path:
    """A released model name (fab-l / fab-s / fab-h, or l / s / h) -> its weights file; any other value is a path."""
    key = MODELS.get(str(name).lower())
    if key is None:
        return Path(name).expanduser()
    return weights_dir() / key / "model.pt"


def shared(name: str) -> Path:
    """A shared weights file: detector.pt, smpl_fitter.pt, smpl_fitter_mesh.pt, contact_video.pt."""
    return weights_dir() / "shared" / name


def require(path, what, hint=""):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} not found at {path}." + (f" {hint}" if hint else "") + " See README.md, Install.")
    return path
