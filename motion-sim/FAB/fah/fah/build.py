import os

import torch

from .config import get_config
from .hand_model import SAM3DHand

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}

IGNORED_MISSING_KEYS = [
    "backbone.encoder.mask_token",
    "head_pose_hand.hand_pose_comps_ori",
]


def find_model_config(checkpoint_path):
    for d in [
        os.path.dirname(checkpoint_path),
        os.path.dirname(os.path.dirname(checkpoint_path)),
    ]:
        path = os.path.join(d, "model_config.yaml")
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"model_config.yaml not found next to {checkpoint_path}")


def load_hand_model(
    checkpoint_path,
    mhr_path="",
    device="cuda",
    backbone_dtype=None,
    verbose=True,
    weights=True,
):
    """Build the hand-only model and load the hand-branch weights of a SAM 3D Body
    checkpoint (or of a trimmed checkpoint written by scripts/export_hand_checkpoint.py).
    weights=False: only the model config next to checkpoint_path (+ mhr_path) is read, the network is
    left at its initialisation (for checkpoints that carry all hand-branch weights, e.g. a FAH model.pt)."""
    if not mhr_path:
        mhr_path = os.path.join(os.path.dirname(checkpoint_path), "assets", "mhr_model.pt")

    cfg = get_config(find_model_config(checkpoint_path))
    cfg.defrost()
    cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH = mhr_path
    cfg.freeze()

    if isinstance(backbone_dtype, str):
        backbone_dtype = DTYPES[backbone_dtype]
    model = SAM3DHand(cfg, backbone_dtype=backbone_dtype)
    if not weights:
        model = model.to(device)
        if torch.device(device).type == "mps":
            model.head_pose_hand.mhr.to("cpu")
        model.eval()
        return model, cfg

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model_keys = set(model.state_dict().keys())
    hand_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
    dropped = sorted({k.split(".")[0] for k in state_dict if k not in model_keys})
    missing, unexpected = model.load_state_dict(hand_state_dict, strict=False)
    # The MHR TorchScript body model is loaded from mhr_path, not from the checkpoint;
    # the other two are unused at inference (also absent in the original checkpoint)
    missing = [
        k
        for k in missing
        if not k.startswith("head_pose_hand.mhr.") and k not in IGNORED_MISSING_KEYS
    ]
    if missing:
        raise RuntimeError(f"Missing hand-branch weights in checkpoint: {missing}")
    if verbose:
        n_params = sum(v.numel() for v in hand_state_dict.values())
        print(
            f"Loaded {len(hand_state_dict)} hand-branch tensors ({n_params / 1e6:.1f}M params); "
            f"dropped modules: {dropped}"
        )

    model = model.to(device)
    if torch.device(device).type == "mps":
        model.head_pose_hand.mhr.to("cpu")
    model.eval()
    return model, cfg


def hand_state_dict(model):
    return {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("head_pose_hand.mhr.")
    }
