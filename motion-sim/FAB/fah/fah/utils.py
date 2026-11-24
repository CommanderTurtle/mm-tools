import numpy as np
import torch


def pick_device(device="auto"):
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def save_obj(path, vertices, faces):
    with open(path, "w") as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces + 1:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")


STACK_KEYS = [
    "bbox",
    "crop_box",
    "score",
    "is_right",
    "keypoints_3d",
    "keypoints_3d_rel",
    "keypoints_2d",
    "keypoints_3d_mano",
    "keypoints_2d_mano",
    "vertices",
    "cam_t",
    "wrist_rot",
    "hand_pose",
    "scale",
    "shape",
]


def hands_to_arrays(hands):
    """Stack per-hand dicts into arrays (N first). Faces are the right-hand topology;
    reverse the winding (faces[:, ::-1]) for left hands."""
    if not hands:
        return {k: np.zeros((0,)) for k in STACK_KEYS}
    out = {k: np.stack([np.asarray(h[k]) for h in hands]) for k in STACK_KEYS if all(k in h for h in hands)}
    out["cam_int"] = hands[0]["cam_int"]
    return out
