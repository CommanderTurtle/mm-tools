"""Conservative, teacher-independent checks for AIST's triangulated camera-space labels."""

import numpy as np

BODY = np.array([5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14])
LIMBS = np.array([(5, 7), (7, 62), (6, 8), (8, 41), (9, 11), (11, 13), (10, 12), (12, 14)])
TORSO = np.array([(5, 9), (6, 10), (5, 6), (9, 10)])


def aist_geometry_valid(gt):
    """Return a mask for (..., 70, 4) labels in cm.

    These deliberately loose bounds reject gross failures, not certify annotation accuracy:
    visible body in front of the camera; limb segments 5–100 cm; torso edges 3–125 cm;
    body diameter at most 3 m. No model predictions or evaluation errors enter this check.
    """
    gt = np.asarray(gt)
    if gt.shape[-2:] != (70, 4):
        raise ValueError("Expected AIST MHR70 coordinates and validity flags")
    xyz = gt[..., :3] / 100.
    body = xyz[..., BODY, :]
    limb_lengths = np.linalg.norm(xyz[..., LIMBS[:, 0], :] - xyz[..., LIMBS[:, 1], :], axis=-1)
    torso_lengths = np.linalg.norm(xyz[..., TORSO[:, 0], :] - xyz[..., TORSO[:, 1], :], axis=-1)
    diameter = np.linalg.norm(body[..., :, None, :] - body[..., None, :, :], axis=-1).max(axis=(-2, -1))
    return (
        np.isfinite(gt).all(axis=(-2, -1))
        & (gt[..., BODY, 3] > 0).all(axis=-1)
        & (body[..., 2] > 0).all(axis=-1)
        & ((limb_lengths >= .05) & (limb_lengths <= 1.)).all(axis=-1)
        & ((torso_lengths >= .03) & (torso_lengths <= 1.25)).all(axis=-1)
        & (diameter <= 3.)
    )
