"""FAH: Fast and Accurate Hand estimation (3D hand joints, MANO mesh and hand parts from images and videos).

    import cv2, fah
    model = fah.load("fah", mode="image")  # weights: $FAH_WEIGHTS or ~/.cache/fah/weights
    hands = model(cv2.imread("hand.jpg"))
"""
from .build import load_hand_model
from .estimator import HandEstimator, default_cam_int, fov_to_cam_int
from .api import FAH, MODELS, MODES, camera_head, download_sam3d, load, method_args, mhr_fitter

__all__ = ["load_hand_model", "HandEstimator", "default_cam_int", "fov_to_cam_int",
           "FAH", "MODELS", "MODES", "camera_head", "download_sam3d", "load", "method_args", "mhr_fitter"]
