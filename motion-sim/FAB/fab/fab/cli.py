"""fab: FAB per-frame outputs for images, through FABPredictor's default call with fast=True (the paper's timed
setting: compiled body path, one CUDA graph per person count, TensorRT engines when built on this GPU).
--model fab-l (default), fab-s or fab-h. An image, or each image of a directory (independently), goes through
.predict. Videos: fab-video (tracking, causal smoothing, video contacts, a render); world motion and scene: fab-world.

Per person (camera frame, metres; docs/usage.md): keypoints2d (the dense head's 70 MHR 2D) / keypoints2d_projected /
keypoints3d, joints_h36m14 + dense_surface_cam (the dense 3D points; dense_surface: the head's raw points), mesh_mhr,
smpl_pose / smpl_betas / mesh_smpl_fit (SMPL fitted to the points),
contact (the image contact head: l/r heel, l/r toe, l/r hand, l/r knee; static 8 + any 8),
parts_crop + crop_affine (body parts in the crop), g1_q / g1_root (Unitree G1 joint angles, URDF order, and root
rotation), box, cam_t. FAB-H: mesh_mhr, keypoints and the SMPL fitted to its MHR mesh (--profile mesh, the default; no
dense points, contacts, parts or G1), or with --profile points the SMPL fitted to its dense points, + those points.

    fab image.jpg [--out DIR] [--K fx,fy,cx,cy] [--model fab-l|fab-s|fab-h] [--profile mesh|points]
    fab photos/ [--out DIR]                        # every image of a directory
    fab image.jpg --boxes boxes.npz                # given boxes (box (N, 4), or (T, N, 4) for a directory), no detector

-> <out>/<name>_fab.pkl: {"source", "frames": [image names], "people": [[{output: array}] per image], "K" (--K or None),
   "K_used" ((T, 3, 3): the intrinsics each image ran with), "call", "profile", "fitter"}
"""

import argparse
import os
import pickle
from pathlib import Path

import cv2
import numpy as np

IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="an image or a directory of images")
    p.add_argument("--out", default="output/fab")
    p.add_argument("--K", default="", help="intrinsics fx,fy,cx,cy (px); default: FABPredictor's own (from the image size)")
    p.add_argument("--boxes", default="", help="npz with box (N, 4) xyxy px, or (T, N, 4) per image of a directory: these "
                   "people, no detector")
    p.add_argument("--model", default="fab-l", help="fab-l (default), fab-s, fab-h (or l / s / h), or a model.pt path")
    p.add_argument("--eager", action="store_true", help="fast=False (the eager PyTorch path; the default is the paper's fast=True)")
    p.add_argument("--profile", default="auto", choices=("auto", "mesh", "points"),
                   help="FAB-H's SMPL fit: mesh (its default: fitted to its MHR mesh) or points (fitted to its dense points)")
    p.add_argument("--weights", default="", help="weights folder (default $FAB_WEIGHTS or $FAB_HOME/weights)")
    a = p.parse_args(argv)
    if a.weights:
        os.environ["FAB_WEIGHTS"] = a.weights

    from fab.predictor import FABPredictor, default_K

    src = Path(a.input)
    if not src.is_dir() and src.suffix.lower() not in IMAGES:
        p.error(f"{src}: not an image or a directory of images (videos: fab-video; world motion: fab-world)")
    K = None
    if a.K:
        fx, fy, cx, cy = map(float, a.K.split(","))
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    names = sorted(f.name for f in src.iterdir() if f.suffix.lower() in IMAGES) if src.is_dir() else [src.name]
    root = src if src.is_dir() else src.parent
    boxes = None
    if a.boxes:
        boxes = np.load(a.boxes)["box"].astype(np.float32)
        boxes = boxes[None] if boxes.ndim == 2 else boxes
    ck = a.model
    fab = FABPredictor(ck, fast=not a.eager, profile=a.profile, **({"detector": ""} if boxes is not None else {}))
    K_used = []  # per image: --K, or FABPredictor's default for the image size

    def load(n):
        img = cv2.cvtColor(cv2.imread(str(root / n)), cv2.COLOR_BGR2RGB)
        K_used.append(K if K is not None else default_K(*img.shape[:2]))
        return img

    images = (load(n) for n in names)
    if boxes is not None:
        def given(t):  # image t's boxes; images past the file's rows: no people
            return boxes[t][np.isfinite(boxes[t]).all(1)] if t < len(boxes) else np.zeros((0, 4), np.float32)

        it = (fab.predict(img, K, given(t)) for t, img in enumerate(images))
        call = "predict(image, K, boxes)"
    else:
        it, call = (fab.predict(img, K) for img in images), "predict(image, K)"
    people = []
    for t, pp in enumerate(it):
        people.append([{k: v for k, v in q.items()} for q in pp])
        if t % 100 == 0:
            print(f"[fab] image {t}: {len(pp)} people", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{src.stem if not src.is_dir() else src.name}_fab.pkl"
    with open(f, "wb") as fh:
        pickle.dump({"source": str(src), "frames": names, "people": people, "K": K,
                     "K_used": np.stack(K_used) if K_used else np.zeros((0, 3, 3), np.float32),
                     "call": f'FABPredictor("{ck}", fast={not a.eager}, profile="{a.profile}").{call}', "profile": fab.profile,
                     "fitter": fab.smpl_fitter_path}, fh)
    print(f"[fab] {len(people)} images, {sum(map(len, people))} people -> {f}")


if __name__ == "__main__":
    main()
