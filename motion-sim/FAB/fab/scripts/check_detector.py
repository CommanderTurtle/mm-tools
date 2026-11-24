"""Detector check: FABPredictor's "fab" detector backend (fab/yolo.py, no ultralytics) against the ultralytics path.

    python scripts/check_detector.py --images path/to/images [more folders or files] [--recursive] [--max 200]

Per image, with the same checkpoint, confidence (FABPredictor's det_conf), the predictor's NMS IoU (fab.yolo.PREDICT_IOU),
max_det 300 and 640 px input:
  host  ultralytics' YOLO.predict on the BGR image (its own letterbox, NMS and scale_boxes; FABPredictor's first frame)
  ult   FABPredictor's detector_backend "ultralytics" path after its first frame (CUDA: det_gpu, the device letterbox,
        the predictor's network and _nms_one_class)
  fab   FABPredictor's detector_backend "fab" path (fab/yolo.py)
and the raw network outputs (1, 5, A) on the same letterboxed input (ultralytics' fused model vs fab.yolo's YOLONet),
plus the letterbox canvas size of every image size (and a few odd ones) against ultralytics' own pre_transform.
Prints every image whose boxes differ (all with --verbose) and a summary; exits 1 when the NMS IoU, a box count or a
canvas size differs or a box coordinate differs by more than --tol px (default 0: bit-identical). Needs ultralytics for the
reference (the [engine] extra). Run it on an otherwise idle GPU.
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fab import yolo  # noqa: E402
from fab.predictor import DEPLOYED_DETECTOR, FABPredictor  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
ODD_SIZES = [(1, 1), (17, 2000), (2000, 17), (333, 777), (639, 641), (640, 640), (1000, 1000), (1080, 1920),
             (1920, 1080), (3024, 4032)]


def image_files(args):
    files = []
    for s in args.images:
        p = Path(s)
        if p.is_dir():
            fs = sorted(f for f in (p.rglob("*") if args.recursive else p.iterdir()) if f.suffix.lower() in EXTS)
        else:
            fs = [p]
        if args.max and len(fs) > args.max:  # evenly over the sorted list
            fs = [fs[int(k)] for k in np.linspace(0, len(fs) - 1, args.max).round()]
        files += fs
    return files


def detector_only(backend, weights, device, conf):
    """FABPredictor's detector alone (no body model): its own _init_detector / detect on a bare instance."""
    p = FABPredictor.__new__(FABPredictor)
    p.device, p.det_conf, p.det_gpu, p.detector = device, conf, str(device).startswith("cuda"), None
    p.detector_backend = backend
    p._init_detector(weights)
    if p.detector_backend != backend:
        raise SystemExit(f"{weights}: the {backend} backend did not load (got {p.detector_backend})")
    return p


def max_diff(a, b):
    """None if the box counts differ, else the largest coordinate difference (px; 0 for no boxes)."""
    if a.shape != b.shape:
        return None
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0


def timed(fn, device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    t = time.perf_counter()
    out = fn()
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", nargs="+", required=True, help="image folders and / or files")
    ap.add_argument("--recursive", action="store_true", help="also the images in sub-folders")
    ap.add_argument("--max", type=int, default=0, help="at most this many images per folder (evenly spaced; 0 = all)")
    ap.add_argument("--scales", default="1", help="comma-separated resize factors per image (e.g. 1,0.61,1.37)")
    ap.add_argument("--weights", default=DEPLOYED_DETECTOR, help="detector checkpoint (.pt)")
    ap.add_argument("--conf", type=float, default=0.25, help="confidence threshold (FABPredictor's det_conf)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tol", type=float, default=0.0, help="largest allowed box coordinate difference (px)")
    ap.add_argument("--verbose", action="store_true", help="print every image")
    a = ap.parse_args()

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    if not Path(a.weights).exists():
        raise SystemExit(f"no detector checkpoint at {a.weights}")
    files = image_files(a)
    if not files:
        raise SystemExit("no images")
    scales = [float(s) for s in a.scales.split(",") if s.strip()]
    print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, ultralytics {ultralytics.__version__}; "
          f"device {a.device}; cudnn.benchmark {torch.backends.cudnn.benchmark}, cudnn.allow_tf32 "
          f"{torch.backends.cudnn.allow_tf32}, matmul.allow_tf32 {torch.backends.cuda.matmul.allow_tf32}", flush=True)
    print(f"{len(files)} images x {len(scales)} scales, conf {a.conf}, tol {a.tol} px, weights {a.weights}", flush=True)

    host = YOLO(a.weights)
    ult = detector_only("ultralytics", a.weights, a.device, a.conf)
    fab = detector_only("fab", a.weights, a.device, a.conf)
    first = cv2.cvtColor(cv2.imread(str(files[0])), cv2.COLOR_BGR2RGB)
    host.predict(np.ascontiguousarray(first[..., ::-1]), classes=[0], conf=a.conf, imgsz=640, verbose=False)
    ult.detect(first)  # the predictor exists from now on: the det_gpu path on CUDA
    fab.detect(first)
    pr = host.predictor
    bad_iou = pr.args.iou != yolo.PREDICT_IOU
    if bad_iou:
        print(f"DIFF NMS IoU: ultralytics {pr.args.iou}, fab.yolo {yolo.PREDICT_IOU}", flush=True)

    fails, worst, worst_raw, n, t_ult, t_fab, sizes = [], 0.0, 0.0, 0, 0.0, 0.0, set()
    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"skip {f}: unreadable", flush=True)
            continue
        for s in scales:
            img = bgr if s == 1 else cv2.resize(bgr, (max(round(bgr.shape[1] * s), 1), max(round(bgr.shape[0] * s), 1)),
                                                interpolation=cv2.INTER_AREA)
            rgb = np.ascontiguousarray(img[..., ::-1])
            H, W = rgb.shape[:2]
            sizes.add((H, W))
            ref = host.predict(img, classes=[0], conf=a.conf, imgsz=640, verbose=False)[0].boxes.xyxy.float().cpu().numpy()
            b_ult, dt = timed(lambda: ult.detect(rgb), a.device)
            t_ult += dt
            b_fab, dt = timed(lambda: fab.detect(rgb), a.device)
            t_fab += dt
            with torch.no_grad():  # the networks alone, on the same input
                im = yolo.letterbox(rgb, 640, fab.detector.stride, fab.detector.device)
                r_ult = pr.inference(im)
                r_ult = r_ult[0] if isinstance(r_ult, (list, tuple)) else r_ult
                raw = float((r_ult.float() - fab.detector.net(im)).abs().max())
            d_ult, d_host = max_diff(b_fab, b_ult), max_diff(b_fab, ref)
            n += 1
            worst_raw = max(worst_raw, raw)
            ok = d_ult is not None and d_host is not None and max(d_ult, d_host) <= a.tol
            if d_ult is not None and d_host is not None:
                worst = max(worst, d_ult, d_host)
            line = (f"{f.name} x{s:g} {H}x{W}: boxes host {len(ref)} ult {len(b_ult)} fab {len(b_fab)}; "
                    f"max |fab - ult| {d_ult if d_ult is not None else 'count'}, max |fab - host| "
                    f"{d_host if d_host is not None else 'count'} px; raw net max |diff| {raw:.3g}")
            if not ok:
                fails.append(line)
            if a.verbose or not ok:
                print(("" if ok else "DIFF ") + line, flush=True)

    bad_shape = []
    for H, W in sorted(sizes | set(ODD_SIZES)):
        mine = yolo.letterbox_shape(H, W, 640, fab.detector.stride)[:2]
        theirs = tuple(pr.pre_transform([np.zeros((H, W, 3), np.uint8)])[0].shape[:2])
        if mine != theirs:
            bad_shape.append(f"{H}x{W}: fab {mine} ultralytics {theirs}")
    for line in bad_shape:
        print(f"DIFF letterbox {line}", flush=True)

    print(f"\n{n - len(fails)} / {n} images identical within {a.tol} px (largest box difference {worst:.3g} px, raw "
          f"network {worst_raw:.3g}); letterbox sizes {len(sizes | set(ODD_SIZES)) - len(bad_shape)} / "
          f"{len(sizes | set(ODD_SIZES))} identical", flush=True)
    if n:
        print(f"detection incl. letterbox + NMS: ultralytics {n / t_ult:.1f} fps, fab {n / t_fab:.1f} fps", flush=True)
    sys.exit(1 if fails or bad_shape or bad_iou else 0)


if __name__ == "__main__":
    main()
