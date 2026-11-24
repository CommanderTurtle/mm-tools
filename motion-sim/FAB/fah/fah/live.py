"""fah-live: hand estimation on a live camera (or a video file) with the FAH real-time preset. On Apple silicon: the Core
ML pose network on the GPU + the Core ML detector on the Neural Engine (fah-build-coreml), else PyTorch (MPS bfloat16 /
CUDA / CPU). Default: FAH-S, online video mode, no mesh fit, fingertip step on. Overlay at camera resolution: 2D joints + skeleton
(handedness-coloured boxes), the fingertip step (head tip -> mask-extended tip), optional instance masks / part maps and
the fitted MANO mesh, fps.

    fah-live                                   # camera 0, window, FAH-S
    fah-live --backbone l --mesh_fit 1         # FAH (ViT-L), fitted MANO mesh as a wireframe
    fah-live --video_mode 0                    # per-frame image mode (key v switches the video refiner on)
    fah-live --camera 1 --fov 78 --record out.mp4             # or --focal <px>
    fah-live --video clip.mp4 --no_window --frames 100        # headless, prints fps

Keys: q quit, k keypoints, m instance masks / parts (Core ML: the part-free pose network runs while they are off; the
first m loads the one with the part head, a few seconds), f mesh fit on / off, v video mode on / off (the causal video refiner +
its One Euro filters, as fah --mode video; its state restarts on every switch), s snapshot (PNG into --out_dir). The fps
counter is the median over the last 30 frames. Video mode: the refiner takes the last 32 processed frames (no time
input; trained with 1-3x frame strides, so frames the newest-frame reader skips at >= 10 fps stay in its training
range); its One Euro filters run on wall-clock time; on Apple silicon it runs on the CPU.
The camera frames are read by a thread that keeps only the newest one, so the display never lags behind the camera.
macOS asks for camera access on the first run (System Settings > Privacy & Security > Camera, for the terminal app).
The Core ML detector runs in the camera's letterbox shape (the common landscape and portrait aspects are built by default,
fah.build_coreml.DEFAULT_FRAMES; other frames are padded up to the nearest built shape; fah-build-coreml --frames builds
exact ones).
"""
import argparse
import os
import sys
import threading
import time
import warnings
from collections import deque

import cv2
import numpy as np

from fah.cli import BACKBONES

NAMES = {"fah-s": "FAH-S", "fah-b": "FAH-B", "fah": "FAH", "fah-h": "FAH-H"}
SIDE = {True: (250, 178, 115), False: (140, 204, 250)}  # BGR: right light blue, left light orange (fah.vis)
PARTS = (np.array([[0, 0, 0]] + [[int(c) for c in cv2.applyColorMap(np.uint8([[i * 15]]), cv2.COLORMAP_HSV)[0, 0]]
                                 for i in range(16)], np.uint8))  # 0 background, 1 palm, 2-16 finger segments


def preset(args):
    """fah method arguments of the real-time preset -> (args, Core ML used, video refiner checkpoint for key v)."""
    from fah.api import coreml_paths, method_args, pick_device

    model, device = BACKBONES[args.backbone], pick_device(args.device)
    use_coreml = args.preset == "coreml" or (args.preset == "auto" and sys.platform == "darwin"
                                             and all(os.path.exists(p) for p in coreml_paths(model, args.weights, parts=False)))
    a = method_args(model, "video", args.weights, device, args.sam3d, mesh="none", coreml=use_coreml,
                    **({"dtype": args.dtype} if args.dtype else {}))  # the video settings act only with the refiner on
    refiner = a["refiner"] if os.path.exists(a["refiner"]) else None
    if args.video_mode and refiner is None:
        warnings.warn(f"no video refiner at {a['refiner']}: per-frame image mode")
    if not args.video_mode or refiner is None:
        a["refiner"] = None  # per-frame image mode
    return a, "pose_coreml" in a, refiner


class LatestFrame:
    """Camera reader thread: keeps only the newest frame (stale frames are dropped, not queued)."""

    def __init__(self, cap):
        self.cap, self.frame, self.n, self.ok = cap, None, 0, True
        self.cond = threading.Condition()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, f = self.cap.read()
            with self.cond:
                if not ok:
                    self.ok = False
                else:
                    self.frame, self.n = f, self.n + 1
                self.cond.notify_all()

    def get(self, last_n, timeout=2.0):
        """The newest frame after frame number last_n -> (frame, n) or (None, last_n) when the camera stopped."""
        with self.cond:
            self.cond.wait_for(lambda: self.n > last_n or not self.ok, timeout)
            return (self.frame, self.n) if self.n > last_n else (None, last_n)


def box_masks(det, idx, boxes, H, W, step=4):
    """Instance masks inside each detector box for display: detector.mask_at on every step-th pixel, upsampled (nearest)
    -> [(x0, y0, (h, w) bool) or None] (full-resolution masks() costs ~3 ms per hand on the CPU)."""
    grids, pts = [], []
    for b in boxes:
        x0, y0 = np.floor(b[:2]).astype(int).clip(0, [W, H])
        x1, y1 = np.ceil(b[2:]).astype(int).clip(0, [W, H])
        gx, gy = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
        grids.append((x0, y0, x1, y1, gx.shape))
        pts.append((gx.ravel(), gy.ravel()))
    out = []
    for (x0, y0, x1, y1, sh), v in zip(grids, det.mask_at(idx, pts)):
        out.append((x0, y0, cv2.resize(v.reshape(sh).astype(np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST) > 0)
                   if v.size else None)
    return out


def draw(img, m, hands, show_kp, show_mask, K, mesh_style="wire"):
    out = img.copy()
    if show_mask and hands and getattr(m.detector, "last", None) is not None:
        db = m.detector.last["boxes"]
        idx = [int(np.abs(db - h["bbox"]).sum(1).argmin()) if len(db) else -1 for h in hands]
        ok = [i >= 0 and np.abs(db[i] - h["bbox"]).sum() < 1 for i, h in zip(idx, hands)]
        keep = [k for k, o in enumerate(ok) if o and hands[k].get("parts_roi") is None]
        masks = dict(zip(keep, box_masks(m.detector, [idx[k] for k in keep], db[[idx[k] for k in keep]], *img.shape[:2]))) if keep else {}
        for k, h in enumerate(hands):
            if h.get("parts_roi") is not None:  # part map (inside the detector mask)
                x0, y0, lab = h["parts_roi"]
                roi = out[y0:y0 + lab.shape[0], x0:x0 + lab.shape[1]]
                sel = lab[:roi.shape[0], :roi.shape[1]] > 0
                roi[sel] = (0.45 * roi[sel] + 0.55 * PARTS[lab[:roi.shape[0], :roi.shape[1]][sel]]).astype(np.uint8)
            elif masks.get(k) is not None:
                x0, y0, mk = masks[k]
                roi = out[y0:y0 + mk.shape[0], x0:x0 + mk.shape[1]]
                roi[mk] = (0.5 * roi[mk] + 0.5 * np.array(SIDE[bool(h["is_right"])])).astype(np.uint8)
    meshes = [(np.asarray(h["verts3d"], np.float64), m._mano_faces(), SIDE[bool(h["is_right"])])
              for h in hands if h.get("mesh_param") and h.get("verts3d") is not None]  # fitted MANO mesh (mesh fit on)
    if meshes and mesh_style == "shaded":
        from fah.vis import render_meshes

        out = render_meshes(out, [(v, f, np.array(c) / 255.0) for v, f, c in meshes], K, alpha=0.75, supersample=1)
    elif meshes:
        from fah.vis import project

        for v, f, c in meshes:
            if (v[:, 2] > 1e-3).all():
                cv2.polylines(out, list(np.round(project(v, K)[f]).astype(np.int32)), True, c, 1, cv2.LINE_AA)
    for h in hands:
        x1, y1, x2, y2 = np.asarray(h["bbox"]).astype(int)
        cv2.rectangle(out, (x1, y1), (x2, y2), SIDE[bool(h["is_right"])], 2)
        cv2.putText(out, f"{'R' if h['is_right'] else 'L'} {h['score']:.2f}", (x1, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    SIDE[bool(h["is_right"])], 1, cv2.LINE_AA)
        if show_kp:
            from fah.vis import draw_hand_skeleton

            out = draw_hand_skeleton(out, np.asarray(h["joints2d"]))
            ext = h.get("tip_ext")
            if ext is not None and h.get("joints2d_head") is not None:  # fingertip step: head tip -> mask-extended tip
                for f, t in enumerate((4, 8, 12, 16, 20)):
                    if ext[f] > 0:
                        a, b = np.round(h["joints2d_head"][t]).astype(int), np.round(h["joints2d"][t]).astype(int)
                        cv2.line(out, tuple(a), tuple(b), (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(out, tuple(b), 5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def open_source(args):
    if args.video:
        return cv2.VideoCapture(os.path.expanduser(args.video))
    cap = cv2.VideoCapture(args.camera, cv2.CAP_AVFOUNDATION) if sys.platform == "darwin" else cv2.VideoCapture(args.camera)
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    return cap


def run(args):
    from bench.methods.fah import FAHMethod
    from fah.api import coreml_paths, loaded

    if args.mano_dir:
        os.environ["MANO_DIR"] = args.mano_dir
    cap = open_source(args)
    if not cap.isOpened():
        print(f"Could not open {'video ' + args.video if args.video else f'camera {args.camera}'}." + ("" if args.video else
              " macOS asks once per terminal app for camera access: accept the prompt (or enable the app in System Settings >"
              " Privacy & Security > Camera) and run again; or try another --camera index, or --video <file>."))
        sys.exit(1)
    a, coreml, refiner = preset(args)  # the model after the source: a missing camera fails fast
    t = time.perf_counter()
    m = FAHMethod(**a)
    m.mesh_fit = args.mesh_fit
    model = NAMES[BACKBONES[args.backbone]] + (" Core ML" if coreml else f" {a['device'].upper()}")
    print(f"{model} loaded in {time.perf_counter() - t:.1f} s")
    loaded()
    src = None if args.video else LatestFrame(cap)
    m.reset()
    show_kp, show_mask, n_done, last_n, K, lat, snap, writer = True, False, 0, 0, None, [], 0, None
    m.est.parts = show_mask  # part maps only while shown (on the CPU: ~15 ms per frame with two hands)
    poses = {False: m.est.student_coreml}  # Core ML: the part-free pose network while the part maps are off
    full = coreml_paths(BACKBONES[args.backbone], args.weights)[0] if coreml else None
    if coreml and a["pose_coreml"] == full:  # no part-free export: the full network either way
        poses[True] = m.est.student_coreml
    periods = deque(maxlen=30)
    t0 = prev = time.perf_counter()
    while args.frames <= 0 or n_done < args.frames:
        if src is None:
            ok, img = cap.read()
            if not ok:
                break
        else:
            img, last_n = src.get(last_n)
            if img is None:
                print("camera stopped delivering frames")
                break
        if K is None:
            from fah import fov_to_cam_int

            h, w = img.shape[:2]
            K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], np.float32) if args.focal else \
                fov_to_cam_int(args.fov, h, w)
            print(f"frames {w}x{h}, focal {K[0, 0]:.0f} px")
        tic = time.perf_counter()
        if m.tracker is not None:
            m.t = (tic - t0) * m.fps  # One Euro filter on wall-clock time (the method counts t / fps)
        hands = m(img, K)
        toc = time.perf_counter()
        lat.append(toc - tic)
        n_done += 1
        periods.append(max(toc - prev, 1e-6))  # loop period (capture wait + inference + drawing)
        prev = toc
        fps = 1 / np.median(periods)
        if args.no_window and n_done % 30 == 0:
            print(f"{n_done} frames: {fps:.1f} fps (loop), model {1 / np.median(lat[-30:]):.1f} fps, {len(hands)} hands")
        if args.no_window and not args.record:
            continue
        vis = draw(img, m, hands, show_kp, show_mask, K, args.mesh_style)
        status = f"{fps:.1f} fps | {model} | mesh {'on' if m.mesh_fit else 'off'} | video {'on' if m.tracker is not None else 'off'}"
        (tw, th), _ = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(vis, (4, 4), (16 + tw, 16 + th), (0, 0, 0), -1)
        cv2.putText(vis, status, (10, 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if args.record:
            if writer is None:
                rate = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(os.path.expanduser(args.record), cv2.VideoWriter_fourcc(*"mp4v"), rate if rate > 1 else 30.0,
                                         (vis.shape[1], vis.shape[0]))
            writer.write(vis)
        if args.no_window:
            continue
        cv2.imshow("FAH live", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("k"):
            show_kp = not show_kp
        elif key == ord("m"):
            show_mask = m.est.parts = not show_mask
            if coreml:
                if show_mask and True not in poses and not os.path.exists(full):
                    print(f"part maps need {full} (fah-build-coreml)")
                    show_mask = m.est.parts = False
                elif show_mask and True not in poses:  # first use: the pose network with the part head, loaded once
                    from fah.coreml import CoreMLPose

                    cv2.putText(vis, "loading the part-map network ...", (10, 30 + 2 * th), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow("FAH live", vis)
                    cv2.waitKey(1)
                    t = time.perf_counter()
                    poses[True] = CoreMLPose(full, m.est.student_coreml.units.name)
                    print(f"part-map network loaded in {time.perf_counter() - t:.1f} s")
                m.est.student_coreml = poses[show_mask]
        elif key == ord("f"):
            m.mesh_fit = 0 if m.mesh_fit else 1
        elif key == ord("v"):
            if refiner is None:
                print("video mode needs the video refiner (<weights>/shared/video_refiner.pt)")
            else:
                m.refiner = None if m.refiner else refiner
                m.tracker = None
                m.reset()
        elif key == ord("s"):
            os.makedirs(os.path.expanduser(args.out_dir), exist_ok=True)
            p = os.path.join(os.path.expanduser(args.out_dir), f"fah_live_{time.strftime('%Y%m%d_%H%M%S')}_{snap}.png")
            cv2.imwrite(p, vis)
            snap += 1
            print("saved", p)
    if src is not None:
        src.ok = False
    if writer is not None:
        writer.release()
        print("recorded", args.record)
    cap.release()
    cv2.destroyAllWindows()
    if lat:
        n = len(lat)
        w = lat[min(20, n // 2):]  # past the first frames (Core ML / compile warm-up)
        print(f"{n} frames: model {1 / np.median(w):.1f} fps (median), loop {1 / np.median(periods):.1f} fps (last 30)" +
              ("" if src is None else f", camera delivered {last_n} frames in {time.perf_counter() - t0:.1f} s"))


def parser():
    from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS

    p = argparse.ArgumentParser(prog="fah-live", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--video", default="", help="video file instead of the camera (every frame processed, none dropped)")
    p.add_argument("--width", type=int, default=1280, help="requested camera size (0: the camera's default)")
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fov", type=float, default=70.0, help="horizontal field of view (deg) -> focal")
    p.add_argument("--focal", type=float, default=0.0, help="focal in pixels (overrides --fov)")
    p.add_argument("--backbone", default="s", choices=list(BACKBONES), help="s / b / l / h = FAH-S (default, real time) / FAH-B / FAH / FAH-H")
    p.add_argument("--preset", default="auto", choices=["auto", "coreml", "torch"],
                   help="auto: the Core ML exports when present (macOS), else PyTorch")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="", choices=["", "bfloat16", "float16", "float32"], help="PyTorch precision (default: the device preset)")
    p.add_argument("--mesh_fit", type=int, default=0, help="1: MANO mesh fit (+4 ms on the CPU), drawn with --mesh_style; key f")
    p.add_argument("--mesh_style", default="wire", choices=["wire", "shaded"])
    p.add_argument("--record", default="", help="write the overlay video (mp4)")
    p.add_argument("--video_mode", type=int, default=1, choices=[0, 1], help="online video refiner (key v); 0: per-frame image mode")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--sam3d", default=DEFAULT_SAM3D)
    p.add_argument("--mano_dir", default=None)
    p.add_argument("--no_window", action="store_true")
    p.add_argument("--frames", type=int, default=0, help="stop after N frames (0: until q / end of video)")
    p.add_argument("--out_dir", default=".", help="snapshots")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if sys.platform == "darwin" and BACKBONES[args.backbone] in ("fah", "fah-h") and args.preset != "torch" and \
            "FAH_LOADED_MARK" not in os.environ:  # Core ML start-up abort of the larger models: one relaunch (fah.api.relaunch_once)
        from fah.api import relaunch_once

        sys.exit(relaunch_once("fah.live", sys.argv[1:] if argv is None else argv))
    run(args)


if __name__ == "__main__":
    main()
