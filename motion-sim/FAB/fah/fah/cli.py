"""FAH: 3D hands from images and videos -> overlays (jpg / mp4) + predictions JSON.

    fah --img hand.jpg                                   # one image (or --img a folder of images)
    fah --vid clip.mp4 --backbone s --dense --parts      # video model, FAH-S, + dense points and part maps
    fah --vid clip.mp4 --mode world                      # + hands in world space with the camera model

JSON (one per image; one per video with per-frame entries), per hand: is_right, score, bbox, joints2d (px), joints3d
(camera frame, m); --mesh mano: verts3d (MANO mesh, 778 x 3); --mesh mhr: verts3d_mhr (MHR-Hand mesh, 2445 x 3),
verts3d_mhr_mano (its MANO-topology map), joints3d_mhr_fit / joints3d_mhr_rig, mhr_params (SAM 3D Body / MHR parameters,
docs/mhr_hand.md; left hands: the right rig fitted to the mirrored hand, mirrored true); --mesh both: both meshes;
--dense: surface2d_head / surface3d_head / surface_visible (40 bone + dense surface points), verts3d_dense; --parts:
parts = {x0, y0, size, labels: {part id: RLE}} (decode_parts); --masks: mask_det (instance mask RLE, bench/seg/rle.py);
--contact: contact (778 vertices), contact_score, contact_fingers; videos (FAH): + contact_video, contact_score_video,
contact_flag_video (the video contact model; drawn instead of the per-frame contact);
world: joints3d_world / verts3d_world (/ verts3d_mhr_world) and the frame's T_world_cam. Values are the Python API's
(fah.FAH) unrounded. --min_score only filters the overlays.
Weights: --weights (layout in fah/api.py and the README); SAM 3D Body files: --sam3d (fah.download_sam3d); MANO:
--mano_dir / $MANO_DIR.
"""
import argparse
import glob
import json
import os
import pickle
import sys
import time

import cv2
import numpy as np

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
RIGHT_COLOR, LEFT_COLOR = (0.98, 0.70, 0.45), (0.55, 0.80, 0.98)  # BGR 0-1
BACKBONES = {"s": "fah-s", "b": "fah-b", "l": "fah", "h": "fah-h"}


def parser():
    from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS, MODES

    p = argparse.ArgumentParser(prog="fah", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--img", help="an image or a folder of images")
    src.add_argument("--vid", help="a video file")
    p.add_argument("--backbone", default="l", choices=list(BACKBONES), help="s / b / l / h = FAH-S / FAH-B / FAH / FAH-H (default l)")
    p.add_argument("--mesh", default="mano", choices=["none", "mano", "mhr", "hmr", "both"],
                   help="parametric hand mesh: none / mano (default) / mhr (= hmr; MHR-Hand) / both (overlay: MANO | MHR-Hand)")
    p.add_argument("--output_dir", default="output/fah", help="overlays + predictions JSON")
    p.add_argument("--dense", action="store_true", help="save and draw the dense points")
    p.add_argument("--parts", action="store_true", help="save (RLE per part) and draw the part maps")
    p.add_argument("--masks", action="store_true", help="save (RLE) and draw the instance masks")
    p.add_argument("--contact", action="store_true", help="FAH: hand-object contact (per-vertex probabilities, hand / finger "
                   "scores in the JSON; vertices in contact drawn); videos: + the video contact model (FAH)")
    p.add_argument("--mode", default=None, choices=MODES,
                   help="--vid only: video (default; causal refiner + filters) / image (per frame) / world (+ camera model)")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    p.add_argument("--dtype", default="auto", choices=["auto", "bfloat16", "float16", "float32"],
                   help="pose network dtype (default: bfloat16 on CUDA, float32 on MPS / CPU)")
    p.add_argument("--trt", action="store_true", help="use the TensorRT engines in <weights>/trt if present (fah-build-trt)")
    p.add_argument("--coreml", action="store_true",
                   help="Apple silicon: the Core ML exports in <weights>/coreml (fah-build-coreml); MPS when missing")
    p.add_argument("--focal", type=float, default=0, help="focal length in px (default: the image diagonal)")
    p.add_argument("--fov", type=float, default=0, help="horizontal field of view in degrees (instead of --focal)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help=f"weights folder (default {DEFAULT_WEIGHTS}; $FAH_WEIGHTS)")
    p.add_argument("--sam3d", default=DEFAULT_SAM3D, help="SAM 3D Body model_config.yaml + assets/mhr_model.pt (fah.download_sam3d)")
    p.add_argument("--mano_dir", default=None, help="MANO_RIGHT.pkl / MANO_LEFT.pkl (default: $MANO_DIR)")
    p.add_argument("--no_tips", action="store_true", help="disable the mask-guided fingertip step")
    p.add_argument("--identity", action="store_true", help="videos: the opt-in identity rules (per-track handedness, a score gate "
                   "for new hands at the image border; fah.api.IDENTITY)")
    p.add_argument("--contact_smooth", type=int, default=1, choices=(1, 2), help="videos with --contact: 2 = the contact flag on "
                   "the mean of 2 consecutive frames (fewer switches, about +1 frame latency; fah.api.CONTACT_SMOOTH_H)")
    p.add_argument("--min_score", type=float, default=0.3, help="display threshold of the overlays (the JSON keeps every hand)")
    p.add_argument("--max_frames", type=int, default=0, help="--vid: stop after this many frames (0: all)")
    w = p.add_argument_group("world mode (docs/world.md)")
    w.add_argument("--cameras", default="", help="camera trajectory npz (T_world_cam [+ index]) instead of the camera model")
    w.add_argument("--dpvo_dir", default=os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "DPVO"), help="DPVO checkout")
    w.add_argument("--dpvo_python", default=sys.executable, help="python of the DPVO environment")
    w.add_argument("--depth_python", default=sys.executable, help="python of the MoGe-2 environment")
    w.add_argument("--camera_version", default="realtime", choices=["realtime", "offline"],
                   help="camera: realtime = the causal camera model (default); offline = + masked DROID-SLAM and Metric3D, not "
                        "real time (docs/world.md)")
    w.add_argument("--droid_dir", default=os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "DROID-SLAM"),
                   help="offline camera: TRAM's mask-aware DROID-SLAM checkout (thirdparty/DROID-SLAM of TRAM, built)")
    w.add_argument("--droid_weights", default="", help="offline camera: droid.pth (default: <droid_dir>/droid.pth)")
    w.add_argument("--metric3d_dir", default=os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "Metric3D"), help="offline camera: Metric3D checkout")
    w.add_argument("--metric3d_weights", default="", help="offline camera: metric_depth_vit_large_800k.pth (default: <metric3d_dir>/weight/)")
    w.add_argument("--slam_python", default=sys.executable, help="offline camera: python of the DROID-SLAM environment")
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    args.mesh = {"hmr": "mhr"}.get(args.mesh, args.mesh)
    if args.img and args.mode not in (None, "image"):
        p.error("--mode video / world need --vid")
    if args.cameras and (args.vid is None or args.mode != "world"):
        p.error("--cameras needs --vid ... --mode world")
    if args.camera_version == "offline" and args.mesh not in ("mano", "both"):
        p.error("--camera_version offline masks the hands with their fitted MANO meshes: --mesh mano or both")
    args.mode = args.mode or ("video" if args.vid else "image")
    args.model = BACKBONES[args.backbone]
    return args


def model_kwargs(args):
    """fah.FAH arguments of the command line (the API run that the JSON reproduces)."""
    from fah.api import IDENTITY, trt_dir

    return dict(weights=args.weights, sam3d=args.sam3d, device=args.device, trt_dir=trt_dir(args.weights) if args.trt else "",
                mano_dir=args.mano_dir, mesh=args.mesh, tips=not args.no_tips, **({"det_masks": True} if args.masks else {}),
                **(IDENTITY if getattr(args, "identity", False) else {}),
                **({"dtype": args.dtype} if args.dtype != "auto" else {}), **({"coreml": True} if args.coreml else {}),
                **({"contact": True, "contact_video": bool(args.vid)} if args.contact else {}),  # videos: + the video contact model
                **({"contact_smooth": 2} if getattr(args, "contact_smooth", 1) == 2 else {}),
                parts=args.parts)  # part maps only with --parts (rest identical)


def intrinsics(args, h, w):
    from fah import default_cam_int, fov_to_cam_int

    if args.focal > 0:
        return np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], np.float32)
    if args.fov > 0:
        return fov_to_cam_int(args.fov, h, w)
    return default_cam_int(h, w)


def draw_parts(img, hands, alpha=0.6):
    """Part maps (palm + 15 finger segments, train/part_masks.py colours) blended over the image."""
    from train.part_masks import COLORS

    out = img.copy()
    H, W = out.shape[:2]
    for h in hands:
        if h.get("parts_roi") is None:
            continue
        x0, y0, roi = h["parts_roi"]
        ys, xs = np.nonzero((roi > 0) & (roi < len(COLORS)))
        ys, xs, lab = ys + y0, xs + x0, roi[ys, xs]
        ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
        ys, xs, lab = ys[ok], xs[ok], lab[ok]
        out[ys, xs] = (out[ys, xs] * (1 - alpha) + COLORS[lab] * alpha).astype(np.uint8)
    return out


def draw_masks(img, hands, alpha=0.35):
    from bench.seg.rle import decode

    out = img.copy()
    for h in hands:
        if h.get("mask_det") is None:
            continue
        m = decode(h["mask_det"])
        color = np.array([255, 128, 0] if h["is_right"] else [0, 128, 255], np.float32)
        out[m] = (out[m] * (1 - alpha) + color * alpha).astype(np.uint8)
        cnt, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnt, -1, tuple(int(c) for c in color), 1, cv2.LINE_AA)
    return out


def draw_dense(img, hands):
    out = img.copy()
    for h in hands:
        if h.get("surface2d_head") is None:
            continue
        uv = np.asarray(h["surface2d_head"])
        vis = np.asarray(h.get("surface_visible", np.ones(len(uv), bool)))
        for (x, y), v in zip(uv, vis):
            cv2.circle(out, (int(round(x)), int(round(y))), 1, (0, 255, 255) if v else (128, 128, 128), -1, cv2.LINE_AA)
    return out


def overlay(img, hands, K, faces, args, faces_mhr=None):
    """--mesh both: MANO | MHR-Hand panels side by side."""
    from fah.vis import draw_hand_skeleton, render_meshes

    hands = [h for h in hands if h["score"] >= args.min_score]
    meshes_of = {"mhr": [("verts3d_mhr", faces_mhr)], "both": [("verts3d", faces), ("verts3d_mhr", faces_mhr)]}
    panels = []
    for key, f in meshes_of.get(args.mesh, [("verts3d", faces)]):
        meshes = [(np.asarray(h[key]), f if h["is_right"] or key == "verts3d" else f[:, ::-1], RIGHT_COLOR if h["is_right"] else LEFT_COLOR)
                  for h in hands if h.get(key) is not None]
        out = render_meshes(img, meshes, K, alpha=0.8) if meshes else img.copy()
        if args.masks:
            out = draw_masks(out, hands)
        if args.parts:
            out = draw_parts(out, hands)
        if args.dense:
            out = draw_dense(out, hands)
        if args.contact:
            out = draw_contact(out, hands, K, key)
        for h in hands:
            out = draw_hand_skeleton(out, np.asarray(h["joints2d"]))
        panels.append(out)
    return np.concatenate(panels, 1) if len(panels) > 1 else panels[0]


def overlay_for_model(img, hands, K, model, args):
    """Load mesh topology only when a requested mesh is actually present in this frame."""
    faces = model.faces if args.mesh in ("mano", "both") and any(h.get("verts3d") is not None for h in hands) else None
    faces_mhr = model.faces_mhr if args.mesh in ("mhr", "both") and any(h.get("verts3d_mhr") is not None for h in hands) else None
    return overlay(img, hands, K, faces, args, faces_mhr)


def draw_contact(img, hands, K, key="verts3d"):
    """Vertices in contact (probability >= CONTACT_THRESH) as dots on the drawn mesh (its MANO-topology vertices)."""
    from fah.api import CONTACT_THRESH
    from fah.vis import project

    out = img.copy()
    for h in hands:
        v = h.get("verts3d_mhr_mano" if key == "verts3d_mhr" else key)
        v = h.get("verts3d_dense") if v is None else v
        c = h.get("contact_video") if h.get("contact_video") is not None else h.get("contact")  # videos: the video contact model's
        if c is None or v is None:
            continue
        on = np.asarray(c, np.float32) >= CONTACT_THRESH
        for x, y in np.round(project(np.asarray(v, np.float64)[on], K)).astype(int):
            cv2.circle(out, (int(x), int(y)), 2, (40, 40, 230), -1, cv2.LINE_AA)
    return out


def encode_parts(parts_roi):
    from bench.seg.rle import encode

    x0, y0, roi = parts_roi
    return dict(x0=int(x0), y0=int(y0), size=list(roi.shape),
                labels={str(int(k)): encode(roi == k) for k in np.unique(roi) if k != 0})


def decode_parts(d):
    """JSON parts -> (x0, y0, part map uint8 (h, w)), as the API's parts_roi."""
    from bench.seg.rle import decode

    roi = np.zeros(d["size"], np.uint8)
    for k, rle in d["labels"].items():
        roi[decode(rle)] = int(k)
    return d["x0"], d["y0"], roi


def to_json(hands, dense=False, parts=False, masks=False, contact=False):
    keys = ["bbox", "joints2d", "joints3d", "joints3d_world", "verts3d", "verts3d_world", "verts3d_mhr", "verts3d_mhr_world",
            "verts3d_mhr_mano", "joints3d_mhr_fit", "joints3d_mhr_rig"] + \
        (["surface2d_head", "surface3d_head", "surface_visible", "verts3d_dense"] if dense else [])
    out = []
    for h in hands:
        d = dict(is_right=bool(h["is_right"]), score=float(h["score"]))
        for k in keys:
            if h.get(k) is not None:
                d[k] = np.asarray(h[k]).tolist()  # exact values of the API's arrays
        if h.get("mhr_params") is not None:
            d["mhr_params"] = {k: bool(v) if k == "mirrored" else np.asarray(v).tolist() for k, v in h["mhr_params"].items()}
        if parts and h.get("parts_roi") is not None:
            d["parts"] = encode_parts(h["parts_roi"])
        if masks and h.get("mask_det") is not None:
            d["mask_det"] = h["mask_det"]
        if contact and h.get("contact") is not None:
            d.update(contact=np.asarray(h["contact"], np.float32).tolist(), contact_score=float(h["contact_score"]),
                     contact_fingers=np.asarray(h["contact_fingers"], np.float32).tolist())
            if h.get("contact_k") is not None:  # several tolerances (heads with contact_dists)
                d.update(contact_k=np.asarray(h["contact_k"], np.float32).tolist(), contact_score_k=np.asarray(h["contact_score_k"]).tolist())
        if contact and h.get("contact_video") is not None:  # videos: the video contact model
            d.update(contact_video=np.asarray(h["contact_video"], np.float32).tolist(), contact_score_video=float(h["contact_score_video"]),
                     contact_flag_video=bool(h["contact_flag_video"]))
        out.append(d)
    return out


def meta(args):
    return dict(model=args.model, mode=args.mode, mesh=args.mesh, tips=not args.no_tips,
                **({"contact": True} if args.contact else {}), **({"camera": args.camera_version} if args.mode == "world" else {}),
                **({"identity": True} if getattr(args, "identity", False) else {}),
                **({"contact_smooth": 2} if args.contact and getattr(args, "contact_smooth", 1) == 2 else {}))


def run_images(args, model, paths):
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"skipping unreadable {p}")
            continue
        K = intrinsics(args, *img.shape[:2])
        t = time.perf_counter()
        hands = model(img, K)
        name = os.path.splitext(os.path.basename(p))[0]
        print(f"{name}: {len(hands)} hands ({1000 * (time.perf_counter() - t):.0f} ms)")
        cv2.imwrite(os.path.join(args.output_dir, f"{name}.jpg"), overlay_for_model(img, hands, K, model, args))
        with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
            json.dump(dict(image=os.path.basename(p), K=K.tolist(), **meta(args),
                           hands=to_json(hands, args.dense, args.parts, args.masks, args.contact)), f)


def frames_of(path, max_frames):
    cap = cv2.VideoCapture(path)
    i = 0
    while max_frames <= 0 or i < max_frames:
        ok, img = cap.read()
        if not ok:
            break
        yield i, img
        i += 1
    cap.release()


def write_frame(args, writer, name, fps, vis):
    if writer is None:
        writer = cv2.VideoWriter(os.path.join(args.output_dir, f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (vis.shape[1], vis.shape[0]))
    writer.write(vis)
    return writer


def run_video(args, model, path):
    from fah.world import estimate_cameras, load_cameras, place

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    model.method.fps = fps
    model.reset()
    name = os.path.splitext(os.path.basename(path))[0]
    world, K = args.mode == "world", None
    work = os.path.join(args.output_dir, f"{name}_world")  # world: frames + hands for the camera stages, their outputs
    estimate = world and not args.cameras
    t0 = time.perf_counter()
    results, writer, frames = [], None, []  # world: every frame's hands first (the camera stages read them), then overlays
    for i, img in frames_of(path, args.max_frames):
        K, hw = intrinsics(args, *img.shape[:2]) if K is None else K, img.shape[:2]
        hands = model(img, K)
        if world:
            results.append(hands)
            if estimate:
                d = os.path.join(work, "frames", name)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            continue
        writer = write_frame(args, writer, name, fps, overlay_for_model(img, hands, K, model, args))
        frames.append(dict(frame=i, hands=to_json(hands, args.dense, args.parts, args.masks, args.contact)))
    n = len(results) if world else len(frames)
    if world:
        if estimate:
            d = os.path.join(work, "frames", name)
            np.save(os.path.join(d, "K.npy"), np.asarray(K, np.float64))
            open(os.path.join(d, "done"), "w").close()
            os.makedirs(os.path.join(work, "hands"), exist_ok=True)
            with open(os.path.join(work, "hands", f"{name}.pkl"), "wb") as f:
                pickle.dump(dict(clip=name, method=model.name, video=True, frames=[
                    dict(key=f"{i:05d}", index=i, K=K, pred=h, image_size=hw) for i, h in enumerate(results)]), f)
            if args.camera_version == "offline":
                from fah.world import estimate_cameras_offline

                args.cameras = estimate_cameras_offline(
                    work, name, model.camera_head, args.dpvo_dir, args.droid_dir, args.droid_weights or os.path.join(args.droid_dir, "droid.pth"),
                    args.metric3d_dir, args.metric3d_weights or os.path.join(args.metric3d_dir, "weight", "metric_depth_vit_large_800k.pth"),
                    args.dpvo_python, args.depth_python, args.slam_python, os.environ.get("MANO_DIR"))
            else:
                args.cameras = estimate_cameras(work, name, model.camera_head, args.dpvo_dir, args.dpvo_python, args.depth_python,
                                                os.environ.get("MANO_DIR"))
        cams = load_cameras(args.cameras)
        idx = sorted(cams)
        np.savez(os.path.join(args.output_dir, f"{name}_cameras.npz"), index=np.array(idx),
                 T_world_cam=np.stack([cams[i] for i in idx]))
        for (i, img), hands in zip(frames_of(path, args.max_frames), results):
            if i in cams:
                place(hands, cams[i])
            writer = write_frame(args, writer, name, fps, overlay_for_model(img, hands, K, model, args))
            frames.append(dict(frame=i, hands=to_json(hands, args.dense, args.parts, args.masks, args.contact),
                               **({"T_world_cam": cams[i].tolist()} if i in cams else {})))
    if writer is not None:
        writer.release()
    dt = time.perf_counter() - t0
    print(f"{name}: {n} frames, {sum(len(f['hands']) for f in frames)} hands ({n / max(dt, 1e-9):.1f} fps incl. decode / drawing)")
    with open(os.path.join(args.output_dir, f"{name}.json"), "w") as f:
        json.dump(dict(video=os.path.basename(path), fps=fps, K=None if K is None else K.tolist(), **meta(args), world=world,
                       frames=frames), f)


def main(argv=None):
    from fah.api import FAH, loaded, relaunch_once

    args = parse_args(argv)
    if sys.platform == "darwin" and args.coreml and args.model in ("fah", "fah-h") and "FAH_LOADED_MARK" not in os.environ:
        sys.exit(relaunch_once("fah.cli", sys.argv[1:] if argv is None else argv))  # Core ML start-up abort: one relaunch
    src = args.img or args.vid
    assert os.path.exists(src), f"{src} not found"
    os.makedirs(args.output_dir, exist_ok=True)
    model = FAH(args.model, args.mode, **model_kwargs(args))
    loaded()
    if args.vid:
        run_video(args, model, args.vid)
    elif os.path.isdir(args.img):
        run_images(args, model, sorted(f for f in glob.glob(os.path.join(args.img, "*")) if f.lower().endswith(IMAGE_EXTS)))
    else:
        run_images(args, model, [args.img])


if __name__ == "__main__":
    main()
