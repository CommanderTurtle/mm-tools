"""fab-video: one video -> every FAB output per frame and person (a results pickle) and one render: the frame with the
2D keypoints, the dense surface points and the feet's contact state over the per-foot contact timeline (H.264, no
audio), plus a poster still.

    fab-video clip.mp4 -o out/clip.mp4                     # FAB-L: tracking, causal smoothing, video contacts, render
    fab-video clip.mp4 -o out/clip.mp4 --model fab-s
    fab-video frames/ -o out/clip.mp4 --fps 30             # a directory of frames (sorted by name)
    fab-video clip.mp4 --no-render --results out/clip.pkl  # results only
    fab-video clip.mp4 --boxes boxes.npz --eager           # given boxes per frame (no detector), eager PyTorch
    fab-video clip.mp4 -o out/clip.mp4 --load out/clip.pkl # render saved results again (no model run)

Frame by frame, in order (causal):
  1. FABPredictor(model, fast=True).predict_video: the person detector every 10th frame, boxes tracked from the
     keypoints in between, each person's previous-frame fit as the temporal input (--boxes: the given boxes, the
     previous frame's same person by order). TensorRT engines are used when built on this GPU (fab-build-trt;
     --no-trt: the compiled PyTorch body).
  2. Tracks: people linked across frames by box IoU (track id). The causal fixed-lag smoother (fab/retarget/smooth.py;
     --lag 3 / --sigma 2.5 frames = 100 ms at 30 fps) runs per track on the G1, SMPL and keypoint outputs; its values
     are stored as <output>_smoothed next to the raw per-frame ones.
  3. Video contacts: the causal contact video model (weights/shared/contact_video.pt, a GRU over a track's contact-head
     features, keypoints and their velocities) -> contact_video (16: 8 static + 8 any, fab/contact_regions.py order)
     per person, in the camera frame: right for a static camera. For a moving camera use fab-world, which runs the same
     model in the reconstructed world.

Results (pickle): video, fps, size (W, H), K, model, names (the frame files of a directory, else None), call,
frames[i] = {index, people: [per person: the FABPredictor outputs (docs/usage.md) + track, <output>_smoothed,
contact_video]}; about 0.5 MB per person and frame (the meshes). Tracks shorter than 4 frames get no contact_video (as
world mode). FAB-H has no contact head: no video contacts.
Rendering uses ffmpeg (H.264) when it is on the PATH, else OpenCV's mp4v.
"""
import argparse
import os
import pickle
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SMOOTH = {"g1_q": "vec", "g1_root": "rot", "smpl_pose": "aa", "smpl_betas": "vec", "cam_t": "vec",
          "mesh_smpl_fit": "vec", "joints_h36m14": "vec", "keypoints3d": "vec", "keypoints2d": "vec"}  # not mesh_mhr (size)
MIN_LEN = 4  # shortest track that gets video contacts (train/extract_contact_tracks.py --min_len, as world mode)
FEET = (("L heel", 0, 17), ("L toe", 1, 15), ("R heel", 2, 20), ("R toe", 3, 18))  # (label, contact channel, MHR70 kp)
BODY_LINKS = ((13, 11), (11, 9), (14, 12), (12, 10), (9, 10), (5, 9), (6, 10), (5, 6), (5, 7), (6, 8), (7, 62), (8, 41),
              (13, 15), (13, 16), (13, 17), (14, 18), (14, 19), (14, 20), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))


def parser():
    p = argparse.ArgumentParser(prog="fab-video", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video", help="a video file or a directory of frames")
    p.add_argument("-o", "--out", default="", help="render (.mp4; default output/<video name>_fab.mp4)")
    p.add_argument("--results", default="", help="results pickle (default: the render path with .pkl)")
    p.add_argument("--load", default="", help="render this results pickle instead of running the model")
    p.add_argument("--no-render", dest="render", action="store_false", help="results only")
    p.add_argument("--model", default="fab-l", help="fab-l (default), fab-s, fab-h (or l / s / h), or a model.pt path")
    p.add_argument("--weights", default="", help="weights folder (default $FAB_WEIGHTS or $FAB_HOME/weights)")
    p.add_argument("--eager", action="store_true", help="fast=False: the eager PyTorch path (no compilation, no engines)")
    p.add_argument("--profile", default="auto", choices=("auto", "mesh", "points"),
                   help="FAB-H's SMPL fit: mesh (its default: fitted to its MHR mesh) or points (fitted to its dense points)")
    p.add_argument("--no-trt", dest="trt", action="store_false", help="fast path without the TensorRT engines")
    p.add_argument("--boxes", default="", help="npz with box (T, 4) or (T, N, 4) xyxy px per frame: these people, no detector")
    p.add_argument("--focal", type=float, default=0, help="focal length in px (default: the image diagonal)")
    p.add_argument("--fov", type=float, default=0, help="horizontal field of view in degrees (instead of --focal)")
    p.add_argument("--K", default="", help="fx,fy,cx,cy (instead of --focal / --fov)")
    p.add_argument("--fps", type=float, default=0, help="frame rate (default: the video's; 30 for a frame directory)")
    p.add_argument("--max_frames", type=int, default=0, help="stop after this many frames (0: all)")
    p.add_argument("--lag", type=int, default=3, help="smoother look-ahead in frames (0: no smoothing)")
    p.add_argument("--sigma", type=float, default=2.5, help="smoother width in frames")
    p.add_argument("--no-contact-video", dest="contact_video", action="store_false", help="skip the video contact model")
    r = p.add_argument_group("render")
    r.add_argument("--width", type=int, default=960, help="render width (px)")
    r.add_argument("--crf", type=int, default=18, help="H.264 quality (lower = better)")
    r.add_argument("--poster", type=float, default=-1, help="poster still at this time (s; default the first frame)")
    r.add_argument("--points", action="store_true", help="also draw the dense surface points")
    return p


def frame_source(path, max_frames=0):
    """-> (generator of RGB frames, fps or None, names or None)."""
    src = Path(path)
    if src.is_dir():
        names = sorted(f.name for f in src.iterdir() if f.suffix.lower() in IMAGES)
        names = names[:max_frames] if max_frames > 0 else names
        return (cv2.cvtColor(cv2.imread(str(src / n)), cv2.COLOR_BGR2RGB) for n in names), None, names

    def gen():
        cap = cv2.VideoCapture(str(src))
        i = 0
        while max_frames <= 0 or i < max_frames:
            ok, f = cap.read()
            if not ok:
                break
            yield cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            i += 1
        cap.release()

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or None
    cap.release()
    return gen(), fps, None


def intrinsics(a, H, W):
    if a.K:
        fx, fy, cx, cy = (float(x) for x in a.K.split(","))
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)
    f = a.focal if a.focal > 0 else (W / 2 / np.tan(np.radians(a.fov) / 2) if a.fov > 0 else float(np.hypot(H, W)))
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float32)


def link_tracks(frames, min_iou=0.3):
    """Track ids by box IoU with the previous frame's people (as FABPredictor.predict_video_smoothed links them)."""
    def iou(p, q):
        w = max(0.0, min(p[2], q[2]) - max(p[0], q[0]))
        h = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - w * h
        return w * h / u if u > 0 else 0.0

    last, next_id = {}, 0
    for people in frames:
        cur, seen = {}, set()
        for pp in people:
            best, bo = None, min_iou
            for tid, box in last.items():
                o = iou(pp["box"], box) if tid not in seen else 0.0
                if o > bo:
                    best, bo = tid, o
            if best is None:
                best, next_id = next_id, next_id + 1
            seen.add(best)
            pp["track"] = best
            cur[best] = pp["box"]
        last = cur


def smooth_tracks(frames, lag, sigma, g1_limits=None):
    """<output>_smoothed per person: the causal fixed-lag smoother over its track (the online smoother's values);
    g1_q_smoothed clamped to the G1 joint limits (the local-linear weights can overshoot)."""
    from fab.retarget.smooth import LagSmoother

    runs = defaultdict(list)  # track -> [(frame, person)]; a track is contiguous (lost = ended)
    for t, people in enumerate(frames):
        for pp in people:
            runs[pp["track"]].append(pp)
    for run in runs.values():
        keys = [k for k in SMOOTH if k in run[0]]
        sm = LagSmoother(lag, sigma, {k: SMOOTH[k] for k in keys})
        outs = [s for pp in run for s in [sm.push({k: pp[k] for k in keys})] if s is not None] + list(sm.flush())
        for pp, s in zip(run, outs):
            for k, v in s.items():
                pp[f"{k}_smoothed"] = np.asarray(v, np.float32)
            if g1_limits is not None and "g1_q_smoothed" in pp:
                pp["g1_q_smoothed"] = np.clip(pp["g1_q_smoothed"], *g1_limits).astype(np.float32)


def video_contacts(a, frames, K, fps, model_file):
    """contact_video per person: the contact head's features on each track's crops (the eager BF16 forward of
    train/extract_contact_tracks.py, as world mode), then the causal GRU (train/train_contact_video.py) per track in the
    camera frame."""
    import torch
    from torchvision.transforms import ToTensor

    from fab import paths
    from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from fab.data.utils.prepare_batch import prepare_batch
    from train import train_contact_video as tcv
    from train.extract_contact_tracks import load_model

    vid_ck = paths.require(paths.shared("contact_video.pt"), "the video contact model",
                           "Download the weights (README.md, Install).")
    model = load_model(str(model_file), str(paths.sam3d_dir()))
    size = model.size
    transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                         VisionTransformWrapper(ToTensor())])
    per = defaultdict(lambda: defaultdict(list))  # track -> field -> per-frame rows
    src, _, _ = frame_source(a.video, len(frames))
    for t, image in enumerate(src):
        people = frames[t]
        if not people:
            continue
        boxes = np.stack([pp["box"] for pp in people]).astype(np.float32)
        b = prepare_batch(image, transform, boxes, None, None, cam_int=torch.from_numpy(K).float()[None])
        crop = b["img"][0]
        aff = b["affine_trans"][0].numpy() * (size / crop.shape[-1])
        if crop.shape[-1] != size:
            crop = torch.nn.functional.interpolate(crop, size=(size, size), mode="area")
        g = [x.cuda().float() for x in (b["bbox_center"][0], b["bbox_scale"][0], b["ori_img_size"][0],
                                        torch.from_numpy(np.repeat(K[None], len(people), 0)), torch.from_numpy(aff))]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(crop.cuda(), g[0], g[1], g[2], g[3], affine=g[4], return_vertices=False)
        cam_t = out["pred_cam_t"].float()
        rows = {"logits": out["contact_logits"].float().cpu().numpy(), "feat": out["contact_feat"].half().cpu().numpy(),
                "pts": (out["contact_pts"].float() + cam_t[:, None]).cpu().numpy()}
        for i, pp in enumerate(people):
            for k, v in rows.items():
                per[pp["track"]][k].append(v[i])
            per[pp["track"]]["time"].append(t / fps)
            per[pp["track"]]["who"].append((t, i))
    del model
    torch.cuda.empty_cache()
    st = torch.load(vid_ck, map_location="cpu", weights_only=False)
    tcv.VEL_WIN = float(st.get("args", {}).get("vel_win", 0.0))
    gru = tcv.ContactGRU(st["d_in"], n_out=st.get("n_out", 8)).cuda().eval()
    gru.load_state_dict(st["model"])
    cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False  # the GRU without cuDNN, as the released model was trained and evaluated
    try:
        for tr in per.values():
            if len(tr["time"]) < MIN_LEN:
                continue
            t = {k: np.asarray(v) for k, v in tr.items() if k != "who"}
            x, _ = tcv.features(t, np.arange(len(t["time"])))
            with torch.no_grad():
                y, _ = gru(torch.from_numpy(x)[None].cuda(), torch.from_numpy(t["logits"].astype(np.float32))[None].cuda())
            prob = torch.sigmoid(y[0]).cpu().numpy().astype(np.float32)
            for (f, i), pr in zip(tr["who"], prob):
                frames[f][i]["contact_video"] = pr
    finally:
        torch.backends.cudnn.enabled = cudnn


def run(a):
    from fab.predictor import FABPredictor

    if a.weights:
        os.environ["FAB_WEIGHTS"] = a.weights
    src, vfps, names = frame_source(a.video, a.max_frames)
    fps = a.fps or vfps or 30.0
    boxes = None
    if a.boxes:
        boxes = np.load(a.boxes)["box"].astype(np.float32)
        boxes = boxes[:, None] if boxes.ndim == 2 else boxes
    kw = {"detector": ""} if boxes is not None else {}
    if not a.trt:
        kw["body_engine"] = ""
    pred = FABPredictor(a.model, fast=not a.eager, profile=a.profile, **kw)
    first = next(src, None)
    if first is None:
        raise SystemExit(f"no frames in {a.video}")
    H, W = first.shape[:2]
    K = intrinsics(a, H, W)

    def stream():
        yield first
        yield from src

    if boxes is not None:  # given boxes: .predict per frame, the previous frame's same person (by order) as its prev
        def per_frame():
            last = None
            for t, img in enumerate(stream()):
                b = boxes[t][np.isfinite(boxes[t]).all(1)] if t < len(boxes) else np.zeros((0, 4), np.float32)
                last = pred.predict(img, K, b, prev=last if last is not None and len(last) == len(b) else None)
                yield last
        it, call = per_frame(), "predict(image, K, boxes, prev)"
    else:
        it, call = pred.predict_video(stream(), K), "predict_video(frames, K)"
    g1_limits = None
    if getattr(pred, "kin", None) is not None:
        g1_limits = (pred.kin.lower.float().cpu().numpy(), pred.kin.upper.float().cpu().numpy())
    frames = []
    for t, people in enumerate(it):
        frames.append([dict(q) for q in people])
        if t % 100 == 0:
            print(f"[fab-video] frame {t}: {len(people)} people", flush=True)
    it = pred = None  # free the predictor (and its compiled graphs): the video contacts load their own forward
    import gc

    import torch

    gc.collect()
    if hasattr(torch, "_dynamo"):
        torch._dynamo.reset()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    link_tracks(frames)
    if a.lag > 0:
        smooth_tracks(frames, a.lag, a.sigma, g1_limits)
    if a.contact_video and frames and any("contact" in pp for f in frames for pp in f):
        from fab import paths

        video_contacts(a, frames, K, fps, paths.model_file(a.model))
    return {"video": str(a.video), "fps": float(fps), "size": (W, H), "K": K, "model": a.model, "names": names,
            "call": f'FABPredictor("{a.model}", fast={not a.eager}).{call}',
            "frames": [{"index": t, "people": f} for t, f in enumerate(frames)]}


def draw_frame(img, people, res, t, width, points, timeline):
    s = width / img.shape[1]
    out = cv2.resize(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), (width, int(round(img.shape[0] * s))))
    for pp in people:
        kp = (pp.get("keypoints2d", pp["keypoints2d_projected"]) * s).astype(np.float32)
        if points and "dense_surface2d" in pp:
            for x, y in pp["dense_surface2d"] * s:
                cv2.circle(out, (int(x), int(y)), 1, (220, 200, 120), -1, cv2.LINE_AA)
        for i, j in BODY_LINKS:
            cv2.line(out, tuple(map(int, kp[i])), tuple(map(int, kp[j])), (255, 255, 255), 2, cv2.LINE_AA)
        c = pp.get("contact_video", pp.get("contact"))
        for _, ch, k in FEET:
            on = c is not None and c[ch] >= 0.5
            cv2.circle(out, tuple(map(int, kp[k])), 6, (60, 200, 60) if on else (160, 160, 160), -1 if on else 2, cv2.LINE_AA)
        x0, y0 = int(pp["box"][0] * s), int(pp["box"][1] * s)
        cv2.putText(out, f"#{pp['track']}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([out, timeline(t, out.shape[1])])


def make_timeline(res):
    """Per-foot contact timeline of the longest track (the video contacts, else the image head's), cursor at t."""
    n = len(res["frames"])
    count = defaultdict(int)
    for f in res["frames"]:
        for pp in f["people"]:
            count[pp["track"]] += 1
    main = max(count, key=count.get) if count else None
    P = np.full((n, 4), np.nan, np.float32)
    for t, f in enumerate(res["frames"]):
        for pp in f["people"]:
            if pp["track"] == main:
                c = pp.get("contact_video", pp.get("contact"))
                if c is not None:
                    P[t] = c[:4]
    src = "video contact model" if any("contact_video" in pp for f in res["frames"] for pp in f["people"]) else "image head"
    row, top, left = 16, 20, 60

    def timeline(t, width):
        im = np.full((top + 4 * row + 6, width, 3), 24, np.uint8)
        cv2.putText(im, f"static foot contact, track #{main} ({src})", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (220, 220, 220), 1, cv2.LINE_AA)
        xs = np.linspace(left, width - 6, n + 1).astype(int)
        for r, (lab, ch, _) in enumerate(FEET):
            y = top + r * row
            cv2.putText(im, lab, (6, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
            for i in range(n):
                v = P[i, ch]
                if np.isfinite(v):
                    g = int(40 + 200 * v)
                    cv2.rectangle(im, (xs[i], y + 2), (max(xs[i], xs[i + 1] - 1), y + row - 3),
                                  (60, g, 60) if v >= 0.5 else (g // 2, g // 2, g // 2), -1)
        cx = int(xs[min(t, n - 1)])
        cv2.line(im, (cx, top - 2), (cx, top + 4 * row), (0, 200, 255), 1)
        return im

    return timeline


def render(res, a, out):
    src, _, _ = frame_source(res["video"], len(res["frames"]))
    timeline = make_timeline(res)
    poster = 0 if a.poster < 0 else int(round(a.poster * res["fps"]))
    writer, proc = None, None
    for t, img in enumerate(src):
        fr = draw_frame(img, res["frames"][t]["people"], res, t, a.width, a.points, timeline)
        fr = fr[: fr.shape[0] // 2 * 2, : fr.shape[1] // 2 * 2]
        if t == poster:
            cv2.imwrite(str(Path(out).with_suffix(".jpg")), fr)
        if writer is None and proc is None:
            h, w = fr.shape[:2]
            if shutil.which("ffmpeg"):
                proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                                         "-s", f"{w}x{h}", "-r", f"{res['fps']:g}", "-i", "-", "-an", "-c:v", "libx264",
                                         "-crf", str(a.crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                                        stdin=subprocess.PIPE)
            else:
                writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), res["fps"], (w, h))
        if proc is not None:
            proc.stdin.write(fr.tobytes())
        else:
            writer.write(fr)
    if proc is not None:
        proc.stdin.close()
        proc.wait()
    elif writer is not None:
        writer.release()


def main(argv=None):
    a = parser().parse_args(argv)
    name = Path(a.video).stem if not Path(a.video).is_dir() else Path(a.video).name
    out = Path(a.out or f"output/{name}_fab.mp4")
    res_path = Path(a.results or out.with_suffix(".pkl"))
    if a.load:
        res = pickle.load(open(a.load, "rb"))
    else:
        res = run(a)
        res_path.parent.mkdir(parents=True, exist_ok=True)
        with open(res_path, "wb") as fh:
            pickle.dump(res, fh)
        n = sum(len(f["people"]) for f in res["frames"])
        print(f"[fab-video] {len(res['frames'])} frames, {n} people -> {res_path}", flush=True)
    if a.render and res["frames"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        render(res, a, out)
        print(f"[fab-video] render -> {out} (poster {out.with_suffix('.jpg')})", flush=True)


if __name__ == "__main__":
    main()
