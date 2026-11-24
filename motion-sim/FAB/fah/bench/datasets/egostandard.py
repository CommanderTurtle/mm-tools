"""EgoStandard (https://huggingface.co/datasets/LightwheelAI/EgoStandard): the head-camera line of Lightwheel's
EgoSuite-Open100K (2026-08). License commercial-training-no-resale-v1.0: training / evaluation / benchmarking are
allowed, mirroring or publishing any frames, crops or records is not (keep everything private; only
aggregate metrics may be published). Cite as LightwheelAI/EgoStandard with a link.

Data: HF Bucket LightwheelAI/EgoStandard, one MCAP per episode under {EgoStand,EgoStandard,EgoStandard-body}/mcap/
<task name>/<uuid>.mcap. Topics used (schemas: github.com/LightwheelAI/LW-Egosuite-DevKit, docs.lightwheel.net):
  /sensor/camera/head_left/video       foxglove.CompressedVideo, undistorted H.264, one frame per message
  /sensor/camera/head_left/intrinsic   foxglove.CameraCalibration (K, P, width, height)
  /sensor/camera/head_left/extrinsic   foxglove.FrameTransforms, camera pose in world (parent=world)
  /pose/headcam                        head_left camera pose in world (fallback for a static extrinsic)
  /pose/{left,right}_hand              21 joints per hand in world (X fwd, Y left, Z up, m), OpenPose order
                                       (wrist, thumb CMC/MCP/IP/tip, index..pinky MCP/PIP/DIP/tip)
  /annotation/bad_frame/pose/hand, /annotation/bad_frame/camera/head_left   optional per-frame QC flags
No MANO / mesh: GT are the capture system's tracked keypoints (pseudo-GT, skeleton != MANO joint definition).

The benchmark reads converted clips (bench.datasets.build_egostandard): <root>/clips/<name>/{%06d.jpg, gt.npz,
meta.json[, mano.npz]}; GT in the head_left camera frame (OpenCV axes), like bench.datasets.hot3d; frame dicts carry
T_world_cam (camera -> world, per frame) for bench.world.
"""
import json
import os
import subprocess
import tempfile

import cv2
import numpy as np

BUCKET = "LightwheelAI/EgoStandard"
SUBSETS = ("EgoStand", "EgoStandard", "EgoStandard-body")
BODY_TO_OPTICAL = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # X fwd/Y left/Z up -> OpenCV


def quat_to_R(w, x, y, z):
    n = np.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _pose(pos, quat):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(quat.w, quat.x, quat.y, quat.z)
    T[:3, 3] = pos.x, pos.y, pos.z
    return T


def _nearest(ts_src, ts_dst, tol):
    """Index into ts_src (sorted) nearest to each ts_dst, -1 if farther than tol."""
    if len(ts_src) == 0:
        return np.full(len(ts_dst), -1)
    j = np.clip(np.searchsorted(ts_src, ts_dst), 1, len(ts_src) - 1) if len(ts_src) > 1 else np.zeros(len(ts_dst), int)
    if len(ts_src) > 1:
        j = np.where(np.abs(ts_src[j - 1] - ts_dst) <= np.abs(ts_src[j] - ts_dst), j - 1, j)
    return np.where(np.abs(ts_src[j] - ts_dst) <= tol, j, -1)


def _flags(d):
    """(bad_left, bad_right) from a decoded BadFrame* message dict (field names vary per variant)."""
    def bad(v):
        return bool(v.get("isBad", v.get("is_bad", False))) if isinstance(v, dict) else bool(v)
    left = [v for k, v in d.items() if "left" in k.lower()]
    right = [v for k, v in d.items() if "right" in k.lower()]
    glob_bad = bad(d) if ("isBad" in d or "is_bad" in d) else False
    return glob_bad or any(bad(v) for v in left), glob_bad or any(bad(v) for v in right)


def read_episode(path, cam="head_left", convention="auto"):
    """Per video frame: timestamps, K, camera->world pose (OpenCV camera axes), world joints per hand (NaN if
    absent), bad-frame flags and session metadata. Video payloads are not decoded."""
    from google.protobuf.json_format import MessageToDict
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    vtopic = f"/sensor/camera/{cam}/video"
    topics = [f"/sensor/camera/{cam}/intrinsic", f"/sensor/camera/{cam}/extrinsic", "/pose/headcam",
              "/pose/left_hand", "/pose/right_hand", "/annotation/bad_frame/pose/hand",
              f"/annotation/bad_frame/camera/{cam}", "/session/metadata"]
    with open(path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        avail = {c.topic for c in summary.channels.values()} if summary else set()
        vts = np.array([m.log_time for _, _, m in reader.iter_messages(topics=[vtopic])], np.int64)
        f.seek(0)
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        calib, extr, headcam, hands, bad_hand, bad_img, meta = None, [], [], {"left": [], "right": []}, [], [], {}
        for _, ch, m, msg in reader.iter_decoded_messages(topics=[t for t in topics if t in avail]):
            t = ch.topic
            if t.endswith("/intrinsic"):
                calib = calib or msg
            elif t.endswith("/extrinsic"):
                for tf in msg.transforms:
                    extr.append((m.log_time, _pose(tf.translation, tf.rotation)))
            elif t == "/pose/headcam":
                if len(msg.transforms):
                    headcam.append((m.log_time, _pose(msg.transforms[0].pos, msg.transforms[0].quat)))
            elif t in ("/pose/left_hand", "/pose/right_hand"):
                j = np.array([[tf.pos.x, tf.pos.y, tf.pos.z] for tf in msg.transforms], np.float64)
                if j.shape == (21, 3) and np.isfinite(j).all() and np.abs(j).sum() > 0:
                    hands["left" if "left" in t else "right"].append((m.log_time, j))
            elif t == "/annotation/bad_frame/pose/hand":
                bad_hand.append((m.log_time, _flags(MessageToDict(msg))))
            elif "/bad_frame/camera/" in t:
                bad_img.append((m.log_time, _flags(MessageToDict(msg))[0]))
            elif t == "/session/metadata":
                meta = MessageToDict(msg)
    n = len(vts)
    dt = np.median(np.diff(vts)) if n > 1 else 33_000_000
    tol = dt / 2
    W, H = int(calib.width), int(calib.height)
    K = np.asarray(calib.K, np.float64).reshape(3, 3)
    P = np.asarray(calib.P, np.float64).reshape(3, 4)[:, :3] if len(calib.P) == 12 else None
    if P is not None and P[0, 0] > 0 and not np.allclose(P, K, rtol=1e-3):
        K = P  # rectified / undistorted projection
    # camera pose: per-frame extrinsic if dense, else head-camera pose topic
    src = extr if len(extr) >= 0.5 * n else headcam
    src.sort(key=lambda x: x[0])
    sts = np.array([s[0] for s in src], np.int64)
    k = _nearest(sts, vts, max(tol, 5 * dt)) if len(src) > 1 else np.zeros(n, int)
    T = np.stack([src[i][1] if i >= 0 else np.full((4, 4), np.nan) for i in k]) if len(src) else np.full((n, 4, 4), np.nan)
    joints = {}
    for side, lst in hands.items():
        lst.sort(key=lambda x: x[0])
        idx = _nearest(np.array([a for a, _ in lst], np.int64), vts, tol)
        joints[side] = np.stack([lst[i][1] if i >= 0 else np.full((21, 3), np.nan) for i in idx]) if n else np.zeros((0, 21, 3))
    flags = {}
    for name, lst, dflt in (("bad_hand", bad_hand, (False, False)), ("bad_img", bad_img, False)):
        lst.sort(key=lambda x: x[0])
        idx = _nearest(np.array([a for a, _ in lst], np.int64), vts, tol)
        flags[name] = np.array([lst[i][1] if i >= 0 else dflt for i in idx], bool)
    ep = dict(ts=vts, K=K, wh=(W, H), T_world_cam_raw=T, joints_left=joints["left"], joints_right=joints["right"],
              bad_left=flags["bad_hand"].reshape(n, -1)[:, 0], bad_right=flags["bad_hand"].reshape(n, -1)[:, -1],
              bad_img=flags["bad_img"].reshape(n), meta=meta, pose_source="extrinsic" if src is extr else "headcam",
              fps=1e9 / dt, counts=dict(video=n, extrinsic=len(extr), headcam=len(headcam), left=len(hands["left"]),
                                        right=len(hands["right"]), bad_hand=len(bad_hand), bad_img=len(bad_img)))
    ep["convention"], ep["in_view"] = choose_convention(ep) if convention == "auto" else (convention, None)
    R_fix = np.eye(3) if ep["convention"] == "optical" else BODY_TO_OPTICAL.T
    ep["T_world_cam"] = T.copy()
    ep["T_world_cam"][:, :3, :3] = T[:, :3, :3] @ R_fix  # camera axes -> OpenCV
    return ep


def to_camera(T_world_cam, X):
    """World points (..., 3) -> camera frame of a camera->world pose."""
    R, t = T_world_cam[:3, :3], T_world_cam[:3, 3]
    return (X - t) @ R


def project(K, Xc):
    return Xc[..., :2] / np.clip(Xc[..., 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]


def in_image(uv, Xc, wh):
    return (Xc[..., 2] > 0.01) & (uv[..., 0] >= 0) & (uv[..., 0] < wh[0]) & (uv[..., 1] >= 0) & (uv[..., 1] < wh[1])


def choose_convention(ep, stride=10):
    """Camera axes of the published pose: 'optical' (OpenCV) or 'body' (X fwd, Y left, Z up), whichever puts more
    tracked hand joints inside the image (egocentric hands are in view most of the time)."""
    score = {}
    for name, R_fix in (("optical", np.eye(3)), ("body", BODY_TO_OPTICAL.T)):
        hit, tot = 0, 0
        for i in range(0, len(ep["ts"]), stride):
            T = ep["T_world_cam_raw"][i].copy()
            if not np.isfinite(T).all():
                continue
            T[:3, :3] = T[:3, :3] @ R_fix
            for side in ("left", "right"):
                j = ep[f"joints_{side}"][i]
                if np.isfinite(j).all():
                    Xc = to_camera(T, j)
                    hit += in_image(project(ep["K"], Xc), Xc, ep["wh"]).sum()
                    tot += 21
        score[name] = hit / max(tot, 1)
    return max(score, key=score.get), score


def iter_frames(path, keep, cam="head_left", threads=4):
    """Yield (frame index, BGR image) for the requested video frame indices in order (decodes up to max(keep))."""
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    keep, last = set(int(i) for i in keep), max(keep)
    with tempfile.NamedTemporaryFile(suffix=".h264") as tmp:
        with open(path, "rb") as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])
            for n, (_, _, _, msg) in enumerate(reader.iter_decoded_messages(topics=[f"/sensor/camera/{cam}/video"])):
                tmp.write(bytes(msg.data))
                if n >= last:
                    break
        tmp.flush()
        wh = subprocess.run(["ffprobe", "-v", "error", "-f", "h264", "-i", tmp.name, "-show_entries", "stream=width,height",
                             "-of", "csv=p=0"], capture_output=True, text=True, check=True).stdout.split()[0]
        W, H = map(int, wh.split(",")[:2])
        proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-threads", str(threads), "-f", "h264", "-i", tmp.name,
                                 "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        n = 0
        try:
            while n <= last:
                buf = proc.stdout.read(W * H * 3)
                if len(buf) < W * H * 3:
                    break
                if n in keep:
                    yield n, np.frombuffer(buf, np.uint8).reshape(H, W, 3).copy()
                n += 1
        finally:
            proc.kill()
            proc.wait()


def decode_frames(path, keep, cam="head_left"):
    """{frame index: BGR image} for the requested video frame indices."""
    return dict(iter_frames(path, keep, cam))


def list_bucket(cache=None, token=None):
    """[{path, size, subset, task, uuid}] for every episode MCAP in the bucket (cached as jsonl)."""
    if cache and os.path.exists(cache):
        return [json.loads(l) for l in open(cache)]
    from huggingface_hub import list_bucket_tree

    eps = []
    for sub in SUBSETS:
        for it in list_bucket_tree(BUCKET, prefix=f"{sub}/mcap", recursive=True, token=token):
            p = it.path
            if getattr(it, "type", "file") != "file" or not p.endswith(".mcap"):
                continue
            parts = p.split("/")
            eps.append(dict(path=p, size=int(it.size), subset=parts[0], task="/".join(parts[2:-1]), uuid=parts[-1][:-5]))
    if cache:
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        with open(cache + ".tmp", "w") as f:
            for e in eps:
                f.write(json.dumps(e) + "\n")
        os.replace(cache + ".tmp", cache)
    return eps


def download(paths, raw_root, token=None):
    """Bucket paths -> <raw_root>/<path> (existing files are kept; written as .part, renamed when complete)."""
    todo = [(p, os.path.join(raw_root, p)) for p in paths if not os.path.exists(os.path.join(raw_root, p))]
    for _, dst in todo:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    if todo:
        from huggingface_hub import download_bucket_files

        for k in range(0, len(todo), 8):  # small batches: a preempted run keeps what finished
            batch = todo[k:k + 8]
            download_bucket_files(BUCKET, files=[(p, dst + ".part") for p, dst in batch], token=token)
            for _, dst in batch:
                os.replace(dst + ".part", dst)
    return [os.path.join(raw_root, p) for p in paths]


def hand_gt(j_world, T_world_cam, K, wh, is_right, v_world=None):
    """Camera-frame GT hand dict (bench format) or None. Box / visible = visible projected joints, or mesh vertices
    when given (MANO-fit targets, as HOT3D)."""
    if not np.isfinite(j_world).all() or not np.isfinite(T_world_cam).all():
        return None
    j = to_camera(T_world_cam, j_world)
    uv = project(K, j)
    inside = in_image(uv, j, wh)
    v = None if v_world is None else to_camera(T_world_cam, v_world)
    pts, pin = (uv, inside) if v is None else (project(K, v), None)
    if v is not None:
        pin = in_image(pts, v, wh)
    if pin.sum() < 2:
        return None
    p = pts[pin]
    box = np.array([*p.min(0), *p.max(0)], dtype=np.float32)
    return dict(is_right=is_right, joints3d=j.astype(np.float32), verts3d=None if v is None else v.astype(np.float32),
                joints2d=uv.astype(np.float32), bbox=box, visible=float(pin.mean()), inside=inside)


class EgoStandardClip:
    """Converted clip: frames + GT (bench.run frame dicts). Hands with < min_visible of their joints (vertices for
    MANO-fit targets) in the image or a bad-frame flag are not scored (detections on them are not false positives).
    targets: "mano" = MANO fits of the keypoints (bench.datasets.egostandard_fit, default),
    "raw" = the tracked keypoints with joint boxes; default $EGOSTD_TARGETS or "mano"."""

    def __init__(self, clip_dir, min_visible=0.5, min_box=16, targets=None):
        self.dir = clip_dir
        self.gt = dict(np.load(os.path.join(clip_dir, "gt.npz")))
        self.targets = targets or os.environ.get("EGOSTD_TARGETS", "mano")
        self.mano = dict(np.load(os.path.join(clip_dir, "mano.npz"))) if self.targets == "mano" else None
        self.meta = json.load(open(os.path.join(clip_dir, "meta.json")))
        self.K = self.gt["K"].astype(np.float32)
        self.wh = tuple(int(v) for v in self.gt["wh"])
        self.min_visible, self.min_box = min_visible, min_box
        self.sequence_id = self.meta["uuid"]

    def __len__(self):
        return len(self.gt["index"])

    def frame(self, i):
        idx = int(self.gt["index"][i])
        img = cv2.imread(os.path.join(self.dir, f"{idx:06d}.jpg"))
        T = self.gt["T_world_cam"][i]
        hands, ignore = [], []
        for side, is_right in (("left", False), ("right", True)):
            if self.mano is None:
                h = hand_gt(self.gt[f"joints_{side}"][i], T, self.K, self.wh, is_right)
            else:
                h = hand_gt(self.mano[f"joints_{side}"][i], T, self.K, self.wh, is_right, self.mano[f"verts_{side}"][i])
            if h is None:
                continue
            b = h["bbox"]
            ok = (h["visible"] >= self.min_visible and not self.gt[f"bad_{side}"][i] and not self.gt["bad_img"][i]
                  and max(b[2] - b[0], b[3] - b[1]) >= self.min_box)
            (hands if ok else ignore).append(h)
        return dict(image=img, K=self.K.copy(), hands=hands, ignore=ignore, key=f"{idx:06d}", sequence_id=self.sequence_id,
                    device="egostandard", sku=self.meta["subset"], T_world_cam=T)  # not "subset": that marks image sets
