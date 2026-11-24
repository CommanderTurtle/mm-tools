"""FAB motions for NVIDIA SONIC (GEAR-SONIC, the universal Unitree G1 tracking controller): FAB's predictor in video
mode on a benchmark sequence, SMPL fitted to the dense head's surface points (the released one-pass graph fit,
--fit_iters 0; or the learned fitter + L-BFGS),
lifted to a gravity-aligned world with the dataset's camera poses (oracle camera), lightly smoothed, resampled to
SONIC's 50 fps. Writes
  <out>/<name>_raw.npz     per-frame camera-frame fit + G1 head outputs (reused with --reuse)
  <out>/<name>_motion.npz  50 fps references: SMPL (SONIC's y-up smpl_filtered convention) and G1 (z-up), see save;
                           G1 from the in-model G1 head
  <out>/<name>__ik_motion.npz  (--g1_ik) the same SMPL, G1 from IK on the reference SMPL (fab/retarget/g1.py)
  <out>/<name>_input.mp4   input video + the smoothed reference SMPL mesh projected back (what SONIC is asked to track)
  <out>/<name>_info.json   diagnostics
SONIC's motion library (GR00T-WholeBodyControl) reads these references after packing them into its PKL format.

World: z up, x along the first frame's horizontal camera viewing direction, first-frame pelvis above the origin. The
dataset world's up axis is the mean of the ground-truth body's and the camera's up directions, snapped to a coordinate
axis (a per-dataset convention, not a per-frame oracle). Uneven ground (stairs) is flattened: heights are measured above a
local floor (rolling minimum of the body's lowest point), so the references live on SONIC's flat ground.

    MUJOCO_GL=egl python -m bench.sonic_motion --dataset emdb --root <EMDB root> --seq P5/42_indoor_dancing \
        --fab_ckpt fab-l --fit_iters 0 --g1_ik --out <dir>
Helpers used elsewhere: run_fab / one_pass_fit (bench/videomimic/recon.py, world mode) and g1_lowest
(bench/videomimic/scene_retarget.py, the retargeter's root start).
"""

import argparse
import json
import os
import pickle
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from scipy.spatial.transform import Rotation

os.environ.setdefault("MUJOCO_GL", "egl")

from fab.retarget import g1_files  # noqa: E402

RX_M90 = Rotation.from_euler("x", -90, degrees=True).as_matrix()  # z-up world -> SMPL's y-up (SONIC smpl_y_up)


def load_sequence(dataset, root, seq, person=0):
    """All frames of a sequence (not only the benchmark's good frames): image paths, K, world -> camera extrinsics
    (N, 4, 4), the ground-truth SMPL global orientations (for the dataset world's up axis only) and the subject's
    dataset box per frame (N, 4) xyxy, NaN where unknown (to pick the subject among FAB's detections only)."""
    from bench.datasets import _bbox_from_keypoints

    if dataset in ("rich", "moyo"):  # bench/sonic_clips.py loaders (RICH val 10 fps frames, MOYO contiguous clips)
        from bench.sonic_clips import load_sequence as clip_sequence

        return clip_sequence(dataset, root, seq)
    root = Path(root)
    if dataset == "emdb":
        pkl = next((root / seq).glob("*_data.pkl"))
        with open(pkl, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        ext = np.asarray(d["camera"]["extrinsics"], np.float64)
        images = [str(pkl.parent / "images" / f"{i:05d}.jpg") for i in range(len(ext))]
        boxes = np.asarray(d["bboxes"]["bboxes"], np.float64).copy()
        boxes[np.asarray(d["bboxes"].get("invalid_idxs", []), int)] = np.nan
        return (images, np.asarray(d["camera"]["intrinsics"], np.float64), ext, np.asarray(d["smpl"]["poses_root"]),
                boxes)
    pkl = next(p for s in ("test", "validation", "train") for p in [root / "sequenceFiles" / s / f"{seq}.pkl"]
               if p.exists())
    with open(pkl, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    ext = np.asarray(d["cam_poses"], np.float64)
    images = [str(root / "imageFiles" / seq / f"image_{i:05d}.jpg") for i in range(len(ext))]
    go = np.concatenate([np.asarray(p)[:, :3] for p in d["poses"]])
    boxes = np.full((len(ext), 4), np.nan)
    for i, p2 in enumerate(np.asarray(d["poses2d"][person])):
        b = _bbox_from_keypoints(p2[:2].T, p2[2])
        if b is not None:
            boxes[i] = b
    return images, np.asarray(d["cam_intrinsics"], np.float64), ext, go, boxes


def scene_surface(pieces, radius=0.0):
    """Convex scene pieces (z-up, [N_i, 3] each) -> heights(xy [N, 2]) = the top surface under each point, 0 (the
    floor) where no piece is; radius > 0: the highest surface within that horizontal radius."""
    from scipy.spatial import ConvexHull

    hulls = []
    for v in pieces:
        try:
            hulls.append(ConvexHull(v).equations)  # n . p + d <= 0 inside
        except Exception:
            continue

    def heights(xy):
        xy = np.asarray(xy, np.float64)
        out = np.zeros(len(xy))
        for eq in hulls:
            n, d = eq[:, :3], eq[:, 3]
            up, dn, side = n[:, 2] > 1e-6, n[:, 2] < -1e-6, np.abs(n[:, 2]) <= 1e-6
            lin = xy @ n[:, :2].T + d
            ztop = (-lin[:, up] / n[up, 2]).min(1) if up.any() else np.full(len(xy), -np.inf)
            zbot = (-lin[:, dn] / n[dn, 2]).max(1) if dn.any() else np.full(len(xy), -np.inf)
            inside = (ztop >= zbot) & ((lin[:, side] <= 1e-9).all(1) if side.any() else True)
            out = np.where(inside, np.maximum(out, ztop), out)
        return out

    if radius <= 0:
        return heights
    ring = [np.zeros(2)] + [r * np.array([np.cos(a), np.sin(a)]) for r in (radius / 2, radius)
                            for a in np.linspace(0, 2 * np.pi, 12, endpoint=False)]
    return lambda xy: np.max([heights(np.asarray(xy, np.float64) + o) for o in ring], 0)


def _top_offsets(search, step, inset):
    g = np.arange(-search, search + 1e-9, step)
    off = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    off = off[np.linalg.norm(off, axis=1) <= search]
    return off[np.argsort(np.linalg.norm(off, axis=1), kind="stable")], \
        [inset * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]


def nearest_top(surf, p, off, ring, min_h=0.15, dh=0.15):
    """Offset (xy) from the limb point p to the nearest support-top point higher than min_h, at least the ring
    radius inside that top (the surface within 2 cm of its height), whose height is within dh of p; None if none."""
    h = surf(p[:2] + off)
    ok = (h > min_h) & (np.abs(h - p[2]) < dh)
    for o in ring:
        ok &= np.abs(surf(p[:2] + off + o) - h) < 0.02
    return off[np.flatnonzero(ok)[0]] if ok.any() else None  # off is sorted by length


def scene_register(pieces, V, part, pred, groups, search=0.35, min_h=0.15, step=0.01, inset=0.02):
    """Horizontal shift (human-scale flat world) that puts the scene's support tops under the predicted hand / knee
    contacts: per contact frame, the offset from the limb's lowest vertex to the nearest support-top point
    (nearest_top: higher than min_h, at least `inset` inside the top, within 0.15 m of the limb's height); the median
    over frames and limbs."""
    surf = scene_surface(pieces)
    off, ring = _top_offsets(search, step, inset)
    shifts = []
    for name in ("lhand", "rhand", "lknee", "rknee"):
        m = pred.get(name)
        if m is None:
            continue
        idx = np.flatnonzero(np.isin(part, groups[name]))
        for t in np.flatnonzero(m)[::2]:
            o = nearest_top(surf, V[t, idx][V[t, idx, 2].argmin()], off, ring, min_h)
            if o is not None:
                shifts.append(-o)
    if not shifts:
        return np.zeros(2), 0
    return np.median(np.array(shifts), 0), len(shifts)


def support_targets(pieces, V, part, labels, groups, search=0.35, min_h=0.15, step=0.01, inset=0.02):
    """Per labelled hand / knee contact frame: the support-top point (xy, human-scale flat world) nearest the limb's
    lowest vertex (nearest_top), NaN where none: the hold position on thin supports (rails, tubes)."""
    surf = scene_surface(pieces)
    off, ring = _top_offsets(search, step, inset)
    out = {}
    for name in ("lhand", "rhand", "lknee", "rknee"):
        m = labels.get(name)
        if m is None or not np.any(m):
            continue
        idx = np.flatnonzero(np.isin(part, groups[name]))
        xy = np.full((len(m), 2), np.nan)
        for t in np.flatnonzero(m):
            p = V[t, idx][V[t, idx, 2].argmin()]
            o = nearest_top(surf, p, off, ring, min_h)
            if o is not None:
                xy[t] = p[:2] + o
        out[name] = xy
    return out


def dataset_up(gt_go, R_wc):
    """Mean body up direction of the ground-truth SMPL plus the mean camera up direction (both lean on stairs, in
    opposite ways), snapped to the nearest world axis."""
    body = (Rotation.from_rotvec(gt_go).as_matrix() @ np.array([0.0, 1.0, 0.0])).mean(0)
    cam = (R_wc @ np.array([0.0, -1.0, 0.0])).mean(0)
    up = body / np.linalg.norm(body) + cam / np.linalg.norm(cam)
    up /= np.linalg.norm(up)
    k = np.abs(up).argmax()
    axis = np.zeros(3)
    axis[k] = np.sign(up[k])
    ang = np.degrees(np.arccos(np.clip(up @ axis, -1, 1)))
    return (axis if ang < 35 else up), up, ang


def rot_between(a, b):
    v, c = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else Rotation.from_rotvec(np.pi * np.array([1.0, 0, 0])).as_matrix()
    return Rotation.from_rotvec(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), c)).as_matrix()


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
    return w * h / max(u, 1e-9)


def run_fab(args, images, K, gt_boxes):
    """Per frame: the subject's dense surface points (camera frame), G1 head outputs, dense mesh. The subject is the
    detection overlapping the dataset's box (identity only); frames without that box are left missing (filled by
    interpolation later). Without any dataset boxes: the person tracked from the previous frame."""
    from fab.predictor import FABPredictor

    one_pass = args.fit_iters == 0  # the released path: the predictor's graph-captured one-pass SMPL fit
    kw = json.loads(args.predictor_kw) if args.predictor_kw else {}  # e.g. a predictor profile's kwargs
    fab = FABPredictor(args.fab_ckpt, detector=args.detector, bm_dir=args.bm_dir,
                       smpl_fitter=kw.pop("smpl_fitter", args.fitter) if one_pass else "", fast=args.fast, **kw)
    load = lambda p: cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)  # noqa: E731
    keys = ("pts", "g1_q", "g1_root", "mesh", "box", "contact") + (("fit_pose", "fit_betas", "fit_mesh") if one_pass else ())
    out = {"valid": [], **{k: [] for k in keys}}
    prev = None
    boxed = np.isfinite(gt_boxes).all(1).any()
    for t, people in enumerate(fab.predict_video((load(p) for p in images), K)):
        pp = None
        if people and np.isfinite(gt_boxes[t]).all():
            best = max(people, key=lambda x: iou(x["box"], gt_boxes[t]))
            pp = best if iou(best["box"], gt_boxes[t]) > 0.2 else None
        elif people and not boxed:  # frames without the dataset's box (subject out of view) stay missing
            if prev is not None:
                best = max(people, key=lambda x: iou(x["box"], prev))
                pp = best if iou(best["box"], prev) > 0.1 else None
            if pp is None:
                pp = max(people, key=lambda x: np.ptp(x["keypoints2d_projected"][:, 1]))  # the most prominent person
        out["valid"].append(pp is not None)
        if pp is None:
            for k in keys:
                out[k].append(out[k][-1] if out[k] else None)
            continue
        prev = pp["box"]
        # FAB-H (no dense head, no G1 head): no surface points (only the --fit_iters L-BFGS path needs them) and
        # identity G1 head outputs (its references use IK on the fitted SMPL)
        out["pts"].append(pp["dense_surface_cam"] if "dense_surface_cam" in pp else np.zeros((1, 3), np.float32))  # camera frame
        out["g1_q"].append(pp.get("g1_q", np.zeros(29, np.float32)))
        out["g1_root"].append(pp.get("g1_root", np.eye(3, dtype=np.float32)))
        out["mesh"].append(pp.get("mesh_smpl_dense", pp["mesh_smpl"]))
        out["box"].append(pp["box"])
        out["contact"].append(pp.get("contact", np.full(8, np.nan, np.float32)))  # contact head (if the checkpoint has one)
        if one_pass:
            out["fit_pose"].append(pp["smpl_pose"])
            out["fit_betas"].append(pp["smpl_betas"])
            out["fit_mesh"].append(pp["mesh_smpl_fit"])
    first = next(i for i, v in enumerate(out["valid"]) if v)
    for k in keys:  # leading misses: first detection
        out[k] = np.stack([out[k][first] if x is None else x for x in out[k]]).astype(np.float32)
    out["valid"] = np.array(out["valid"])
    dense = getattr(fab.model, "dense_head", None)
    surface_idx = dense.surface_idx.cpu().numpy() if dense is not None else np.zeros(0, np.int64)
    if getattr(fab, "kin", None) is not None:
        joints = list(fab.kin.revolute)
    else:
        from fab.retarget.g1 import G1Kinematics

        joints = list(G1Kinematics(g1_files.urdf()).revolute)
    del fab
    torch.cuda.empty_cache()
    return out, surface_idx, joints


def one_pass_fit(body, fab_out):
    """The predictor's one-pass fit (camera frame): pose / betas as predicted, the translation that places
    SMPL(pose, betas) on the returned mesh (which is the fit placed at the points' centroid)."""
    pose, betas, mesh = (torch.from_numpy(fab_out[k]).float().cuda() for k in ("fit_pose", "fit_betas", "fit_mesh"))
    trans = []
    with torch.no_grad():
        for i in range(0, len(pose), 256):
            v, _ = body(pose[i : i + 256, :3], pose[i : i + 256, 3:], betas[i : i + 256], None)
            trans.append(mesh[i : i + 256].mean(1) - v.mean(1))
    return {"pose": fab_out["fit_pose"], "betas": fab_out["fit_betas"], "trans": torch.cat(trans).cpu().numpy(),
            "verts": fab_out["fit_mesh"]}


def fit_smpl_sequence(args, body, pts, idx):
    from fab.smpl_fit import fit_smpl
    from fab.smpl_fitter import load_fitter

    if args.fitter == "deployed":  # the L-BFGS path (--fit_iters > 0): the points-first heads' released fitter (the assert checks it)
        from fab.predictor import DEPLOYED_FITTERS

        args.fitter = DEPLOYED_FITTERS["absolute"]
    st = torch.load(args.fitter, map_location="cpu", weights_only=False)
    assert np.array_equal(np.asarray(st["idx"]), idx), "fitter / dense-head surface points differ"
    fitter = load_fitter(args.fitter)
    idx_t = torch.as_tensor(idx).cuda()
    res = {"pose": [], "betas": [], "verts": []}
    for i in range(0, len(pts), 128):
        p = torch.from_numpy(pts[i : i + 128]).cuda()
        with torch.no_grad():
            init = fitter(p, body, idx_t)
        pose, betas, trans, v = fit_smpl(body, p, idx_t, iters=args.fit_iters, init=init)
        res["pose"].append(pose.cpu().numpy())
        res["betas"].append(betas.cpu().numpy())
        res["verts"].append(v.cpu().numpy())
        res.setdefault("trans", []).append(trans.cpu().numpy())
    return {k: np.concatenate(v) for k, v in res.items()}


def smooth_quats(q, sigma):
    """q (N, ..., 4) xyzw: sign continuity along time, Gaussian filter, renormalise."""
    q = q.copy()
    for t in range(1, len(q)):
        flip = (q[t] * q[t - 1]).sum(-1) < 0
        q[t][flip] *= -1
    if sigma > 0:
        q = gaussian_filter1d(q, sigma, axis=0, mode="nearest")
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def fill_invalid(x, valid):
    """Linear interpolation over frames without a detection (x (N, ...))."""
    if valid.all():
        return x
    t, good = np.arange(len(x)), np.flatnonzero(valid)
    flat = x.reshape(len(x), -1)
    out = np.stack([np.interp(t, good, flat[good, j]) for j in range(flat.shape[1])], 1)
    return out.reshape(x.shape)


def resample(x, fps_in, fps_out, quat=False):
    n = len(x)
    t = np.arange(0, (n - 1) / fps_in + 1e-9, 1 / fps_out) * fps_in
    i0 = np.clip(np.floor(t).astype(int), 0, n - 1)
    i1 = np.clip(i0 + 1, 0, n - 1)
    a = (t - i0).reshape(-1, *([1] * (x.ndim - 1)))
    y = (1 - a) * x[i0] + a * x[i1]  # quaternions: sign-continuous input, nlerp
    return y / np.linalg.norm(y, axis=-1, keepdims=True) if quat else y


def g1_lowest(mjcf, joints, root_quat_wxyz, q, stride=4):
    """Lowest point of the G1 model (all geoms; meshes by vertex) per frame, root at the origin."""
    import mujoco

    m = mujoco.MjModel.from_xml_path(mjcf)
    d = mujoco.MjData(m)
    adr = [m.joint(n).qposadr[0] for n in joints]
    meshes = {g: m.mesh_vert[m.mesh_vertadr[m.geom_dataid[g]] : m.mesh_vertadr[m.geom_dataid[g]]
                             + m.mesh_vertnum[m.geom_dataid[g]]][::stride].astype(np.float64)
              for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH}
    floor = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    low = np.zeros(len(q))
    for t in range(len(q)):
        d.qpos[:3], d.qpos[3:7] = 0.0, root_quat_wxyz[t]
        d.qpos[adr] = q[t]
        mujoco.mj_kinematics(m, d)
        z = []
        for g in range(m.ngeom):
            if g in floor:
                continue
            if g in meshes:
                z.append((meshes[g] @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
            else:
                z.append(d.geom_xpos[g][2] - m.geom_rbound[g])
        low[t] = min(z)
    return low


def write_video(frames, path, fps):
    raw = path.with_name(path.stem + "_raw.mp4")
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w // 2 * 2, h // 2 * 2))
    for f in frames:
        wr.write(cv2.cvtColor(f[: h // 2 * 2, : w // 2 * 2], cv2.COLOR_RGB2BGR))
    wr.release()
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "22", str(path)], check=True)
        raw.unlink()
    else:
        raw.rename(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["emdb", "3dpw", "rich", "moyo"], default="emdb")
    p.add_argument("--root", default="", help="dataset root (not read with --world_only)")
    p.add_argument("--seq", required=True)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=450)
    p.add_argument("--person", type=int, default=0, help="3DPW subject index")
    p.add_argument("--fps", type=float, default=30.0, help="input video frame rate")
    p.add_argument("--fab_ckpt", required=True)
    p.add_argument("--fitter", default="deployed", help="default: the released fitter by head type (FABPredictor's "
                   "DEPLOYED_FITTERS: the points fitter for FAB-L / FAB-S, the mesh fitter for FAB-H)")
    p.add_argument("--fit_iters", type=int, default=0, help="0: the released one-pass graph fit; >0: L-BFGS iterations")
    p.add_argument("--fast", action="store_true", help="FABPredictor(fast=True): compiled body path in CUDA graphs")
    p.add_argument("--predictor_kw", default="", help="JSON of extra FABPredictor kwargs (a profile, e.g. "
                   "fab.predictor.SIMPLE_PROFILE); its smpl_fitter overrides --fitter")
    p.add_argument("--g1_ik", action="store_true", help="also write the IK-on-SMPL G1 reference (<name>__ik)")
    p.add_argument("--ik_iters", type=int, default=150)
    p.add_argument("--sigma_ik", type=float, default=1.0, help="Gaussian smoothing of the IK joint angles (50 fps)")
    p.add_argument("--ik_seq", action="store_true", help="IK reference by retarget_sequence (temporally consistent)")
    p.add_argument("--ik_suffix", default="__ik", help="motion key suffix of the IK reference")
    p.add_argument("--ik_motion", action="store_true",
                   help="IK reference by the motion-based world-space retargeter (fab/retarget/motion.py)")
    p.add_argument("--motion_cfg", default="", help="JSON overrides of fab.retarget.motion.DEFAULTS")
    p.add_argument("--contact_model", action="store_true",
                   help="with --ik_motion: contact labels from the checkpoint's contact head")
    p.add_argument("--contact_pred", default="", help="with --ik_motion: npz (frame, prob (N, 8)) of predicted contacts")
    p.add_argument("--contact_mode", default="replace", choices=["replace", "and", "or", "limbs"],
                   help="predicted contacts replace / gate (and) / extend (or) the mesh-rule labels; limbs: replace only "
                   "the hands / knees (feet keep the mesh rule)")
    p.add_argument("--contact_static", action="store_true", help="split (16-channel) head: every group (hands / knees "
                   "too) on its static channel >= 0.5, not hands / knees on any contact")
    p.add_argument("--contact_runs", action="store_true", help="predicted contacts as stance runs (gaps < 0.1 s filled, "
                   "runs < 0.2 s dropped; the stance pin's definition)")
    p.add_argument("--export_human", default="", help="with --ik_motion: also write the retargeter's input (50 fps, flat "
                   "reference world, human scale: joints, verts, final contact labels) as <dir>/<name>.npz + a flat "
                   "<name>_hfield.npz (bench.videomimic.scene_retarget's recon format: one solver for every clip)")
    p.add_argument("--ik_flex_init", action="store_true", help="sequence / motion IK: start the per-frame IK from the "
                   "human's knee / hip-pitch angles (fab.retarget.g1.flexion_init; knees lock straight otherwise)")
    p.add_argument("--ik_init", default="", help="with --ik_motion: <dir>:<fmt> of another retargeter's reference "
                   "motions (e.g. SOMA / GMR, {c} = clip name) to refine instead of starting from the sequence IK")
    p.add_argument("--no_elev", action="store_true", help="zero elevation above the local floor (no jumps / root-height variation)")
    p.add_argument("--scene", default="", help="with --ik_motion: npz of the clip's scene as convex pieces (pieces: object "
                   "array of (N, 3) dataset-world vertices; ground_y: optional scan ground height along the dataset's y "
                   "up axis, which then replaces the rolling local floor): contacts are held at the scene surface under "
                   "the limb and the limbs gate is 'within --scene_gate of any surface'; writes <name><suffix>_scene_ref.npz "
                   "(pieces in the reference world, G1 scale) for the harness")
    p.add_argument("--scene_gate", type=float, default=0.10, help="limbs gate: limb within this of a surface (m)")
    p.add_argument("--scene_radius", type=float, default=0.0, help="--scene surface under a limb: the highest surface "
                   "within this horizontal radius (m; thin supports such as rails / tubes)")
    p.add_argument("--scene_hold_xy", action="store_true", help="--scene: hand / knee contacts are also held at the "
                   "nearest support-top point horizontally (support_targets), not only at its height")
    p.add_argument("--scene_register", action="store_true", help="--scene with --contact_pred: shift the scene "
                   "horizontally so the support tops lie under the predicted hand / knee contacts (oracle scan, estimated "
                   "person: the person's depth error otherwise puts the limbs beside a thin support)")
    p.add_argument("--up", default="", help="dataset-world up direction 'x,y,z', or 'scene' (the --scene npz's up: the "
                   "scan ground normal), instead of the RICH / MOYO camera-up snap (RICH's world is "
                   "the camera-0 frame: 7-11 deg off gravity)")
    p.add_argument("--root_free_limbs", action="store_true", help="with --ik_motion: no root-height prior on frames with "
                   "a hand / knee contact (the contacts place the root, not the lowest-point rule)")
    p.add_argument("--detector", default="deployed", help="person detector weights (default: the released one)")
    p.add_argument("--bm_dir", default="", help="body models dir (default: fab.paths.body_models_dir())")
    p.add_argument("--mjcf", default=str(g1_files.mjcf()))
    p.add_argument("--sigma_rot", type=float, default=1.5, help="Gaussian smoothing of rotations (input frames)")
    p.add_argument("--sigma_pos", type=float, default=3.0, help="Gaussian smoothing of positions (input frames)")
    p.add_argument("--sigma_root", type=float, default=-1, help="root-orientation smoothing (-1: --sigma_rot)")
    p.add_argument("--sigma_body", type=float, default=-1, help="body-pose / G1 joint smoothing (-1: --sigma_rot)")
    p.add_argument("--g1_smoother", default="gauss", choices=["gauss", "lag"],
                   help="G1 head joints / root: Gaussian (--sigma_rot) or the causal LagSmoother (fab/retarget/smooth.py)")
    p.add_argument("--no_input_video", action="store_true", help="skip the input panel video (references only)")
    p.add_argument("--input_rgb", default="", help="input panel mesh colour r,g,b in 0-1 (default green; paper: sand 0.87,0.81,0.72)")
    p.add_argument("--input_alpha", type=float, default=0.75, help="input panel mesh opacity")
    p.add_argument("--input_edge", type=float, default=0.0, help="> 0: an opaque silhouette outline, this x the mesh colour")
    p.add_argument("--input_shade", type=float, default=0.25, help="input panel: the darkest flat shade (paper sand: 0.6)")
    p.add_argument("--input_blank", action="store_true",
                   help="input panel: the reference mesh on a plain background (always for RICH / MOYO: no dataset pixels)")
    p.add_argument("--g1_lag", type=int, default=3)
    p.add_argument("--g1_lag_sigma", type=float, default=2.5)
    p.add_argument("--height", type=int, default=720, help="input panel height")
    p.add_argument("--out", required=True)
    p.add_argument("--name", default="")
    p.add_argument("--reuse", action="store_true", help="reuse <name>_raw.npz (skip FAB + fit)")
    p.add_argument("--world_only", action="store_true", help="with --world, --reuse and --no_input_video: the clip's frames and cameras "
                   "from the world file, the per-frame fields from <name>_raw.npz; no dataset files are read "
                   "(scripts/reproduce_g1_refs.sh)")
    p.add_argument("--export_pts", action="store_true", help="with --export_human and a raw with dense points (pts, surface_idx): "
                   "also export them in the retargeter's frame (pts, pts_idx; each frame's points registered onto the reference SMPL "
                   "at their vertex ids by a similarity, the residual smoothed as the body rotations, resampled to 50 fps)")
    p.add_argument("--pts_sigma", type=float, default=2.0, help="--export_pts: Gaussian sigma (input frames) of the points' "
                   "residual to the reference SMPL")
    p.add_argument("--world", default="", help="world-space pipeline npz (frames, T_world_cam, gravity_world): estimated "
                   "camera instead of the dataset's (oracle) poses")
    p.add_argument("--world_gravity", default="up", choices=["up", "down"], help="the world file's gravity_world: the up axis "
                   "(world_vipe_g / _u exports) or the true gravity direction (sonic/world_one: z-up world, gravity (0, 0, -1))")
    p.add_argument("--world_scale", action="store_true", help="divide the world file's camera centres by its "
                   "scale_fix (their metric correction), keeping our fitted body size and path self-consistent")
    p.add_argument("--world_scale_mode", default="divide", choices=["divide", "whole"],
                   help="with --world_scale: divide = camera centres / scale_fix (our body scale); whole = keep the "
                        "corrected camera centres and scale the person (camera-frame path and body size) by scale_fix, "
                        "the world fork's whole-world convention (world_export.py); identical G1 references after the "
                        "robot's body-size normalisation")
    args = p.parse_args()
    if not args.bm_dir:  # the installed body models (fab.paths)
        from fab import paths

        args.bm_dir = str(paths.body_models_dir())
    from bench.body_models import SMPL

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.name or f"{args.dataset}_{args.seq.replace('/', '_')}"
    if args.world_only:  # frames + cameras from the world file, the rest from the raw: no dataset files
        assert args.world and args.reuse and args.no_input_video, "--world_only needs --world, --reuse and --no_input_video"
        sel = np.asarray(np.load(args.world, allow_pickle=True)["frames"]).astype(int)
        images, K, ext, gt_go, gt_boxes = [], None, None, None, None
    else:
        images, K, ext, gt_go, gt_boxes = load_sequence(args.dataset, args.root, args.seq, args.person)
        sel = np.arange(args.start, min(len(images), args.start + args.max_frames))
        images, ext, gt_boxes = [images[i] for i in sel], ext[sel], gt_boxes[sel]
    world, body_scale = None, 1.0
    if args.world:  # estimated camera (no oracle): camera-to-world poses + gravity from the world-space pipeline
        world = dict(np.load(args.world, allow_pickle=True))
        assert np.array_equal(np.asarray(world["frames"]).astype(int), sel), "world file frames differ from the clip"
        T_wc_est = np.asarray(world["T_world_cam"], np.float64).copy()
        if args.world_scale and "scale_fix" in world and args.world_scale_mode == "whole":
            body_scale = float(world["scale_fix"])
        elif args.world_scale and "scale_fix" in world:  # undo the world's metric correction: camera centres at the
            T_wc_est[:, :3, 3] /= float(world["scale_fix"])  # camera's own scale, consistent with our body size
        ext = np.linalg.inv(T_wc_est)  # world -> camera, as the dataset's
    body = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
    raw_path = out / f"{name}_raw.npz"
    if args.reuse and raw_path.exists():
        raw = dict(np.load(raw_path, allow_pickle=True))
    else:
        fab_out, idx, joints = run_fab(args, images, K, gt_boxes)
        fit = one_pass_fit(body, fab_out) if args.fit_iters == 0 else fit_smpl_sequence(args, body, fab_out["pts"], idx)
        err = np.linalg.norm(fit["verts"] - fab_out["mesh"], axis=-1).mean(-1) * 1000
        raw = {"frames": sel, "valid": fab_out["valid"], "pose_c": fit["pose"], "betas": fit["betas"],
               "trans_c": fit["trans"], "g1_q": fab_out["g1_q"], "g1_root_c": fab_out["g1_root"],
               "box": fab_out["box"], "fit_vs_dense_mm": err, "joints": np.array(joints), "K": K, "ext": ext,
               "fit_iters": args.fit_iters, "fab_ckpt": args.fab_ckpt, "contact": fab_out["contact"]}
        np.savez(raw_path, **raw)
    valid = raw["valid"].astype(bool)
    n = len(valid)
    betas = np.median(raw["betas"][valid], 0).astype(np.float32)
    zero = torch.zeros(n, 3, device="cuda")

    # camera-frame pelvis of the per-frame fits; global rotation about the pelvis -> any betas keep it
    with torch.no_grad():
        pose_c = torch.from_numpy(raw["pose_c"]).float().cuda()
        _, jc = body(pose_c[:, :3], pose_c[:, 3:], torch.from_numpy(raw["betas"]).float().cuda(),
                     torch.from_numpy(raw["trans_c"]).float().cuda())
        _, j_rest = body(zero[:1], zero[:1].repeat(1, 23), torch.from_numpy(betas)[None].cuda(), None)
        v_rest, _ = body(zero[:1], zero[:1].repeat(1, 23), torch.from_numpy(betas)[None].cuda(), None)
    pelvis_c = jc[:, 0].cpu().numpy().astype(np.float64) * body_scale  # whole-world scale: the person's path too
    J0 = j_rest[0, 0].cpu().numpy().astype(np.float64)

    # dataset world (oracle camera) -> z-up world: up axis, then yaw so the first camera looks along +x
    T_wc = np.linalg.inv(ext)
    R_wc, t_wc = T_wc[:, :3, :3], T_wc[:, :3, 3]
    if args.up:  # explicit gravity (e.g. the scan ground normal), not snapped
        up = (np.asarray(np.load(args.scene, allow_pickle=True)["up"], np.float64) if args.up == "scene"
              else np.array([float(x) for x in args.up.split(",")]))
        up = up / np.linalg.norm(up)
        up_raw, up_ang = up, 0.0
    elif world is None and args.dataset in ("rich", "moyo"):  # static calibrated rigs: camera up snapped to a world axis
        cam_up = (R_wc @ np.array([0.0, -1.0, 0.0])).mean(0)  # (body-up averaging breaks on inversions / lying poses)
        cam_up /= np.linalg.norm(cam_up)
        up = np.zeros(3)
        up[np.abs(cam_up).argmax()] = np.sign(cam_up[np.abs(cam_up).argmax()])
        up_raw, up_ang = cam_up, float(np.degrees(np.arccos(np.clip(cam_up @ up, -1, 1))))
    elif world is None:
        up, up_raw, up_ang = dataset_up(gt_go, R_wc)
    else:  # gravity from the floor fit of the world-space pipeline (not snapped)
        up = np.asarray(world["gravity_world"], np.float64) * (-1.0 if args.world_gravity == "down" else 1.0)
        up = up / np.linalg.norm(up)
        up_raw, up_ang = up, 0.0
    A = rot_between(up, np.array([0.0, 0, 1]))
    fwd = A @ R_wc[0][:, 2]
    yaw = np.arctan2(fwd[1], fwd[0])
    A = Rotation.from_euler("z", -yaw).as_matrix() @ A
    pelvis_w = np.einsum("ij,nj->ni", A, np.einsum("nij,nj->ni", R_wc, pelvis_c) + t_wc)
    R_loc = Rotation.from_rotvec(raw["pose_c"].reshape(n, 24, 3).reshape(-1, 3)).as_matrix().reshape(n, 24, 3, 3)
    R_loc[:, 0] = A @ R_wc @ R_loc[:, 0]  # SMPL global orientation in the z-up world
    g1_R = A @ R_wc @ raw["g1_root_c"].astype(np.float64)

    # fill misses, smooth (input rate), resample to 50 fps
    q_smpl = Rotation.from_matrix(R_loc.reshape(-1, 3, 3)).as_quat().reshape(n, 24, 4)
    q_g1 = Rotation.from_matrix(g1_R).as_quat()
    s_root = args.sigma_rot if args.sigma_root < 0 else args.sigma_root
    s_body = args.sigma_rot if args.sigma_body < 0 else args.sigma_body
    q_smpl = fill_invalid(smooth_quats(q_smpl, 0), valid)
    q_smpl = np.concatenate([smooth_quats(q_smpl[:, :1], s_root), smooth_quats(q_smpl[:, 1:], s_body)], 1)
    q_g1 = smooth_quats(fill_invalid(smooth_quats(q_g1, 0), valid), s_root)
    pelvis_w = fill_invalid(pelvis_w, valid)
    pelvis_w = gaussian_filter1d(pelvis_w, args.sigma_pos, axis=0, mode="nearest") if args.sigma_pos > 0 else pelvis_w
    g1_q = fill_invalid(raw["g1_q"].astype(np.float64), valid)
    if s_body > 0:
        g1_q = gaussian_filter1d(g1_q, s_body, axis=0, mode="nearest")
    if args.g1_smoother == "lag":  # the causal fixed-lag smoother on the in-model G1 head (smooth.py)
        from fab.retarget.smooth import lag_filter

        q_g1 = lag_filter(fill_invalid(smooth_quats(Rotation.from_matrix(g1_R).as_quat(), 0), valid), args.g1_lag,
                          args.g1_lag_sigma)
        q_g1 = q_g1 / np.linalg.norm(q_g1, axis=-1, keepdims=True)
        g1_q = lag_filter(fill_invalid(raw["g1_q"].astype(np.float64), valid), args.g1_lag, args.g1_lag_sigma)
    fps = 50.0
    Q = resample(q_smpl, args.fps, fps, quat=True)
    Qg = resample(q_g1, args.fps, fps, quat=True)
    P = resample(pelvis_w, args.fps, fps)
    G = resample(g1_q, args.fps, fps)
    T = len(P)

    def smpl_world(quats, pelvis):
        """z-up world vertices of the reference SMPL (sequence betas)."""
        R = torch.from_numpy(Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(-1, 24, 3, 3)).float().cuda()
        z = torch.zeros(len(R), 3, device="cuda")
        b = torch.from_numpy(betas)[None].cuda().expand(len(R), -1)
        vs = []
        with torch.no_grad():
            for i in range(0, len(R), 256):
                v, _ = body(z[i : i + 256], z[i : i + 256].repeat(1, 23), b[i : i + 256], None, rotmats=R[i : i + 256])
                vs.append(v.cpu().numpy())
        return (np.concatenate(vs).astype(np.float64) - J0) * body_scale + pelvis[:, None]

    def smpl_world_joints(quats, pelvis):
        """z-up world SMPL joints (24) of the reference SMPL."""
        R = torch.from_numpy(Rotation.from_quat(quats.reshape(-1, 4)).as_matrix().reshape(-1, 24, 3, 3)).float().cuda()
        z = torch.zeros(len(R), 3, device="cuda")
        b = torch.from_numpy(betas)[None].cuda().expand(len(R), -1)
        js = []
        with torch.no_grad():
            for i in range(0, len(R), 256):
                _, j = body(z[i : i + 256], z[i : i + 256].repeat(1, 23), b[i : i + 256], None, rotmats=R[i : i + 256])
                js.append(j.cpu().numpy())
        return (np.concatenate(js).astype(np.float64) - J0) * body_scale + pelvis[:, None]

    V = smpl_world(Q, P)
    lowest = V[..., 2].min(1)
    win = int(round(fps))  # local floor: rolling minimum (+- 1 s) of the lowest point, smoothed
    floor = gaussian_filter1d(minimum_filter1d(lowest, 2 * win + 1, mode="nearest"), fps / 2, mode="nearest")
    scene = np.load(args.scene, allow_pickle=True) if args.scene else None
    if scene is not None and "ground_point" in scene.files:  # the scan's ground (under the person) replaces the
        floor = np.full_like(floor, float((A @ np.asarray(scene["ground_point"], np.float64))[2]))  # rolling floor
    elif scene is not None and "ground_y" in scene.files:  # the scan's ground replaces the rolling local floor
        floor = np.full_like(floor, float((A @ np.array([0.0, float(scene["ground_y"]), 0.0]))[2]))
    elev = np.clip(lowest - floor, 0, None)
    if args.no_elev:  # the body never leaves the local floor (no jumps / root-height variation)
        elev = np.zeros_like(elev)
    P_flat = P.copy()
    P_flat[:, 2] -= floor
    P_flat[:, :2] -= P_flat[0, :2]

    # human -> G1 scale from standing pelvis heights; G1 root height: its lowest point at the scaled elevation
    vr = v_rest[0].cpu().numpy()
    human_pelvis_h = (J0[1] - vr[:, 1].min()) * body_scale
    g1_pelvis_h = 0.793
    s = g1_pelvis_h / human_pelvis_h
    joints = [str(j) for j in raw["joints"]]
    Rg = Rotation.from_quat(Qg)
    g1_low = g1_lowest(args.mjcf, joints, Rg.as_quat()[:, [3, 0, 1, 2]], G)
    g1_root = np.concatenate([s * P_flat[:, :2], (s * elev - g1_low)[:, None]], 1)
    g1_root[:, 2] = gaussian_filter1d(g1_root[:, 2], 1.0, mode="nearest")

    # SONIC SMPL convention: y-up world, global orientation Rx(-90) R_zup, transl so SMPL(params) = y-up mesh
    Rq = Rotation.from_quat(Q.reshape(-1, 4)).as_matrix().reshape(T, 24, 3, 3)
    Rq[:, 0] = RX_M90 @ Rq[:, 0]
    pose_aa = Rotation.from_matrix(Rq.reshape(-1, 3, 3)).as_rotvec().reshape(T, 72)
    transl = P_flat @ RX_M90.T - J0
    np.savez(out / f"{name}_motion.npz", fps=fps, smpl_pose_aa=pose_aa.astype(np.float32),
             smpl_transl=transl.astype(np.float32), smpl_betas=betas, smpl_pelvis_zup=P_flat.astype(np.float32),
             g1_dof=G.astype(np.float32), g1_joint_names=np.array(joints), g1_root_pos=g1_root.astype(np.float32),
             g1_root_quat_xyzw=Qg.astype(np.float32), human_elev=elev.astype(np.float32), scale=s,
             frames=raw["frames"], input_fps=args.fps)
    ik_info = {}
    if args.g1_ik:  # G1 from IK on the reference SMPL: the same 50 fps world joints SONIC's SMPL path receives
        from fab.retarget.g1 import G1Kinematics, retarget_sequence, retarget_smpl

        kin = G1Kinematics(g1_files.urdf()).cuda()
        assert list(kin.revolute) == joints, "IK and head joint orders differ"
        Jw = torch.from_numpy(smpl_world_joints(Q, P)).float().cuda()
        qs, Rs, res = [], [], []
        if args.ik_motion:  # motion-based retargeting (fab/retarget/motion.py): world space, contacts, velocities
            from fab.retarget.contact import (human_contact_labels, human_support_heights, human_support_speeds,
                                                  support_points)
            from fab.retarget.motion import retarget_motion

            Jf = torch.from_numpy(smpl_world_joints(Q, P_flat)).float().cuda()
            t_init = None
            if args.ik_init:  # refinement of another retargeter's per-frame motion (same clip, 50 fps, same world)
                from fab.retarget.g1 import BONES

                d_i, f_i = args.ik_init.rsplit(":", 1)
                zi = np.load(Path(d_i) / f"{f_i.format(c=name)}_motion.npz")
                jn_i = [str(j) for j in zi["g1_joint_names"]]
                assert len(zi["g1_dof"]) == len(Jf), (len(zi["g1_dof"]), len(Jf))
                q0 = torch.from_numpy(np.asarray(zi["g1_dof"], np.float64)[:, [jn_i.index(n) for n in joints]]).float().cuda()
                q0 = torch.minimum(torch.maximum(q0, kin.lower), kin.upper)
                R0 = torch.from_numpy(Rotation.from_quat(np.asarray(zi["g1_root_quat_xyzw"], np.float64)).as_matrix()).float().cuda()
                t_init = np.asarray(zi["g1_root_pos"], np.float64)
                hum_len = torch.stack([(Jf[:, a] - Jf[:, b]).norm(dim=-1) for a, b in BONES], -1)
                s_ik = (kin.rest_lengths().sum() / hum_len.sum(-1).clamp_min(1e-3)).median()
                root_z0 = t_init[:, 2]
                if json.loads(args.motion_cfg or "{}").get("init_root_z", "init") == "lowest":
                    # our root height on their pose: the G1's lowest point at the human's (scaled) elevation
                    Q0 = Rotation.from_matrix(R0.cpu().numpy().astype(np.float64)).as_quat()
                    low0 = g1_lowest(args.mjcf, joints, Q0[:, [3, 0, 1, 2]], q0.cpu().numpy().astype(np.float64))
                    root_z0 = gaussian_filter1d(float(s_ik) * elev - low0, 1.0, mode="nearest")
                    t_init = t_init.copy()
                    t_init[:, 2] = root_z0
            else:
                q0, R0, s_ik, _ = retarget_sequence(kin, Jf, flex_init=args.ik_flex_init)
                Q0 = Rotation.from_matrix(R0.cpu().numpy().astype(np.float64)).as_quat()
                low0 = g1_lowest(args.mjcf, joints, Q0[:, [3, 0, 1, 2]], q0.cpu().numpy().astype(np.float64))
                root_z0 = gaussian_filter1d(float(s_ik) * elev - low0, 1.0, mode="nearest")
            mcfg = json.loads(args.motion_cfg or "{}")
            mcfg.pop("init_root_z", None)
            ground_h = float(mcfg.pop("ground_h", 0.02))  # the human counts as on the floor below this elevation
            Vf = smpl_world(Q, P_flat)
            labels = human_contact_labels(Vf, body.weights.argmax(1).cpu().numpy(), fps,
                                          mcfg.pop("h_contact", 0.03), mcfg.pop("v_contact", 0.5))
            surf, surf_h, s_world = None, None, s if t_init is not None else float(s_ik)
            pc, pred, scene_info = None, None, {}
            if args.contact_pred:  # per-frame probabilities of another model (frame, prob (N, 8)): the input frames
                zc = np.load(args.contact_pred)
                row = {int(f_): i for i, f_ in enumerate(zc["frame"])}
                pc = np.array([zc["prob"][row[int(f_)]] if int(f_) in row else np.full(zc["prob"].shape[1], np.nan)
                               for f_ in raw["frames"]])
                if pc.shape[1] == 16:  # split head (c4 / c5): static feet for the lock, any-contact hands / knees
                    pc = pc[:, [0, 1, 2, 3, 12, 13, 14, 15]]
            elif args.contact_model and "contact" in raw and np.isfinite(raw["contact"]).any():
                pc = raw["contact"].astype(np.float64)  # the FAB checkpoint's contact head
            if pc is not None and not (np.isfinite(pc).all(1) & valid).any():
                pc = None
            if pc is not None:  # explicit contacts instead of / with the mesh rule
                ok = np.isfinite(pc).all(1) & valid
                pc = resample(fill_invalid(np.nan_to_num(pc), ok), args.fps, fps)[:T] >= 0.5
                pred = {"lfoot": pc[:, 0] | pc[:, 1], "rfoot": pc[:, 2] | pc[:, 3], "lhand": pc[:, 4], "rhand": pc[:, 5],
                        "lknee": pc[:, 6], "rknee": pc[:, 7]}
                if pc.shape[1] >= 16 and not args.contact_static:  # split head (static, then any contact): feet lock
                    pred.update(lhand=pc[:, 12], rhand=pc[:, 13], lknee=pc[:, 14], rknee=pc[:, 15])  # on static, hands / knees on any
                if args.contact_runs:  # the stance pin's run definition (g1_contact_check.stance_runs): gaps < 0.1 s filled,
                    # runs < 0.2 s dropped, so the solver and the pin hold the same contacts
                    gap, mrun = int(round(0.1 * fps)), int(round(0.2 * fps))
                    for g_, m_ in pred.items():
                        m_ = np.asarray(m_, bool).copy()
                        e_ = np.flatnonzero(np.diff(np.r_[0, m_.astype(int), 0]))
                        for b_, f_ in zip(e_[2::2], e_[1:-1:2]):
                            if b_ - f_ < gap:
                                m_[f_:b_] = True
                        e_ = np.flatnonzero(np.diff(np.r_[0, m_.astype(int), 0]))
                        for b_, f_ in zip(e_[::2], e_[1::2]):
                            if f_ - b_ < mrun:
                                m_[b_:f_] = False
                        pred[g_] = m_
            if scene is not None:  # scene pieces -> the flat reference world (human scale): A, start xy, floor
                from fab.retarget.contact import HUMAN_PARTS

                pieces = [np.asarray(v, np.float64) @ A.T - np.array([P[0, 0], P[0, 1], float(np.median(floor))])
                          for v in scene["pieces"]]
                part_ = body.weights.argmax(1).cpu().numpy()
                if args.scene_register and pred is not None:  # the oracle scene onto the estimated person (xy shift)
                    sh, n_reg = scene_register(pieces, Vf, part_, pred, HUMAN_PARTS)
                    pieces = [v + np.array([sh[0], sh[1], 0.0]) for v in pieces]
                    scene_info = {"scene_register_xy_m": sh.tolist(), "scene_register_frames": n_reg,
                                  "scene_register_rich": (A.T @ np.array([sh[0], sh[1], 0.0])).tolist()}
                surf = scene_surface(pieces, args.scene_radius)
                surf_h = {g_: surf(Vf[:, np.flatnonzero(np.isin(part_, js)), :2].mean(1)) for g_, js in HUMAN_PARTS.items()}
                np.savez(out / f"{name}{args.ik_suffix}_scene_ref.npz", pieces=np.array([v * s_world for v in pieces],
                         dtype=object), scale=s_world, note="reference world (G1 scale): x_ref = s * (A x - [xy0, floor])"
                         + (f"; registered by {scene_info['scene_register_xy_m']} m (human scale)" if scene_info else ""))
            if pred is not None:
                hh = human_support_heights(Vf, body.weights.argmax(1).cpu().numpy())
                if surf is not None:  # height above the scene surface under the limb, not above the floor
                    hh = {g_: hh[g_] - surf_h[g_] for g_ in hh}
                for g_ in labels:
                    if args.contact_mode == "limbs":  # any contact is with any surface (the body too): surface-near only
                        if g_ not in ("lfoot", "rfoot"):
                            labels[g_] = pred[g_] & (hh[g_] < args.scene_gate)
                    elif args.contact_mode == "replace":
                        labels[g_] = pred[g_]
                    elif args.contact_mode == "and":
                        labels[g_] = labels[g_] & pred[g_]
                    else:
                        labels[g_] = labels[g_] | pred[g_]
            hold_xy = None
            if surf is not None and args.scene_hold_xy:
                hold_xy = {g_: s_world * v for g_, v in support_targets(pieces, Vf, part_, labels, HUMAN_PARTS).items()}
                scene_info["scene_hold_xy_frames"] = {g_: int(np.isfinite(v[:, 0]).sum()) for g_, v in hold_xy.items()}
            if args.export_human:  # the same input for bench.videomimic.scene_retarget (+ bench.stance_pin), flat floor z = 0
                ex = Path(args.export_human)
                ex.mkdir(parents=True, exist_ok=True)
                Jn = Jf.cpu().numpy().astype(np.float64)
                lo_, hi_ = Jn[:, 0, :2].min(0) - 3.0, Jn[:, 0, :2].max(0) + 3.0
                nxy = np.ceil((hi_ - lo_) / 0.02).astype(int) + 1
                np.savez(ex / f"{name}_hfield.npz", x0=lo_[0], y0=lo_[1], res=0.02, height=np.zeros((nxy[1], nxy[0])))
                xpts = {}
                if args.export_pts:  # each frame's dense points onto the reference SMPL at their vertex ids (similarity), the
                    # per-point residual smoothed (--pts_sigma input frames, the world body's rotation smoothing) and resampled
                    from fab.retarget.dense_inputs import umeyama

                    pidx = np.asarray(raw["surface_idx"], int)
                    Vi = smpl_world(q_smpl, pelvis_w)[:, pidx]
                    Xp = np.asarray(raw["pts"], np.float64)
                    dres, sc_ = np.zeros_like(Vi), np.ones(n)
                    for i_ in range(n):
                        sc_[i_], Rr_, tr_ = umeyama(Xp[i_], Vi[i_])
                        dres[i_] = sc_[i_] * Xp[i_] @ Rr_.T + tr_ - Vi[i_]
                    dres = fill_invalid(dres, valid)
                    if args.pts_sigma > 0:
                        dres = gaussian_filter1d(dres, args.pts_sigma, axis=0, mode="nearest")
                    d50 = resample(dres, args.fps, fps)[:T]
                    rn = np.linalg.norm(dres[valid], axis=-1)
                    xpts = dict(pts=(np.asarray(Vf, np.float64)[:, pidx] + d50).astype(np.float32), pts_idx=pidx,
                                pts_info=json.dumps({"residual_mm_median": float(np.median(rn) * 1000),
                                                     "residual_mm_p95": float(np.percentile(rn, 95) * 1000),
                                                     "sim_scale_median": float(np.median(sc_[valid])), "pts_sigma": args.pts_sigma}))
                np.savez(ex / f"{name}.npz", joints=Jn.astype(np.float32), verts=np.asarray(Vf, np.float32), fps=float(fps),
                         frames=np.arange(T), src_frames=raw["frames"], src_fps=args.fps,
                         contacts=json.dumps({g_: np.asarray(v, bool).astype(int).tolist() for g_, v in labels.items()}),
                         contacts_source=(f"head >= 0.5 ({args.contact_mode}{', static channels' if args.contact_static else ''})"
                                          if pred is not None else "mesh rule"), **xpts)
            urdf = g1_files.urdf()
            q, R, tr, _, mrep = retarget_motion(kin, Jf, fps, labels, support_points(args.mjcf), root_z0, urdf,
                                                init=(q0, R0, s_ik), cfg=mcfg, t_init=t_init,
                                                grounded=torch.as_tensor(elev < ground_h, device="cuda"),
                                                heights=human_support_heights(Vf, body.weights.argmax(1).cpu().numpy()),
                                                speeds=human_support_speeds(Vf, body.weights.argmax(1).cpu().numpy(), fps),
                                                contact_z={g_: s_world * h_ for g_, h_ in surf_h.items()} if surf_h else None,
                                                contact_xy=hold_xy,
                                                root_free=(np.any([labels[g_] for g_ in ("lhand", "rhand", "lknee", "rknee")
                                                                   if g_ in labels], 0) if args.root_free_limbs else None))
            q_ik = q.cpu().numpy().astype(np.float64)
            Q_ik = Rotation.from_matrix(R.cpu().numpy().astype(np.float64)).as_quat()
            root_ik = tr.cpu().numpy().astype(np.float64)
            np.savez(out / f"{name}{args.ik_suffix}_motion.npz", fps=fps, smpl_pose_aa=pose_aa.astype(np.float32),
                     smpl_transl=transl.astype(np.float32), smpl_betas=betas, smpl_pelvis_zup=P_flat.astype(np.float32),
                     g1_dof=q_ik.astype(np.float32), g1_joint_names=np.array(joints), g1_root_pos=root_ik.astype(np.float32),
                     g1_root_quat_xyzw=Q_ik.astype(np.float32), human_elev=elev.astype(np.float32), scale=float(s_ik),
                     frames=raw["frames"], input_fps=args.fps)
            ik_info = {"ik_mode": "motion", "ik_residual_mm": mrep["residual_mm"], "motion_terms": mrep, **scene_info,
                       "contact_fraction": {g: float(np.mean(v)) for g, v in labels.items()},
                       "motion_cfg": json.loads(args.motion_cfg or "{}"), "ik_init": args.ik_init,
                       "ik_flex_init": args.ik_flex_init,
                       "ik_vs_head_dof_deg": float(np.degrees(np.abs(q_ik - G).mean()))}
        elif args.ik_seq:  # temporally consistent IK of the whole clip (no extra smoothing)
            q, R, _, r = retarget_sequence(kin, Jw, flex_init=args.ik_flex_init)
            qs, Rs, res, sig = [q.cpu().numpy()], [R.cpu().numpy()], [r.cpu().numpy()], 0.0
        else:
            for i in range(0, T, 512):
                q, R, _, r = retarget_smpl(kin, Jw[i : i + 512], iters=args.ik_iters)
                qs.append(q.cpu().numpy())
                Rs.append(R.cpu().numpy())
                res.append(r.cpu().numpy())
            sig = args.sigma_ik
        if not args.ik_motion:
            q_ik = gaussian_filter1d(np.concatenate(qs).astype(np.float64), sig, axis=0, mode="nearest") if sig > 0 \
                else np.concatenate(qs).astype(np.float64)
            Q_ik = smooth_quats(Rotation.from_matrix(np.concatenate(Rs).astype(np.float64)).as_quat(), sig)
            low_ik = g1_lowest(args.mjcf, joints, Q_ik[:, [3, 0, 1, 2]], q_ik)
            root_ik = np.concatenate([s * P_flat[:, :2], (s * elev - low_ik)[:, None]], 1)
            root_ik[:, 2] = gaussian_filter1d(root_ik[:, 2], 1.0, mode="nearest")
            np.savez(out / f"{name}{args.ik_suffix}_motion.npz", fps=fps, smpl_pose_aa=pose_aa.astype(np.float32),
                     smpl_transl=transl.astype(np.float32), smpl_betas=betas, smpl_pelvis_zup=P_flat.astype(np.float32),
                     g1_dof=q_ik.astype(np.float32), g1_joint_names=np.array(joints), g1_root_pos=root_ik.astype(np.float32),
                     g1_root_quat_xyzw=Q_ik.astype(np.float32), human_elev=elev.astype(np.float32), scale=s,
                     frames=raw["frames"], input_fps=args.fps)
            ik_info = {"ik_mode": "sequence" if args.ik_seq else "per-frame",
                       "ik_residual_mm": float(np.concatenate(res).mean() * 1000),
                       "ik_vs_head_dof_deg": float(np.degrees(np.abs(q_ik - G).mean()))}

    # input panel: the smoothed reference mesh (input rate) projected back into the camera
    if not args.no_input_video:
        raise NotImplementedError("rendering is not part of this release")

    V30 = smpl_world(q_smpl, pelvis_w)
    Vc = np.einsum("nij,nvj->nvi", ext[:, :3, :3], np.einsum("ji,nvj->nvi", A, V30)) + ext[:, None, :3, 3]
    faces = body.faces.cpu().numpy()
    frames = []
    for t, path in enumerate([] if args.no_input_video else images):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        sc = args.height / img.shape[0]
        img = cv2.resize(img, (int(round(img.shape[1] * sc)), args.height), interpolation=cv2.INTER_AREA)
        if args.input_blank or args.dataset in ("rich", "moyo"):  # licence: no RICH / MOYO pixels, the mesh only
            img = np.full_like(img, 48)
        Ks = K.copy()
        Ks[:2] *= sc
        col = tuple(int(round(255 * float(x))) for x in args.input_rgb.split(",")) if args.input_rgb else (110, 200, 120)
        col = col if valid[t] else (200, 120, 110)
        frames.append(render(img, Vc[t], faces, Ks, col, alpha=args.input_alpha, shade_min=args.input_shade,
                             edge=tuple(int(x * args.input_edge) for x in col) if args.input_edge > 0 else None))
    if frames:
        write_video(frames, out / f"{name}_input.mp4", args.fps)

    acc = lambda x: float(np.linalg.norm(np.diff(x, 2, axis=0), axis=-1).mean() * args.fps ** 2)  # noqa: E731
    info = {"name": name, "frames": [int(sel[0]), int(sel[-1])], "n_input": n, "n_50fps": T,
            "valid_fraction": float(valid.mean()),
            "fit_vs_dense_mesh_mm": float(raw["fit_vs_dense_mm"][valid].mean()) if "fit_vs_dense_mm" in raw else None,
            "dataset_up": up_raw.tolist(), "dataset_up_snap_deg": float(up_ang), "scale_g1_over_human": float(s),
            "human_pelvis_height": float(human_pelvis_h), "floor_range_m": float(np.ptp(floor)),
            "pelvis_acc_raw_m_s2": acc(np.einsum("ij,nj->ni", A, np.einsum("nij,nj->ni", R_wc, pelvis_c) + t_wc)),
            "pelvis_acc_smoothed_m_s2": acc(pelvis_w), "g1_root_z_range": [float(g1_root[:, 2].min()),
                                                                        float(g1_root[:, 2].max())],
            "fit_iters": int(raw.get("fit_iters", -1)), "camera": "estimated: " + args.world if args.world else "dataset (oracle)",
            # dataset world -> reference world (G1 root path, 50 fps frame t): x_ref = s * (A @ x - [xy0, floor_t])
            "world_to_ref": {"A": A.tolist(), "xy0": P[0, :2].tolist(), "floor_z_50fps": floor.tolist(), "scale": float(s),
                             "body_scale": float(body_scale)},
            **ik_info}
    (out / f"{name}_info.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info), flush=True)


if __name__ == "__main__":
    main()
