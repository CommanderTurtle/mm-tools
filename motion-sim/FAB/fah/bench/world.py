"""World-frame hand evaluation over video clips (W-MPJPE / WA-MPJPE / RTE on true-positive hand tracks; WHAM's
protocol by default).

Camera-frame hands from bench.run pickles are mapped to a world frame with per-frame camera poses
T_world_cam (4x4, camera -> world, camera = the loader's upright pinhole): predicted (SLAM, feed-forward or
in-model camera, any world origin, metric scale) or GT ("gt" = oracle upper bound for the hand model).
GT hands always use the GT poses (pickle field T_world_cam, else recovered from the HOT3D clip tars and cached).

Track = (clip, GT hand side), or (clip, segment, side) with --segment N: the frames where that GT hand has a
true-positive prediction (box IoU >= 0.3, handedness not required, finite joints) and a camera pose; tracks with < 3
frames are dropped. --common restricts every run to the hand-frames that are true positives in all listed runs (paper:
the intersection of the compared methods).
--protocol wham (default, the paper's): WHAM's definitions (github.com/yohanshin/WHAM lib/eval/evaluate_emdb.py,
eval_utils.py; alignment from SLAHMR) with HaWoR's hand conventions (github.com/ThunderVVV/HaWoR
scripts/scripts_eval/eval_hawor_hot3d.py): each track is split at missing frames into contiguous runs, runs < --min_run
(16) frames dropped; every run is cut into --chunk (100) frame chunks from its start, the shorter remainder kept; per
chunk, Sim(3) (Umeyama, with scale) of predicted to GT world joints:
  W-MPJPE100  (mm): fit on the 21 joints of the chunk's first two frames, applied to the chunk
  WA-MPJPE100 (mm): fit on all joints of the chunk
  RTE (%)         : per run, rigid (no scale) fit of the predicted to the GT wrist trajectory, per-frame wrist error /
                    GT wrist path length (sum of frame-to-frame displacements) x 100, runs moving < 1 mm excluded
all three averaged over frames (both hands); *_bridge: the same with the missing frames skipped instead of split
(WHAM's EMDB masks), *_right / *_left: one hand side only (HaWoR evaluates the right hand). --chunk 0: whole tracks
(one chunk per run: W-MPJPE on the run's first two frames, WA-MPJPE on the whole run), the paper's second protocol.
--protocol rigid (legacy, not in the paper; gaps allowed): per track, one rigid (R, t, no scale) alignment of predicted
to GT world joints:
  W-MPJPE  (mm): fit on the 21 joints of the first evaluated frame, applied to all frames
  WA-MPJPE (mm): fit on all joints of all evaluated frames
  RTE (%)      : mean wrist error after the WA fit / max GT wrist displacement from the first frame x 100,
                 tracks moving < 1 mm excluded
W-/WA-MPJPE average over all evaluated frames; RTE over tracks (frame-weighted variant also stored).
Camera metrics (when predicted poses are given): ATE of camera centres after rigid / Sim(3) alignment,
path-length scale ratio, rotation drift after aligning the first frame's orientation.
--cams: gt (oracle) | none (static camera) | pickle:<frame field> | dir of <clip>.npz with index (N,) and T_world_cam (N, 4, 4) (bench/world_slam.py) |
<dir>@<key>: same, translation scaled by the npz scalar <key> (bench/world_dpvo.py scale candidates);
a trailing "@gtscale" rescales any source with the per-clip GT-optimal scale (diagnostic); "<dir>#<key>[@...]" reads
the poses from npz key <key> instead of T_world_cam (bench/world_dpvo.py --causal latency variants).

    python -m bench.world <pred_dir> [<pred_dir> ...] --cams gt,<slam dir>,... [--common d1,d2,...] [--summary x.md]
-> $RESULTS/world/eval/<run>/world_<cams name>[_common][_wham100 | _wham0].json
"""
import argparse
import glob
import json
import os
import pickle
import tarfile
from collections import defaultdict

import numpy as np

from bench.common import match_hands

DATA_ROOT = os.environ.get("DATA_ROOT", "data")
RESULTS = os.environ.get("RESULTS", "output")
ROT_CW = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # = bench.datasets.hot3d.ROT_CW


def rigid_fit(src, dst):
    """R, t minimising |R src + t - dst| (Kabsch, no scale); src, dst (N, 3)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((src - mu_s).T @ (dst - mu_d))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    return R, mu_d - R @ mu_s


def sim_fit(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    p, g = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(p.T @ g)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = (U @ D @ Vt).T
    s = (S * np.diag(D)).sum() / max((p**2).sum(), 1e-12)
    return s, R, mu_d - s * R @ mu_s


def to_world(T, X):
    return X @ T[:3, :3].T + T[:3, 3]


# ---------------------------------------------------------------------------- camera poses
def hot3d_gt_cams(tar_path, stream="214-1"):
    """{frame key: T_world_cam} with bench.datasets.hot3d's convention (upright pinhole for Aria)."""
    from hand_tracking_toolkit import camera as htt_camera

    tar = tarfile.open(tar_path)
    keys = sorted({n.split(".")[0] for n in tar.getnames() if n[:6].isdigit()})
    rot = json.load(tar.extractfile(f"{keys[0]}.info.json"))["device"] == "Aria"
    R = ROT_CW if rot else np.eye(3)
    out = {}
    for key in keys:
        cam = htt_camera.from_json(json.load(tar.extractfile(f"{key}.cameras.json"))[stream])
        Tcw = np.linalg.inv(cam.T_world_from_eye)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R @ Tcw[:3, :3], R @ Tcw[:3, 3]
        out[key] = np.linalg.inv(T)
    return out


def gt_cams_for_clip(clip, data_root, cache_dir):
    path = os.path.join(cache_dir, f"{clip['clip']}.npz")
    try:
        d = np.load(path)
        return dict(zip([str(k) for k in d["keys"]], d["T_world_cam"]))
    except (OSError, EOFError, ValueError, KeyError):  # missing or partially written
        pass
    tars = glob.glob(os.path.join(data_root, "**", f"{clip['clip']}.tar"), recursive=True)
    assert tars, f"no tar for {clip['clip']} under {data_root} (needed for GT camera poses)"
    cams = hot3d_gt_cams(tars[0])
    os.makedirs(cache_dir, exist_ok=True)
    keys = sorted(cams)
    tmp = f"{path}.{os.getpid()}.npz"
    np.savez(tmp, keys=np.array(keys), T_world_cam=np.stack([cams[k] for k in keys]))
    os.replace(tmp, path)
    return cams


def load_pred_cams(spec, clip):
    """{frame index: T_world_cam} or None (= GT) for one clip."""
    if spec == "gt":
        return None
    if spec == "none":  # static camera: world = camera frame
        return {fr["index"]: np.eye(4) for fr in clip["frames"]}
    if spec.startswith("pickle:"):
        k = spec.split(":", 1)[1]
        return {fr["index"]: fr[k] for fr in clip["frames"] if fr.get(k) is not None}
    spec, key = spec.split("@") if "@" in spec else (spec, None)
    spec, tkey = spec.split("#") if "#" in spec else (spec, "T_world_cam")
    path = os.path.join(spec, f"{clip['clip']}.npz")
    if not os.path.exists(path):
        return {}
    d = np.load(path)
    T = (d[tkey] if tkey in d else d["T_world_cam"]).copy()  # e.g. static clips without a scale estimate
    if key is not None:  # "<dir>@<scale key>": metric scale stored separately (bench/world_dpvo.py)
        s = float(d[key])
        T[:, :3, 3] *= s if np.isfinite(s) and s > 0 else 1.0
    return {int(i): T for i, T in zip(d["index"], T) if np.isfinite(T).all()}


def camera_metrics(T_pred, T_gt):
    """Camera-centre ATE (m) after rigid / Sim(3) alignment, path-length ratio, rotation error (deg)."""
    cp, cg = np.stack([T[:3, 3] for T in T_pred]), np.stack([T[:3, 3] for T in T_gt])
    R, t = rigid_fit(cp, cg)
    ate = float(np.sqrt((np.linalg.norm(cp @ R.T + t - cg, axis=1) ** 2).mean()))
    s, Rs, ts = sim_fit(cp, cg)
    ate_sim = float(np.sqrt((np.linalg.norm(s * cp @ Rs.T + ts - cg, axis=1) ** 2).mean()))
    lp, lg = np.linalg.norm(np.diff(cp, axis=0), axis=1).sum(), np.linalg.norm(np.diff(cg, axis=0), axis=1).sum()
    # rotation: align the first-frame orientation, then measure drift
    R0 = T_gt[0][:3, :3] @ T_pred[0][:3, :3].T
    rot = [np.degrees(np.arccos(np.clip((np.trace((R0 @ a[:3, :3]).T @ b[:3, :3]) - 1) / 2, -1, 1))) for a, b in zip(T_pred, T_gt)]
    return dict(ate_rigid=ate, ate_sim3=ate_sim, sim3_scale=float(s), scale_err_pct=float(abs(s - 1) * 100),
                path_ratio=float(lp / max(lg, 1e-9)),
                gt_path=float(lg), rot_err_deg=float(np.mean(rot)))


# ---------------------------------------------------------------------------- tracks
def tp_hands(fr, iou_thresh, key="joints3d"):
    """{GT side: (pred joints, GT joints)} for true-positive hands of one frame."""
    matches, _ = match_hands(fr["pred"], fr["gt"], iou_thresh)
    out = {}
    for gi, pi in matches:
        if pi is None:
            continue
        pj = fr["pred"][pi].get(key)
        if pj is None or not np.isfinite(pj).all():
            continue
        out[bool(fr["gt"][gi]["is_right"])] = (np.asarray(pj, np.float64), np.asarray(fr["gt"][gi]["joints3d"], np.float64))
    return out


def common_support(dirs, iou_thresh, key="joints3d"):
    """Set of (clip, frame index, GT side) true positives in every run of dirs."""
    sets = []
    for d in dirs:
        s = set()
        for path in sorted(glob.glob(os.path.join(d, "*.pkl"))):
            clip = pickle.load(open(path, "rb"))
            for fr in clip["frames"]:
                s.update((clip["clip"], fr["index"], side) for side in tp_hands(fr, iou_thresh, key))
        sets.append(s)
    return set.intersection(*sets) if sets else None


def track_errors(P, G, min_disp=1e-3):
    """P, G (T, 21, 3) world joints (m) -> per-frame W / WA errors (mm), RTE (%) or None."""
    R, t = rigid_fit(P[0], G[0])
    w = np.linalg.norm(P @ R.T + t - G, axis=-1).mean(-1) * 1000
    R, t = rigid_fit(P.reshape(-1, 3), G.reshape(-1, 3))
    Pa = P @ R.T + t
    wa = np.linalg.norm(Pa - G, axis=-1).mean(-1) * 1000
    disp = np.linalg.norm(G[:, 0] - G[0, 0], axis=-1).max()
    rte = float(np.linalg.norm(Pa[:, 0] - G[:, 0], axis=-1).mean() / disp * 100) if disp >= min_disp else None
    return w, wa, rte


def wham_errors(P, G, chunk=100, min_disp=1e-3):
    """P, G (T, 21, 3) world joints (m) of one contiguous run -> per-frame W-MPJPE100 / WA-MPJPE100 (mm), per-frame RTE (%)
    or None (WHAM: first_align_joints / global_align_joints per chunk, compute_rte on the wrist over the run)."""
    w, wa, step = [], [], chunk or len(P)
    for a in range(0, len(P), step):
        p, g = P[a:a + step], G[a:a + step]
        s, R, t = sim_fit(p[:2].reshape(-1, 3), g[:2].reshape(-1, 3))
        w.append(np.linalg.norm(s * p @ R.T + t - g, axis=-1).mean(-1) * 1000)
        s, R, t = sim_fit(p.reshape(-1, 3), g.reshape(-1, 3))
        wa.append(np.linalg.norm(s * p @ R.T + t - g, axis=-1).mean(-1) * 1000)
    R, t = rigid_fit(P[:, 0], G[:, 0])
    path = np.linalg.norm(np.diff(G[:, 0], axis=0), axis=-1).sum()
    rte = np.linalg.norm(P[:, 0] @ R.T + t - G[:, 0], axis=-1) / path * 100 if path >= min_disp else None
    return np.concatenate(w), np.concatenate(wa), rte


def evaluate_world(pred_dir, cams="gt", iou_thresh=0.3, data_root=None, gt_cache=None, common=None, key="joints3d",
                   min_frames=3, segment=0, protocol="wham", chunk=100, min_run=16):
    data_root = data_root or os.path.join(DATA_ROOT, "hot3d")
    gt_cache = gt_cache or os.path.join(RESULTS, "world", "gt_cams")
    w_all, wa_all, rte_tr, rte_w, cam_m = [], [], [], [], defaultdict(list)
    n_tracks, n_short, n_nocam, method = 0, 0, 0, None
    n_runs, n_short_run, n_tp, bridge, by_side = 0, 0, 0, [], defaultdict(list)  # --protocol wham
    files = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    for path in files:
        clip = pickle.load(open(path, "rb"))
        method = clip["method"]
        frames = clip["frames"]
        if any(fr.get("T_world_cam") is None for fr in frames):
            gtc = gt_cams_for_clip(clip, data_root, gt_cache)
            gt_T = {fr["index"]: gtc[fr["key"]] for fr in frames}
        else:
            gt_T = {fr["index"]: np.asarray(fr["T_world_cam"]) for fr in frames}
        pc = load_pred_cams(cams.replace("@gtscale", ""), clip)
        if cams.endswith("@gtscale") and pc:  # diagnostic: per-clip GT-optimal (Sim3) translation scale
            idx = [i for i in sorted(gt_T) if i in pc]
            s, _, _ = sim_fit(np.stack([pc[i][:3, 3] for i in idx]), np.stack([gt_T[i][:3, 3] for i in idx]))
            pc = {i: np.vstack([np.hstack([T[:3, :3], s * T[:3, 3:]]), T[3:]]) for i, T in pc.items()}
        pred_T = gt_T if pc is None else pc
        if pc is not None and cams != "none":
            idx = [i for i in sorted(gt_T) if i in pc]
            if len(idx) >= 2:
                for k, v in camera_metrics([pc[i] for i in idx], [gt_T[i] for i in idx]).items():
                    cam_m[k].append(v)
        tracks, tidx = defaultdict(list), defaultdict(list)
        n_seg = len(frames) // segment if segment else 1
        for fr in frames:
            seg = fr["index"] // segment if segment else 0  # non-overlapping segments, tail dropped
            if seg >= n_seg:
                continue
            for side, (pj, gj) in tp_hands(fr, iou_thresh, key).items():
                if common is not None and (clip["clip"], fr["index"], side) not in common:
                    continue
                if fr["index"] not in pred_T:
                    n_nocam += 1
                    continue
                tracks[side, seg].append((to_world(pred_T[fr["index"]], pj), to_world(gt_T[fr["index"]], gj)))
                tidx[side, seg].append(fr["index"])
        for side, tr in tracks.items():
            if len(tr) < min_frames:
                n_short += 1
                continue
            P, G = np.stack([a for a, _ in tr]), np.stack([b for _, b in tr])
            if protocol == "wham":
                n_tracks, n_tp = n_tracks + 1, n_tp + len(tr)
                idx = np.asarray(tidx[side])
                cuts = np.flatnonzero(np.diff(idx) != 1) + 1  # HaWoR parse_chunks: a missing frame ends the run
                for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(idx)]):
                    if b - a < min_run:
                        n_short_run += 1
                        continue
                    w, wa, rte = wham_errors(P[a:b], G[a:b], chunk)
                    w_all.append(w)
                    wa_all.append(wa)
                    by_side[side[0]].append((w, wa, rte))
                    n_runs += 1
                    if rte is not None:
                        rte_tr.append(float(rte.mean()))
                        rte_w.append((float(rte.mean()), b - a))
                bridge.append(wham_errors(P, G, chunk))
                continue
            w, wa, rte = track_errors(P, G)
            w_all.append(w)
            wa_all.append(wa)
            n_tracks += 1
            if rte is not None:
                rte_tr.append(rte)
                rte_w.append((rte, len(tr)))
    w_all = np.concatenate(w_all) if w_all else np.zeros(0)
    wa_all = np.concatenate(wa_all) if wa_all else np.zeros(0)
    res = dict(
        method=method, cams=cams, common=common is not None, segment=segment, num_clips=len(files), num_tracks=n_tracks,
        num_frames=int(len(w_all)), num_short_tracks=n_short, num_hand_frames_without_cam=n_nocam,
        w_mpjpe=float(w_all.mean()) if len(w_all) else None, wa_mpjpe=float(wa_all.mean()) if len(wa_all) else None,
        rte=float(np.mean(rte_tr)) if rte_tr else None, num_rte_tracks=len(rte_tr),
        rte_frame_weighted=float(sum(r * n for r, n in rte_w) / sum(n for _, n in rte_w)) if rte_w else None,
        w_mpjpe_median=float(np.median(w_all)) if len(w_all) else None,
    )
    res.update({f"cam_{k}": float(np.mean(v)) for k, v in cam_m.items()})
    res.update({f"cam_{k}_median": float(np.median(v)) for k, v in cam_m.items()})
    if cam_m:
        res["cam_num_clips"] = len(cam_m["ate_rigid"])
    if protocol == "wham":  # rte = WHAM's mean over frames; the mean over runs is kept as rte_run_mean
        res.update(protocol="wham", chunk=chunk, min_run=min_run, num_runs=n_runs, num_short_runs=n_short_run,
                   num_tp_frames=n_tp, rte=res["rte_frame_weighted"], rte_run_mean=res["rte"])
        for name, v in (("bridge", bridge), ("right", by_side[True]), ("left", by_side[False])):
            for i, k in enumerate(("w_mpjpe", "wa_mpjpe", "rte")):
                x = [e[i] for e in v if e[i] is not None]
                res[f"{k}_{name}"] = float(np.concatenate(x).mean()) if x else None
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("pred_dirs", nargs="+")
    p.add_argument("--cams", default="gt", help="comma-separated: gt | none | pickle:<frame field> | dir of <clip>.npz")
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--common", default="", help="comma-separated run dirs; evaluate their common true-positive hand-frames")
    p.add_argument("--data_root", default=None, help="HOT3D clip tars (GT poses for pickles without T_world_cam)")
    p.add_argument("--gt_cache", default=None)
    p.add_argument("--key", default="joints3d")
    p.add_argument("--name", default="", help="output world_<name>.json (default from --cams)")
    p.add_argument("--out_root", default=os.path.join(RESULTS, "world", "eval"))
    p.add_argument("--summary", default="", help="also append a markdown table of all results to this file")
    p.add_argument("--segment", type=int, default=0, help="score non-overlapping N-frame segments of each clip as separate "
                   "tracks (tail < N dropped); 0 = whole clip")
    p.add_argument("--protocol", default="wham", choices=["wham", "rigid"], help="wham (default, the paper's): WHAM's "
                   "W-MPJPE100 / WA-MPJPE100 / RTE with HaWoR's runs (output world_<name>_wham<chunk>); rigid: one rigid "
                   "alignment per track (legacy)")
    p.add_argument("--chunk", type=int, default=100,
                   help="--protocol wham: chunk length (100: the paper's chunks; 0: whole tracks)")
    p.add_argument("--min_run", type=int, default=16, help="--protocol wham: shortest contiguous run scored (HaWoR: 16)")
    p.add_argument("--export", default="", help="only write the (first) --cams poses of pred_dirs[0] as <dir>/<clip>.npz")
    a = p.parse_args()
    if a.export:  # e.g. a method's own camera (pickle:<field>), to pair it with other methods' hands
        os.makedirs(a.export, exist_ok=True)
        for path in sorted(glob.glob(os.path.join(a.pred_dirs[0], "*.pkl"))):
            clip = pickle.load(open(path, "rb"))
            pc = load_pred_cams(a.cams.split(",")[0], clip)
            idx = sorted(pc)
            np.savez(os.path.join(a.export, f"{clip['clip']}.npz"), index=np.array(idx), T_world_cam=np.stack([pc[i] for i in idx]))
        raise SystemExit
    common = common_support([d for d in a.common.split(",") if d], a.iou, a.key) if a.common else None
    rows = []
    for d in a.pred_dirs:
        run = os.path.basename(os.path.normpath(d))
        for cams in a.cams.split(","):
            res = evaluate_world(d, cams, a.iou, a.data_root, a.gt_cache, common, a.key, segment=a.segment,
                                 protocol=a.protocol, chunk=a.chunk, min_run=a.min_run)
            name = a.name or os.path.basename(os.path.normpath(cams.replace("pickle:", ""))).replace("@", "_").replace("#", "_")
            name += "_common" if common is not None and not a.name else ""
            name += f"_seg{a.segment}" if a.segment and not a.name else ""
            name += f"_wham{a.chunk}" if a.protocol == "wham" and not a.name else ""
            os.makedirs(os.path.join(a.out_root, run), exist_ok=True)
            with open(os.path.join(a.out_root, run, f"world_{name}.json"), "w") as f:
                json.dump(res, f, indent=1)
            print(run, json.dumps(res), flush=True)
            rows.append((run, name, res))
    if a.summary:
        fmt = lambda v, k=1: "-" if v is None else f"{v:.{k}f}"
        with open(a.summary, "a") as f:
            f.write("| run | cams | tracks | frames | W-MPJPE | WA-MPJPE | RTE % | cam ATE-rigid mm | cam Sim3 scale | cam scale err % (median) | cam rot deg |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|\n")
            for run, name, r in rows:
                ate = r.get("cam_ate_rigid")
                f.write(f"| {run} | {name} | {r['num_tracks']} | {r['num_frames']} | {fmt(r['w_mpjpe'])} | {fmt(r['wa_mpjpe'])} | "
                        f"{fmt(r['rte'])} | {fmt(ate * 1000 if ate is not None else None)} | {fmt(r.get('cam_sim3_scale'), 3)} | "
                        f"{fmt(r.get('cam_scale_err_pct_median'))} | {fmt(r.get('cam_rot_err_deg'), 2)} |\n")
            f.write("\n")
