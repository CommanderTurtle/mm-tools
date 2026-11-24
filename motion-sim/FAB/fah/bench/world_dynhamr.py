"""World-frame Dyn-HaMR (CVPR 2025, https://github.com/ZhengdiYu/Dyn-HaMR) on benchmark clips: the runs behind the
supplement's Dyn-HaMR paragraph. Dyn-HaMR is not part of this repository: scripts/dynhamr/setup.sh fetches it (and its
dependencies) at a pinned commit into $THIRDPARTY and builds its own environment; this driver only feeds it frames and
cameras and reads its results back.

Runs the released pipeline (run_opt.py: its HaMeR fork -> hand tracks, root fit, smooth fit; the HMP motion-prior stage
III only with --run_prior 1, it asserts exactly 128-frame track spans) on the loader frames (upright pinhole, q95 JPEGs,
on a local --work disk: it rewrites a 12 MB optimiser state every step). Cameras are given to it as its cameras.npz
(true intrinsics + --cams poses, e.g. the bench/world_slam.py run of the WiLoR + SLAM row), so no SLAM runs inside
Dyn-HaMR; its world-scale optimisation stays on unless --overrides model.opt_scale=False (the paper's setting).
Resumable per clip and --exp_name. --vipe <ViPE results dir> instead gives it ViPE poses + ViPE intrinsics, as its
README's ViPE path (data=video_vipe).
-> <out>/<clip>.pkl in bench.run format: Dyn-HaMR's world hands (its MANO, 21 joints) mapped to the camera frame
   through its own optimised cameras; <out>/cams/<clip>.npz (index, T_world_cam = those cameras) so that
   `bench.world <out> --cams <out>/cams` scores its world output exactly; <work>/<clip>/ keeps inputs, logs, timings.
    DYNHAMR_ROOT=$THIRDPARTY/Dyn-HaMR-camfix <dynhamr env>/bin/python -m bench.world_dynhamr --dataset hot3d \
        --split bench/splits/hot3d_bench.json --data_root $DATA_ROOT/hot3d --mano_dir <numpy-1 MANO dir> \
        --cams $RESULTS/world/hot3d/slam_hawor_r1 --overrides model.opt_scale=False --work <local dir> \
        --out $RESULTS/hot3d/dynhamr --gpu 0
(scripts/dynhamr/run.sh shards it over GPUs; docs/world.md "Dyn-HaMR".)
"""
import argparse
import glob
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np

from bench.run import iter_clips

DYN = os.environ.get("DYNHAMR_ROOT") or os.path.join(os.environ.get("THIRDPARTY", "thirdparty"), "Dyn-HaMR")  # scripts/dynhamr/setup.sh


def load_cams(cams_dir, clip, n):
    """(n, 4, 4) T_world_cam, frames without a pose take the nearest posed frame."""
    d = np.load(os.path.join(cams_dir, f"{clip}.npz"))
    idx, T = np.asarray(d["index"]).astype(int), d["T_world_cam"].astype(np.float64)
    ok = np.isfinite(T).reshape(len(T), -1).all(1)
    idx, T = idx[ok], T[ok]
    near = np.abs(np.arange(n)[:, None] - idx[None]).argmin(1)
    return T[near], int((np.abs(idx[near] - np.arange(n)) > 0).sum())


def load_vipe(vipe_dir, clip, n):
    """ViPE results as Dyn-HaMR's own ViPE path reads them (data/vidproc.py load_vipe_cameras): pose/<clip>.npz c2w,
    intrinsics/<clip>.npz (fx, fy, cx, cy); frames without a pose take the nearest posed frame."""
    pose, intr = (np.load(os.path.join(vipe_dir, k, f"{clip}.npz")) for k in ("pose", "intrinsics"))
    idx = np.asarray(pose["inds"]).astype(int)
    near = np.abs(np.arange(n)[:, None] - idx[None]).argmin(1)
    return pose["data"].astype(np.float64)[near], intr["data"].astype(np.float32)[near], int((idx[near] != np.arange(n)).sum())


def write_inputs(clip, name, root, cams_dir, cam_dir, vipe_dir=""):
    img_dir = os.path.join(root, "images", name)
    os.makedirs(img_dir, exist_ok=True)
    have = len(glob.glob(os.path.join(img_dir, "*.jpg"))) == len(clip)  # another config already wrote them
    frames, Ks = [], []
    for i in range(len(clip)):
        fr = clip.frame(i)
        if not have:
            cv2.imwrite(os.path.join(img_dir, f"{i + 1:06d}.jpg"), fr["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append(dict(key=fr["key"], index=i, K=fr["K"], gt=fr["hands"], ignore=fr.get("ignore", []), pred=[], latency=0.0,
                           image_size=fr["image"].shape[:2], subset=fr.get("subset"),
                           fully_annotated=fr.get("fully_annotated", True), T_world_cam=fr.get("T_world_cam")))
        Ks.append(np.asarray(fr["K"], np.float64))
    H, W = frames[0]["image_size"]
    if vipe_dir:  # its ViPE path: ViPE poses and ViPE intrinsics
        T, intrins, n_fill = load_vipe(vipe_dir, name, len(frames))
    else:  # given poses, true intrinsics
        T, n_fill = load_cams(cams_dir, name, len(frames))
        Ks = np.stack(Ks)
        intrins = np.stack([Ks[:, 0, 0], Ks[:, 1, 1], Ks[:, 0, 2], Ks[:, 1, 2]], 1).astype(np.float32)
    os.makedirs(cam_dir, exist_ok=True)
    np.savez(os.path.join(cam_dir, "cameras.npz"), height=H, width=W, focal=float(intrins[0, :2].mean()), intrins=intrins,
             w2c=np.linalg.inv(T).astype(np.float32))
    return frames, n_fill


def final_result(log_dir, stages=("prior", "smooth_fit")):
    """Last saved stage: prior (HMP) if present, else smooth_fit (stages=("init",): its HaMeR initialiser)."""
    for stage in stages:
        res = sorted(glob.glob(os.path.join(log_dir, stage, "*_world_results.npz")),
                     key=lambda p: int(re.findall(r"_(\d+)_(?:init_)?world_results", p)[-1]))
        if res:
            return stage, res[-1]
    return None, None


def world_hands(res_path, log_dir):
    """Dyn-HaMR's own MANO on its result params -> joints / verts (B, T, ., 3) world, w2c (T, 4, 4), is_right (B, T)."""
    sys.path.insert(0, os.path.join(DYN, "dyn-hamr"))
    import torch
    from body_model import MANO, run_mano

    r = np.load(res_path)
    trans, root, pose = (torch.as_tensor(r[k]).float() for k in ("trans", "root_orient", "pose_body"))
    B, T = trans.shape[:2]
    betas = torch.as_tensor(r["betas"]).float().reshape(B, -1)
    is_right = torch.as_tensor(r["is_right"]).float().reshape(B, -1)
    if is_right.shape[1] != T:
        is_right = is_right[:, :1].expand(B, T)
    mano = MANO(batch_size=B * T, pose2rot=True, model_path=os.path.join(DYN, "_DATA/data/mano"), gender="neutral",
                num_hand_joints=15, mean_params=os.path.join(DYN, "_DATA/data/mano_mean_params.npz"), create_body_pose=False)
    with torch.no_grad():
        out = run_mano(mano, trans, root, pose.reshape(B, T, -1), is_right, betas=betas[:, : mano.num_betas])
    cam_R, cam_t = np.asarray(r["cam_R"], np.float64), np.asarray(r["cam_t"], np.float64)
    cam_R, cam_t = (cam_R[0], cam_t[0]) if cam_R.ndim == 4 else (cam_R, cam_t)
    w2c = np.tile(np.eye(4), (T, 1, 1))
    w2c[:, :3, :3], w2c[:, :3, 3] = cam_R, cam_t
    info = json.load(open(os.path.join(log_dir, "track_info.json")))
    vis = -np.ones((B, T))  # Dyn-HaMR's ternary mask: -1 outside the track span, 0 occluded (infilled), 1 observed
    for tid, t in info["tracks"].items():
        m = np.asarray(t["vis_mask"], bool)[info["meta"]["seq_interval"][0]:][:T]
        on = np.flatnonzero(m)
        if len(on):
            vis[t["index"], on[0]:on[-1] + 1] = m[on[0]:on[-1] + 1]
    intr = np.asarray(r["intrins"], np.float64).reshape(-1, 4)  # the intrinsics it fitted with (ViPE's on the ViPE path)
    intr = np.broadcast_to(intr, (T, 4)) if len(intr) == 1 else intr[:T]
    return (out["joints"].numpy().astype(np.float64), out["vertices"].numpy().astype(np.float64), w2c,
            is_right.numpy() > 0.5, vis, info["meta"]["seq_interval"][0], list(r.keys()), intr)


def fill_preds(frames, joints, verts, w2c, is_right, vis, s0, intr):
    B, T = joints.shape[:2]
    T_world_cam = {}
    for t in range(T):
        fi = s0 + t
        if fi >= len(frames):
            break
        fr = frames[fi]
        K = np.array([[intr[t, 0], 0, intr[t, 2]], [0, intr[t, 1], intr[t, 3]], [0, 0, 1]])
        T_world_cam[fi] = np.linalg.inv(w2c[t])
        for b in range(B):
            if vis[b, t] < 0:  # outside the track span (Dyn-HaMR's -1); 0 = occluded / infilled, 1 = observed
                continue
            j3 = joints[b, t] @ w2c[t, :3, :3].T + w2c[t, :3, 3]
            v3 = verts[b, t] @ w2c[t, :3, :3].T + w2c[t, :3, 3]
            if not (np.isfinite(j3).all() and (v3[:, 2] > 1e-3).all()):
                continue
            uv = v3[:, :2] / v3[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            juv = j3[:, :2] / j3[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            fr["pred"].append(dict(is_right=bool(is_right[b, t]), score=1.0 if vis[b, t] > 0 else 0.5,
                                   bbox=np.r_[uv.min(0), uv.max(0)].astype(np.float32), joints3d=j3.astype(np.float32),
                                   verts3d=v3.astype(np.float32), joints2d=juv.astype(np.float32)))
    return T_world_cam


def stage_times(log_path):
    txt = open(log_path, errors="ignore").read()
    out = {}
    m = re.search(r"frame, hand track, slam camera:\s+([\d.e-]+)\s+([\d.e-]+)\s+([\d.e-]+)", txt)
    if m:
        out.update(frames_s=float(m.group(1)), hamer_tracks_s=float(m.group(2)), cameras_s=float(m.group(3)))
    for k, pat in (("root_s", r"root optimization time:\s+([\d.e-]+)"), ("smooth_s", r"Smooth optimization time:\s+([\d.e-]+)"),
                   ("prior_s", r"prior optimization time:\s+([\d.e-]+)")):
        m = re.findall(pat, txt)
        if m:
            out[k] = float(m[-1])
    return out


def run_cmd(args, py, root, name, log_root, cam_dir, run_opt):
    return [py, "run_opt.py", f"data={args.data_cfg}", f"data.root={root}", f"data.seq={name}", "data.end_idx=-1",
            run_opt, "run_vis=False", f"run_prior={bool(args.run_prior)}", "is_static=False",
            f"log_root={log_root}", f"exp_name={args.exp_name}", "gpu=0", f"fps={args.fps}",
            f"data.sources.cameras={cam_dir}"] + args.overrides.split()


def run_env(args, py, root):
    """Dyn-HaMR's gpu=<i> is cuda:<i> after CUDA was initialised at import (HMP/fitting.py) and is pasted into its HaMeR
    command as CUDA_VISIBLE_DEVICES=<i>: run it with gpu=0 inside CUDA_VISIBLE_DEVICES=<physical>, and give the HaMeR
    subprocess a `python` that restores the physical id."""
    shim = os.path.join(root, "bin")
    os.makedirs(shim, exist_ok=True)
    hamer_py = os.environ.get("DYNHAMR_HAMER_PY") or py  # the 98e1622 HaMeR fork (ViTDet + ViTPose) has its own env
    with open(os.path.join(shim, "python"), "w") as f:
        f.write(f'#!/bin/bash\nexport CUDA_VISIBLE_DEVICES={args.gpu}\nexec {hamer_py} "$@"\n')
    os.chmod(os.path.join(shim, "python"), 0o755)
    return dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONPATH=os.path.join(DYN, "shim"),
                PATH=shim + ":" + os.path.dirname(py) + ":" + os.environ.get("PATH", ""))


def main(args):
    os.makedirs(os.path.join(args.out, "cams"), exist_ok=True)
    py = sys.executable
    for name, make_clip in iter_clips(args):
        out_path = os.path.join(args.out, f"{name}.pkl")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        root = os.path.join(args.work, name)
        t0 = time.perf_counter()
        cam_dir = os.path.join(root, "dynhamr", f"cameras_{args.exp_name}", name, "shot-0")
        frames, n_fill = write_inputs(make_clip(), name, root, args.cams, cam_dir, args.vipe)
        t_in = time.perf_counter() - t0
        log_root = os.path.join(root, "logs")
        log_dir = os.path.join(log_root, "video-custom", args.exp_name, f"{name}-all-shot-0-0--1")
        stages = ("init",) if args.result_stage == "init" else ("prior", "smooth_fit")
        stage, res = final_result(log_dir, stages)
        t1 = time.perf_counter()
        if args.tracks_only:  # frames + HaMeR tracks only (shared by every config of this clip), no optimisation
            if not glob.glob(os.path.join(root, "dynhamr", "track_preds", name, "*")):
                with open(os.path.join(root, "run_opt_tracks.log"), "w") as f:
                    subprocess.call(run_cmd(args, py, root, name, log_root, cam_dir, "run_opt=False"), cwd=os.path.join(DYN, "dyn-hamr"),
                                    env=run_env(args, py, root), stdout=f, stderr=subprocess.STDOUT)
            print(f"[dynhamr] {name}: tracks {len(glob.glob(os.path.join(root, 'dynhamr', 'track_preds', name, '*')))}", flush=True)
            continue
        if res is None or (args.run_prior and stage != "prior"):
            with open(os.path.join(root, f"run_opt_{args.exp_name}.log"), "w") as f:
                rc = subprocess.call(run_cmd(args, py, root, name, log_root, cam_dir, "run_opt=True"),
                                     cwd=os.path.join(DYN, "dyn-hamr"), env=run_env(args, py, root), stdout=f, stderr=subprocess.STDOUT)
            stage, res = final_result(log_dir, stages)
            if rc != 0 or res is None:
                print(f"[dynhamr] {name}: run_opt failed (rc {rc}), see {root}/run_opt_{args.exp_name}.log", flush=True)
                continue
        t_run = time.perf_counter() - t1
        joints, verts, w2c, is_right, vis, s0, keys, intr = world_hands(res, log_dir)
        T_world_cam = fill_preds(frames, joints, verts, w2c, is_right, vis, s0, intr)
        times = dict(stage_times(os.path.join(root, f"run_opt_{args.exp_name}.log")), inputs_s=t_in, run_opt_s=t_run, stage=stage,
                     result=os.path.relpath(res, root), result_keys=keys, cams_filled=n_fill, n_frames=len(frames))
        json.dump(times, open(os.path.join(root, f"timing_{args.exp_name}.json"), "w"), indent=1)
        for fr in frames:
            fr["latency"] = t_run / len(frames)
        idx = np.array(sorted(T_world_cam))
        np.savez(os.path.join(args.out, "cams", f"{name}.npz"), index=idx,
                 T_world_cam=np.stack([T_world_cam[i] for i in idx]) if len(idx) else np.zeros((0, 4, 4)))
        fd, tmp = tempfile.mkstemp(dir=args.out, prefix=f"{name}.pkl.", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(dict(clip=name, method="dynhamr", dataset=args.dataset, frames=frames, stride=1, video=True,
                             timing=times), f)
        os.replace(tmp, out_path)
        n_pred = sum(len(fr["pred"]) for fr in frames)
        print(f"[dynhamr] {name}: {len(frames)} frames, {n_pred} hands, stage {stage}, run_opt {t_run:.0f} s "
              f"({1000 * t_run / len(frames):.0f} ms/frame)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["hot3d", "egostandard"])
    p.add_argument("--split", default="bench/splits/hot3d_bench.json")
    p.add_argument("--data_root", required=True)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--cams", default="", help="dir of <clip>.npz (index, T_world_cam metric, loader camera -> world)")
    p.add_argument("--vipe", default="", help="ViPE results dir (pose/, intrinsics/): its own ViPE camera path instead of --cams")
    p.add_argument("--work", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--run_prior", type=int, default=0, help="HMP stage III (needs exactly 128-frame track spans)")
    p.add_argument("--exp_name", default="dynhamr", help="Dyn-HaMR log sub-dir (one per config)")
    p.add_argument("--overrides", default="", help="extra hydra overrides, e.g. 'model.opt_scale=False'")
    p.add_argument("--tracks_only", action="store_true", help="write frames + run its HaMeR tracking, no optimisation")
    p.add_argument("--data_cfg", default="video_driod", help="its data config (paper-time release 98e1622: video)")
    p.add_argument("--result_stage", default="final", choices=["final", "init"],
                   help="init: export its HaMeR initialiser (<log>/init) instead of the optimised result")
    p.add_argument("--clips", default="", help="comma-separated clip names (smoke tests)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.clips:  # filter first, then shard the kept clips
        keep, _iter = set(a.clips.split(",")), iter_clips

        def iter_clips(args):
            whole = argparse.Namespace(**dict(vars(args), shard=0, num_shards=1))
            kept = [(n, m) for n, m in _iter(whole) if n in keep]
            return kept[args.shard::args.num_shards]
    main(a)
