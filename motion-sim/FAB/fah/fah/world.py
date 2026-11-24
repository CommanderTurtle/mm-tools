"""World-space hands: camera-frame hands of the video model placed with per-frame camera poses T_world_cam (4x4,
camera -> world, metric). The FAH camera (causal DPVO + metric-depth scale + learned camera-motion head, fused) is
produced by bench/world_dpvo.py, bench/world_camhead.py and bench/world_fuse.py (docs/world.md; estimate_cameras runs
them on one clip); any other metric camera trajectory in the same format works too. The offline camera (camera
"offline", estimate_cameras_offline) adds a masked DROID-SLAM trajectory with a Metric3D scale (bench/world_slam.py),
combined with the real-time camera by fuse_offline.

Camera file (.npz, as bench/world.py reads camera sources): T_world_cam (N, 4, 4) and optionally index (N,) = the
video frame of each pose (default 0..N-1).
"""
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")  # torch_scatter (DPVO, DROID-SLAM), mmcv (Metric3D)
# the camera configuration (docs/world.md): DPVO 384 px every 2nd frame, re-initialisation, MoGe-2 scale calls (15, one
# every 30 fed frames, their running median per DPVO instance), fused with the camera-motion head, whose scale correction
# fades out where DPVO sees parallax (--ev_lo / --ev_hi); the head's translation replaces DPVO's where they disagree by
# more than 10x the head's predicted error, 3x on steps where the hand masks leave < 70 % of the frame (--t_sigma*)
DPVO_ARGS = ["--size", "384", "--config", "default", "--stride", "2", "--causal", "--mask_box", "--mask_delay", "1",
             "--scale_at", "10", "--scale_pw", "--reinit_sig", "rot", "--reinit_rot", "14.2", "--reinit_back", "8",
             "--reinit_scale", "2.0", "--signals", "--depth_model", "moge2", "--scale_calls", "15", "--scale_every", "30",
             "--scale_rmed"]
FUSE_ARGS = ["--win", "30", "--w_head", "0.5", "--thr", "0", "--k_sigma", "3", "--bridge", "--ev_lo", "0.5", "--ev_hi", "1.5",
             "--t_sigma", "10", "--t_sigma_low", "3", "--t_sigma_cov", "0.7"]
SCALE_CAL, GATE = 1.0338939640317917, 1.5  # HOT3D-train correction of the MoGe-2 scale calls; translation gate
OFFLINE_ALPHA, OFFLINE_FALLBACK = 0.5, 0.05  # offline camera: SLAM scale moved halfway to ours; fall back to ours above this


def load_cameras(path, key="T_world_cam"):
    d = np.load(path)
    T = np.asarray(d[key], np.float64)
    idx = np.asarray(d["index"]).astype(int) if "index" in d else np.arange(len(T))
    return dict(zip(idx.tolist(), T))


def to_world(T, X):
    return np.asarray(X, np.float64) @ T[:3, :3].T + T[:3, 3]


def place(hands, T):
    """Adds joints3d_world / verts3d_world / verts3d_mhr_world (m) to each hand of one frame."""
    for h in hands:
        h["joints3d_world"] = to_world(T, h["joints3d"])
        if h.get("verts3d") is not None:
            h["verts3d_world"] = to_world(T, h["verts3d"])
        if h.get("verts3d_mhr") is not None:
            h["verts3d_mhr_world"] = to_world(T, h["verts3d_mhr"])
    return hands


def estimate_cameras(work, name, camera_head, dpvo_dir, dpvo_python=sys.executable, depth_python=sys.executable,
                     mano_dir=None, seed=0):
    """FAH camera for one clip: <work>/frames/<name>/{%05d.jpg, K.npy, done} + <work>/hands/<name>.pkl (bench.run-style
    hand predictions: the box masks / hand tokens) -> <work>/cams/<name>.npz (T_world_cam, metric). DPVO runs in
    dpvo_python from its checkout dpvo_dir, the MoGe-2 scale calls in depth_python."""
    from bench.world_camera import calibrate, gate

    f, h = os.path.join(work, "frames"), os.path.join(work, "hands")
    run = lambda cmd, cwd=REPO, path=REPO: subprocess.run(cmd, cwd=cwd, check=True, env=dict(
        os.environ, PYTHONPATH=os.pathsep.join([path] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))))
    mano = ["--mano_dir", mano_dir or os.environ.get("MANO_DIR", "data/mano")]
    run([dpvo_python, os.path.join(REPO, "bench", "world_dpvo.py"), "--frames", f, "--mask_run", h, "--out", os.path.join(work, "dpvo"),
         *DPVO_ARGS, "--depth_py", depth_python, "--seed", str(seed), *mano], cwd=dpvo_dir,
        path=os.pathsep.join([VENDOR, REPO, dpvo_dir]))  # the torch_scatter shim first
    run([sys.executable, "-m", "bench.world_camhead", "--ckpt", camera_head, "--frames", f, "--mask_run", h,
         "--out", os.path.join(work, "head"), "--stride", "2"])
    calibrate(SimpleNamespace(src=os.path.join(work, "dpvo"), dst=os.path.join(work, "dpvo_cal"), cal=SCALE_CAL))
    gate(SimpleNamespace(src=os.path.join(work, "dpvo_cal"), dst=os.path.join(work, "dpvo_cal_g"), g=GATE))
    run([sys.executable, "-m", "bench.world_fuse", "--dpvo", os.path.join(work, "dpvo_cal_g") + "#T_world_cam_d0@scale_cal",
         "--head", os.path.join(work, "head"), "--out", os.path.join(work, "cams"), *FUSE_ARGS])
    return os.path.join(work, "cams", f"{name}.npz")


def fuse_offline(T_slam, T_ours, alpha=OFFLINE_ALPHA, fallback=OFFLINE_FALLBACK):
    """Offline camera of one clip: the SLAM trajectory (rotations and shape) with its metric scale moved towards the
    real-time camera's. k = the similarity scale mapping the SLAM camera centres onto ours (frames where both are finite;
    1 if ours does not move); the SLAM centres are rescaled about the first frame by k^alpha (0.5: the geometric mean of
    the two scale estimates). Where the trajectories disagree (mean similarity residual / our path length > fallback) the
    real-time camera is kept. -> (T_world_cam (N, 4, 4), k, used the real-time camera)."""
    from bench.world import sim_fit

    Ts, To = np.asarray(T_slam, np.float64).copy(), np.asarray(T_ours, np.float64).copy()
    n = min(len(Ts), len(To))
    ok = np.isfinite(Ts[:n]).all((1, 2)) & np.isfinite(To[:n]).all((1, 2))
    A, B = Ts[:n][ok, :3, 3], To[:n][ok, :3, 3]
    k, R, t = sim_fit(A, B) if ok.sum() > 3 and np.linalg.norm(B - B.mean(0), axis=1).max() > 1e-3 else (1.0, np.eye(3), np.zeros(3))
    k = k if np.isfinite(k) and k > 0 else 1.0
    T = Ts.copy()
    c0 = T[0, :3, 3].copy()
    T[:, :3, 3] = c0 + (T[:, :3, 3] - c0) * k ** alpha
    if fallback > 0 and ok.sum() > 3:
        res = np.linalg.norm(k * A @ R.T + t - B, axis=1).mean()
        path = np.linalg.norm(np.diff(B, axis=0), axis=1).sum()
        if path > 1e-3 and res / path > fallback:
            return To, k, True
    return T, k, False


def estimate_cameras_offline(work, name, camera_head, dpvo_dir, droid_dir, droid_weights, metric3d_dir, metric3d_weights,
                             dpvo_python=sys.executable, depth_python=sys.executable, slam_python=sys.executable, mano_dir=None,
                             seed=0):
    """Offline FAH camera for one clip (docs/world.md): the real-time camera (estimate_cameras), masked DROID-SLAM +
    Metric3D in slam_python (bench/world_slam.py; droid_dir: TRAM's thirdparty/DROID-SLAM, built; metric3d_dir: a
    Metric3D checkout), fused by fuse_offline. Needs the hands' fitted meshes (verts3d) in <work>/hands/<name>.pkl.
    -> <work>/cams_offline/<name>.npz (T_world_cam, k_ours_over_slam, fallback)."""
    ours = estimate_cameras(work, name, camera_head, dpvo_dir, dpvo_python, depth_python, mano_dir, seed)
    path = os.pathsep.join([VENDOR, REPO, droid_dir, os.path.join(droid_dir, "droid_slam"), metric3d_dir]
                           + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    subprocess.run([slam_python, os.path.join(REPO, "bench", "world_slam.py"), "--frames", os.path.join(work, "frames"),
                    "--mask_run", os.path.join(work, "hands"), "--out", os.path.join(work, "slam"), "--droid_weights", droid_weights,
                    "--metric3d_dir", metric3d_dir, "--metric3d_weights", metric3d_weights,
                    "--mano_dir", mano_dir or os.environ.get("MANO_DIR", "data/mano")],
                   cwd=metric3d_dir, check=True, env=dict(os.environ, PYTHONPATH=path))
    T, k, fb = fuse_offline(np.load(os.path.join(work, "slam", f"{name}.npz"))["T_world_cam"], np.load(ours)["T_world_cam"])
    out = os.path.join(work, "cams_offline", f"{name}.npz")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out, index=np.arange(len(T)), T_world_cam=T, k_ours_over_slam=k, fallback=fb)
    return out

