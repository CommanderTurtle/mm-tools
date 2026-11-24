"""Training smoke test: a few synthetic samples in the pool / sidecar format of docs/training.md, the FAB-S body recipe
for 3 steps and the contact-head recipe for 2 steps on its output (one GPU), then a bundle. Checks that every logged
loss is finite and that the checkpoints and the bundle are written. Synthetic labels: the numbers mean nothing.

    python scripts/train_smoke.py --out /tmp/fab_train_smoke

Needs one CUDA GPU, SAM 3D Body's model_config.yaml + assets/mhr_model.pt ($FAB_SAM3D; fab.download_sam3d()) and
$FAB_HOME/body_models (SMPL_{NEUTRAL,MALE,FEMALE}.pkl, J_regressor_h36m.npy, the MHR -> SMPL mapping files). The G1 head
is trained when the G1 description is installed (models/fab_assets.py), else switched off for the test. No weights
are downloaded (--no_pretrained --no_teacher_weights).
"""
import argparse
import io
import math
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fab import paths  # noqa: E402
from fab.retarget.g1_files import urdf  # noqa: E402

N = 8  # samples


def records(rng):
    """N synthetic samples: (key, crop jpg bytes, record dict)."""
    W, H, S = 640, 480, 256
    K = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]], np.float32)
    out = []
    for i in range(N):
        cx, cy, side = 300.0 + 10 * i, 260.0, 400.0
        s = S / side
        affine = np.array([[s, 0, S / 2 - s * cx], [0, s, S / 2 - s * cy]], np.float32)
        crop = rng.integers(0, 255, (S, S, 3), dtype=np.uint8)
        jpg = cv2.imencode(".jpg", crop)[1].tobytes()
        kp2d = np.zeros((70, 3), np.float32)
        kp2d[:17, :2] = rng.uniform(40, S - 40, (17, 2))
        kp2d[:17, 2] = 1.0
        pose = np.zeros(72, np.float32)
        pose[0] = math.pi  # SMPL y-up -> camera y-down
        rec = dict(
            affine=affine, cam_int=K, ori_size=np.array([W, H], np.float32),
            bbox=np.array([cx - 100, cy - 160, cx + 100, cy + 160], np.float32),
            bbox_center=np.array([cx, cy], np.float32), bbox_scale=np.array([side, side], np.float32),
            gt_kp2d=kp2d, gt_kp3d=np.zeros((70, 4), np.float32),
            gt_mhr_valid=np.float32(0), gt_model_params=np.zeros(204, np.float32), gt_shape=np.zeros(45, np.float32),
            gt_smpl_pose=pose, gt_smpl_betas=np.zeros(11, np.float32), gt_smpl_trans=np.array([0, 0.2, 3.0], np.float32),
            gt_smpl_valid=np.float32(i % 2), gt_smpl_gender=np.int64(0),
            t_kp2d=rng.uniform(40, S - 40, (70, 2)).astype(np.float32),
            t_kp3d=rng.normal(0, 0.3, (70, 3)).astype(np.float32), t_cam_t=np.array([0, 0.3, 4.0], np.float32),
            t_model_params=np.zeros(204, np.float32), t_shape=np.zeros(45, np.float32), t_pose_raw=np.zeros(266, np.float32),
            t_global_rot=np.zeros(3, np.float32), t_body_pose=np.zeros(133, np.float32), t_scale=np.zeros(28, np.float32),
            t_hand=np.zeros(108, np.float32))
        out.append((f"synth_{i:04d}", jpg, rec))
    return out


def write_pool(d, rng):
    d.mkdir(parents=True, exist_ok=True)
    tar = d / "000000.tar"
    recs = records(rng)
    with tarfile.open(tar, "w", format=tarfile.PAX_FORMAT, pax_headers={"pool.source": "synth"}) as tw:
        for key, jpg, rec in recs:
            buf = io.BytesIO()
            np.savez(buf, **rec)
            for ext, data in (("jpg", jpg), ("npz", buf.getvalue())):
                info = tarfile.TarInfo(f"{key}.{ext}")
                info.size = len(data)
                tw.addfile(info, io.BytesIO(data))
    keys = np.array([k for k, _, _ in recs])
    stem = str(tar)[:-4]
    t_fields = {k: np.stack([r[k] for _, _, r in recs]) for k in recs[0][2] if k.startswith("t_")}
    t_fields["t_kp2d"] = t_fields["t_kp2d"] + rng.normal(0, 2, t_fields["t_kp2d"].shape).astype(np.float32)
    np.savez(f"{stem}.teacher.npz", keys=keys, **t_fields)  # the teacher (FAB-H) labels
    wb = np.zeros((N, 70, 3), np.float32)
    wb[:, 15:21, :2] = rng.uniform(40, 216, (N, 6, 2))
    wb[:, 15:21, 2] = 1.0
    np.savez(f"{stem}.wholebody.npz", keys=keys, kp2d=wb)  # human whole-body 2D
    np.savez(f"{stem}.g1_body.npz", keys=keys, q=np.zeros((N, 29), np.float32), R=np.tile(np.eye(3, dtype=np.float32), (N, 1, 1)))
    np.savez(f"{stem}.contact.npz", keys=keys, c=rng.integers(0, 2, (N, 8)).astype(np.uint8), valid=np.ones((N, 8), np.uint8),
             h=rng.uniform(0, 0.2, (N, 8)).astype(np.float16))
    return tar


def config(src, out, overrides):
    import yaml

    cfg = yaml.safe_load(Path(src).read_text())
    cfg.update(overrides)
    Path(out).write_text(yaml.safe_dump(cfg, sort_keys=True))
    return out


def train(cfg, out, extra=()):
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    cmd = [sys.executable, "-m", "train.train_fab", "--config", str(cfg), "--out", str(out), "--no_pretrained",
           "--no_teacher_weights", *extra]
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    log = r.stdout + r.stderr
    (Path(out) / "smoke.log").parent.mkdir(parents=True, exist_ok=True)
    (Path(out) / "smoke.log").write_text(log)
    if r.returncode:
        sys.exit(f"train_fab failed (rc {r.returncode}); last lines:\n" + "\n".join(log.splitlines()[-25:]))
    steps = [ln for ln in log.splitlines() if ln.startswith("step ")]
    vals = [float(v) for ln in steps for v in re.findall(r"=(-?[0-9.eE+-]+|nan|inf)", ln)]
    assert steps, "no step lines logged"
    assert all(math.isfinite(v) for v in vals), f"non-finite loss logged:\n{steps[-1]}"
    assert (Path(out) / "model.pt").exists(), "no model.pt written"
    print(f"{out}: {len(steps)} steps logged, all finite; last: {steps[-1][:160]}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", default="/tmp/fab_train_smoke")
    a = p.parse_args()
    import torch

    if not torch.cuda.is_available():
        sys.exit("train_smoke: needs a CUDA GPU")
    sam3d, bm = paths.sam3d_dir(), paths.body_models_dir()
    for f, what in ((sam3d / "model_config.yaml", "SAM 3D Body config (fab.download_sam3d())"),
                    (sam3d / "assets" / "mhr_model.pt", "MHR model (fab.download_sam3d())"),
                    (bm / "smpl" / "SMPL_NEUTRAL.pkl", "SMPL model files (README.md, Install)"),
                    (bm / "smpl" / "J_regressor_h36m.npy", "J_regressor_h36m.npy (README.md, Install)")):
        paths.require(f, what)
    g1 = urdf().exists()
    print(f"G1 head: {'on' if g1 else 'off (no G1 description; scripts/download_g1.sh)'}", flush=True)
    out = Path(a.out)
    pool = write_pool(out / "data" / "pools" / "synth", np.random.default_rng(0))
    common = {"pools": str(pool), "pool_list": None, "batch": 2, "workers": 1, "log_every": 1, "save_every": 1,
              "bm_dir": str(bm), "g1_head": g1}
    body = config(ROOT / "configs/train/fab-s.yaml", out / "body.yaml", {**common, "steps": 3, "foot_reg": ""})
    train(body, out / "body")
    head = config(ROOT / "configs/train/heads-contact-s.yaml", out / "contact.yaml",
                  {**common, "steps": 2, "init": str(out / "body" / "model.pt")})
    train(head, out / "contact")
    env = {**os.environ, "PYTHONPATH": str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    r = subprocess.run([sys.executable, "scripts/build_bundle.py", "--body", str(out / "body" / "model.pt"),
                        "--contact", str(out / "contact" / "model.pt"), "--out", str(out / "bundle" / "model.pt")],
                       cwd=ROOT, env=env, capture_output=True, text=True)
    print(r.stdout.strip(), flush=True)
    if r.returncode:
        sys.exit(f"build_bundle failed:\n{r.stderr[-2000:]}")
    st = torch.load(out / "bundle" / "model.pt", map_location="cpu", weights_only=False)
    n = sum(k.startswith("contact_head.") for k in st["model"])
    assert n > 0 and st.get("body_args"), "bundle without the contact head / body_args"
    print(f"train_smoke OK: body 3 steps, contact head 2 steps, bundle {out / 'bundle' / 'model.pt'} ({n} contact-head tensors)")


if __name__ == "__main__":
    main()
