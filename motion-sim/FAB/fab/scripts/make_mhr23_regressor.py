"""Build the 23-joint regressor ($FAB_HOME/extras/mhr23_regressor.npz): the 23 MHR body keypoints (MHR70 0-20, 41, 62)
as affine combinations (weights summing to 1) of the 48 nearest SMPL-topology vertices of the model's own mesh, fitted
by least squares (ridge 1e-4) on (mesh_smpl, keypoints3d) pairs of FAB-L's outputs on 3,000 3DPW test people (seed 0).
It is needed by FABPredictor(dense_fit=True) and by the body training recipe's foot terms (docs/training.md). It is not
distributed because it is tied to the SMPL topology; building it needs SMPL (README, Install, step 3) and 3DPW
(datasets/README.md).

    python scripts/make_mhr23_regressor.py --root $FAB_DATA/3dpw [--model fab-l] [--n_fit 3000] [--out PATH]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fab import paths  # noqa: E402

BODY23 = list(range(21)) + [41, 62]


def fit_regressor(V, Kp, n_near=48, ridge=1e-4):
    """V (S, 6890, 3) meshes, Kp (S, 23, 3) keypoints -> reg (23, 6890) float32, per-keypoint fit error (mm)."""
    reg = np.zeros((Kp.shape[1], V.shape[1]))
    err = []
    for k in range(Kp.shape[1]):
        d = np.linalg.norm(V - Kp[:, k : k + 1], axis=-1).mean(0)
        nn_ = np.argsort(d)[:n_near]
        Xn = V[:, nn_]  # (S, n, 3)
        A = Xn.transpose(0, 2, 1).reshape(-1, n_near)  # (3S, n)
        b = Kp[:, k].reshape(-1)
        KKT = np.zeros((n_near + 1, n_near + 1))
        KKT[:n_near, :n_near] = A.T @ A + ridge * len(V) * np.eye(n_near)
        KKT[:n_near, n_near] = KKT[n_near, :n_near] = 1.0
        sol = np.linalg.solve(KKT, np.concatenate([A.T @ b, [1.0]]))
        reg[k, nn_] = sol[:n_near]
        err.append(np.linalg.norm(Xn.transpose(0, 2, 1) @ sol[:n_near] - Kp[:, k], axis=-1).mean() * 1000)
    return reg.astype(np.float32), np.array(err)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, help="the 3DPW release folder (imageFiles/, sequenceFiles/)")
    p.add_argument("--split", default="test")
    p.add_argument("--model", default="fab-l")
    p.add_argument("--n_fit", type=int, default=3000)
    p.add_argument("--out", default=str(paths.home() / "extras" / "mhr23_regressor.npz"))
    a = p.parse_args()

    from bench.datasets import DATASETS
    from fab.predictor import FABPredictor

    samples = DATASETS["3dpw"](a.root) if a.split == "test" else DATASETS[f"3dpw_{a.split}"](a.root)
    rng = np.random.default_rng(0)
    sel = rng.choice(len(samples), min(a.n_fit, len(samples)), replace=False)
    fab = FABPredictor(a.model, fast=False, detector="", smpl_fitter="")
    V, K = [], []
    for j, i in enumerate(sel):
        s = samples[int(i)]
        img = cv2.cvtColor(cv2.imread(s["image"]), cv2.COLOR_BGR2RGB)
        pp = fab.predict(img, np.asarray(s["K"], np.float32), np.asarray(s["bbox"], np.float32)[None])
        if pp:
            V.append(pp[0]["mesh_smpl"]); K.append(pp[0]["keypoints3d"][BODY23])
        if j % 500 == 0:
            print(f"[mhr23] {j}/{len(sel)}", flush=True)
    reg, err = fit_regressor(np.asarray(V, np.float64), np.asarray(K, np.float64))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, reg=reg, err=err, body23=np.array(BODY23))
    print(f"[mhr23] {len(V)} people, fit error mean {err.mean():.1f} mm (max {err.max():.1f}) -> {a.out}")


if __name__ == "__main__":
    main()
