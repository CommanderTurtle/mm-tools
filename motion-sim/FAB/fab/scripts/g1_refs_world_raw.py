"""The retargeter table's reference input, step 1 (scripts/reproduce_g1_refs.sh): a clip's fab-world body (the fab-world output
<clip>.npz: SMPL pose / trans / betas in the z-up world, its cameras) written in bench/sonic_motion.py's raw format, the body in
each camera's frame via inv(T_world_cam), so that sonic_motion's world lift (--world <clip>.npz) puts exactly that body back into
its world. The per-frame raw <clip>_raw.npz (FAB's per-frame outputs with its dense points: pts, surface_idx) supplies the fields
the SMPL path does not read.
    python scripts/g1_refs_world_raw.py <world dir> <raw dir> <out dir> <clip> [<clip> ...]"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.body_models import SMPL  # noqa: E402
from fab import paths  # noqa: E402


def main():
    wd, rd, od = (Path(x) for x in sys.argv[1:4])
    od.mkdir(parents=True, exist_ok=True)
    clips = sys.argv[4:]
    body = SMPL(paths.smpl_dir() / "SMPL_NEUTRAL.pkl").eval()
    for c in clips:
        w, r = np.load(wd / f"{c}.npz", allow_pickle=True), dict(np.load(rd / f"{c}_raw.npz", allow_pickle=True))
        assert np.array_equal(np.asarray(w["frames"], int), np.asarray(r["frames"], int)), c
        T = len(w["frames"])
        Tcw = np.linalg.inv(np.asarray(w["T_world_cam"], np.float64))
        betas = np.asarray(w["betas"], np.float32).reshape(1, -1)
        with torch.no_grad():  # the rest pelvis J0(betas): SMPL rotates the global orientation about it
            _, j0 = body(torch.zeros(1, 3), torch.zeros(1, 69), torch.from_numpy(betas), torch.zeros(1, 3))
        J0 = j0[0, 0].numpy().astype(np.float64)
        pose = np.asarray(w["pose"], np.float64).reshape(T, 72).copy()
        R_w = Rotation.from_rotvec(pose[:, :3]).as_matrix()
        pose[:, :3] = Rotation.from_matrix(Tcw[:, :3, :3] @ R_w).as_rotvec()
        trans_w = np.asarray(w["trans"], np.float64)
        trans_c = np.einsum("nij,nj->ni", Tcw[:, :3, :3], trans_w + J0) + Tcw[:, :3, 3] - J0
        r.update(pose_c=pose.astype(np.float32), betas=np.repeat(betas, T, 0), trans_c=trans_c.astype(np.float32),
                 valid=np.asarray(w["valid"], bool), K=np.asarray(w["K"], np.float64), ext=Tcw,
                 source=f"fab-world body ({wd / (c + '.npz')}), camera frame via inv(T_world_cam); other fields: {rd}")
        np.savez(od / f"{c}_raw.npz", **r)
        print(c, T, "frames", flush=True)


if __name__ == "__main__":
    main()
