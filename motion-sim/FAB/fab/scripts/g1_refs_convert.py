"""The G1 reference files of the paper's retargeter table (scripts/reproduce_g1_refs.sh, last step): each clip's pinned G1
reference (<run>/p/<clip>_motion.npz) with the SMPL fields of its reference lift (<lift>/<clip>__ikmot_motion.npz: the SMPL the
trackers read, its G1 joint order), one npz per clip -> <out>/<clip>__ikmot_motion.npz.
    python scripts/g1_refs_convert.py <run dir> <lift dir> <out dir> <solver note> <clip> [<clip> ...]"""
import sys
from pathlib import Path

import numpy as np


def main():
    run, lift, out, note = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    out.mkdir(parents=True, exist_ok=True)
    for c in sys.argv[5:]:
        z, b = np.load(run / "p" / f"{c}_motion.npz", allow_pickle=True), dict(np.load(lift / f"{c}__ikmot_motion.npz", allow_pickle=True))
        T, Tb = len(z["g1_dof"]), len(b["g1_dof"])
        # the solver's 50 fps grid floor((n - 1) / fps * 50) + 1 can lose the last frame to rounding at fps 50
        assert Tb - 1 <= T <= Tb, (c, T, Tb)
        b = {k: (v[:T] if isinstance(v, np.ndarray) and v.ndim and len(v) == Tb and k != "frames" else v) for k, v in b.items()}
        jn, jb = [str(j) for j in z["g1_joint_names"]], [str(j) for j in b["g1_joint_names"]]
        b.update(g1_dof=np.asarray(z["g1_dof"], np.float32)[:, [jn.index(j) for j in jb]], g1_root_pos=np.asarray(z["g1_root_pos"], np.float32),
                 g1_root_quat_xyzw=np.asarray(z["g1_root_quat_xyzw"], np.float32), scale=float(z["scale"]), solver=note)
        np.savez(out / f"{c}__ikmot_motion.npz", **b)
        print(c, T, "frames ->", out / f"{c}__ikmot_motion.npz", flush=True)


if __name__ == "__main__":
    main()
