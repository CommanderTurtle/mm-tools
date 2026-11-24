"""DexYCB checks of a real-to-sim clip (track.py output vs gt.npz): object shape (recon vs YCB mesh, Chamfer after rigid
ICP from the tracked pose at the init frame), object track in the camera frame (centroid error, depth error, posed-mesh
Chamfer), hand (FAH root depth / camera-frame MPJPE before and after the MoGe-2 depth shift), hand-object relative
placement (wrist -> object centroid vector error), support plane (normal angle / offset vs the AprilTag table) ->
<clip>/eval_gt.json.

    python scripts/real2sim/eval_gt.py --clip $R2S/mustard
"""
import argparse
import json
import os
import pickle

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def chamfer(A, B):
    return 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


def icp(A, B, T, iters=50):
    """Rigid ICP of points A onto B from T (4x4)."""
    tree = cKDTree(B)
    for _ in range(iters):
        X = A @ T[:3, :3].T + T[:3, 3]
        d, j = tree.query(X)
        keep = d < np.percentile(d, 90)
        P, Q = X[keep], B[j[keep]]
        mp, mq = P.mean(0), Q.mean(0)
        U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
        R = (U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt).T
        dT = np.eye(4)
        dT[:3, :3], dT[:3, 3] = R, mq - R @ mp
        T = dT @ T
    return T


def main(args):
    c = json.load(open(os.path.join(args.clip, "clip.json")))
    g = np.load(os.path.join(args.clip, "gt.npz"))
    tr = np.load(os.path.join(args.clip, args.track))
    hp = pickle.load(open(os.path.join(args.clip, "hands.pkl"), "rb"))
    gm = trimesh.load(c["gt_mesh"], force="mesh")
    rd = str(tr["recon"]) if "recon" in tr.files else "recon"
    rm = trimesh.load(os.path.join(args.clip, rd, "mesh_metric.glb" if args.track == "track.npz" else args.track[:-4] + "_mesh.glb"), force="mesh")
    Pg, Pr = np.asarray(gm.sample(5000)), np.asarray(rm.sample(5000))
    n, f0 = len(g["T_cam_obj"]), int(tr["frame0"])
    T, Tg = tr["T_cam_obj"], g["T_cam_obj"]
    cam = lambda P, M: P @ M[:3, :3].T + M[:3, 3]
    # shape: recon in the GT object frame via the tracked + GT poses at f0, refined by ICP
    T0 = np.linalg.inv(Tg[f0]) @ T[f0]
    Ti = icp(Pr, Pg, T0)
    shape_cd = chamfer(cam(Pr, Ti), Pg)
    shape_cd0 = chamfer(cam(Pr, T0), Pg)
    gc, rc = gm.bounding_box_oriented.primitive.extents, rm.bounding_box_oriented.primitive.extents
    cen_g = np.stack([cam(Pg, Tg[i]).mean(0) for i in range(n)])
    cen_r = np.stack([cam(Pr, T[i]).mean(0) for i in range(n)])
    cd = np.array([chamfer(cam(Pr, T[i]), cam(Pg, Tg[i])) for i in range(n)])
    out = dict(object=c["object"], frames=n, scale_m=float(tr["scale"]),
               extent_recon_mm=np.round(np.sort(rc)[::-1] * 1e3).tolist(), extent_gt_mm=np.round(np.sort(gc)[::-1] * 1e3).tolist(),
               shape_chamfer_mm=round(shape_cd * 1e3, 1), shape_chamfer_tracked_pose_mm=round(shape_cd0 * 1e3, 1),
               obj_centroid_err_mm_median=round(float(np.median(np.linalg.norm(cen_r - cen_g, axis=1))) * 1e3, 1),
               obj_centroid_err_mm_max=round(float(np.max(np.linalg.norm(cen_r - cen_g, axis=1))) * 1e3, 1),
               obj_depth_err_mm_median=round(float(np.median(cen_r[:, 2] - cen_g[:, 2])) * 1e3, 1),
               obj_posed_chamfer_mm_median=round(float(np.median(cd)) * 1e3, 1),
               obj_motion_gt_mm=round(float(np.linalg.norm(cen_g[-1] - cen_g[0])) * 1e3, 1),
               obj_motion_est_mm=round(float(np.linalg.norm(cen_r[-1] - cen_r[0])) * 1e3, 1))
    # hands: frames with GT and a tracked hand
    ok = g["hand_annotated"] & tr["hand_valid"]
    Jg, Ja = g["hand_joints"][ok], tr["hand_joints"][ok]
    side = bool(tr["hand_side"])
    raw = []
    for i in np.nonzero(ok)[0]:
        h = max([h for h in hp["frames"][i] if bool(h["is_right"]) == side], key=lambda h: h["score"])
        raw.append(np.asarray(h["joints3d"]))
    Jr = np.stack(raw)
    rel_g = Jg[:, 0] - cen_g[ok]
    out.update(hand_frames=int(ok.sum()),
               hand_root_depth_err_mm_raw=round(float(np.median(Jr[:, 0, 2] - Jg[:, 0, 2])) * 1e3, 1),
               hand_root_depth_err_mm_aligned=round(float(np.median(Ja[:, 0, 2] - Jg[:, 0, 2])) * 1e3, 1),
               hand_mpjpe_cam_mm_raw=round(float(np.linalg.norm(Jr - Jg, axis=-1).mean()) * 1e3, 1),
               hand_mpjpe_cam_mm_aligned=round(float(np.linalg.norm(Ja - Jg, axis=-1).mean()) * 1e3, 1),
               hand_pampjpe_rootrel_mm=round(float(np.linalg.norm((Jr - Jr[:, :1]) - (Jg - Jg[:, :1]), axis=-1).mean()) * 1e3, 1),
               wrist_obj_rel_err_mm_raw=round(float(np.median(np.linalg.norm((Jr[:, 0] - cen_r[ok]) - rel_g, axis=1))) * 1e3, 1),
               wrist_obj_rel_err_mm_aligned=round(float(np.median(np.linalg.norm((Ja[:, 0] - cen_r[ok]) - rel_g, axis=1))) * 1e3, 1))
    # support plane vs the AprilTag table (z = 0 of T_world_cam)
    Twc = g["T_world_cam"]
    ng = Twc[2, :3]
    dg = Twc[2, 3]
    ng, dg = (ng, dg) if dg > 0 else (-ng, -dg)
    pl = tr["plane"]
    out.update(plane_normal_err_deg=round(float(np.degrees(np.arccos(np.clip(pl[:3] @ ng, -1, 1)))), 2),
               plane_offset_err_mm=round(float(pl[3] - dg) * 1e3, 1))
    k = float(tr["hand_scale"]) if "hand_scale" in tr.files else 1.0  # object size / distance factor from the hand
    ratio = np.median((np.sort(rc)[::-1] / np.sort(gc)[::-1])[:2])  # two largest extents: the depth one is a guess
    out.update(hand_scale=round(k, 3), scale_err_pct=round(float(ratio - 1) * 100, 1),
               scale_err_pct_before_hand_scale=round(float(ratio / k - 1) * 100, 1))
    if "init_scale" in tr.files:  # the frame-0 initialisation's own scale (SAM 3D Objects layout, or the search's guess)
        out["init_scale_err_pct"] = round(float(ratio / k * float(tr["init_scale"]) / float(tr["fit_scale_f0"]) - 1) * 100, 1)
    d = np.linalg.norm(T[:, :3, 3] - T[0, :3, 3], axis=1)
    still = np.linalg.norm(Tg[:, :3, 3] - Tg[0, :3, 3], axis=1) < 0.005
    out["obj_drift_mm_while_static_max"] = round(float(d[still].max()) * 1e3, 1)
    json.dump(out, open(os.path.join(args.clip, "eval_gt.json" if args.track == "track.npz" else "eval_" + args.track[:-4] + ".json"), "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--track", default="track.npz")
    main(p.parse_args())
