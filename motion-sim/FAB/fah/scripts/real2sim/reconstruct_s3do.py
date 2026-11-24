"""Object mesh + layout from one clear frame with SAM 3D Objects (github.com/facebookresearch/sam-3d-objects, in its own
environment; SAM3D_OBJECTS_REPO = its checkout, SAM3D_OBJECTS_CONFIG = the checkpoints' pipeline.yaml): the
frame, the object mask (segment.py) and the MoGe-2 metric pointmap of depth.npz (PyTorch3D camera convention, as the
pipeline expects) -> <clip>/recon_s3do[_s<seed>]/mesh.glb + layout.json (R, t, s with X_cam = s R v + t, OpenCV camera,
v = the glb vertices) + recon.json. The silhouette IoU of the projected layout on the frame is a sanity check.
--seeds: extra samples with the pipeline loaded once (shape only: no texture baking).

    LIDRA_SKIP_INIT=true CONDA_PREFIX=$CUDA_HOME $SAM3DO_PY scripts/real2sim/reconstruct_s3do.py --clips <clip>,<clip> --frame -1 \
        --texture
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))
REPO = os.environ.get("SAM3D_OBJECTS_REPO", os.path.join(FAH_HOME, "sam-3d-objects"))
CFG = os.environ.get("SAM3D_OBJECTS_CONFIG", os.path.join(REPO, "checkpoints", "hf", "pipeline.yaml"))
GLB_TO_LOCAL = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float64)  # to_glb: v_glb = v_local @ M -> v_local = M v_glb
P3D = np.diag([-1.0, -1.0, 1.0])  # OpenCV (x right, y down) <-> PyTorch3D (x left, y up) camera


def pointmap_p3d(depth, K):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = np.stack([(u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth], -1)
    X[depth <= 0] = np.nan
    return torch.from_numpy((X @ P3D).astype(np.float32))


def layout_cv(out):
    """SAM 3D layout (local -> PyTorch3D camera: X = (s p) @ R(q) + t, row vectors) as X_cv = s R v_glb + t."""
    from pytorch3d.transforms import quaternion_to_matrix

    Rq = quaternion_to_matrix(out["rotation"].float()).reshape(3, 3).cpu().numpy().astype(np.float64)
    s = out["scale"].reshape(-1)[:3].float().cpu().numpy().astype(np.float64)
    t = out["translation"].reshape(-1)[:3].float().cpu().numpy().astype(np.float64)
    return P3D @ Rq.T @ GLB_TO_LOCAL, float(s.mean()), P3D @ t, s


def silhouette_iou(V, K, mask):
    uv = V[:, :2] / V[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
    H, W = mask.shape
    m = np.zeros((H, W), np.uint8)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H) & (V[:, 2] > 0)
    m[uv[ok, 1].astype(int), uv[ok, 0].astype(int)] = 1
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
    return float((m & mask).sum() / max((m | mask).sum(), 1))


def save(od, out, K, mask, f, seed, tex, dt, faces=40000):
    import trimesh

    os.makedirs(od, exist_ok=True)
    glb = out["glb"]
    mesh = glb if isinstance(glb, trimesh.Trimesh) else trimesh.util.concatenate(list(glb.geometry.values()))
    if len(mesh.faces) > faces and not hasattr(mesh.visual, "uv"):  # raw marching-cubes output (~400k faces): decimate
        import open3d as o3d

        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(mesh.vertices)), o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        vc = getattr(mesh.visual, "vertex_colors", None)
        if vc is not None and len(vc) == len(mesh.vertices):
            om.vertex_colors = o3d.utility.Vector3dVector(np.asarray(vc)[:, :3] / 255.0)
        om = om.simplify_quadric_decimation(faces)
        mesh = trimesh.Trimesh(np.asarray(om.vertices), np.asarray(om.triangles),
                               vertex_colors=(np.asarray(om.vertex_colors) * 255).astype(np.uint8) if om.has_vertex_colors() else None)
    mesh.export(os.path.join(od, "mesh.glb"))
    R, s, t, s3 = layout_cv(out)
    V = s * np.asarray(mesh.vertices) @ R.T + t
    iou = silhouette_iou(np.asarray(trimesh.Trimesh(V, mesh.faces, process=False).sample(20000)), K, mask)
    json.dump(dict(R=R.tolist(), t=t.tolist(), s=s, s_xyz=s3.tolist(), silhouette_iou=iou, frame=f), open(os.path.join(od, "layout.json"), "w"),
              indent=1)
    json.dump(dict(frame=f, model="SAM 3D Objects" + (" (texture baked)" if tex else ""), seed=seed, faces=len(mesh.faces),
                   shape_s=round(dt, 1), total_s=round(dt, 1)), open(os.path.join(od, "recon.json"), "w"), indent=1)
    ext = (V.max(0) - V.min(0)) * 1e3
    print(od, "seed", seed, len(mesh.faces), "faces, layout scale %.3f, extent %s mm, depth %.3f m, silhouette IoU %.3f, %.0f s" %
          (s, np.round(ext).astype(int).tolist(), t[2], iou, dt))


def pick_frame(clip, upto=0.5):
    """Reconstruction frame by mask quality: among the frames before a (dilated) FAH hand mask first touches the object mask
    (at most the first `upto` of the clip), the one with the largest object mask x its SAM 3 score."""
    import pickle

    from bench.seg.rle import decode

    m = np.load(os.path.join(clip, "masks.npz"))
    obj, score = m["obj"], m["score"]
    hp = pickle.load(open(os.path.join(clip, "hands.pkl"), "rb"))["frames"]
    last = max(int(len(obj) * upto), 1)
    for i, f in enumerate(hp[:last]):
        hm = np.zeros(obj.shape[1:], bool)
        for h in f:
            if "mask_rle" in h:
                hm |= decode(h["mask_rle"])
        if (cv2.dilate(hm.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & obj[i]).sum() >= 20:
            last = max(i, 1)
            break
    q = obj[:last].sum((1, 2)) * np.clip(score[:last], 0, None)
    if q.max() <= 0:  # no mask in that window (masks propagated from a later prompt frame): the best masked frame of the clip
        q = obj.sum((1, 2)) * np.clip(score, 0, None)
    return int(np.argmax(q))


def main(args):
    os.environ.setdefault("CONDA_PREFIX", "/usr/local/cuda")  # notebook/inference.py sets CUDA_HOME from it
    sys.path.insert(0, os.path.join(REPO, "notebook"))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from inference import Inference

    inf = None
    for clip in (args.clips.split(",") if args.clips else [args.clip]):
        obj = np.load(os.path.join(clip, "masks.npz"))["obj"]
        f = args.frame if args.frame >= 0 else pick_frame(clip)
        mask = obj[f]
        img = cv2.imread(os.path.join(clip, "frames", f"{f:06d}.jpg"))[..., ::-1].copy()
        dep = np.load(os.path.join(clip, "depth.npz"))
        K = np.asarray(dep["K"], np.float64)
        pm = pointmap_p3d(dep["depth"][f].astype(np.float32), K) if args.pointmap == "moge2" else None
        inf = inf or Inference(CFG, compile=False)  # loaded once for all clips
        for i, sd in enumerate([args.seed] + [int(v) for v in args.seeds.split(",") if v]):
            t0 = time.time()
            tex = args.texture and i == 0
            out = inf._pipeline.run(np.concatenate([img, mask[..., None].astype(np.uint8) * 255], -1), None, sd, stage1_only=False,
                                    with_mesh_postprocess=tex, with_texture_baking=tex, with_layout_postprocess=args.layout_post,
                                    use_vertex_color=not tex, pointmap=pm)
            save(os.path.join(clip, args.out if i == 0 else f"{args.out}_s{sd}"), out, K, mask, f, sd, tex, time.time() - t0)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", default="")
    p.add_argument("--clips", default="", help="comma-separated clips, the model loaded once")
    p.add_argument("--frame", type=int, default=0, help="-1: by mask quality before the hand touches the object (pick_frame)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seeds", default="", help="extra shape samples -> <out>_s<seed>/")
    p.add_argument("--texture", action="store_true", help="bake a UV texture for the first sample (mesh post-processing on)")
    p.add_argument("--pointmap", default="moge2", choices=["moge2", "s3do"], help="moge2: depth.npz + K; s3do: SAM 3D Objects' own MoGe pointmap")
    p.add_argument("--layout_post", action="store_true", help="the pipeline's layout post-optimisation")
    p.add_argument("--out", default="recon_s3do")
    main(p.parse_args())
