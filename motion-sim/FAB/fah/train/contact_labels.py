"""Per-vertex hand-object contact labels on the GT MANO meshes of the crop pools whose datasets ship object meshes and
per-frame object poses, the held-out test crop sets with the same labels (--test), their per-frame GT (--frames), the
per-frame GT of bench.run prediction pickles for the video model (--video_gt), and the training label root (--make_root).

Label rule (HACO's: a MANO vertex is in contact iff it is within 10 mm of the object): signed distance of every GT MANO
vertex to the nearest object surface of the frame. Distance = exact point-to-triangle distance of the vertex (the pool's
*_v.npy, camera frame, float16) to each object mesh of the frame placed in that camera frame (Open3D RaycastingScene),
negative when the vertex lies inside the mesh (generalised winding number > 0.5, igl: robust to the non-watertight dataset
meshes; tested on every vertex inside the object's bounding box), min over the frame's objects; clipped to +-CLIP_MM.
Penetrating vertices are contact at any depth. The head trains on this distance at 5 / 10 / 20 / 35 mm (train.contact_head
--dists; 10 mm = the main output). Objects only: tables / scene / the other hand are not surfaces here. contact.jsonl keeps
min_mm and n10 so other rules can be derived. InterHand2.6M crops carry no objects: train.contact_head --zero_sources labels
every vertex 0 (no label files needed).

pool                 raw data (poses)                                              object meshes
hot3d_lab_mesh         HOT3D-Clips tars: <frame>.objects.json T_world_from_object,   object_models_eval/*.glb (BOP id)
                     <frame>.cameras.json (world -> upright rectified camera as bench.datasets.hot3d)
dexycb_lab           DexYCB clips (bench.datasets.dexycb): labels_*.npz pose_y      models/*/textured_simple.obj
ho3d_lab             HO3D_v3.zip meta/*.pkl objRot / objTrans / objName            the same YCB models
oakink2_ego_lab,     anno_preview/<seq>.pkl obj_transf (object -> world), cam_extr  object_repair/align_ds/*/model.obj
oakink2_lab
valid = every object of the frame has a pose and a mesh (else the crop's labels are ignored, not negatives). ARCTIC, HOI4D
and MOW labels have their own builders (train.build_arctic_contact, train.build_hoi4d_contact, train.build_mow_contact) with
the same output layout and distance (train.contact_sdf).

Held-out test crop sets (--test, crops / labels / mirroring by the pool builders' own _work): hot3d_bench = the HOT3D
benchmark clips (bench/splits/hot3d_bench.json), every 5th frame; dexycb_s0test = DexYCB s0 test, every 8th frame per
camera. Written to <out>/test/<set>/ (index.jsonl, crops, *_v.npy), then labelled like a pool. --frames <set>: every frame,
every GT hand of the set's clips -> <out>/test/<set>_frames/<clip>.npz = key, is_right, sdf (n, 778) fp16 mm, vis (n, 778)
(z-buffer: the vertex is the first hit of its camera ray among the hand + object meshes, 2 mm tolerance, inside the image).
--video_gt <pred dir> --dataset hot3d | dexycb | oakink2: every GT / ignore hand of every frame of the pickles -> <out>/video_gt/
<dataset>/<clip>.npz = frame, kind (0 gt, 1 ignore), hand index, is_right, valid, sdf (n, 778) fp16.

Output <out>/<pool>/: sdf.npy (N, 778) float16 signed distance (mm), contact.jsonl one line per crop in the pool's index.jsonl
order (file, src, is_right, valid, min_mm, n10, obj); per-group parts in <out>/<pool>/parts/ (resumable). Pools are only read.
--make_root <dir>: the training label root of the released heads = links to the label dirs of --root_sources + a filtered
HOI4D copy (Chair sequences and GT penetration deeper than 15 mm marked invalid; the HOI4D left hands only occur in Chair).

    python -m train.contact_labels --pool hot3d_lab_mesh --crops $DATA_ROOT/crops --data_root $DATA_ROOT --out $DATA_ROOT/contact
    python -m train.contact_labels --test hot3d_bench --data_root $DATA_ROOT --out $DATA_ROOT/contact
    python -m train.contact_labels --frames hot3d_bench --data_root $DATA_ROOT --out $DATA_ROOT/contact
    python -m train.contact_labels --video_gt $TRACKS/hot3d/train-fold --dataset hot3d --data_root $DATA_ROOT --out $DATA_ROOT/contact
    python -m train.contact_labels --make_root $DATA_ROOT/contact/root --out $DATA_ROOT/contact
"""
import argparse
import glob
import io
import json
import os
import zipfile
from collections import defaultdict
from multiprocessing import Pool

import numpy as np

from train.contact_sdf import CLIP_MM, THRESH_MM, MeshSDF, mesh_sdf, signed_mm  # noqa: F401  (one distance rule, re-exported)

MIN_VERTS = 1
ROOT_SOURCES = "hot3d_lab_mesh,dexycb_lab,ho3d_lab,oakink2_ego_lab,oakink2_lab,mow_train_c35,arctic_ego"
_MESH = {}


def configure(data_root):
    """Raw data locations under data_root (docs/contact.md): HOT3D-Clips tars, HOT3D object models, DexYCB clips + YCB models,
    HO3D_v3.zip, OakInk2 annotations + object models."""
    global D, HOT3D_TARS, HOT3D_MODELS, YCB_MODELS, OAK_MODELS
    D = data_root
    HOT3D_TARS = [os.path.join(D, "hot3d_train600", "train_aria"), os.path.join(D, "hot3d_train_rest", "train_aria"),
                  os.path.join(D, "hot3d", "train_aria")]
    HOT3D_MODELS = os.path.join(D, "hot3d_objects", "object_models_eval")
    YCB_MODELS = os.path.join(D, "dexycb", "models")
    OAK_MODELS = os.path.join(D, "oakink2_objects", "object_repair", "align_ds")


configure(os.environ.get("DATA_ROOT", "data"))


def load_mesh(path):
    """(RaycastingScene, V, F) of a mesh file (metres, object frame); duplicated seam vertices merged; one per worker."""
    if path not in _MESH:
        import open3d as o3d
        import trimesh

        m = trimesh.load(path, force="mesh", process=True)
        V, F = np.asarray(m.vertices, np.float64), np.asarray(m.faces, np.int64)
        sc = o3d.t.geometry.RaycastingScene(nthreads=1)  # one thread per worker (the default takes every core)
        sc.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
        _MESH[path] = (sc, V, F)
    return _MESH[path]


def signed_dist(path, T, pts):
    """Signed distance (mm) of camera-frame points (n, 3) to the mesh at pose T (object -> camera, 4 x 4)."""
    sc, V, F = load_mesh(path)
    return signed_mm(sc, V, F, (pts - T[:3, 3]) @ T[:3, :3])


def label(verts, objs):
    """verts (778, 3) camera frame; objs [(name, mesh path, T)] -> (sdf mm (778,), nearest object name)."""
    best, name = np.full(len(verts), CLIP_MM), None
    for o, path, T in objs:
        d = signed_dist(path, T, verts)
        if name is None or d.min() < best.min():
            name = o
        best = np.minimum(best, d)
    return np.clip(best, -CLIP_MM, CLIP_MM), name


def rigid(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t, np.float64).reshape(3)
    return T


def quat_T(p):
    from scipy.spatial.transform import Rotation

    w, x, y, z = p["quaternion_wxyz"]
    return rigid(Rotation.from_quat([x, y, z, w]).as_matrix(), p["translation_xyz"])


# ---- per-dataset frame objects: group -> {frame key: (valid, [(name, mesh path, T object -> camera)])}

def hot3d_objects(clip, keys):
    import tarfile

    from hand_tracking_toolkit import camera as htt_camera

    from bench.datasets.hot3d import ROT_CW

    tar = next((os.path.join(d, clip + ".tar") for d in HOT3D_TARS if os.path.exists(os.path.join(d, clip + ".tar"))), None)
    if tar is None:
        return None
    t = tarfile.open(tar)
    names = set(t.getnames())
    js = lambda k, kind: json.load(t.extractfile(f"{k}.{kind}.json"))
    rot = js(sorted(n.split(".")[0] for n in names if n[:6].isdigit())[0], "info")["device"] == "Aria"
    R4 = rigid(ROT_CW if rot else np.eye(3), np.zeros(3))
    out = {}
    for k in keys:
        if f"{k}.objects.json" not in names:
            out[k] = (False, [])
            continue
        T_cam = R4 @ np.linalg.inv(htt_camera.from_json(js(k, "cameras")["214-1"]).T_world_from_eye)
        objs, ok = [], True
        for bop, insts in js(k, "objects").items():
            path = os.path.join(HOT3D_MODELS, f"obj_{int(bop):06d}.glb")
            ok &= os.path.exists(path)
            objs += [(insts[0].get("object_name", bop), path, T_cam @ quat_T(q["T_world_from_object"])) for q in insts if os.path.exists(path)]
        out[k] = (ok and bool(objs), objs)
    return out


def dexycb_objects(clip, keys):
    import tarfile

    from bench.datasets.dexycb import SUBJECTS

    subj = {s[:8]: s for s in SUBJECTS}[clip[:8]]
    t = tarfile.open(os.path.join(D, "dexycb", "clips", subj, clip + ".tar"))
    ycb = sorted(os.listdir(YCB_MODELS))  # DexYCB ycb_ids are 1-based indices of the 21 sorted YCB model names
    ids = json.load(t.extractfile("info.json"))["meta"]["ycb_ids"]
    out = {}
    for k in keys:
        P = np.load(io.BytesIO(t.extractfile(f"labels_{k}.npz").read()))["pose_y"]
        objs = [(ycb[i - 1], os.path.join(YCB_MODELS, ycb[i - 1], "textured_simple.obj"), rigid(p[:, :3], p[:, 3])) for i, p in zip(ids, P)]
        out[k] = (bool(len(P) == len(ids) and all(np.abs(p).sum() > 0 for p in P)), objs)
    return out


def ho3d_objects(seq, keys, zpath=None):
    from scipy.spatial.transform import Rotation

    from train.build_ho3d_crops import load_meta

    z = zipfile.ZipFile(zpath or os.path.join(D, "ho3d_raw", "HO3D_v3.zip"))
    root = "" if "train.txt" in z.namelist() else z.namelist()[0].split("/")[0] + "/"
    gl = np.diag([1.0, -1.0, -1.0, 1.0])
    out = {}
    for k in keys:
        a = load_meta(z, root, seq, k)
        if a.get("objRot") is None or a.get("objName") is None:
            out[k] = (False, [])
            continue
        T = gl @ rigid(Rotation.from_rotvec(np.asarray(a["objRot"], np.float64).reshape(3)).as_matrix(), a["objTrans"])
        path = os.path.join(YCB_MODELS, a["objName"], "textured_simple.obj")
        out[k] = (os.path.exists(path), [(a["objName"], path, T)] if os.path.exists(path) else [])
    return out


def oakink2_objects(seq, keys):
    """keys = (view, fid)."""
    from train.build_oakink2_crops import load_anno

    anno = load_anno(os.path.join(D, "oakink2_raw", "anno_preview", f"{seq}.pkl"))
    out = {}
    for view, fid in keys:
        E = np.asarray(anno["cam_extr"][view][fid], np.float64)
        objs, ok = [], True
        for o in anno["obj_list"]:
            path, T = os.path.join(OAK_MODELS, o, "model.obj"), anno["obj_transf"].get(o, {}).get(fid)
            if T is None or not os.path.exists(path):
                ok = False
                continue
            objs.append((o, path, E @ np.asarray(T, np.float64)))
        out[(view, fid)] = (ok and bool(objs), objs)
    return out


def group_of(pool, r):
    """(group, frame key) of a pool record."""
    if pool.startswith("hot3d"):
        return r["file"].split("/")[0], r["src"].split(":")[1]
    if pool.startswith("dexycb"):
        return r["file"].split("/")[0], r["src"].split("/")[-1]
    if pool.startswith("ho3d"):
        return tuple(r["src"].split(":"))
    if pool.startswith("oakink2"):
        seq, rest = r["src"].split(":")
        view, fid = rest.split("/")
        return seq, (view, int(fid))
    raise ValueError(pool)


OBJECTS = dict(hot3d=hot3d_objects, dexycb=dexycb_objects, ho3d=ho3d_objects, oakink2=oakink2_objects)


def _work(arg):
    pool, crops, part, group, items = arg  # items: [(row, record)]
    if os.path.exists(part):
        return part
    keys = sorted({group_of(pool, r)[1] for _, r in items})
    frames = OBJECTS[next(k for k in OBJECTS if pool.startswith(k))](group, keys)
    rows, sdf, meta = [], [], []
    for row, r in items:
        v = np.load(os.path.join(crops, pool, r["verts_file"])).astype(np.float64)
        if not r["is_right"]:  # pool verts are mirrored into the right-hand frame
            v[:, 0] *= -1
        valid, objs = (False, []) if frames is None else frames[group_of(pool, r)[1]]
        d, o = label(v, objs) if objs else (np.full(778, CLIP_MM), None)
        rows.append(row)
        sdf.append(d.astype(np.float16))
        meta.append(dict(valid=bool(valid), min_mm=round(float(d.min()), 2), n10=int((d < THRESH_MM).sum()), obj=o))
    tmp = part + ".tmp.npz"
    np.savez(tmp, rows=np.asarray(rows), sdf=np.stack(sdf), meta=json.dumps(meta))
    os.replace(tmp, part)
    return part


def label_pool(pool, crops, out, workers):
    recs = [json.loads(l) for l in open(os.path.join(crops, pool, "index.jsonl"))]
    dst = os.path.join(out, pool)
    os.makedirs(os.path.join(dst, "parts"), exist_ok=True)
    groups = defaultdict(list)
    for i, r in enumerate(recs):
        if "verts_file" in r:
            groups[group_of(pool, r)[0]].append((i, r))
    jobs = [(pool, crops, os.path.join(dst, "parts", f"{g}.npz"), g, it) for g, it in sorted(groups.items())]
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_work, jobs)):
            if k % 50 == 0:
                print(f"{pool}: {k}/{len(jobs)} groups", flush=True)
    sdf = np.full((len(recs), 778), CLIP_MM, np.float16)
    meta = [dict(valid=False, min_mm=CLIP_MM, n10=0, obj=None)] * len(recs)
    for j in jobs:
        z = np.load(j[2])
        sdf[z["rows"]] = z["sdf"]
        for row, m in zip(z["rows"].tolist(), json.loads(str(z["meta"]))):
            meta[row] = m
    np.save(os.path.join(dst, "sdf.tmp.npy"), sdf)
    os.replace(os.path.join(dst, "sdf.tmp.npy"), os.path.join(dst, "sdf.npy"))
    with open(os.path.join(dst, "contact.jsonl"), "w") as f:
        for r, m in zip(recs, meta):
            f.write(json.dumps(dict(file=r["file"], src=r.get("src"), is_right=r["is_right"], **m)) + "\n")
    v = [m for m in meta if m["valid"]]
    n = np.asarray([m["n10"] for m in v])
    print(json.dumps(dict(pool=pool, crops=len(recs), valid=len(v), hands_in_contact=float((n >= MIN_VERTS).mean()) if len(v) else None,
                          contact_vertex_frac=float(n.mean() / 778) if len(v) else None,
                          penetrating_gt_5mm=float(np.mean([m["min_mm"] < -5 for m in v])) if len(v) else None)), flush=True)


def _frames_work(arg):
    name, path, dst, mano_dir = arg
    import open3d as o3d

    clip_name = os.path.basename(path)[:-4]
    part = os.path.join(dst, clip_name + ".npz")
    if os.path.exists(part):
        return part
    if name == "hot3d_bench":
        from bench.common import ManoLayer
        from bench.datasets.hot3d import Hot3DClip

        if "hot3d" not in _MESH:
            _MESH["hot3d"] = ManoLayer(mano_dir, num_pca_comps=15)
        clip = Hot3DClip(path, mano_dir, min_visible=0.0, mano=_MESH["hot3d"])
        objs = hot3d_objects(clip_name, clip.frame_keys)
        frames = (clip.frame(i) for i in range(len(clip)))
    else:
        from bench.datasets.dexycb import DexYCBClip, DexYCBMano

        if "dexycb" not in _MESH:
            _MESH["dexycb"] = DexYCBMano(mano_dir)
        clip = DexYCBClip(path, min_visible=0.0, mano=_MESH["dexycb"])
        objs = dexycb_objects(clip_name, clip.frame_keys)
        frames = (clip.frame(i, image=False) for i in range(len(clip)))
    faces = (_MESH["hot3d"] if name == "hot3d_bench" else _MESH["dexycb"].layer).faces
    keys, side, sdf, vis = [], [], [], []
    for fr in frames:
        H, W = (clip.size, clip.size) if name == "hot3d_bench" else clip.size
        valid, ob = objs[fr["key"]]
        for h in fr["hands"] + fr["ignore"]:
            v = np.asarray(h["verts3d"], np.float64)
            d, _ = label(v, ob) if ob else (np.full(778, CLIP_MM), None)
            sc = o3d.t.geometry.RaycastingScene(nthreads=1)
            sc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(faces[bool(h["is_right"])].astype(np.uint32)))
            for _, mp, T in ob:
                _, V, F = load_mesh(mp)
                sc.add_triangles(o3d.core.Tensor((V @ T[:3, :3].T + T[:3, 3]).astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
            dist = np.linalg.norm(v, axis=1)
            rays = np.concatenate([np.zeros_like(v), v / dist[:, None]], 1).astype(np.float32)
            t = sc.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            K = fr["K"]
            uv = v[:, :2] / v[:, 2:] * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
            keys.append(fr["key"])
            side.append(bool(h["is_right"]))
            sdf.append(d.astype(np.float16))
            vis.append((t >= dist - 0.002) & inside)
    tmp = part + ".tmp.npz"
    np.savez(tmp, key=np.asarray(keys), is_right=np.asarray(side), sdf=np.stack(sdf) if sdf else np.zeros((0, 778), np.float16),
             vis=np.stack(vis) if vis else np.zeros((0, 778), bool))
    os.replace(tmp, part)
    return part


def make_root(dst, labels, sources=ROOT_SOURCES, hoi4d="hoi4d", max_pen=15.0):
    """The training label root: symlinks to <labels>/<source> + a copy of the HOI4D labels with Chair sequences and GT
    penetration deeper than max_pen mm marked invalid (row order unchanged; sdf.npy linked)."""
    from collections import Counter

    os.makedirs(dst, exist_ok=True)
    for s in sources.split(","):
        if os.path.exists(os.path.join(labels, s)) and not os.path.lexists(os.path.join(dst, s)):
            os.symlink(os.path.abspath(os.path.join(labels, s)), os.path.join(dst, s))
    src = os.path.join(labels, hoi4d)
    if not os.path.exists(os.path.join(src, "contact.jsonl")):
        return
    d = os.path.join(dst, hoi4d)
    os.makedirs(d, exist_ok=True)
    for f in ("sdf.npy", "stats.json", "channels.json"):
        if os.path.exists(os.path.join(src, f)) and not os.path.lexists(os.path.join(d, f)):
            os.symlink(os.path.abspath(os.path.join(src, f)), os.path.join(d, f))
    rows, cnt = [], Counter()
    for line in open(os.path.join(src, "contact.jsonl")):
        q = json.loads(line)
        why = "chair" if q["obj"] == "Chair" else ("pen15" if q["min_mm"] < -max_pen else ("invalid" if not q["valid"] else "kept"))
        cnt[why] += 1
        if why != "kept":
            q["valid"], q["dropped"] = False, why
        rows.append(json.dumps(q))
    with open(os.path.join(d, "contact.jsonl.tmp"), "w") as f:
        f.write("\n".join(rows) + "\n")
    os.replace(os.path.join(d, "contact.jsonl.tmp"), os.path.join(d, "contact.jsonl"))
    json.dump(dict(cnt), open(os.path.join(d, "filter.json"), "w"))
    print(dict(cnt), sorted(os.listdir(dst)), flush=True)


def build_frames(name, out, workers, mano_dir):
    dst = os.path.join(out, "test", name + "_frames")
    os.makedirs(dst, exist_ok=True)
    if name == "hot3d_bench":
        clips = [os.path.join(D, "hot3d", c) for c in json.load(open("bench/splits/hot3d_bench.json"))["clips"]["Aria"]]
    else:
        clips = [os.path.join(D, "dexycb", c) for c in json.load(open("bench/splits/dexycb_s0_test_full.json"))["clips"]]
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_frames_work, [(name, c, dst, mano_dir) for c in clips])):
            if k % 50 == 0:
                print(f"{name} frames: {k}/{len(clips)} clips", flush=True)


def _video_gt_work(arg):
    path, dataset, dst = arg
    import pickle

    clip = os.path.basename(path)[:-4]
    part = os.path.join(dst, clip + ".npz")
    if os.path.exists(part):
        return part
    d = pickle.load(open(path, "rb"))
    keys = [fr["key"] for fr in d["frames"]]
    if dataset == "oakink2":
        objs = oakink2_objects(clip.rsplit("_", 1)[0], [("egocentric", int(k)) for k in keys])
        objs = {k: objs[("egocentric", int(k))] for k in keys}
    else:
        objs = OBJECTS[dataset](clip, keys)
    fi, kind, hi, side, sdf, valid = [], [], [], [], [], []
    for t, fr in enumerate(d["frames"]):
        ok, ob = (False, []) if objs is None else objs[fr["key"]]
        for kk, lst in enumerate((fr["gt"], fr.get("ignore", []))):
            for h_i, h in enumerate(lst):
                dd, _ = label(np.asarray(h["verts3d"], np.float64), ob) if ob else (np.full(778, CLIP_MM), None)
                fi.append(t), kind.append(kk), hi.append(h_i), side.append(bool(h["is_right"])), sdf.append(dd.astype(np.float16)), valid.append(bool(ok))
    tmp = part + ".tmp.npz"
    np.savez(tmp, frame=np.asarray(fi, np.int32), kind=np.asarray(kind, np.int8), hand=np.asarray(hi, np.int16), is_right=np.asarray(side),
             valid=np.asarray(valid), sdf=np.stack(sdf) if sdf else np.zeros((0, 778), np.float16))
    os.replace(tmp, part)
    return part


def build_video_gt(pred_dir, dataset, out, workers):
    dst = os.path.join(out, "video_gt", dataset)
    os.makedirs(dst, exist_ok=True)
    pk = sorted(glob.glob(os.path.join(pred_dir, "*.pkl")))
    with Pool(workers) as p:
        for k, _ in enumerate(p.imap_unordered(_video_gt_work, [(f, dataset, dst) for f in pk])):
            if k % 100 == 0:
                print(f"{pred_dir}: {k}/{len(pk)} clips", flush=True)


def build_test(name, out, workers, mano_dir):
    """Held-out crop set via the pool builders' own _work (identical crops / labels / mirroring)."""
    dst = os.path.join(out, "test", name)
    os.makedirs(dst, exist_ok=True)
    if name == "hot3d_bench":
        from train import build_hot3d_crops as B

        split = json.load(open("bench/splits/hot3d_bench.json"))
        jobs = [(os.path.join(D, "hot3d", c), dst, 5, mano_dir, True, True) for c in split["clips"]["Aria"]]
    elif name == "dexycb_s0test":
        from train import build_dexycb_crops as B

        jobs = [(os.path.join(D, "dexycb", c), dst, 8, mano_dir, True, True)
                for c in json.load(open("bench/splits/dexycb_s0_test_full.json"))["clips"]]
    else:
        raise ValueError(name)
    with Pool(workers) as p:
        recs = [r for rs in p.imap_unordered(B._work, jobs) for r in rs]
    with open(os.path.join(dst, "index.jsonl"), "w") as f:
        for r in sorted(recs, key=lambda r: r["file"]):
            f.write(json.dumps(r) + "\n")
    print(f"{name}: {len(recs)} crops from {len(jobs)} clips -> {dst}", flush=True)
    return os.path.join(out, "test")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", default="", help="crop pool to label (hot3d_lab_mesh, dexycb_lab, ho3d_lab, oakink2_ego_lab, oakink2_lab)")
    p.add_argument("--crops", default="", help="root holding <pool>/index.jsonl + *_v.npy")
    p.add_argument("--test", default="", help="build + label a held-out crop set: hot3d_bench | dexycb_s0test")
    p.add_argument("--frames", default="", help="per-frame GT (sdf + visibility) of a test set's clips: hot3d_bench | dexycb_s0test")
    p.add_argument("--video_gt", default="", help="bench.run prediction dir: per-frame GT contact of its GT hands")
    p.add_argument("--dataset", default="", help="with --video_gt: hot3d | dexycb | oakink2")
    p.add_argument("--make_root", default="", help="write the training label root (links + filtered HOI4D) to this dir")
    p.add_argument("--root_sources", default=ROOT_SOURCES, help="--make_root: label dirs linked into the root (+ hoi4d, filtered)")
    p.add_argument("--data_root", default=os.environ.get("DATA_ROOT", "data"), help="raw datasets (see configure)")
    p.add_argument("--out", required=True, help="the contact data root (labels at <out>/<pool>, test sets at <out>/test), e.g. $DATA_ROOT/contact")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=24)
    a = p.parse_args()
    configure(a.data_root)
    if a.make_root:
        make_root(a.make_root, a.out, a.root_sources)
    elif a.video_gt:
        build_video_gt(a.video_gt, a.dataset, a.out, a.workers)
    elif a.frames:
        build_frames(a.frames, a.out, a.workers, a.mano_dir)
    elif a.test:
        label_pool(a.test, build_test(a.test, a.out, a.workers, a.mano_dir), os.path.join(a.out, "test"), a.workers)
    else:
        label_pool(a.pool, a.crops, a.out, a.workers)
