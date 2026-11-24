"""HOI4D (Liu et al., CVPR 2022, https://hoi4d.github.io; CC BY-NC 4.0) contact data: an image crop pool, a held-out image test
set and converted video clips, labelled with the contact rule of train.contact_sdf (signed distance of each GT MANO vertex to
the GT object surface, inside negative; contact iff < 10 mm). The data is fetched by each user from HOI4D under its licence;
nothing of it is redistributed. Data files are only read, never executed; pickles go through a numpy-only unpickler.

Inputs (--raw, the HOI4D downloads; each .zip may also be given unzipped as the directory of the same name):
  HOI4D_Hand_pose.zip        Hand_pose/handpose_{left,right}_hand/<seq>/<frame>.pickle: poseCoeff (3 global + 45 axis-angle),
                             beta (10), trans (camera frame, m), kps2D (21 px).
  HOI4D_annotations.zip      HOI4D_annotations/<seq>/objpose/<frame>.json: dataList[0] center (m) + rotation (Euler 'XYZ') =
                             CAD -> camera; <seq>/action/color.json: the capture duration.
  HOI4D_CAD_Model_for_release/  (unzipped) .../rigid/<Category>/<instance:03d>.obj, metres.
  camera_params.zip          camera_params/<camera>/intrin.npy (3 x 3, 1920 x 1080, no distortion).
  HOI4D_release.zip          HOI4D_release/<seq>/align_rgb/image.mp4: 300 frames; decoded frame i = annotation frame i.
<seq> = ZY2021080000<camera>/H<subject>/C<category>/N<instance>/S<room>/s<layout>/T<task>. Rigid categories with hand pose
and object pose: ToyCar (C1), Mug (C2), Bottle (C5), Bowl (C7), Kettle (C12), Knife (C13), Chair (C20).

Hand convention (checked on the images): HOI4D's fits use manopth with flat_hand_mean=True, 45 axis-angle, and the left hand
of MANO_LEFT.pkl as shipped (no smplx left-shapedirs fix). This convention reproduces HOI4D's kps2D (0.8 px median, both
sides), and HOI4D's own 2D hand masks agree with it: 86 % of the projected left-hand silhouette falls inside the left-hand
mask, vs 59 % for "mean pose + shapedirs fix", which moves left hands 36 mm and flips ~20 % of their contact labels.
Keep it. Left hands occur only in Chair sequences (two-hand chair lifting).

Split (bench/splits/hoi4d_contact.json): test = every sequence of subject H3; train = H1 / H2 / H4. Chair (C20) GT is
unreliable (frequent hand-object penetration). The contact training recipe drops Chair crops and crops with > 15 mm GT
penetration from the pool (train.contact_head's label root), and HOI4D test is reported without Chair
(bench/splits/hoi4d_contact_test_nochair.json: 149 of the 176 test sequences / clips).

Outputs (--out, the contact data root, default $CONTACT_ROOT or data/contact):
  pools/hoi4d/               train crops (train.build_crops.render_crop, 640 px, left hands mirrored), index.jsonl with file,
                             src "<seq>:<frame>", is_right, box, M, hw, K, kp2d, joints3d (HOT3D tips), verts_file (right-hand
                             frame, float16), sequence_id, visible, obj
  hoi4d/                     sdf.npy (N x 778 float16 mm) + contact.jsonl (pool index order)
  test/hoi4d_test/           test crops + index.jsonl + sdf.npy + contact.jsonl (same builder)
  test/hoi4d_test_frames/    per-frame GT of the test clips (key, is_right, sdf, vis: z-buffer over hand + object, 2 mm)
  video/hoi4d/clips/<clip>/  image.mp4 (copied from HOI4D_release) + gt.npz + meta.json (bench.datasets.contactclip)
  video_gt/hoi4d/<clip>.npz  per-frame GT contact through the loader (frame, kind 0 gt / 1 ignore, hand, is_right, valid, sdf)

    python -m train.build_hoi4d_contact inventory --raw $DATA_ROOT/hoi4d          # (re)writes the split file
    python -m train.build_hoi4d_contact check --raw $DATA_ROOT/hoi4d --n 20 --out /tmp/hoi4d_check
    python -m train.build_hoi4d_contact pool --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split train --stride 4
    python -m train.build_hoi4d_contact pool --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split test --stride 5
    python -m train.build_hoi4d_contact video --raw $DATA_ROOT/hoi4d --out $DATA_ROOT/contact --split train,test
"""
import argparse
import io
import json
import os
import pickle
import tempfile
import zipfile
import zlib
from collections import defaultdict
from multiprocessing import Pool

import cv2
import numpy as np

OFF = os.environ.get("HOI4D_ROOT", os.path.join("data", "hoi4d"))  # --raw (configure)
C = os.environ.get("CONTACT_ROOT", os.path.join("data", "contact"))  # --out (configure)
SPLIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "splits", "hoi4d_contact.json")
RIGID = {1: "ToyCar", 2: "Mug", 5: "Bottle", 7: "Bowl", 12: "Kettle", 13: "Knife", 20: "Chair"}
TEST_SUBJECTS = ("H3",)
SIDES = (("right", True), ("left", False))
MANOPTH_TIPS = [745, 317, 444, 556, 673]  # manopth fingertip vertices (HOI4D's kps2D)
_G = {}


class NumpyUnpickler(pickle.Unpickler):
    OK = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"), ("numpy", "ndarray"),
          ("numpy", "dtype"), ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar")}

    def find_class(self, module, name):
        if (module, name) in self.OK:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked {module}.{name}")


ZIPS = {"hand": "HOI4D_Hand_pose.zip", "ann": "HOI4D_annotations.zip", "cam": "camera_params.zip", "rel": "HOI4D_release.zip"}
UNZIPPED = {"hand": "HOI4D_Hand_pose", "ann": "HOI4D_annotations"}


def configure(raw="", out=""):
    """The input / output roots (before any worker forks: workers inherit them)."""
    global OFF, C
    OFF, C = raw or OFF, out or C


def zf(name):
    """One ZipFile per process (never shared across a fork: workers open their own). Member lists are always read from the
    zips (the index); member bytes from the unzipped copies when present (read)."""
    if name not in _G:
        _G[name] = zipfile.ZipFile(os.path.join(OFF, ZIPS[name]))
    return _G[name]


def read(kind, member):
    """Bytes of a zip member, from the unzipped copy (<raw>/<part>/, same member path) when it exists."""
    d = UNZIPPED.get(kind)
    if d and os.path.isfile(os.path.join(OFF, d, member)):
        with open(os.path.join(OFF, d, member), "rb") as f:
            return f.read()
    return zf(kind).read(member)


def drop_zips():
    """Before forking workers: they reopen the zips themselves."""
    for k in ("hand", "ann", "cam", "rel"):
        _G.pop(k, None)


def index():
    """{seq: {"hand": {side: {frame: member}}, "obj": {frame: member}}} over the zips' member lists (main process only;
    jobs carry their sequence's entry)."""
    if "index" not in _G:
        idx = defaultdict(lambda: dict(hand=defaultdict(dict), obj={}))
        for n in zf("hand").namelist():
            p = n.split("/")
            if n.endswith(".pickle") and len(p) == 10:
                idx["/".join(p[2:9])]["hand"][p[1].split("_")[1]][int(p[-1][:-7])] = n
        for n in zf("ann").namelist():
            p = n.split("/")
            if len(p) == 10 and p[8] == "objpose" and n.endswith(".json"):
                idx["/".join(p[1:8])]["obj"][int(p[-1][:-5])] = n
        _G["index"] = {s: dict(hand={k: dict(v) for k, v in d["hand"].items()}, obj=d["obj"]) for s, d in idx.items()}
    return _G["index"]


def rigid_sequences():
    rel = {n.split("/", 1)[1].rsplit("/", 2)[0] for n in zf("rel").namelist() if n.endswith("align_rgb/image.mp4")}
    return sorted(s for s, d in index().items() if d["hand"] and d["obj"] and s in rel and int(s.split("/")[2][1:]) in RIGID)


def intrinsics(seq):
    return np.load(io.BytesIO(zf("cam").read(f"camera_params/{seq.split('/')[0]}/intrin.npy")), allow_pickle=False).astype(np.float64)


def cad_path(seq):
    c, n = seq.split("/")[2:4]
    rel = os.path.join("rigid", RIGID[int(c[1:])], f"{int(n[1:]):03d}.obj")
    for d in (os.path.join(OFF, "HOI4D_CAD_Model_for_release", "HOI4D_CAD_Model_for_release"), os.path.join(OFF, "HOI4D_CAD_Model_for_release")):
        if os.path.exists(os.path.join(d, rel)):  # the zip extracts to a nested directory of the same name
            return os.path.join(d, rel)
    return os.path.join(OFF, "HOI4D_CAD_Model_for_release", "HOI4D_CAD_Model_for_release", rel)


def object_sdf(seq):
    from train.contact_sdf import MeshSDF

    p = cad_path(seq)
    if ("mesh", p) not in _G:
        import trimesh

        m = trimesh.load(p, force="mesh", process=True)
        _G[("mesh", p)] = MeshSDF(np.asarray(m.vertices), np.asarray(m.faces))
    return _G[("mesh", p)]


def object_pose(member):
    from scipy.spatial.transform import Rotation

    j = json.loads(read("ann", member))
    o = (j.get("dataList") or j.get("objects"))[0]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("XYZ", [float(o["rotation"][k]) for k in "xyz"]).as_matrix()
    T[:3, 3] = [float(o["center"][k]) for k in "xyz"]
    return T


def hand_params(member):
    d = NumpyUnpickler(io.BytesIO(read("hand", member)), encoding="latin1").load()
    p, b, t = (np.asarray(d.get(k), np.float64).reshape(-1) if d.get(k) is not None else np.zeros(0) for k in ("poseCoeff", "beta", "trans"))
    ok = p.size == 48 and b.size == 10 and t.size == 3 and np.isfinite(np.r_[p, b, t]).all()
    kp = np.asarray(d.get("kps2D", np.zeros((0, 2))), np.float64).reshape(-1, 2)
    return ok, p, b, t, kp


class HOI4DMano:
    """manopth convention (flat_hand_mean, 45 axis-angle, left shapedirs as shipped) on bench.common.ManoLayer."""

    def __init__(self, mano_dir):
        import torch

        from bench.common import ManoLayer

        self.layer = ManoLayer(mano_dir, flat_hand_mean=True)
        sd = pickle.load(open(os.path.join(mano_dir, "MANO_LEFT.pkl"), "rb"), encoding="latin1")["shapedirs"]  # MANO model file
        self.layer.layers[False].shapedirs[:] = torch.as_tensor(np.asarray(sd, np.float32))

    def __call__(self, is_right, pose, beta, trans):
        j, v = self.layer(is_right, beta, pose[3:], pose[:3], trans)
        return j.astype(np.float64), v.astype(np.float64)


def mano(mano_dir):
    if "mano" not in _G:
        _G["mano"] = HOI4DMano(mano_dir)
    return _G["mano"]


def frames_of(info):
    """Sorted frames with any hand annotation."""
    return sorted({f for s in info["hand"].values() for f in s})


def proj(X, K):
    return X[:, :2] / np.clip(X[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]


def hands_at(info, f, K, H, W, mano_dir):
    """GT hands of frame f (right first): list of dict(is_right, ok, verts3d, joints3d, joints2d, inside, kp_err, mano)."""
    out = []
    for side, is_right in SIDES:
        m = info["hand"].get(side, {}).get(f)
        if m is None:
            continue
        ok, p, b, t, kp = hand_params(m)
        if not ok:
            out.append(dict(is_right=is_right, ok=False))
            continue
        j, v = mano(mano_dir)(is_right, p, b, t)
        uv = proj(v, K)
        inside = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        jt = j.copy()
        jt[[4, 8, 12, 16, 20]] = v[MANOPTH_TIPS]  # manopth tips, for the kps2D check only
        kp_err = float(np.median(np.linalg.norm(proj(jt, K) - kp, axis=1))) if kp.shape == (21, 2) else None
        out.append(dict(is_right=is_right, ok=True, verts3d=v, joints3d=j, joints2d=proj(j, K), inside=inside, kp_err=kp_err,
                        mano=np.r_[p, b, t]))
    return out


class Video:
    """Sequential frame reader of a sequence's image.mp4 (decoded frame i = annotation frame i)."""

    def __init__(self, seq):
        fd, self.path = tempfile.mkstemp(suffix=".mp4", dir="/dev/shm" if os.path.isdir("/dev/shm") else None)
        with os.fdopen(fd, "wb") as f:
            f.write(zf("rel").read(f"HOI4D_release/{seq}/align_rgb/image.mp4"))
        self.cap, self.pos = cv2.VideoCapture(self.path), 0

    def get(self, i):
        if i < self.pos:
            self.cap.release()
            self.cap, self.pos = cv2.VideoCapture(self.path), 0
        img = None
        while self.pos <= i:
            ok, img = self.cap.read()
            self.pos += 1
            if not ok:
                return None
        return img

    def close(self):
        self.cap.release()
        os.remove(self.path)


def capture_fps(seq):
    """300 frames over action/color.json info.duration: 20 s (15 fps) for 1,252 of the 1,327 rigid sequences, 10 s for 69
    (-> 30 fps; their per-frame hand motion is 0.65 x), missing for 6 (-> 15). Every mp4 header says 15 fps."""
    j = json.loads(read("ann", f"HOI4D_annotations/{seq}/action/color.json"))
    d = j.get("info", {}).get("duration")
    return round(300 / d) if d else 15


def split_of(seq):
    return "test" if seq.split("/")[1] in TEST_SUBJECTS else "train"


def seq_name(seq):
    return seq.replace("/", "_")


def mkdir(d):
    try:
        os.makedirs(d, exist_ok=True)
    except FileExistsError:
        pass


# ---------------------------------------------------------------------------- inventory / split

def cmd_inventory(a):
    seqs = rigid_sequences()
    by = defaultdict(list)
    for s in seqs:
        by[split_of(s)].append(s)
    stats = {}
    for k, v in by.items():
        stats[k] = dict(sequences=len(v), hand_frames=sum(len(fr) for s in v for fr in index()[s]["hand"].values()),
                        both_hands=sum(len(index()[s]["hand"]) == 2 for s in v),
                        categories={RIGID[c]: sum(int(s.split("/")[2][1:]) == c for s in v) for c in RIGID})
    split = dict(
        name="hoi4d_contact",
        definition="HOI4D rigid-object sequences (ToyCar C1, Mug C2, Bottle C5, Bowl C7, Kettle C12, Knife C13, Chair C20) with "
                   "MANO hand pose (Hand_pose) + object pose (objpose) + RGB (HOI4D_release align_rgb/image.mp4). Test = every "
                   "sequence of subject H3 (= camera ZY20210800003); train = H1 / H2 / H4. Held out of every training set. "
                   "Object instances are shared across subjects (unseen subject, not unseen objects). HACO trained on the HOI4D "
                   "release, so HOI4D test is in-domain (seen sequences) for HACO. Licence CC BY-NC 4.0.",
        test_subjects=list(TEST_SUBJECTS), stats=stats, train=by["train"], test=by["test"])
    with open(SPLIT, "w") as f:
        json.dump(split, f, indent=1)
    print(json.dumps(stats, indent=1))


def load_split(name, only=""):
    """The split's sequences; only: a comma-separated subset (e.g. sample sequences for a parity check)."""
    s = json.load(open(SPLIT))
    out = []
    for k in name.split(","):
        out += s[k]
    if only:
        keep = set(only.split(","))
        out = [x for x in out if x in keep or seq_name(x) in keep]
    return out


def jobs_for(seqs, *rest):
    idx = index()
    jobs = [(s, idx[s], *rest) for s in seqs]
    drop_zips()
    return jobs


# ---------------------------------------------------------------------------- image pool

def _pool_work(arg):
    seq, info, out, stride, mano_dir = arg
    import torch

    from train.build_crops import crop_keypoints, render_crop

    torch.set_num_threads(1)
    done = os.path.join(out, seq_name(seq), ".done.json")
    if os.path.exists(done):
        return json.load(open(done))
    K = intrinsics(seq)
    ph = zlib.crc32(seq.encode()) % stride
    fr = [f for f in frames_of(info) if f % stride == ph]
    sdf_of = object_sdf(seq)
    obj_name = RIGID[int(seq.split("/")[2][1:])]
    vid = Video(seq)
    mkdir(os.path.join(out, seq_name(seq)))
    recs = []
    try:
        for f in fr:
            img = vid.get(f)
            if img is None:
                continue
            H, W = img.shape[:2]
            T = object_pose(info["obj"][f]) if f in info["obj"] else None
            for h in hands_at(info, f, K, H, W, mano_dir):
                if not h["ok"] or h["inside"].mean() < 0.3:
                    continue
                uv = proj(h["verts3d"], K)[h["inside"]]
                b = np.array([*uv.min(0), *uv.max(0)], np.float32)
                if max(b[2] - b[0], b[3] - b[1]) < 32:
                    continue
                is_right = h["is_right"]
                crop, M = render_crop(img, b, is_right)
                rel = f"{seq_name(seq)}/{f:05d}_{'r' if is_right else 'l'}.jpg"
                cv2.imwrite(os.path.join(out, rel), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                j2 = h["joints2d"]
                ins = (j2[:, 0] >= 0) & (j2[:, 0] < W) & (j2[:, 1] >= 0) & (j2[:, 1] < H)
                Km, v3, j3 = K.copy(), h["verts3d"].copy(), h["joints3d"].copy()
                if not is_right:  # right-hand (mirrored) frame, as every pool
                    Km[0, 2] = W - 1 - Km[0, 2]
                    v3[:, 0] *= -1
                    j3[:, 0] *= -1
                np.save(os.path.join(out, rel[:-4] + "_v.npy"), v3.astype(np.float16))
                sd = sdf_of.posed(h["verts3d"], T) if T is not None else np.full(778, 500.0)
                recs.append(dict(file=rel, src=f"{seq}:{f:05d}", is_right=bool(is_right), box=[float(x) for x in b], M=M.round(5).tolist(),
                                 hw=[H, W], K=Km.round(4).tolist(), kp2d=crop_keypoints(np.concatenate([j2, ins[:, None]], 1), M, is_right, W).round(2).tolist(),
                                 joints3d=j3.round(5).tolist(), sequence_id=seq, verts_file=rel[:-4] + "_v.npy", visible=float(h["inside"].mean()),
                                 obj=obj_name, valid=T is not None, kp_err=h["kp_err"], sdf=sd.astype(np.float16).tolist()))
    finally:
        vid.close()
    json.dump(recs, open(done, "w"))
    return recs


def write_labels(recs, lab):
    mkdir(lab)
    sdf = np.asarray([r.pop("sdf") for r in recs], np.float16).reshape(-1, 778)
    np.save(os.path.join(lab, "sdf.tmp.npy"), sdf)
    os.replace(os.path.join(lab, "sdf.tmp.npy"), os.path.join(lab, "sdf.npy"))
    with open(os.path.join(lab, "contact.jsonl"), "w") as f:
        for r, d in zip(recs, sdf.astype(np.float32)):
            f.write(json.dumps(dict(file=r["file"], src=r["src"], is_right=r["is_right"], valid=bool(r["valid"]), min_mm=round(float(d.min()), 2),
                                    n10=int((d < 10).sum()), obj=r["obj"])) + "\n")
    return label_stats(sdf, [r["valid"] for r in recs])


def label_stats(sdf, valid):
    v = np.asarray(sdf, np.float32)[np.asarray(valid, bool)]
    n = (v < 10).sum(1)
    return dict(crops=len(sdf), valid=len(v), hands_in_contact=float((n > 0).mean()) if len(v) else None,
                contact_vertex_frac=float(n.mean() / 778) if len(v) else None,
                penetrating_gt_5mm=float((v.min(1) < -5).mean()) if len(v) else None,
                penetrating_gt_25mm=float((v.min(1) < -25).mean()) if len(v) else None,
                penetrating_gt_50mm=float((v.min(1) < -50).mean()) if len(v) else None)


def cmd_pool(a):
    seqs = load_split(a.split, a.sequences)
    test = a.split == "test"
    if test:
        out = a.out or os.path.join(C, "test", "hoi4d_test")
        lab = a.labels or out
    else:
        out = a.out or os.path.join(C, "pools", "hoi4d")
        lab = a.labels or os.path.join(C, "hoi4d")
        assert not set(load_split("test")) & set(seqs), "test sequence in the training pool"
    mkdir(out)
    jobs = jobs_for(seqs, out, a.stride, a.mano_dir)
    recs = []
    with Pool(a.workers) as p:
        for k, rs in enumerate(p.imap_unordered(_pool_work, jobs)):
            recs += rs
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} sequences, {len(recs)} crops", flush=True)
    recs.sort(key=lambda r: r["file"])
    stats = write_labels(recs, lab)
    with open(os.path.join(out, "index.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    kp = [r["kp_err"] for r in recs if r.get("kp_err") is not None]
    stats.update(split=a.split, sequences=len(seqs), stride=a.stride, kps2d_err_px_median=float(np.median(kp)) if kp else None,
                 kps2d_err_px_p95=float(np.percentile(kp, 95)) if kp else None, left_share=float(np.mean([not r["is_right"] for r in recs])))
    sdf = np.load(os.path.join(lab, "sdf.npy"))
    objs = np.asarray([r["obj"] for r in recs])
    stats["per_object"] = {o: label_stats(sdf[objs == o], [r["valid"] for r in recs if r["obj"] == o]) for o in sorted(set(objs))}
    json.dump(stats, open(os.path.join(lab, "stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


# ---------------------------------------------------------------------------- video clips + per-frame GT

def _video_work(arg):
    seq, info, root, gt_dir, frames_dir, mano_dir = arg
    import torch

    torch.set_num_threads(1)
    name = seq_name(seq)
    d = os.path.join(root, "clips", name)
    if os.path.exists(os.path.join(d, "meta.json")) and os.path.exists(os.path.join(gt_dir, name + ".npz")):
        return name
    mkdir(d)
    K = intrinsics(seq)
    H, W = 1080, 1920
    fr = frames_of(info)
    f0, f1 = fr[0], fr[-1]
    sdf_of = object_sdf(seq)
    rows = defaultdict(list)
    for f in range(f0, f1 + 1):
        T = object_pose(info["obj"][f]) if f in info["obj"] else None
        for h in hands_at(info, f, K, H, W, mano_dir):
            if not h["ok"]:
                continue
            sd = sdf_of.posed(h["verts3d"], T) if T is not None else np.full(778, 500.0)
            for k, v in (("frame", f), ("is_right", h["is_right"]), ("verts", h["verts3d"].astype(np.float32)), ("joints", h["joints3d"].astype(np.float32)),
                         ("mano", h["mano"].astype(np.float32)), ("sdf", sd.astype(np.float16)), ("valid", T is not None),
                         ("T_obj", (T if T is not None else np.eye(4)).astype(np.float32))):
                rows[k].append(v)
    gt = {k: np.asarray(v) for k, v in rows.items()}
    tmp = os.path.join(d, "gt.tmp.npz")
    np.savez(tmp, **gt)
    os.replace(tmp, os.path.join(d, "gt.npz"))
    with open(os.path.join(d, "image.tmp.mp4"), "wb") as f:
        f.write(zf("rel").read(f"HOI4D_release/{seq}/align_rgb/image.mp4"))
    os.replace(os.path.join(d, "image.tmp.mp4"), os.path.join(d, "image.mp4"))
    meta = dict(dataset="hoi4d", sequence=seq, K=K.round(5).tolist(), hw=[H, W], fps=capture_fps(seq), fps_mp4=15, first=int(f0), last=int(f1), video="image.mp4",
                obj=RIGID[int(seq.split("/")[2][1:])], cad=os.path.relpath(cad_path(seq), OFF), split=split_of(seq), licence="CC BY-NC 4.0")
    json.dump(meta, open(os.path.join(d, "meta.tmp.json"), "w"))
    os.replace(os.path.join(d, "meta.tmp.json"), os.path.join(d, "meta.json"))
    video_gt(d, os.path.join(gt_dir, name + ".npz"), frames_dir and os.path.join(frames_dir, name + ".npz"), object_sdf(seq), mano(mano_dir).layer.faces)
    return name


def video_gt(clip_dir, dst, frames_dst=None, obj_mesh=None, faces=None):
    """Per-frame GT through the loader itself (the same hand lists / order bench.run stores in its pickles); with frames_dst
    also the test categories' per-frame file (key, is_right, sdf, vis: z-buffer over hand + object, 2 mm, inside the image).
    obj_mesh = the clip's train.contact_sdf.MeshSDF (its raycasting scene, object frame)."""
    import open3d as o3d

    from bench.datasets.contactclip import ContactClip

    clip = ContactClip(clip_dir)
    fi, kind, hi, side, sdf, valid = [], [], [], [], [], []
    fkey, fside, fsdf, fvis = [], [], [], []
    H, W = clip.size
    for t in range(len(clip)):
        fr = clip.frame(t, image=False)
        for kk, lst in enumerate((fr["hands"], fr["ignore"])):
            for h_i, h in enumerate(lst):
                fi.append(t), kind.append(kk), hi.append(h_i), side.append(bool(h["is_right"])), sdf.append(h["sdf"]), valid.append(bool(h["valid"]))
                if not frames_dst:
                    continue
                if "vis" in h:  # stored by the builder (ARCTIC: articulated object, scene per frame)
                    fkey.append(fr["key"]), fside.append(bool(h["is_right"])), fsdf.append(h["sdf"]), fvis.append(np.asarray(h["vis"], bool))
                    continue
                v = np.asarray(h["verts3d"], np.float64)
                sc = o3d.t.geometry.RaycastingScene(nthreads=1)
                sc.add_triangles(o3d.core.Tensor(v.astype(np.float32)), o3d.core.Tensor(faces[bool(h["is_right"])].astype(np.uint32)))
                dist = np.linalg.norm(v, axis=1)
                rays = np.concatenate([np.zeros_like(v), v / dist[:, None]], 1)
                th = sc.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))["t_hit"].numpy()
                if h["valid"] and obj_mesh is not None:  # the object's own scene (object frame), rays moved into it
                    T = np.asarray(h["T_obj"], np.float64)
                    lr = np.concatenate([np.broadcast_to(-T[:3, 3] @ T[:3, :3], v.shape), rays[:, 3:] @ T[:3, :3]], 1)
                    th = np.minimum(th, obj_mesh.sc.cast_rays(o3d.core.Tensor(lr.astype(np.float32)))["t_hit"].numpy())
                uv = proj(v, fr["K"])
                ins = (v[:, 2] > 0.01) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
                fkey.append(fr["key"]), fside.append(bool(h["is_right"])), fsdf.append(h["sdf"]), fvis.append((th >= dist - 0.002) & ins)
    tmp = dst + ".tmp.npz"
    np.savez(tmp, frame=np.asarray(fi, np.int32), kind=np.asarray(kind, np.int8), hand=np.asarray(hi, np.int16), is_right=np.asarray(side, bool),
             valid=np.asarray(valid, bool), sdf=np.stack(sdf).astype(np.float16) if sdf else np.zeros((0, 778), np.float16))
    os.replace(tmp, dst)
    if frames_dst:
        tmp = frames_dst + ".tmp.npz"
        np.savez(tmp, key=np.asarray(fkey), is_right=np.asarray(fside, bool), sdf=np.stack(fsdf).astype(np.float16) if fsdf else np.zeros((0, 778), np.float16),
                 vis=np.stack(fvis) if fvis else np.zeros((0, 778), bool))
        os.replace(tmp, frames_dst)


def video_stats(gt_dir, names):
    """Label stats of the video GT (gt hands, valid rows)."""
    sd, ok, n = [], [], 0
    for c in names:
        z = np.load(os.path.join(gt_dir, c + ".npz"))
        k = z["kind"] == 0
        sd.append(z["sdf"][k]), ok.append(z["valid"][k])
        n += len(np.unique(z["frame"]))
    st = label_stats(np.concatenate(sd), np.concatenate(ok))
    st["hand_frames"] = st.pop("crops")
    st.update(clips=len(names), frames_with_hands=n)
    return st


def cmd_video(a):
    seqs = load_split(a.split, a.sequences)
    root = a.out or os.path.join(C, "video", "hoi4d")
    gt_dir = os.path.join(C, "video_gt", "hoi4d")
    fdir = os.path.join(C, "test", "hoi4d_test_frames")
    for d in (os.path.join(root, "clips"), gt_dir, fdir):
        mkdir(d)
    jobs = jobs_for(seqs, root, gt_dir, None, a.mano_dir)
    jobs = [(s, i, r, g, fdir if split_of(s) == "test" else None, m) for s, i, r, g, _, m in jobs]
    with Pool(a.workers) as p:
        for k, _ in enumerate(p.imap_unordered(_video_work, jobs)):
            if k % 50 == 0:
                print(f"{k}/{len(jobs)} clips", flush=True)
    sp = defaultdict(list)
    for s in seqs:
        sp[split_of(s)].append(seq_name(s))
    stats = {}
    for k, v in sp.items():  # bench.run --dataset contactclip --split <this> --data_root <root>
        json.dump(dict(dataset="hoi4d", split=k, clips=sorted(v)), open(os.path.join(root, f"split_{k}.json"), "w"), indent=1)
        stats[k] = video_stats(gt_dir, v)
    json.dump(stats, open(os.path.join(root, "stats.json"), "w"), indent=1)
    print(json.dumps(stats), flush=True)


# ---------------------------------------------------------------------------- sanity sheet

def cmd_check(a):
    """Random frames: projected hand mesh (green right / orange left), object points (grey), contact vertices (red), HOI4D's
    kps2D (yellow crosses); kps2D residual of our MANO (manopth tips) per side, and left hands with / without the smplx
    left-shapedirs fix."""
    import torch

    from bench.common import ManoLayer

    torch.set_num_threads(4)
    rng = np.random.default_rng(a.seed)
    seqs = load_split(a.split)
    pick = [str(s) for s in rng.choice(seqs, a.n, replace=False)]
    if a.left:  # also frames of two-hand sequences
        two = [s for s in seqs if len(index()[s]["hand"]) == 2]
        pick += [str(s) for s in rng.choice(two, a.left, replace=False)]
    mkdir(a.out)
    fixed = ManoLayer(a.mano_dir, flat_hand_mean=True)
    errs, left_cmp, tiles = defaultdict(list), [], []
    for s in pick:
        info = index()[s]
        f = int(rng.choice(frames_of(info)))
        K = intrinsics(s)
        vid = Video(s)
        img = vid.get(f)
        vid.close()
        H, W = img.shape[:2]
        T = object_pose(info["obj"][f])
        m = object_sdf(s)
        P = m.V @ T[:3, :3].T + T[:3, 3]
        vis = img.copy()
        for u, v in proj(P[rng.choice(len(P), min(3000, len(P)), replace=False)], K).astype(int):
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(vis, (int(u), int(v)), 1, (180, 180, 180), -1)
        txt = []
        for h in hands_at(info, f, K, H, W, a.mano_dir):
            if not h["ok"]:
                txt.append("bad params")
                continue
            sd = m.posed(h["verts3d"], T)
            uv = proj(h["verts3d"], K).astype(int)
            col = (0, 200, 0) if h["is_right"] else (0, 140, 255)
            for (u, v), c in zip(uv, sd < 10):
                cv2.circle(vis, (int(u), int(v)), 2 if c else 1, (0, 0, 255) if c else col, -1)
            _, p, b, t, kp = hand_params(info["hand"]["right" if h["is_right"] else "left"][f])
            for u, v in kp.astype(int):
                cv2.drawMarker(vis, (int(u), int(v)), (0, 255, 255), cv2.MARKER_CROSS, 8, 1)
            if h["kp_err"] is not None:
                errs["right" if h["is_right"] else "left"].append(h["kp_err"])
            if not h["is_right"] and kp.shape == (21, 2):
                j2, v2 = fixed(False, b, p[3:], p[:3], t)
                jt = np.asarray(j2, np.float64).copy()
                jt[[4, 8, 12, 16, 20]] = v2[MANOPTH_TIPS]
                left_cmp.append((h["kp_err"], float(np.median(np.linalg.norm(proj(jt, K) - kp, axis=1)))))
            txt.append(f"{'R' if h['is_right'] else 'L'} min {sd.min():.0f}mm n10 {(sd < 10).sum()} kp {h['kp_err'] or -1:.1f}px")
        cv2.putText(vis, f"{s} f{f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        for k, l in enumerate(txt):
            cv2.putText(vis, l, (10, 60 + 30 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        out = os.path.join(a.out, f"{seq_name(s)}_{f:05d}.jpg")
        cv2.imwrite(out, cv2.resize(vis, (W // 2, H // 2)), [cv2.IMWRITE_JPEG_QUALITY, 85])
        tiles.append(out)
    res = dict(kp_err_px={k: [float(np.median(v)), float(np.max(v)), len(v)] for k, v in errs.items()},
               left_shipped_vs_fixed_kp_err=left_cmp, tiles=tiles)
    json.dump(res, open(os.path.join(a.out, "check.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "tiles"}, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["inventory", "check", "pool", "video"])
    p.add_argument("--split", default="train", help="train | test | train,test")
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--out", default="")
    p.add_argument("--labels", default="")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--left", type=int, default=0, help="check: extra frames from two-hand sequences")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=64)
    p.add_argument("--raw", default="", help="the HOI4D downloads (default $HOI4D_ROOT or data/hoi4d)")
    p.add_argument("--sequences", default="", help="pool / video: only these split sequences (comma-separated, '/' or '_' form)")
    a = p.parse_args()
    if a.cmd in ("pool", "video"):
        configure(a.raw, a.out)
        a.out = ""
    else:
        configure(a.raw)
    dict(inventory=cmd_inventory, check=cmd_check, pool=cmd_pool, video=cmd_video)[a.cmd](a)
