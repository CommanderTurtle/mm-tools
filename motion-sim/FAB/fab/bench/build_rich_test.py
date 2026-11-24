"""RICH test bench set (Huang et al., CVPR 2022): every --stride-th fitted frame of every test sequence, all its cameras,
half-resolution JPEGs, oracle boxes from the projected GT mesh.

Ground truth: the released SMPL-X fit mesh (test_body/<seq>/<frame>/<subject>.ply, world = scan frame) mapped to SMPL
topology by the official transfer matrix (<bm_dir>/smplx_transfer/smplx_to_smpl.pkl), no refit; posed in each camera
by scan_calibration/<scene>/calibration/<cam>.xml. Scored like any SMPL GT (H36M-14 / SMPL-24 regressors on the mesh).

  gt      selected frames -> <out>/gt_verts.npy (N, 6890, 3) float32 world + <out>/gt_rows.pkl [(seq, frame, subject)]
  frames  one pass over the release's JPG_images/test.tar.gz, selected frames of every camera at --scale -> <out>/images/
  index   per (row, camera): K (scaled), extrinsic, tight box of the projected mesh (people mostly out of frame / shorter
          than --min_height px dropped) -> <out>/index.pkl (bench/datasets.wds_test; eval_hmr reads gt_verts_file / row)

    python -m bench.build_rich_test gt --data $FAB_DATA/rich --out $FAB_DATA/rich_test --stride 30
    python -m bench.build_rich_test frames --archive $FAB_DATA/rich/JPG_images/test.tar.gz --out $FAB_DATA/rich_test
    python -m bench.build_rich_test index --data $FAB_DATA/rich --out $FAB_DATA/rich_test
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import pickle
import re
import tarfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from bench.testset_utils import _save_frame, read_camera, read_ply_vertices
from fab import paths


def stage_gt(a):
    with open(Path(a.bm_dir) / "smplx_transfer" / "smplx_to_smpl.pkl", "rb") as f:
        M = np.asarray(pickle.load(f, encoding="latin1")["matrix"], np.float32)  # (6890, 10475)
    import scipy.sparse as sp

    M = sp.csr_matrix(M)
    z = zipfile.ZipFile(Path(a.data) / "test_body.zip")
    names = set(z.namelist())
    items = defaultdict(list)
    for n in names:
        m = re.match(r"(?:test_body|test)/([^/]+)/(\d+)/(\d+)\.ply$", n)
        if m:
            items[m.group(1)].append((int(m.group(2)), int(m.group(3)), n))
    rows, verts = [], []
    for seq in sorted(items):
        frames = sorted({f for f, _, _ in items[seq]})
        keep = set(frames[:: a.stride])
        for fr, subj, n in sorted(items[seq]):
            if fr not in keep:
                continue
            v = read_ply_vertices(z.read(n)).astype(np.float64)
            verts.append((M @ v).astype(np.float32))
            rows.append((seq, fr, subj))
        print(f"{seq}: {len(frames)} fitted frames, kept {len(keep)}", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "gt_verts.npy", np.stack(verts))
    (out / "gt_rows.pkl").write_bytes(pickle.dumps(rows))
    print(f"{len(rows)} (frame, subject) GT meshes from {len(items)} sequences", flush=True)


def stage_frames(a):
    out = Path(a.out)
    want = defaultdict(set)
    for seq, fr, _ in pickle.loads((out / "gt_rows.pkl").read_bytes()):
        want[seq].add(fr)
    fileobj = open(a.archive, "rb")
    t0, n_seen, n_kept, sizes, pending = time.time(), 0, 0, {}, []
    with cf.ProcessPoolExecutor(a.procs) as ex, tarfile.open(fileobj=fileobj, mode="r|gz") as tf:
        for m in tf:
            n_seen += 1
            mm = re.match(r"test/([^/]+)/cam_(\d+)/(\d+)_\d+\.jpe?g$", m.name)
            if not m.isfile() or mm is None:
                continue
            seq, cam, fr = mm.group(1), mm.group(2), int(mm.group(3))
            if fr not in want.get(seq, ()):
                continue
            dst = out / "images" / seq / f"cam_{cam}" / f"{fr:05d}.jpg"
            if dst.exists():
                continue
            pending.append(((seq, cam), ex.submit(_save_frame, tf.extractfile(m).read(), str(dst), a.scale)))
            if len(pending) >= 4 * a.procs:
                for key, fut in pending[: 2 * a.procs]:
                    shape = fut.result()
                    if shape is not None:
                        sizes.setdefault(key, shape)
                        n_kept += 1
                pending = pending[2 * a.procs :]
                if n_kept % 2000 < 2 * a.procs:
                    print(f"{n_seen} members, {n_kept} kept, {time.time() - t0:.0f} s", flush=True)
        for key, fut in pending:
            shape = fut.result()
            if shape is not None:
                sizes.setdefault(key, shape)
                n_kept += 1
    meta = out / "image_sizes.pkl"
    old = pickle.loads(meta.read_bytes()) if meta.exists() else {}
    old.update({k: (int(h), int(w), a.scale) for k, (h, w) in sizes.items()})
    meta.write_bytes(pickle.dumps(old))
    print(f"done: {n_seen} members, {n_kept} kept, {time.time() - t0:.0f} s", flush=True)


def stage_index(a):
    out = Path(a.out)
    rows = pickle.loads((out / "gt_rows.pkl").read_bytes())
    verts = np.load(out / "gt_verts.npy", mmap_mode="r")
    sizes = pickle.loads((out / "image_sizes.pkl").read_bytes())
    row_of = {(s, f, j): i for i, (s, f, j) in enumerate(rows)}
    by_frame = defaultdict(list)
    for (s, f, j), i in row_of.items():
        by_frame[(s, f)].append((j, i))
    index, stats = [], defaultdict(int)
    for (seq, cam), (H0, W0, sc) in sorted(sizes.items()):
        scene = seq.split("_")[0]
        xml = Path(a.data) / "scan_calibration" / scene / "calibration" / f"{int(cam):03d}.xml"
        if not xml.exists():
            stats["no_calibration"] += 1
            continue
        R, t, K0, _ = read_camera(xml)
        K = K0.copy()
        K[:2] *= sc
        H, W = int(round(H0 * sc)), int(round(W0 * sc))
        ext = np.eye(4)
        ext[:3, :3], ext[:3, 3] = R, t
        for img in sorted((out / "images" / seq / f"cam_{cam}").glob("*.jpg")):
            fr = int(img.stem)
            for subj, i in sorted(by_frame.get((seq, fr), [])):
                vc = verts[i].astype(np.float64) @ R.T + t
                if (vc[:, 2] < 0.1).any():
                    stats["behind"] += 1
                    continue
                uv = vc @ K.T
                uv = uv[:, :2] / uv[:, 2:3]
                full = np.concatenate([uv.min(0), uv.max(0)])
                clip = np.array([np.clip(full[0], 0, W), np.clip(full[1], 0, H), np.clip(full[2], 0, W), np.clip(full[3], 0, H)])
                area = lambda b: max(b[2] - b[0], 0) * max(b[3] - b[1], 0)  # noqa: E731
                if area(clip) < a.min_visible * area(full) or clip[3] - clip[1] < a.min_height:
                    stats["out_of_frame_or_small"] += 1
                    continue
                index.append({"id": f"{seq}/cam_{cam}/{fr:05d}/{subj}", "image": str(img), "bbox": clip.astype(np.float32),
                              "K": K.astype(np.float32), "extrinsic": ext.astype(np.float32), "gender": "neutral",
                              "gt_verts_file": str(out / "gt_verts.npy"), "gt_verts_row": int(i)})
                stats["kept"] += 1
    with open(out / "index.pkl", "wb") as f:
        pickle.dump(index, f)
    stats.update(sequences=len({s["id"].split("/")[0] for s in index}), units=len(sizes))
    (out / "build_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(dict(stats), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=("gt", "frames", "index"))
    p.add_argument("--data", default=str(paths.home() / "data" / "rich"), help="the local RICH release folder")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=30, help="gt: every n-th fitted frame of a sequence")
    p.add_argument("--archive", default="", help="frames: the release's JPG_images/test.tar.gz")
    p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--procs", type=int, default=24)
    p.add_argument("--min_height", type=float, default=96.0)
    p.add_argument("--min_visible", type=float, default=0.6)
    a = p.parse_args()
    {"gt": stage_gt, "frames": stage_frames, "index": stage_index}[a.stage](a)


if __name__ == "__main__":
    main()
