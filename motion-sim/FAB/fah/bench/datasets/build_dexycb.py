"""Download DexYCB, convert it to per-clip tars (bench.datasets.dexycb) and write the benchmark / training splits.

    python -m bench.datasets.build_dexycb --raw $DATA_ROOT/dexycb_raw --out $DATA_ROOT/dexycb \
        --tmp /tmp/dexycb_tmp --download --check_dir $RESULTS/dexycb_check
Steps (each resumable): download the 10 subject archives + calibration from the project's Google Drive (public, no
login; <raw>/<file>.done after gzip -t), extract one subject at a time to --tmp without depth, pack
<out>/clips/<subject>/<sequence>_<serial>.tar (<out>/clips/<subject>/.done), splits, GT checks.
Splits (toolkit setup s0, the default DexYCB hand-pose protocol: seen subjects / views / objects):
  bench/splits/dexycb_s0_test_full.json  all 1280 s0 test clips (160 sequences x 8 cameras, every frame)
  bench/splits/dexycb_s0_test.json       benchmark: each s0 test sequence once (160 clips, all frames at 30 fps),
                                         camera (object index + subject index) % 8 -> 20 clips per camera, every
                                         object seen from 8 cameras across the 8 test subjects
  bench/splits/dexycb_s0_train.json      all 6400 s0 train clips (crop pool, video tracks)
"""
import argparse
import glob
import io
import json
import os
import shutil
import subprocess
import tarfile
from multiprocessing import Pool

import cv2
import numpy as np
import yaml

from bench.datasets.dexycb import DEXYCB_TIP_VERTS, SERIALS, SUBJECTS, DexYCBClip, DexYCBMano, split_clips

FILES = {  # https://dex-ycb.github.io/ (Google Drive ids)
    "calibration.tar.gz": "1UAwVKT4Rgb1fLcFoa1o71_-0NtSvvLAQ",
    "20200709-subject-01.tar.gz": "1Ehh92wDE3CWAiKG7E9E73HjN2Xk2XfEk",
    "20200813-subject-02.tar.gz": "1Uo7MLqTbXEa-8s7YQZ3duugJ1nXFEo62",
    "20200820-subject-03.tar.gz": "1FkUxas8sv8UcVGgAzmSZlJw1eI5W5CXq",
    "20200903-subject-04.tar.gz": "14up6qsTpvgEyqOQ5hir-QbjMB_dHfdpA",
    "20200908-subject-05.tar.gz": "1NBA_FPyGWOQF5-X9ueAat5g8lDMz-EmS",
    "20200918-subject-06.tar.gz": "1UWIN2-wOBZX2T0dkAi4ctAAW8KffkXMQ",
    "20200928-subject-07.tar.gz": "1oWEYD_o3PVh39pLzMlJcArkDtMj4nzI0",
    "20201002-subject-08.tar.gz": "1GTNZwhWbs7Mfez0krTgXwLPndvrw1Ztv",
    "20201015-subject-09.tar.gz": "1j0BLkaCjIuwjakmywKdOO9vynHTWR0UH",
    "20201022-subject-10.tar.gz": "1FvFlRfX-p5a5sAWoKEGc17zKJWwKaSB-",
}
LICENSE = "CC BY-NC 4.0 (DexYCB, Chao et al. CVPR 2021): non-commercial research use, attribution"
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]


class _Loader(yaml.SafeLoader):  # calibration files use !!python/tuple
    pass


_Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda l, n: l.construct_sequence(n))


def load_yaml(path):
    return yaml.load(open(path), Loader=_Loader)


def _download(arg):
    raw, name, fid = arg
    import gdown

    path = os.path.join(raw, name)
    for _ in range(5):
        if os.path.exists(path + ".done"):
            return name
        try:
            gdown.download(id=fid, output=path, quiet=True, resume=True)
            subprocess.run(["gzip", "-t", path], check=True)
            open(path + ".done", "w").close()
        except Exception as e:
            print("retry", name, e, flush=True)
    return name if os.path.exists(path + ".done") else None


def cam_to_world(extr, serial):
    """camera -> table (AprilTag) frame from extrinsics.yml (3x4 camera -> master camera, apriltag -> master)."""
    T = lambda k: np.vstack([np.asarray(extr["extrinsics"][k], np.float64).reshape(3, 4), [0, 0, 0, 1]])
    return np.linalg.inv(T("apriltag")) @ T(serial)


def convert_subject(arg):
    raw, out, tmp, subject = arg
    dst = os.path.join(out, "clips", subject)
    if os.path.exists(os.path.join(dst, ".done")):
        return subject, 0
    os.makedirs(dst, exist_ok=True)
    os.makedirs(tmp, exist_ok=True)
    src = os.path.join(tmp, subject)
    if not os.path.exists(os.path.join(tmp, f".{subject}.extracted")):
        shutil.rmtree(src, ignore_errors=True)
        subprocess.run(["tar", "--exclude=*aligned_depth_to_color_*", "-xzf", os.path.join(raw, f"{subject}.tar.gz"),
                        "-C", tmp], check=True)
        open(os.path.join(tmp, f".{subject}.extracted"), "w").close()
    calib = os.path.join(out, "calibration")
    n = 0
    for seq in sorted(os.listdir(src)):
        meta = load_yaml(os.path.join(src, seq, "meta.yml"))
        extr = load_yaml(os.path.join(calib, f"extrinsics_{meta['extrinsics']}", "extrinsics.yml"))
        betas = load_yaml(os.path.join(calib, f"mano_{meta['mano_calib'][0]}", "mano.yml"))["betas"]
        for serial in meta["serials"]:
            path = os.path.join(dst, f"{seq}_{serial}.tar")
            if os.path.exists(path):
                continue
            intr = load_yaml(os.path.join(calib, "intrinsics", f"{serial}_640x480.yml"))["color"]
            K = [[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]]
            d = os.path.join(src, seq, serial)
            nf = meta["num_frames"]
            info = dict(subject=subject, sequence=seq, serial=serial, num_frames=nf, fps=30, hw=[480, 640], K=K,
                        side=meta["mano_sides"][0], betas=betas, T_world_cam=cam_to_world(extr, serial).tolist(),
                        world="DexYCB AprilTag (table) frame", meta=meta, license=LICENSE)
            with tarfile.open(path + ".tmp", "w") as t:
                b = json.dumps(info).encode()
                ti = tarfile.TarInfo("info.json")
                ti.size = len(b)
                t.addfile(ti, io.BytesIO(b))
                for i in range(nf):
                    for f in (f"color_{i:06d}.jpg", f"labels_{i:06d}.npz"):
                        t.add(os.path.join(d, f), f)
            os.replace(path + ".tmp", path)
            n += 1
    open(os.path.join(dst, ".done"), "w").close()
    shutil.rmtree(src, ignore_errors=True)
    os.remove(os.path.join(tmp, f".{subject}.extracted"))
    return subject, n


def sequences(out):
    seqs = {}
    for s in SUBJECTS:
        names = sorted({os.path.basename(p)[:-4].rsplit("_", 1)[0] for p in glob.glob(os.path.join(out, "clips", s, "*.tar"))})
        seqs[s] = names
    return seqs


def write_splits(out, split_dir):
    seqs = sequences(out)
    rel = lambda s, q, c: f"clips/{s}/{q}_{c}.tar"
    base = dict(source="DexYCB (dex-ycb.github.io), converted by bench.datasets.build_dexycb", license=LICENSE, setup="s0",
                fps=30)
    test = split_clips(seqs, "s0", "test")
    train = split_clips(seqs, "s0", "train")
    assert not {q for _, q, _ in test} & {q for _, q, _ in train}
    full = dict(base, split="test", rule="toolkit s0 test: subjects 03-10, sequences i % 5 == 4, all 8 cameras",
                test_sequences=sorted({q for _, q, _ in test}), clips=[rel(*c) for c in test])
    bench = []
    for si, s in enumerate(SUBJECTS[2:]):
        qs = [q for i, q in enumerate(seqs[s]) if i % 5 == 4]  # one per grasped object (5 trials each)
        bench += [rel(s, q, SERIALS[(o + si) % 8]) for o, q in enumerate(qs)]
    sub = dict(base, split="test", rule="each s0 test sequence once, camera SERIALS[(object index + test subject index) % 8]",
               test_sequences=full["test_sequences"], clips=bench)
    trn = dict(base, split="train", rule="toolkit s0 train: all subjects, sequences i % 5 != 4, all 8 cameras",
               clips=[rel(*c) for c in train])
    for name, d in (("dexycb_s0_test_full", full), ("dexycb_s0_test", sub), ("dexycb_s0_train", trn)):
        json.dump(d, open(os.path.join(split_dir, f"{name}.json"), "w"), indent=1)
        print(name, len(d["clips"]), "clips")


def _stats(arg):
    """Per clip: frames, annotated frames, scored hands, box size, MANO-vs-label joint error (toolkit tips)."""
    path, mano_dir = arg
    clip = DexYCBClip(path, mano_dir)
    ann, scored, sizes, err = 0, 0, [], []
    for i in range(len(clip)):
        fr = clip.frame(i, image=False)
        ann += fr["annotated"]
        for h in fr["hands"]:
            scored += 1
            b = h["bbox"]
            sizes.append(max(b[2] - b[0], b[3] - b[1]))
        for h in fr["hands"] + fr["ignore"]:  # toolkit joints from our mesh: 16 MANO joints + manopth fingertips
            vj = h["joints3d"].copy()
            vj[[4, 8, 12, 16, 20]] = h["verts3d"][DEXYCB_TIP_VERTS[clip.is_right]]
            err.append(float(np.abs(vj - h["joints3d_dexycb"]).max() * 1000))
    return dict(clip=path, side=clip.info["side"], raw_frames=clip.info["num_frames"], frames=len(clip), annotated=ann, scored=scored,
                box_med=float(np.median(sizes)) if sizes else None, err_max=max(err) if err else None)


def draw_check(path, mano_dir, out_png, frames=(0.3, 0.6)):
    clip = DexYCBClip(path, mano_dir)
    tiles = []
    for t in frames:
        fr = clip.frame(int(t * (len(clip) - 1)))
        img = fr["image"].copy()
        for h in fr["hands"] + fr["ignore"]:
            uv = h["verts3d"][:, :2] / h["verts3d"][:, 2:] * fr["K"][[0, 1], [0, 1]] + fr["K"][[0, 1], [2, 2]]
            for u, v in uv[::2].astype(int):
                cv2.circle(img, (u, v), 1, (255, 200, 0), -1)
            j2 = h["joints2d"].astype(int)
            for a, b in BONES:
                cv2.line(img, tuple(j2[a]), tuple(j2[b]), (0, 0, 255), 1)
            b = h["bbox"].astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0) if h in fr["hands"] else (128, 128, 128), 1)
        cv2.putText(img, f"{clip.info['sequence']} {clip.info['serial']} {fr['key']} {clip.info['side']}", (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        tiles.append(img)
    cv2.imwrite(out_png, np.concatenate(tiles, 1), [cv2.IMWRITE_JPEG_QUALITY, 90])


def main(args):
    os.makedirs(args.raw, exist_ok=True)
    if args.download:
        with Pool(args.dl_workers) as pool:
            got = pool.map(_download, [(args.raw, k, v) for k, v in FILES.items()])
        assert all(got), [k for k, g in zip(FILES, got) if not g]
    calib = os.path.join(args.out, "calibration")
    if not os.path.exists(calib):
        os.makedirs(args.out, exist_ok=True)
        subprocess.run(["tar", "xzf", os.path.join(args.raw, "calibration.tar.gz"), "-C", args.out], check=True)
    subjects = [s for s in SUBJECTS if os.path.exists(os.path.join(args.raw, f"{s}.tar.gz.done"))]
    with Pool(args.workers) as pool:
        for s, n in pool.imap_unordered(convert_subject, [(args.raw, args.out, args.tmp, s) for s in subjects]):
            print(f"{s}: {n} clips written", flush=True)
    write_splits(args.out, args.split_dir)
    if args.check_dir:
        os.makedirs(args.check_dir, exist_ok=True)
        split = json.load(open(os.path.join(args.split_dir, "dexycb_s0_test.json")))
        clips = split["clips"]
        with Pool(args.workers * 4) as pool:
            st = pool.map(_stats, [(os.path.join(args.out, c), args.mano_dir) for c in clips])
        json.dump(st, open(os.path.join(args.check_dir, "stats_s0_test.json"), "w"), indent=1)
        a = lambda k: np.array([s[k] for s in st if s[k] is not None], np.float64)
        print(f"s0_test bench: {len(st)} clips, {a('frames').sum():.0f} frames ({a('frames').min():.0f}-{a('frames').max():.0f}"
              f" per clip; {a('raw_frames').sum():.0f} before trimming), annotated {a('annotated').sum() / a('frames').sum():.4f}, "
              f"scored hands {a('scored').sum():.0f}, "
              f"left clips {sum(s['side'] == 'left' for s in st)}, median box {np.median(a('box_med')):.0f} px, "
              f"MANO vs label joint_3d max err {a('err_max').max():.4f} mm")
        for c in clips[:: max(len(clips) // args.n_check, 1)][: args.n_check]:
            draw_check(os.path.join(args.out, c), args.mano_dir, os.path.join(args.check_dir, os.path.basename(c)[:-4] + ".jpg"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="downloaded archives (<file>.tar.gz + .done markers)")
    p.add_argument("--out", required=True)
    p.add_argument("--tmp", default="/tmp/dexycb_tmp", help="local scratch for one extracted subject (~20 GB)")
    p.add_argument("--download", action="store_true")
    p.add_argument("--split_dir", default="bench/splits")
    p.add_argument("--check_dir", default="")
    p.add_argument("--n_check", type=int, default=8)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--dl_workers", type=int, default=5)
    main(p.parse_args())
