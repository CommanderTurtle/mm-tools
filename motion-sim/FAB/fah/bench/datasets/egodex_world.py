"""EgoDex world clips: the camera model's second validation domain in the supplement ("EgoDex p2"), 80 episodes of
EgoDex part 2 listed in bench/splits/egodex_world_p2.json (with their selection rule), the first 900 frames of each.
No data is redistributed: `build` reads the official part2.zip (EgoDex download; its licence applies).
  build  frames <out>/frames/<episode>/%05d.jpg (JPEG q90) + K.npy + done, the input of the camera stages; GT pickles
         <out>/gt/<episode>.pkl (bench.run frame dicts: index, key, K, image_size, T_world_cam = the ARKit camera pose,
         gt hands with joints3d = the ARKit joints in the camera frame (OpenPose order), the box of the projected
         joints and is_right; hand_valid hands only); <out>/lag/<episode>.json: video / pose sync check, the lag in [-4, 4]
         frames that maximises the correlation of the image motion (Farneback flow at 320 px) with the GT camera's angular
         speed (a record; no episode was dropped)
  merge  GT pickles + hands predicted on the frames (python -m bench.world_camera hands --frames <out>/frames --out <dir>)
         -> bench.run-style pickles with pred and gt per frame, scored by bench.world like any benchmark run
    python -m bench.datasets.egodex_world build --zip <EgoDex part2.zip> --out $DATA_ROOT/egodex_world_p2 [--shard i --num_shards n]
    python -m bench.datasets.egodex_world merge --gt $DATA_ROOT/egodex_world_p2/gt --pred <hands dir> --out $RESULTS/egodex_world_p2/<run>
"""
import argparse
import glob
import json
import os
import pickle
import zipfile

import numpy as np

SPLIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "splits", "egodex_world_p2.json")


def lag_check(imgs, T):
    import cv2

    g = [cv2.cvtColor(cv2.resize(im, (320, int(320 * im.shape[0] / im.shape[1]))), cv2.COLOR_BGR2GRAY) for im in imgs]
    m = np.array([np.median(np.linalg.norm(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0), axis=-1))
                  for a, b in zip(g[:-1], g[1:])])
    R = T[:, :3, :3]
    w = np.array([np.degrees(np.arccos(np.clip((np.trace(R[i].T @ R[i + 1]) - 1) / 2, -1, 1))) for i in range(len(R) - 1)])
    cs = {}
    for lag in range(-4, 5):  # video step i vs pose step i + lag
        a, b = m[max(0, -lag):len(m) - max(0, lag)], w[max(0, lag):len(w) - max(0, -lag)]
        cs[lag] = float(np.corrcoef(a, b)[0, 1])
    return max(cs, key=cs.get), cs


def build(args):
    import cv2

    from train.egodex import EgoDexEpisode

    split = json.load(open(args.split))
    z = zipfile.ZipFile(args.zip)
    for e in split["episodes"][args.shard::args.num_shards]:
        name = e.replace("/", "__")
        fd = os.path.join(args.out, "frames", name)
        if os.path.exists(os.path.join(fd, "done")):
            continue
        raw = os.path.join(args.out, "raw")
        for ext in (".hdf5", ".mp4"):
            if not os.path.exists(os.path.join(raw, e + ext)):
                z.extract(e + ext, raw)
        ep = EgoDexEpisode(os.path.join(raw, e + ".hdf5"))
        os.makedirs(fd, exist_ok=True)
        frames, imgs = [], []
        for fr in ep.frames(0, split.get("max_frames", 900)):
            i = fr["frame_idx"]
            cv2.imwrite(os.path.join(fd, f"{i:05d}.jpg"), fr["image"], [cv2.IMWRITE_JPEG_QUALITY, 90])
            imgs.append(cv2.resize(fr["image"], (480, 270)))
            gt = [dict(is_right=h["is_right"], joints3d=h["joints_cam"], bbox=h["bbox"]) for h in fr["hands"] if h["hand_valid"]]
            frames.append(dict(key=f"{i:05d}", index=i, K=fr["K"], image_size=fr["image"].shape[:2], T_world_cam=fr["T_world_cam"],
                               gt=gt, ignore=[], pred=[]))
        np.save(os.path.join(fd, "K.npy"), ep.K)
        lag, cs = lag_check(imgs, np.stack([f["T_world_cam"] for f in frames]))
        for sub in ("gt", "lag"):
            os.makedirs(os.path.join(args.out, sub), exist_ok=True)
        pickle.dump(dict(clip=name, dataset="egodex", frames=frames, video=True), open(os.path.join(args.out, "gt", name + ".pkl"), "wb"))
        json.dump(dict(lag=lag, corr=cs, frames=len(frames), task=ep.task), open(os.path.join(args.out, "lag", name + ".json"), "w"))
        open(os.path.join(fd, "done"), "w").close()
        print(name, len(frames), "frames, lag", lag, "corr %.2f" % cs[lag], flush=True)


def merge(args):
    os.makedirs(args.out, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(args.pred, "*.pkl"))):
        name = os.path.basename(f)
        g = pickle.load(open(os.path.join(args.gt, name), "rb"))
        run = pickle.load(open(f, "rb"))
        p = {fr["index"]: fr for fr in run["frames"]}
        for fr in g["frames"]:
            fr["pred"] = p[fr["index"]]["pred"]
        g["method"] = run.get("method")
        pickle.dump(g, open(os.path.join(args.out, name + ".tmp"), "wb"))
        os.replace(os.path.join(args.out, name + ".tmp"), os.path.join(args.out, name))
        n += 1
    print(args.out, n, "clips")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build")
    a.add_argument("--zip", required=True, help="EgoDex part2.zip")
    a.add_argument("--out", required=True)
    a.add_argument("--split", default=SPLIT)
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--num_shards", type=int, default=1)
    a = sub.add_parser("merge")
    a.add_argument("--gt", required=True, help="<build out>/gt")
    a.add_argument("--pred", required=True, help="hand pickles on <build out>/frames (bench.world_camera hands)")
    a.add_argument("--out", required=True)
    args = p.parse_args()
    {"build": build, "merge": merge}[args.cmd](args)
