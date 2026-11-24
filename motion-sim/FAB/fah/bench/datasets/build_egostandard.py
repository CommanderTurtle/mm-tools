"""Build the EgoStandard 3D video benchmark (egostandard) from the LightwheelAI/EgoStandard bucket.

There is no official test split, so the benchmark holds out whole tasks: a task is a test task iff crc32(task name) % test_mod
== 0 (all its episodes, in every sub-SKU, are excluded from training -- train.build_egostandard_crops asserts it).
One episode per test task (smallest crc32(uuid) among episodes of min_mb..max_mb), n_clips tasks in crc32 order;
from each, the window of `window_s` seconds of consecutive frames (head_left, native fps, no subsampling) with the
most scorable hands. Defaults: 120 clips x 30 s = 60 min.

    HF_TOKEN=... python -m bench.datasets.build_egostandard --raw $DATA_ROOT/egostandard_raw \
        --out $DATA_ROOT/egostandard --check_dir $RESULTS/egostandard_check
-> bench/splits/egostandard_bench.json, <out>/clips/<uuid>/{%06d.jpg, gt.npz, meta.json},
   <out>/annotations.json (bench2d format, every stride2d-th frame = 1 Hz: 2D PCK via --dataset bench2d)
Then bench.datasets.egostandard_fit (MANO-fit targets, the benchmark default). EgoStandard is test-only.
--task_rem r (r != 0) builds a dev split instead, from the tasks with crc32(task) % test_mod == r (disjoint from the test
tasks; same selection and converter), e.g. egostandard_dev (r 1, 80 clips) for camera-scale selection.
"""
import argparse
import json
import os
import zlib
from multiprocessing import Pool

import cv2
import numpy as np

from bench.datasets.egostandard import EgoStandardClip, download, hand_gt, list_bucket, read_episode

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
LICENSE = "LightwheelAI commercial-training-no-resale-v1.0 (eval/benchmark OK; no redistribution of frames/records)"


def crc(s):
    return zlib.crc32(s.encode())


def is_test_task(task, test_mod):
    return crc(task) % test_mod == 0


def make_split(eps, args):
    tasks = sorted({e["task"] for e in eps})
    test_tasks = [t for t in tasks if crc(t) % args.test_mod == args.task_rem]  # task_rem 0: the test tasks
    sel = set(test_tasks)
    by_task = {}
    for e in eps:
        if e["task"] in sel and args.min_mb * 1e6 <= e["size"] <= args.max_mb * 1e6:
            by_task.setdefault(e["task"], []).append(e)
    pick = [min(v, key=lambda e: crc(e["uuid"])) for v in by_task.values()]
    pick = sorted(pick, key=lambda e: crc(e["task"] + "/bench"))[: args.n_clips]
    rule = (f"test tasks: crc32(task) % {args.test_mod} == 0; never train on any episode of a test task" if not args.task_rem else
            f"dev tasks: crc32(task) % {args.test_mod} == {args.task_rem} (disjoint from the test tasks, == 0); camera-scale "
            f"selection only; do not train on them either")
    return dict(source=f"HF bucket LightwheelAI/EgoStandard ({', '.join(sorted({e['subset'] for e in eps}))}), head_left camera",
                license=LICENSE, rule=rule, test_mod=args.test_mod, task_rem=args.task_rem, num_tasks=len(tasks),
                num_episodes=len(eps), **{"dev_tasks" if args.task_rem else "test_tasks": test_tasks},
                clips=[e["uuid"] for e in pick], episodes={e["uuid"]: dict(path=e["path"], subset=e["subset"], task=e["task"],
                                                                           size=e["size"]) for e in pick})


def scorable(ep, min_visible=0.5):
    """(n_frames,) number of hands that EgoStandardClip would score."""
    n = len(ep["ts"])
    cnt = np.zeros(n, int)
    for i in range(n):
        if ep["bad_img"][i]:
            continue
        for side, is_right in (("left", False), ("right", True)):
            h = hand_gt(ep[f"joints_{side}"][i], ep["T_world_cam"][i], ep["K"], ep["wh"], is_right)
            cnt[i] += int(h is not None and h["visible"] >= min_visible and not ep[f"bad_{side}"][i])
    return cnt


def _convert(arg):
    uuid, info, raw, out, window_s = arg
    from bench.datasets.egostandard import iter_frames

    d = os.path.join(out, "clips", uuid)
    if os.path.exists(os.path.join(d, "meta.json")):
        return uuid, json.load(open(os.path.join(d, "meta.json")))
    ep = read_episode(os.path.join(raw, info["path"]))
    n = len(ep["ts"])
    cnt = scorable(ep)
    W = min(int(round(window_s * ep["fps"])), n)
    run = np.convolve(cnt, np.ones(W, int), "valid")
    s = int(info.get("window", [int(run.argmax())])[0])
    idx = np.arange(s, s + W)
    os.makedirs(d, exist_ok=True)
    got = 0
    for i, img in iter_frames(os.path.join(raw, info["path"]), idx):  # streamed: 30 s of 1920x1456 frames is ~7.5 GB raw
        cv2.imwrite(os.path.join(d, f"{i:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        got += 1
    assert got == len(idx), f"{uuid}: decoded {got} of {len(idx)} frames"
    np.savez_compressed(os.path.join(d, "gt.npz"), index=idx, ts=ep["ts"][idx], K=ep["K"], wh=np.array(ep["wh"]),
                        T_world_cam=ep["T_world_cam"][idx], joints_left=ep["joints_left"][idx],
                        joints_right=ep["joints_right"][idx], bad_left=ep["bad_left"][idx], bad_right=ep["bad_right"][idx],
                        bad_img=ep["bad_img"][idx])
    meta = dict(uuid=uuid, subset=info["subset"], task=info["task"], path=info["path"], window=[s, s + W], num_frames=n,
                fps=ep["fps"], wh=list(ep["wh"]), convention=ep["convention"], in_view=ep["in_view"],
                pose_source=ep["pose_source"], counts=ep["counts"], scorable_hands=int(cnt[idx].sum()),
                bad_frac=dict(left=float(ep["bad_left"].mean()), right=float(ep["bad_right"].mean()), img=float(ep["bad_img"].mean())),
                session=ep["meta"])
    json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
    print(f"{uuid} [{info['subset']}] {info['task']}: {n} frames @ {ep['fps']:.1f} fps, {ep['wh']}, window {s}-{s + W}, "
          f"{meta['scorable_hands']} scorable hands, convention {ep['convention']} {ep['in_view']}", flush=True)
    return uuid, meta


def draw(img, fr):
    vis = img.copy()
    for hands, col in ((fr["hands"], None), (fr["ignore"], (128, 128, 128))):
        for h in hands:
            c = col or ((0, 0, 255) if h["is_right"] else (255, 128, 0))
            uv = h["joints2d"]
            for a, b in BONES:
                if h["inside"][a] and h["inside"][b]:
                    cv2.line(vis, tuple(int(v) for v in uv[a]), tuple(int(v) for v in uv[b]), c, 2, cv2.LINE_AA)
            for k in np.flatnonzero(h["inside"]):
                cv2.circle(vis, tuple(int(v) for v in uv[k]), 3, c, -1, cv2.LINE_AA)
            b = h["bbox"].astype(int)
            cv2.rectangle(vis, (b[0], b[1]), (b[2], b[3]), c, 1)
    return vis


def main(args):
    token = os.environ.get("HF_TOKEN")
    split = json.load(open(args.split)) if os.path.exists(args.split) else None
    if split is None:
        eps = list_bucket(os.path.join(args.raw, "listing.jsonl"), token=token)
        split = make_split(eps, args)
        print(f"{split['num_episodes']} episodes / {split['num_tasks']} tasks; {len(split.get('test_tasks', split.get('dev_tasks', [])))} {'dev' if split.get('task_rem') else 'test'} tasks; "
              f"{len(split['clips'])} clips", flush=True)
        json.dump(split, open(args.split, "w"), indent=1)  # fixed before any download (resumable)
    clips = split["clips"][: args.limit or None]
    download([split["episodes"][u]["path"] for u in clips], args.raw, token=token)
    with Pool(args.workers) as pool:
        metas = dict(pool.imap_unordered(_convert, [(u, split["episodes"][u], args.raw, args.out, args.window_s) for u in clips]))
    for u, m in metas.items():
        split["episodes"][u].update(window=m["window"], convention=m["convention"])
    json.dump(split, open(args.split, "w"), indent=1)

    recs = []
    for k, u in enumerate(clips):
        clip = EgoStandardClip(os.path.join(args.out, "clips", u), targets="raw")  # 2D GT = tracked keypoints
        for i in range(0, len(clip), args.stride2d):
            fr = clip.frame(i)
            hands = []
            for h in fr["hands"]:
                hands.append(dict(is_right=bool(h["is_right"]), kp2d=h["joints2d"].round(2).tolist(),
                                  vis=h["inside"].astype(int).tolist(), bbox=[float(v) for v in h["bbox"]]))
            if hands:
                recs.append(dict(id=f"egostandard/{u}_{fr['key']}", file=f"clips/{u}/{fr['key']}.jpg", subset=clip.meta["subset"],
                                 width=clip.wh[0], height=clip.wh[1], K=clip.K.round(3).tolist(), fully_annotated=not fr["ignore"],
                                 sequence=u, hands=hands))
            if args.check_dir and k < args.check and i in (0, len(clip) // 2 // args.stride2d * args.stride2d):
                os.makedirs(args.check_dir, exist_ok=True)
                cv2.imwrite(os.path.join(args.check_dir, f"{u}_{fr['key']}.jpg"), draw(fr["image"], fr), [cv2.IMWRITE_JPEG_QUALITY, 85])
    stats = {}
    for r in recs:
        s = stats.setdefault(r["subset"], dict(images=0, hands=0, right=0))
        s["images"] += 1
        s["hands"] += len(r["hands"])
        s["right"] += sum(h["is_right"] for h in r["hands"])
    meta = dict(version=os.path.basename(args.out.rstrip("/")), keypoints="OpenPose 21 (wrist, thumb1-4, index1-4, middle1-4, ring1-4, pinky1-4)",
                stats=stats, params=dict(stride2d=args.stride2d, window_s=args.window_s), sources=dict(egostandard=LICENSE))
    json.dump(dict(meta=meta, images=recs), open(os.path.join(args.out, "annotations.json"), "w"))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", required=True, help="bucket mirror root (listing.jsonl + <subset>/mcap/<task>/<uuid>.mcap)")
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="bench/splits/egostandard_bench.json")
    p.add_argument("--test_mod", type=int, default=25)
    p.add_argument("--task_rem", type=int, default=0, help="0: test split; r: dev split from crc32(task) % test_mod == r")
    p.add_argument("--n_clips", type=int, default=120)
    p.add_argument("--min_mb", type=float, default=45, help="~37 s at the ~1.2 MB/s of a stereo-head episode")
    p.add_argument("--max_mb", type=float, default=400)
    p.add_argument("--window_s", type=float, default=30.0)
    p.add_argument("--stride2d", type=int, default=30)
    p.add_argument("--limit", type=int, default=0, help="only the first N clips (smoke test)")
    p.add_argument("--check_dir", default="")
    p.add_argument("--check", type=int, default=6, help="overlay frames for the first N clips")
    p.add_argument("--workers", type=int, default=8)
    main(p.parse_args())
