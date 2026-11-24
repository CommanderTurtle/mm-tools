"""Run a method over a benchmark dataset shard and store GT + predictions per clip.

    python -m bench.run --dataset hot3d --split bench/splits/hot3d_bench.json \
        --data_root $DATA_ROOT/hot3d --method fah_pipeline --method_args '{"input_size": 384}' \
        --out_root /results/bench --shard 0 --num_shards 8
-> <out_root>/<dataset>/<method_name>/<clip>.pkl
"""
import argparse
import importlib
import json
import os
import pickle
import tempfile
import time

import numpy as np

METHODS = {
    "fah": "fah.api:FAH",  # fah.FAH arguments: {"model": "fah", "mode": "image" | "video" | "world", "weights": ...}
    "fah_pipeline": "bench.methods.fah:FAHMethod",
    "wilor": "bench.methods.wilor:WiLoRMethod",
    "hawor": "bench.methods.hawor:HaWoRMethod",
    "hamer": "bench.methods.hamer:HaMeRMethod",
    "mediapipe": "bench.methods.mediapipe_hands:MediaPipeHandsMethod",
    "rtmpose": "bench.methods.rtmpose_hand:RTMPoseHandMethod",
    "wilor_anyhand": "bench.methods.anyhand:WiLoRAnyHandMethod",
    "hamer_anyhand": "bench.methods.anyhand:HaMeRAnyHandMethod",
    "hamba": "bench.methods.hamba:HambaMethod",
    "haco": "bench.methods.haco:HACOMethod",  # contact baseline (dense MANO-vertex contact, no pose)
    "sam3db_hand": "bench.methods.sam3db:SAM3DBodyHandMethod",  # SAM 3D Body's hand branch off the shelf
}


def load_method(name, args):
    mod, cls = METHODS[name].split(":")
    return getattr(importlib.import_module(mod), cls)(**args)


def iter_clips(args):
    if args.dataset == "hot3d":
        from bench.datasets.hot3d import Hot3DClip
        from bench.common import ManoLayer

        split = json.load(open(args.split))
        clips = [c for dev in split["clips"] for c in split["clips"][dev]]
        clips = clips[args.shard :: args.num_shards]
        mano = ManoLayer(args.mano_dir, num_pca_comps=15)
        for c in clips:
            path = os.path.join(args.data_root, c)
            yield os.path.splitext(os.path.basename(c))[0], (lambda p=path: Hot3DClip(p, args.mano_dir, mano=mano))
    elif args.dataset == "egostandard":
        from bench.datasets.egostandard import EgoStandardClip

        clips = json.load(open(args.split))["clips"][args.shard :: args.num_shards]
        for c in clips:
            yield c, (lambda c=c: EgoStandardClip(os.path.join(args.data_root, "clips", c)))
    elif args.dataset == "dexycb":
        from bench.datasets.dexycb import DexYCBClip, DexYCBMano

        clips = json.load(open(args.split))["clips"][args.shard :: args.num_shards]
        mano = DexYCBMano(args.mano_dir)
        for c in clips:
            yield os.path.splitext(os.path.basename(c))[0], (lambda c=c: DexYCBClip(os.path.join(args.data_root, c), mano=mano))
    elif args.dataset == "oakink2":
        from bench.datasets.oakink2 import OakInk2Clip

        clips = json.load(open(args.split))["clips"][args.shard :: args.num_shards]
        for c in clips:
            yield c.replace("@", "_"), (lambda c=c: OakInk2Clip(args.data_root, c, args.mano_dir))
    elif args.dataset == "contactclip":
        from bench.datasets.contactclip import ContactClip

        clips = json.load(open(args.split))["clips"][args.shard :: args.num_shards]
        for c in clips:
            yield c, (lambda c=c: ContactClip(os.path.join(args.data_root, "clips", c)))
    elif args.dataset == "freihand":
        from bench.datasets.freihand import FreiHANDEval

        ds = FreiHANDEval(args.data_root, args.mano_dir)
        chunks = [list(range(i, min(i + 200, len(ds)))) for i in range(0, len(ds), 200)]
        for ci, ids in enumerate(chunks):
            if ci % args.num_shards == args.shard:
                yield f"chunk{ci:03d}", (lambda ids=ids: ds.subset(ids))
    elif args.dataset == "bench2d":
        from bench.bench2d.dataset import Bench2D

        ds = Bench2D(args.data_root, subsets=args.subsets.split(",") if args.subsets else None)
        for ci, chunk in ds.chunks(100):
            if ci % args.num_shards == args.shard:
                yield f"chunk{ci:04d}", (lambda c=chunk: c)
    elif args.dataset == "seg":
        from bench.seg.dataset import BenchSeg

        ds = BenchSeg(args.data_root)
        for ci, chunk in ds.chunks(100):
            if ci % args.num_shards == args.shard:
                yield f"chunk{ci:04d}", (lambda c=chunk: c)
    else:
        raise ValueError(args.dataset)


def main(args):
    method = load_method(args.method, json.loads(args.method_args))
    out_dir = os.path.join(args.out_root, args.dataset, args.run_name or method.name)
    os.makedirs(out_dir, exist_ok=True)
    for clip_name, make_clip in iter_clips(args):
        out_path = os.path.join(out_dir, f"{clip_name}.pkl")
        if os.path.exists(out_path) and not args.overwrite:
            continue
        clip = make_clip()
        frames = []
        idxs = range(0, len(clip), args.stride)
        if args.max_frames:
            idxs = list(idxs)[: args.max_frames]
        reset = getattr(method, "reset", None)
        if reset:
            reset()
        for i in idxs:
            fr = clip.frame(i)
            t0 = time.perf_counter()
            preds = method(fr["image"], fr["K"])
            dt = time.perf_counter() - t0
            frames.append(dict(key=fr["key"], index=i, K=fr["K"], gt=fr["hands"], ignore=fr.get("ignore", []), pred=preds, latency=dt,
                               image_size=fr["image"].shape[:2], subset=fr.get("subset"),
                               fully_annotated=fr.get("fully_annotated", True), T_world_cam=fr.get("T_world_cam")))
        finalize = getattr(method, "finalize_clip", None)
        if finalize:  # video methods may refine all frames of a clip at the end
            finalize(frames)
        # uniquely created temp file (O_EXCL; a PID alone repeats across machines): no writer can rename another's file
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(os.path.abspath(out_path)), prefix=os.path.basename(out_path) + ".", suffix=".tmp")
        with os.fdopen(fd, "wb") as f:
            pickle.dump(dict(clip=clip_name, method=method.name, dataset=args.dataset, frames=frames,
                             stride=args.stride, video=args.dataset in ("hot3d", "egostandard", "dexycb", "oakink2", "contactclip")), f)
        try:
            os.replace(tmp, out_path)
        except FileNotFoundError:  # network file systems: retry once after the attribute cache settles
            time.sleep(2)
            os.replace(tmp, out_path)
        lat = np.median([f["latency"] for f in frames]) * 1000
        print(f"[{method.name}] {clip_name}: {len(frames)} frames, median {lat:.1f} ms/frame", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, choices=["hot3d", "egostandard", "dexycb", "oakink2", "contactclip", "freihand", "bench2d", "seg"])
    p.add_argument("--subsets", default="", help="bench2d: comma-separated subsets (default all)")
    p.add_argument("--split", default="bench/splits/hot3d_bench.json")
    p.add_argument("--data_root", required=True)
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--method", required=True, choices=list(METHODS))
    p.add_argument("--method_args", default="{}")
    p.add_argument("--run_name", default="")
    p.add_argument("--out_root", default="output/bench")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    main(p.parse_args())
