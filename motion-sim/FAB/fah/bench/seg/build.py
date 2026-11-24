"""Hand instance-segmentation benchmark (bench_seg) from EPIC-KITCHENS VISOR GroundTruth-SparseAnnotations.

v1 = VISOR *val*: every frame with >= 1 hand instance ("left hand" / "right hand"; gloves counted as hands);
each instance -> RLE mask (bench.seg.rle) + handedness + box. VISOR annotations are exhaustive per frame,
so precision is scored too.

    python -m bench.seg.build --visor_root $DATA_ROOT/visor/.../GroundTruth-SparseAnnotations --split val \
        --out $DATA_ROOT/bench_seg --stride 1
"""
import argparse
import glob
import json
import os
import re
import zipfile

import cv2
import numpy as np

from bench.seg.rle import area, encode

HAND = re.compile(r"^(left|right) (hand|glove)$")


def polygons_to_mask(segments, hw, scale=(1.0, 1.0)):
    m = np.zeros(hw, np.uint8)
    for poly in segments:
        p = np.asarray(poly, np.float64)
        if len(p) < 3:
            continue
        p = np.round(p * np.asarray(scale) * 8).astype(np.int32)  # 3-bit subpixel
        cv2.fillPoly(m, [p], 1, cv2.LINE_8, 3)
    return m.astype(bool)


def main(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    recs = []
    for jf in sorted(glob.glob(os.path.join(args.visor_root, "annotations", args.split, "*.json"))):
        video = os.path.splitext(os.path.basename(jf))[0]
        part = video.split("_")[0]
        zpath = os.path.join(args.visor_root, "rgb_frames", args.split, part, video + ".zip")
        dpath = os.path.join(args.visor_root, "rgb_frames", args.split, part, video)
        if os.path.exists(zpath):
            zf = zipfile.ZipFile(zpath)
            names = {os.path.basename(n): n for n in zf.namelist()}
            read = zf.read
        elif os.path.isdir(dpath):  # some mirrors ship extracted frame folders
            names = {os.path.basename(n): n for n in glob.glob(os.path.join(dpath, "**", "*.jpg"), recursive=True)}
            read = lambda n: open(n, "rb").read()
        else:
            print("missing frames", zpath, flush=True)
            continue
        va = json.load(open(jf))["video_annotations"]
        for k, fr in enumerate(va):
            if k % args.stride:
                continue
            hands = [a for a in fr["annotations"] if HAND.match(a["name"].strip().lower())]
            if not hands or fr["image"]["name"] not in names:
                continue
            buf = np.frombuffer(read(names[fr["image"]["name"]]), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            H, W = img.shape[:2]
            # VISOR polygons are in 1920x1080 coordinates; frames may be stored smaller
            scale = (W / args.ann_width, H / args.ann_height)
            inst = []
            for a in hands:
                m = polygons_to_mask(a["segments"], (H, W), scale)
                if m.sum() < args.min_area:
                    continue
                ys, xs = np.nonzero(m)
                rle = encode(m)
                inst.append(dict(is_right=a["name"].lower().startswith("right"), mask_rle=rle, area=area(rle),
                                 bbox=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)],
                                 glove="glove" in a["name"].lower(), in_contact=a.get("in_contact_object") not in (None, "", "none")))
            if not inst:
                continue
            rel = f"images/{video}/{fr['image']['name']}"
            os.makedirs(os.path.dirname(os.path.join(args.out, rel)), exist_ok=True)
            with open(os.path.join(args.out, rel), "wb") as f:
                f.write(buf.tobytes())
            recs.append(dict(id=f"visor/{video}/{fr['image']['name'][:-4]}", file=rel, subset="visor", width=W, height=H,
                             K=None, fully_annotated=True, hands=inst))
        print(f"{video}: total {len(recs)} frames", flush=True)
    stats = dict(images=len(recs), hands=sum(len(r["hands"]) for r in recs),
                 right=sum(h["is_right"] for r in recs for h in r["hands"]),
                 gloves=sum(h["glove"] for r in recs for h in r["hands"]))
    meta = dict(version=args.version, source="EPIC-KITCHENS VISOR GroundTruth-SparseAnnotations "
                f"({args.split}; Darkhalil et al. 2022, CC BY-NC 4.0)", stats=stats, stride=args.stride,
                masks="visible instance masks (polygons rasterised), RLE column-major (bench.seg.rle)")
    json.dump(dict(meta=meta, images=recs), open(os.path.join(args.out, "annotations.json"), "w"))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--visor_root", required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--out", required=True)
    p.add_argument("--version", default="bench_seg")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--min_area", type=int, default=100)
    p.add_argument("--ann_width", type=int, default=1920)
    p.add_argument("--ann_height", type=int, default=1080)
    main(p.parse_args())
