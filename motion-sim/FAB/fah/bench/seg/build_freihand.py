"""bench_seg-format hand-mask set from the FreiHAND evaluation split (3,960 224x224 crops, one right hand each): the
GT mask is the union of the hand part labels of `evaluation/segmap` (rendered from the GT MANO fit: hand only, cut at
the MANO wrist, amodal). Images are linked, not copied.

    python -m bench.seg.build_freihand --freihand $DATA_ROOT/freihand --out $DATA_ROOT/bench_seg_freihand
"""
import argparse
import json
import os

import cv2
import numpy as np

from bench.seg.rle import area, encode


def main(args):
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    seg_dir = os.path.join(args.freihand, "evaluation", "segmap")
    recs = []
    for f in sorted(os.listdir(seg_dir)):
        stem = os.path.splitext(f)[0]
        m = cv2.imread(os.path.join(seg_dir, f), cv2.IMREAD_UNCHANGED) > 0
        if m.sum() < args.min_area:
            continue
        link = os.path.join(args.out, "images", stem + ".jpg")
        if not os.path.lexists(link):
            os.symlink(os.path.join(os.path.abspath(args.freihand), "evaluation", "rgb", stem + ".jpg"), link)
        ys, xs = np.nonzero(m)
        rle = encode(m)
        recs.append(dict(id=f"freihand/{stem}", file=f"images/{stem}.jpg", subset="freihand", width=m.shape[1],
                         height=m.shape[0], K=None, fully_annotated=True,
                         hands=[dict(is_right=True, mask_rle=rle, area=area(rle),
                                     bbox=[float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])]))
    meta = dict(version="bench_seg_freihand", source="FreiHAND evaluation split (Zimmermann et al. 2019), segmap of the MANO fit",
                stats=dict(images=len(recs), hands=len(recs)), masks="hand only (MANO wrist), amodal, RLE column-major")
    json.dump(dict(meta=meta, images=recs), open(os.path.join(args.out, "annotations.json"), "w"))
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--freihand", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--min_area", type=int, default=50)
    main(p.parse_args())
