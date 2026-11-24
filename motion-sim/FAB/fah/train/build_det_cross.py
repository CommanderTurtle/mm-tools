"""Overlapping / crossed two-hand detector frames from HaGRID v2 (https://github.com/hukenovs/hagrid): the frames of
train/lists/det_hagrid_cross.json (3,001 images whose two hand boxes overlap, box IoU >= 0.1; selection rule in the
list), read from the official release. Per hand: the HaGRID box (every hand box of the image: the gesture hand and the
"no_gesture" hand), 2D keypoints and side from the FAH pose model on that box (run as right and as left, P(right) =
mean(side_score_R, 1 - side_score_L)); a frame with any unsure side (|P - 0.5| < --side_margin) is dropped. Output in the
YOLO-pose layout of train.build_det_dataset (images <= 1024 px, val = 1 / 50 by name) for train.build_det_seg_sam3 (SAM 3
masks, --split_pairs; no SAM 2.1 labels: --sam2_dir / --old empty).

    python -m train.build_det_cross --annotations $DATA_ROOT/hagrid/annotations --images $DATA_ROOT/hagrid/images \
        --out $DATA_ROOT/det_hagrid --method_args "$(python -c "import fah, json; print(json.dumps(fah.method_args('fah', 'image')))")"

--annotations: HaGRID v2's annotations/{train,val,test}/<gesture>.json (or annotations.zip); --images: the per-gesture
image archives unpacked as <images>/<gesture>/<image id>.jpg.
"""
import argparse
import json
import os
import zipfile

import cv2
import numpy as np

from train.build_det_dataset import _write

LIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lists", "det_hagrid_cross.json")


def read_annotations(root, split, gesture):
    name = f"{split}/{gesture}.json"
    if root.endswith(".zip"):
        with zipfile.ZipFile(root) as z:
            return json.loads(z.read(f"annotations/{name}"))
    return json.load(open(os.path.join(root, name)))


def frames(args):
    """[(gesture, image path, image id, HaGRID hand boxes x, y, w, h relative to the image)] of the list."""
    sel = json.load(open(args.list))["frames"]
    out = []
    for g, splits in sel.items():
        for split, ids in splits.items():
            ann = read_annotations(args.annotations, split, g)
            out += [(g, os.path.join(args.images, g, f"{k}.jpg"), k, ann[k]["bboxes"]) for k in ids]
    return out


def main(args):
    import torch

    from bench.methods.fah import FAHMethod

    m = FAHMethod(**json.loads(os.path.expandvars(args.method_args)))
    stats = dict(listed=0, side_unsure_frames=0, written=0, hands=0)
    rows = []
    for g, path, iid, rel in frames(args):
        stats["listed"] += 1
        img = cv2.imread(path)
        if img is None:
            print("missing", path, flush=True)
            continue
        H, W = img.shape[:2]
        boxes = np.asarray([[round(x * W, 1), round(y * H, 1), round((x + w) * W, 1), round((y + h) * H, 1)] for x, y, w, h in rel],
                           np.float32)
        with torch.inference_mode():
            hr = m.est.process_image(img, boxes=boxes, is_right=np.ones(len(boxes), bool))
            hl = m.est.process_image(img, boxes=boxes, is_right=np.zeros(len(boxes), bool))
        pr = [0.5 * (a["side_score"] + 1 - b["side_score"]) for a, b in zip(hr, hl)]
        if any(abs(p - 0.5) < args.side_margin for p in pr):
            stats["side_unsure_frames"] += 1
            continue
        hands = []
        for b, p, a, l in zip(boxes, pr, hr, hl):
            kp = (a if p > 0.5 else l)["keypoints_2d"].astype(np.float64)
            hands.append((p > 0.5, b, np.concatenate([kp, np.ones((21, 1))], 1)))
        n = _write(args.out, "hgx_hg_" + iid, img, hands, args.max_side)
        stats["written"] += n > 0
        stats["hands"] += n
        rows.append(dict(id=iid, gesture=g, p_right=[float(p) for p in pr]))
    y = f"path: {args.out}\ntrain: images/train\nval: images/val\nkpt_shape: [21, 3]\nflip_idx: {list(range(21))}\nnames:\n  0: left\n  1: right\n"
    open(os.path.join(args.out, "hands.yaml"), "w").write(y)
    json.dump(dict(stats=stats, frames=rows), open(os.path.join(args.out, "build_cross.json"), "w"), indent=1)
    print(json.dumps(stats))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--annotations", required=True, help="HaGRID v2 annotations dir (<split>/<gesture>.json) or annotations.zip")
    p.add_argument("--images", required=True, help="<images>/<gesture>/<image id>.jpg")
    p.add_argument("--out", required=True)
    p.add_argument("--method_args", required=True, help="bench.methods.fah args (FAH pose model)")
    p.add_argument("--list", default=LIST)
    p.add_argument("--side_margin", type=float, default=0.2)
    p.add_argument("--max_side", type=int, default=1024)
    main(p.parse_args())
