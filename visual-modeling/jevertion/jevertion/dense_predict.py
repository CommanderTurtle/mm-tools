"""Dense prediction on a picture: 13-class segmentation, depth and surface normals, one forward pass.

    python dense_predict.py --weights <jeveration.pt> --image room.jpg --out room_dense/

Writes into <out>/ (maps at the input picture's size):
    seg.png          class index per pixel (8-bit; the classes are jeveration.config.NYU13, in that order)
    seg_color.png    the same in colour, over a faint copy of the picture
    depth.npy        depth in meters, float32 [H, W];  depth.png: a view of it, nearer is brighter
    normal.npy       unit surface normals, float32 [H, W, 3];  normal.png: (n + 1) / 2 as RGB
The picture is resized to 576 x 768 for the model (trained on NYUv2 indoor scenes, 4:3), whatever its shape.
"""
import argparse
import os

import numpy as np
from PIL import Image


def main():
    from jeveration import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="the Jeveration checkpoint")
    ap.add_argument("--image", required=True)
    ap.add_argument("--tasks", default=",".join(config.DENSE_TASKS), help="comma list of seg, depth, normal")
    ap.add_argument("--out", default="dense")
    ap.add_argument("--backbone", default=None, help="local copy of Qwen3.5-4B (default: the Hugging Face Hub)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    tasks = tuple(t for t in a.tasks.split(",") if t)
    assert tasks and all(t in config.DENSE_TASKS for t in tasks), a.tasks

    from jeveration import JeverationPipeline
    from jeveration.dense import depth_view, normal_view, seg_view
    pipe = JeverationPipeline(a.weights, backbone=a.backbone, device=a.device)
    image = np.asarray(Image.open(a.image).convert("RGB"), dtype=np.uint8)
    pred = {t: v[0].cpu().numpy() for t, v in pipe.dense_predict(image[None], tasks).items()}
    os.makedirs(a.out, exist_ok=True)
    if "seg" in pred:
        seg = pred["seg"].astype(np.uint8)
        Image.fromarray(seg).save(os.path.join(a.out, "seg.png"))
        Image.fromarray(seg_view(seg, image)).save(os.path.join(a.out, "seg_color.png"))
        print("classes present:", ", ".join(config.NYU13[i] for i in np.unique(seg)))
    if "depth" in pred:
        d = pred["depth"].astype(np.float32)
        np.save(os.path.join(a.out, "depth.npy"), d)
        Image.fromarray(depth_view(d)).save(os.path.join(a.out, "depth.png"))
        print(f"depth: {d.min():.2f} to {d.max():.2f} m")
    if "normal" in pred:
        n = pred["normal"].astype(np.float32)
        np.save(os.path.join(a.out, "normal.npy"), n)
        Image.fromarray(normal_view(n)).save(os.path.join(a.out, "normal.png"))
    print(a.out)


if __name__ == "__main__":
    main()
