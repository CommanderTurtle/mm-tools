"""Supplementary figure of custom-benchmark samples (fig:supp_samples): four shadow_v1 images (types A / B) and four
synthetic low-light copies of bench2d v2 COCO val2017 images, each a square crop with the ground-truth 2D keypoints;
low-light tiles carry the original crop as an inset. Two rows of four 400 px tiles, numbered 1-8.

The selection (bench/splits/supp_samples.json) is fixed: shadow tiles are visual picks from the per-type candidate
lists; low-light tiles are the first images of a seeded order over the licence / size filters that pass a visual check,
recorded per rank. `select` re-derives both candidate lists and the order from the annotations and checks the recorded
picks against them; `render` writes the figure.

    python scripts/supp_samples.py select --shadow $DATA_ROOT/shadow_v1 --dark $DATA_ROOT/bench2d_v2_dark \
        --coco_wholebody $DATA_ROOT/coco/coco_wholebody_val_v1.0.json
    python scripts/supp_samples.py render --shadow $DATA_ROOT/shadow_v1 --bench2d $DATA_ROOT/bench2d_v2 \
        --dark $DATA_ROOT/bench2d_v2_dark --out supp_bench_samples.jpg

bench2d_v2_dark: python -m bench.bench2d.make_dark --src $DATA_ROOT/bench2d_v2 --dst $DATA_ROOT/bench2d_v2_dark --level dark.

Image credits (Flickr photos, all CC BY 2.0; cropped, with our overlays; 5-8 darkened by our low-light synthesis):
1 "Delta_100_Ring_Park073" by Maggie Brauer, 2 "Zebra hand" by Zepfanman.com, 3 "Kinu as Capsicum" by Nagarjun Kandukuru,
4 "bratmobile" by super awesome (1, 3, 4 via Open Images V7, 2 via Openverse); 5 "20111005-first lady host wine tasting
for nga staff-jb" by Maryland GovPics, 6 "wii bowling" by rachaelvoorhees, 7 "68/365 - I'm too sexy to match." by Becky
Wetherington, 8 "Jelly Doughnut Incident" by Brad Greenlee (5-8 via COCO val2017). Photo pages: bench/splits/supp_samples.json.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fah.vis import draw_hand_skeleton  # noqa: E402

PICKS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench", "splits", "supp_samples.json")
T = 400
SHADOW_SOURCES = ("Open Images V7", "Openverse (Flickr)", "Wikimedia Commons")


def square(box, H, W, scale):
    """Square crop (x0, y0, side) around a box, scaled, clipped to the image."""
    x0, y0, x1, y1 = box
    side = int(min(max(x1 - x0, y1 - y0) * scale, H, W))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return int(np.clip(cx - side / 2, 0, W - side)), int(np.clip(cy - side / 2, 0, H - side)), side


def tile(img, hands, crop):
    X0, Y0, side = crop
    t = cv2.resize(img[Y0:Y0 + side, X0:X0 + side], (T, T), interpolation=cv2.INTER_AREA)
    for h in hands:
        k = (np.asarray(h["kp2d"], float)[:, :2] - [X0, Y0]) * T / side
        if (k.min(0) > -0.2 * T).all() and (k.max(0) < 1.2 * T).all():  # hands mostly outside the crop are not drawn
            t = draw_hand_skeleton(t, k, thickness=3, radius=4)
    return t


def number(t, n):
    cv2.rectangle(t, (0, T - 30), (30, T), (255, 255, 255), -1)
    cv2.putText(t, str(n), (8, T - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return t


def shadow_candidates(shadow):
    """{type: [file name]} of the licence-clean shadow_v1 candidates."""
    src = json.load(open(os.path.join(shadow, "sources.json")))
    ann = {im["file"].split("/")[-1]: im for im in json.load(open(os.path.join(shadow, "annotations.json")))["images"]}
    cand = [k for k, v in src.items() if v["source"] in SHADOW_SOURCES and not v.get("borderline") and k in ann and ann[k]["hands"]]
    return {t: [k for k in cand if src[k]["type"] == t] for t in "AB"}


def dark_order(images, licence_of, licences, min_side, min_visible, seed):
    """Seeded order [(id, eligible hand indices)] of the low-light COCO images that pass the licence / size filters."""
    el = []
    for im in images:
        if im["subset"] != "coco" or licence_of.get(int(im["id"].split("/")[1])) not in set(licences):
            continue
        hs = [i for i, h in enumerate(im["hands"]) if max(h["bbox"][2] - h["bbox"][0], h["bbox"][3] - h["bbox"][1]) >= min_side
              and sum(h.get("vis", [1] * 21)) >= min_visible]
        if hs:
            el.append((im["id"], hs))
    rng = np.random.default_rng(seed)
    return [el[i] for i in rng.permutation(len(el))]


def dark_picks(order, decisions, n=4):
    """The first n ranks whose recorded check passed (the decisions must follow the order)."""
    picks = []
    for d in decisions:
        assert order[d["rank"]][0] == d["id"], (d["rank"], d["id"], order[d["rank"]][0])
        if d["pass"]:
            picks.append(d["id"])
        if len(picks) == n:
            break
    return picks


def select(a):
    cfg = json.load(open(a.picks))
    cand = shadow_candidates(a.shadow)
    for p in cfg["shadow"]["picks"]:
        assert p["id"] in cand[p["type"]], p
    print("shadow candidates:", {t: len(v) for t, v in cand.items()}, "picks:", [p["id"] for p in cfg["shadow"]["picks"]])
    lic = {im["id"]: im["license"] for im in json.load(open(a.coco_wholebody))["images"]}
    dk = cfg["dark"]
    order = dark_order(json.load(open(os.path.join(a.dark, "annotations.json")))["images"], lic, dk["licences"], dk["min_side"],
                       dk["min_visible"], dk["seed"])
    picks = dark_picks(order, dk["decisions"])
    assert len(order) == dk["eligible"], len(order)
    assert sorted(picks) == sorted(dk["tile_order"]), picks
    print("low-light eligible:", len(order), "picks (order):", picks, "tiles:", dk["tile_order"])
    return order, picks


def render(a):
    cfg = json.load(open(a.picks))
    src = json.load(open(os.path.join(a.shadow, "sources.json")))
    ann = {im["file"].split("/")[-1]: im for im in json.load(open(os.path.join(a.shadow, "annotations.json")))["images"]}
    row1 = []
    for i, p in enumerate(cfg["shadow"]["picks"]):
        im = ann[p["id"]]
        img = cv2.imread(os.path.join(a.shadow, im["file"]))
        H, W = img.shape[:2]
        row1.append(number(tile(img, im["hands"], square(src[p["id"]]["box"], H, W, p["scale"])), i + 1))
    dk = {im["id"]: im for im in json.load(open(os.path.join(a.dark, "annotations.json")))["images"]}
    og = {im["id"]: im for im in json.load(open(os.path.join(a.bench2d, "annotations.json")))["images"]}
    sel = {d["id"]: d["rank"] for d in cfg["dark"]["decisions"] if d["pass"]}
    d = cfg["dark"]
    row2 = []
    for i, key in enumerate(d["tile_order"]):
        im = dk[key]
        hs = [j for j, h in enumerate(im["hands"]) if max(h["bbox"][2] - h["bbox"][0], h["bbox"][3] - h["bbox"][1]) >= d["min_side"]
              and sum(h.get("vis", [1] * 21)) >= d["min_visible"]]  # the crop covers the eligible hands (as the order)
        img = cv2.imread(os.path.join(a.dark, im["file"]))
        org = cv2.imread(os.path.join(a.bench2d, og[key]["file"]))
        assert org.shape == img.shape, (key, org.shape, img.shape)
        H, W = img.shape[:2]
        bx = np.array([im["hands"][j]["bbox"] for j in hs])
        crop = square([bx[:, 0].min(), bx[:, 1].min(), bx[:, 2].max(), bx[:, 3].max()], H, W, d["scale"])
        t = tile(img, im["hands"], crop)
        X0, Y0, side = crop
        s = T * 3 // 10  # inset: the original (not darkened) crop, top right
        t[0:s + 3, T - s - 3:T] = 255
        t[0:s, T - s:T] = cv2.resize(org[Y0:Y0 + side, X0:X0 + side], (s, s), interpolation=cv2.INTER_AREA)
        row2.append(number(t, i + 5))
        print(f"tile {i + 5}: {key} (rank {sel[key]}) crop {crop}")

    def row(ts):
        return np.concatenate([np.pad(t, ((4, 4), (4, 4), (0, 0)), constant_values=255) for t in ts], 1)

    g = np.concatenate([row(row1), row(row2)], 0)
    cv2.imwrite(a.out, g, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(g.shape[1], "x", g.shape[0], "->", a.out)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["select", "render"])
    p.add_argument("--picks", default=PICKS)
    p.add_argument("--shadow", required=True, help="shadow_v1 folder (annotations.json, sources.json, images)")
    p.add_argument("--bench2d", default="", help="render: bench2d v2 folder (inset originals)")
    p.add_argument("--dark", required=True, help="bench2d_v2_dark folder (bench/bench2d/make_dark.py --level dark)")
    p.add_argument("--coco_wholebody", default="", help="select: COCO-WholeBody val json (image licence ids)")
    p.add_argument("--out", default="supp_bench_samples.jpg")
    a = p.parse_args()
    dict(select=select, render=render)[a.cmd](a)
