"""Foot contacts for world mode (the joint human-scene step's contact terms), around the two contact models.
Stages:
  prep    the clip list + its FAB raw npz (box (T, 4), valid (T,), K, frames) -> a clips spec json for
          train/extract_contact_tracks.py --source clips:<json> (the image contact head of the FAB model), and the
          clip's ViPE cameras (T_world_cam, camera-to-world, metric) -> the video model's --test_cams dir (frame, T_world_cam)
  export  the track npz (image head: sigmoid(logits); the 2D heel / toe = its contact points in image px) and the video
          model's predictions (causal GRU on world-frame keypoints) -> <out>/<clip>__0.contact.npz with frame, time, box,
          ids, {image,video}_{static,any}_prob (T, 4: l_heel, l_toe, r_heel, r_toe), the bools at --thr, *_hk_* (l_hand,
          r_hand, l_knee, r_knee), heel_xy / toe_xy (T, 2, 2) px (foot L / R), channels, thr, fps. Frames without a valid
          box are left out.
The inputs come from environment variables (scripts/fab_world.sh sets them): FAB_CONTACT_CLIPS (clip list json, clip ->
images), FAB_CONTACT_RAW and FAB_CONTACT_CAMS (path patterns with {c}), CLIPS_ONLY (a comma list of clips).

    python scripts/contact_clips.py prep --set clip --work <dir>
    python scripts/contact_clips.py export --set clip --work <dir> --out <dir>
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

SETS = {  # set -> (clip list json candidates, raw npz pattern, camera npz pattern, clip names or None = every clip of the list)
    "clip": ([os.environ.get("FAB_CONTACT_CLIPS", "")], os.environ.get("FAB_CONTACT_RAW", ""), os.environ.get("FAB_CONTACT_CAMS", ""),
             None),
}
CH = ["l_heel", "l_toe", "r_heel", "r_toe"]
HK = ["l_hand", "r_hand", "l_knee", "r_knee"]
# contact points (fab/contact_head.CONTACT_POINTS = MHR70 13 14 15 16 17 18 19 20 11 12 62 41): heel / toe rows
HEEL, TOES = (4, 7), ((2, 3), (5, 6))


def clips_of(st):
    js, raw, cam, names = SETS[st]
    lists = [json.load(open(j)) for j in js if j and Path(j).exists()]
    if isinstance(names, str):  # a comma / whitespace separated list file
        names = [x for x in open(names).read().replace(",", " ").split() if x]
    names = names or sorted(lists[0])
    only = os.environ.get("CLIPS_ONLY", "")  # a comma list: these clips only
    if only:
        names = [c for c in names if c in set(only.split(","))]
    return names, lists, raw, cam


def prep(a):
    names, lists, raw, cam = clips_of(a.set)
    work = Path(a.work)
    (work / "cams").mkdir(parents=True, exist_ok=True)
    spec, rep = {}, {}
    for c in names:
        z = np.load(raw.format(c=c), allow_pickle=True)
        T = len(z["box"])
        imgs = next((d[c]["images"] for d in lists if c in d and len(d[c]["images"]) == T), None)
        keys = z.files if hasattr(z, "files") else list(z)
        if imgs is None and "frames" in keys:  # a subsampled raw: its frames index the clip list's images
            full = next((d[c]["images"] for d in lists if c in d), None)
            fr0 = np.asarray(z["frames"]).astype(int)
            if full is not None and len(fr0) == T and fr0.min() >= 0 and fr0.max() < len(full):
                imgs = [full[f] for f in fr0]
        fps = float(z["fps"]) if "fps" in keys else 30.0
        if imgs is None:
            rep[c] = f"no clip list with {T} images"
            continue
        fr = np.asarray(z["frames"]).astype(int) if "frames" in keys else np.arange(T)
        spec[f"{c}__0"] = dict(images=imgs, box=z["box"].astype(float).tolist(), valid=z["valid"].astype(bool).tolist(),
                               K=z["K"].astype(float).tolist(), frames=fr.tolist(), fps=fps)
        cf = Path(cam.format(c=c))
        if cf.exists():
            cz = np.load(cf, allow_pickle=True)
            Tw = cz["T_world_cam"].astype(np.float64)
            # the video model was trained in a first-camera world (y down, ViPE's convention: T_world_cam[0] = I); a
            # gravity-aligned world lowers its static recall a lot, so every camera path is re-expressed there
            Tw = np.einsum("ij,tjk->tik", np.linalg.inv(Tw[0]), Tw)
            if len(Tw) == T:
                np.savez(work / "cams" / f"{c}__0.npz", frame=fr, T_world_cam=Tw, fix=np.float32(1.0))
        rep[c] = dict(frames=T, valid=int(np.asarray(z["valid"]).sum()), camera=cf.exists())
    (work / "clips_spec.json").write_text(json.dumps(spec))
    print(json.dumps(rep, indent=1))


def export(a):
    names, _, _, _ = clips_of(a.set)
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = json.load(open(work / "clips_spec.json"))
    vid = {}
    vf = work / "pred" / "tracks.npz"
    if vf.exists():
        z = np.load(vf, allow_pickle=True)
        vid = dict(zip(z["ids"].astype(str), z["prob"].astype(np.float32)))
    rep = {}
    for c in names:
        tf = work / "tracks" / f"{c}__0.npz"
        if not tf.exists():
            rep[c] = "no track"
            continue
        t = np.load(tf, allow_pickle=True)
        sp = spec[f"{c}__0"]
        K = np.asarray(sp["K"], np.float64)
        box = np.asarray(sp["box"], np.float32)
        frames = t["frame"].astype(int)
        row = {int(f): i for i, f in enumerate(sp["frames"])}
        P = 1 / (1 + np.exp(-t["logits"].astype(np.float64)))
        if "uv" in t.files:  # the contact points' image px (train/extract_contact_tracks: dense uv through the crop affine)
            uv = t["uv"].astype(np.float64)
        else:  # older tracks: pts + cam_t through K, which is not a consistent camera frame (px off by ~1 box width)
            pts = t["pts"].astype(np.float64)
            uv = pts @ K.T
            uv = uv[..., :2] / np.clip(uv[..., 2:3], 1e-6, None)
        d = dict(ids=t["ids"], frame=frames, time=t["time"], box=box[[row[int(f)] for f in frames]],
                 heel_xy=np.stack([uv[:, HEEL[0]], uv[:, HEEL[1]]], 1).astype(np.float32),
                 toe_xy=np.stack([uv[:, list(TOES[0])].mean(1), uv[:, list(TOES[1])].mean(1)], 1).astype(np.float32),
                 channels=np.array(CH), hk_channels=np.array(HK), thr=np.float32(a.thr), fps=np.float32(t["fps"]),
                 px_valid=np.bool_("uv" in t.files))
        for head, Q in (("image", P), ("video", np.stack([vid[i] for i in t["ids"].astype(str)]) if vid and
                                         all(i in vid for i in t["ids"].astype(str)) else None)):
            if Q is None:
                continue
            d.update({f"{head}_static_prob": Q[:, :4].astype(np.float32), f"{head}_any_prob": Q[:, 8:12].astype(np.float32),
                      f"{head}_hk_static_prob": Q[:, 4:8].astype(np.float32), f"{head}_hk_any_prob": Q[:, 12:16].astype(np.float32),
                      f"{head}_static": Q[:, :4] >= a.thr, f"{head}_any": Q[:, 8:12] >= a.thr})
        tmp = out / f".{c}__0.contact.tmp.npz"  # atomic: readers never see a half-written file
        np.savez(tmp, **d)
        os.replace(tmp, out / f"{c}__0.contact.npz")
        rep[c] = dict(frames=len(frames), video="video_static_prob" in d, image_static=round(float(d["image_static"].mean()), 3),
                      image_any=round(float(d["image_any"].mean()), 3),
                      video_static=round(float(d["video_static"].mean()), 3) if "video_static" in d else None)
    ix = out / f"index_{a.set}.json"
    if os.environ.get("CLIPS_ONLY") and ix.exists():  # several runs into one dir: merge into the index
        rep = {**json.load(open(ix)), **rep}
    ix.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "export"])
    ap.add_argument("--set", default="clip", choices=sorted(SETS))
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", default="", help="export: the contact files' folder")
    ap.add_argument("--thr", type=float, default=0.5)
    a = ap.parse_args()
    if a.stage == "export" and not a.out:
        ap.error("export needs --out")
    prep(a) if a.stage == "prep" else export(a)


if __name__ == "__main__":
    main()
