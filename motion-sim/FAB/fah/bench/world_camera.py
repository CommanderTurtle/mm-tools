"""Steps of the FAH world camera around bench/world_dpvo.py, bench/world_camhead.py and bench/world_fuse.py
(docs/world.md has the full sequence):
  frames     a video -> frame dir (<out>/<name>/%05d.jpg + K.npy + done), the input of the camera stages
  hands      FAH video hands (fah.FAH video model, mesh stage off) on frame dirs -> bench.run-style pickles (the hand masks of
             the camera stages and the hands placed in the world)
  calibrate  causal DPVO npz dir -> copy with cal = the HOT3D-train metric-scale correction and scale_cal = cal x scale_causal
  gate       translation gate on a causal DPVO npz dir with sig_res (bench/world_dpvo.py --signals): a fed step whose
             rotation-only residual < g x the full-pose residual keeps the camera centre

    python -m bench.world_camera frames --video clip.mp4 --out frames/ [--focal 600]
    python -m bench.world_camera hands --frames frames/ --out hands/ --model fah
    python -m bench.world_camera calibrate --src cams/dpvo --dst cams/dpvo_cal --cal 1.034
    python -m bench.world_camera gate --src cams/dpvo_cal --dst cams/dpvo_cal_g15 --g 1.5
"""
import argparse
import glob
import os
import pickle

import numpy as np


def frames(args):
    import cv2

    name = os.path.splitext(os.path.basename(args.video))[0]
    d = os.path.join(args.out, name)
    os.makedirs(d, exist_ok=True)
    cap, i, hw = cv2.VideoCapture(args.video), 0, None
    while args.max_frames <= 0 or i < args.max_frames:
        ok, img = cap.read()
        if not ok:
            break
        cv2.imwrite(os.path.join(d, f"{i:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        hw, i = img.shape[:2], i + 1
    h, w = hw
    f = args.focal or (h ** 2 + w ** 2) ** 0.5  # default: focal = image diagonal (as the hand model)
    np.save(os.path.join(d, "K.npy"), np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64))
    open(os.path.join(d, "done"), "w").close()
    print(d, i, "frames")


def hands(args):
    import cv2

    from fah.api import FAH

    paths = {k: v for k, v in dict(weights=args.weights, sam3d=args.sam3d).items() if v}
    model = FAH(args.model, "world", trt_dir=args.trt_dir, mano_dir=args.mano_dir, mesh="none", **paths)  # the camera stages need the hand boxes only
    os.makedirs(args.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(args.frames) if os.path.exists(os.path.join(args.frames, d, "done")))
    for c in clips[args.shard::args.num_shards]:
        out = os.path.join(args.out, c + ".pkl")
        if os.path.exists(out):
            continue
        K = np.load(os.path.join(args.frames, c, "K.npy")).astype(np.float32)
        model.reset()
        fr = []
        for i, f in enumerate(sorted(glob.glob(os.path.join(args.frames, c, "*.jpg")))):
            img = cv2.imread(f)
            fr.append(dict(key=f"{i:05d}", index=i, K=K, pred=model(img, K), image_size=img.shape[:2]))
        pickle.dump(dict(clip=c, method=model.name, frames=fr, video=True), open(out + ".tmp", "wb"))
        os.replace(out + ".tmp", out)
        print(c, len(fr), flush=True)


def partial(name):
    """A stage's temporary output (<clip>.npz.<pid>.npz, renamed when complete); clip names may contain dots."""
    return ".npz." in name


def calibrate(args):
    os.makedirs(args.dst, exist_ok=True)
    for f in glob.glob(f"{args.src}/*.npz"):
        if partial(os.path.basename(f)):
            continue
        d = dict(np.load(f))
        d["cal"] = args.cal
        if "scale_causal" in d:
            d["scale_cal"] = args.cal * float(d["scale_causal"])
        np.savez(os.path.join(args.dst, os.path.basename(f)), **d)
    print(args.dst, args.cal, len(glob.glob(args.dst + "/*.npz")))


def gate(args):
    os.makedirs(args.dst, exist_ok=True)
    share = []
    for f in sorted(glob.glob(args.src + "/*.npz")):
        if partial(os.path.basename(f)):
            continue
        d = dict(np.load(f))
        T = d["T_world_cam_d0"].copy()
        n, nf = len(T), int(d["n_fed"])
        st = int(np.ceil(n / nf))
        r = d["sig_res"]
        g = (r[:, 0] > 0) & (r[:, 1] < args.g * r[:, 0])
        step = np.arange(n) // st
        gf = g[step] & (np.arange(n) % st == 0)  # the frame where the step lands
        p = T[:, :3, 3]
        inc = np.diff(p, axis=0)
        inc[gf[1:]] = 0
        T[:, :3, 3] = np.concatenate([p[:1], p[:1] + np.cumsum(inc, 0)])
        d.update(T_world_cam_d0=T, T_world_cam=T, gated=g)
        np.savez(os.path.join(args.dst, os.path.basename(f)), **d)
        share.append(g.mean())
    print(args.dst, len(share), "gated step share mean %.3f" % np.mean(share))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("frames")
    a.add_argument("--video", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--focal", type=float, default=0)
    a.add_argument("--max_frames", type=int, default=0)
    a = sub.add_parser("hands")
    a.add_argument("--frames", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--model", default="fah")
    a.add_argument("--weights", default=None, help="default: fah.api.DEFAULT_WEIGHTS")
    a.add_argument("--sam3d", default=None, help="default: fah.api.DEFAULT_SAM3D")
    a.add_argument("--trt_dir", default="")
    a.add_argument("--mano_dir", default=None)
    a.add_argument("--shard", type=int, default=0)
    a.add_argument("--num_shards", type=int, default=1)
    a = sub.add_parser("calibrate")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    a.add_argument("--cal", type=float, default=1.0338939640317917, help="HOT3D-train correction of the MoGe-2 scale calls (bench.world_calib fit)")
    a = sub.add_parser("gate")
    a.add_argument("--src", required=True)
    a.add_argument("--dst", required=True)
    a.add_argument("--g", type=float, default=1.5)
    args = p.parse_args()
    {"frames": frames, "hands": hands, "calibrate": calibrate, "gate": gate}[args.cmd](args)
