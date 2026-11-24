"""Camera trajectories from the learned camera-motion head (train/camera_motion.py), causal: every --stride-th frame is fed
and its pose = the previous fed frame's pose x the head's relative pose (previous fed frame -> this frame); skipped frames
hold the last pose. Hand tokens: the padded predicted hand boxes of frame t - --mask_delay (--mask_run pickles; what the
concurrent pipeline can pass on). Frames from the world frame caches (bench/world_frames.py frames704 / EgoStandard frame
dirs with K.npy); the model sees the centred 4:3 crop of the frame at its input size.
    python -m bench.world_camhead --ckpt <model.pt> --frames <dir> --mask_run <run> --out <dir> --stride 2
-> <out>/<clip>.npz: T_world_cam = T_world_cam_d0 (metric, scale_causal 1), rel (fed, 4, 4) relative pose of each fed step
   (identity for the first), logvar (fed, 2) predicted log-variances (rotation deg, translation cm), rel_lag<k> for --lags,
   t_ms_per_fed (backbone + pair, batched)
"""
import argparse
import glob
import os
import pickle
import time

import cv2
import numpy as np
import torch

from train.camera_motion import CameraMotion, box_tokens, crop_K, crop_image, crop_params


def load_head(ckpt, dev):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck["args"]
    size = tuple(int(x) for x in a["size"].split("x"))
    model = CameraMotion(a["backbone"], size, a["dim"], a["layers"], pretrained=False, corr=not a.get("no_corr", True),
                         raft=a.get("raft", False))
    model.load_state_dict(ck["model"])
    return model.to(dev).eval()


def main(args):
    dev = torch.device("cuda")
    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    model = load_head(args.ckpt, dev)
    size = model.size
    lags = [int(x) for x in args.lags.split(",")] if args.lags else []
    os.makedirs(args.out, exist_ok=True)
    clips = sorted(d for d in os.listdir(args.frames) if os.path.exists(os.path.join(args.frames, d, "done")))
    for name in clips[args.shard::args.num_shards]:
        out_path = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(out_path):
            continue
        files = sorted(glob.glob(os.path.join(args.frames, name, "*.jpg")))
        K0 = np.load(os.path.join(args.frames, name, "K.npy"))
        n = len(files)
        hw = cv2.imread(files[0]).shape[:2]
        crop = crop_params(hw, size)
        K = torch.as_tensor(crop_K(K0, crop, size), dtype=torch.float32, device=dev)[None]
        frames = {f["index"]: f for f in pickle.load(open(os.path.join(args.mask_run, f"{name}.pkl"), "rb"))["frames"]}
        fed = list(range(0, n, args.stride))
        imgs, hands = [], []
        for i in fed:  # decode / crop / boxes: shared with the hand pipeline, not timed
            imgs.append(crop_image(cv2.imread(files[i]), crop, size)[:, :, ::-1].transpose(2, 0, 1))
            src = frames.get(i - args.mask_delay)
            b = np.zeros((0, 5), np.float32)
            if src is not None and not args.no_mask:
                s = hw[1] / src["image_size"][1]
                b = np.array([[*(np.asarray(p["bbox"], np.float64) * s), 1.0] for p in src["pred"]], np.float32).reshape(-1, 5)
            hands.append(box_tokens(b, crop, size).reshape(-1))
        x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs))).to(dev)
        hm = torch.from_numpy(np.stack(hands)).to(dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            tok = torch.cat([model.encode(x[k:k + 64]) for k in range(0, len(x), 64)])
            ray = model.rays(K)

            def pairs(lag):
                a = torch.arange(len(fed) - lag, device=dev)
                R, t, lr, lt = [], [], [], []
                for k in range(0, len(a), 128):
                    ia, ib = a[k:k + 128], a[k:k + 128] + lag
                    im = (x[ia], x[ib], K.expand(len(ia), -1, -1)) if model.use_raft else ()
                    o = model.pair(tok[ia], tok[ib], ray.expand(len(ia), -1, -1), hm[ia], hm[ib], *im)
                    R.append(o[0]), t.append(o[1]), lr.append(o[2]), lt.append(o[3])
                M = torch.eye(4, device=dev).repeat(len(a), 1, 1)
                M[:, :3, :3], M[:, :3, 3] = torch.cat(R), torch.cat(t)
                return M.double().cpu().numpy(), torch.stack([torch.cat(lr), torch.cat(lt)], 1).cpu().numpy()

            rel1, lv = pairs(1)
            extra = {f"rel_lag{k}": pairs(k)[0] for k in lags if k > 1 and k < len(fed)}
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        Tf = np.tile(np.eye(4), (len(fed), 1, 1))
        for j in range(1, len(fed)):
            Tf[j] = Tf[j - 1] @ rel1[j - 1]
        T = Tf[np.searchsorted(np.array(fed), np.arange(n), side="right") - 1]
        rel = np.concatenate([np.eye(4)[None], rel1])
        logvar = np.concatenate([np.zeros((1, 2)), lv])
        tmp = f"{out_path}.{os.getpid()}.npz"
        np.savez(tmp, index=np.arange(n), T_world_cam=T, T_world_cam_d0=T, scale_causal=1.0, rel=rel, logvar=logvar,
                 n_fed=len(fed), t_ms_per_fed=(t1 - t0) * 1000 / len(fed), **extra)
        os.replace(tmp, out_path)
        print(f"[camhead] {name}: {n} frames, {len(fed)} fed, {(t1 - t0) * 1000 / len(fed):.2f} ms / fed frame (batched)", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--frames", required=True)
    p.add_argument("--mask_run", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--mask_delay", type=int, default=1)
    p.add_argument("--no_mask", action="store_true")
    p.add_argument("--lags", default="", help="also store the relative poses at these fed-frame lags, e.g. 4,8")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
