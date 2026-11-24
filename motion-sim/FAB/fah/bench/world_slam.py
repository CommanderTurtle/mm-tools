"""Offline camera of FAH world space (camera "offline", docs/world.md): masked DROID-SLAM trajectory with a per-clip
metric scale from Metric3D depth; fah.world.fuse_offline then combines it with the real-time camera. Runs in a
DROID-SLAM environment (fah.world.estimate_cameras starts it as a subprocess):
  - frames as JPEGs (--frames: one dir per clip with *.jpg + K.npy + done; or a benchmark dataset, --dataset / --split /
    --data_root as bench.run), hand pixels masked with the hands' fitted meshes (--mask_run: bench.run-style pickles
    with verts3d for every frame, all hands rasterised);
  - masked DROID-SLAM (the mask-aware DROID-SLAM of TRAM, droid.pth, DROID's default arguments, ~384 x 512 area input,
    upsampled disparities); if the global bundle adjustment has no proximity factors (a near-static clip), it is skipped
    and the frontend poses are kept (global_ba_failed in the npz);
  - Metric3D ViT-L depth on every keyframe; per keyframe the scale of the SLAM depth (1 / disparity) to the metric depth
    outside the hands, in a 0.4-0.7 m depth band widened by 0.1 m on each side while undefined (the widening persists);
    the clip scale is the median over keyframes.
-> <out>/<clip>.npz: index, T_world_cam (N, 4, 4: camera -> SLAM world, metric), traj (DROID c2w, unscaled xyz +
   quaternion xyzw), scale, kf_scales, tstamp (keyframes), kf_disps / kf_depths (1/4 resolution, fp16), K, slam_size,
   timings (s), global_ba_failed

The DROID input stream, the mask resizing, the masked tracking loop and the depth-scale fit (scale_fit) are adapted
from TRAM (https://github.com/yufu-wang/tram, lib/camera/slam_utils.py, masked_droid_slam.py, est_scale.py), which builds
on DROID-SLAM (https://github.com/princeton-vl/DROID-SLAM; BSD 3-Clause License, Copyright (c) 2021 Princeton Vision &
Learning Lab; THIRD_PARTY_NOTICES.md). TRAM's licence:

    MIT License

    Copyright (c) 2024 Yufu Wang

    Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
    documentation files (the "Software"), to deal in the Software without restriction, including without limitation
    the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
    permit persons to whom the Software is furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all copies or substantial portions of
    the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
    THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
    TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

    python bench/world_slam.py --frames <dir> --mask_run <hands dir> --out <dir> --droid_weights droid.pth \
        --metric3d_dir <Metric3D checkout> --metric3d_weights metric_depth_vit_large_800k.pth
    (PYTHONPATH: fah/_vendor, this repository, TRAM's thirdparty/DROID-SLAM and its droid_slam/, Metric3D's checkout)
"""
import argparse
import glob
import math
import os
import pickle
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import cv2
import numpy as np
import torch


def droid_args(weights, image_size):
    """DROID-SLAM's default arguments (its demo parser), monocular, upsampled disparities, no viewer."""
    return SimpleNamespace(weights=weights, buffer=512, image_size=list(image_size), disable_vis=True, stereo=False, upsample=True,
                           beta=0.3, filter_thresh=2.4, warmup=8, keyframe_thresh=4.0, frontend_thresh=16.0, frontend_window=25,
                           frontend_radius=2, frontend_nms=1, backend_thresh=22.0, backend_radius=2, backend_nms=3,
                           reconstruction_path=None, t0=0, stride=1)


def slam_size(h0, w0):
    """DROID's input size: the ~384 x 512 area resize, cropped to a multiple of 8 -> (resized (h1, w1), cropped (H, W))."""
    h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
    w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))
    return (h1, w1), (h1 - h1 % 8, w1 - w1 % 8)


def image_stream(imgfiles, calib):
    """DROID input: (t, image (1, 3, H, W) uint8 BGR, intrinsics at that size) per frame."""
    fx, fy, cx, cy = calib[:4]
    for t, f in enumerate(imgfiles):
        image = cv2.imread(f)
        h0, w0 = image.shape[:2]
        (h1, w1), (H, W) = slam_size(h0, w0)
        image = cv2.resize(image, (w1, h1))[:H, :W]
        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= w1 / w0
        intrinsics[1::2] *= h1 / h0
        yield t, torch.as_tensor(image).permute(2, 0, 1)[None], intrinsics


def resize_masks(masks, H, W):
    """(N, h0, w0) hand masks -> image masks (N, H, W) and DROID confidence masks (N, H / 8, W / 8)."""
    from torchvision.transforms import Resize

    r1, r2 = Resize((H, W), antialias=True), Resize((H // 8, W // 8), antialias=True)
    return (torch.cat([r1(masks[i:i + 500]) for i in range(0, len(masks), 500)]),
            torch.cat([r2(masks[i:i + 500]) for i in range(0, len(masks), 500)]))


def run_masked_droid(imgfiles, masks, calib, weights):
    """Masked DROID-SLAM over the frames -> (droid, traj (N, 7) c2w xyz + quaternion xyzw, number of skipped global BAs)."""
    import droid_backend
    from droid import Droid

    failed = []
    ba = droid_backend.DroidBackend.__call__

    def safe_ba(self, steps=12):  # near-static clips: too few keyframes for proximity factors -> keep the frontend poses
        try:
            return ba(self, steps)
        except ValueError:
            failed.append(steps)

    droid_backend.DroidBackend.__call__ = safe_ba
    try:
        H, W = slam_size(*cv2.imread(imgfiles[0]).shape[:2])[1]
        img_msks, conf_msks = resize_masks(masks, H, W)
        droid = None
        for t, image, intrinsics in image_stream(imgfiles, calib):
            if droid is None:
                droid = Droid(droid_args(weights, image.shape[2:]))
            droid.track(t, image * (img_msks[t] < 0.5), intrinsics=intrinsics, depth=None, mask=conf_msks[t])
        traj = droid.terminate(image_stream(imgfiles, calib))
    finally:
        droid_backend.DroidBackend.__call__ = ba
    return droid, traj, len(failed)


def gmof(x, sigma):
    """Geman-McClure error."""
    return (sigma ** 2 * x ** 2) / (sigma ** 2 + x ** 2)


def scale_fit(slam_depth, pred_depth, msk, near, far, sigma=0.5):
    """Scale of slam_depth to the metric pred_depth outside the hand mask with both depths in (near, far): iterated
    median ratio, then a Geman-McClure robust fit (BFGS); nan if no pixel qualifies."""
    from torchmin import minimize

    msk = cv2.resize(msk, (pred_depth.shape[1], pred_depth.shape[0]))
    s = pred_depth / slam_depth
    ok = (msk < 0.5) & (near < pred_depth) & (pred_depth < far)
    scale = np.median(s[ok])
    for _ in range(10):
        d0 = slam_depth * scale
        ok = (msk < 0.5) & (0 < d0) & (d0 < far) & (near < pred_depth) & (pred_depth < far)
        scale = np.median(s[ok])
    # the robust fit keeps the last iteration's support (d0 from the scale before the final median)
    pm, sm = torch.from_numpy(pred_depth[ok]), torch.from_numpy(slam_depth[ok])
    res = minimize(lambda x: gmof(sm * x - pm, sigma).mean(), torch.tensor([scale]), method="bfgs")
    return res.x.detach().cpu().item()


def clip_scale(disps, depths, masks):
    """Per-keyframe scales in the 0.4-0.7 m band, widened by 0.1 m on each side while undefined (the widening persists
    for the later keyframes; bounded at far < 10) -> (clip scale = median, per-keyframe scales)."""
    lo, hi, scales = 0.4, 0.7, []
    for disp, depth, msk in zip(disps, depths, masks):
        s = scale_fit(1 / disp, depth, msk, lo, hi)
        while math.isnan(s) and hi < 10:
            lo, hi = lo - 0.1, hi + 0.1
            s = scale_fit(1 / disp, depth, msk, lo, hi)
        scales.append(s)
    scales = np.array(scales)
    return (float(np.nanmedian(scales)) if np.isfinite(scales).any() else 1.0), scales


# Metric3DDepth follows Metric3D's inference path (https://github.com/YvanYin/Metric3D; BSD 2-Clause License, Copyright
# (c) 2024 Wei Yin and Mu Hu; THIRD_PARTY_NOTICES.md)
class Metric3DDepth:
    """Metric3D ViT-L (its own inference path, mono/utils/do_test.py): metric depth (m) of a BGR image at its resolution."""

    def __init__(self, repo, weights):
        sys.path[:0] = [p for p in (repo,) if p not in sys.path]
        from mmengine import Config
        from mono.model.monodepth_model import get_configured_monodepth_model
        from mono.utils.mldb import load_data_info, reset_ckpt_path
        from mono.utils.running import load_ckpt

        cfg = Config.fromfile(os.path.join(repo, "mono", "configs", "HourglassDecoder", "vit.raft5.large.py"))
        cfg.load_from = weights
        info = {}
        load_data_info("data_info", data_info=info)
        cfg.mldb_info = info
        reset_ckpt_path(cfg.model, info)
        cfg.distributed = False
        model = torch.nn.DataParallel(get_configured_monodepth_model(cfg)).cuda()
        self.model, _, _, _ = load_ckpt(cfg.load_from, model, strict_match=False)
        self.model.eval()
        self.cfg = cfg

    @torch.no_grad()
    def __call__(self, bgr, calib):
        from mono.utils.do_test import transform_test_data_scalecano

        rgb = bgr[:, :, ::-1].copy()
        x, cam, pad, label_scale = transform_test_data_scalecano(rgb, list(calib), self.cfg.data_basic)
        depth, _, _ = self.model.module.inference(dict(input=x[None], cam_model=cam))
        depth = depth.squeeze()
        depth = depth[pad[0]:depth.shape[0] - pad[1], pad[2]:depth.shape[1] - pad[3]]
        depth = torch.nn.functional.interpolate(depth[None, None], rgb.shape[:2], mode="bilinear").squeeze()
        depth = depth * self.cfg.data_basic.depth_range[1] / label_scale
        return ((depth > 0) & (depth < 300)).float().mul(depth).cpu().numpy()


def hand_masks(frames, shape, faces):
    """(N, H, W) uint8 masks of the hands' fitted meshes (all hands of a frame)."""
    H, W = shape
    out = np.zeros((len(frames), H, W), np.uint8)
    for i, fr in enumerate(frames):
        for p in fr["pred"]:
            v = p.get("verts3d")
            if v is None or not np.isfinite(v).all():
                continue
            K = fr["K"]
            uv = v[:, :2] / np.clip(v[:, 2:], 1e-6, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
            tri = faces[(v[faces, 2] > 0.01).all(1)]
            cv2.fillPoly(out[i], list(np.round(uv[tri]).astype(np.int32)), 1)
    return out


def main(args):
    from scipy.spatial.transform import Rotation

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    os.makedirs(args.out, exist_ok=True)
    faces = pickle.load(open(os.path.join(args.mano_dir, "MANO_RIGHT.pkl"), "rb"))["f"].astype(np.int64)
    metric = None
    if args.frames:  # frame dirs (*.jpg + K.npy + done)
        names = sorted(d for d in os.listdir(args.frames) if os.path.exists(os.path.join(args.frames, d, "done")))
        clips = [(n, None) for n in names[args.shard::args.num_shards]]
    else:
        from bench.run import iter_clips

        clips = iter_clips(args)
    for name, make_clip in clips:
        out_path = os.path.join(args.out, f"{name}.npz")
        if os.path.exists(out_path):
            continue
        tmp = tempfile.mkdtemp(prefix="fah_slam_")
        if make_clip is None:
            imgfiles = sorted(glob.glob(os.path.join(args.frames, name, "*.jpg")))
            K = np.load(os.path.join(args.frames, name, "K.npy"))
            H0, W0 = cv2.imread(imgfiles[0]).shape[:2]
        else:  # benchmark clips: frames as q95 JPEGs, the first frame's intrinsics
            clip = make_clip()
            imgfiles, Ks = [], []
            for i in range(len(clip)):
                fr = clip.frame(i)
                imgfiles.append(os.path.join(tmp, f"{i:05d}.jpg"))
                cv2.imwrite(imgfiles[-1], fr["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
                Ks.append(fr["K"])
            K = Ks[0]
            H0, W0 = fr["image"].shape[:2]
        mframes = pickle.load(open(os.path.join(args.mask_run, f"{name}.pkl"), "rb"))["frames"]
        assert [f["index"] for f in mframes] == list(range(len(imgfiles))), "the mask run must cover every frame (stride 1)"
        masks = torch.from_numpy(hand_masks(mframes, (H0, W0), faces))
        calib = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        droid, traj, ba_failed = run_masked_droid(imgfiles, masks, calib, args.droid_weights)
        n = droid.video.counter.value
        tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
        disps = droid.video.disps_up.cpu().numpy()[:n]
        torch.cuda.synchronize()
        t_slam = time.perf_counter() - t0
        del droid
        torch.cuda.empty_cache()

        if metric is None:
            metric = Metric3DDepth(args.metric3d_dir, args.metric3d_weights)
        H, W = slam_size(H0, W0)[1]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        depths = [cv2.resize(metric(cv2.imread(imgfiles[t]), calib), (W, H)) for t in tstamp]
        torch.cuda.synchronize()
        t_depth = time.perf_counter() - t0
        t0 = time.perf_counter()
        scale, scales = clip_scale(disps, depths, [masks[t].numpy().astype(np.uint8) for t in tstamp])
        t_scale = time.perf_counter() - t0

        T = np.tile(np.eye(4), (len(traj), 1, 1))
        T[:, :3, :3] = Rotation.from_quat(traj[:, 3:7]).as_matrix()
        T[:, :3, 3] = traj[:, :3] * scale
        tmp_out = f"{out_path}.{os.getpid()}.npz"
        np.savez(tmp_out, index=np.arange(len(traj)), T_world_cam=T, traj=traj, scale=scale, kf_scales=scales, tstamp=tstamp,
                 kf_disps=disps[:, ::4, ::4].astype(np.float16), kf_depths=np.stack(depths)[:, ::4, ::4].astype(np.float16), K=K,
                 slam_size=np.array([H, W]), t_slam=t_slam, t_depth=t_depth, t_scale=t_scale, num_frames=len(traj),
                 global_ba_failed=ba_failed)
        os.replace(tmp_out, out_path)
        shutil.rmtree(tmp, ignore_errors=True)
        print(f"[world_slam] {name}: {len(traj)} frames, {n} keyframes, scale {scale:.3f}, global BA skipped {ba_failed}, "
              f"slam {t_slam:.1f} s, depth {t_depth:.1f} s, scale fit {t_scale:.1f} s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default="", help="frame dirs (one per clip: *.jpg + K.npy + done)")
    p.add_argument("--dataset", default="hot3d", choices=["hot3d", "egostandard"], help="instead of --frames: benchmark clips")
    p.add_argument("--split", default="")
    p.add_argument("--data_root", default="")
    p.add_argument("--mask_run", required=True, help="hand pickles per clip (verts3d for every frame)")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--droid_weights", required=True, help="droid.pth (DROID-SLAM)")
    p.add_argument("--metric3d_dir", required=True, help="Metric3D checkout (its mono/ package)")
    p.add_argument("--metric3d_weights", required=True, help="metric_depth_vit_large_800k.pth")
    p.add_argument("--out", required=True)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    main(p.parse_args())
