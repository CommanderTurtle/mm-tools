"""Frame cache for the camera-motion head (train/camera_motion.py): small frames + metric camera poses + hand boxes.
  hot3d : HOT3D-Clips tars (bench.datasets.hot3d: rectified upright pinhole, Aria MPS T_world_cam, boxes from the GT MANO
          meshes of all hands in view)
  egodex: EgoDex episodes (train.egodex: ARKit T_world_cam + intrinsics, boxes from the ARKit joints padded 0.3 since its
          2D projections are 5-15 % of the hand size off)
  arctic: ARCTIC egocentric view, protocol-p2 train subjects (undistorted; mocap ego-camera poses; MANO-mesh boxes)
  hoi4d : HOI4D released RGB videos (reconstruction camera poses; boxes from the annotated 2D hand keypoints)
-> <out>/<source>/<clip>/frames.bin (concatenated JPEGs, --size long side) + meta.npz: offsets (n + 1), T_world_cam (n, 4, 4),
   K (3, 3) at the cached size, boxes (n, 2, 5) x0 y0 x1 y1 valid (cached px), fps, sequence (HOT3D sequence id / EgoDex
   task: held-out splits are by sequence). One file pair per clip (network-file-system friendly); a clip with meta.npz is done (resumable).
    python -m train.build_cam_frames hot3d --src $DATA_ROOT/hot3d_train600 --out $DATA_ROOT/camframes --shard 0 --num_shards 8
    python -m train.build_cam_frames egodex --src $DATA_ROOT/egodex/part2 --out $DATA_ROOT/camframes --max_frames 450
    python -m train.build_cam_frames arctic --src $DATA_ROOT/arctic/downloads/data --out $DATA_ROOT/camframes
    python -m train.build_cam_frames hoi4d --src $DATA_ROOT/hoi4d --out $DATA_ROOT/camframes
"""
import argparse
import glob
import os

import cv2
import numpy as np


def write_clip(out_dir, frames, T, K, boxes, fps, sequence, size):
    """frames: iterable (generator: the clip is never held in memory) of BGR images, all the same size; K for that size;
    T (n, 4, 4) and boxes (n, 2, 5) in its pixels -- lists filled while the generator runs are fine."""
    os.makedirs(out_dir, exist_ok=True)
    offs, s = [0], None
    with open(os.path.join(out_dir, "frames.bin.tmp"), "wb") as fb:
        for img in frames:
            if s is None:
                s = size / max(img.shape[:2])
            small = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
            buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            fb.write(buf)
            offs.append(offs[-1] + len(buf))
    n = len(offs) - 1
    T, boxes = np.asarray(T), np.asarray(boxes)
    K = np.asarray(K, np.float64).copy()
    K[:2] *= s  # pixel-centre offsets are negligible at this scale
    boxes = boxes[:n].astype(np.float32).copy()
    boxes[..., :4] *= s
    os.replace(os.path.join(out_dir, "frames.bin.tmp"), os.path.join(out_dir, "frames.bin"))
    np.savez(os.path.join(out_dir, "meta.tmp.npz"), offsets=np.array(offs, np.int64), T_world_cam=T[:n].astype(np.float64), K=K,
             boxes=boxes, fps=float(fps), sequence=str(sequence), size=np.array(small.shape[:2]))
    os.replace(os.path.join(out_dir, "meta.tmp.npz"), os.path.join(out_dir, "meta.npz"))
    return n


def hot3d(args):
    from bench.datasets.hot3d import Hot3DClip

    tars = sorted(glob.glob(os.path.join(args.src, "*.tar")))[args.shard::args.num_shards]
    for tp in tars:
        name = os.path.basename(tp)[:-4]
        out = os.path.join(args.out, "hot3d", name)
        if os.path.exists(os.path.join(out, "meta.npz")):
            continue
        try:
            clip = Hot3DClip(tp, args.mano_dir)
        except Exception as e:  # e.g. Quest3 clips without the Aria RGB stream
            print(f"[skip] {name}: {e}", flush=True)
            continue
        if clip.device != "Aria":
            print(f"[skip] {name}: {clip.device}", flush=True)
            continue
        T, boxes = [], np.zeros((len(clip), 2, 5), np.float32)

        def images():
            for i in range(len(clip)):
                fr = clip.frame(i)
                T.append(fr["T_world_cam"])
                for h in fr["hands"] + fr["ignore"]:
                    boxes[i, int(h["is_right"])] = [*h["bbox"], 1.0]
                yield fr["image"]

        clip.frame(0)  # sets up the rectification (clip.K)
        n = write_clip(out, images(), T, clip.K, boxes, 30.0, clip.sequence_id, args.size)
        print(f"[hot3d] {name}: {n} frames, sequence {clip.sequence_id}", flush=True)


def egodex(args):
    from train.egodex import EgoDexEpisode, list_episodes
    from train.release_guards import egodex_camera_holdout

    eps = list_episodes(args.src)
    eps = [p for p in eps if not egodex_camera_holdout(p)]
    if args.max_episodes:  # spread over the tasks: every k-th episode of the sorted list
        eps = eps[:: max(1, len(eps) // args.max_episodes)][: args.max_episodes]
    for hp in eps[args.shard::args.num_shards]:
        rel = os.path.relpath(os.path.splitext(hp)[0], args.src).replace(os.sep, "__")
        out = os.path.join(args.out, "egodex", rel)
        if os.path.exists(os.path.join(out, "meta.npz")):
            continue
        try:
            ep = EgoDexEpisode(hp, bbox_pad=0.3)
        except Exception as e:
            print(f"[skip] {rel}: {e}", flush=True)
            continue
        stop = min(len(ep), args.max_frames) if args.max_frames else len(ep)
        if stop < 32:
            continue
        boxes = np.zeros((stop, 2, 5), np.float32)
        for i in range(stop):
            for h in (ep.hand_annotation(i, "left"), ep.hand_annotation(i, "right")):
                if h["bbox"] is not None:
                    boxes[i, int(h["is_right"])] = [*h["bbox"], 1.0]
        imgs = (fr["image"] for fr in ep.frames(0, stop))
        n = write_clip(out, imgs, ep.T_world_cam[:stop], ep.K, boxes, ep.fps or 30.0, ep.task, args.size)
        print(f"[egodex] {rel}: {n} frames, task {ep.task}", flush=True)


def arctic(args):
    """ARCTIC egocentric view (view 0 of the official cropped_images: the full ego frame at x0.3, distorted), protocol-p2
    train subjects only (s03 / s05 = test / val never cached). Frames undistorted with the 8-coefficient model (same K);
    T_world_cam = inv([R_k_cam | T_k_cam]) (mocap world -> ego camera, m); boxes from the projected MANO meshes; mocap frame
    vidx = image index - the subject's ioi_offset (meta/misc.json)."""
    import json
    import zipfile

    import smplx
    import torch

    A = args.src  # .../arctic/repo/downloads/data
    split = json.loads(zipfile.ZipFile(os.path.join(A, "splits_json.zip")).read("splits_json/protocol_p2.json"))
    misc = json.loads(zipfile.ZipFile(os.path.join(A, "meta.zip")).read("meta/misc.json"))
    raw = zipfile.ZipFile(os.path.join(A, "raw_seqs.zip"))
    mano = {r: smplx.MANO(args.mano_dir, is_rhand=r, use_pca=False, flat_hand_mean=False) for r in (True, False)}
    for seq in sorted(split["train"])[args.shard::args.num_shards]:
        s, name = seq.split("/")
        out = os.path.join(args.out, "arctic", f"{s}_{name}")
        if os.path.exists(os.path.join(out, "meta.npz")):
            continue
        cam = np.load(raw.open(f"raw_seqs/{s}/{name}.egocam.dist.npy"), allow_pickle=True).item()
        mp = np.load(raw.open(f"raw_seqs/{s}/{name}.mano.npy"), allow_pickle=True).item()
        zi = zipfile.ZipFile(os.path.join(A, "cropped_images_zips", s, f"{name}.zip"))
        off, n_mocap = int(misc[s]["ioi_offset"]), len(cam["R_k_cam_np"])
        have = {int(os.path.basename(x)[:-4]) for x in zi.namelist() if x.startswith("0/") and x.endswith(".jpg")}
        vids = [v for v in range(n_mocap) if v + off in have]
        if len(vids) < 32:
            continue
        K = np.asarray(cam["intrinsics"], np.float64).copy()
        K[:2] *= 0.3  # the cropped_images ego frames are the full frame at x0.3
        dist = np.asarray(cam["dist8"], np.float64)
        img0 = cv2.imdecode(np.frombuffer(zi.read(f"0/{vids[0] + off:05d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
        h0, w0 = img0.shape[:2]
        mx, my = cv2.initUndistortRectifyMap(K, dist, None, K, (w0, h0), cv2.CV_32FC1)
        Rk = np.asarray(cam["R_k_cam_np"], np.float64)[vids]
        Tk = np.asarray(cam["T_k_cam_np"], np.float64)[vids, :, 0]
        E = np.tile(np.eye(4), (len(vids), 1, 1))
        E[:, :3, :3], E[:, :3, 3] = Rk, Tk
        T = np.linalg.inv(E)
        boxes = np.zeros((len(vids), 2, 5), np.float32)
        with torch.no_grad():
            for side, r in (("right", True), ("left", False)):
                p = {k: torch.tensor(np.asarray(v, np.float32)) for k, v in mp[side].items()}
                v = mano[r](global_orient=p["rot"][vids], hand_pose=p["pose"][vids], betas=p["shape"][None].expand(len(vids), -1),
                            transl=p["trans"][vids]).vertices.numpy().astype(np.float64)
                vc = np.einsum("nij,nvj->nvi", Rk, v) + Tk[:, None]
                ok = vc[..., 2] > 0.05
                uv = vc[..., :2] / np.clip(vc[..., 2:], 0.05, None) * K[[0, 1], [0, 1]] + K[[0, 1], [2, 2]]
                for i in range(len(vids)):
                    q = uv[i][ok[i]]
                    if len(q) < 50:
                        continue
                    x0, y0 = np.clip(q.min(0), 0, [w0 - 1, h0 - 1])
                    x1, y1 = np.clip(q.max(0), 0, [w0 - 1, h0 - 1])
                    if x1 - x0 > 4 and y1 - y0 > 4:
                        boxes[i, int(r)] = [x0, y0, x1, y1, 1.0]

        def images():
            for v in vids:
                img = cv2.imdecode(np.frombuffer(zi.read(f"0/{v + off:05d}.jpg"), np.uint8), cv2.IMREAD_COLOR)
                yield cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        n = write_clip(out, images(), T, K, boxes, 30.0, seq, args.size)
        print(f"[arctic] {seq}: {n} frames", flush=True)


def hoi4d(args):
    """HOI4D (released sequences; head-mounted Kinect RGB, 1920 x 1080 at 15 fps): align_rgb/image.mp4 from
    downloads/HOI4D_release.zip, T_world_cam from HOI4D_annotations/<seq>/3Dseg/output.log (Redwood .log: one camera-to-world
    pose per frame, m, first = identity), K = camera_params/<ZY id>/intrin.npy, boxes from the annotated 2D hand keypoints
    (HOI4D_Hand_pose kps2D) padded 0.3. sequence = the object instance (ZY / H / C / N), so held-out splits keep instances."""
    import pickle
    import zipfile

    cv2.setNumThreads(1)
    H = args.src  # .../hoi4d
    ann = os.path.join(H, "official", "HOI4D_annotations", "HOI4D_annotations")
    hp = os.path.join(H, "official", "HOI4D_Hand_pose", "Hand_pose")
    z = zipfile.ZipFile(os.path.join(H, "downloads", "HOI4D_release.zip"))
    vids = sorted(x for x in z.namelist() if x.endswith("align_rgb/image.mp4"))
    for v in vids[args.shard::args.num_shards]:
        seq = v[len("HOI4D_release/"):-len("/align_rgb/image.mp4")]
        out = os.path.join(args.out, "hoi4d", seq.replace("/", "_"))
        log = os.path.join(ann, seq, "3Dseg", "output.log")
        if os.path.exists(os.path.join(out, "meta.npz")) or not os.path.exists(log):
            continue
        L = open(log).read().split()
        T = np.array([np.array(L[i + 3:i + 19], np.float64).reshape(4, 4) for i in range(0, len(L) - 18, 19)])
        K = np.load(os.path.join(H, "official", "camera_params", "camera_params", seq.split("/")[0], "intrin.npy"))
        n = len(T)
        boxes = np.zeros((n, 2, 5), np.float32)
        for side, r in (("left", 0), ("right", 1)):
            d = os.path.join(hp, f"handpose_{side}_hand", seq)
            for i in range(n):
                f = os.path.join(d, f"{i}.pickle")
                if not os.path.exists(f):
                    continue
                k = np.asarray(pickle.load(open(f, "rb"))["kps2D"], np.float64).reshape(-1, 2)
                if len(k) < 2 or not np.isfinite(k).all():
                    continue
                (x0, y0), (x1, y1) = k.min(0), k.max(0)
                p = 0.3 * max(x1 - x0, y1 - y0)
                boxes[i, r] = [x0 - p, y0 - p, x1 + p, y1 + p, 1.0]
        tmp = f"/tmp/hoi4d_{os.getpid()}.mp4"
        with open(tmp, "wb") as f:
            f.write(z.read(v))

        def images(cap=cv2.VideoCapture(tmp)):  # streamed: a clip is never held in memory
            for _ in range(n):
                ok, img = cap.read()
                if not ok:
                    return
                yield img

        n = write_clip(out, images(), T, K, boxes, 15.0, "/".join(seq.split("/")[:4]), args.size)
        os.remove(tmp)
        print(f"[hoi4d] {seq}: {n} frames", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", choices=["hot3d", "egodex", "arctic", "hoi4d"])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=320, help="cached long side (px)")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", "data/mano"))
    p.add_argument("--max_frames", type=int, default=450, help="egodex: frames per episode (0 = all)")
    p.add_argument("--max_episodes", type=int, default=0, help="egodex: episodes (0 = all), spread over the task list")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    args = p.parse_args()
    dict(hot3d=hot3d, egodex=egodex, arctic=arctic, hoi4d=hoi4d)[args.source](args)
