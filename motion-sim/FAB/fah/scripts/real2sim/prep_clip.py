"""Real-to-sim clip preparation: one DexYCB sequence x camera (or any video) -> <out>/frames/%06d.jpg + video.mp4 +
clip.json (K, fps, source, licence) and, for DexYCB, gt.npz (grasped object's pose per frame in the camera frame, its
YCB mesh, MANO GT hand per frame, GT masks, camera -> table frame) for the quantitative checks.

    python scripts/real2sim/prep_clip.py --dexycb $DATA/dexycb/clips/20200918-subject-06/20200918_114030_836212060125.tar \
        --out $R2S/mustard
    python scripts/real2sim/prep_clip.py --video in.mp4 --focal 0 --licence "CC0 (...)" --out $R2S/my_clip
    python scripts/real2sim/prep_clip.py --hot3d $DATA/hot3d/train_aria/clip-001884.tar --out $R2S/h3_mug
"""
import argparse
import io
import json
import os
import sys
import tarfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))

YCB = {1: "002_master_chef_can", 2: "003_cracker_box", 3: "004_sugar_box", 4: "005_tomato_soup_can", 5: "006_mustard_bottle",
       6: "007_tuna_fish_can", 7: "008_pudding_box", 8: "009_gelatin_box", 9: "010_potted_meat_can", 10: "011_banana",
       11: "019_pitcher_base", 12: "021_bleach_cleanser", 13: "024_bowl", 14: "025_mug", 15: "035_power_drill",
       16: "036_wood_block", 17: "037_scissors", 18: "040_large_marker", 19: "051_large_clamp", 20: "052_extra_large_clamp",
       21: "061_foam_brick"}


def write_video(frames_dir, n, fps, path):
    first = cv2.imread(os.path.join(frames_dir, "000000.jpg"))
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0]))
    for i in range(n):
        vw.write(cv2.imread(os.path.join(frames_dir, f"{i:06d}.jpg")))
    vw.release()


def prep_dexycb(args):
    from bench.datasets.dexycb import DexYCBMano

    t = tarfile.open(args.dexycb)
    info = json.load(t.extractfile("info.json"))
    meta = info["meta"]
    gi = meta["ycb_grasp_ind"]
    ycb = meta["ycb_ids"][gi]
    n = info["num_frames"]
    mano = DexYCBMano(args.mano_dir)
    fd = os.path.join(args.out, "frames")
    os.makedirs(fd, exist_ok=True)
    pose, seg_obj, seg_hand, hv, hj, ann = [], [], [], [], [], []
    for i in range(n):
        open(os.path.join(fd, f"{i:06d}.jpg"), "wb").write(t.extractfile(f"color_{i:06d}.jpg").read())
        lab = np.load(io.BytesIO(t.extractfile(f"labels_{i:06d}.npz").read()))
        pose.append(np.vstack([lab["pose_y"][gi], [0, 0, 0, 1]]))
        seg_obj.append(lab["seg"] == ycb), seg_hand.append(lab["seg"] == 255)
        pm = lab["pose_m"][0]
        ann.append(bool(np.any(pm != 0)))
        j, v = mano(info["side"] == "right", np.asarray(info["betas"], np.float32), pm) if ann[-1] else (np.zeros((21, 3)), np.zeros((778, 3)))
        hj.append(j), hv.append(v)
    name = YCB[ycb]
    np.savez_compressed(os.path.join(args.out, "gt.npz"), T_cam_obj=np.stack(pose), seg_obj=np.stack(seg_obj),
                        seg_hand=np.stack(seg_hand), hand_verts=np.stack(hv).astype(np.float32),
                        hand_joints=np.stack(hj).astype(np.float32), hand_annotated=np.array(ann),
                        T_world_cam=np.asarray(info["T_world_cam"]), is_right=info["side"] == "right")
    clip = dict(source="DexYCB " + os.path.basename(args.dexycb)[:-4], subject=info["subject"], n=n, fps=info["fps"],
                hw=info["hw"], K=info["K"], object=name, ycb_id=ycb, side=info["side"],
                gt_mesh=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(args.dexycb))), "models", name, "textured_simple.obj"),
                licence=info["license"], static_camera=True)
    return clip


def prep_video(args):
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    fd = os.path.join(args.out, "frames")
    os.makedirs(fd, exist_ok=True)
    n, hw = 0, None
    while not args.max_frames or n - args.start < args.max_frames:
        ok, img = cap.read()
        if not ok:
            break
        if n >= args.start:
            if args.resize:
                s = args.resize / max(img.shape[:2])
                img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(fd, f"{n - args.start:06d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            hw = img.shape[:2]  # of the last written frame (the final failed read leaves img = None)
        n += 1
    cap.release()
    n = max(n - args.start, 0)
    if not n:
        raise SystemExit(f"no frames from {args.video} at --start {args.start}")
    H, W = hw
    f = args.focal
    K = [[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]] if f else None  # None: MoGe-2's estimate (depth.py)
    return dict(source=args.video, n=n, fps=fps, hw=[H, W], K=K, object=args.object, licence=args.licence,
                static_camera=args.static_camera)


def prep_hot3d(args):
    """HOT3D-Clips tar (Aria RGB, rectified to an upright pinhole by bench.datasets.hot3d) -> frames, cams.npz (T_world_cam per
    frame: the headset's own tracking, the camera moves) and the moving object's name (largest motion) for the text prompt."""
    from bench.datasets.hot3d import Hot3DClip

    clip = Hot3DClip(args.hot3d, args.mano_dir, out_size=args.size or None)
    fd = os.path.join(args.out, "frames")
    os.makedirs(fd, exist_ok=True)
    T, pos, names = [], {}, {}
    for i in range(len(clip)):
        fr = clip.frame(i)
        cv2.imwrite(os.path.join(fd, f"{i:06d}.jpg"), fr["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        T.append(fr["T_world_cam"])
        for oid, lst in json.load(clip.tar.extractfile(f"{fr['key']}.objects.json")).items():
            pos.setdefault(oid, []).append(lst[0]["T_world_from_object"]["translation_xyz"])
            names[oid] = lst[0]["object_name"]
    moved = {o: np.linalg.norm(np.asarray(P) - P[0], axis=1).max() for o, P in pos.items() if len(P) == len(T)}
    name = args.object if args.object != "object" else names[max(moved, key=moved.get)].replace("_", " ")
    np.savez(os.path.join(args.out, "cams.npz"), T_world_cam=np.stack(T))
    H, W = fr["image"].shape[:2]
    return dict(source="HOT3D " + os.path.basename(args.hot3d)[:-4], n=len(T), fps=30, hw=[H, W], K=clip.K.tolist(), object=name,
                licence=args.licence or "HOT3D (Banerjee et al., CVPR 2025): see the dataset's licence",
                static_camera=False, device=clip.device)


def main(args):
    os.makedirs(args.out, exist_ok=True)
    clip = prep_dexycb(args) if args.dexycb else prep_hot3d(args) if args.hot3d else prep_video(args)
    write_video(os.path.join(args.out, "frames"), clip["n"], clip["fps"], os.path.join(args.out, "video.mp4"))
    json.dump(clip, open(os.path.join(args.out, "clip.json"), "w"), indent=1)
    print(args.out, clip["n"], "frames", clip.get("object"))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dexycb", default="", help="converted DexYCB clip tar (bench.datasets.build_dexycb)")
    p.add_argument("--video", default="")
    p.add_argument("--hot3d", default="", help="HOT3D-Clips tar (egocentric, moving camera: writes cams.npz)")
    p.add_argument("--size", type=int, default=704, help="hot3d: rectified image size in px")
    p.add_argument("--focal", type=float, default=0.0, help="video: focal length in px (0: MoGe-2 estimate)")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--resize", type=int, default=0, help="video: longer side in px")
    p.add_argument("--object", default="object")
    p.add_argument("--licence", default="")
    p.add_argument("--static_camera", action="store_true")
    p.add_argument("--mano_dir", default=os.environ.get("MANO_DIR", os.path.join(FAH_HOME, "mano")))
    p.add_argument("--out", required=True)
    main(p.parse_args())
