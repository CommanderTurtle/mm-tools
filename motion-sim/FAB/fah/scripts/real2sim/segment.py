"""Video masks of the manipulated object -> <clip>/masks.npz (obj (N, H, W) bool, score (N,), prompt).
--model sam3 (default; the SAM 3 video predictor, github.com/facebookresearch/sam3, in its own environment): a box prompt on one frame (--box, or for DexYCB --gt_box:
the GT mask's box on that frame, i.e. the click a user would give) or a text prompt (--text, default the clip's object
name, e.g. "mustard bottle"; the instance overlapping the box, else the most confident one, is kept), propagated forward
and backward. --search N (text): the prompt frame is searched every N frames for the most confident instance no FAH hand
mask touches (then the name's last word as the prompt). --model sam2: SAM 2.1 hiera-L from a box prompt (real2sim env).

    $SAM3_PY scripts/real2sim/segment.py --clip $R2S/mustard --prompt text --text_rule dataset --search 5 --relink 0.2
    $SAM3_PY scripts/real2sim/segment.py --clip <clip> --prompt text --text "power drill"
    $SAM3_PY scripts/real2sim/segment.py --clip <clip> --frame 0 --gt_box            # DexYCB: the GT box picks the instance
    python scripts/real2sim/segment.py --model sam2 --clip <clip> --frame 10 --box 120,80,260,300
"""
import argparse
import json
import os

import numpy as np
import torch

FAH_HOME = os.environ.get("FAH_HOME", os.path.join(os.path.expanduser("~"), ".cache", "fah"))
CKPT = os.environ.get("SAM2_CKPT", os.path.join(FAH_HOME, "sam2.1", "sam2.1_hiera_large.pt"))
CKPT3 = os.environ.get("SAM3_CKPT", os.path.join(FAH_HOME, "sam3", "sam3.pt"))


# HOT3D object name -> text prompt (one table per dataset, fixed 2026-10-28 before the round-2 runs; round 1's clip.json names
# were edited per clip); keys with spaces as prep_clip.py writes them, plus the round-1 edited names
HOT3D_TEXT = {"aria small": "glasses", "birdhouse toy": "toy birdhouse", "bottle bbq": "bbq sauce bottle", "bottle mustard": "mustard bottle",
              "bottle ranch": "ranch dressing bottle", "bowl": "bowl", "can parmesan": "parmesan cheese can", "can soup": "soup can",
              "can tomato sauce": "tomato sauce can", "carton milk": "milk carton", "carton oj": "orange juice carton",
              "cellphone": "cellphone", "coffee pot": "coffee pot", "dino toy": "toy dinosaur", "dumbbell 5lb": "dumbbell", "flask": "flask",
              "food vegetables": "toy vegetables", "food waffles": "waffles", "holder black": "black holder", "holder gray": "gray holder",
              "keyboard": "keyboard", "mouse": "computer mouse", "mug patterned": "patterned mug", "mug white": "white mug",
              "plate bamboo": "bamboo plate", "potato masher": "potato masher", "puzzle toy": "toy puzzle", "spatula red": "red spatula",
              "spoon wooden": "wooden spoon", "vase": "vase", "whiteboard eraser": "whiteboard eraser"}
HOT3D_TEXT.update({"ranch dressing bottle": "ranch dressing bottle", "tomato sauce can": "tomato sauce can", "milk carton": "milk carton",
                   "white vase": "vase", "parmesan cheese can": "parmesan cheese can", "white mug": "white mug",
                   "orange juice carton": "orange juice carton", "soup can": "soup can", "wooden spoon": "wooden spoon"})


def object_text(clip, rule="clip"):
    """'006_mustard_bottle' -> 'mustard bottle'. rule 'dataset': from the dataset's own object name (DexYCB: the YCB name of
    clip.json's ycb_id; HOT3D: HOT3D_TEXT), ignoring per-clip edits of clip.json's "object"."""
    c = json.load(open(os.path.join(clip, "clip.json")))
    name = c.get("object", "object")
    if rule == "dataset" and "ycb_id" in c:
        import sys

        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        from scripts.real2sim.prep_clip import YCB

        name = YCB[int(c["ycb_id"])]
    elif rule == "dataset" and c.get("source", "").startswith("HOT3D"):
        return HOT3D_TEXT[name]
    w = name.split("_")
    return " ".join(w[1:] if w[0].isdigit() else w)


def hand_masks(clip, n, shape):
    """Per-frame union of FAH's hand masks (hands.pkl), None without it."""
    import pickle
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from bench.seg.rle import decode

    f = os.path.join(clip, "hands.pkl")
    if not os.path.exists(f):
        return None
    fr = pickle.load(open(f, "rb"))["frames"]
    return [np.any([decode(h["mask_rle"]) for h in fr[i] if "mask_rle" in h], 0) if i < len(fr) and any("mask_rle" in h for h in fr[i])
            else np.zeros(shape, bool) for i in range(n)]


def search_prompt(pred, sid, texts, n, hands, stride):
    """Prompt frame by search: the text prompt on every `stride`-th frame; the most confident instance that no (dilated) hand
    mask touches, else the most confident one; the texts in order (the next only if the previous found no untouched one).
    Returns (frame, text, mask) or None."""
    import cv2

    best = None
    for text in texts:
        for f in range(0, n, stride):
            pred.handle_request(dict(type="reset_session", session_id=sid))
            o = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=f, text=text))["outputs"]
            hd = None if hands is None else cv2.dilate(hands[f].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            for m, pr in zip(o["out_binary_masks"], o["out_probs"]):
                key = (hd is None or (hd & m).sum() < 20, float(pr))
                if best is None or key > best[0]:
                    best = (key, f, text, m)
        if best is not None and best[0][0]:
            break
    pred.handle_request(dict(type="reset_session", session_id=sid))
    return None if best is None else best[1:]


def sam2_masks(args, box):
    from sam2.build_sam import build_sam2_video_predictor

    pred = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_l.yaml", CKPT, device="cuda")
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = pred.init_state(video_path=os.path.join(args.clip, "frames"), offload_video_to_cpu=True)
        n, H, W = st["num_frames"], st["video_height"], st["video_width"]
        obj, score = np.zeros((n, H, W), bool), np.zeros(n, np.float32)
        pred.add_new_points_or_box(st, frame_idx=args.frame, obj_id=1, box=box)
        for rev in (False, True):
            for i, _, logits in pred.propagate_in_video(st, start_frame_idx=args.frame, reverse=rev):
                lg = logits[0, 0].float()
                obj[i], score[i] = (lg > 0).cpu().numpy(), float(lg.max())
    return obj, score


def sam3_masks(args, box, text):
    from sam3.model_builder import build_sam3_video_predictor

    pred = build_sam3_video_predictor(checkpoint_path=CKPT3, gpus_to_use=[0])
    sid = pred.handle_request(dict(type="start_session", resource_path=os.path.join(args.clip, "frames")))["session_id"]
    import glob

    import cv2

    paths = sorted(glob.glob(os.path.join(args.clip, "frames", "*.jpg")))
    H, W = cv2.imread(paths[0]).shape[:2]
    found = None
    if args.prompt == "text" and args.search:  # prompt frame where the object is found untouched (both directions from there)
        texts = [text] + ([text.split()[-1]] if len(text.split()) > 1 else [])
        found = search_prompt(pred, sid, texts, len(paths), hand_masks(args.clip, len(paths), (H, W)), args.search)
        if found is None:
            raise SystemExit("SAM 3 found no object on any searched frame")
        args.frame, text, fmask = found
    if args.prompt == "text":
        r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=args.frame, text=text))
    else:
        x0, y0, x1, y1 = box
        r = pred.handle_request(dict(type="add_prompt", session_id=sid, frame_index=args.frame,
                                     bounding_boxes=[[x0 / W, y0 / H, (x1 - x0) / W, (y1 - y0) / H]], bounding_box_labels=[1]))
    o = r["outputs"]
    if not len(o["out_obj_ids"]):
        raise SystemExit("SAM 3 found no object on the prompt frame")
    if found is not None:  # the searched instance
        k = int(np.argmax([(m & fmask).sum() / max((m | fmask).sum(), 1) for m in o["out_binary_masks"]]))
    elif box is not None and len(o["out_obj_ids"]) > 1:  # the instance that overlaps the box most
        bm = np.zeros((H, W), bool)
        bm[int(box[1]):int(box[3]) + 1, int(box[0]):int(box[2]) + 1] = True
        k = int(np.argmax([(m & bm).sum() / max(m.sum(), 1) for m in o["out_binary_masks"]]))
    else:
        k = int(np.argmax(o["out_probs"]))
    oid = int(o["out_obj_ids"][k])
    obj, score = np.zeros((len(paths), H, W), bool), np.zeros(len(paths), np.float32)
    inst = {}  # frame -> [(mask, prob)] of the other instances (--relink)
    for resp in pred.handle_stream_request(dict(type="propagate_in_video", session_id=sid, propagation_direction="both",
                                                start_frame_index=args.frame)):
        out, i = resp["outputs"], resp["frame_index"]
        ids = list(out["out_obj_ids"])
        if oid in ids:
            j = ids.index(oid)
            obj[i], score[i] = out["out_binary_masks"][j], float(out["out_probs"][j])
        if args.relink:
            inst[i] = [(np.asarray(m, bool), float(pr)) for d_, m, pr in zip(ids, out["out_binary_masks"], out["out_probs"]) if d_ != oid]
    pred.handle_request(dict(type="close_session", session_id=sid))
    relinked = relink(obj, score, inst, args.frame, args.relink) if args.relink else 0
    return obj, score, dict(n_instances=len(o["out_obj_ids"]), picked=oid, search_frame=None if found is None else found[0],
                            search_text=None if found is None else found[1], relinked=relinked)


def relink(obj, score, inst, f0, min_iou, max_gap=15):
    """Identity switches of the SAM 3 tracker: frames without the picked instance take the other instance that overlaps the
    last kept mask most (IoU of 15 px-dilated masks >= min_iou, within max_gap frames), from the prompt frame outwards."""
    import cv2

    k = np.ones((15, 15), np.uint8)
    n, done = len(obj), 0
    for order in (range(f0, n), range(f0, -1, -1)):
        last, gap = None, 0
        for i in order:
            if obj[i].any():
                last, gap = cv2.dilate(obj[i].astype(np.uint8), k).astype(bool), 0
                continue
            gap += 1
            if last is None or gap > max_gap or not inst.get(i):
                continue
            ious = [(cv2.dilate(m.astype(np.uint8), k).astype(bool) & last).sum() / max((cv2.dilate(m.astype(np.uint8), k).astype(bool) | last).sum(), 1)
                    for m, _ in inst[i]]
            j = int(np.argmax(ious))
            if ious[j] >= min_iou:
                obj[i], score[i] = inst[i][j]
                last, gap, done = cv2.dilate(obj[i].astype(np.uint8), k).astype(bool), 0, done + 1
    return done


def main(args):
    box = None
    if args.gt_box:
        m = np.load(os.path.join(args.clip, "gt.npz"))["seg_obj"][args.frame]
        ys, xs = np.nonzero(m)
        box = np.array([xs.min(), ys.min(), xs.max(), ys.max()], np.float32)
    elif args.box:
        box = np.array([float(v) for v in args.box.split(",")], np.float32)
    text = args.text or object_text(args.clip, args.text_rule)
    extra = {}
    if args.model == "sam2":
        obj, score = sam2_masks(args, box)
    else:
        obj, score, extra = sam3_masks(args, box, text)
    prompt = dict(model=args.model, frame=args.frame, search=args.search, box=None if box is None else box.tolist(), gt_box=args.gt_box,
                  prompt=args.prompt if args.model == "sam3" else "box", text=text if args.prompt == "text" else None, **extra)
    np.savez_compressed(os.path.join(args.clip, args.out), obj=obj, score=score, prompt=json.dumps(prompt))
    print(args.clip, args.model, prompt["prompt"], len(obj), "frames, mask area median %d px, empty %d" %
          (np.median(obj.sum((1, 2))), int((obj.sum((1, 2)) == 0).sum())))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", required=True)
    p.add_argument("--model", default="sam3", choices=["sam3", "sam2"])
    p.add_argument("--prompt", default="box", choices=["box", "text"], help="sam3: box (--box / --gt_box) or text")
    p.add_argument("--text", default="", help="sam3 text prompt (default: the clip's object name)")
    p.add_argument("--text_rule", default="clip", choices=["clip", "dataset"], help="default text: clip.json's object, or the "
                   "dataset's own object name (object_text)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--relink", type=float, default=0.0, help="sam3: > 0: frames that lose the picked instance take the other "
                   "instance overlapping the last kept mask (dilated IoU >= this)")
    p.add_argument("--search", type=int, default=0, help="sam3 text: search the prompt frame every N frames (untouched instance, "
                   "most confident; then the name's last word), instead of --frame")
    p.add_argument("--box", default="", help="x0,y0,x1,y1 px on --frame")
    p.add_argument("--gt_box", action="store_true")
    p.add_argument("--out", default="masks.npz")
    main(p.parse_args())
