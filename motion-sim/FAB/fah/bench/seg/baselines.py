"""Segmentation baselines of the paper's mask table, as callables on one BGR frame (scripts/seg_compare.py runs them on
bench_seg frames and times them). None of their code is included: each arm imports the official package / checkout,
installed by scripts/setup_seg_baselines.sh, and is used under its own licence.

  sam2b_ours / sam2l_ours  SAM 2.1 hiera-B+ / hiera-L (`sam2`, Apache-2.0; HF facebook/sam2.1-hiera-*), one box prompt
                           per hand, single-mask output, bf16 autocast
  sam2b_gt / sam2l_gt      the same prompted with the GT instance boxes (oracle)
  sam3_ours                SAM 3 interactive / tracker head (`transformers` Sam3TrackerModel; HF facebook/sam3, gated,
                           SAM License), box prompts, bf16
  sam3_text                SAM 3 concept segmentation (`transformers` Sam3Model), text prompt "hand", score >= 0.5, bf16
  hos                      VISOR-HOS PointRend R-50 (github.com/epic-kitchens/VISOR-HOS; hand + arm masks like VISOR),
                           hand score >= 0.5, Detectron2 fp32
  hands23                  Hands23 Mask R-CNN X-101 (github.com/EvaCheng-cty/hands23_detector), hand score >= 0.7 (the
                           repository's demo default), Detectron2 fp32
Box arms return one mask per prompt box (in order); the others return their own detections.
"""
import os
import sys

import numpy as np
import torch

BOX_ARMS = ("sam2b_ours", "sam2l_ours", "sam2b_gt", "sam2l_gt", "sam3_ours")
OWN_ARMS = ("sam3_text", "hos", "hands23")
SAM2 = {"sam2b": "facebook/sam2.1-hiera-base-plus", "sam2l": "facebook/sam2.1-hiera-large"}


def mask_box(m):
    """Tight xyxy box of a binary mask (None for an empty mask)."""
    ys, xs = np.nonzero(m)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32) if len(xs) else None


class Sam2Arm:
    def __init__(self, model):
        from sam2.build_sam import build_sam2_hf
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.p = SAM2ImagePredictor(build_sam2_hf(model, device="cuda"))

    def __call__(self, img_bgr, boxes):
        if not len(boxes):
            return []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            self.p.set_image(np.ascontiguousarray(img_bgr[:, :, ::-1]))
            m, _, _ = self.p.predict(box=np.asarray(boxes, np.float32), multimask_output=False)
        return list(np.asarray(m).reshape(len(boxes), -1, *img_bgr.shape[:2])[:, 0] > 0)


class Sam3TextArm:
    def __init__(self, model="facebook/sam3", prompt="hand", thresh=0.5):
        from transformers import Sam3Model, Sam3Processor

        self.m = Sam3Model.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.proc = Sam3Processor.from_pretrained(model)
        self.prompt, self.thresh = prompt, thresh

    def __call__(self, img_bgr):
        inputs = self.proc(images=img_bgr[:, :, ::-1].copy(), text=self.prompt, return_tensors="pt").to("cuda")
        inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.m(**inputs)
        r = self.proc.post_process_instance_segmentation(out, threshold=self.thresh, mask_threshold=0.5,
                                                         target_sizes=[img_bgr.shape[:2]])[0]
        masks = [m.cpu().numpy().astype(bool) for m in r["masks"]]
        scores = r["scores"].float().cpu().numpy()
        return [dict(bbox=mask_box(m), score=float(s), is_right=None, m=m) for m, s in zip(masks, scores) if m.any()]


class Sam3BoxArm:
    def __init__(self, model="facebook/sam3"):
        from transformers import Sam3TrackerModel, Sam3TrackerProcessor

        self.m = Sam3TrackerModel.from_pretrained(model, torch_dtype=torch.bfloat16).to("cuda").eval()
        self.proc = Sam3TrackerProcessor.from_pretrained(model)

    def __call__(self, img_bgr, boxes):
        if not len(boxes):
            return []
        inputs = self.proc(images=img_bgr[:, :, ::-1].copy(), input_boxes=[[list(map(float, b)) for b in boxes]],
                           return_tensors="pt").to("cuda")
        inputs = {k: v.to(torch.bfloat16) if torch.is_floating_point(v) else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = self.m(**inputs, multimask_output=False)
        m = self.proc.post_process_masks(out.pred_masks.float().cpu(), inputs["original_sizes"].cpu())[0]
        return list(np.asarray(m).reshape(len(boxes), -1, *img_bgr.shape[:2])[:, 0] > 0)


class D2Arm:
    """Detectron2 hand(-object) segmenters from their repository checkouts (their own environment):
    hos = VISOR-HOS PointRend R-50 (Darkhalil et al. 2022), hands23 = Hands23 Mask R-CNN X-101 (Cheng et al. 2023)."""

    def __init__(self, kind, repo, weights, thresh):
        sys.path.insert(0, repo)
        from detectron2.config import get_cfg
        from detectron2.engine import DefaultPredictor

        cfg = get_cfg()
        if kind == "hos":
            import hos  # noqa: F401  (registers HOSROIHeads)
            from detectron2.projects import point_rend

            point_rend.add_pointrend_config(cfg)
            cfg.merge_from_file(os.path.join(repo, "configs/hos/hos_pointrend_rcnn_R_50_FPN_1x.yaml"))
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = cfg.MODEL.POINT_HEAD.NUM_CLASSES = 2
        else:
            from hodetector.modeling import roi_heads  # noqa: F401  (registers hoRCNNROIHeads)

            cfg.merge_from_file(os.path.join(repo, "faster_rcnn_X_101_32x8d_FPN_3x_Hands23.yaml"))
            cfg.HAND, cfg.FIRSTOBJ, cfg.SECONDOBJ, cfg.HAND_RELA, cfg.OBJ_RELA = thresh, 0.5, 0.3, 0.3, 0.7  # demo defaults
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh if kind == "hos" else min(thresh, 0.3)
        cfg.MODEL.WEIGHTS = weights
        self.kind, self.thresh, self.p = kind, thresh, DefaultPredictor(cfg)

    def __call__(self, img_bgr):
        with torch.inference_mode():
            ins = self.p(img_bgr)["instances"].to("cpu")
        hand = (ins.pred_classes == 0) & (ins.scores >= self.thresh)
        side = ins.pred_handsides.argmax(1) == 1 if self.kind == "hos" else ins.pred_dz[:, 5] == 1  # right = 1
        return [dict(bbox=b.numpy(), score=float(s), is_right=bool(r), m=m.numpy().astype(bool))
                for b, s, r, m, k in zip(ins.pred_boxes.tensor, ins.scores, side, ins.pred_masks, hand) if k]


def build_arms(names, thirdparty="thirdparty", sam3="facebook/sam3", text="hand", sam3_thresh=0.5):
    """name -> callable for the requested arms (models shared between the *_ours / *_gt variants of one SAM 2.1 size)."""
    arms, sam2 = {}, {}
    for a in names:
        if a[:5] in SAM2:
            sam2.setdefault(a[:5], Sam2Arm(SAM2[a[:5]]))
            arms[a] = sam2[a[:5]]
        elif a == "sam3_ours":
            arms[a] = Sam3BoxArm(sam3)
        elif a == "sam3_text":
            arms[a] = Sam3TextArm(sam3, text, sam3_thresh)
        elif a == "hos":
            repo = os.path.join(thirdparty, "VISOR-HOS")
            arms[a] = D2Arm("hos", repo, os.path.join(repo, "checkpoints", "model_final_hos.pth"), 0.5)
        elif a == "hands23":
            repo = os.path.join(thirdparty, "hands23_detector")
            arms[a] = D2Arm("hands23", repo, os.path.join(repo, "model_weights", "model_hands23.pth"), 0.7)
        elif a != "det":
            raise ValueError(f"unknown arm {a}; known: det, {', '.join(BOX_ARMS + OWN_ARMS)}")
    return arms
