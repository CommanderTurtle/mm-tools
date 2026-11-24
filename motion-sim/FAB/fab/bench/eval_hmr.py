"""Camera-space HMR evaluation (oracle or detected boxes, GT intrinsics), one shard per GPU.

Predictions are MHR meshes, transferred to SMPL topology by MHR's barycentric mapping (bench/mhr_smpl.py) and
scored against gendered SMPL ground truth posed in the camera frame:
  - 14 joints from the H36M regressor, pelvis = hip midpoint (the 3DPW / EMDB protocol of SPIN, CLIFF, HMR 2.0,
    CameraHMR, SAM 3D Body): MPJPE, PA-MPJPE
  - 24 SMPL joints from the SMPL regressor, pelvis = joint 0: MPJPE24, PA-MPJPE24
  - PVE: vertices after the same 14-joint pelvis alignment
Per-sample errors and latency go to <out>/<dataset>_shard<k>.npz; bench/summarize.py merges the shards. With
--save_pred, the meshes go to <out>/<dataset>_pred_shard<k>.npz for bench/eval_standard_protocol.py (the paper's tables).

    # the released FAB-L call (FABPredictor's released profile + fitter), PyTorch body, saved for the standard protocol
    python -m bench.eval_hmr --dataset 3dpw --root $FAB_DATA/3dpw --fab_ckpt fab-l --compile --upright 0 \
        --predictor --save_dense --predictor_kw FINAL --save_pred --out $FAB_HOME/bench/preds/fab-l --shard 0 --num_shards 4
    # SAM 3D Body itself (needs its network weights, <SAM 3D Body dir>/model.ckpt)
    python -m bench.eval_hmr --dataset 3dpw --root $FAB_DATA/3dpw --save_pred --out $FAB_HOME/bench/preds/sam3db
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from bench.body_models import SMPL
from bench.corruptions import CORRUPTIONS
from bench.datasets import DATASETS, WORLD_FRAME
from bench.mhr_smpl import MHRToSMPL
from fab import paths

H36M_TO_J14 = [3, 2, 1, 4, 5, 6, 16, 15, 14, 11, 12, 13, 8, 10]


def procrustes(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Similarity-align pred [J, 3] to gt [J, 3]."""
    mu_p, mu_g = pred.mean(0), gt.mean(0)
    p, g = pred - mu_p, gt - mu_g
    u, s, vt = np.linalg.svd(p.T @ g)
    z = np.eye(3)
    z[-1, -1] = np.sign(np.linalg.det(u @ vt))
    rot = vt.T @ z @ u.T
    scale = (s * np.diag(z)).sum() / (p ** 2).sum()
    return scale * p @ rot.T + mu_g


def errors(pred_verts: np.ndarray, gt_verts: np.ndarray, j14: np.ndarray, j24: np.ndarray) -> dict[str, float]:
    """All inputs in meters, camera frame; returns mm."""
    pj14, gj14 = j14 @ pred_verts, j14 @ gt_verts
    pp, gp = pj14[[2, 3]].mean(0), gj14[[2, 3]].mean(0)
    pj14, gj14 = pj14 - pp, gj14 - gp
    pj24, gj24 = j24 @ pred_verts, j24 @ gt_verts
    pj24, gj24 = pj24 - pj24[:1], gj24 - gj24[:1]
    mm = 1000.0
    return {
        "mpjpe": float(np.linalg.norm(pj14 - gj14, axis=1).mean() * mm),
        "pa_mpjpe": float(np.linalg.norm(procrustes(pj14, gj14) - gj14, axis=1).mean() * mm),
        "pve": float(np.linalg.norm((pred_verts - pp) - (gt_verts - gp), axis=1).mean() * mm),
        "mpjpe24": float(np.linalg.norm(pj24 - gj24, axis=1).mean() * mm),
        "pa_mpjpe24": float(np.linalg.norm(procrustes(pj24, gj24) - gj24, axis=1).mean() * mm),
    }


def build_fab(args):
    """FAB student: the teacher's crop (same box transform), resized to the student's input size."""
    from fab.data.utils.prepare_batch import prepare_batch
    from fab import load_fab_image

    ckpt_dir = Path(args.ckpt).parent
    state = torch.load(args.fab_ckpt, map_location="cpu", weights_only=False) if args.fab_ckpt != "teacher" else {}
    train_args = state.get("args", {"arch": "fabd", "size": 512})
    if train_args.get("arch") in ("fabd", "fabt"):
        return build_fabd(args, state, train_args)
    model = load_fab_image(ckpt_dir / "model_config.yaml", ckpt_dir / "assets" / "mhr_model.pt", args.fab_ckpt,
                           backbone=train_args.get("backbone", "dinov3_vitb16"), size=train_args.get("size", 384),
                           pool_layers=train_args.get("pool_layers", 2), iters=train_args.get("iters", 1))
    # the teacher's box transform (no teacher weights are run)
    from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from torchvision.transforms import ToTensor

    transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                         VisionTransformWrapper(ToTensor())])

    @torch.inference_mode()
    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        batch = prepare_batch(image, transform, bbox[None].astype(np.float32), None, None,
                              cam_int=torch.from_numpy(K).float()[None])
        img = batch["img"][0].cuda()
        if img.shape[-1] != model.size:
            img = torch.nn.functional.interpolate(img, size=(model.size, model.size), mode="area")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(img, batch["bbox_center"][0].cuda(), batch["bbox_scale"][0].cuda(),
                        batch["ori_img_size"][0].cuda(), batch["cam_int"].cuda())
        return out["pred_vertices"][0].float().cpu().numpy()

    return run


def build_fabd(args, state, train_args):
    """FAB: SAM 3D Body's decoder on the student backbone (fab/decoder_model.py), 512 px crops.
    --fab_ckpt teacher: the same wrapper around the teacher's own backbone (must reproduce SAM 3D Body)."""
    import copy

    from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from fab.data.utils.prepare_batch import prepare_batch
    from fab.decoder_model import build_fab_decoder
    from torchvision.transforms import ToTensor

    model, teacher_backbone = build_fab_decoder(args.ckpt, train_args.get("backbone", "dinov3_vitb16"),
                                                train_args.get("size", 512), pretrained=False,
                                                dec_layers=args.dec_layers or train_args.get("dec_layers", 0))
    if train_args.get("arch") == "fabt":  # fine-tuned SAM 3D Body: its own (fine-tuned) backbone, saved in the state
        model.body.backbone = copy.deepcopy(teacher_backbone)
        model.body.backbone_dtype = torch.bfloat16
    if state:
        weights = dict(state["model"])
        weights.update(state.get("ema") or {})
        if "dense_head.surface_idx" in weights:  # direct dense keypoints: evaluated alongside the mesh
            from fab.dense_head import dense_hr, dense_mask_depth, dense_mask_query, dense_part_depth, dense_xyz_layers
            from fab.dense_head import dense_head_kwargs

            model.add_dense_head(weights["dense_head.surface_idx"].tolist(),
                                 residual=bool(int(weights.get("dense_head.residual", torch.tensor(0)))), parts="dense_head.part_emb" in weights,
                                 xyz_layers=dense_xyz_layers(weights), part_depth=dense_part_depth(weights),
                                 hr=dense_hr(weights), mask_depth=dense_mask_depth(weights), mask_query=dense_mask_query(weights),
                                 **dense_head_kwargs(weights)).cuda()
        from fab.contact_head import contact_head_args, has_contact_head

        if has_contact_head(weights) and model.dense_head is not None:  # explicit contacts (saved with --save_pred)
            model.add_contact_head(**contact_head_args(weights)).cuda()
        from fab.vis_head import has_vis_head, vis_head_args

        if getattr(args, "vis_head_from", ""):  # a visibility head trained on the frozen model
            vs = torch.load(args.vis_head_from, map_location="cpu", weights_only=False, mmap=True)
            weights.update({k: v for k, v in {**vs["model"], **(vs.get("ema") or {})}.items() if k.startswith("vis_head.")})
        if has_vis_head(weights) and model.dense_head is not None:  # per-keypoint visibility (out["vis_logits"])
            model.add_vis_head(**vis_head_args(weights)).cuda()
        model.load_state_dict(weights, strict=False)
        if getattr(args, "dense_25d_mix", -1.0) >= 0 and model.dense_head is not None:  # 2.5D lift at test time
            model.dense_head.p25d_mix = args.dense_25d_mix
        if getattr(args, "dense_25d_vis_gate", False) and model.dense_head is not None:  # lift x P(visible)
            if model.vis_head is None:
                raise ValueError("--dense_25d_vis_gate needs a visibility head (saved, or --vis_head_from)")
            model.dense_head.p25d_vis_gate = True
    else:
        model.body.backbone = copy.deepcopy(teacher_backbone)
        model.body.backbone_dtype = torch.bfloat16
    del teacher_backbone
    engine_path = getattr(args, "backbone_engine", "")
    if engine_path:
        if not state or train_args.get("arch") != "fabd":
            raise ValueError("--backbone_engine requires a FAB student checkpoint")
        from fab.tensorrt import TensorRTBackbone

        model.body.backbone = TensorRTBackbone(engine_path, args.fab_ckpt)
    model = model.eval()
    model.body.head_pose.enable_fast()
    if getattr(args, "fast_inference", False) or getattr(args, "compile", False):  # the released numerics
        from fab.precision import compile_body, fast_inference

        fast_inference(model)
        if args.compile:
            compile_body(model)
    decoder_fp32 = getattr(args, "decoder_fp32", False)
    if decoder_fp32:
        model.body.backbone.forward = torch.autocast("cuda", dtype=torch.bfloat16)(
            model.body.backbone.forward)
    transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                         VisionTransformWrapper(ToTensor())])

    trt = None
    if getattr(args, "trt_engine", ""):  # the whole body forward from a verified TensorRT engine (fab.tensorrt.TensorRTBundle):
        # its outputs replace the model's dict; the model only supplies head metadata
        from fab.tensorrt import TensorRTBundle

        trt = TensorRTBundle(args.trt_engine, args.fab_ckpt)
        mix = (trt.meta.get("config") or {}).get("p25d_mix")
        if model.dense_head is not None and mix is not None and abs(model.dense_head.p25d_mix - mix) > 1e-9:
            raise ValueError(f"--trt_engine has the 2.5D mix {mix}: pass --dense_25d_mix {mix}")

    fp32 = getattr(args, "fp32", False)  # --fp32: the reference numerics (no autocast anywhere, TF32 off)
    if fp32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    def forward(img, center, scale, ori, cam, affine, mask=None, mask_score=None):
        if trt is not None and mask is None:
            return trt(img.cuda().float(), center.cuda().float(), scale.cuda().float(), ori.cuda().float(),
                       cam.cuda().float(), affine.cuda().float())
        kw = {} if mask is None else {"mask": mask.cuda(), "mask_score": mask_score.cuda()}  # --mask_prompt
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=not (decoder_fp32 or fp32)):
            return model(img.cuda(), center.cuda(), scale.cuda(), ori.cuda(), cam.cuda(), affine=affine.cuda(), **kw)

    scales = [float(x) for x in args.ms_tta.split(",")] if args.ms_tta else [1.0]
    extra = {}  # dense-head 14 evaluation joints of the last call (root-relative, camera axes)

    if model.dense_head is not None:
        extra["surface_idx"] = model.dense_head.surface_idx.cpu().numpy()
        extra["residual"] = bool(int(model.dense_head.residual))

    def dense_j14(out, R3=None):
        if "dense_xyz" not in out:
            return None
        j = out["dense_xyz"][0, 70:].float().cpu().numpy().astype(np.float64)  # 14 joints, then the surface
        return j if R3 is None else j @ R3

    @torch.inference_mode()
    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        if getattr(args, "box_scale", 1.0) != 1.0:  # every box scaled about its centre (crop-size study)
            c, wh = (bbox[:2] + bbox[2:]) / 2, bbox[2:] - bbox[:2]
            bbox = np.concatenate([c - wh * args.box_scale / 2, c + wh * args.box_scale / 2]).astype(np.float32)
        if len(scales) > 1:  # multi-scale TTA: mean of the (same-topology, untranslated) meshes of rescaled boxes
            c, wh = (bbox[:2] + bbox[2:]) / 2, bbox[2:] - bbox[:2]
            return np.mean([run_one(image, np.concatenate([c - wh * f / 2, c + wh * f / 2]), K) for f in scales], 0)
        verts = run_one(image, bbox, K)
        if getattr(args, "flip_tta", False):  # mean with the mirrored prediction on the flipped image
            W = image.shape[1]
            Kf = K.copy()
            Kf[0, 2] = W - 1 - K[0, 2]
            bf = np.array([W - 1 - bbox[2], bbox[1], W - 1 - bbox[0], bbox[3]], np.float32)
            dense, kp = extra.get("dense_j14"), extra.get("kp2d")
            vf = run_one(np.ascontiguousarray(image[:, ::-1]), bf, Kf)
            extra["dense_j14"], extra["kp2d"] = dense, kp  # keep the unflipped auxiliary outputs
            if "mirror" not in extra:
                from scipy.spatial import cKDTree

                rest = model.body.head_pose.mhr.character_torch.mesh.rest_vertices.float().cpu().numpy()
                extra["mirror"] = cKDTree(rest).query(rest * np.array([-1.0, 1.0, 1.0]))[1]  # left <-> right vertex
            verts = 0.5 * (verts + (vf * np.array([-1.0, 1.0, 1.0]))[extra["mirror"]])
        extra["recropped"] = 0.0
        if getattr(args, "recrop", 0) > 0 and "kp2d" in extra:  # re-crop on the predicted body keypoints' extent
            kp = extra["kp2d"]
            H, W = image.shape[:2]
            if args.recrop_border > 0:  # only bodies whose keypoints reach the crop's border (as FAH's hands)
                lo_c, hi_c = extra["crop"][:2], extra["crop"][2:]
                m = args.recrop_border * (hi_c - lo_c)
                if not ((kp.min(0) < lo_c + m).any() or (kp.max(0) > hi_c - m).any()):
                    return verts
            extra["recropped"] = 1.0
            lo, hi = kp.min(0), kp.max(0)
            c, wh = (lo + hi) / 2, (hi - lo) * args.recrop
            nb = np.concatenate([c - wh / 2, c + wh / 2]).clip([0, 0, 0, 0], [W - 1, H - 1, W - 1, H - 1])
            if (nb[2:] - nb[:2]).min() > 16:
                verts = run_one(image, nb.astype(np.float32), K)
        return verts

    def run_one(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        mfull = extra.get("mask_full")  # --mask_prompt: full-image person mask of this sample
        batch = prepare_batch(image, transform, bbox[None].astype(np.float32),
                              None if mfull is None else mfull[None, :, :, None],
                              None if mfull is None else np.ones(1, np.float32), cam_int=torch.from_numpy(K).float()[None])
        args_ = (batch["img"][0], batch["bbox_center"][0], batch["bbox_scale"][0], batch["ori_img_size"][0],
                 batch["cam_int"], batch["affine_trans"][0])
        out = forward(*args_) if mfull is None else forward(*args_, mask=batch["mask"][0], mask_score=batch["mask_score"][0])
        verts = out["pred_vertices"][0].float().cpu().numpy()
        extra["dense_j14"] = dense_j14(out)
        if "contact_logits" in out:
            extra["contact"] = out["contact_logits"][0].float().cpu().numpy()
        if "vis_logits" in out:  # (84, 3) P(visible / occluded / outside the crop): 70 MHR keypoints + J14
            extra["vis"] = out["vis_logits"][0].float().softmax(-1).cpu().numpy()
        if out.get("part_logits") is not None:  # predicted part map + the crop geometry (scored against the GT mesh)
            pl = out["part_logits"][0:1].float()
            pr = getattr(args, "parts_res", 0)  # (callers that build their own args may lack it)
            if pr:  # the logits upsampled first (bilinear), as FABPredictor's 256 crop maps
                pl = torch.nn.functional.interpolate(pl, size=(pr,) * 2, mode="bilinear", align_corners=False)
            extra["parts"] = (pl[0].argmax(0).cpu().numpy(),
                              batch["affine_trans"][0, 0].numpy().astype(np.float64), K.astype(np.float64),
                              batch["img"].shape[-1])
        # body, feet and wrist keypoints (recrop box)
        extra["kp2d"] = out["pred_keypoints_2d"][0, list(range(21)) + [41, 62], :2].float().cpu().numpy()
        if "dense_uv" in out:  # the dense head's own MHR-keypoint 2D (heatmaps), in image pixels
            Ai = np.linalg.inv(np.vstack([batch["affine_trans"][0, 0].numpy().astype(np.float64), [0, 0, 1]]))
            uv = out["dense_uv"][0, list(range(21)) + [41, 62]].float().cpu().numpy() * batch["img"].shape[-1]
            extra["kp2d_dense"] = (uv @ Ai[:2, :2].T + Ai[:2, 2]).astype(np.float32)
        if "pred_raw" in out:
            extra["pred_raw"] = out["pred_raw"][0:1].detach().float()
        c_, s_ = batch["bbox_center"][0, 0].numpy(), batch["bbox_scale"][0, 0].numpy()
        extra["crop"] = np.concatenate([c_ - s_ / 2, c_ + s_ / 2])  # the crop's image-space box
        extra["cam_t"] = out["pred_cam_t"][0].float().cpu().numpy()  # --save_dense: the mesh's camera translation
        extra["kp3d"] = out["pred_keypoints_3d"][0].float().cpu().numpy()  # --save_dense: the mesh's 70 MHR keypoints
        extra["kp2d_full"] = out["pred_keypoints_2d"][0, :, :2].float().cpu().numpy()
        if "dense_uv" in out:  # --save_dense: the dense head's own 2D of its 14 joints / 70 MHR keypoints (image px)
            Ai = np.linalg.inv(np.vstack([batch["affine_trans"][0, 0].numpy().astype(np.float64), [0, 0, 1]]))
            uvd = out["dense_uv"][0, :84].float().cpu().numpy().astype(np.float64) * batch["img"].shape[-1]
            uvd = uvd @ Ai[:2, :2].T + Ai[:2, 2]
            extra["dense_uv14"], extra["dense_kp2d"] = uvd[70:84], uvd[:70]
            kd = out["dense_xyz"][0, :70].float().cpu().numpy().astype(np.float64)
            if extra.get("residual"):  # offsets on the MHR keypoints (the residual head's anchors)
                kd = kd + extra["kp3d"].astype(np.float64)
            extra["dense_kp3d"] = kd
        if not args.upright:
            return verts
        # upright canonicalisation: if the torso (hips -> shoulders, 2D) is far from vertical, roll the camera so it
        # is upright and predict again; rolling the crop about its centre is exactly the crop of the image rotated
        # about the principal point (K unchanged), so the second prediction is un-rolled exactly by R3^T
        kp = out["pred_keypoints_2d"][0].float().cpu().numpy()
        t = kp[[5, 6]].mean(0) - kp[[9, 10]].mean(0)
        phi_t = np.degrees(np.arctan2(t[0], -t[1]))
        if abs(phi_t) < args.upright:
            return verts
        phi = np.radians(-phi_t)
        R2 = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        S = batch["img"].shape[-1]
        u0 = np.full(2, (S - 1) / 2.0)
        Rc = np.eye(3)
        Rc[:2, :2], Rc[:2, 2] = R2, u0 - R2 @ u0
        pp = K[:2, 2].astype(np.float64)
        Ri_inv = np.eye(3)
        Ri_inv[:2, :2], Ri_inv[:2, 2] = R2.T, pp - R2.T @ pp
        A = np.vstack([batch["affine_trans"][0, 0].numpy().astype(np.float64), [0, 0, 1]])
        A2 = Rc @ A @ Ri_inv
        crop = batch["img"][0, 0].permute(1, 2, 0).numpy()
        crop = cv2.warpAffine(crop, Rc[:2], (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        centre = np.linalg.solve(A2, np.array([S / 2, S / 2, 1.0]))[:2]
        out2 = forward(torch.from_numpy(crop).permute(2, 0, 1)[None].float(),
                       torch.from_numpy(centre[None]).float(), args_[2], args_[3], args_[4],
                       torch.from_numpy(A2[None, :2]).float())
        R3 = np.eye(3)
        R3[:2, :2] = R2
        extra["dense_j14"] = dense_j14(out2, R3)
        return out2["pred_vertices"][0].float().cpu().numpy() @ R3  # (R3^T v)^T = v^T R3

    run.extra = extra
    return run


FAST3DB_FLAGS = {  # Fast SAM 3D Body's own eval settings (mhr2smpl/multi_view/step0_infer_*.py), minus TensorRT
    "GPU_HAND_PREP": "1", "LAYER_DTYPE": "fp32", "SKIP_KEYPOINT_PROMPT": "1", "USE_COMPILE_BACKBONE": "1",
    "USE_COMPILE": "1", "DECODER_COMPILE": "1", "MHR_USE_CUDA_GRAPH": "0", "KEYPOINT_PROMPT_INTERM_INTERVAL": "999",
    "BODY_INTERM_PRED_LAYERS": "0,1,2", "HAND_INTERM_PRED_LAYERS": "0,1", "MHR_NO_CORRECTIVES": "1", "IMG_SIZE": "512",
}


def build_fast3db(args):
    """Fast SAM 3D Body (github.com/yangtiming/Fast-SAM-3D-Body) with its eval flags; YOLO11-Pose hand boxes."""
    import os
    import sys

    for k, v in FAST3DB_FLAGS.items():
        os.environ.setdefault(k, v)
    sys.path.insert(0, args.fast3db_root)
    # detectron2 is only used by their ViTDet detector (we use YOLO11-Pose) and imported at the top of a
    # visualisation helper; an empty stub avoids building it without touching the inference path
    import types

    if "detectron2" not in sys.modules:
        d2, d2cfg = types.ModuleType("detectron2"), types.ModuleType("detectron2.config")
        d2cfg.LazyConfig = d2cfg.instantiate = None
        d2.config = d2cfg
        sys.modules.update({"detectron2": d2, "detectron2.config": d2cfg})
    from notebook.utils import setup_sam_3d_body

    estimator = setup_sam_3d_body(local_checkpoint_path=str(Path(args.ckpt).parent), detector_name="yolo_pose",
                                  detector_model=args.fast3db_pose_model, segmentor_name="", fov_name="")

    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        out = estimator.process_one_image(image, bboxes=bbox[None].astype(np.float32),
                                          cam_int=torch.from_numpy(K).float()[None], hand_box_source="yolo_pose")
        return out[0]["pred_vertices"]

    return run


def build_nlf(args):
    """NLF (github.com/isarandi/nlf, multi-person TorchScript): box-prompted SMPL fit; returns SMPL vertices (m)."""
    import torchvision  # noqa: F401  (registers torchvision::nms, which the TorchScript graph calls)

    model = torch.jit.load(args.nlf_model).cuda().eval()

    @torch.inference_mode()
    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)[None].cuda()
        x1, y1, x2, y2 = bbox.tolist()
        box = [x1, y1, x2 - x1, y2 - y1] if args.nlf_box == "xywh" else [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
        pred = model.estimate_smpl_batched(img, [torch.tensor([box], dtype=torch.float32, device="cuda")],
                                           intrinsic_matrix=torch.from_numpy(K).float()[None].cuda())
        return (pred["vertices3d"][0][0] / 1000.0).float().cpu().numpy()

    return run


def build_predictor(args):
    """--predictor: FAB-L / FAB-S / FAB-H through fab/predictor.FABPredictor on the given box, its released path
    (smpl_fitter="deployed": the released fitter; foot refine and target, 2.5D mix: FABPredictor's defaults, overridden by
    --predictor_kw); the prediction = its SMPL fit (mesh_smpl_fit, SMPL topology); the dense rows = its raw
    points (joints_h36m14 + dense_surface, absolute)."""
    from fab.predictor import FABPredictor

    kw = {"smpl_fitter": "deployed", "detector": "", "bm_dir": args.bm_dir, "fast": bool(args.compile),
          "teacher_dir": str(Path(args.ckpt).parent), "parts": False, "mask": False}
    if args.dense_25d_mix >= 0:
        kw["dense_25d_mix"] = args.dense_25d_mix
    if args.predictor_kw.split(":")[0] == "DEPLOYED":  # FABPredictor(<model>, smpl_fitter="deployed") as released: its own
        for tok in args.predictor_kw.split(":")[1:]:  # defaults (incl. body_engine auto: the TensorRT body built on this GPU, fast=True);
            if tok == "parts":  # DEPLOYED:parts: + the part head's map (as in the default call; read in the crop and
                kw.update(parts=True, crop_maps=True)  # scored against the GT mesh's rendered parts)
    elif args.predictor_kw == "DEPLOYED_NOFIT":  # FABPredictor(<model>, smpl_fitter=""): the MHR mesh (no SMPL fit)
        kw["smpl_fitter"] = ""
    elif args.predictor_kw.startswith("FINAL"):  # FINAL[:eager]: SIMPLE_PROFILE (the released FAB-L / FAB-S call),
        from fab.predictor import SIMPLE_PROFILE  # PyTorch body

        kw.update(SIMPLE_PROFILE, body_engine="")
        if "eager" in args.predictor_kw.split(":")[1:]:
            kw["fast"] = False
    else:
        kw.update(json.loads(args.predictor_kw or "{}"))
    fab = FABPredictor(args.fab_ckpt, **kw)
    print("predictor:", {k: v for k, v in kw.items() if k != "bm_dir"}, "foot_target", fab.foot_target,
          "foot_after", fab.foot_after, "p25d_mix", getattr(fab.model.dense_head, "p25d_mix", None), flush=True)
    extra = {}
    if fab.model.dense_head is not None:
        extra["surface_idx"] = fab.model.dense_head.surface_idx.cpu().numpy()
        extra["residual"] = False  # the predictor's joints / surface are absolute (residual heads put on the mesh)

    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        pp = fab.predict(image, K=np.asarray(K, np.float32), boxes=np.asarray(bbox, np.float32)[None])[0]
        if "joints_h36m14" in pp:
            extra["dense_j14"] = np.concatenate([pp["joints_h36m14"], pp["dense_surface"]]).astype(np.float64)
        if "joints2d_h36m14" in pp:  # --save_dense: the dense head's own 2D of its 14 joints (image px)
            extra["dense_uv14"] = np.asarray(pp["joints2d_h36m14"], np.float32)
        extra["cam_t"] = np.asarray(pp["cam_t"], np.float32)
        extra["kp3d"] = np.asarray(pp["keypoints3d"] - pp["cam_t"], np.float32)
        extra["kp2d_full"] = np.asarray(pp["keypoints2d_projected"], np.float32)
        if "keypoints2d" in pp:  # the dense head's own 2D of the 70 MHR keypoints (image px)
            extra["kp2d_dense70"] = np.asarray(pp["keypoints2d"], np.float32)
        if "keypoint_visibility" in pp:
            extra["vis"] = np.asarray(pp["keypoint_visibility"], np.float32)
        if "parts_crop" in pp:  # the predictor's part map in the crop (256 cells) + its image -> crop affine
            extra["parts"] = (np.asarray(pp["parts_crop"]), np.asarray(pp["crop_affine"], np.float64),
                              np.asarray(K, np.float64), fab.model.size)
        return pp["mesh_smpl_fit"] if "mesh_smpl_fit" in pp else pp["mesh_smpl"]

    run.extra = extra
    return run


def build_method(args):
    if getattr(args, "predictor", False):
        return build_predictor(args)
    if getattr(args, "body_engine", ""):
        if not args.fab_ckpt or args.backbone_engine or args.ms_tta or args.upright:
            raise ValueError("--body_engine requires --fab_ckpt, --upright 0, and no backbone engine or TTA")
        from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
        from fab.data.utils.prepare_batch import prepare_batch
        from fab.tensorrt import TensorRTBody
        from torchvision.transforms import ToTensor

        engine = TensorRTBody(args.body_engine, args.fab_ckpt)
        transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                             VisionTransformWrapper(ToTensor())])

        @torch.inference_mode()
        def run_body(image, bbox, K):
            batch = prepare_batch(image, transform, bbox[None].astype(np.float32), None, None,
                                  cam_int=torch.from_numpy(K).float()[None])
            out = engine(batch["img"][0].cuda(), batch["bbox_center"][0].cuda(),
                         batch["bbox_scale"][0].cuda(), batch["ori_img_size"][0].cuda(),
                         batch["cam_int"].cuda(), affine=batch["affine_trans"][0].cuda())
            return out["pred_vertices"][0].cpu().numpy()

        return run_body
    if args.fab_ckpt:
        return build_fab(args)
    if args.method == "fast3db":
        return build_fast3db(args)
    if args.method == "nlf":
        return build_nlf(args)
    from fab import SAM3DBodyEstimator, load_sam_3d_body

    mhr = args.mhr or str(Path(args.ckpt).parent / "assets" / "mhr_model.pt")
    model, cfg = load_sam_3d_body(checkpoint_path=args.ckpt, device="cuda", mhr_path=mhr)
    if args.dec_layers:  # truncated decoder: the k-th layer's estimate is the output
        model.decoder.layers = model.decoder.layers[: args.dec_layers]
    if args.img_size:  # crop resolution (the DINOv3 backbone uses RoPE; the decoder reads any token grid)
        for c in (cfg, model.cfg):
            c.defrost()
            c.MODEL.IMAGE_SIZE = [args.img_size, args.img_size]
            c.freeze()
    estimator = SAM3DBodyEstimator(sam_3d_body_model=model, model_cfg=cfg)

    def run(image: np.ndarray, bbox: np.ndarray, K: np.ndarray) -> np.ndarray:
        mfull = run.extra.get("mask_full")  # --mask_prompt
        out = estimator.process_one_image(
            image,
            bboxes=bbox[None].astype(np.float32),
            cam_int=torch.from_numpy(K).float()[None],
            inference_type=args.inference_type,
            **({} if mfull is None else {"masks": mfull[None]}),
        )
        run.extra["cam_t"] = np.asarray(out[0]["pred_cam_t"], np.float32)  # --save_dense
        run.extra["kp3d"] = np.asarray(out[0]["pred_keypoints_3d"], np.float32)
        run.extra["kp2d_full"] = np.asarray(out[0]["pred_keypoints_2d"], np.float32)[:, :2]
        return out[0]["pred_vertices"]

    run.extra = {}
    return run


def silhouette(verts, faces, K, H, W):
    """(H, W) uint8 mask of a camera-frame mesh projected with K (--mask_prompt gt)."""
    uvw = verts @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:], 1e-6, None)
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, list(np.round(uv[faces]).astype(np.int32)), 1)
    return m


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    parser.add_argument("--root", required=True)
    parser.add_argument("--bm_dir", default=str(paths.body_models_dir()), help="body_models dir with smpl/ and mhr/ (docs/evaluation.md)")
    parser.add_argument("--ckpt", default=str(paths.sam3d_dir() / "model.ckpt"),
                        help="SAM 3D Body's checkpoint path; its folder gives the config and MHR model (FAB models need only "
                        "those), the network weights are read only for the SAM 3D Body rows")
    parser.add_argument("--mhr", default="")
    parser.add_argument("--inference_type", default="body", choices=["body", "full"])
    parser.add_argument("--fab_ckpt", default="", help="evaluate a FAB model (fab-l / fab-s / fab-h or a path) instead of SAM 3D Body")
    parser.add_argument("--backbone_engine", default="", help="verified TensorRT FAB backbone for this checkpoint")
    parser.add_argument("--body_engine", default="", help="verified fixed one-person full-body TensorRT engine")
    parser.add_argument("--fp32", action="store_true", help="FAB in FP32 throughout (no autocast, TF32 off): the "
                        "reference numerics for engine / precision checks (not with --compile / --fast_inference)")
    parser.add_argument("--trt_engine", default="", help="a verified TensorRT bundle engine of --fab_ckpt: the whole body "
                        "forward (all saved heads) from TensorRT, every metric as for the PyTorch model")
    parser.add_argument("--det_model", default="", help="automatic person detector; empty uses oracle boxes")
    parser.add_argument("--det_conf", type=float, default=.25)
    parser.add_argument("--det_preprocess", choices=["none", "exposure"], default="none",
                        help="opt-in exposure normalization for detection only; body input is unchanged")
    parser.add_argument("--det_min_iou", type=float, default=.3,
                        help="one-to-one evaluation association threshold; missed people remain in recall")
    parser.add_argument("--method", default="sam3db", choices=["sam3db", "fast3db", "nlf"])
    parser.add_argument("--nlf_model", default="", help="--method nlf: NLF's official TorchScript model (download it yourself)")
    parser.add_argument("--nlf_box", default="xywh", choices=["xywh", "cxcywh"])
    parser.add_argument("--fast3db_root", default="", help="--method fast3db: a checkout of Fast SAM 3D Body's official code")
    parser.add_argument("--fast3db_pose_model", default="yolo11m-pose.pt")
    parser.add_argument("--ms_tta", default="", help="FAB: comma-separated box scales averaged (e.g. 0.9,1.0,1.1)")
    parser.add_argument("--box_scale", type=float, default=1.0, help="FAB: scale every box about its centre")
    parser.add_argument("--mask_prompt", default="none", choices=["none", "box", "gt", "neg"],
                        help="person-mask prompt (SAM 3D Body / FAB): the box filled, or the GT mesh silhouette (oracle); "
                        "neg: +1 in the box, -1 in the image's other boxes (the other detections with --det_model, else "
                        "the other annotated people), 0 where both (train_fab --neg_prompt_p)")
    parser.add_argument("--upright", type=float, default=90.0,
                        help="FAB: re-predict on an upright-rolled crop when the torso is tilted more than this (deg)")
    parser.add_argument("--img_size", type=int, default=0, help="SAM 3D Body crop size (0 = its 512)")
    parser.add_argument("--dec_layers", type=int, default=0, help="keep only the first k decoder layers (0 = all)")
    parser.add_argument("--decoder_fp32", action="store_true",
                        help="FAB wrapper: BF16 backbone with FP32 decoder/heads")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--max_samples", type=int, default=-1, help="cap per shard (smoke tests)")
    parser.add_argument("--stride", type=int, default=1, help="evaluate every n-th sample (quick runs)")
    parser.add_argument("--require_images", action="store_true", help="drop samples whose image is missing (smoke tests)")
    parser.add_argument("--flip_tta", action="store_true", help="FAB: average with the mirrored flipped-image mesh")
    parser.add_argument("--track_every", type=int, default=1,
                        help="video mode: boxes from the detector (or oracle) every N frames, else from the previous "
                             "frame's predicted keypoints (needs --stride 1 ordering)")
    parser.add_argument("--track_scale", type=float, default=1.1)
    parser.add_argument("--frame_gap", type=int, default=1, help="source frames between consecutive samples")
    parser.add_argument("--recrop_border", type=float, default=0.0,
                        help="with --recrop: re-crop only when a body keypoint is within this share of the crop size "
                             "of the crop's border (0 = always)")
    parser.add_argument("--recrop", type=float, default=0.0,
                        help="FAB: second pass on the box of the predicted body keypoints scaled by this (0 = off)")
    parser.add_argument("--points_map", default="",
                        help="dense head: surface points -> full mesh map (docs/usage.md; default "
                             "<bm_dir>/smpl/dense512_to_mesh.npz when its points match) -> pve_points (no parameters)")
    parser.add_argument("--fit_smpl", default="", help="SMPL fitter checkpoint (fab/smpl_fitter.py): also fit SMPL to the "
                        "dense surface points and score the fitted mesh (*_fit)")
    parser.add_argument("--fit_iters", type=int, default=30)
    parser.add_argument("--fit_mesh", action="store_true", help="--fit_smpl on the predicted mesh's SMPL-topology "
                        "surface points instead of the dense surface (any model): one pass, *_fit")
    parser.add_argument("--fast_inference", action="store_true",
                        help="FAB: fab/precision.fast_inference (BF16 weights, CSR correctives; same outputs)")
    parser.add_argument("--compile", action="store_true",
                        help="FAB: + torch.compile of backbone and decoder (the released numerics; not bit-identical)")
    parser.add_argument("--save_pred", action="store_true",
                        help="also save SMPL-topology meshes (pred and GT; fp16 offsets from fp32 hip midpoints) to "
                             "<out>/<dataset>_pred_shard<k>.npz (bench/eval_standard_protocol.py, bench/eval_ssp3d.py)")
    parser.add_argument("--foot_refine", type=int, default=0,
                        help="SMPL fit: re-solve the ankle / foot rotations on the MHR mesh's feet (Adam iterations; -k = the "
                             "closed form with k passes, CUDA graph; 0 = off)")
    parser.add_argument("--dense_25d_vis_gate", action="store_true",
                        help="dense head 2.5D lift: each keypoint's lift weighted by the visibility head's P(visible)")
    parser.add_argument("--vis_head_from", default="",
                        help="load the visibility head (vis_head.*) from this checkpoint")
    parser.add_argument("--dense_25d_mix", type=float, default=-1.0,
                        help="dense head: weight of the 2.5D lift (each point on its heatmap 2D ray at its depth; "
                             "fab/dense_head.lift25d) at test time, any head (-1 = the checkpoint's own)")
    parser.add_argument("--save_dense", action="store_true",
                        help="with --save_pred: also save cam_t (the mesh's camera translation), the mesh's 70 MHR keypoints "
                        "(kp3d in the saved mesh frame, kp2d image px) and the dense head's own 70 MHR keypoints / 14 "
                        "joints (dense_kp3d / dense_j14 hip-centred 3D, dense_kp2d / dense_uv14 image px)")
    parser.add_argument("--foot_refine_mode", default="kabsch", choices=["kabsch", "swing", "hybrid"],
                        help="closed-form foot refine: whole-foot Kabsch, or heel -> toe direction only (fit's sole roll kept)")
    parser.add_argument("--foot_refine_gate", type=float, default=0.0, help="leave feet within this many deg of the target")
    parser.add_argument("--foot_refine_gate_hi", type=float, default=20.0,
                        help="--foot_refine_mode hybrid: kabsch for feet more than this many deg off the target, swing below")
    parser.add_argument("--save_fit", action="store_true",
                        help="with --save_pred and --fit_smpl: also save the fitted SMPL (fit_verts / fit_root)")
    parser.add_argument("--predictor", action="store_true",
                        help="with --fab_ckpt: the released path through FABPredictor (build_predictor); the scored mesh "
                        "is its SMPL fit, the *_dense rows its raw points; --fit_smpl / --foot_refine unused")
    parser.add_argument("--predictor_kw", default="", help="--predictor: FINAL[:eager] (the released FAB-L / FAB-S call, "
                        "PyTorch body), DEPLOYED[:parts] (FABPredictor's defaults, incl. engines built on this GPU), "
                        "DEPLOYED_NOFIT (the MHR mesh, no SMPL fit) or FABPredictor keyword overrides (JSON)")
    parser.add_argument("--parts_res", type=int, default=0, help="part maps scored at this resolution (logits upsampled "
                        "bilinear first, as FABPredictor's 256 crop maps); 0 = the head's own")
    parser.add_argument("--parts_bary", default="3", choices=["3", "adaptive"], help="GT part render: 3 samples per "
                        "triangle edge (the protocol) or body_parts.adaptive_bary (no holes at fine resolutions)")
    args = parser.parse_args()
    if args.fab_ckpt and args.fab_ckpt != "teacher":  # a released model name (fab-l / fab-s / fab-h) -> its weights file
        args.fab_ckpt = str(paths.model_file(args.fab_ckpt))
    if args.decoder_fp32 and (not args.fab_ckpt or args.body_engine):
        parser.error("--decoder_fp32 requires a FAB wrapper checkpoint and no full-body engine")

    samples = DATASETS[args.dataset](args.root)
    if args.dataset in ("aistpp", "aistpp_dark") and samples and all(
        not np.any(sample["model_params"]) for sample in samples
    ):
        parser.error("AIST MHR targets are zero-parameter placeholders, not ground-truth meshes. "
                     "Score AIST++ on its calibrated joints instead.")
    if args.require_images:
        samples = [s for s in samples if Path(s["image"]).exists()]
    boxes_of_image = {}  # --mask_prompt neg (annotated boxes): the image's people
    for s_ in samples:
        boxes_of_image.setdefault(s_["image"], []).append((s_["id"], np.asarray(s_["bbox"], np.float64)))
    detector = None
    if args.det_model:
        from bench.detected_boxes import DetectedBoxes

        detector = DetectedBoxes(samples, args.det_model, args.det_conf, args.det_min_iou, args.det_preprocess)
    samples = samples[:: args.stride]
    shard = samples[args.shard :: args.num_shards]
    if args.max_samples > 0:
        shard = shard[: args.max_samples]
    print(f"{args.dataset}: {len(samples)} samples, shard {args.shard}/{args.num_shards}: {len(shard)}", flush=True)

    bm = Path(args.bm_dir)
    smpl = {g: SMPL(bm / "smpl" / f"SMPL_{g.upper()}.pkl").cuda().eval() for g in ("neutral", "male", "female")}
    to_smpl = MHRToSMPL(bm / "mhr").cuda()
    j14 = np.load(bm / "smpl" / "J_regressor_h36m.npy")[H36M_TO_J14].astype(np.float64)
    j24 = smpl["neutral"].J_regressor.cpu().numpy().astype(np.float64)
    run = build_method(args)
    corrupt = CORRUPTIONS.get(args.dataset.rsplit("_", 1)[-1]) if "_" in args.dataset else None
    gt_rig = None
    if shard and "model_params" in shard[0]:  # MHR ground truth
        from fab.models.heads.mhr_fast import SubsetMHR

        mhr_path = args.mhr or str(Path(args.ckpt).parent / "assets" / "mhr_model.pt")
        gt_rig = SubsetMHR(torch.jit.load(mhr_path, map_location="cuda"), None).cuda()
        yz_flip = torch.tensor([1.0, -1.0, -1.0], device="cuda")
    world = args.dataset.split("_")[0] in WORLD_FRAME

    ids, rows, times, saved, missing_images = [], [], [], {}, []
    tracks = {}  # --track_every: last frame and keypoint box per track
    gt_files = {}  # memory-mapped GT vertex files
    dense_interp = None
    points_map = None
    part_geo = None
    smpl_fitter = graph_fit = None
    foot_refiner = None
    if args.foot_refine != 0:
        from fab.foot_refine import FootRefine

        from fab.foot_refine import GraphFootRefine

        foot_refiner = (FootRefine(smpl["neutral"], iters=args.foot_refine) if args.foot_refine > 0 else
                        GraphFootRefine(smpl["neutral"], passes=-args.foot_refine, mode=args.foot_refine_mode,
                                        gate=args.foot_refine_gate, gate_hi=args.foot_refine_gate_hi))  # -k: closed form, k passes, graphed
    if getattr(run, "extra", {}).get("surface_idx") is not None:
        path = Path(args.points_map or bm / "smpl" / "dense512_to_mesh.npz")
        if path.exists():
            z = np.load(path)
            if np.array_equal(z["idx"], run.extra["surface_idx"]):
                points_map = z["A"].astype(np.float64)
    for i, s in enumerate(shard):
        image = cv2.imread(s["image"])
        if image is None:
            print(f"missing image {s['image']}", flush=True)
            missing_images.append(s["id"])
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if corrupt is not None:
            image = corrupt(image, s["id"])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        bbox = detector.select(image, s) if detector is not None else s["bbox"]
        if args.track_every > 1 and bbox is not None:
            # video mode: detect every N frames, otherwise crop on the person's
            # previous-frame predicted keypoints (x args.track_scale); a track = sequence + person
            parts = s["id"].split("/")  # EMDB: <subject>/<sequence>/<frame>; 3DPW: <sequence>/<frame>/<person>
            if args.dataset.startswith("emdb"):
                tid, fr = "/".join(parts[:2]), int(parts[2])
            else:
                tid, fr = (parts[0], parts[2]), int(parts[1])
            prev = tracks.get(tid)
            if prev is not None and fr - prev[0] <= args.frame_gap and (fr // args.frame_gap) % args.track_every:
                bbox = prev[1]
        if bbox is None:
            continue
        if args.mask_prompt != "none":  # person-mask prompt: the box, or the GT mesh's silhouette (oracle)
            H_, W_ = image.shape[:2]
            m_ = np.zeros((H_, W_), np.uint8)
            if args.mask_prompt == "box":
                x1_, y1_, x2_, y2_ = np.clip(np.round(bbox), 0, [W_, H_, W_, H_]).astype(int)
                m_[y1_:y2_, x1_:x2_] = 1
            elif args.mask_prompt == "neg":  # +1 own box, -1 the other boxes, 0 both
                own_, oth_ = np.zeros((H_, W_), np.float32), np.zeros((H_, W_), np.float32)
                x1_, y1_, x2_, y2_ = np.clip(np.round(bbox), 0, [W_, H_, W_, H_]).astype(int)
                own_[y1_:y2_, x1_:x2_] = 1
                others_ = (getattr(detector, "last_others", []) if detector is not None else
                           [b_ for i_, b_ in boxes_of_image.get(s["image"], []) if i_ != s["id"]])
                for ob_ in others_:
                    x1_, y1_, x2_, y2_ = np.clip(np.round(ob_), 0, [W_, H_, W_, H_]).astype(int)
                    oth_[y1_:y2_, x1_:x2_] = 1
                m_ = own_ - oth_
            else:
                t_ = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None].cuda()  # noqa: E731
                with torch.no_grad():
                    gw_, _ = smpl[s["gender"]](t_(s["global_orient"]), t_(s["body_pose"]), t_(s["betas"]), t_(s["transl"]))
                ext_ = s["extrinsic"].astype(np.float64)
                g_ = gw_[0].cpu().numpy().astype(np.float64) @ ext_[:3, :3].T + ext_[:3, 3]
                m_ = silhouette(g_, smpl["neutral"].faces.cpu().numpy(), s["K"].astype(np.float64), H_, W_)
            run.extra["mask_full"] = m_
        pred = run(image, bbox, s["K"])
        if args.track_every > 1 and "kp2d" in getattr(run, "extra", {}):
            kp = run.extra["kp2d"]
            H, W = image.shape[:2]
            c, wh = (kp.min(0) + kp.max(0)) / 2, (kp.max(0) - kp.min(0)) * args.track_scale
            tracks[tid] = (fr, np.concatenate([c - wh / 2, c + wh / 2]).clip(0, [W - 1, H - 1, W - 1, H - 1]).astype(np.float32))
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)

        with torch.no_grad():
            pred = np.asarray(pred, dtype=np.float32)
            # MHR meshes (18439 vertices) go through the barycentric transfer; SMPL methods (6890) are scored as is
            pred_smpl = pred if pred.shape[0] == 6890 else to_smpl(torch.from_numpy(pred).cuda()).cpu().numpy()
            t = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32))[None].cuda()  # noqa: E731
            if gt_rig is not None:
                jp, skel = gt_rig.skeleton(t(s["model_params"]))
                gt_mhr = gt_rig.vertices(t(s["shape_params"]), jp, skel)[0] / 100 * yz_flip
                gt_cam = to_smpl(gt_mhr).cpu().numpy()
            elif "gt_verts_file" in s:  # SMPL-topology GT mesh in the world frame (bench/build_rich_test.py)
                if s["gt_verts_file"] not in gt_files:
                    gt_files[s["gt_verts_file"]] = np.load(s["gt_verts_file"], mmap_mode="r")
                ext = s["extrinsic"]
                gt_cam = gt_files[s["gt_verts_file"]][s["gt_verts_row"]].astype(np.float32) @ ext[:3, :3].T + ext[:3, 3]
            else:
                gt_world, _ = smpl[s["gender"]](t(s["global_orient"]), t(s["body_pose"]), t(s["betas"]), t(s["transl"]))
                ext = s["extrinsic"]
                gt_cam = gt_world[0].cpu().numpy() @ ext[:3, :3].T + ext[:3, 3]
        row = errors(pred_smpl.astype(np.float64), gt_cam.astype(np.float64), j14, j24)
        vfit = None
        if args.recrop > 0:
            row["recropped"] = float(getattr(run, "extra", {}).get("recropped", 0.0))
        dpts = getattr(run, "extra", {}).get("dense_j14")
        if args.fit_smpl and args.fit_mesh:  # one-pass fit to the mesh's own surface points (replaces the dense-surface fit)
            if graph_fit is None:
                from fab.dense_head import symmetric_subset
                from fab.fast_fit import GraphFit
                from fab.smpl_fitter import load_fitter

                mesh_idx = symmetric_subset(smpl["neutral"].v_template.cpu().numpy(), 512)[0]
                graph_fit = GraphFit(load_fitter(args.fit_smpl), smpl["neutral"], mesh_idx)
            torch.cuda.synchronize()
            t_fit = time.perf_counter()
            R_f, b_f, vfit = graph_fit(torch.from_numpy(pred_smpl[mesh_idx][None]).float().cuda())
            if args.foot_refine != 0:  # feet re-solved on the MHR mesh's feet (fab/foot_refine.py)
                vfit = foot_refiner(R_f, b_f, vfit, torch.from_numpy(pred_smpl[None]).float().cuda())[1]
            torch.cuda.synchronize()
            row["ms_fit"] = (time.perf_counter() - t_fit) * 1000
            e = errors(vfit[0].cpu().numpy().astype(np.float64), gt_cam.astype(np.float64), j14, j24)
            for k in ("pve", "mpjpe", "pa_mpjpe", "mpjpe24", "pa_mpjpe24"):
                row[f"{k}_fit"] = e[k]
        if dpts is not None:  # the dense head's own skeleton (not regressed from the mesh) and surface
            dj, dv = dpts[:14], dpts[14:]
            if run.extra.get("residual"):  # offsets on the mesh's own regressed joints / mapped vertices
                dj = dj + j14 @ pred_smpl.astype(np.float64)
                dv = dv + pred_smpl.astype(np.float64)[run.extra["surface_idx"]]
            gj = j14 @ gt_cam.astype(np.float64)
            pr, gr = dj - dj[[2, 3]].mean(0), gj - gj[[2, 3]].mean(0)
            row["mpjpe_dense"] = float(np.linalg.norm(pr - gr, axis=1).mean() * 1000)
            row["pa_mpjpe_dense"] = float(np.linalg.norm(procrustes(pr, gr) - gr, axis=1).mean() * 1000)
            pm = j14 @ pred_smpl.astype(np.float64)
            pf = 0.5 * (pr + pm - pm[[2, 3]].mean(0))  # fused: mean of the dense and the mesh skeletons
            row["mpjpe_fuse_dense"] = float(np.linalg.norm(pf - gr, axis=1).mean() * 1000)
            row["pa_mpjpe_fuse_dense"] = float(np.linalg.norm(procrustes(pf, gr) - gr, axis=1).mean() * 1000)
            if dense_interp is None:  # template-space interpolation of the surface points' offsets to all vertices
                sidx = run.extra["surface_idx"]
                tmpl = smpl["neutral"].v_template.cpu().numpy().astype(np.float64)
                d = np.linalg.norm(tmpl[:, None] - tmpl[sidx][None], axis=-1)
                nn_ = np.argsort(d, 1)[:, :8]
                wgt = np.exp(-(np.take_along_axis(d, nn_, 1) / 0.04) ** 2) + 1e-9
                dense_interp = (sidx, nn_, wgt / wgt.sum(1, keepdims=True))
            sidx, nn_, wgt = dense_interp
            ps = pred_smpl.astype(np.float64)
            pj_m = j14 @ ps
            dv_cam = dv - dj[[2, 3]].mean(0) + pj_m[[2, 3]].mean(0)  # surface in the mesh's hip frame
            off = dv_cam - ps[sidx]
            mesh_d = ps + (off[nn_] * wgt[..., None]).sum(1)
            row["pve_dense"] = errors(mesh_d, gt_cam.astype(np.float64), j14, j24)["pve"]
            if points_map is not None:  # parameter-free mesh: the surface points alone through the linear map
                c = dv.mean(0)
                e = errors(points_map @ (dv - c) + c, gt_cam.astype(np.float64), j14, j24)
                row["pve_points"], row["mpjpe_points"], row["pa_mpjpe_points"] = e["pve"], e["mpjpe"], e["pa_mpjpe"]
                row["mpjpe24_points"], row["pa_mpjpe24_points"] = e["mpjpe24"], e["pa_mpjpe24"]
            if args.fit_smpl and not args.fit_mesh:  # SMPL parameters fitted to the dense surface (learned fitter + L-BFGS), its mesh scored
                from fab.smpl_fit import fit_smpl

                if smpl_fitter is None:
                    from fab.smpl_fitter import load_fitter

                    smpl_fitter = load_fitter(args.fit_smpl)
                    # a head with more surface points than the fitter's (--dense_remap): fit the fitter's subset
                    fit_idx = np.asarray(torch.load(args.fit_smpl, map_location="cpu", weights_only=False)["idx"])
                    pos_of = {int(v): i for i, v in enumerate(run.extra["surface_idx"])}
                    fit_pos = np.arange(len(run.extra["surface_idx"])) if np.array_equal(
                        fit_idx, run.extra["surface_idx"]) else np.array([pos_of[int(v)] for v in fit_idx])
                    if args.fit_iters == 0:  # the deployment path (fab/fast_fit.py, as FABPredictor runs it)
                        from fab.fast_fit import GraphFit

                        graph_fit = GraphFit(smpl_fitter, smpl["neutral"], run.extra["surface_idx"][fit_pos])
                pts = torch.from_numpy(dv[fit_pos][None]).float().cuda()
                sidx_t = torch.as_tensor(run.extra["surface_idx"][fit_pos]).cuda()
                torch.cuda.synchronize()
                t_fit = time.perf_counter()
                if args.fit_iters > 0:
                    with torch.no_grad():
                        init = smpl_fitter(pts, smpl["neutral"], sidx_t)
                    _, _, _, vfit = fit_smpl(smpl["neutral"], pts, sidx_t, iters=args.fit_iters, init=init)
                else:  # the learned fitter alone (the one-pass deployment path): its mesh placed at the points
                    R_f, b_f, vfit = graph_fit(pts)
                    if args.foot_refine != 0:  # feet re-solved on the MHR mesh's feet (fab/foot_refine.py)
                        vfit = foot_refiner(R_f, b_f, vfit, torch.from_numpy(pred_smpl[None]).float().cuda())[1]
                torch.cuda.synchronize()
                row["ms_fit"] = (time.perf_counter() - t_fit) * 1000
                vf = vfit[0].cpu().numpy().astype(np.float64)
                e = errors(vf, gt_cam.astype(np.float64), j14, j24)
                for k in ("pve", "mpjpe", "pa_mpjpe", "mpjpe24", "pa_mpjpe24"):
                    row[f"{k}_fit"] = e[k]
                # cheap fusions with the parametric mesh: the fit in the mesh's global orientation (Kabsch about the
                # centroids), and the vertex mean of the fit and the mesh (centroids aligned)
                ps = pred_smpl.astype(np.float64)
                cf, cm = vf.mean(0), ps.mean(0)
                U, _, Vt = np.linalg.svd((vf - cf).T @ (ps - cm))
                Rk = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
                for tag, v in (("fitor", (vf - cf) @ Rk.T + cf), ("fitmix", 0.5 * (vf - cf + ps - cm) + cf)):
                    e = errors(v, gt_cam.astype(np.float64), j14, j24)
                    for k in ("pve", "mpjpe", "pa_mpjpe", "mpjpe24", "pa_mpjpe24"):
                        row[f"{k}_{tag}"] = e[k]
        parts = getattr(run, "extra", {}).get("parts")
        if parts is not None:  # part segmentation vs the GT mesh's rendered parts (fab/body_parts.py)
            from fab.body_parts import N_PARTS, barycentric_grid, face_parts, render_parts

            pred_lab, A_c, K_c, S_c = parts
            if part_geo is None:
                body = smpl["neutral"]
                part_geo = (body.faces.long().cpu(), face_parts(body.weights.cpu(), body.faces.cpu()), barycentric_grid(3))
            g = gt_cam.astype(np.float64) @ K_c.T
            uv = g[:, :2] / np.clip(g[:, 2:3], 1e-3, None)
            uv = (uv @ A_c[:, :2].T + A_c[:, 2]) / S_c
            uv_t = torch.from_numpy(uv[None]).float()
            bary = part_geo[2]
            if args.parts_bary == "adaptive":  # GT samples dense enough for the map's resolution (no holes on close-ups)
                from fab.body_parts import adaptive_bary

                bary = adaptive_bary(uv_t, part_geo[0], pred_lab.shape)
            gl = render_parts(uv_t, torch.from_numpy(gt_cam[None, :, 2]).float(),
                              part_geo[0], part_geo[1], pred_lab.shape, bary)[0].numpy()
            fg_p, fg_g = pred_lab > 0, gl > 0
            row["sil_iou_parts"] = float((fg_p & fg_g).sum() / max((fg_p | fg_g).sum(), 1))
            ious = [((pred_lab == c) & (gl == c)).sum() / ((pred_lab == c) | (gl == c)).sum()
                    for c in range(1, N_PARTS) if ((pred_lab == c) | (gl == c)).any()]
            row["miou_parts"] = float(np.mean(ious)) if ious else 1.0
        if args.save_pred:
            keep = (("pred", pred_smpl), ("gt", gt_cam))
            if args.save_fit and vfit is not None:
                keep += (("fit", vfit[0].float().cpu().numpy()),)
            for name, v in keep:
                c = (j14[[2, 3]] @ v).mean(0)
                saved.setdefault(f"{name}_root", []).append(c.astype(np.float32))
                saved.setdefault(f"{name}_verts", []).append((v - c).astype(np.float16))
            ex = getattr(run, "extra", {})
            if args.save_dense:  # camera translation, and the dense head's own 14 joints (3D hip-centred, 2D image px)
                saved.setdefault("cam_t", []).append(np.asarray(ex.get("cam_t", np.full(3, np.nan)), np.float32))
                c_ = (j14[[2, 3]] @ pred_smpl).mean(0)  # the mesh's 70 MHR keypoints in the saved mesh frame (+ pred_root)
                saved.setdefault("kp3d", []).append((np.asarray(ex.get("kp3d", np.full((70, 3), np.nan)), np.float64)
                                                     - c_).astype(np.float32))
                saved.setdefault("kp2d", []).append(np.asarray(ex.get("kp2d_full", np.full((70, 2), np.nan)), np.float32))
                if ex.get("dense_kp3d") is not None:  # the dense head's 70 MHR keypoints: 3D hip-centred, 2D image px
                    kd_ = np.asarray(ex["dense_kp3d"], np.float64)
                    saved.setdefault("dense_kp3d", []).append((kd_ - kd_[[9, 10]].mean(0)).astype(np.float32))
                    saved.setdefault("dense_kp2d", []).append(np.asarray(ex["dense_kp2d"], np.float32))
                dj_ = ex.get("dense_j14")
                if dj_ is not None:
                    dj_ = np.asarray(dj_, np.float64)[:14]  # extra["dense_j14"]: the 14 joints, then the surface
                    if ex.get("residual"):
                        dj_ = dj_ + j14 @ pred_smpl.astype(np.float64)
                    saved.setdefault("dense_j14", []).append((dj_ - dj_[[2, 3]].mean(0)).astype(np.float32))
                    saved.setdefault("dense_uv14", []).append(np.asarray(ex.get("dense_uv14", np.full((14, 2), np.nan)),
                                                                         np.float32))
            if "contact" in getattr(run, "extra", {}):
                saved.setdefault("contact", []).append(run.extra["contact"].astype(np.float32))
            if args.save_dense and "vis" in ex:  # visibility head probabilities
                saved.setdefault("vis", []).append(ex["vis"].astype(np.float16))
        if world:
            row = {k: (v if k.startswith("pa_") else float("nan")) for k, v in row.items()}
        rows.append(row)
        ids.append(s["id"])
        if i % 200 == 0:
            mean = {k: np.mean([r[k] for r in rows]) for k in rows[0]}
            print(f"[{i}/{len(shard)}] " + " ".join(f"{k}={v:.1f}" for k, v in mean.items())
                  + f" ms={np.median(times):.1f}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ("mpjpe", "pa_mpjpe", "pve", "mpjpe24", "pa_mpjpe24")
    keys += tuple(k for k in (rows[0] if rows else {}) if k.endswith(("_dense", "_points", "_parts", "_fit", "_fitor", "_fitmix", "recropped")))  # dense-head metrics
    np.savez(out / f"{args.dataset}_shard{args.shard}.npz", ids=np.array(ids, dtype=str),
             **{k: np.array([r[k] for r in rows]) for k in keys}, ms=np.array(times))
    if detector is not None:
        report = {"detector": args.det_model, "confidence": args.det_conf, "min_iou": args.det_min_iou,
                  "detector_preprocess": args.det_preprocess,
                  "requested_samples": len(shard), "missing_images": missing_images,
                  "detected_samples": len(ids), "records": detector.records,
                  "protocol": "Known intrinsics; GT boxes only associate detections one-to-one. "
                              "No oracle fallback. Pose metrics condition on detected people; report recall "
                              "alongside them. Unannotated detections are not labeled false positives. "
                              "Dark corruption remains deterministic per sample, matching the oracle harness."}
        (out / f"{args.dataset}_detections_shard{args.shard}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.save_pred and saved:
        np.savez(out / f"{args.dataset}_pred_shard{args.shard}.npz", ids=np.array(ids),
                 **{k: np.stack(v) for k, v in saved.items()})
    print("done", {k: round(float(np.mean([r[k] for r in rows])), 2) for k in keys} if rows else
          {"detected_samples": 0}, flush=True)


if __name__ == "__main__":
    main()
