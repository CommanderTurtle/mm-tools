"""2D keypoints on COCO val2017 (GT person boxes, OKS keypoint AP via pycocotools): the 17 COCO keypoints taken from
(a) the projection of the 3D MHR keypoints (the parametric output: SAM 3D Body, FAB) and (b) FAB's dense-head
heatmaps (an independent 2D output). Default intrinsics (no camera is known for COCO images). The paper's COCO rows:
--predictor --body_engine torch (the released call with the PyTorch body).

    python -m bench.eval_coco2d --root $FAB_DATA/coco_val --fab_ckpt fab-l --predictor --body_engine torch --out <dir> --shard 0 --num_shards 4
    python -m bench.eval_coco2d --root $FAB_DATA/coco_val --out <dir> --merge
"""

import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from fab import paths

MHR_TO_COCO = [0, 1, 2, 3, 4, 5, 6, 7, 8, 62, 41, 9, 10, 11, 12, 13, 14]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--fab_ckpt", default="", help="FAB model (fab-l / fab-s / fab-h or a path; empty: SAM 3D Body)")
    p.add_argument("--ckpt", default=str(paths.sam3d_dir() / "model.ckpt"), help="SAM 3D Body (its network weights only for the SAM 3D Body rows)")
    p.add_argument("--out", required=True)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--merge", action="store_true")
    p.add_argument("--fit_smpl", default="", help="FAB: SMPL fitter checkpoint: also score the fitted SMPL's MHR body "
                   "keypoints (regressed from its vertices by the 23-joint regressor --reg23): one-pass (smpl_onepass) and "
                   "the refined fit (smpl_densefit, fab/gn_fit.DenseFit)")
    p.add_argument("--reg23", default=str(paths.home() / "extras" / "mhr23_regressor.npz"), help="optional (not part of the download): the fit row")
    p.add_argument("--predictor", action="store_true", help="the released path, FABPredictor(--fab_ckpt) with its "
                   "defaults: rows dense (its keypoints2d), projected (the MHR head's keypoints2d_projected) and fit (the "
                   "fitted SMPL mesh_smpl_fit's MHR body keypoints through --reg23, projected with the default K; only when --reg23 exists)")
    p.add_argument("--body_engine", default="auto", help="--predictor: FABPredictor's body_engine ('auto' = the default, the bundle's "
                   "TensorRT engine when one was built; 'torch' = the PyTorch body, the paper's COCO row; or an engine path)")
    p.add_argument("--dense_w2d", default="0.1", help="--fit_smpl: 2D weights of the refined fit (one row each)")
    p.add_argument("--bm_dir", default=str(paths.body_models_dir()))
    p.add_argument("--decoder_fp32", action="store_true",
                   help="FAB wrapper: BF16 backbone, FP32 decoder/heads (precision-parity diagnostic)")
    p.add_argument("--local_windows", default="",
                   help="dense head: also score the soft-argmax within r heatmap cells of each peak (comma-separated r)")
    args = p.parse_args()
    windows = [int(r) for r in args.local_windows.split(",") if r]
    from pycocotools.coco import COCO

    ann_file = Path(args.root) / "annotations" / "person_keypoints_val2017.json"
    coco = COCO(str(ann_file))
    out = Path(args.out)
    if args.merge:
        from pycocotools.cocoeval import COCOeval

        res = {}
        for f in sorted(glob.glob(str(out / "shard*.json"))):
            for name, rows in json.loads(Path(f).read_text()).items():
                res.setdefault(name, []).extend(rows)
        summary = {}
        for name, rows in res.items():
            ev = COCOeval(coco, coco.loadRes(rows), "keypoints")
            ev.params.imgIds = sorted({r["image_id"] for r in rows})
            ev.evaluate(); ev.accumulate(); ev.summarize()
            summary[name] = {"AP": ev.stats[0], "AP50": ev.stats[1], "AP75": ev.stats[2], "AR": ev.stats[5],
                             "n": len(rows)}
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        return

    from fab.data.transforms import Compose, GetBBoxCenterScale, TopdownAffine, VisionTransformWrapper
    from fab.data.utils.prepare_batch import prepare_batch
    from torchvision.transforms import ToTensor

    anns = [a for a in coco.loadAnns(coco.getAnnIds(catIds=[1], iscrowd=False)) if a["num_keypoints"] > 0]
    anns = anns[args.shard :: args.num_shards]
    if args.predictor:  # the released path (FABPredictor defaults), K = the image-diagonal focal as prepare_batch's
        from fab.predictor import FABPredictor

        fab = FABPredictor(args.fab_ckpt, detector="", bm_dir=args.bm_dir, fast=True, parts=False, mask=False,
                           body_engine="" if args.body_engine == "torch" else args.body_engine)
        reg23 = np.load(args.reg23)["reg"].astype(np.float64) if Path(args.reg23).exists() else None
        body23 = list(range(21)) + [41, 62]
        rows = {"dense": [], "projected": [], "fit": []}
        for k, a in enumerate(anns):
            info = coco.loadImgs(a["image_id"])[0]
            img = cv2.cvtColor(cv2.imread(str(Path(args.root) / "val2017" / info["file_name"])), cv2.COLOR_BGR2RGB)
            x, y, w_, h_ = a["bbox"]
            H, W = img.shape[:2]
            f = float(np.hypot(H, W))
            K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], np.float64)
            pp = fab.predict(img, K.astype(np.float32), boxes=np.array([[x, y, x + w_, y + h_]], np.float32))[0]
            rows["dense"].append((a, np.asarray(pp["keypoints2d"], np.float64)))
            rows["projected"].append((a, np.asarray(pp["keypoints2d_projected"], np.float64)))
            if reg23 is not None:
                q = (reg23 @ np.asarray(pp["mesh_smpl_fit"], np.float64)) @ K.T
                full70 = np.zeros((70, 2))
                full70[body23] = q[:, :2] / np.maximum(q[:, 2:], 1e-3)
                rows["fit"].append((a, full70))
            if k % 500 == 0:
                print(f"[{k}/{len(anns)}]", flush=True)
        res = {name: [{"image_id": a["image_id"], "category_id": 1, "score": 1.0,
                       "keypoints": np.concatenate([kp[MHR_TO_COCO], np.full((17, 1), 2.0)], 1).ravel().tolist()}
                      for a, kp in rr] for name, rr in rows.items() if rr}
        out.mkdir(parents=True, exist_ok=True)
        (out / f"shard{args.shard}.json").write_text(json.dumps(res))
        print("done", {k: len(v) for k, v in res.items()}, flush=True)
        return
    transform = Compose([GetBBoxCenterScale(), TopdownAffine(input_size=[512, 512], use_udp=False),
                         VisionTransformWrapper(ToTensor())])
    if args.fab_ckpt:
        from fab.decoder_model import build_fab_decoder

        # "teacher": SAM 3D Body's own weights through the same one-pass wrapper (path check)
        state = torch.load(args.fab_ckpt, map_location="cpu", weights_only=False) if args.fab_ckpt != "teacher" else \
            {"args": {"arch": "fabt", "size": 512}, "model": {}}
        ta = state.get("args", {})
        model, tb = build_fab_decoder(args.ckpt, ta.get("backbone", "dinov3_vitb16"), ta.get("size", 512),
                                      pretrained=False, dec_layers=ta.get("dec_layers", 0))
        if ta.get("arch") == "fabt":  # fine-tuned SAM 3D Body: its own backbone (weights in the state)
            model.body.backbone = tb
            model.body.backbone_dtype = torch.bfloat16
        else:
            del tb
        w = dict(state["model"])
        w.update(state.get("ema") or {})
        if "dense_head.surface_idx" in w:
            from fab.dense_head import dense_hr, dense_mask_depth, dense_mask_query, dense_part_depth, dense_xyz_layers
            from fab.dense_head import dense_head_kwargs

            model.add_dense_head(w["dense_head.surface_idx"].tolist(),
                                 residual=bool(int(w.get("dense_head.residual", torch.tensor(0)))), parts="dense_head.part_emb" in w,
                                 xyz_layers=dense_xyz_layers(w), part_depth=dense_part_depth(w),
                                 hr=dense_hr(w), mask_depth=dense_mask_depth(w), mask_query=dense_mask_query(w),
                                 **dense_head_kwargs(w)).cuda()
            model.dense_head.keep_logits = bool(args.local_windows)
        model.load_state_dict(w, strict=False)
        model = model.eval()
        model.body.head_pose.enable_fast()
        if args.decoder_fp32:
            model.body.backbone.forward = torch.autocast("cuda", dtype=torch.bfloat16)(
                model.body.backbone.forward)
    else:
        from fab import SAM3DBodyEstimator, load_sam_3d_body

        sam, cfg = load_sam_3d_body(checkpoint_path=args.ckpt, device="cuda",
                                    mhr_path=str(Path(args.ckpt).parent / "assets" / "mhr_model.pt"))
        est = SAM3DBodyEstimator(sam_3d_body_model=sam, model_cfg=cfg)

    rows = {"projected": [], "dense": []}
    fits = None
    if args.fab_ckpt and args.fit_smpl and model.dense_head is not None:  # SMPL fits of the dense head's outputs
        from bench.body_models import SMPL
        from bench.mhr_smpl import MHRToSMPL
        from fab.fast_fit import GraphFit
        from fab.gn_fit import DenseFit, GraphDenseFit, crop_camera
        from fab.smpl_fitter import load_fitter

        body = SMPL(Path(args.bm_dir) / "smpl" / "SMPL_NEUTRAL.pkl").cuda().eval()
        sidx = model.dense_head.surface_idx
        reg23 = torch.from_numpy(np.load(args.reg23)["reg"]).float().cuda()
        one = GraphFit(load_fitter(args.fit_smpl), body, sidx)
        dense_fits = {float(w): GraphDenseFit(DenseFit(one.fit, body, sidx, reg23, w_2d=float(w)))
                      for w in args.dense_w2d.split(",")}
        fits = (MHRToSMPL(Path(args.bm_dir) / "mhr").cuda(), one, dense_fits, reg23, crop_camera)
        body23 = list(range(21)) + [41, 62]  # MHR70 body keypoints (their COCO subset scored like the others)
    for k, a in enumerate(anns):
        info = coco.loadImgs(a["image_id"])[0]
        img = cv2.cvtColor(cv2.imread(str(Path(args.root) / "val2017" / info["file_name"])), cv2.COLOR_BGR2RGB)
        x, y, w_, h_ = a["bbox"]
        box = np.array([x, y, x + w_, y + h_], np.float32)
        if args.fab_ckpt:
            b = prepare_batch(img, transform, box[None], None, None)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                        enabled=not args.decoder_fp32):
                o = model(b["img"][0].cuda(), b["bbox_center"][0].cuda(), b["bbox_scale"][0].cuda(),
                          b["ori_img_size"][0].cuda(), b["cam_int"].cuda(), affine=b["affine_trans"][0].cuda(),
                          return_vertices=fits is not None)
            if fits is not None and "dense_uv" in o:
                to_smpl, one, dense_fits, reg23, crop_camera = fits
                with torch.no_grad():
                    mesh = to_smpl(o["pred_vertices"][0].float())[None]
                    off = o["dense_xyz"][0].float()
                    shift = off[70:84][[2, 3]].mean(0)
                    surf = (off[84:] - shift + mesh[0, sidx])[None]
                    mo = off[:70][body23]
                    kp3 = (mo - mo[[9, 10]].mean(0) + reg23 @ mesh[0])[None]
                    K = b["cam_int"][0].float().cuda()
                    Kc = crop_camera(K[None], b["affine_trans"][0].float().cuda())
                    cam_t = o["pred_cam_t"][0].float()
                    _, _, v1 = one(surf)
                    v1 = v1[0] + cam_t
                    fitted = [(f"smpl_densefit_w{w:g}", dense(surf, mesh, kp3, (o["dense_uv"][0].float() * 512)[body23][None],
                                                             Kc)[2][0].clone()) for w, dense in dense_fits.items()]
                    for name, v in [("mesh_reg23", mesh[0] + cam_t), ("smpl_onepass", v1)] + fitted:
                        q = (reg23 @ v) @ K.T  # MHR body keypoints of the fitted mesh, projected (image px)
                        full70 = np.zeros((70, 2))
                        full70[body23] = (q[:, :2] / q[:, 2:].clamp_min(1e-3)).cpu().numpy()
                        rows.setdefault(name, []).append((a, full70))
            kp = o["pred_keypoints_2d"][0, :, :2].float().cpu().numpy()
            if "dense_uv" in o:
                A = np.vstack([b["affine_trans"][0, 0].numpy().astype(np.float64), [0, 0, 1]])
                uv = o["dense_uv"][0, :70].float().cpu().numpy() * 512
                full = (np.linalg.inv(A) @ np.concatenate([uv, np.ones((70, 1))], 1).T).T[:, :2]
                rows["dense"].append((a, full))
                for r in windows:  # the same heatmaps, local soft-argmax decoding
                    from fab.dense_head import local_soft_argmax

                    lg = getattr(model.dense_head, "last_logits_hr", None)
                    lg, rr = (model.dense_head.last_logits[:, :70], r) if lg is None else (lg[:, :70], 2 * r)
                    uv = local_soft_argmax(lg.float(), rr)[0].cpu().numpy() * 512
                    full = (np.linalg.inv(A) @ np.concatenate([uv, np.ones((70, 1))], 1).T).T[:, :2]
                    rows.setdefault(f"dense_lw{r}", []).append((a, full))
        else:
            o = est.process_one_image(img, bboxes=box[None], inference_type="body")[0]
            kp = np.asarray(o["pred_keypoints_2d"])[:, :2]
        rows["projected"].append((a, kp))
        if k % 500 == 0:
            print(f"[{k}/{len(anns)}]", flush=True)
    res = {name: [{"image_id": a["image_id"], "category_id": 1, "score": 1.0,
                   "keypoints": np.concatenate([kp[MHR_TO_COCO], np.full((17, 1), 2.0)], 1).ravel().tolist()}
                  for a, kp in rr] for name, rr in rows.items() if rr}
    out.mkdir(parents=True, exist_ok=True)
    (out / f"shard{args.shard}.json").write_text(json.dumps(res))
    print("done", {k: len(v) for k, v in res.items()}, flush=True)


if __name__ == "__main__":
    main()
