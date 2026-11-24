"""Adapter: the FAH pipeline (detector, pose network, mesh fit, video refiner) -> benchmark prediction format."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fah import HandEstimator, load_hand_model  # noqa: E402
from fah.detector import build_hand_detector  # noqa: E402
from fah.frame_cache import frame_cache  # noqa: E402
from fah.utils import pick_device  # noqa: E402

VERTEX_REGRESSOR = os.path.join(os.path.dirname(os.path.abspath(sys.modules["fah"].__file__)), "assets", "mhr_to_mano778_regressor.npz")


def box_ioa(a, b):
    """Fraction of box a inside box b."""
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    return max(x2 - x1, 0) * max(y2 - y1, 0) / max((a[2] - a[0]) * (a[3] - a[1]), 1e-9)


def split_shared_masks(dm, dj, hands, faces, hw):
    """Hands that share one detector box (pair_wrist's two side runs of a box over crossed / praying hands): detector.masks
    computes identical logits for them, so the overlap rule (ties to the lower index) gave the first hand the whole box mask
    and the others an empty one. Splits each shared box's mask by the hands' mesh silhouettes (fah.vis.render_instance_masks,
    mutual occlusion resolved); mask pixels no mesh covers stay with the first hand. dm: per-hand masks (None: no box mask),
    dj: per-hand detector box index (None: none), faces: of the hands' vertices_mano; in place."""
    from fah.vis import render_instance_masks

    groups = {}
    for k, i in enumerate(dj):
        if i is not None and dm[k] is not None:
            groups.setdefault(i, []).append(k)
    for ks in (g for g in groups.values() if len(g) > 1):
        full = np.any([dm[k] for k in ks], 0)
        meshes = [(hands[k]["vertices_mano"], faces) if "vertices_mano" in hands[k] else
                  (hands[k]["keypoints_3d"], np.zeros((0, 3), int)) for k in ks]
        ids = render_instance_masks(meshes, hands[ks[0]]["cam_int"], hw)
        for a, k in enumerate(ks):
            dm[k] = full & (ids == a + 1)
        dm[ks[0]] |= full & (ids == 0)  # mask pixels no mesh covers stay with the first hand
    return dm


class FAHMethod:
    name = "fah"

    def __init__(self, checkpoint="checkpoints/sam-3d-body-dinov3/model.ckpt", device="auto",
                 input_size=None, decoder_depth=None, dtype=None, det_conf=0.3, cuda_graph=False, student=None, fast_det=False, compile=False, finetune_ckpt=None, det_every=1,
                 masks=False, detector_weights="", detector="wilor", refine_box_iters=0, recrop_border=0.0, bbox_padding=0.9, min_crop=0.0, gpu_crop=False, gpu_frame=False, fast_batch=False, det_half=False, det_iou=None, det_trt="", pose_trt="", smooth=False, fps=30.0,
                 beta_trans=20.0, beta_rel=40.0, verify=False, presence_thresh=0.5, side_thresh=0.5, fuse_thresh=0.25, warp_size=None, det_size=None, refine_mesh_iters=0, refiner=None, refiner_blend=0.75, mesh_follow=False, refiner_euro=None, refiner_anchor_2d=False, refiner_keep_2d=False, refiner_euro_2d=None, mesh_fit=0, mesh_fit_warm=0, mesh_fit_wv=0.3, mesh_fit_every=1, mesh_fit_lmf="", mesh_fit_stride=1, mesh_fit_jac="ad", mesh_align_2d=True, mesh_align_tipend=False, mesh_align_surface=False, tip_extend=0.0, mano_head=True, surface_conf=0.0, heatmap_local=0, heatmap_local_gate=0.0, lift_2d=False, side_memory=0.0, surface_smooth=0,
                 det_masks=False, parts_in_mask=True, contain_thresh=1.0, pair_keep=0.0, pair_fuse=None, pair_probe=0.0, pair_wrist=0.0, pair_wrist_side=0.0, mask_split=True, side_track=False, st_margin=1.0, st_n=3, st_lock=1.5, st_lock_hits=3, st_iou=0.3, st_amb=0.1, st_cap=3.0, edge_score=0.0, edge_track=False, tf32=False,
                 mano_vertex_regressor=VERTEX_REGRESSOR, mano_regressor=None, persp_crop=False, base_weights=True,
                 pose_coreml="", pose_coreml_units="CPU_AND_GPU", det_coreml="", det_coreml_units="ALL", mesh_fit_device=None, mesh_fit_centre=False, mesh_fit_fd_eps=1e-3, refiner_device=None,
                 mhr_fit=0, mhr_fit_wv=0.3, mhr_fit_init="", mhr_fit_vreg="", mhr_fit_asset="", mhr_fit_align_2d=True, mhr_fit_args=None,
                 vd_propose=False, vd_max_missing=3, vd_min_hits=2, vd_enlarge=1.1, vd_cover_iou=0.5, vd_cover_score=0.25, vd_prop_score=0.05,
                 vd_presence=None, vd_dedup=0.15, vd_consist=0.3, vd_gate_fill=False, vd_min_inside=0.3,
                 vd_dedup_wrist=0.0, vd_match_wrist=0.0, parts=True, contact_ckpt="", contact_video="", contact_smooth=1, contact_h=None, **_):
        device = pick_device(device)
        self.tf32 = bool(tf32)  # TF32 tensor cores for the fp32 parts of the pose model (point-head einsums / linears), per call
        model, cfg = load_hand_model(checkpoint, device=device, backbone_dtype=dtype, verbose=False, weights=base_weights)
        ft = None
        if finetune_ckpt:  # train/finetune.py checkpoint: decoder ("model.*") and optionally student ("encoder.*")
            ft = torch.load(finetune_ckpt, map_location="cpu", weights_only=False)
            sd = {k[6:]: v for k, v in ft["trainable"].items() if k.startswith("model.")}
            if contact_ckpt:  # hand-object contact head trained on the frozen network (its keypoint_head.contact_* tensors)
                sd.update({k[6:]: v for k, v in torch.load(os.path.expandvars(contact_ckpt), map_location="cpu", weights_only=False)
                           ["trainable"].items() if k.startswith("model.keypoint_head.contact_")})
            if any(k.startswith("slim_decoder.") for k in sd):
                from fah.models.decoders.slim import SlimHandDecoder

                d = sd["slim_decoder.queries.weight"].shape[1]
                nl = len({k.split(".")[3] for k in sd if k.startswith("slim_decoder.decoder.layers.")})
                kq = sd["slim_decoder.kp.weight"].shape[0] if "slim_decoder.kp.weight" in sd else 0
                model.slim_decoder = SlimHandDecoder(dim=d, layers=nl, heads=d // 64, kp_queries=kq).to(model.device)
            if any(k.startswith("mask_head.") for k in sd):
                from fah.models.heads.mask_head import HandMaskHead

                model.mask_head = HandMaskHead(img_dim=sd["mask_head.img_proj.0.weight"].shape[1]).to(model.device)
            if any(k.startswith("keypoint_head.") for k in sd):
                from fah.models.heads.keypoint_head import HandKeypointHead

                kp_only = "keypoint_head.pool.0.weight" in sd  # decoder-free keypoint model (finetune --kp_only)
                dense = sd["keypoint_head.dense"].tolist() if "keypoint_head.dense" in sd else (0, 0)
                model.keypoint_head = HandKeypointHead(img_dim=sd["keypoint_head.img_proj.0.weight"].shape[1],
                                                       surface_idx=sd["keypoint_head.surface_idx"].tolist(), pool=kp_only,
                                                       dense=dense, dim=sd["keypoint_head.point"].shape[1],
                                                       stride=4 if "keypoint_head.up.3.weight" in sd else 8,
            depth="keypoint_head.depth.1.weight" in sd, vis="keypoint_head.vis.1.weight" in sd,
            dense_vidx=sd["keypoint_head.dense_vidx"].tolist() if "keypoint_head.dense_vidx" in sd else None,
            parts=sd["keypoint_head.part_emb"].shape[0] if "keypoint_head.part_emb" in sd else 0,
            part_hr="keypoint_head.part_refine.0.weight" in sd,
            part_depth=sum(1 for k in sd if k.startswith("keypoint_head.part_conv.") and k.endswith(".weight") and sd[k].dim() == 4) or 2,
            contact=len({k.split(".")[2] for k in sd if k.startswith("keypoint_head.contact_mix.")}),
            contact_dists=sd["keypoint_head.contact_dists"].tolist() if "keypoint_head.contact_dists" in sd else (10.0,),
            contact_dim=sd["keypoint_head.contact_pos"].shape[1] if "keypoint_head.contact_pos" in sd else 128,
            contact_tokmem="keypoint_head.contact_tokmem.1.weight" in sd,
            contact_geo="keypoint_head.contact_in.1.weight" in sd or "keypoint_head.contact_mix.0.norm1.weight" not in sd).to(model.device)
                model.kp_only = kp_only
                if any(k.startswith("mano_head.") for k in sd):  # parametric MANO mesh head (MANO from MANO_DIR)
                    from fah.models.heads.mano_head import ManoParamHead

                    pd = sd["mano_head.feat.0.weight"].shape[0]
                    model.mano_head = ManoParamHead(tok_dim=sd["mano_head.mlp.0.weight"].shape[0] - 61 * 34, pt_dim=pd,
                                                    mano_dir=os.environ.get("MANO_DIR", "data/mano")).to(model.device)
            if any(k.startswith("verify_head.") for k in sd):
                from fah.models.heads.mask_head import HandVerifyHead

                model.verify_head = HandVerifyHead().to(model.device)
            pre = "keypoint_head.contact_branch."  # trainable contact branch: copies of the encoder's last blocks (after the encoder)
            sd_br = {k[len(pre):]: sd.pop(k) for k in list(sd) if k.startswith(pre)}
            missing, unexpected = model.load_state_dict(sd, strict=False)
            kpo = getattr(model, "kp_only", False)  # decoder-free: the (unused) decoder / MHR heads are not saved
            missing = [k for k in missing if not (k.startswith("backbone.") or "mhr" in k or k in ("keypoint_head.dense", "keypoint_head.contact_bias"))
                       and not (kpo and not k.startswith(("keypoint_head.", "ray_cond_emb_hand.", "verify_head.")))]
            assert not unexpected and not missing, (missing[:5], unexpected[:5])
            if not mano_head and hasattr(model, "mano_head"):  # optionally ignore a learned MANO parameter head (mesh from the dense points)
                del model.mano_head
        if ft is not None and not decoder_depth:  # early-exit decoder trained with train/finetune.py --decoder_depth
            decoder_depth = ft.get("args", {}).get("decoder_depth") or None
        self.detector = build_hand_detector(detector, device=device, conf=det_conf, fast=fast_det, weights=detector_weights,
                                            **({"imgsz": det_size} if det_size else {}), **({"half": True} if det_half else {}),
                                            **({"iou": det_iou} if det_iou else {}), **({"trt": os.path.expandvars(det_trt)} if det_trt else {}),
                                            **({"coreml": os.path.expandvars(det_coreml), "coreml_units": det_coreml_units} if det_coreml else {}))
        enc = None
        if student or (ft is not None and ft.get("student_name")):  # train/distill.py (or finetune) student
            from fah.student import StudentEncoder

            if ft is not None and ft.get("student_name"):
                enc = StudentEncoder(ft["student_name"], pretrained=False, size=ft["args"].get("student_size") or None)
                enc.load_state_dict({k[8:]: v for k, v in ft["trainable"].items() if k.startswith("encoder.")})
            else:
                ck = torch.load(student, map_location="cpu", weights_only=False)
                enc = StudentEncoder(ck["student_name"], pretrained=False)
                enc.load_state_dict(ck["student"])
            enc = enc.to(device).eval()
            if ft is not None and sd_br:
                from fah.models.heads.keypoint_head import attach_contact_branch

                attach_contact_branch(enc, model.keypoint_head, len({k.split(".")[1] for k in sd_br if k.startswith("blocks.")}))
                model.keypoint_head.contact_branch.load_state_dict(sd_br)
            if dtype:
                enc = enc.to(getattr(torch, dtype))
        if gpu_frame:  # the frame goes to the device once: detector letterbox and pose crops sampled there
            gpu_crop = True
            for d in (self.detector, getattr(self.detector, "det", None)):
                if d is not None and hasattr(d, "imgsz"):
                    d.gpu_letterbox = True
        self.est = HandEstimator(model, cfg, detector=self.detector, input_size=input_size, refine_box_iters=refine_box_iters, recrop_border=recrop_border, bbox_padding=bbox_padding, min_crop=min_crop, gpu_crop=gpu_crop, fast_batch=fast_batch, student_trt=os.path.expandvars(pose_trt) if pose_trt else "",
                                 student_coreml=os.path.expandvars(pose_coreml) if pose_coreml else "", student_coreml_units=pose_coreml_units,
                                 decoder_depth=decoder_depth, student=enc, cuda_graph=cuda_graph,
                                 compile=compile, warp_size=warp_size, refine_mesh_iters=refine_mesh_iters,
                                 surface_conf=surface_conf, heatmap_local=heatmap_local, heatmap_local_gate=heatmap_local_gate, lift_2d=lift_2d, surface_smooth=surface_smooth,
                                 **({"persp_crop": True} if persp_crop else {}),  # virtual camera rotated onto each box-centre ray
                                 **({"mano_regressor_path": mano_regressor} if mano_regressor else {}))  # MHR -> MANO21 joints
        self.est.parts = parts  # False: no part maps (the other outputs unchanged)
        kh = getattr(model, "keypoint_head", None)
        if getattr(kh, "contact_branch", None) is not None and self.est.student_trt is not None and \
                self.est.student_trt.hidden_block != kh.contact_branch_block:  # the engine must give the branch's input
            import warnings

            warnings.warn(f"{pose_trt} has no hidden{kh.contact_branch_block} output for the contact head's branch (build the engine "
                          "with fah-build-trt for this contact.pt): the PyTorch encoder runs instead")
            self.est.student_trt = None
        self.vreg = None
        if mano_vertex_regressor and os.path.exists(mano_vertex_regressor):
            from fah.estimator import load_joint_regressor

            num_verts = model.head_pose_hand.mhr.character_torch.mesh.rest_vertices.shape[0]
            W = load_joint_regressor(mano_vertex_regressor, num_verts)
            cols = np.nonzero(W.any(0))[0]
            remap = -np.ones(W.shape[1], dtype=np.int64)
            remap[self.est.hand_vert_idx] = np.arange(len(self.est.hand_vert_idx))
            assert (remap[cols] >= 0).all()
            self.vreg = (W[:, cols], remap[cols])
        self.name = "fah" + (f"-d{decoder_depth}" if decoder_depth else "")
        self.schedule, self.prev = None, []
        self.smooth, self.fps, self.t = smooth, fps, 0  # video: One-Euro smoothing via fah.video.HandTracker
        self.tracker_args = dict(beta_trans=beta_trans, beta_rel=beta_rel)
        self.tracker = None
        self.refiner, self.refiner_blend = refiner, refiner_blend  # video: learned online refiner (fah.video.LearnedSmoother)
        self.mesh_follow = mesh_follow  # the mesh follows the refined articulation (not only the root shift)
        self.refiner_euro = refiner_euro  # [min_cutoff, beta]: One Euro filter after the refiner (fah.video.LearnedSmoother)
        self.refiner_anchor_2d = refiner_anchor_2d  # video 2D = per-frame 2D + the refiner's change of the projection
        self.refiner_keep_2d = refiner_keep_2d  # video 2D = per-frame 2D (refiner changes 3D / mesh only)
        self.refiner_euro_2d = refiner_euro_2d  # [min_cutoff, beta per px/s]: One Euro filter on the exported 2D joints
        self.refiner_device = refiner_device  # device of the refiner (default: the model's; "cpu" on Apple silicon: 0.8 vs 2.4 ms on MPS)
        self.mesh_fit = mesh_fit  # > 0: Gauss-Newton iterations of a MANO fit to the dense surface (mesh output)
        self.mesh_align_2d = mesh_align_2d  # fitted mesh placed rigidly onto the 2D joints (PnP)
        self.mesh_align_tipend = mesh_align_tipend  # PnP tips = the dense fingertip-end points' 2D (train --tipend_w)
        self.tip_extend = tip_extend  # > 0: fingertips (2D + the mesh's distal phalanges) out to the instance mask's end, at most x the phalanx (fah.tips)
        self.mesh_align_surface = mesh_align_surface  # PnP also onto the dense surface points' 2D (mask-aware head output); a number k > 1 = joints counted k times
        self.mesh_fit_warm = mesh_fit_warm  # video: iterations when starting from the previous frame's fit (0: always cold)
        self.mesh_fit_wv = mesh_fit_wv  # weight of the dense surface vs the joints in the fit (EgoStandard best at 0.3)
        self.mesh_fit_every = mesh_fit_every  # video: refit every k-th frame; in between the last fit follows the joints
        self.mesh_fit_lmf = os.path.expandvars(mesh_fit_lmf) if mesh_fit_lmf else ""  # learned fitter checkpoint: init of the fit
        self.mesh_fit_stride = mesh_fit_stride  # > 1: Gauss-Newton residuals on every stride-th surface vertex
        self.mesh_fit_jac = mesh_fit_jac  # Gauss-Newton Jacobian: ad (vmap(jacfwd)) or fd (central differences, compiled)
        self.mesh_fit_device = mesh_fit_device  # device of the mesh stage (default: the model's; "cpu" on Apple silicon: MPS is launch-bound)
        # mesh stage Gauss-Newton residuals relative to the target wrist + finite-difference step (default off / 1e-3; True + 1e-2:
        # 1e-6 input changes move far hands' meshes 0.01 instead of 0.7 mm, same accuracy, fah.fast_mano.MeshStage)
        self.mesh_fit_centre, self.mesh_fit_fd_eps = mesh_fit_centre, mesh_fit_fd_eps
        # > 0: Levenberg-Marquardt iterations of an MHR-hand fit (SAM 3D Body's rig, fah.mhr_fit) to the dense surface + joints,
        # with or without the MANO fit: verts3d_mhr / verts3d_mhr_mano / joints3d_mhr_fit / joints3d_mhr_rig / mhr_params (the MANO
        # outputs unchanged); mhr_fit_init: learned initialiser (train/mhr_fitter.py); mhr_fit_vreg: MHR -> MANO-778 vertex regressor
        self.mhr_fit, self.mhr_fit_wv, self.mhr_fit_args = mhr_fit, mhr_fit_wv, dict(mhr_fit_args or {})
        self.mhr_fit_init = os.path.expandvars(mhr_fit_init) if mhr_fit_init else ""
        self.mhr_fit_vreg = os.path.expandvars(mhr_fit_vreg) if mhr_fit_vreg else ""
        self.mhr_fit_asset = os.path.expandvars(mhr_fit_asset) if mhr_fit_asset else ""  # cached fah.mhr_fit.build_asset (default: from the model)
        self.mhr_fit_align_2d = mhr_fit_align_2d  # fitted MHR hand placed rigidly onto the 2D joints (PnP), as mesh_align_2d
        self.refiner_graph = cuda_graph  # its full-window forward as a CUDA graph too
        # pose-model verification (needs a finetune checkpoint with verify_head): the detector runs at low
        # confidence, boxes are re-scored by presence, handedness flipped when the side score is low
        self.verify, self.presence_thresh, self.side_thresh, self.fuse_thresh = verify, presence_thresh, side_thresh, fuse_thresh
        self.contain_thresh = contain_thresh  # also drop a hand whose box lies more than this inside a better one's
        self.pair_keep = pair_keep  # > 0: keep an opposite-side hand overlapping a kept one if their 2D joints differ by > this x box diagonal
        self.pair_fuse = pair_fuse  # fused-score threshold for such partners (default: fuse_thresh)
        self.pair_probe = pair_probe  # > 0: probe unpaired hands' boxes (x this) with the other side for a hidden partner
        # > 0: identity NMS instead of the box NMS + pair rule: every (box, side) run of the verification (incl. the other side of
        # doubtful boxes) ranked by score x side score; a run whose wrist lies within this x palm length (wrist - middle MCP) of
        # a kept hand's wrist is that hand (side-conditioned runs of a box over two hands find different wrists)
        self.pair_wrist = pair_wrist
        self.pair_wrist_side = pair_wrist_side  # > 0: runs of opposite sides at one wrist stay two hands when both side scores >= this
        self.mask_split = mask_split  # hands sharing one detector box split its instance mask by their mesh silhouettes
        # video: > 0 -> a box on a tracked hand keeps its handedness; the side score's log-odds evidence accumulates
        # with this decay per frame and the side flips only when the evidence for the other side outweighs it
        self.side_memory = side_memory
        # side_track (video, needs side_memory; opt-in): causal per-track handedness. Boxes join the previous frame's hands on
        # boxes moved by their video track's constant velocity (IoU > st_iou with a clear best by st_amb; overlapping hands keep
        # their memory); a locked track (st_lock_hits frames, belief >= st_lock) changes side only after its belief stays below
        # -st_margin for st_n frames in a row (tentative tracks follow side_memory's rule); a box's two side runs rank by the
        # track belief in the pair_wrist selection, and a locked box's other-side run is no second hand. Belief capped at st_cap.
        self.side_track, self.st_margin, self.st_n, self.st_lock, self.st_lock_hits = side_track, st_margin, st_n, st_lock, st_lock_hits
        self.st_iou, self.st_amb, self.st_cap = st_iou, st_amb, st_cap
        # edge_score (opt-in): > 0 -> detector boxes touching the image border need score >= this; edge_track: video only, and
        # a border box that continues a live video-detector track is kept (only new tracks born at the border need the score)
        self.edge_score, self.edge_track = edge_score, edge_track
        self.masks = masks  # instance masks from the rendered predicted meshes (mutual hand occlusion only)
        self.det_masks = det_masks  # instance masks of a YOLO-seg detector (fah.detector masks()), det_every 1
        self.parts_in_mask = parts_in_mask  # part map = the detector's instance mask x the part head's best hand part per pixel
        # video detector (video mode only, fah.video.TrackProposals): vd_propose -> a box for each confirmed track (>= vd_min_hits
        # matched frames, unmatched for <= vd_max_missing) without a detector box (IoU >= vd_cover_iou, score >= vd_cover_score) at its
        # predicted position, run with the detector boxes and kept if its presence >= vd_presence (default presence_thresh), detector
        # score vd_prop_score
        self.vd_propose, self.vd_prop_score, self.vd_presence = vd_propose, vd_prop_score, vd_presence
        self.vd_dedup = vd_dedup  # a kept proposal whose 2D joints lie on a detected hand's (mean distance < this x its box diagonal) is dropped
        # identity rules (pair_wrist's wrist distance in palm lengths; opt-in): > 0 -> the proposal dedup tests the wrists instead
        # (a kept proposal within vd_dedup_wrist palms of a detected hand's wrist is dropped), and hands join tracks by wrist
        # distance to the track's predicted wrist (< vd_match_wrist palms, nearest first) instead of box IoU
        self.vd_dedup_wrist = vd_dedup_wrist
        self.vd_consist = vd_consist  # a proposal is kept only if its 2D joints stay within this x the box diagonal of its track's predicted joints
        self.vd_gate_fill = vd_gate_fill  # no refiner gap fill for a track whose proposal found no hand (presence below its threshold) or that left the image
        self.vd_args = dict(max_missing=vd_max_missing, min_hits=vd_min_hits, enlarge=vd_enlarge, cover_iou=vd_cover_iou, cover_score=vd_cover_score,
                            min_inside=vd_min_inside, match_wrist=vd_match_wrist)  # leaving: predicted box < min_inside in the image
        self._vd = self.vd_last = None
        # video contact (fah.contact_video): online over a clip's frames (reset() per clip), fed the per-frame hands from
        # before the video refiner (what it was trained on); its outputs are added to the same hands
        self.contact_video, self._vc = os.path.expandvars(contact_video) if contact_video else "", None
        self.contact_smooth, self.contact_h = contact_smooth, contact_h  # flag rule (fah.contact_video.VideoContact smooth / h)
        if det_every > 1:
            from fah.video import DetectionSchedule

            self.schedule = DetectionSchedule(self.detector, det_every)

    def reset(self):  # new clip
        self._fit_prev = {}
        if getattr(self, "_vc", None) is not None:
            self._vc.reset()
        self.prev = []
        self.t = 0
        if self.refiner:
            from fah.video import LearnedSmoother

            if self.tracker is None:
                self.tracker = LearnedSmoother(self.refiner, self.refiner_blend, device=getattr(self, "refiner_device", None) or self.est.model.device,
                                               cuda_graph=self.refiner_graph, mesh_follow=self.mesh_follow,
                                               euro=self.refiner_euro, anchor_2d=self.refiner_anchor_2d,
                                               keep_2d=self.refiner_keep_2d, euro_2d=self.refiner_euro_2d)
            self.tracker.reset()
        elif self.smooth:
            from fah.video import HandTracker

            self.tracker = HandTracker(**self.tracker_args)
        if self.schedule is not None:
            self.schedule.reset()
        self._vd = None
        if self.tracker is not None and getattr(self, "vd_propose", False):
            from fah.video import TrackProposals

            self._vd = TrackProposals(**self.vd_args)

    @frame_cache()
    def _verified(self, image_bgr, K):
        from bench.common import box_iou

        # video with det_every > 1: decoder-predicted boxes of the previous frame between detector frames
        boxes, scores, is_right = self.schedule(image_bgr, self.prev) if self.schedule is not None else self.detector(image_bgr)
        vd = getattr(self, "_vd", None) if self.tracker is not None else None  # video detector (video mode only)
        n_det = len(boxes)
        if vd is not None:  # track-propagated proposals after the detector boxes, one pose call
            pb, ps, pid = vd.proposals(boxes, scores, image_bgr.shape[:2])
            expect = [vd.expected(k) for k in pid]  # the track's last 2D joints moved by its velocity
            self.vd_last, self._vd_empty = dict(props=len(pb)), set(vd.leaving)
            if len(pb):
                boxes = np.concatenate([np.asarray(boxes, np.float32).reshape(-1, 4), pb])
                scores = np.concatenate([np.asarray(scores, np.float32), np.full(len(pb), self.vd_prop_score, np.float32)])
                is_right = np.concatenate([np.asarray(is_right, bool), ps])
        if len(boxes) == 0:
            return []
        lo = np.zeros(len(boxes))
        memory = self.side_memory and self.tracker is not None  # video mode only (images would leak across frames)
        st = memory and getattr(self, "side_track", False)
        kc, lock, age = np.zeros(len(boxes), int), np.zeros(len(boxes), bool), np.zeros(len(boxes), int)
        if st and getattr(self, "prev", None):
            is_right = np.array(is_right, dtype=bool)
            vel = {tr["id"]: tr["vel"] for tr in vd.tracks} if vd is not None else {}
            pbx = []
            for q in self.prev:  # the previous frame's boxes moved by their track's constant velocity
                b, v = np.asarray(q["bbox"], np.float64).copy(), vel.get(q.get("vd_track"))
                if v is not None:
                    b[[0, 2]] += v[0]
                    b[[1, 3]] += v[1]
                pbx.append(b)
            iou0 = np.array([[box_iou(b, q) for q in pbx] for b in boxes[:n_det]]).reshape(n_det, len(pbx))
            iou = iou0.copy()
            second = lambda x, k: np.sort(np.delete(x, k))[-1] if len(x) > 1 else 0.0
            while iou.size and iou.max() > self.st_iou:  # unique matches, a clear best both ways
                i, j = np.unravel_index(iou.argmax(), iou.shape)
                if iou0[i, j] - max(second(iou0[i], j), second(iou0[:, j], i)) >= self.st_amb:
                    q = self.prev[j]
                    is_right[i], lo[i] = q["is_right"], q.get("side_lo", 0.0)
                    kc[i], lock[i], age[i] = q.get("side_k", 0), q.get("side_lock", False), q.get("side_age", 0)
                iou[i, :], iou[:, j] = 0, 0
            if vd is not None:  # proposals: their track's state
                by_track = {q.get("vd_track"): q for q in self.prev}
                for i in range(n_det, len(boxes)):
                    q = by_track.get(pid[i - n_det])
                    if q is not None and bool(q["is_right"]) == bool(is_right[i]):
                        lo[i], kc[i], lock[i], age[i] = q.get("side_lo", 0.0), q.get("side_k", 0), q.get("side_lock", False), q.get("side_age", 0)
        elif memory and getattr(self, "prev", None):
            is_right = np.array(is_right, dtype=bool)
            iou = np.array([[box_iou(b, q["bbox"]) for q in self.prev] for b in boxes[:n_det]])  # proposals keep their track's side
            sep = [all(box_iou(b, c) < 0.1 for k, c in enumerate(boxes[:n_det]) if k != i) for i, b in enumerate(boxes[:n_det])]
            while iou.size and iou.max() > 0.5:  # unique matches; interacting (overlapping) hands keep no memory
                i, j = np.unravel_index(iou.argmax(), iou.shape)
                if sep[i]:
                    is_right[i], lo[i] = self.prev[j]["is_right"], self.prev[j].get("side_lo", 0.0)
                iou[i, :], iou[:, j] = 0, 0
        hands = self.est.process_image(image_bgr, boxes=boxes, is_right=is_right, scores=scores, cam_int=K)
        logit = lambda h: float(np.log(np.clip(h.get("side_score", 0.5), 1e-4, 1 - 1e-4) / np.clip(1 - h.get("side_score", 0.5), 1e-4, 1)))
        low = [i for i, h in enumerate(hands) if h.get("side_score", 1.0) < self.side_thresh]
        other = []  # pair_wrist: the losing run of each doubtful box stays a candidate (the box's other hand)
        if low:  # try the other handedness for doubtful boxes, keep the more consistent hypothesis
            alt = self.est.process_image(image_bgr, boxes=boxes[low], is_right=~is_right[low], scores=scores[low], cam_int=K)
            first = [hands[i] for i in low]
            for i, h in zip(low, alt):
                if memory:
                    lo[i] = self.side_memory * lo[i] + float(np.clip(logit(hands[i]) - logit(h), -4, 4))
                    if st and lock[i]:  # hysteresis: the opposite side must win st_n frames in a row by st_margin
                        kc[i] = kc[i] + 1 if lo[i] < -self.st_margin else 0
                        if kc[i] >= self.st_n:
                            hands[i], lo[i], kc[i] = h, -lo[i], 0
                    elif lo[i] < 0:
                        hands[i], lo[i] = h, -lo[i]
                elif h.get("side_score", 0) > hands[i].get("side_score", 0):
                    hands[i] = h
            if self.pair_wrist > 0:
                other = [(i, a if hands[i] is f else f) for i, f, a in zip(low, first, alt)]
                if st:  # a locked track's box: its other-side run is that hand's wrong side, no second hand; the box's runs rank by
                    other = [(i, o) for i, o in other if not lock[i]]  # the track belief (held: the track's run first)
                    for i, o in other:
                        b = abs(lo[i])
                        hands[i]["_rank_side"], o["_rank_side"] = float(1 / (1 + np.exp(-b))), float(1 / (1 + np.exp(b)))
        if memory:
            for i, h in enumerate(hands):
                if i not in low:
                    lo[i] = self.side_memory * lo[i] + float(np.clip(logit(h), 0, 4))
                    kc[i] = 0
                h["side_lo"] = float(min(lo[i], self.st_cap if st else 3.0))
                if st:
                    h["side_k"], h["side_age"] = int(kc[i]), int(age[i]) + 1
                    h["side_lock"] = bool(lock[i]) or (age[i] + 1 >= self.st_lock_hits and lo[i] >= self.st_lock)
        for h in hands + [o for _, o in other]:
            h["det_score"] = h["score"]
            h["score"] = float(np.sqrt(h["det_score"] * h.get("presence", 1.0)))
        pres_thr = lambda h: self.presence_thresh
        fuse_thr = lambda h: self.fuse_thresh
        if vd is not None:  # proposals: their own thresholds
            def consistent(h, e):  # a proposal's hand must continue its track's pose (not a forearm / background / other hand)
                if e is None or not self.vd_consist:
                    return True
                d = np.linalg.norm(np.asarray(h["keypoints_2d"], np.float64) - e, axis=-1).mean()
                return d < self.vd_consist * np.hypot(*(e.max(0) - e.min(0)))

            for i, h in enumerate(hands):
                h["vd_prop"] = i >= n_det
                h["vd_ok"] = i < n_det or consistent(h, expect[i - n_det])
            for i, o in other:
                o["vd_prop"] = hands[i]["vd_prop"]
                o["vd_ok"] = i < n_det or consistent(o, expect[i - n_det])
            pres_thr = lambda h: (2.0 if not h["vd_ok"] else self.vd_presence if self.vd_presence is not None else self.presence_thresh) \
                if h["vd_prop"] else self.presence_thresh
            # proposals: accepted on presence alone, ranked below every detector box (detector score vd_prop_score)
            fuse_thr = lambda h: 0.0 if h["vd_prop"] else self.fuse_thresh
        lo_thr = min(self.fuse_thresh, self.pair_fuse) if self.pair_keep > 0 and self.pair_fuse is not None else self.fuse_thresh
        if vd is not None:  # sides the video detector looked at and found empty (vd_gate_fill)
            self._vd_empty |= {pid[i - n_det] for i, h in enumerate(hands) if h["vd_prop"] and i - n_det < len(pid) and h["vd_ok"]
                               and h.get("presence", 1.0) < pres_thr(h)}
        hands = [h for h in hands if h.get("presence", 1.0) >= pres_thr(h) and h["score"] >= min(lo_thr, fuse_thr(h))]
        if getattr(self, "edge_score", 0) > 0 and (not self.edge_track or vd is not None):  # truncated hands at the border
            ih, iw = image_bgr.shape[:2]
            edge = lambda b: b[0] <= 2 or b[1] <= 2 or b[2] >= iw - 3 or b[3] >= ih - 3
            live = [b for tr, b in vd.predicted() if tr["hits"] >= 1] if self.edge_track else []
            tracked = lambda b: any(box_iou(b, q) > 0.3 or vd._covers(b, q) for q in live)
            ok = lambda h: h.get("vd_prop") or not edge(h["bbox"]) or h["score"] >= self.edge_score or (self.edge_track and tracked(h["bbox"]))
            hands = [h for h in hands if ok(h)]
            other = [(i, o) for i, o in other if ok(o)]
        hands.sort(key=lambda h: -h["score"])
        if self.pair_wrist > 0:  # identity NMS: one hand per wrist, whatever the boxes' overlap / side
            cand = hands + [o for _, o in other if o.get("presence", 1.0) >= pres_thr(o)]
            cand.sort(key=lambda h: -h["score"] * h.get("_rank_side", h.get("side_score", 1.0)))

            def same(h, k):  # one hand: wrists within pair_wrist x palm, unless two confident opposite sides touch there
                j, q = np.asarray(h["keypoints_2d"], np.float64), np.asarray(k["keypoints_2d"], np.float64)
                if np.linalg.norm(j[0] - q[0]) >= self.pair_wrist * 0.5 * (np.linalg.norm(j[0] - j[9]) + np.linalg.norm(q[0] - q[9])):
                    return False
                return not (self.pair_wrist_side > 0 and bool(h["is_right"]) != bool(k["is_right"])
                            and min(h.get("side_score", 1.0), k.get("side_score", 1.0)) >= self.pair_wrist_side)

            keep = []
            for h in cand:
                if h["score"] >= fuse_thr(h) and not any(same(h, k) for k in keep):
                    keep.append(h)
            return keep
        keep = []  # cross-handedness NMS (YOLO's NMS is per class)
        for h in hands:
            clash = [k for k in keep if box_iou(h["bbox"], k["bbox"]) >= 0.5 or box_ioa(h["bbox"], k["bbox"]) > self.contain_thresh]
            if not clash:
                if h["score"] >= fuse_thr(h):  # below it (pair_fuse): only as the partner of a kept hand
                    keep.append(h)
            elif self.pair_keep > 0 and len(clash) == 1 and bool(clash[0]["is_right"]) != bool(h["is_right"]) and not clash[0].get("paired"):
                # interacting hands (clasped / stacked): the other side kept when its joints lie on another hand
                k = clash[0]
                d = np.linalg.norm(np.asarray(h["keypoints_2d"]) - np.asarray(k["keypoints_2d"]), axis=-1).mean()
                if d > self.pair_keep * np.hypot(k["bbox"][2] - k["bbox"][0], k["bbox"][3] - k["bbox"][1]):
                    h["paired"] = k["paired"] = True
                    keep.append(h)
        if self.pair_probe > 0 and self.pair_keep > 0:  # the other hand of a clasped pair often has no detector box at all:
            single = [k for k in keep if not k.get("paired")]  # probe each unpaired hand's (enlarged) box with the other side
            if single:
                bx = np.stack([np.asarray(k["bbox"], np.float32) for k in single])
                c, half = (bx[:, :2] + bx[:, 2:]) / 2, (bx[:, 2:] - bx[:, :2]) / 2 * self.pair_probe
                probes = self.est.process_image(image_bgr, boxes=np.concatenate([c - half, c + half], 1).astype(np.float32),
                                                is_right=np.array([not bool(k["is_right"]) for k in single]),
                                                scores=np.array([k["det_score"] for k in single], np.float32), cam_int=K)
                for k, h in zip(single, probes):
                    h["det_score"] = h["score"]
                    h["score"] = float(np.sqrt(h["det_score"] * h.get("presence", 1.0)))
                    if h.get("presence", 1.0) < self.presence_thresh or h["score"] < self.fuse_thresh or h.get("side_score", 1.0) < self.side_thresh:
                        continue
                    d = np.linalg.norm(np.asarray(h["keypoints_2d"]) - np.asarray(k["keypoints_2d"]), axis=-1).mean()
                    others = [o for o in keep if o is not k]
                    if d > self.pair_keep * np.hypot(k["bbox"][2] - k["bbox"][0], k["bbox"][3] - k["bbox"][1]) and \
                            box_iou(h["bbox"], k["bbox"]) > 0.1 and all(box_iou(h["bbox"], o["bbox"]) < 0.5 for o in others):
                        h["paired"] = k["paired"] = h["probed"] = True
                        keep.append(h)
        return keep

    def _surface_2d(self, h):
        """Dense surface points at their predicted 2D positions (what the masks supervise), back-projected at the
        depth of the 3D points so that mutual hand occlusion is still ordered."""
        v = h["vertices_mano"]
        if "surface_vidx" in h:  # subset model: the full surface exists only as the reconstructed 3D mesh
            return v
        uv = h["surface_2d_head"][-len(v):]
        K = h["cam_int"]
        z = np.clip(v[:, 2:], 1e-3, None)
        return np.concatenate([(uv[:, :1] - K[0, 2]) / K[0, 0] * z, (uv[:, 1:] - K[1, 2]) / K[1, 1] * z, z], 1)

    def _fit_meshes(self, hands):
        """MANO mesh fitted to each hand's dense surface + joints (fah.fast_mano.MeshFitter, Gauss-Newton from the
        closed-form per-part Procrustes pose; all hands of the frame in one batch). vertices_mano <- fitted MANO mesh,
        the dense surface kept as vertices_dense."""
        sel = [h for h in hands if h.get("vertices_mano") is not None and len(h["vertices_mano"]) == 778 and not h.get("mesh_param")]
        if len({bool(h["is_right"]) for h in sel}) < len(sel):  # two hands claim the same side: no warm start this frame
            self._fit_prev = {}
        if not sel:
            self._fit_prev = {}
            return
        if getattr(self, "_fitter", None) is None:
            from fah.fast_mano import MeshFitter
            from fah.mano_procrustes import ManoProcrustes

            self._ppm = ManoProcrustes()
            self._fitter = MeshFitter(iters=self.mesh_fit, solver="gn", device=self.est.model.device, wv=self.mesh_fit_wv,
                                      centre=getattr(self, "mesh_fit_centre", False), fd_eps=getattr(self, "mesh_fit_fd_eps", 1e-3))
        video = self.tracker is not None and self.mesh_fit_warm > 0
        prev = getattr(self, "_fit_prev", {}) if video else {}
        warm = video and all(bool(h["is_right"]) in prev for h in sel)  # every hand continues a fitted track: warm start
        self._fit_n = getattr(self, "_fit_n", 0) + 1
        if warm and self.mesh_fit_every > 1 and self._fit_n % self.mesh_fit_every:  # between refits: last fit + joint change
            from fah.video import follow_matrix

            A = follow_matrix(778)
            meshes, M = [], []
            for h in sel:
                m = np.array([1.0 if h["is_right"] else -1.0, 1.0, 1.0])
                pm, pj = self._fit_mesh[bool(h["is_right"])]
                j = np.asarray(h.get("keypoints_3d_mano", h["keypoints_3d"]), np.float64) * m
                meshes.append(pm + A @ (j - pj)), M.append(m)
        else:
            meshes, M = self._fit_now(sel, video, prev, warm)
        for h, mesh, m in zip(sel, meshes, M):
            mesh = mesh * m
            if self.mesh_align_2d and h.get("keypoints_2d_mano", h.get("keypoints_2d")) is not None:
                uv, tipv = np.array(h.get("keypoints_2d_mano", h["keypoints_2d"]), np.float64), None
                if self.mesh_align_tipend and h.get("surface_2d_head") is not None:
                    rows, tipv = self._tip_rows()
                    uv[[4, 8, 12, 16, 20]] = np.asarray(h["surface_2d_head"], np.float64)[40 + rows]
                extra = None
                if self.mesh_align_surface and h.get("surface_2d_head") is not None:
                    kept = self._tip_rows() and self._dense_kept
                    s2 = np.asarray(h["surface_2d_head"], np.float64)
                    if len(s2) == 40 + len(kept):
                        extra = (kept, s2[40:])
                mesh = self._align_2d(mesh, uv, h["cam_int"], tipv, extra)
            h["vertices_dense"] = h["vertices_mano"]
            h["vertices_mano"], h["mesh_param"] = mesh.astype(np.float32), True

    def _fit_now(self, sel, video, prev, warm):
        if not warm and not video:  # whole mesh stage on the device in one CUDA graph (fah.fast_mano.MeshStage)
            if getattr(self, "_stage", None) is None:
                from fah.fast_mano import MeshStage

                self._stage = MeshStage(device=self.mesh_fit_device or self.est.model.device, iters=self.mesh_fit, wv=self.mesh_fit_wv, lmf=self.mesh_fit_lmf or None,
                                        stride=self.mesh_fit_stride, jac=self.mesh_fit_jac, centre=getattr(self, "mesh_fit_centre", False),
                                        fd_eps=getattr(self, "mesh_fit_fd_eps", 1e-3))
            M = [np.array([1.0 if h["is_right"] else -1.0, 1.0, 1.0]) for h in sel]
            P = np.stack([np.asarray(h["vertices_mano"], np.float64) * m for h, m in zip(sel, M)])
            J = np.stack([np.asarray(h.get("keypoints_3d_mano", h["keypoints_3d"]), np.float64) * m for h, m in zip(sel, M)])
            return list(self._stage(P, J)), M
        P, J, R0, P0, B0, M = [], [], [], [], [], []
        for h in sel:
            m = np.array([1.0 if h["is_right"] else -1.0, 1.0, 1.0])
            v = np.asarray(h["vertices_mano"], np.float64) * m
            if warm:
                r0, p0, b0 = prev[bool(h["is_right"])]
            else:
                R, _ = self._ppm.rotations(v)
                (r0, p0), b0 = self._ppm.local_pose(R), np.zeros(10)
            P.append(v), J.append(np.asarray(h.get("keypoints_3d_mano", h["keypoints_3d"]), np.float64) * m), R0.append(r0), P0.append(p0)
            B0.append(b0), M.append(m)
        if self.mesh_fit_lmf and not warm:  # learned one-pass fitter (train/mesh_fitter.py) as the start of the GN steps
            import torch

            from fah.fast_mano import rodrigues
            from train.mesh_fitter import Fitter, features

            if getattr(self, "_lmf", None) is None:
                ck = torch.load(self.mesh_fit_lmf, map_location=self.est.model.device, weights_only=False)
                self._lmf = Fitter(ck["d_in"], ck.get("hidden", 1024)).to(self.est.model.device).eval()
                self._lmf.load_state_dict(ck["model"])
            dev = self.est.model.device
            X = torch.as_tensor(np.stack([features(v, j, r0) for v, j, r0 in zip(P, J, R0)]), dtype=torch.float32, device=dev)
            with torch.no_grad():
                pose, betas, dr, _ = self._lmf(X, torch.as_tensor(np.stack(P0), dtype=torch.float32, device=dev))
                Rn = torch.as_tensor(np.stack(R0), dtype=torch.float32, device=dev) @ rodrigues(dr)
            R0, P0, B0 = list(Rn.cpu().numpy()), list(pose.cpu().numpy()), list(betas.cpu().numpy())
        meshes, (Rf, pf, bf) = self._fitter.fit(np.stack(P), np.stack(J), np.stack(R0), np.stack(P0), np.stack(B0),
                                                iters=self.mesh_fit_warm if warm else self.mesh_fit, params=True)
        if video:
            self._fit_prev = {bool(h["is_right"]): (Rf[k], pf[k], bf[k]) for k, h in enumerate(sel)}
            self._fit_mesh = {bool(h["is_right"]): (meshes[k].astype(np.float64), J[k]) for k, h in enumerate(sel)}  # right-hand frame
        return meshes, M

    def _fit_mhr(self, hands):
        """MHR hand (SAM 3D Body's rig, fah.mhr_fit.fit_mhr_hand) fitted to each hand's dense surface + joints, all
        hands of the frame in one batch, placed onto the 2D joints (PnP). Sets vertices_mhr (MHR hand mesh, est.faces /
        faces_left), vertices_mhr_mano (its MANO-topology map), keypoints_3d_mhr_fit (MANO-convention joints),
        keypoints_3d_mhr_rig (the rig's own hand keypoints), mhr_params."""
        sel = [h for h in hands if len(h.get("vertices_dense", h.get("vertices_mano", ()))) == 778]
        if not sel:
            return
        from fah.mhr_fit import MHRHandFitter, build_asset, fit_mhr_hand

        if getattr(self, "_mhr", None) is None:
            if self.mhr_fit_asset:
                a = torch.load(self.mhr_fit_asset, map_location="cpu", weights_only=False)
            else:
                a = build_asset(self.est.model.head_pose_hand, **({"vreg": self.mhr_fit_vreg} if self.mhr_fit_vreg else {}))
            kw = {"iters": self.mhr_fit, "wv": self.mhr_fit_wv, **self.mhr_fit_args}  # e.g. {"iters": 0}: the learned initialiser alone
            self._mhr = MHRHandFitter(a, getattr(self, "mesh_fit_device", None) or self.est.model.device, init_net=self.mhr_fit_init or None, **kw)
        uv = [h.get("keypoints_2d_mano", h.get("keypoints_2d")) for h in sel]
        place = self.mhr_fit_align_2d and all(u is not None for u in uv)
        res = fit_mhr_hand(self._mhr, np.stack([h.get("vertices_dense", h.get("vertices_mano")) for h in sel]),
                           np.stack([h.get("keypoints_3d_mano", h["keypoints_3d"]) for h in sel]), [bool(h["is_right"]) for h in sel],
                           K=sel[0]["cam_int"] if place else None, joints2d=np.stack(uv) if place else None)
        for h, r in zip(sel, res):
            h["vertices_mhr"], h["vertices_mhr_mano"], h["mhr_params"] = r["verts"], r["verts_mano"], r["params"]
            h["keypoints_3d_mhr_fit"], h["keypoints_3d_mhr_rig"] = r["joints"], r["joints_rig"]

    def _extend_tips(self, outs, det_idx):
        """Mask-guided fingertips (fah.tips.extend_tips) on the exported hands (outs, their detections det_idx): 2D tips
        moved out to the instance mask's end, the mask read only at the ray pixels (detector.mask_at, one transfer for all
        hands); the mesh's distal phalanges stretched along their 3D axis so that the rendered fingertip follows (3D joints
        keep the benchmark's MANO tip convention)."""
        from fah.tips import tip_rays, tips_from_hits

        cell = max(self.detector.last["hw"]) / ((getattr(self, "det_size", None) or 640) / 4)  # YOLO-seg prototype cell in image px
        R = [tip_rays(np.asarray(o["joints2d"], np.float64), max_ext=float(self.tip_extend), mask_px=cell) for o in outs]
        hits = self.detector.mask_at(det_idx, [(xs, ys) for _, xs, ys in R])
        for o, (rays, _, _), hit in zip(outs, R, hits):
            new, ext = tips_from_hits(o["joints2d"], rays, hit, mask_px=cell)
            o["joints2d"], o["tip_ext"] = new.astype(np.float32), ext.astype(np.float32)
            self._stretch_distal(o, ext)

    def _stretch_distal(self, o, ext):
        v = o.get("verts3d")
        if v is None or len(v) != 778 or not ext.any():
            return
        if getattr(self, "_distal", None) is None:  # MANO vertices skinned mostly to each finger's distal joint (thumb..pinky)
            import pickle

            mm = pickle.load(open(os.path.join(os.environ.get("MANO_DIR", "data/mano"), "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
            wj = np.asarray(mm["weights"]).argmax(1)  # MANO joint order: index 1-3, middle 4-6, pinky 7-9, ring 10-12, thumb 13-15
            self._distal = [np.where(wj == j)[0] for j in (15, 3, 6, 12, 9)]
            jr = mm["J_regressor"]
            self._jreg = np.asarray(jr.todense() if hasattr(jr, "todense") else jr, np.float64)
        v = np.asarray(v, np.float64).copy()
        J = self._jreg @ v
        for f, (jd, tv) in enumerate(zip((15, 3, 6, 12, 9), (744, 320, 443, 554, 671))):
            if ext[f] <= 0:
                continue
            a, b = J[jd], v[tv]
            L = np.linalg.norm(b - a)
            if L < 1e-4:
                continue
            u = (b - a) / L
            idx = self._distal[f]
            s = np.clip((v[idx] - a) @ u / L, 0, None)  # 0 at the DIP joint, 1 at the tip vertex
            v[idx] += (ext[f] * L) * s[:, None] * u
        o["verts3d"] = v.astype(np.float32)

    def _tip_rows(self):
        """Fingertip-end points of a dense model: rows (in its dense subset) and MANO vertices of the kept surface points
        nearest the MANO tip vertices, thumb..pinky (as train/finetune.py --tipend_w)."""
        if getattr(self, "_tiprows", None) is None:
            from bench.common import ManoLayer

            kept = getattr(self.est.model.keypoint_head, "dense_vidx", None)
            kept = np.arange(778) if kept is None else np.asarray(kept.cpu() if torch.is_tensor(kept) else kept)
            _, rv = ManoLayer(os.environ.get("MANO_DIR", "data/mano"))(True, np.zeros(10), np.zeros(45), np.zeros(3), np.zeros(3))
            rows = np.array([int(np.argmin(np.linalg.norm(rv[kept] - rv[v], axis=1))) for v in (744, 320, 443, 554, 671)])
            self._tiprows, self._dense_kept = (rows, kept[rows]), kept
        return self._tiprows

    def _align_2d(self, mesh, uv, K, tip_verts=None, extra=None):
        """Rigid placement of a MANO mesh so that its joints project onto the 2D joints (PnP from the 3D placement):
        overlays line up with the image; shape / articulation (all Procrustes-aligned metrics) unchanged."""
        import cv2

        if getattr(self, "_jreg", None) is None:
            import pickle

            mm = pickle.load(open(os.path.join(os.environ.get("MANO_DIR", "data/mano"), "MANO_RIGHT.pkl"), "rb"), encoding="latin1")
            jr = mm["J_regressor"]
            self._jreg = np.asarray(jr.todense() if hasattr(jr, "todense") else jr, np.float64)
        tv = [744, 320, 443, 554, 671] if tip_verts is None else list(tip_verts)
        J = np.concatenate([self._jreg @ mesh, mesh[tv]])[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
        c = J[0].copy()
        if extra is not None:  # + the dense surface points (mesh vertices <-> their direct 2D); joints repeated k times = their weight
            k = max(int(round(float(self.mesh_align_surface))), 1)
            J, uv = np.concatenate([np.tile(J, (k, 1)), mesh[extra[0]]]), np.concatenate([np.tile(uv, (k, 1)), extra[1]])
        ok, rv, tv = cv2.solvePnP(J - c, np.ascontiguousarray(uv), np.asarray(K, np.float64), None, rvec=np.zeros(3), tvec=c.copy(),
                                  useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok or not np.isfinite(tv).all() or tv[2, 0] <= 0.05:
            return mesh
        return (mesh - c) @ cv2.Rodrigues(rv)[0].T + tv.reshape(3)

    def _mano_faces(self):
        if getattr(self, "mano_faces", None) is None:
            from bench.common import ManoLayer

            self.mano_faces = ManoLayer(os.environ.get("MANO_DIR", "data/mano")).faces[True]
        return self.mano_faces

    def __call__(self, image_bgr, K):
        prev = torch.backends.cuda.matmul.allow_tf32  # scoped: other models in the process keep their own setting
        torch.backends.cuda.matmul.allow_tf32 = getattr(self, "tf32", False)
        try:
            return self._call(image_bgr, K)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev

    @frame_cache()
    def _call(self, image_bgr, K):
        if self.verify:
            hands = self._verified(image_bgr, K)
            if getattr(self, "_vd", None) is not None and self.tracker is not None:  # video detector tracks: this frame's kept hands
                if getattr(self, "vd_dedup_wrist", 0) > 0:  # a proposal whose wrist lies on a detected hand's (pair_wrist's identity rule)
                    from fah.video import wrist_dist

                    real = [k["keypoints_2d"] for k in hands if not k.get("vd_prop")]
                    hands = [h for h in hands if not h.get("vd_prop") or all(wrist_dist(h["keypoints_2d"], q) >= self.vd_dedup_wrist for q in real)]
                elif self.vd_dedup > 0:  # a proposal whose joints lie on a detected hand's: that hand (the pose run on it found it again)
                    real = [k for k in hands if not k.get("vd_prop")]
                    same = lambda h, k: np.linalg.norm(np.asarray(h["keypoints_2d"]) - np.asarray(k["keypoints_2d"]), axis=-1).mean() < \
                        self.vd_dedup * np.hypot(k["bbox"][2] - k["bbox"][0], k["bbox"][3] - k["bbox"][1])
                    hands = [h for h in hands if not h.get("vd_prop") or not any(same(h, k) for k in real)]
                self._vd.update(hands)
                self.vd_last = dict(self.vd_last or {}, kept_props=sum(1 for h in hands if h.get("vd_prop")))
            self.prev = hands
        elif self.schedule is None:
            hands = self.est.process_image(image_bgr, cam_int=K)
        else:  # video: detector every N frames, decoder-predicted boxes in between
            boxes, scores, is_right = self.schedule(image_bgr, self.prev)
            hands = self.est.process_image(image_bgr, boxes=boxes, is_right=is_right, scores=scores, cam_int=K) \
                if len(boxes) else []
            self.prev = hands
        tap = None
        if getattr(self, "contact_video", ""):  # video contact inputs: per-frame values, before the refiner changes the joints
            tap = [(h, dict(is_right=h["is_right"], score=h["score"], joints3d=np.array(h.get("keypoints_3d_mano", h["keypoints_3d"])),
                            contact_dense=h["contact_dense"].astype(np.float16), contact_score=h["contact_score"],
                            **({"token": h["token"]} if "token" in h else {})))
                   for h in hands if "contact_dense" in h]
        if self.tracker is not None:
            if getattr(self, "vd_gate_fill", False) and hasattr(self.tracker, "no_fill"):
                self.tracker.no_fill = getattr(self, "_vd_empty", set()) if getattr(self, "_vd", None) is not None else set()
            hands = self.tracker.update(hands, self.t / self.fps)
            self.t += 1
        if self.mesh_fit:  # full fp32 even under tf32: the Gauss-Newton step's finite-difference Jacobian needs it
            tf, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, False
            try:
                self._fit_meshes(hands)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf
        if getattr(self, "mhr_fit", 0):  # (objects built without __init__, e.g. in tests, have no MHR fit)
            tf, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, False
            try:
                self._fit_mhr(hands)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = tf
        out = []
        ids = None
        if self.masks and hands:
            from bench.seg.rle import encode
            from fah.vis import render_instance_masks

            meshes = [(h["vertices"], self.est.faces) if "vertices" in h else  # dense keypoint model: its surface points
                      (h["vertices_mano"], self._mano_faces()) if h.get("mesh_param") else  # its MANO head mesh
                      (self._surface_2d(h), self._mano_faces()) if "vertices_mano" in h else (h["keypoints_3d"], np.zeros((0, 3), int))
                      for h in hands]
            ids = render_instance_masks(meshes, hands[0]["cam_int"], image_bgr.shape[:2])
        if any("mask" in h or "mask_arm" in h for h in hands):
            from bench.seg.rle import encode
        dm, dj = None, None
        need_parts = self.parts_in_mask and any("parts_fg_roi" in h for h in hands)
        if (self.det_masks or self.tip_extend or need_parts) and hands and getattr(self.detector, "last", None) is not None:
            from bench.seg.rle import encode

            db = self.detector.last["boxes"]  # hands keep their detector box
            j = [int(np.abs(db - h["bbox"]).sum(1).argmin()) if len(db) else -1 for h in hands]
            ok = [i >= 0 and np.abs(db[i] - h["bbox"]).sum() < 1 for i, h in zip(j, hands)]
            dj = [i if o else None for i, o in zip(j, ok)]
            if self.det_masks:  # "crop": per hand (x0, y0, box crop), no full-frame masks and no RLE (bench.seg.rle paste_crop / encode_crop)
                crop = self.det_masks == "crop"
                m = iter((self.detector.masks_crop if crop else self.detector.masks)([i for i, o in zip(j, ok) if o]))
                dm = [next(m) if o else None for o in ok]
                if getattr(self, "mask_split", True):  # one detector box, two hands (pair_wrist runs): its mask split by the hands' mesh silhouettes
                    if crop:  # only the shared boxes' masks go full-frame for the split, then back to their box
                        from bench.seg.rle import paste_crop

                        hw = image_bgr.shape[:2]
                        shared = [k for k, i in enumerate(dj) if i is not None and dm[k] is not None and sum(x == i for x in dj) > 1]
                        if shared:
                            full = [paste_crop(dm[k][2], dm[k][0], dm[k][1], hw) if k in shared else None for k in range(len(dm))]
                            split_shared_masks(full, dj, hands, self._mano_faces(), hw)
                            for k in shared:
                                x0, y0, c = dm[k]
                                dm[k] = (x0, y0, full[k][y0:y0 + c.shape[0], x0:x0 + c.shape[1]])
                    elif any(i is not None and dj.count(i) > 1 for i in dj):
                        split_shared_masks(dm, dj, hands, self._mano_faces(), image_bgr.shape[:2])
        n0 = len(out)
        for k, h in enumerate(hands):
            verts = None
            if "vertices_mano" in h:  # MHR-free dense keypoint model: MANO-topology vertices predicted directly
                verts = h["vertices_mano"]
            if self.vreg is not None and "vertices" in h:
                W, idx = self.vreg
                verts = W @ h["vertices"][idx]  # MANO-topology vertices (camera frame)
            out.append(dict(
                is_right=h["is_right"], score=h["score"], bbox=h["bbox"],
                joints3d=h.get("keypoints_3d_mano", h["keypoints_3d"]),
                joints3d_mhr=h["keypoints_3d"],
                verts3d=verts,
                joints2d=h.get("keypoints_2d_mano", h["keypoints_2d"]),
                joints2d_mhr=h["keypoints_2d"],
                **({"token": h["token"]} if "token" in h else {}),
                **({"track_id": h["vd_track"]} if "vd_track" in h else {}),  # video detector track (gap fills: their track's)
                **({"mesh_param": True} if h.get("mesh_param") else {}),
                **({"verts3d_dense": h["vertices_dense"]} if "vertices_dense" in h else {}),
                **({"verts3d_mhr": h["vertices_mhr"], "verts3d_mhr_mano": h["vertices_mhr_mano"], "joints3d_mhr_fit": h["keypoints_3d_mhr_fit"],
                    "joints3d_mhr_rig": h["keypoints_3d_mhr_rig"], "mhr_params": h["mhr_params"]} if "vertices_mhr" in h else {}),
                **({"parts_roi": h["parts_roi"], "parts_fg_roi": h["parts_fg_roi"]} if "parts_roi" in h else {}),
                **({"contact": h["contact"].astype(np.float16), "contact_dense": h["contact_dense"].astype(np.float16),
                    "contact_score": h["contact_score"], "contact_fingers": h["contact_fingers"],
                    **({"contact_k": h["contact_k"].astype(np.float16), "contact_score_k": h["contact_score_k"]} if "contact_k" in h else {})}
                   if "contact" in h else {}),
            ))
            if "keypoints_2d_head" in h:  # direct keypoint head; 3D = head root-relative + MHR/MANO root
                root = out[-1]["joints3d"][0]
                out[-1].update(joints2d_head=h["keypoints_2d_head"], joints3d_head=h["keypoints_3d_head_rel"] + root,
                               surface2d_head=h["surface_2d_head"], surface3d_head=h["surface_3d_head_rel"] + root,
                               **({"surface_visible": h["surface_visible"]} if "surface_visible" in h else {}))
            if ids is not None:
                out[-1]["mask_render"] = out[-1]["mask_rle"] = encode(ids == k + 1)
            if self.det_masks and dm is not None and dm[k] is not None:  # detector masks (hand only, inside the box), when exported
                if self.det_masks == "crop":
                    out[-1]["mask_det_crop"] = dm[k]
                else:
                    out[-1]["mask_det"] = out[-1]["mask_rle"] = encode(dm[k])
            if "mask" in h:  # learned mask head (preferred as the default "mask_rle")
                out[-1]["mask_head"] = out[-1]["mask_rle"] = encode(h["mask"])
            if "mask_arm" in h:  # hand + visible arm (VISOR convention)
                out[-1]["mask_head_arm"] = encode(h["mask_arm"])
        if self.tip_extend and dj is not None and any(i is not None for i in dj):
            self._extend_tips([out[n0 + k] for k, i in enumerate(dj) if i is not None], [i for i in dj if i is not None])
        if self.parts_in_mask and dj is not None:  # parts and instance mask agree by construction
            todo = [(out[n0 + k], i, k) for k, i in enumerate(dj) if i is not None and out[n0 + k].get("parts_fg_roi") is not None]
            if todo:
                if dm is not None and self.det_masks != "crop":  # reuse the exported mask: exact same pixels, no second mask inference or pixel grid
                    values = []
                    for o, _, k in todo:
                        x0, y0, roi = o["parts_fg_roi"]
                        values.append(dm[k][y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]].reshape(-1))
                elif hasattr(self.detector, "mask_roi"):  # the ROI pixel grid on the device
                    values = self.detector.mask_roi([i for _, i, _ in todo],
                                                    [(x0, y0, roi.shape[1], roi.shape[0]) for x0, y0, roi in (o["parts_fg_roi"] for o, _, _ in todo)])
                else:
                    pts = []
                    for o, _, _ in todo:
                        x0, y0, roi = o["parts_fg_roi"]
                        ys, xs = np.mgrid[y0:y0 + roi.shape[0], x0:x0 + roi.shape[1]]
                        pts.append((xs.ravel(), ys.ravel()))
                    values = self.detector.mask_at([i for _, i, _ in todo], pts)
                for (o, _, _), m in zip(todo, values):
                    x0, y0, roi = o["parts_fg_roi"]
                    o["parts_roi_head"] = o.get("parts_roi")
                    o["parts_roi"] = (x0, y0, np.where(m.reshape(roi.shape), roi, 0).astype(np.uint8))
        if tap is not None:
            self._video_contact(tap, hands, out[n0:])
        return out

    def _video_contact(self, tap, hands, outs):
        """contact_video / contact_score_video / contact_flag_video (fah.contact_video.VideoContact) of the hands in `tap`
        (hand, its per-frame inputs); refiner gap-fill hands (not in `tap`) get none."""
        if self._vc is None:
            if not tap:  # before the first hand: nothing to add (the window starts zero-padded either way)
                return
            from fah.contact_video import VideoContact

            kh = self.est.model.keypoint_head
            bias = float(kh.contact_bias[kh.contact_main]) if getattr(kh, "contact_bias", None) is not None else 0.0
            self._vc = VideoContact(self.contact_video, self.est.sub_R, bias, device=self.est.model.device,
                                    cuda_graph=getattr(self, "refiner_graph", False), smooth=getattr(self, "contact_smooth", 1),
                                    h=getattr(self, "contact_h", None))
        self._vc([d for _, d in tap])
        by = {id(h): d for h, d in tap}
        for h, o in zip(hands, outs):
            d = by.get(id(h), {})
            if "contact_video" in d:
                o.update(contact_video=d["contact_video"], contact_score_video=d["contact_score_video"], contact_flag_video=d["contact_flag_video"])
