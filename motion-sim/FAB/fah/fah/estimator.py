import os

import cv2
import numpy as np
import torch
from torch.utils.data import default_collate
from torchvision.transforms import ToTensor

from .data.transforms import (
    Compose,
    GetBBoxCenterScale,
    TopdownAffine,
    VisionTransformWrapper,
)
from .frame_cache import frame_cache
from .metadata import MHR70_TO_OPENPOSE_HAND, MHR_RIGHT_WRIST_JOINT
from .models.heads.mhr_hand_fast import hand_vertex_mask
from .vis import project

BATCH_KEYS = [
    "img",
    "img_size",
    "ori_img_size",
    "bbox_center",
    "bbox_scale",
    "bbox",
    "affine_trans",
]
# MHR -> camera axes (applied to vertices / keypoints inside MHRHead)
MHR_TO_CAM = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float32)
DEFAULT_MANO_REGRESSOR = os.path.join(
    os.path.dirname(__file__), "assets", "mhr_to_mano21_regressor.npz"
)


def default_cam_int(height, width):
    # Same default as SAM 3D Body: focal length = image diagonal
    f = (height**2 + width**2) ** 0.5
    return np.array(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float32
    )


def fov_to_cam_int(fov_deg, height, width):
    # Horizontal field of view in degrees
    f = (width / 2.0) / np.tan(np.deg2rad(fov_deg) / 2.0)
    return np.array(
        [[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], dtype=np.float32
    )


def load_joint_regressor(path, num_verts):
    """Sparse (21, num_verts) MHR-vertex -> MANO-convention joint regressor."""
    data = np.load(path)
    W = np.zeros((int(data["num_joints"]), num_verts), dtype=np.float32)
    W[data["joint"], data["vertex"]] = data["weight"]
    return W


def flip_boxes(boxes, width):
    flipped = boxes.copy()
    flipped[:, 0] = width - boxes[:, 2] - 1
    flipped[:, 2] = width - boxes[:, 0] - 1
    return flipped


HAND_PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19])


def rotation_between(a, b):
    a, b = a / max(np.linalg.norm(a), 1e-9), b / max(np.linalg.norm(b), 1e-9)
    v, c = np.cross(a, b), float(a @ b)
    if c < -0.999999:  # opposite: half turn about an axis orthogonal to a
        u = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        u /= np.linalg.norm(u)
        return 2 * np.outer(u, u) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def perspective_view(K, box):
    """Virtual camera rotated onto the ray through the box centre: R (virtual <- camera axes), K_v (same focal,
    principal point at the box centre) and H = K_v R K^-1 (image -> virtual image pixels)."""
    c = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 1.0])
    K = np.asarray(K, np.float64)
    R = rotation_between(np.linalg.inv(K) @ c, np.array([0.0, 0.0, 1.0]))
    Kv = K.copy()
    Kv[0, 2], Kv[1, 2] = c[0], c[1]
    return R, Kv, Kv @ R @ np.linalg.inv(K)


def warp_points(H, uv):
    p = np.concatenate([np.asarray(uv, np.float64), np.ones((len(uv), 1))], 1) @ H.T
    return p[:, :2] / p[:, 2:]


def warp_box(H, box):
    """Axis-aligned box around the 4 warped corners of an xyxy box."""
    p = warp_points(H, [[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]])
    return np.array([*p.min(0), *p.max(0)], dtype=np.float32)


def unwarp_hand(h, R, H, cam_int, height, width):
    """Hand dict predicted in a virtual camera (perspective_view, unmirrored frame) -> the real camera: 3D by R^T,
    2D / boxes by H^-1, masks and part ROIs warped back."""
    Hi = np.linalg.inv(H)
    for k in ("keypoints_3d", "keypoints_3d_mano", "vertices", "vertices_mano", "vertices_dense", "keypoints_3d_rel", "keypoints_3d_head_rel",
              "surface_3d_head_rel", "cam_t"):
        if h.get(k) is not None:
            h[k] = (np.asarray(h[k], np.float64) @ R).astype(np.float32)  # rows: R^T x
    if h.get("wrist_rot") is not None:
        h["wrist_rot"] = (R.T @ h["wrist_rot"]).astype(np.float32)
    for k in ("keypoints_2d", "keypoints_2d_mano", "keypoints_2d_head", "surface_2d_head"):
        if h.get(k) is not None:
            h[k] = warp_points(Hi, h[k]).astype(np.float32)
    for k in ("crop_box", "pred_box"):
        if h.get(k) is not None:
            h[k] = warp_box(Hi, h[k])
    for k in ("mask", "mask_arm"):
        if h.get(k) is not None:
            h[k] = cv2.warpPerspective(h[k].astype(np.uint8), Hi, (width, height), flags=cv2.INTER_NEAREST) > 0
    for k in ("parts_roi", "parts_fg_roi"):
        if h.get(k) is None:
            continue
        x, y, lab = h[k]
        if not lab.size:
            h.pop(k)
            continue
        # Source ROI pixels -> virtual image -> real image. Bound the nearest-neighbour pixel support, and retain
        # foreground candidates over the real detector box for subsequent per-pixel mask conditioning.
        T = Hi @ np.array([[1., 0, x], [0, 1., y], [0, 0, 1.]])
        rh, rw = lab.shape
        corners = warp_points(T, [[-.5, -.5], [rw - .5, -.5], [-.5, rh - .5], [rw - .5, rh - .5]])
        lo, hi = corners.min(0), corners.max(0)
        if h.get("bbox") is not None:
            lo, hi = np.minimum(lo, h["bbox"][:2]), np.maximum(hi, h["bbox"][2:])
        x0, y0 = np.floor(lo).astype(int).clip(0, [width, height])
        x1, y1 = np.ceil(hi).astype(int).clip(0, [width, height])
        if x1 <= x0 or y1 <= y0:
            h.pop(k)
            continue
        T = np.array([[1., 0, -x0], [0, 1., -y0], [0, 0, 1.]]) @ T
        h[k] = (int(x0), int(y0), cv2.warpPerspective(lab, T, (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE if k == "parts_fg_roi" else cv2.BORDER_CONSTANT, borderValue=0))
    h["cam_int"] = cam_int
    return h


def lift_to_2d(P, uv, cam, pts=None):
    """3D joints P (21, 3, camera frame) moved onto the camera rays of their 2D joints uv (21, 2), root to tips,
    keeping every bone length and the predicted depth order along each bone (the closest ray point when the ray is
    out of reach); points pts (N, 3) follow their nearest bone rigidly. Returns the new joints and points."""
    fx, fy, cx, cy = cam
    rays = np.stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))], 1)
    Q = P.copy()
    Q[0] = rays[0] * P[0, 2]
    for c in range(1, 21):
        p = HAND_PARENTS[c]
        r, L = rays[c], np.linalg.norm(P[c] - P[p])
        a, b = r @ r, r @ Q[p]
        disc = b * b - a * (Q[p] @ Q[p] - L * L)
        if disc <= 0:
            s = b / a
        else:
            zt = Q[p, 2] + P[c, 2] - P[p, 2]
            s = min(((b - np.sqrt(disc)) / a, (b + np.sqrt(disc)) / a), key=lambda z: abs(z - zt))
        Q[c] = s * r
    if pts is None:
        return Q, None
    A, B = P[HAND_PARENTS[1:]], P[1:]
    d = B - A
    tt = np.clip(np.einsum("nkc,kc->nk", pts[:, None] - A[None], d) / np.maximum((d * d).sum(-1), 1e-12), 0, 1)
    k = np.linalg.norm(pts[:, None] - (A[None] + tt[..., None] * d[None]), axis=-1).argmin(1)
    out = np.empty_like(pts)
    for j in range(20):
        sel = k == j
        if sel.any():
            R = rotation_between(d[j], Q[j + 1] - Q[HAND_PARENTS[j + 1]])
            out[sel] = Q[HAND_PARENTS[j + 1]] + (pts[sel] - A[j]) @ R.T
    return Q, out


class HandEstimator:
    """Detect hands in an image and reconstruct each one with the SAM 3D Body hand
    decoder. Left hands are mirrored into right hands before inference (as in
    SAM 3D Body) and mirrored back afterwards."""

    def __init__(
        self,
        model,
        cfg,
        detector=None,
        bbox_padding=0.9,
        min_crop=0.0,
        refine_box_iters=0,
        recrop_border=0.0,
        gpu_crop=False,
        fast_batch=False,
        student_trt="",
        student_coreml="",
        student_coreml_units="CPU_AND_GPU",
        max_batch_size=8,
        mano_regressor_path=DEFAULT_MANO_REGRESSOR,
        fast_mhr=True,
        input_size=None,
        decoder_depth=None,
        student=None,
        cuda_graph=False,
        compile=False,
        warp_size=None,
        refine_mesh_iters=0,
        surface_conf=0.0,
        heatmap_local=0,
        heatmap_local_gate=0.0,
        lift_2d=False,
        surface_smooth=0,
        persp_crop=False,
    ):
        self.model = model
        # crops from a virtual camera rotated onto each box-centre ray (homography warp, on-axis view of off-axis
        # hands); outputs are mapped back to the real camera
        self.persp_crop = persp_crop
        # fit the mesh (rotation + finger articulation) to the direct keypoint head's 2D for N Adam steps
        self.refine_mesh_iters = refine_mesh_iters
        self.surface_conf = surface_conf  # dense model: heatmap peak below this -> projected 3D surface point
        if heatmap_local and getattr(model, "keypoint_head", None) is not None:
            model.keypoint_head.local_r = abs(heatmap_local)  # local soft-argmax window radius (heatmap cells)
            # < 0: local window for the dense surface 2D only (joints, root and 3D keep the global soft-argmax)
            model.keypoint_head.local_split = heatmap_local < 0
        self.lift_2d = lift_2d  # dense model: 3D joints onto the rays of the heatmap 2D joints (bone lengths kept)
        self.surface_smooth = surface_smooth  # dense model: K -> surface (3D, 2D) projected onto the K lowest Laplacian modes
        self.spec_basis = None
        self.parts = True  # part maps from a part head's logits (False: skipped, other outputs unchanged; fah-live until key m)
        self.local_gate = heatmap_local_gate  # > 0: local surface 2D only where it is this far (crop units) from the global mean
        self.cfg = cfg
        self.device = model.device
        self.detector = detector
        self.bbox_padding = bbox_padding
        self.min_crop = min_crop
        self.refine_box_iters = refine_box_iters
        self.recrop_border = recrop_border  # > 0: re-run hands whose joints come within this crop fraction of a crop edge
        self.gpu_crop = gpu_crop and str(self.device).startswith("cuda")  # crops sampled on the device (frame uploaded once)
        self.fast_batch = fast_batch  # with gpu_crop: _make_batch_fast (closed-form affines, one host-to-device copy)
        self.student_trt = None  # TensorRT engine of the student encoder (scripts/build_pose_engine.py)
        if student_trt and str(self.device).startswith("cuda") and student is not None:
            from .student import TRTEncoder

            self.student_trt = TRTEncoder(student_trt)
        # Core ML (Apple silicon) student encoder, or the whole decoder-free pose network (scripts/build_coreml.py pose
        # --heads): the latter takes host crops, so the batch stays on the CPU
        self.student_coreml = None
        if student_coreml and student is not None:
            from .coreml import CoreMLPose

            self.student_coreml = CoreMLPose(student_coreml, student_coreml_units)
        self.host_batch = self.student_coreml is not None and self.student_coreml.heads
        self.batch_device = "cpu" if self.host_batch else self.device
        self.max_batch_size = max_batch_size
        # Crop resolution (trained at 512; smaller crops trade accuracy for speed)
        if isinstance(input_size, int):
            input_size = [input_size, input_size]
        self.input_size = list(input_size or cfg.MODEL.IMAGE_SIZE)
        # Optional distilled encoder (fah.student) replacing the backbone; the decoder builds
        # rays from the crop at patch 16, so the crop is 16 x the student's token grid
        self.student = student
        if student is not None:
            self.input_size = [16 * student.grid] * 2
        # Replay backbone + decoder + rig as one CUDA graph per crop count (the decoder/rig issue
        # ~1e3 tiny kernels per layer, so eager GPU inference is launch-bound)
        # torch.compile (inductor) fuses the decoder/rig's many tiny elementwise kernels; compiled
        # once per model (the compiled callables are shared by every estimator on that model)
        if compile:
            from torch import _dynamo

            # per-layer / per-crop-count specialisations exceed the default limit of 8
            _dynamo.config.cache_size_limit = max(_dynamo.config.cache_size_limit, 64)
            if not getattr(model, "_fah_compiled", False):
                model.forward_decoder_hand = torch.compile(model.forward_decoder_hand)
                if student is None:
                    model.backbone.forward = torch.compile(model.backbone.forward)
                model._fah_compiled = True
            if student is not None and not getattr(student, "_fah_compiled", False):
                student.forward = torch.compile(student.forward)
                student._fah_compiled = True
        self.graphed = None
        if cuda_graph and torch.device(self.device).type == "cuda":
            from .cuda_graph import GraphedForward

            self.graphed = GraphedForward(self._forward)

        self._placeholder = np.zeros((1, 1, 3), np.uint8)  # transform only computes the affine; crops are warped in _make_batch
        self._to_tensor = ToTensor()
        self.transform = Compose(
            [
                GetBBoxCenterScale(padding=bbox_padding),
                TopdownAffine(input_size=self.input_size, use_udp=False),
                VisionTransformWrapper(ToTensor()),
            ]
        )

        # Hand-only sub-mesh of the MHR body mesh
        faces = model.head_pose_hand.faces.cpu().numpy()
        vmask = hand_vertex_mask(model.head_pose_hand.mhr)
        self.hand_vert_idx = np.nonzero(vmask)[0]
        remap = -np.ones(len(vmask), dtype=np.int64)
        remap[self.hand_vert_idx] = np.arange(len(self.hand_vert_idx))
        face_sel = vmask[faces].all(axis=1)
        self.faces = remap[faces[face_sel]]
        self.faces_left = self.faces[:, ::-1].copy()
        self.full_faces = faces
        # Decoder early exit (None = all 6 layers). Applied per call in _run_crops, since
        # several estimators may share one model.
        n_layers = len(model.decoder_hand.layers)
        assert decoder_depth is None or 1 <= decoder_depth <= n_layers, f"decoder_depth must be in [1, {n_layers}]"
        self.decoder_depth = decoder_depth
        # crops warped at warp_size, then antialiased down to input_size on device (as in training, where
        # 384 crops are resized to the student size); None = warp straight at input_size
        self.warp_size = warp_size if warp_size and warp_size > self.input_size[0] else None
        # Skeleton-only rig for intermediate decoder layers, hand-only skinning at the end
        self.fast_mhr = fast_mhr
        if fast_mhr and model.head_pose_hand.hand_mhr is None:
            model.head_pose_hand.enable_fast_hand(self.hand_vert_idx)

        # Optional MANO/FreiHAND-convention joints (scripts/fit_mano_joint_regressor.py)
        self.mano_regressor = None
        if mano_regressor_path and os.path.exists(mano_regressor_path):
            W = load_joint_regressor(mano_regressor_path, len(vmask))
            reg_verts = np.nonzero(W.any(0))[0]
            assert vmask[reg_verts].all(), "regressor uses non-hand vertices"
            self.mano_regressor = W[:, reg_verts]
            self.mano_regressor_vert_idx = remap[reg_verts]  # into the hand sub-mesh

    # ------------------------------------------------------------------ batching
    def _make_batch(self, crops):
        """crops: list of (img_bgr, box_xyxy, cam_int, flip) -- box/cam_int in the mirrored frame when
        flip -- or legacy (img_rgb, box_xyxy, cam_int). Each crop is warped straight from the input
        frame (mirrored affine for flip), so no full-frame RGB / mirrored copies are made."""
        if self.fast_batch and (self.gpu_crop or getattr(self, "host_batch", False)) and all(len(c) < 5 or c[4] is None for c in crops):
            return self._make_batch_fast(crops)
        data_list, gpu = [], []
        for c in crops:
            img, box, _, flip, Hv = c if len(c) == 5 else (*c, None) if len(c) == 4 else (*c, None, None)
            if self.min_crop > 0:  # tall boxes: the body crop rule gives only bbox_padding x height; widen the box so the
                bw, bh = box[2] - box[0], box[3] - box[1]  # crop covers min_crop x its long side (wide boxes unchanged)
                need = self.min_crop * bh / (4 / 3 * self.bbox_padding)
                if bh > bw and bw < need:
                    box = np.array([box[0] - (need - bw) / 2, box[1], box[2] + (need - bw) / 2, box[3]], np.float32)
            d = self.transform(dict(img=self._placeholder, bbox=box.astype(np.float32), bbox_format="xyxy"))
            H, W = img.shape[:2]
            A = d["affine_trans"].astype(np.float64)
            size = tuple(self.input_size)
            if self.warp_size:
                k = self.warp_size / self.input_size[0]
                A, size = A * k, (self.warp_size, self.warp_size)
                A[:, 2] += (k - 1) / 2  # pixel centres line up with the align_corners=False resize below
            F = np.array([[-1.0, 0, W - 1], [0, 1, 0], [0, 0, 1]])  # sample the original frame: x_mirrored = W - 1 - x
            if self.gpu_crop and Hv is None:  # warped on the device from the frame uploaded once (below)
                gpu.append((img, A @ F if flip else A, size, flip is not None))
            else:
                if Hv is not None:  # perspective crop: original -> (mirrored) frame -> virtual camera -> crop
                    M = np.vstack([A, [0, 0, 1]]) @ Hv @ (F if flip else np.eye(3))
                    crop = cv2.warpPerspective(img, M, size, flags=cv2.INTER_LINEAR)
                else:
                    crop = cv2.warpAffine(img, A @ F if flip else A, size, flags=cv2.INTER_LINEAR)
                if flip is not None:
                    crop = crop[:, :, ::-1]  # BGR -> RGB on the crop
                d["img"] = torch.from_numpy(np.ascontiguousarray(crop.transpose(2, 0, 1)))  # uint8; [0, 1] float on device
            d["ori_img_size"] = np.array([W, H])
            data_list.append(d)
        batch = default_collate(data_list)
        if gpu:
            img = self._gpu_warp(gpu)  # (n, 3, h, w) uint8-range float
        else:
            img = batch.pop("img").to(self.batch_device, non_blocking=True)  # 4x smaller transfer than float crops
        for key in BATCH_KEYS[1:]:
            batch[key] = batch[key].unsqueeze(1).float()
        n = img.shape[0]
        h, w = img.shape[-2:]
        batch["mask_score"] = torch.zeros((n, 1))
        batch["mask_prompt"] = False
        batch["img_unit_range"] = True  # [0, 1] crops
        batch["cam_int"] = torch.from_numpy(np.stack([c[2] for c in crops])).float()
        batch = {k: v.to(self.batch_device) if torch.is_tensor(v) else v for k, v in batch.items()}
        img = img.float().div_(255.0)
        if self.warp_size:
            h, w = self.input_size[1], self.input_size[0]
            img = torch.nn.functional.interpolate(img, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        batch["img"] = img.unsqueeze(1)
        batch["mask"] = torch.zeros((n, 1, 1, h, w), device=self.batch_device)
        return batch

    def _make_batch_fast(self, crops):
        """_make_batch for on-device crops (gpu_crop, no perspective crops) without the per-crop transform pipeline:
        the GetBBoxCenterScale + TopdownAffine affine (rotation 0: a scale about the box centre) in closed form for all
        crops at once, the per-crop arrays in one host-to-device copy, and only the batch entries the model reads.
        Host crops (cv2.warpAffine, as _make_batch) when the batch stays on the CPU (Core ML pose network)."""
        img = crops[0][0]
        H, W = img.shape[:2]
        n = len(crops)
        boxes = np.stack([np.asarray(c[1], np.float32) for c in crops]).astype(np.float64)
        flips = [c[3] if len(c) >= 4 else None for c in crops]
        if self.min_crop > 0:  # as _make_batch: tall boxes widened to min_crop x their long side
            bw, bh = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
            need = self.min_crop * bh / (4 / 3 * self.bbox_padding)
            d = np.where((bh > bw) & (bw < need), (need - bw) / 2, 0.0)
            boxes[:, 0] -= d
            boxes[:, 2] += d
        w, h = self.input_size
        center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        scale = (boxes[:, 2:] - boxes[:, :2]) * self.bbox_padding
        for ar in (0.75, w / h):  # TopdownAffine: fix_aspect_ratio to 0.75, then to the input aspect
            sw, sh = scale[:, 0], scale[:, 1]
            scale = np.where((sw > sh * ar)[:, None], np.stack([sw, sw / ar], 1), np.stack([sh * ar, sh], 1))
        s = w / scale[:, 0]
        A = np.zeros((n, 2, 3))
        A[:, 0, 0] = A[:, 1, 1] = s
        A[:, :, 2] = np.array([w * 0.5, h * 0.5]) - s[:, None] * center
        Aw, size = A, tuple(self.input_size)
        if self.warp_size:
            k = self.warp_size / self.input_size[0]
            Aw, size = A * k, (self.warp_size, self.warp_size)
            Aw[:, :, 2] += (k - 1) / 2
        F = np.array([[-1.0, 0, W - 1], [0, 1, 0], [0, 0, 1]])
        gpu = [(img, Aw[i] @ F if f else Aw[i], size, f is not None) for i, f in enumerate(flips)]
        if self.gpu_crop:
            x = self._gpu_warp(gpu).float().div_(255.0)
        else:  # cv2.warpAffine as _make_batch (BGR -> RGB for the (img, box, K, flip) crops), [0, 1] NCHW in one pass
            x = torch.from_numpy(np.concatenate([cv2.dnn.blobFromImage(cv2.warpAffine(im, M, sz, flags=cv2.INTER_LINEAR), 1 / 255.0,
                                                                       swapRB=rgb) for im, M, sz, rgb in gpu]))
        packed = np.concatenate([np.tile([w, h, W, H], (n, 1)), center, scale, boxes, A.reshape(n, 6),
                                 np.stack([c[2] for c in crops]).reshape(n, 9)], 1).astype(np.float32)
        dev = self.batch_device
        P = torch.from_numpy(packed).to(dev, non_blocking=True)
        batch = dict(img_size=P[:, None, 0:2], ori_img_size=P[:, None, 2:4], bbox_center=P[:, None, 4:6],
                     bbox_scale=P[:, None, 6:8], bbox=P[:, None, 8:12], affine_trans=P[:, None, 12:18].reshape(n, 1, 2, 3),
                     cam_int=P[:, 18:27].reshape(n, 3, 3).contiguous(), mask_score=torch.zeros((n, 1), device=dev),
                     mask_prompt=False, img_unit_range=True)
        if self.warp_size:
            h, w = self.input_size[1], self.input_size[0]
            x = torch.nn.functional.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
        batch["img"] = x.unsqueeze(1)
        batch["mask"] = torch.zeros((n, 1, 1, h, w), device=dev)
        return batch

    def _gpu_warp(self, items):
        """Affine crops (cv2.warpAffine semantics: dst pixel -> A^-1 -> src pixel, bilinear, zero border) sampled on the
        device from the frame, uploaded once per frame (cached across the passes of one frame)."""
        from .frame_cache import frame_tensor

        img = items[0][0]
        self._frame = frame_tensor(img, self.device)  # (1, 3, H, W) BGR, 0..255, shared with the detector letterbox
        Hf, Wf = img.shape[:2]
        w, h = items[0][2]
        if getattr(self, "_grid_base", None) is None or self._grid_base.shape[:2] != (h, w):
            v, u = torch.meshgrid(torch.arange(h, device=self.device, dtype=torch.float32),
                                  torch.arange(w, device=self.device, dtype=torch.float32), indexing="ij")
            self._grid_base = torch.stack([u, v, torch.ones_like(u)], -1)  # (h, w, 3)
        A3 = np.zeros((len(items), 3, 3))
        A3[:, :2] = np.stack([A for _, A, _, _ in items])
        A3[:, 2, 2] = 1
        Minv = np.linalg.inv(A3)[:, :2]  # crop -> frame pixels
        Minv = Minv * np.array([2 / (Wf - 1), 2 / (Hf - 1)])[:, None]  # -> grid_sample's [-1, 1] (align_corners)
        Minv[:, :, 2] -= 1
        grid = torch.einsum("hwk,nck->nhwc", self._grid_base, torch.as_tensor(Minv, dtype=torch.float32).to(self.device, non_blocking=True))
        out = torch.nn.functional.grid_sample(self._frame.expand(len(items), -1, -1, -1), grid, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)
        flags = [it[3] for it in items]
        if all(flags):  # BGR -> RGB for the (img, box, K, flip) crops
            return out.flip(1)
        rgb = torch.tensor(flags, device=self.device)[:, None, None, None]
        return torch.where(rgb, out.flip(1), out)

    def _forward(self, batch):
        emb = None
        if getattr(self, "host_batch", False):
            return self._forward_coreml(batch)
        if self.student is not None:
            x = torch.nn.functional.interpolate(batch["img"][:, 0], size=(self.student.input_size,) * 2,
                                                mode="bilinear", align_corners=False, antialias=True)
            if getattr(self, "student_coreml", None) is not None:
                emb = torch.from_numpy(self.student_coreml(img=x.cpu().numpy())["feat"]).to(self.device)
            else:
                emb = self.student_trt(x) if self.student_trt is not None else self.student(x.to(self.student.mean.dtype)).float()
                kh = getattr(self.model, "keypoint_head", None)
                if self.student_trt is not None and getattr(kh, "contact_branch", None) is not None:  # the branch's input: the engine's
                    kh._br_in = (self.student_trt.last_hidden, kh.contact_branch_kw)  # hidden state (PyTorch: the encoder's pre-hook)
        return self.model(batch, image_embeddings=emb)

    def _forward_coreml(self, batch):
        """The kp_only branch of SAM3DHand.forward in Core ML: crops, the per-crop ray grid as x = sx u + ox, y = sy v + oy
        (SAM3DHand.get_ray_condition) and the CLIFF condition in; host tensors out."""
        m = self.model
        A, K = batch["affine_trans"][:, 0].double(), batch["cam_int"].double()
        sx, sy = 1 / (A[:, 0, 0] * K[:, 0, 0]), 1 / (A[:, 1, 1] * K[:, 1, 1])
        ox = (-A[:, 0, 2] / A[:, 0, 0] - K[:, 0, 2]) / K[:, 0, 0]
        oy = (-A[:, 1, 2] / A[:, 1, 1] - K[:, 1, 2]) / K[:, 1, 1]
        m._initialize_batch(batch)
        cond = m._get_decoder_condition(batch)
        out = self.student_coreml(img=batch["img"][:, 0].numpy(), ray=torch.stack([sx, ox, sy, oy], 1).float().numpy(),
                                  cond=cond.float().numpy())
        out = {k: torch.from_numpy(v) for k, v in out.items()}
        if getattr(self, "_kp_dense", None) is None:
            self._kp_dense = m.keypoint_head.dense.cpu()
        out["kp_dense"] = self._kp_dense[None].expand(len(cond), -1)
        return out

    @torch.no_grad()
    def _run_crops(self, crops):
        outputs = []
        for i in range(0, len(crops), self.max_batch_size):
            batch = self._make_batch(crops[i : i + self.max_batch_size])
            kh = getattr(self.model, "keypoint_head", None)
            if kh is not None:  # part maps off: the part head is not run (its own graphs: the flag is part of the key)
                kh.parts_on = batch["parts_on"] = bool(getattr(self, "parts", True))
            self.model.decoder_hand.active_depth = self.decoder_depth
            self.model.head_pose_hand.use_fast_hand = self.fast_mhr
            out = self.graphed(batch) if self.graphed is not None else self._forward(batch)
            if self.refine_mesh_iters and "pt_uv" in out and "pred_vertices" in out:
                out = self._refine_mesh_to_head(batch, dict(out))
            if not getattr(self, "parts", True) and "part_logits" in out:  # (objects built without __init__ in tests: on)
                out = {k: v for k, v in out.items() if k != "part_logits"}
            if torch.is_tensor(out.get("part_logits")):  # part labels made on the device: two uint8 maps per crop, not 17 float channels
                out = dict(out)
                S = self.input_size[0]
                lg = torch.nn.functional.interpolate(out.pop("part_logits").float(), size=(S, S), mode="bilinear", align_corners=False)
                if lg.device.type == "cpu":  # Core ML: the argmax over contiguous channels (same labels; 15 -> 2 ms per crop on an M3 Pro)
                    lg = lg.contiguous(memory_format=torch.channels_last)
                out["part_lab"] = lg.argmax(1).to(torch.uint8)  # logits upsampled, then argmax
                out["part_fg"] = (lg[:, 1:].argmax(1) + 1).to(torch.uint8)  # the best hand part everywhere (for a part map inside a given mask)
            if self.fast_batch:  # every output (and the batch geometry) in one device-to-host copy
                n = batch["img"].shape[0]
                ten = {k: v for k, v in out.items() if torch.is_tensor(v) and v.dim() > 0 and v.shape[0] == n}
                rest = {k: v.float().cpu().numpy() for k, v in out.items() if torch.is_tensor(v) and k not in ten}
                ten.update(bbox_center=batch["bbox_center"][:, 0], bbox_scale=batch["bbox_scale"][:, 0], affine_trans=batch["affine_trans"][:, 0])
                # Pack bytes in the same single transfer: casting the two uint8 part maps to float32 here would
                # quadruple their payload. Float outputs keep their exact bits, including non-contiguous tensors.
                byte_pack = any(v.dtype == torch.uint8 for v in ten.values())
                packed = [v.reshape(n, -1) if v.dtype == torch.uint8 else
                          v.float().reshape(n, -1).contiguous().view(torch.uint8) for v in ten.values()] if byte_pack else \
                         [v.float().reshape(n, -1) for v in ten.values()]
                flat = torch.cat(packed, 1).cpu().numpy()
                out, o = rest, 0
                for k, v in ten.items():
                    dtype = np.dtype(np.uint8 if v.dtype == torch.uint8 else np.float32)
                    m = v[0].numel() * (dtype.itemsize if byte_pack else 1)
                    out[k] = flat[:, o:o + m].view(dtype).reshape(v.shape)
                    o += m
            else:
                out = {
                    k: (v if v.dtype == torch.uint8 else v.float()).cpu().numpy()
                    for k, v in out.items()
                    if torch.is_tensor(v)
                }
                out["bbox_center"] = batch["bbox_center"][:, 0].cpu().numpy()
                out["bbox_scale"] = batch["bbox_scale"][:, 0].cpu().numpy()
                out["affine_trans"] = batch["affine_trans"][:, 0].cpu().numpy()
            outputs.append(out)
        return {k: np.concatenate([o[k] for o in outputs]) for k in outputs[0]}

    def _predicted_box(self, out, i):
        """Right-hand box predicted by the hand decoder, as a square xyxy box in the
        (possibly flipped) full image."""
        cx, cy, bw, bh = out["hand_box"][i, 1] * self.input_size[0]
        side = max(bw, bh)
        A = out["affine_trans"][i]
        s = A[0, 0]
        cx, cy, side = (cx - A[0, 2]) / s, (cy - A[1, 2]) / s, side / s
        return np.array(
            [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2],
            dtype=np.float32,
        )

    # ------------------------------------------------------------------ main API
    @torch.no_grad()
    @frame_cache()
    def process_image(self, img_bgr, boxes=None, is_right=None, cam_int=None, scores=None):
        """
        Args:
            img_bgr: HxWx3 uint8 BGR image
            boxes: optional (N, 4) xyxy hand boxes; runs the detector if None
            is_right: (N,) bool handedness, required with boxes
            cam_int: optional 3x3 intrinsics; defaults to SAM 3D Body's default
        Returns:
            list of per-hand dicts (camera coordinates, meters; see README)
        """
        height, width = img_bgr.shape[:2]
        if boxes is None:
            assert self.detector is not None, "No detector: pass boxes and is_right"
            boxes, scores, is_right = self.detector(img_bgr)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if len(boxes) == 0:
            return []
        if is_right is None:
            raise ValueError("is_right (per-box handedness) is required when boxes are given")
        is_right = np.asarray(is_right, dtype=bool).reshape(-1)
        if len(is_right) != len(boxes):
            raise ValueError(f"{len(boxes)} boxes but {len(is_right)} is_right flags")
        if scores is None:
            scores = np.ones(len(boxes), dtype=np.float32)

        if cam_int is None:
            cam_int = default_cam_int(height, width)
        cam_int = np.asarray(cam_int, dtype=np.float32)
        cam_int_flip = cam_int.copy()
        cam_int_flip[0, 2] = width - 1 - cam_int[0, 2]

        boxes_in = np.where(is_right[:, None], boxes, flip_boxes(boxes, width))
        views = []  # perspective crops: (R, K_v, H) per box, in the (mirrored) crop frame

        def crops_for(bxs):
            if self.persp_crop:
                views[:] = [perspective_view(cam_int if r else cam_int_flip, b) for b, r in zip(bxs, is_right)]
                return [(img_bgr, warp_box(H, b), Kv.astype(np.float32), not r, H) for (_, Kv, H), b, r in zip(views, bxs, is_right)]
            return [
                (img_bgr, b, cam_int, False) if r else (img_bgr, b, cam_int_flip, True)
                for b, r in zip(bxs, is_right)
            ]

        out = self._run_crops(crops_for(boxes_in))
        for _ in range(self.refine_box_iters):
            boxes_in = np.stack([self._predicted_box(out, i) for i in range(len(boxes))])
            if self.persp_crop:  # predicted boxes are in the virtual frames
                boxes_in = np.stack([warp_box(np.linalg.inv(v[2]), b) for v, b in zip(views, boxes_in)])
            out = self._run_crops(crops_for(boxes_in))
        if self.recrop_border > 0 and "pt_uv" in out and "pred_keypoints_3d" not in out and not self.persp_crop:
            boxes_in = boxes_in.copy()
            redo = []
            for i in range(len(boxes)):  # joints at the crop border (inside the image): crop cut the hand
                uv = out["pt_uv"][i, :21]
                m = self.recrop_border
                if not ((uv < m) | (uv > 1 - m)).any():
                    continue
                A = out["affine_trans"][i]
                p = (uv * self.input_size[0] - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
                lo, hi = (0 - A[:, 2]) / A[0, 0], (self.input_size[0] - A[:, 2]) / A[0, 0]
                inner = ((uv < m) & (lo > 0)[None]) | ((uv > 1 - m) & (hi < [width, height])[None])
                if not inner.any():
                    continue
                ext = 0.25 * self.input_size[0] / A[0, 0]  # the cut points sit on the edge: push the touched sides out
                lo_t, hi_t = ((uv < m) & (lo > 0)[None]).any(0), ((uv > 1 - m) & (hi < [width, height])[None]).any(0)
                b = boxes_in[i]
                boxes_in[i] = [*np.minimum(b[:2], p.min(0) - ext * lo_t), *np.maximum(b[2:], p.max(0) + ext * hi_t)]
                redo.append(i)
            self.recrop_stats = [getattr(self, "recrop_stats", [0, 0])[0] + len(boxes), getattr(self, "recrop_stats", [0, 0])[1] + len(redo)]
            if redo:
                o2 = self._run_crops([crops_for(boxes_in)[i] for i in redo])
                for k in out:
                    if k in o2 and len(out[k]) == len(boxes):
                        if k == "verify_logits":  # side / presence from the more confident of the two passes
                            v = out[k][redo].copy()
                            keep = np.abs(o2[k][:, 1]) > np.abs(v[:, 1])
                            v[keep] = o2[k][keep]
                            out[k][redo] = v
                        else:
                            out[k][redo] = o2[k]

        if not self.persp_crop:
            return self._results(out, boxes, boxes_in, is_right, scores, cam_int, height, width)
        F = np.diag([-1.0, 1.0, 1.0])
        P = np.array([[-1.0, 0, width - 1], [0, 1, 0], [0, 0, 1]])
        results = []
        for i, (R, Kv, H) in enumerate(views):  # per hand in its virtual camera, then back to the real one
            oi = {k: v[i:i + 1] for k, v in out.items() if isinstance(v, np.ndarray) and len(v) == len(boxes)}
            crop_box = warp_box(H, boxes_in[i])[None]  # crop frame, virtual camera
            if not is_right[i]:  # crop-frame (mirrored) view -> the unmirrored frame
                R, Kv, H = F @ R @ F, P @ Kv @ F, P @ H @ P
            h = self._results(oi, boxes[i:i + 1], crop_box, is_right[i:i + 1], scores[i:i + 1], Kv.astype(np.float32),
                              height, width)
            results += [unwarp_hand(x, R, H, cam_int, height, width) for x in h]
        return results

    def _results(self, out, boxes, boxes_in, is_right, scores, cam_int, height, width):
        """Per-hand result dicts from the model outputs (boxes_in: crop boxes in the crop frame)."""
        if "pred_keypoints_3d" not in out:  # decoder-free keypoint model (finetune --kp_only)
            res = self._kp_only_results(out, boxes, boxes_in, is_right, scores, cam_int, width)
            if ("part_lab" in out or "part_logits" in out) and res:
                self._paste_parts(res, out, is_right, height, width)
            return res
        results = []
        for i in range(len(boxes)):
            right = bool(is_right[i])
            kp3d = out["pred_keypoints_3d"][i][MHR70_TO_OPENPOSE_HAND].copy()
            kp2d = out["pred_keypoints_2d"][i][MHR70_TO_OPENPOSE_HAND].copy()
            verts = out["pred_vertices"][i]
            if verts.shape[0] != len(self.hand_vert_idx):  # full-body mesh (fast_mhr off)
                verts = verts[self.hand_vert_idx]
            verts = verts.copy()
            cam_t = out["pred_cam_t"][i].copy()
            wrist_rot = MHR_TO_CAM @ out["joint_global_rots"][i][MHR_RIGHT_WRIST_JOINT]
            crop_box = boxes_in[i].copy()
            pred_box = self._predicted_box(out, i)  # the decoder's own hand box (detector convention)
            kp3d_mano = None
            if self.mano_regressor is not None:
                kp3d_mano = self.mano_regressor @ verts[self.mano_regressor_vert_idx]
            if not right:
                if kp3d_mano is not None:
                    kp3d_mano[:, 0] *= -1
                kp3d[:, 0] *= -1
                verts[:, 0] *= -1
                cam_t[0] *= -1
                kp2d[:, 0] = width - 1 - kp2d[:, 0]
                wrist_rot = MIRROR_X @ wrist_rot @ MIRROR_X
                crop_box = flip_boxes(crop_box[None], width)[0]
                pred_box = flip_boxes(pred_box[None], width)[0]
            hand = {
                "bbox": boxes[i],
                "crop_box": crop_box,
                "pred_box": pred_box,
                **({"presence": float(1 / (1 + np.exp(-out["verify_logits"][i, 0]))),
                    "side_score": float(1 / (1 + np.exp(-out["verify_logits"][i, 1])))} if "verify_logits" in out else {}),
                "score": float(scores[i]),
                "is_right": right,
                # camera frame (x right, y down, z forward), meters
                "keypoints_3d": kp3d + cam_t,
                "keypoints_3d_rel": kp3d,  # relative to cam_t (wrist-ish origin)
                "vertices": verts + cam_t,
                "faces": self.faces if right else self.faces_left,
                "keypoints_2d": kp2d,
                "cam_t": cam_t,
                "focal_length": float(cam_int[0, 0]),
                "cam_int": cam_int,
                "wrist_rot": wrist_rot,
                # MHR parameters (left hands are in the mirrored right-hand frame)
                "hand_pose": out["hand"][i][54:].copy(),
                "scale": out["scale"][i].copy(),
                "shape": out["shape"][i].copy(),
            }
            if kp3d_mano is not None:
                hand["keypoints_3d_mano"] = kp3d_mano + cam_t
                hand["keypoints_2d_mano"] = project(kp3d_mano + cam_t, cam_int)
            if "pt_uv" in out:  # direct keypoint head: joints then dense surface points (crop of the mirrored frame)
                A = out["affine_trans"][i]
                uv = (out["pt_uv"][i] * self.input_size[0] - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
                xyz = out["pt_xyz"][i].copy()  # root-relative (m), camera axes
                if not right:
                    uv[:, 0] = width - 1 - uv[:, 0]
                    xyz[:, 0] *= -1
                hand["keypoints_2d_head"], hand["surface_2d_head"] = uv[:21], uv[21:]
                hand["keypoints_3d_head_rel"], hand["surface_3d_head_rel"] = xyz[:21], xyz[21:]
            results.append(hand)
        if "mask_logits" in out and results:
            self._paste_masks(results, out, is_right, height, width)
        return results

    def _refine_mesh_to_head(self, batch, out, lr=0.03, w3d=0.1, prior=0.01):
        """Adjust the mesh so its MANO joints reproject onto the keypoint head's 2D (crop pixels), with a weak
        pull towards the head's root-relative 3D and a prior on the parameter change; camera translation fixed."""
        head = self.model.head_pose_hand
        n_pre = 6 + head.body_cont_dim + head.num_shape_comps + head.num_scale_comps
        pred0 = torch.cat([out["pred_pose_raw"], out["shape"], out["scale"], out["hand"], out["face"]], 1).detach().float()
        mask = torch.zeros_like(pred0[:1])
        mask[:, :6] = 1  # global rotation (6d)
        mask[:, n_pre:n_pre + 2 * head.num_hand_comps] = 1  # finger articulation
        W = torch.as_tensor(self.mano_regressor, dtype=torch.float32, device=pred0.device)
        idx = torch.as_tensor(self.mano_regressor_vert_idx, device=pred0.device)
        S = float(self.input_size[0])
        # the model's per-batch bookkeeping is stale when the forward was replayed from a CUDA graph
        self.model._batch_size, self.model._max_num_person = pred0.shape[0], 1
        K = self.model._per_crop_cam_int(batch).float()
        A = batch["affine_trans"][:, 0].float()
        cam_t = out["pred_cam_t"].detach().float()
        tgt = out["pt_uv"][:, :21].detach().float() * S
        tgt3 = out["pt_xyz"][:, :21].detach().float()

        def joints(pred):
            po = head.forward_from_pred(pred, return_vertices=True)
            j = torch.einsum("jv,bvc->bjc", W, po["pred_vertices"][:, idx]) + cam_t[:, None]
            uv = j[..., :2] / j[..., 2:].clamp(min=1e-4) * torch.stack([K[:, 0, 0], K[:, 1, 1]], 1)[:, None] + K[:, :2, 2][:, None]
            return po, j, uv @ A[:, :, :2].mT + A[:, :, 2][:, None]

        delta = torch.zeros_like(pred0, requires_grad=True)
        opt = torch.optim.Adam([delta], lr=lr)
        with torch.enable_grad():
            for _ in range(self.refine_mesh_iters):
                _, j, uv = joints(pred0 + delta * mask)
                rel = j - j[:, :1]
                loss = (uv - tgt).norm(dim=-1).mean() / S * 100 + w3d * (rel - tgt3).abs().sum(-1).mean() * 100 \
                    + prior * (delta * mask).pow(2).sum(-1).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            po, _, _ = joints(pred0 + delta * mask)
            po["pred_cam"] = out["pred_cam"]
            po = self.model.camera_project_hand(po, batch)
            for k in ("pred_vertices", "pred_keypoints_3d", "pred_keypoints_2d", "joint_global_rots", "hand", "global_rot",
                      "pred_cam_t"):
                if k in po and po[k] is not None:
                    out[k] = po[k]
        return out

    def _kp_only_results(self, out, boxes, boxes_in, is_right, scores, cam_int, width):
        """Hands from the direct keypoint head alone: 2D from the heatmaps, 3D = root-relative head joints placed
        by the camera translation that best reprojects them onto the 2D (linear least squares; no mesh)."""
        results = []
        fx, fy, cx, cy = cam_int[0, 0], cam_int[1, 1], cam_int[0, 2], cam_int[1, 2]
        for i in range(len(boxes)):
            right = bool(is_right[i])
            A = out["affine_trans"][i]
            uv = (out["pt_uv"][i] * self.input_size[0] - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
            xyz = out["pt_xyz"][i].astype(np.float64)
            if not right:
                uv[:, 0] = width - 1 - uv[:, 0]
                xyz[:, 0] *= -1
            nb, nv = (int(v) for v in out["kp_dense"][i]) if "kp_dense" in out else (0, 0)
            sel = slice(0, xyz.shape[0]) if nv else slice(0, 21)  # dense layout: every point constrains the translation
            j, u = xyz[sel], uv[sel] - [cx, cy]
            if "pt_depth" in out:  # learned root depth (normalised by the crop focal / size); x, y from the 2D points
                tz = float(np.exp(out["pt_depth"][i]) * fx * A[0, 0] / self.input_size[0])
                # At fixed depth the x/y columns are disjoint constants, so their
                # least-squares solution is the mean translation along each axis.
                xy = (u * (j[:, 2:3] + tz) / [fx, fy] - j[:, :2]).mean(0)
                t = np.array([*xy, tz])
            else:
                # fx (X + tx) - u (Z + tz) = 0, fy (Y + ty) - v (Z + tz) = 0 -> M t = b
                n = len(j)
                M = np.zeros((2 * n, 3))
                M[:n, 0], M[:n, 2] = fx, -u[:, 0]
                M[n:, 1], M[n:, 2] = fy, -u[:, 1]
                rhs = np.concatenate([u[:, 0] * j[:, 2] - fx * j[:, 0], u[:, 1] * j[:, 2] - fy * j[:, 1]])
                t = np.linalg.lstsq(M, rhs, rcond=None)[0]
            j = xyz[:21]
            if self.lift_2d:  # the surface / bone points follow the lifted skeleton
                Q, pts = lift_to_2d(j + t, uv[:21], (fx, fy, cx, cy), xyz[21:] + t)
                j, xyz = Q - t, np.concatenate([Q, pts]) - t
            extra = {"vertices_mano": (xyz[21 + nb:21 + nb + nv] + t).astype(np.float32)} if nv else {}
            vidx = getattr(self.model.keypoint_head, "dense_vidx", None) if nv and nv < 778 else None
            if vidx is not None:  # subset of the template vertices: full surface from a weighted fit of the spectral basis
                from .models.heads.spectral import smooth_surface, template_basis

                if getattr(self, "sub_vidx", None) is None:
                    self.sub_vidx = vidx.cpu().numpy().copy()
                vi = self.sub_vidx
                if getattr(self, "sub_R", None) is None:  # the weighted basis fit is linear in the kept points: one (778, n) map
                    U, M = template_basis(os.environ.get("MANO_DIR", "data/mano"), min(256, nv // 2))
                    U, m = U.astype(np.float64), M.astype(np.float64)[vi]
                    UW = U[vi].T * m[None]
                    self.sub_R = (U @ np.linalg.solve(UW @ U[vi] + 1e-3 * np.eye(U.shape[1]), UW)).astype(np.float32)
                x = extra["vertices_mano"]
                c3 = x.mean(0, keepdims=True)
                extra["vertices_mano"] = (self.sub_R @ (x - c3) + c3).astype(np.float32)
                extra["surface_vidx"] = vi.copy()
            if "mano_verts" in out:  # parametric MANO mesh (learned IK from the 2D points), placed at the dense root
                mv = out["mano_verts"][i].astype(np.float64)
                if not right:
                    mv[:, 0] *= -1
                dense = {k: extra[k] for k in ("surface_vidx",) if k in extra}
                if "vertices_mano" in extra:  # the dense surface stays available (pixel-aligned rendering / masks)
                    dense["vertices_dense"] = extra["vertices_mano"]
                extra = {"vertices_mano": (mv + j[0] + t).astype(np.float32), "mesh_param": True, **dense}
            if "contact_logits" in out and nv:  # hand-object contact per dense point -> the 778 template vertices (same basis map)
                pc = 1 / (1 + np.exp(-out["contact_logits"][i].astype(np.float64)))
                extra["contact_dense"] = pc.astype(np.float32)
                extra["contact"] = (np.clip(self.sub_R @ pc, 0, 1) if vidx is not None else pc).astype(np.float32)
                extra["contact_score"] = float(1 / (1 + np.exp(-out["contact_hand"][i])))
                if getattr(self, "contact_groups", None) is None:  # palm, thumb, index, middle, ring, pinky (MANO skinning argmax)
                    import pickle

                    w = np.asarray(pickle.load(open(os.path.join(os.environ.get("MANO_DIR", "data/mano"), "MANO_RIGHT.pkl"), "rb"),
                                               encoding="latin1")["weights"]).argmax(1)
                    g = {0: 0, 13: 1, 14: 1, 15: 1, 1: 2, 2: 2, 3: 2, 4: 3, 5: 3, 6: 3, 10: 4, 11: 4, 12: 4, 7: 5, 8: 5, 9: 5}
                    self.contact_groups = np.array([g[k] for k in w])
                extra["contact_fingers"] = np.array([extra["contact"][self.contact_groups == k].max() for k in range(6)], np.float32)
                if "contact_logits_k" in out:  # contact at several tolerances (contact_dists mm), monotone in the tolerance
                    pk = 1 / (1 + np.exp(-out["contact_logits_k"][i].astype(np.float64)))
                    pk = np.maximum.accumulate(np.clip(self.sub_R @ pk, 0, 1) if vidx is not None else pk, axis=1)
                    extra["contact_k"] = pk.astype(np.float32)
                    extra["contact_score_k"] = np.maximum.accumulate(1 / (1 + np.exp(-out["contact_hand_k"][i].astype(np.float64)))).astype(np.float32)
            if self.surface_smooth and nv and vidx is None and not extra.get("mesh_param"):  # non-parametric low-pass of the dense surface
                from .models.heads.spectral import smooth_surface, template_basis

                if self.spec_basis is None:
                    self.spec_basis = template_basis(os.environ.get("MANO_DIR", "data/mano"), self.surface_smooth)
                v3 = torch.as_tensor(extra["vertices_mano"])[None]
                c3 = v3.mean(1, keepdim=True)
                extra["vertices_mano"] = (smooth_surface(v3 - c3, self.spec_basis) + c3)[0].numpy()
                s2 = torch.as_tensor(uv[21 + nb:21 + nb + nv], dtype=torch.float32)[None]
                w2 = torch.as_tensor(1 / (1 + np.exp(-out["pt_vis"][i, 21 + nb:21 + nb + nv])))[None] + 0.05 if "pt_vis" in out else None
                c2 = s2.mean(1, keepdim=True)
                uv[21 + nb:21 + nb + nv] = (smooth_surface(s2 - c2, self.spec_basis, w2) + c2)[0].numpy()
            if "pt_vis" in out and nv:  # hidden points: 2D from the projected mesh / 3D points instead of the heatmap
                hid = out["pt_vis"][i, 21:] < 0
                dv = extra.get("vertices_dense", extra["vertices_mano"])
                vm = dv if vidx is None else dv[extra["surface_vidx"]]
                pv = np.concatenate([xyz[21:21 + nb] + t, vm.astype(np.float64)])
                proj = np.stack([fx * pv[:, 0] / pv[:, 2] + cx, fy * pv[:, 1] / pv[:, 2] + cy], 1)
                uv[21:][hid] = proj[hid]
                extra["surface_visible"] = ~hid
            if nv and "pt_conf" in out and self.surface_conf > 0:  # flat heatmaps (truncated / ambiguous points):
                low = out["pt_conf"][i, 21:] < self.surface_conf  # use the projected 3D surface point instead
                pv = xyz[21:] + t
                proj = np.stack([fx * pv[:, 0] / pv[:, 2] + cx, fy * pv[:, 1] / pv[:, 2] + cy], 1)
                uv[21:][low] = proj[low]
            if "pt_uv_local" in out:
                ul = (out["pt_uv_local"][i] * self.input_size[0] - A[:, 2]) @ np.linalg.inv(A[:, :2]).T
                if not right:
                    ul[:, 0] = width - 1 - ul[:, 0]
                far = np.linalg.norm(out["pt_uv_local"][i, 21:] - out["pt_uv"][i, 21:], axis=-1) > self.local_gate
                uv[21:][far] = ul[21:][far]
            lo, hi = uv.min(0), uv.max(0)  # next-frame box (video detection schedule): square around all predicted points
            c, side = (lo + hi) / 2, (hi - lo).max()
            results.append({
                **extra,
                "pred_box": np.array([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2], np.float32),
                **({"token": out["pt_token"][i].astype(np.float16)} if "pt_token" in out else {}),
                "bbox": boxes[i], "crop_box": boxes_in[i] if right else flip_boxes(boxes_in[i][None], width)[0],
                "score": float(scores[i]), "is_right": right, "cam_int": cam_int, "cam_t": t.astype(np.float32),
                **({"presence": float(1 / (1 + np.exp(-out["verify_logits"][i, 0]))),
                    "side_score": float(1 / (1 + np.exp(-out["verify_logits"][i, 1])))} if "verify_logits" in out else {}),
                "keypoints_2d": uv[:21].astype(np.float32), "keypoints_2d_head": uv[:21].astype(np.float32),
                "surface_2d_head": uv[21:].astype(np.float32),
                "keypoints_3d": (j + t).astype(np.float32), "keypoints_3d_mano": (j + t).astype(np.float32),
                "keypoints_3d_head_rel": j.astype(np.float32), "surface_3d_head_rel": xyz[21:].astype(np.float32),
                "focal_length": float(fx),
            })
        return results

    def _paste_parts(self, results, out, is_right, height, width):
        """Per-hand part segmentation in the frame (0 background, 1 palm, 2-16 finger segments, train.part_masks): the
        argmax of each crop's part logits warped back through the inverse crop affine (mirrored for left hands) over
        the crop's footprint -> hand["parts_roi"] = (x0, y0, uint8 labels)."""
        S = self.input_size[0]
        F = np.array([[-1.0, 0, width - 1], [0, 1, 0], [0, 0, 1]])
        for i, h in enumerate(results):
            if "part_lab" in out:  # made on the device (_run_crops)
                lab, fg = out["part_lab"][i].astype(np.uint8), out["part_fg"][i].astype(np.uint8)
            else:
                lg = cv2.resize(np.ascontiguousarray(out["part_logits"][i].transpose(1, 2, 0).astype(np.float32)), (S, S),
                                interpolation=cv2.INTER_LINEAR)  # logits upsampled, then argmax
                lab = lg.argmax(-1).astype(np.uint8)
                fg = (lg[..., 1:].argmax(-1) + 1).astype(np.uint8)  # the best hand part everywhere (for a part map inside a given mask)
            A = np.vstack([out["affine_trans"][i].astype(np.float64), [0, 0, 1]])
            if not is_right[i]:
                A = A @ F
            Ainv = np.linalg.inv(A)
            corners = np.array([[0, 0, 1], [S, 0, 1], [0, S, 1], [S, S, 1]], np.float64) @ Ainv[:2].T
            x0, y0 = np.floor(corners.min(0)).astype(int).clip(0, [width, height])
            x1, y1 = np.ceil(corners.max(0)).astype(int).clip(0, [width, height])
            # The detector mask can extend beyond the pose crop. Keep a part
            # assignment available over its entire box, including those pixels.
            bx0, by0, bx1, by1 = np.asarray(h["bbox"], np.float64)
            x0, y0 = np.minimum([x0, y0], np.floor([bx0, by0])).astype(int).clip(0, [width, height])
            x1, y1 = np.maximum([x1, y1], np.ceil([bx1, by1])).astype(int).clip(0, [width, height])
            if x1 <= x0 or y1 <= y0:
                continue
            T = np.array([[1.0, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ Ainv  # crop -> ROI pixels
            h["parts_roi"] = (int(x0), int(y0), cv2.warpAffine(lab, T[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST, borderValue=0))
            h["parts_fg_roi"] = (int(x0), int(y0), cv2.warpAffine(fg, T[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_NEAREST,
                                                                borderMode=cv2.BORDER_REPLICATE))

    def _paste_masks(self, results, out, is_right, height, width, thresh=0.5):
        """Per-hand instance masks in the full frame ("mask" = hand, "mask_arm" = hand + visible arm): each
        crop's mask probability is warped back through the inverse crop affine (mirrored for left hands),
        only over the crop's footprint; overlapping hands are resolved by the per-pixel argmax."""
        logits = out["mask_logits"]
        if logits.ndim == 3:
            logits = logits[:, None]
        for ch, key in enumerate(["mask", "mask_arm"][: logits.shape[1]]):
            self._paste_channel(results, logits[:, ch], out, is_right, height, width, thresh, key)

    def _paste_channel(self, results, logits, out, is_right, height, width, thresh, key):
        S = self.input_size[0]
        F = np.array([[-1.0, 0, width - 1], [0, 1, 0], [0, 0, 1]])
        probs = []
        for i in range(len(results)):
            p = 1 / (1 + np.exp(-logits[i].astype(np.float32)))
            p = cv2.resize(p, (S, S), interpolation=cv2.INTER_LINEAR)
            A = np.vstack([out["affine_trans"][i].astype(np.float64), [0, 0, 1]])
            if not is_right[i]:
                A = A @ F
            Ainv = np.linalg.inv(A)
            corners = np.array([[0, 0, 1], [S, 0, 1], [0, S, 1], [S, S, 1]], np.float64) @ Ainv[:2].T
            x0, y0 = np.floor(corners.min(0)).astype(int).clip(0, [width, height])
            x1, y1 = np.ceil(corners.max(0)).astype(int).clip(0, [width, height])
            if x1 <= x0 or y1 <= y0:
                probs.append(None)
                continue
            T = np.array([[1.0, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ Ainv  # crop -> ROI pixels
            roi = cv2.warpAffine(p, T[:2], (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR, borderValue=0)
            probs.append((x0, y0, roi))
        # ROI-only instance resolution: a pixel goes to the hand with the higher probability (lower index on
        # ties), compared only where two hands' ROIs overlap -- no full-frame float buffers
        for k, h in enumerate(results):
            m = np.zeros((height, width), bool)
            if probs[k] is not None:
                x0, y0, pk = probs[k]
                keep = pk > thresh
                for j, pj in enumerate(probs):
                    if j == k or pj is None:
                        continue
                    a0, b0, pr = pj
                    ix0, iy0 = max(x0, a0), max(y0, b0)
                    ix1, iy1 = min(x0 + pk.shape[1], a0 + pr.shape[1]), min(y0 + pk.shape[0], b0 + pr.shape[0])
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    sk = pk[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
                    sj = pr[iy0 - b0:iy1 - b0, ix0 - a0:ix1 - a0]
                    keep[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] &= (sk > sj) if j < k else (sk >= sj)
                m[y0:y0 + pk.shape[0], x0:x0 + pk.shape[1]] = keep
            h[key] = m
