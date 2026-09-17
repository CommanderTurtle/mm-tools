import argparse
import json
import os
import pickle
import sys
import time
from typing import List, Optional

import numpy as np
import open3d as o3d
import torch
import trimesh

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from utils.det_seg import (
    assign_point_instances_with_background,
    create_bbox_mesh_frame,
    foreground_winner_indices,
    get_dino_model,
    nms_process,
    process_pred_poses_w_seg,
    remove_unreasonable_preds_by_scale,
    visualize_results,
)
from utils.discrete import continue_transform
from utils.load_model import (
    load_config,
    load_model_obj_pose_w_seg_voxelize,
)

try:
    from eval.perception.det_seg_eval_utils import calculate_mAP_and_mIoU
except ModuleNotFoundError:
    from det_seg_eval_utils import calculate_mAP_and_mIoU

DINO_IMAGE_DOWNSAMPLE = 16


_ANYUP_UPSAMPLER = {}
_SR_MODEL = {}


def _super_resolve(args, rgbs):
    """Super-resolve (N, H, W, 3) float RGB in [0,1] by `--rgb_upsample`.

    Swin2SR is run tiled: the classical-SR checkpoint is trained on small
    patches and its attention cost grows with area, so a 1296x968 frame goes
    through in tiles with overlap, blended back with a linear ramp to avoid
    visible seams at the joins.
    """

    import numpy as np
    import torch
    import torch.nn.functional as F

    factor = max(1, int(getattr(args, "rgb_upsample", 1) or 1))
    if factor == 1:
        return rgbs
    frames = torch.from_numpy(np.ascontiguousarray(rgbs)).permute(0, 3, 1, 2).float()

    if getattr(args, "rgb_upsampler", "swin2sr") == "bicubic":
        out = F.interpolate(frames, scale_factor=factor, mode="bicubic",
                            align_corners=False).clamp(0, 1)
        return out.permute(0, 2, 3, 1).contiguous().numpy()

    from transformers import Swin2SRForImageSuperResolution

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = f"caidas/swin2SR-classical-sr-x{factor}-64"
    if name not in _SR_MODEL:
        _SR_MODEL[name] = (
            Swin2SRForImageSuperResolution.from_pretrained(name).eval().to(device)
        )
    model = _SR_MODEL[name]

    tile, overlap = 256, 32
    outputs = []
    with torch.no_grad():
        for frame in frames:
            _, height, width = frame.shape
            canvas = torch.zeros(3, height * factor, width * factor, device=device)
            weight = torch.zeros(1, height * factor, width * factor, device=device)
            for top in range(0, height, tile - overlap):
                for left in range(0, width, tile - overlap):
                    bottom, right = min(top + tile, height), min(left + tile, width)
                    patch = frame[:, top:bottom, left:right].unsqueeze(0).to(device)
                    up = model(pixel_values=patch).reconstruction.clamp(0, 1)[0]
                    ramp_h = torch.ones(up.shape[1], device=device)
                    ramp_w = torch.ones(up.shape[2], device=device)
                    blend = factor * overlap
                    if top > 0:
                        ramp_h[:blend] = torch.linspace(0, 1, blend, device=device)
                    if left > 0:
                        ramp_w[:blend] = torch.linspace(0, 1, blend, device=device)
                    mask = ramp_h[:, None] * ramp_w[None, :]
                    ys, xs = top * factor, left * factor
                    canvas[:, ys:ys + up.shape[1], xs:xs + up.shape[2]] += up * mask
                    weight[:, ys:ys + up.shape[1], xs:xs + up.shape[2]] += mask
            outputs.append((canvas / weight.clamp_min(1e-6)).cpu())
    result = torch.stack(outputs).permute(0, 2, 3, 1).contiguous().numpy()
    print(f"[rgb-upsample] {getattr(args, 'rgb_upsampler', 'swin2sr')} x{factor}: "
          f"{rgbs.shape[1]}x{rgbs.shape[2]} -> {result.shape[1]}x{result.shape[2]}",
          flush=True)
    return result


def _tiled_patch_tokens(args, dino_model, dino_transform_fn, rgbs):
    """DINO patch tokens over a super-resolved frame, tiled at the native size.

    Running the ViT on the whole 4x image would mean ~78 k tokens in one pass,
    which is quadratic-attention infeasible. Splitting it into `factor x factor`
    tiles of the ORIGINAL frame size keeps every forward pass exactly the shape
    the backbone normally sees, and the tiles' patch grids concatenate into the
    dense lattice without overlap or interpolation.
    """

    import torch

    factor = max(1, int(getattr(args, "rgb_upsample", 1) or 1))
    frames, height, width = rgbs.shape[0], rgbs.shape[1], rgbs.shape[2]
    # Split on PATCH-aligned boundaries, not at height/factor: 3872/4 = 968 is
    # not a multiple of 16, so equal-pixel tiles would each drop half a patch
    # row and the lattice would come out 240 instead of 242, desynchronising
    # features from points. Dividing the patch counts instead keeps the total
    # exact and every tile a whole number of patches.
    def spans(total_pixels):
        patches = total_pixels // DINO_IMAGE_DOWNSAMPLE
        base, extra = divmod(patches, factor)
        edges, start = [], 0
        for index in range(factor):
            count = base + (1 if index < extra else 0)
            edges.append((start, start + count * DINO_IMAGE_DOWNSAMPLE))
            start += count * DINO_IMAGE_DOWNSAMPLE
        return edges

    row_spans, col_spans = spans(height), spans(width)
    rows = []
    for top, bottom in row_spans:
        cols = []
        for left, right in col_spans:
            tile = rgbs[:, top:bottom, left:right, :]
            tensor = dino_transform_fn(tile).to(rgbs_device(dino_model))
            n, _, h, w = tensor.shape
            out = dino_model(tensor, is_training=True)["x_norm_patchtokens"]
            cols.append(out.reshape(n, h // DINO_IMAGE_DOWNSAMPLE,
                                    w // DINO_IMAGE_DOWNSAMPLE, -1))
        rows.append(torch.cat(cols, dim=2))
    grid = torch.cat(rows, dim=1)
    print(f"[rgb-upsample] tiled DINO {factor}x{factor} -> lattice "
          f"{grid.shape[1]}x{grid.shape[2]} per frame", flush=True)
    return grid.reshape(frames * grid.shape[1] * grid.shape[2], grid.shape[-1])


def rgbs_device(module):
    return next(module.parameters()).device


def _anyup_features(args, dino_model, rgbs_tensor):
    """DINO features upsampled by AnyUp, flattened to one row per point.

    Mirrors the reconstruction path (`models/dino_anyup.run_dino_anyup_features`)
    so perception and reconstruction see features built the same way.
    """

    from models.dino_anyup import load_anyup_upsampler, run_dino_anyup_features

    device = rgbs_tensor.device
    key = str(device)
    if key not in _ANYUP_UPSAMPLER:
        _ANYUP_UPSAMPLER[key] = load_anyup_upsampler({}, device=device)
    features = run_dino_anyup_features(
        dino_model=dino_model,
        anyup_upsampler=_ANYUP_UPSAMPLER[key],
        rgbs_tensor=rgbs_tensor,
        dino_downsample=DINO_IMAGE_DOWNSAMPLE,
        dino_upsample=int(args.dino_upsample),
        anyup_frame_batch_size=int(getattr(args, "anyup_frame_batch_size", 1) or 1),
    )
    return features.reshape(-1, features.shape[-1])


def _subsample_feature_lattice(args, points_np, feats, rgbs):
    """Keep every K-th cell of the feature lattice, in points and features alike.

    Both tensors are laid out as (frames, H//stride, W//stride) flattened, so the
    stride has to be applied to the same 2D lattice in both or they desynchronise
    -- which is the failure the size assertion downstream would catch anyway.
    """

    keep = max(1, int(getattr(args, "feature_subsample", 1) or 1))
    if keep == 1:
        return points_np, feats
    stride = perception_feature_stride(args)
    frames = len(rgbs)
    height, width = rgbs.shape[1], rgbs.shape[2]
    lattice_h, lattice_w = height // stride, width // stride
    expected = frames * lattice_h * lattice_w
    if points_np.shape[0] != expected or feats.shape[0] != expected:
        raise ValueError(
            f"--feature_subsample expects a {frames}x{lattice_h}x{lattice_w} "
            f"lattice ({expected} cells), got points={points_np.shape[0]} "
            f"feats={feats.shape[0]}"
        )
    index = np.arange(expected).reshape(frames, lattice_h, lattice_w)
    index = index[:, ::keep, ::keep].reshape(-1)
    print(
        f"[feature-subsample] {expected} -> {index.shape[0]} cells "
        f"(every {keep}th on a {lattice_h}x{lattice_w} lattice)",
        flush=True,
    )
    return points_np[index], feats[torch.from_numpy(index).to(feats.device)]


def perception_feature_stride(args) -> int:
    """Pixels per conditioning point == pixels per DINO feature.

    `forward_inference` consumes one feature vector per point, so the point
    cloud stride is not an independent knob: raw ViT patch tokens are one per
    16x16 pixels, and AnyUp upsampling by `dino_upsample` divides that.
    """

    upsample = max(1, int(getattr(args, "dino_upsample", 1) or 1))
    rgb_upsample = max(1, int(getattr(args, "rgb_upsample", 1) or 1))
    total = upsample * rgb_upsample
    if DINO_IMAGE_DOWNSAMPLE % total:
        raise ValueError(
            f"--dino_upsample x --rgb_upsample must divide "
            f"{DINO_IMAGE_DOWNSAMPLE}, got {upsample} x {rgb_upsample}"
        )
    return DINO_IMAGE_DOWNSAMPLE // total
OBB_JSON_NAME = "oriented_bboxes.json"
RAW_OBB_JSON_NAME = "raw_oriented_bboxes.json"
GT_OBB_JSON_NAME = "gt_oriented_bboxes.json"
RAW_GT_OBB_JSON_NAME = "raw_gt_oriented_bboxes.json"
POINT_INSTANCE_PKL_NAME = "point_instance_masks.pkl"
DET_SEG_METRICS_JSON_NAME = "det_seg_metrics.json"
NETWORK_TIMING_JSON_NAME = "network_timing.json"


def timed_cuda_network_call(function):
    """Measure only queued CUDA work for one neural forward call."""
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = function()
        end.record()
        end.synchronize()
        return result, float(start.elapsed_time(end) / 1000.0)
    started = time.perf_counter()
    result = function()
    return result, float(time.perf_counter() - started)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir_seg", type=str, required=True)
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=(
            "Optional explicit det/seg config. Defaults to "
            "<run_dir_seg>/config.yaml for backward compatibility."
        ),
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional explicit det/seg checkpoint. Defaults to the latest .pt "
            "under <run_dir_seg>/checkpoints for backward compatibility."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Optional exact output directory. When omitted, the legacy "
            "run_dir_seg/eval_dir_name or run_dir_seg/inference_<dataset> "
            "layout is retained."
        ),
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip val_<index> scenes that already contain det_seg_metrics.json.",
    )
    parser.add_argument(
        "--dino_repo_dir",
        type=str,
        default=None,
        help="Optional DINOv3 torch.hub repository override.",
    )
    parser.add_argument(
        "--dino_model_path",
        type=str,
        default=None,
        help="Optional DINOv3 weights override.",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default=None,
        help="Optional iTHOR or Imaginarium dataset-root override.",
    )
    parser.add_argument(
        "--dino_frame_batch_size",
        type=int,
        default=None,
        help=(
            "Optional number of RGB frames per DINO forward. This is useful "
            "for dense multi-frame AEO inference; omitted preserves the "
            "historical single-batch path."
        ),
    )

    parser.add_argument("--num_eval_scenes", type=int, default=30)
    parser.add_argument("--eval_idx", type=int, default=None)
    parser.add_argument(
        "--eval_indices",
        type=str,
        default=None,
        help="Comma-separated eval indices or ranges, e.g. 0,2,5-8",
    )
    parser.add_argument("--nms_iou_threshold", type=float, default=0.4)
    parser.add_argument("--anyup_frame_batch_size", type=int, default=1)
    parser.add_argument(
        "--rgb_upsample",
        type=int,
        default=1,
        help=(
            "Super-resolve the RGB by this factor before DINO, then run DINO at "
            "its normal patch-16 stride on the enlarged image. Gives a stride "
            "16//rgb_upsample feature lattice in ORIGINAL pixels using native "
            "patch tokens -- the image-space alternative to AnyUp feature "
            "upsampling. DINO is run tiled at the original frame size, so every "
            "tile is the resolution the backbone normally sees."
        ),
    )
    parser.add_argument(
        "--rgb_upsampler",
        choices=("swin2sr", "bicubic"),
        default="swin2sr",
        help="swin2sr = caidas/swin2SR-classical-sr-x4-64; bicubic is the control.",
    )
    parser.add_argument(
        "--feature_subsample",
        type=int,
        default=1,
        help=(
            "After building points and features at stride 16//dino_upsample, "
            "keep every K-th cell on the feature lattice in both dimensions. "
            "K=4 with --dino_upsample 4 restores the stride-16 point density "
            "while keeping AnyUp features, which separates the feature change "
            "from the density change."
        ),
    )
    parser.add_argument(
        "--dino_upsample",
        type=int,
        default=1,
        help=(
            "AnyUp factor for the DINO features, which also sets the "
            "conditioning-cloud stride: the model needs exactly one feature per "
            "point, so stride = 16 // dino_upsample. 1 keeps the raw ViT patch "
            "grid (stride 16, the historical behaviour); 4 gives stride 4 and "
            "16x the points, which matters on single_image where a single view "
            "yields only ~4.9 k points at stride 16. Matches the reconstruction "
            "path's --dino-upsample."
        ),
    )
    parser.add_argument("--valid_score_threshold", type=float, default=0.2)
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="ithor",
        choices=[
            "ithor",
            "scannetpp",
            "imaginarium",
            "single_image",
        ],
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--eval_dir_name", type=str, default=None)
    parser.add_argument("--no_post_process_perception", action="store_true")
    parser.add_argument(
        "--no_clean_instance_labels_by_obbs",
        action="store_true",
        help=(
            "Do not erase predicted instance-mask points outside the predicted "
            "OBB. This is now the default; the flag is kept so existing "
            "run scripts keep working."
        ),
    )
    parser.add_argument(
        "--clean_instance_labels_by_obbs",
        action="store_true",
        help=(
            "Opt back in to erasing predicted instance-mask points outside the "
            "predicted OBB. Off by default: the shipped percept+recon protocol "
            "runs without OBB cleanup."
        ),
    )
    args = parser.parse_args()
    # The shipped protocol runs without OBB cleanup, so the effective switch is
    # the positive opt-in flag. `--no_clean_instance_labels_by_obbs` stays
    # accepted (and remains a no-op now that it matches the default) because
    # roughly twenty run scripts still pass it explicitly.
    args.no_clean_instance_labels_by_obbs = not args.clean_instance_labels_by_obbs
    return args


def get_latest_ckpt(ckpt_dir: str) -> str:
    ckpt_files = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    if len(ckpt_files) == 0:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    ckpt_paths = [os.path.join(ckpt_dir, f) for f in ckpt_files]
    ckpt_paths.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return ckpt_paths[-1]


def get_eval_indices(num_eval_scenes: int, eval_idx: Optional[int], eval_indices: Optional[str]) -> List[int]:
    if eval_indices is not None and eval_idx is not None:
        raise ValueError("Use either --eval_idx or --eval_indices, not both.")

    if eval_indices is not None:
        out = []
        for token in eval_indices.split(","):
            token = token.strip()
            if token == "":
                continue
            if "-" in token:
                parts = token.split("-")
                if len(parts) != 2:
                    raise ValueError(f"Invalid range token in --eval_indices: {token}")
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                if start < 0 or end < 0 or end < start:
                    raise ValueError(f"Invalid range token in --eval_indices: {token}")
                out.extend(list(range(start, end + 1)))
            else:
                idx = int(token)
                if idx < 0:
                    raise ValueError(f"Invalid index in --eval_indices: {token}")
                out.append(idx)
        if len(out) == 0:
            raise ValueError("--eval_indices is empty after parsing.")
        return sorted(list(set(out)))

    if eval_idx is not None:
        if eval_idx < 0:
            raise ValueError("--eval_idx must be >= 0")
        return [eval_idx]

    return list(range(num_eval_scenes))


def get_seg_eval_dir(
    run_dir_seg: str,
    dataset_type: str,
    eval_dir_name: Optional[str],
    output_dir: Optional[str] = None,
) -> str:
    if output_dir is not None:
        return output_dir
    if eval_dir_name is not None:
        return os.path.join(run_dir_seg, eval_dir_name)
    return os.path.join(run_dir_seg, f"inference_{dataset_type}")


def load_pose_inference_data(args):
    if args.dataset_type == "ithor":
        from utils.data_ithor import get_inference_data, load_ithor_data

        data_list = load_ithor_data(data_root=args.dataset_root)
    elif args.dataset_type == "scannetpp":
        from utils.data_scannetpp import get_inference_data, load_scannetpp_data

        data_list = load_scannetpp_data()
    elif args.dataset_type == "imaginarium":
        from utils.data_imaginarium import get_inference_data, load_imaginarium_data

        data_list = load_imaginarium_data(data_root=args.dataset_root)
    elif args.dataset_type == "single_image":
        from utils.data_single_image import (
            get_inference_data,
            load_single_image_data,
        )

        data_list = load_single_image_data()
    else:
        raise ValueError(f"Invalid dataset_type: {args.dataset_type}")

    def get_item(eval_i, image_downsample):
        return get_inference_data(
            data_list,
            eval_i,
            image_downsample=image_downsample,
        )

    return get_item


def quat_wxyz_from_sxyz_angles(angles):
    x, y, z = [float(v) for v in angles]
    cx, sx = np.cos(x * 0.5), np.sin(x * 0.5)
    cy, sy = np.cos(y * 0.5), np.sin(y * 0.5)
    cz, sz = np.cos(z * 0.5), np.sin(z * 0.5)

    quat = np.array(
        [
            cz * cy * cx + sz * sy * sx,
            cz * cy * sx - sz * sy * cx,
            cz * sy * cx + sz * cy * sx,
            sz * cy * cx - cz * sy * sx,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if norm > 0.0:
        quat = quat / norm
    return [float(v) for v in quat]


def quat_wxyz_to_matrix(quat_wxyz):
    quat = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_wxyz(matrix):
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = np.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )

    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    quat = quat / norm
    return [float(v) for v in quat]


def transform_obbs(obbs, transform):
    transform = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    linear = transform[:3, :3]
    transformed_obbs = []

    for obb in obbs:
        transformed_obb = dict(obb)
        translate = np.asarray(obb["translate"], dtype=np.float64).reshape(3)
        rotation = quat_wxyz_to_matrix(obb["rotation"])

        translate_h = np.concatenate([translate, np.array([1.0], dtype=np.float64)])
        transformed_translate = (transform @ translate_h)[:3]

        transformed_rotation = linear @ rotation
        axis_lengths = np.linalg.norm(transformed_rotation, axis=0)
        scale_factor = float(axis_lengths.mean()) if np.all(axis_lengths > 1e-12) else 1.0
        transformed_rotation = transformed_rotation / scale_factor
        u, _, vt = np.linalg.svd(transformed_rotation)
        transformed_rotation = u @ vt
        if np.linalg.det(transformed_rotation) < 0:
            u[:, -1] *= -1
            transformed_rotation = u @ vt

        transformed_obb["translate"] = [float(v) for v in transformed_translate]
        transformed_obb["rotation"] = matrix_to_quat_wxyz(transformed_rotation)
        if "scale" in transformed_obb:
            scale = np.asarray(transformed_obb["scale"], dtype=np.float64)
            transformed_scale = scale * scale_factor
            if transformed_scale.ndim == 0:
                transformed_obb["scale"] = float(transformed_scale)
            else:
                transformed_obb["scale"] = [float(v) for v in transformed_scale.reshape(-1)]
        transformed_obbs.append(transformed_obb)

    return transformed_obbs


def euler_sxyz_from_quat_wxyz(quat_wxyz):
    rotation = quat_wxyz_to_matrix(quat_wxyz)
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    return [float(v) for v in trimesh.transformations.euler_from_matrix(matrix, axes="sxyz")]


def save_obbs_ply(save_path, obbs):
    if not obbs:
        return None

    box_meshes = []
    colors = colors_from_instance_ids(np.arange(1, len(obbs) + 1, dtype=np.int64))
    for idx, obb in enumerate(obbs):
        scale = float(np.asarray(obb["scale"], dtype=np.float64).reshape(-1)[0])
        box_meshes.append(
            create_bbox_mesh_frame(
                obb["translate"],
                euler_sxyz_from_quat_wxyz(obb["rotation"]),
                scale,
                0.01,
                color=colors[idx],
            )
        )

    trimesh.util.concatenate(box_meshes).export(save_path)
    return save_path


def colors_from_instance_ids(instance_ids):
    instance_ids = np.asarray(instance_ids, dtype=np.int64).reshape(-1)
    hashed = instance_ids.astype(np.uint64)
    colors = np.stack(
        [
            ((hashed * 37 + 17) % 255) / 254.0,
            ((hashed * 67 + 71) % 255) / 254.0,
            ((hashed * 97 + 131) % 255) / 254.0,
        ],
        axis=1,
    ).astype(np.float64)
    colors[instance_ids <= 0] = np.array([0.18, 0.18, 0.18], dtype=np.float64)
    return colors


def write_point_cloud(path, points, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    o3d.io.write_point_cloud(path, pcd)


def save_gt_instance_point_cloud(save_dir, points, instance_ids):
    if instance_ids is None:
        return None

    instance_ids = np.asarray(instance_ids).reshape(-1)
    points = np.asarray(points)
    if instance_ids.shape[0] != points.shape[0]:
        raise ValueError(
            f"GT instance ids length {instance_ids.shape[0]} does not match points {points.shape[0]}"
        )

    save_path = os.path.join(save_dir, "gt_instance_points.ply")
    write_point_cloud(save_path, points, colors_from_instance_ids(instance_ids))
    return save_path


def save_raw_visualization_plys(save_dir, raw_transform):
    raw_transform = np.asarray(raw_transform, dtype=np.float64).reshape(4, 4)
    raw_paths = []
    for filename in sorted(os.listdir(save_dir)):
        if not filename.endswith(".ply") or filename.startswith("raw_"):
            continue

        src_path = os.path.join(save_dir, filename)
        dst_path = os.path.join(save_dir, f"raw_{filename}")
        if "boxes" in filename:
            mesh = trimesh.load(src_path, force="mesh", process=False)
            if mesh.is_empty:
                continue
            mesh.apply_transform(raw_transform)
            mesh.export(dst_path)
        else:
            pcd = o3d.io.read_point_cloud(src_path)
            if not pcd.has_points():
                continue
            pcd.transform(raw_transform)
            o3d.io.write_point_cloud(dst_path, pcd)
        raw_paths.append(dst_path)

    return raw_paths


def extract_pred_point_instances_from_results(
    points_np,
    feats_np,
    results_dict,
    valid_score_threshold,
    nms_iou_threshold,
    no_post_process_perception=False,
):
    num_points = points_np.shape[0]
    voxel_inverse_indices = results_dict["voxel_inverse_indices"]
    pos_bin_logits = results_dict["pos_bin_logits"]
    angle_bin_logits = results_dict["angle_bin_logits"]
    scale_bin_logits = results_dict["scale_bin_logits"]
    valid_logits = results_dict["valid_logits"]
    pred_masks_logits = results_dict["pred_masks_logits"]
    seg_feats = results_dict["seg_feats"]
    background_mask = None
    if "background_pred_masks_logits" in results_dict:
        background_mask = (
            results_dict["background_pred_masks_logits"]
            .sigmoid()[:, :, voxel_inverse_indices][0, 0]
            .detach().cpu().numpy()
        )

    pred_masks = pred_masks_logits.sigmoid()
    pred_masks = pred_masks[:, :, voxel_inverse_indices]

    pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, _, seg_feats = process_pred_poses_w_seg(
        pos_bin_logits=pos_bin_logits,
        angle_bin_logits=angle_bin_logits,
        scale_bin_logits=scale_bin_logits,
        valid_logits=valid_logits,
        pred_masks=pred_masks,
        threshold=valid_score_threshold,
        seg_feats=seg_feats,
    )
    if pred_pos_bins is None:
        if background_mask is not None:
            return np.zeros((num_points,), dtype=np.int32)
        return np.full((num_points,), -1, dtype=np.int32)

    pred_pos_bins = pred_pos_bins.cpu().numpy()
    pred_angle_bins = pred_angle_bins.cpu().numpy()
    pred_scale_bins = pred_scale_bins.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    pred_valid_logits = pred_valid_logits.cpu().numpy()
    seg_feats = seg_feats.cpu().numpy()

    if not no_post_process_perception:
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, seg_feats = nms_process(
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            pred_valid_logits,
            points_np,
            feats_np,
            iou_threshold=nms_iou_threshold,
            seg_feats=seg_feats,
        )

        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, _ = remove_unreasonable_preds_by_scale(
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            seg_feats,
            points_np,
        )

    if pred_masks.shape[0] == 0:
        if background_mask is not None:
            return np.zeros((num_points,), dtype=np.int32)
        return np.full((num_points,), -1, dtype=np.int32)
    return assign_point_instances_with_background(
        pred_masks, background_mask
    ).reshape(-1).astype(np.int32)


def save_point_instance_masks_pkl(save_dir, gt_instance_ids, pred_instance_ids):
    pred = np.asarray(pred_instance_ids, dtype=np.int32).reshape(-1)
    gt = (
        None
        if gt_instance_ids is None
        else np.asarray(gt_instance_ids, dtype=np.int32).reshape(-1)
    )
    if gt is not None and gt.shape != pred.shape:
        raise ValueError(f"GT/pred point instance shapes do not match: {gt.shape} vs {pred.shape}")

    save_path = os.path.join(save_dir, POINT_INSTANCE_PKL_NAME)
    with open(save_path, "wb") as f:
        pickle.dump({"pred": pred, "gt": gt}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return save_path


def points_inside_obb(points, obb, eps=1e-8):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    center = np.asarray(obb["translate"], dtype=np.float64).reshape(3)
    rotation = quat_wxyz_to_matrix(obb.get("rotation", [1.0, 0.0, 0.0, 0.0]))
    scale = np.asarray(obb.get("scale", 1.0), dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(float(scale), 3)
    scale = scale.reshape(3)

    local_points = (points - center) @ rotation
    return np.all(np.abs(local_points) <= scale * 0.5 + eps, axis=1)


def clean_instance_labels_by_obbs(points, instance_ids, obbs):
    cleaned = np.asarray(instance_ids, dtype=np.int32).reshape(-1).copy()
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if cleaned.shape[0] != points.shape[0]:
        raise ValueError(
            f"Instance ids length {cleaned.shape[0]} does not match points {points.shape[0]}"
        )

    num_cleaned = 0
    for obb_idx, obb in enumerate(obbs):
        instance_id = int(obb.get("index", obb_idx))
        instance_mask = cleaned == instance_id
        if not np.any(instance_mask):
            continue

        inside = points_inside_obb(points[instance_mask], obb)
        outlier_mask = instance_mask.copy()
        outlier_mask[instance_mask] = ~inside
        num_cleaned += int(outlier_mask.sum())
        cleaned[outlier_mask] = -1

    return cleaned, num_cleaned


def assign_point_instances_from_obbs(
    points,
    obbs,
    background_id=0,
    scale_factor=1.0,
    overlap_mode="confidence",
):
    """Create point-instance labels using only predicted rotated OBB volumes.

    Points outside every OBB receive ``background_id``.  When OBBs overlap,
    ``confidence`` gives the point to the highest-confidence prediction, while
    ``normalized_center`` gives it to the box in which it lies deepest
    (smallest normalized radius), using confidence as a deterministic tie
    breaker.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    scale_factor = float(scale_factor)
    if not np.isfinite(scale_factor) or scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if overlap_mode not in {"confidence", "normalized_center"}:
        raise ValueError(
            "overlap_mode must be 'confidence' or 'normalized_center', "
            f"got {overlap_mode!r}"
        )

    labels = np.full(points.shape[0], int(background_id), dtype=np.int32)
    if not obbs or points.shape[0] == 0:
        return labels

    best_primary = np.full(points.shape[0], -np.inf, dtype=np.float64)
    best_secondary = np.full(points.shape[0], -np.inf, dtype=np.float64)
    for obb_idx, obb in enumerate(obbs):
        center = np.asarray(obb["translate"], dtype=np.float64).reshape(3)
        rotation = quat_wxyz_to_matrix(
            obb.get("rotation", [1.0, 0.0, 0.0, 0.0])
        )
        scale = np.asarray(obb.get("scale", 1.0), dtype=np.float64)
        if scale.ndim == 0:
            scale = np.repeat(float(scale), 3)
        half_extent = np.maximum(
            scale.reshape(3) * scale_factor * 0.5, 1e-12
        )
        local = (points - center) @ rotation
        normalized_radius = np.max(
            np.abs(local) / half_extent.reshape(1, 3), axis=1
        )
        inside = normalized_radius <= 1.0 + 1e-8
        if not np.any(inside):
            continue

        confidence = float(
            obb.get("confidence", obb.get("source_prob", 1.0))
        )
        if overlap_mode == "confidence":
            primary = np.full(points.shape[0], confidence, dtype=np.float64)
            secondary = -normalized_radius
        else:
            primary = -normalized_radius
            secondary = np.full(points.shape[0], confidence, dtype=np.float64)

        better = inside & (
            (primary > best_primary)
            | (
                np.isclose(primary, best_primary, rtol=0.0, atol=1e-12)
                & (secondary > best_secondary)
            )
        )
        if not np.any(better):
            continue
        labels[better] = int(obb.get("index", obb_idx))
        best_primary[better] = primary[better]
        best_secondary[better] = secondary[better]

    return labels


def json_safe_metric_value(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [json_safe_metric_value(v) for v in value.reshape(-1)]
    if isinstance(value, list):
        return [json_safe_metric_value(v) for v in value]
    return value


def save_det_seg_metrics_json(save_dir, mAP, mIoU, iou_per_class):
    metrics = {
        "mAP": json_safe_metric_value(mAP),
        "mIoU": json_safe_metric_value(mIoU),
        "iou_per_class": json_safe_metric_value(iou_per_class),
    }
    save_path = os.path.join(save_dir, DET_SEG_METRICS_JSON_NAME)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return save_path


def pose_bins_to_obbs(
    pos_bins,
    angle_bins,
    scale_bins,
    masks,
    scores=None,
    instance_index_offset=0,
):
    if pos_bins.shape[0] == 0:
        return []

    instance_ids = np.argmax(masks, axis=0).reshape(-1)
    existing_instance_ids = np.unique(instance_ids)

    obbs = []
    for obj_i in existing_instance_ids:
        d_trans = [float(v) for v in pos_bins[obj_i].reshape(3)]
        d_angles = [float(v) for v in angle_bins[obj_i].reshape(3)]
        d_scale = float(scale_bins[obj_i].reshape(-1)[0])
        scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)

        obb = {
            "index": int(obj_i) + int(instance_index_offset),
            "translate": [float(v) for v in trans],
            "rotation": quat_wxyz_from_sxyz_angles(angles),
            "scale": float(scale),
        }
        if scores is not None:
            obb["confidence"] = float(scores[obj_i])
        obbs.append(obb)

    return obbs


def extract_obbs_from_results(
    points_np,
    feats_np,
    results_dict,
    valid_score_threshold,
    nms_iou_threshold,
    no_post_process_perception=False,
):
    voxel_inverse_indices = results_dict["voxel_inverse_indices"]
    pos_bin_logits = results_dict["pos_bin_logits"]
    angle_bin_logits = results_dict["angle_bin_logits"]
    scale_bin_logits = results_dict["scale_bin_logits"]
    valid_logits = results_dict["valid_logits"]
    pred_masks_logits = results_dict["pred_masks_logits"]
    seg_feats = results_dict["seg_feats"]
    background_mask = None
    if "background_pred_masks_logits" in results_dict:
        background_mask = (
            results_dict["background_pred_masks_logits"]
            .sigmoid()[:, :, voxel_inverse_indices][0, 0]
            .detach().cpu().numpy()
        )

    pred_masks = pred_masks_logits.sigmoid()
    pred_masks = pred_masks[:, :, voxel_inverse_indices]

    pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, _, seg_feats = process_pred_poses_w_seg(
        pos_bin_logits=pos_bin_logits,
        angle_bin_logits=angle_bin_logits,
        scale_bin_logits=scale_bin_logits,
        valid_logits=valid_logits,
        pred_masks=pred_masks,
        threshold=valid_score_threshold,
        seg_feats=seg_feats,
    )
    if pred_pos_bins is None:
        return []

    pred_pos_bins = pred_pos_bins.cpu().numpy()
    pred_angle_bins = pred_angle_bins.cpu().numpy()
    pred_scale_bins = pred_scale_bins.cpu().numpy()
    pred_masks = pred_masks.cpu().numpy()
    pred_valid_logits = pred_valid_logits.cpu().numpy()
    seg_feats = seg_feats.cpu().numpy()

    if not no_post_process_perception:
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, pred_valid_logits, seg_feats = nms_process(
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            pred_valid_logits,
            points_np,
            feats_np,
            iou_threshold=nms_iou_threshold,
            seg_feats=seg_feats,
        )

        scored_pose_keys = {
            (
                tuple(np.asarray(pos).reshape(-1).tolist()),
                tuple(np.asarray(angle).reshape(-1).tolist()),
                tuple(np.asarray(scale).reshape(-1).tolist()),
            ): float(1.0 / (1.0 + np.exp(-float(np.asarray(score).reshape(-1)[0]))))
            for pos, angle, scale, score in zip(
                pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_valid_logits
            )
        }
        pred_pos_bins, pred_angle_bins, pred_scale_bins, pred_masks, _ = remove_unreasonable_preds_by_scale(
            pred_pos_bins,
            pred_angle_bins,
            pred_scale_bins,
            pred_masks,
            seg_feats,
            points_np,
        )

    if no_post_process_perception:
        scores = 1.0 / (1.0 + np.exp(-np.asarray(pred_valid_logits).reshape(-1)))
    else:
        scores = [
            scored_pose_keys[
                (
                    tuple(np.asarray(pos).reshape(-1).tolist()),
                    tuple(np.asarray(angle).reshape(-1).tolist()),
                    tuple(np.asarray(scale).reshape(-1).tolist()),
                )
            ]
            for pos, angle, scale in zip(pred_pos_bins, pred_angle_bins, pred_scale_bins)
        ]
    instance_index_offset = 0
    if background_mask is not None:
        foreground_keep = foreground_winner_indices(
            pred_masks, background_mask
        )
        if foreground_keep.size == 0:
            return []
        pred_pos_bins = pred_pos_bins[foreground_keep]
        pred_angle_bins = pred_angle_bins[foreground_keep]
        pred_scale_bins = pred_scale_bins[foreground_keep]
        pred_masks = pred_masks[foreground_keep]
        scores = np.asarray(scores)[foreground_keep]
        instance_index_offset = 1
    return pose_bins_to_obbs(
        pred_pos_bins,
        pred_angle_bins,
        pred_scale_bins,
        pred_masks,
        scores=scores,
        instance_index_offset=instance_index_offset,
    )


def save_obbs_json(save_dir, scene_name, data_name, obbs, filename=OBB_JSON_NAME):
    obbs_data = {
        "scene_name": scene_name,
        "data_name": data_name,
        "rotation_convention": "quaternion_wxyz",
        "euler_source_convention": "sxyz",
        "objects": obbs,
    }
    save_path = os.path.join(save_dir, filename)
    with open(save_path, "w") as f:
        json.dump(obbs_data, f, indent=2)
    return save_path


def run_stage_pose(args, eval_dir_seg, device, eval_indices, pose_bundle, inference_data_map=None, data_load_time_map=None):
    print("=== Stage 1/1: pose+seg voxelize from images ===")

    dino_model = pose_bundle["dino_model"]
    dino_transform_fn = pose_bundle["dino_transform_fn"]
    seg_model = pose_bundle["seg_model"]
    get_inference_data = pose_bundle["get_inference_data"]

    scene_obbs = {}
    pose_timing_map = {}

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.float):
            for eval_i in eval_indices:
                if inference_data_map is not None and eval_i in inference_data_map:
                    inference_data = inference_data_map[eval_i]
                    data_load_elapsed = None if data_load_time_map is None else data_load_time_map.get(eval_i)
                else:
                    data_t0 = time.perf_counter()
                    inference_data = get_inference_data(
                        eval_i, image_downsample=perception_feature_stride(args)
                    )
                    data_load_elapsed = time.perf_counter() - data_t0

                points_np = inference_data["points"]
                points_rgbs = inference_data["points_rgbs"]
                rgbs = inference_data["rgbs"]
                data_name = inference_data.get("data_name", f"{args.split}_{eval_i}")
                scene_name = f"{args.split}_{eval_i}"
                print(f"Stage1 scene {eval_i}: {data_name}")

                if int(getattr(args, "rgb_upsample", 1) or 1) > 1:
                    rgbs = _super_resolve(args, rgbs)

                dino_batch_size = args.dino_frame_batch_size
                if int(getattr(args, "rgb_upsample", 1) or 1) > 1:
                    feats = _tiled_patch_tokens(
                        args, dino_model, dino_transform_fn, rgbs
                    )
                    dino_seconds = 0.0
                elif dino_batch_size is None or len(rgbs) <= dino_batch_size:
                    rgbs_tensor = dino_transform_fn(rgbs).to(device, non_blocking=True)
                    n, _, h, w = rgbs_tensor.shape
                    outputs, dino_seconds = timed_cuda_network_call(
                        lambda: dino_model(rgbs_tensor, is_training=True)
                    )
                    if int(getattr(args, "dino_upsample", 1) or 1) > 1:
                        feats = _anyup_features(args, dino_model, rgbs_tensor)
                    else:
                        feats = outputs["x_norm_patchtokens"].reshape(
                            n
                            * (h // DINO_IMAGE_DOWNSAMPLE)
                            * (w // DINO_IMAGE_DOWNSAMPLE),
                            -1,
                        )
                else:
                    if dino_batch_size <= 0:
                        raise ValueError(
                            "--dino_frame_batch_size must be positive when set"
                        )
                    feature_chunks = []
                    dino_seconds = 0.0
                    for frame_start in range(0, len(rgbs), dino_batch_size):
                        frame_end = min(frame_start + dino_batch_size, len(rgbs))
                        rgbs_tensor = dino_transform_fn(
                            rgbs[frame_start:frame_end]
                        ).to(device, non_blocking=True)
                        n, _, h, w = rgbs_tensor.shape
                        outputs, chunk_seconds = timed_cuda_network_call(
                            lambda: dino_model(rgbs_tensor, is_training=True)
                        )
                        dino_seconds += chunk_seconds
                        if int(getattr(args, "dino_upsample", 1) or 1) > 1:
                            feature_chunks.append(
                                _anyup_features(args, dino_model, rgbs_tensor)
                            )
                        else:
                            feature_chunks.append(
                                outputs["x_norm_patchtokens"].reshape(
                                    n
                                    * (h // DINO_IMAGE_DOWNSAMPLE)
                                    * (w // DINO_IMAGE_DOWNSAMPLE),
                                    -1,
                                )
                            )
                    feats = torch.cat(feature_chunks, dim=0)

                points_np, feats = _subsample_feature_lattice(
                    args, points_np, feats, rgbs
                )
                points = torch.from_numpy(points_np).float().unsqueeze(0).to(device)
                colors = feats.float().unsqueeze(0).to(device)

                if "mask" in inference_data:
                    mask = torch.from_numpy(inference_data["mask"]).bool().unsqueeze(0).to(device)
                    colors = colors[mask].unsqueeze(0)

                results_dict, det_seg_seconds = timed_cuda_network_call(
                    lambda: seg_model.forward_inference(points=points, colors=colors)
                )
                infer_elapsed = dino_seconds + det_seg_seconds

                eval_iter_dir = os.path.join(eval_dir_seg, scene_name)
                os.makedirs(eval_iter_dir, exist_ok=True)
                camera_pose_override = inference_data.get("camera_pose_override")
                if camera_pose_override is not None:
                    with open(
                        os.path.join(eval_iter_dir, "camera_pose_override.json"),
                        "w",
                        encoding="utf-8",
                    ) as handle:
                        json.dump(camera_pose_override, handle, indent=2)

                model_points_np = points[0].cpu().numpy()
                model_feats_np = colors[0].cpu().numpy()
                obbs = extract_obbs_from_results(
                    points_np=model_points_np,
                    feats_np=model_feats_np,
                    results_dict=results_dict,
                    valid_score_threshold=args.valid_score_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    no_post_process_perception=args.no_post_process_perception,
                )
                obbs_path = save_obbs_json(eval_iter_dir, scene_name, data_name, obbs)
                raw_obbs_path = None
                gt_obbs_path = None
                raw_gt_obbs_path = None
                gt_boxes_path = None
                raw_gt_boxes_path = None
                preprocess_transform = inference_data.get("preprocess_transform")
                raw_transform = None
                gt_obbs = inference_data.get("gt_obbs")
                if gt_obbs:
                    gt_obbs_path = save_obbs_json(
                        eval_iter_dir,
                        scene_name,
                        data_name,
                        gt_obbs,
                        filename=GT_OBB_JSON_NAME,
                    )
                    gt_boxes_path = save_obbs_ply(
                        os.path.join(eval_iter_dir, "gt_boxes.ply"),
                        gt_obbs,
                    )
                if preprocess_transform is not None:
                    raw_transform = np.linalg.inv(preprocess_transform)
                    raw_obbs = transform_obbs(obbs, raw_transform)
                    raw_obbs_path = save_obbs_json(
                        eval_iter_dir,
                        scene_name,
                        data_name,
                        raw_obbs,
                        filename=RAW_OBB_JSON_NAME,
                    )
                    if gt_obbs:
                        raw_gt_obbs = transform_obbs(gt_obbs, raw_transform)
                        raw_gt_obbs_path = save_obbs_json(
                            eval_iter_dir,
                            scene_name,
                            data_name,
                            raw_gt_obbs,
                            filename=RAW_GT_OBB_JSON_NAME,
                        )
                        raw_gt_boxes_path = save_obbs_ply(
                            os.path.join(eval_iter_dir, "raw_gt_boxes.ply"),
                            raw_gt_obbs,
                        )
                scene_obbs[scene_name] = obbs
                gt_instance_pcd_path = save_gt_instance_point_cloud(
                    eval_iter_dir,
                    points_np,
                    inference_data.get("points_instance_masks"),
                )
                point_instance_pkl_path = None
                det_seg_metrics_path = None
                pred_point_instances = extract_pred_point_instances_from_results(
                    points_np=model_points_np,
                    feats_np=model_feats_np,
                    results_dict=results_dict,
                    valid_score_threshold=args.valid_score_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    no_post_process_perception=args.no_post_process_perception,
                )
                num_cleaned_instances = 0
                if not args.no_clean_instance_labels_by_obbs:
                    pred_point_instances, num_cleaned_instances = clean_instance_labels_by_obbs(
                        model_points_np,
                        pred_point_instances,
                        obbs,
                    )
                gt_instance_ids = inference_data.get("points_instance_masks")
                point_instance_pkl_path = save_point_instance_masks_pkl(
                    eval_iter_dir,
                    gt_instance_ids,
                    pred_point_instances,
                )
                if gt_instance_ids is not None and gt_obbs:
                    mAP, mIoU, iou_per_class = calculate_mAP_and_mIoU(
                        pred_bboxes=obbs,
                        gt_bboxes=gt_obbs,
                        pred_labels=pred_point_instances,
                        gt_labels=gt_instance_ids,
                        preserve_background_label=(
                            "background_pred_masks_logits" in results_dict
                        ),
                    )
                    det_seg_metrics_path = save_det_seg_metrics_json(
                        eval_iter_dir,
                        mAP,
                        mIoU,
                        iou_per_class,
                    )

                success = visualize_results(
                    points=points,
                    points_rgbs=points_rgbs,
                    colors=colors,
                    results_dict=results_dict,
                    vis_save_dir=eval_iter_dir,
                    valid_score_threshold=args.valid_score_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    save_feats=False,
                    visualize_point_dino_feats_with_pca=True,
                )
                raw_ply_paths = []
                if raw_transform is not None:
                    raw_ply_paths = save_raw_visualization_plys(eval_iter_dir, raw_transform)
                if not success:
                    print(f"Stage1 scene {scene_name}: no valid predicted objects after visualization filtering.")
                if len(obbs) == 0:
                    print(f"Stage1 scene {scene_name}: no detected OBBs saved to {obbs_path}")
                else:
                    print(f"Stage1 scene {scene_name}: saved {len(obbs)} OBBs to {obbs_path}")
                if gt_instance_pcd_path is not None:
                    print(f"Stage1 scene {scene_name}: saved GT instance points to {gt_instance_pcd_path}")
                if point_instance_pkl_path is not None:
                    print(f"Stage1 scene {scene_name}: saved point instance masks to {point_instance_pkl_path}")
                    if args.no_clean_instance_labels_by_obbs:
                        print(f"Stage1 scene {scene_name}: OBB point cleanup disabled")
                    else:
                        print(
                            f"Stage1 scene {scene_name}: bbox-cleaned "
                            f"{num_cleaned_instances} predicted instance points"
                        )
                if det_seg_metrics_path is not None:
                    print(f"Stage1 scene {scene_name}: saved det/seg metrics to {det_seg_metrics_path}")
                if gt_obbs_path is not None:
                    print(f"Stage1 scene {scene_name}: saved GT OBBs to {gt_obbs_path}")
                if gt_boxes_path is not None:
                    print(f"Stage1 scene {scene_name}: saved GT boxes to {gt_boxes_path}")
                if raw_obbs_path is not None:
                    print(f"Stage1 scene {scene_name}: saved raw-frame OBBs to {raw_obbs_path}")
                if raw_gt_obbs_path is not None:
                    print(f"Stage1 scene {scene_name}: saved raw-frame GT OBBs to {raw_gt_obbs_path}")
                if raw_gt_boxes_path is not None:
                    print(f"Stage1 scene {scene_name}: saved raw-frame GT boxes to {raw_gt_boxes_path}")
                if raw_ply_paths:
                    print(f"Stage1 scene {scene_name}: saved {len(raw_ply_paths)} raw-frame PLYs")

                timing_payload = {
                    "schema": "ff_perception_network_timing_v1",
                    "scene_name": scene_name,
                    "data_name": data_name,
                    "definition": (
                        "CUDA-event device time around neural forwards only; excludes data IO, "
                        "DINO transforms, host/device tensor preparation, postprocessing, metrics, "
                        "serialization, and visualization"
                    ),
                    "dino_seconds": dino_seconds,
                    "det_seg_seconds": det_seg_seconds,
                    "network_total_seconds": infer_elapsed,
                    "num_predicted_objects": len(obbs),
                    "amortized_seconds_per_predicted_object": (
                        infer_elapsed / len(obbs) if obbs else None
                    ),
                }
                with open(os.path.join(eval_iter_dir, NETWORK_TIMING_JSON_NAME), "w") as f:
                    json.dump(timing_payload, f, indent=2)

                if data_load_elapsed is None:
                    print(f"Stage1 timing {scene_name}: model_inference={infer_elapsed:.3f}s")
                else:
                    print(
                        f"Stage1 timing {scene_name}: data_load={data_load_elapsed:.3f}s, "
                        f"model_inference={infer_elapsed:.3f}s"
                    )
                pose_timing_map[scene_name] = {
                    "data_load": data_load_elapsed,
                    "model_inference": infer_elapsed,
                    "dino": dino_seconds,
                    "det_seg": det_seg_seconds,
                }

    return scene_obbs, pose_timing_map


def preload_resources(args, device):
    print("=== Preloading pose checkpoint/model ===")

    dino_model, dino_transform_fn = get_dino_model(
        model_path=args.dino_model_path,
        repo_dir=args.dino_repo_dir,
    )
    get_inference_data = load_pose_inference_data(args)
    seg_config_path = args.config_path or os.path.join(args.run_dir_seg, "config.yaml")
    seg_ckpt = args.checkpoint_path or get_latest_ckpt(
        os.path.join(args.run_dir_seg, "checkpoints")
    )
    print(f"Preload pose config: {seg_config_path}")
    seg_config = load_config(seg_config_path)
    print(f"Preload pose ckpt: {seg_ckpt}")
    seg_model = load_model_obj_pose_w_seg_voxelize(seg_ckpt, seg_config).to(device).eval()
    seg_model.get_voxelizer_inference()

    print("=== Preloading finished ===")
    return {
        "pose": {
            "dino_model": dino_model,
            "dino_transform_fn": dino_transform_fn,
            "seg_model": seg_model,
            "get_inference_data": get_inference_data,
        }
    }


def main():
    args = parse_args()
    device = "cuda"
    eval_indices = get_eval_indices(args.num_eval_scenes, args.eval_idx, args.eval_indices)
    print(f"Eval indices: {eval_indices}")

    eval_dir_seg = get_seg_eval_dir(
        args.run_dir_seg,
        args.dataset_type,
        args.eval_dir_name,
        output_dir=args.output_dir,
    )
    os.makedirs(eval_dir_seg, exist_ok=True)
    print(f"Pose save directory: {eval_dir_seg}")

    if args.skip_existing:
        requested_count = len(eval_indices)
        eval_indices = [
            eval_i
            for eval_i in eval_indices
            if not os.path.isfile(
                os.path.join(eval_dir_seg, f"{args.split}_{eval_i}", DET_SEG_METRICS_JSON_NAME)
            )
        ]
        print(
            f"Skip-existing filter: pending={len(eval_indices)} "
            f"already_complete={requested_count - len(eval_indices)}"
        )
        if len(eval_indices) == 0:
            print("All requested perception scenes already contain metrics; nothing to do.")
            return

    resources = preload_resources(args, device)

    per_scene_stats = []
    total_start = time.perf_counter()

    for eval_i in eval_indices:
        scene_name = f"{args.split}_{eval_i}"
        print(f"=== Scene pose start: {scene_name} ===")
        scene_start = time.perf_counter()
        scene_stat = {
            "scene_name": scene_name,
            "data_load": 0.0,
            "model_inference": 0.0,
            "stage_pose": 0.0,
            "num_obbs": 0,
            "scene_total": 0.0,
        }

        data_t0 = time.perf_counter()
        inference_data = resources["pose"]["get_inference_data"](
            eval_i, image_downsample=perception_feature_stride(args)
        )
        data_load_elapsed = time.perf_counter() - data_t0

        pose_t0 = time.perf_counter()
        scene_obbs, pose_timing_map = run_stage_pose(
            args,
            eval_dir_seg,
            device,
            [eval_i],
            resources["pose"],
            inference_data_map={eval_i: inference_data},
            data_load_time_map={eval_i: data_load_elapsed},
        )

        scene_stat["stage_pose"] = time.perf_counter() - pose_t0
        scene_stat["data_load"] = data_load_elapsed
        scene_stat["model_inference"] = pose_timing_map.get(scene_name, {}).get("model_inference", 0.0)
        scene_stat["num_obbs"] = len(scene_obbs.get(scene_name, []))
        scene_stat["scene_total"] = time.perf_counter() - scene_start
        per_scene_stats.append(scene_stat)

        print(f"=== Scene pose done: {scene_name}, elapsed={scene_stat['scene_total']:.2f}s ===")
        print(
            f"Scene timing {scene_name}: data_load={scene_stat['data_load']:.2f}s, "
            f"model_inference={scene_stat['model_inference']:.2f}s, "
            f"pose={scene_stat['stage_pose']:.2f}s, "
            f"num_obbs={scene_stat['num_obbs']}, "
            f"total={scene_stat['scene_total']:.2f}s"
        )

    total_elapsed = time.perf_counter() - total_start
    if len(per_scene_stats) > 0:
        n = len(per_scene_stats)
        avg_data_load = sum(x["data_load"] for x in per_scene_stats) / n
        avg_model_infer = sum(x["model_inference"] for x in per_scene_stats) / n
        avg_pose = sum(x["stage_pose"] for x in per_scene_stats) / n
        avg_scene = sum(x["scene_total"] for x in per_scene_stats) / n
        total_obbs = sum(x["num_obbs"] for x in per_scene_stats)
        print(
            f"Timing summary: scenes={n}, total={total_elapsed:.2f}s, "
            f"avg_data_load={avg_data_load:.2f}s, avg_model_inference={avg_model_infer:.2f}s, "
            f"avg_pose={avg_pose:.2f}s, avg_per_scene={avg_scene:.2f}s, total_obbs={total_obbs}"
        )


if __name__ == "__main__":
    main()
