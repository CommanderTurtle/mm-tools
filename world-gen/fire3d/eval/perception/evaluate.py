"""Deterministic, balanced det/seg benchmark on the training val split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from training.datasets.scene_object_poses_dataset_multiple_syn2real import get_dataset
from training.config import load_config
from eval.perception.det_seg_eval_utils import _obb_iou
from eval.perception.eval_perception import quat_wxyz_from_sxyz_angles
from models.object_pose_w_seg_voxelize_dino import ObjectPoseWSegVoxelize
from utils.det_seg import (
    nms_process,
    process_pred_poses_w_seg,
    remove_unreasonable_preds_by_scale,
)
from utils.discrete import continue_transform

DEFAULT_VALID_SCORE_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(REPO_ROOT / "configs/training/perception/default.yaml"),
    )
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--checkpoint", default="latest")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--eval_name", default="detseg_val100_balanced")
    parser.add_argument("--split", default="val", choices=["trainval", "val"])
    parser.add_argument("--num_scenes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--valid_score_threshold",
        type=float,
        default=None,
        help="Evaluate one validity threshold instead of the default sweep.",
    )
    parser.add_argument(
        "--valid_score_thresholds",
        default=None,
        help=(
            "Comma-separated threshold sweep; model inference is shared across thresholds. "
            "Defaults to 0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8."
        ),
    )
    parser.add_argument("--nms_iou_threshold", type=float, default=0.4)
    parser.add_argument("--selection_only", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


def parse_thresholds(args):
    if args.valid_score_thresholds is not None:
        values = [float(v.strip()) for v in args.valid_score_thresholds.split(",") if v.strip()]
    elif args.valid_score_threshold is not None:
        values = [float(args.valid_score_threshold)]
    else:
        values = list(DEFAULT_VALID_SCORE_THRESHOLDS)
    if not values or any(v < 0.0 or v > 1.0 for v in values):
        raise ValueError(f"Invalid validity thresholds: {values}")
    return sorted(set(values))


def threshold_key(value):
    return f"{float(value):.6g}"


def checkpoint_step(path):
    name = Path(path).stem
    try:
        return int(name.split("_")[-1])
    except ValueError:
        return None


def latest_checkpoint(run_dir):
    ckpt_dir = Path(run_dir) / "checkpoints"
    paths = list(ckpt_dir.glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No checkpoint under {ckpt_dir}")
    return str(max(paths, key=lambda p: checkpoint_step(p) or -1))


def resolve_checkpoint(args):
    if args.checkpoint == "latest":
        if args.run_dir is None:
            raise ValueError("--checkpoint latest requires --run_dir")
        return latest_checkpoint(args.run_dir)
    path = Path(args.checkpoint)
    if path.is_dir():
        return latest_checkpoint(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return str(path)


def stable_rank(seed, data_name):
    digest = hashlib.sha256(f"{seed}:{data_name}".encode()).hexdigest()
    return digest, data_name


def balanced_selection(dataset, num_scenes, seed):
    groups = defaultdict(list)
    seen = set()
    for dataset_index, item in enumerate(dataset.data_list):
        key = (item["dataset_name"], item["data_name"])
        if key in seen:
            continue
        seen.add(key)
        groups[item["dataset_name"]].append((dataset_index, item))

    names = sorted(groups)
    if not names:
        raise RuntimeError("Val dataset is empty")
    base, remainder = divmod(num_scenes, len(names))
    quotas = {name: base + (i < remainder) for i, name in enumerate(names)}
    selected = []
    for name in names:
        ranked = sorted(
            groups[name], key=lambda pair: stable_rank(seed, pair[1]["data_name"])
        )
        if len(ranked) < quotas[name]:
            raise RuntimeError(
                f"Dataset {name} has {len(ranked)} unique val scenes, needs {quotas[name]}"
            )
        for dataset_index, item in ranked[: quotas[name]]:
            selected.append(
                {
                    "dataset_index": dataset_index,
                    "dataset_name": name,
                    "data_name": item["data_name"],
                    "scene_id": item.get("scene_id"),
                    "video_id": item.get("video_id"),
                }
            )
    return selected, quotas


def bins_to_obb(pos_bins, angle_bins, scale_bins, index):
    d_trans = np.asarray(pos_bins).reshape(3)
    d_angles = np.asarray(angle_bins).reshape(3)
    d_scale = float(np.asarray(scale_bins).reshape(-1)[0])
    scale, angles, trans = continue_transform(d_scale, d_angles, d_trans)
    return {
        "index": int(index),
        "translate": [float(v) for v in np.asarray(trans).reshape(3)],
        "rotation": quat_wxyz_from_sxyz_angles(angles),
        "scale": float(np.asarray(scale).reshape(-1)[0]),
    }


def postprocess_predictions(results, points, feats, valid_threshold, nms_threshold):
    inverse = results["voxel_inverse_indices"]
    masks = results["pred_masks_logits"].sigmoid()
    masks = masks[:, :, inverse]
    processed = process_pred_poses_w_seg(
        pos_bin_logits=results["pos_bin_logits"],
        angle_bin_logits=results["angle_bin_logits"],
        scale_bin_logits=results["scale_bin_logits"],
        valid_logits=results["valid_logits"],
        pred_masks=masks,
        threshold=valid_threshold,
        seg_feats=results["seg_feats"],
    )
    if processed[0] is None:
        return [], np.full(points.shape[0], -1, dtype=np.int32)

    pos, angles, scales, masks, valid, _, seg_feats = processed
    pos = pos.cpu().numpy()
    angles = angles.cpu().numpy()
    scales = scales.cpu().numpy()
    masks = masks.float().cpu().numpy()
    valid = valid.float().cpu().numpy()
    seg_feats = seg_feats.float().cpu().numpy()
    pos, angles, scales, masks, valid, seg_feats = nms_process(
        pos,
        angles,
        scales,
        masks,
        valid,
        points,
        feats,
        iou_threshold=nms_threshold,
        seg_feats=seg_feats,
    )
    pos, angles, scales, masks, seg_feats = remove_unreasonable_preds_by_scale(
        pos, angles, scales, masks, seg_feats, points
    )
    if pos.shape[0] == 0:
        return [], np.full(points.shape[0], -1, dtype=np.int32)

    obbs = [bins_to_obb(pos[i], angles[i], scales[i], i) for i in range(pos.shape[0])]
    pred_labels = np.argmax(masks, axis=0).astype(np.int32)
    return obbs, pred_labels


def build_gt(batch):
    pos = batch["object_translations"][0].cpu().numpy()
    angles = batch["object_angles"][0].cpu().numpy()
    scales = batch["object_scales"][0].cpu().numpy()
    # Token 0 is the layout/background. Detection metrics score instances >= 1.
    obbs = [bins_to_obb(pos[i], angles[i], scales[i], i) for i in range(1, pos.shape[0])]
    labels = batch["instance_ids"][0].cpu().numpy().astype(np.int32)
    return obbs, labels


def greedy_matches(pred_obbs, gt_obbs, threshold):
    candidates = []
    for pred_i, pred in enumerate(pred_obbs):
        for gt_i, gt in enumerate(gt_obbs):
            iou = _obb_iou(pred, gt)
            if iou >= threshold:
                candidates.append((iou, pred_i, gt_i))
    candidates.sort(reverse=True)
    used_pred, used_gt, matches = set(), set(), []
    for iou, pred_i, gt_i in candidates:
        if pred_i in used_pred or gt_i in used_gt:
            continue
        used_pred.add(pred_i)
        used_gt.add(gt_i)
        matches.append((pred_i, gt_i, iou))
    return matches


def segmentation_mask_ious(pred_labels, gt_labels, matches):
    """Return per-GT and matched-only point-mask IoUs after OBB association."""
    pred_to_gt = {pred_i: gt_i + 1 for pred_i, gt_i, _ in matches}
    matched_gt_ids = set(pred_to_gt.values())
    reassigned = np.full(pred_labels.shape, -1, dtype=np.int32)
    for pred_i, gt_id in pred_to_gt.items():
        reassigned[pred_labels == pred_i] = gt_id
    gt_ids = sorted(int(v) for v in np.unique(gt_labels) if v >= 1)
    all_gt_ious = []
    matched_ious = []
    for gt_id in gt_ids:
        pred_mask = reassigned == gt_id
        gt_mask = gt_labels == gt_id
        union = np.logical_or(pred_mask, gt_mask).sum()
        iou = float(np.logical_and(pred_mask, gt_mask).sum() / union) if union else 0.0
        all_gt_ious.append(iou)
        if gt_id in matched_gt_ids:
            matched_ious.append(iou)
    return all_gt_ious, matched_ious


def scene_metrics(pred_obbs, gt_obbs, pred_labels, gt_labels):
    result = {
        "num_pred_objects": len(pred_obbs),
        "num_gt_objects": len(gt_obbs),
    }
    for threshold, suffix in ((0.25, "25"), (0.50, "50")):
        matches = greedy_matches(pred_obbs, gt_obbs, threshold)
        tp = len(matches)
        fp = len(pred_obbs) - tp
        fn = len(gt_obbs) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        # Detection has no useful true-negative count. This set accuracy is the
        # Jaccard score over matched predictions and GT objects.
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        result.update(
            {
                f"tp{suffix}": tp,
                f"fp{suffix}": fp,
                f"fn{suffix}": fn,
                f"precision{suffix}": precision,
                f"recall{suffix}": recall,
                f"f1_{suffix}": f1,
                f"accuracy{suffix}": accuracy,
                f"matched_iou{suffix}": float(np.mean([m[2] for m in matches])) if matches else 0.0,
            }
        )
        all_gt_ious, matched_ious = segmentation_mask_ious(
            pred_labels, gt_labels, matches
        )
        result[f"mask_miou_all_gt_at{suffix}"] = (
            float(np.mean(all_gt_ious)) if all_gt_ious else None
        )
        result[f"mask_miou_matched_at{suffix}"] = (
            float(np.mean(matched_ious)) if matched_ious else None
        )
        result[f"mask_ious_all_gt_at{suffix}"] = all_gt_ious
        result[f"mask_ious_matched_at{suffix}"] = matched_ious
        if threshold == 0.25:
            # Backward-compatible names used by the first benchmark report.
            result["instance_miou_fg_at25"] = result["mask_miou_all_gt_at25"]
            result["instance_iou_fg_at25"] = all_gt_ious
    return result


def aggregate(scenes):
    if not scenes:
        return {}
    result = {"num_scenes": len(scenes)}
    result["num_pred_objects"] = sum(s["num_pred_objects"] for s in scenes)
    result["num_gt_objects"] = sum(s["num_gt_objects"] for s in scenes)
    for suffix in ("25", "50"):
        tp = sum(s[f"tp{suffix}"] for s in scenes)
        fp = sum(s[f"fp{suffix}"] for s in scenes)
        fn = sum(s[f"fn{suffix}"] for s in scenes)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        accuracy = tp / (tp + fp + fn) if tp + fp + fn else 0.0
        result.update(
            {
                f"tp{suffix}": tp,
                f"fp{suffix}": fp,
                f"fn{suffix}": fn,
                f"micro_precision{suffix}": precision,
                f"micro_recall{suffix}": recall,
                f"micro_f1_{suffix}": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                f"micro_accuracy{suffix}": accuracy,
                f"macro_precision{suffix}": float(np.mean([s[f"precision{suffix}"] for s in scenes])),
                f"macro_recall{suffix}": float(np.mean([s[f"recall{suffix}"] for s in scenes])),
                f"macro_f1_{suffix}": float(np.mean([s[f"f1_{suffix}"] for s in scenes])),
                f"macro_accuracy{suffix}": float(np.mean([s[f"accuracy{suffix}"] for s in scenes])),
            }
        )
    valid_miou = [s["instance_miou_fg_at25"] for s in scenes if s["instance_miou_fg_at25"] is not None]
    result["macro_instance_miou_fg_at25"] = float(np.mean(valid_miou)) if valid_miou else None
    for suffix in ("25", "50"):
        for population in ("all_gt", "matched"):
            scene_key = f"mask_miou_{population}_at{suffix}"
            scene_values = [s[scene_key] for s in scenes if s[scene_key] is not None]
            result[f"macro_scene_{scene_key}"] = (
                float(np.mean(scene_values)) if scene_values else None
            )
            instance_key = f"mask_ious_{population}_at{suffix}"
            instance_values = [iou for s in scenes for iou in s[instance_key]]
            result[f"dataset_instance_mask_miou_{population}_at{suffix}"] = (
                float(np.mean(instance_values)) if instance_values else None
            )
    comprehensive_components = (
        result["micro_f1_25"],
        result["micro_f1_50"],
        result["dataset_instance_mask_miou_all_gt_at25"],
        result["dataset_instance_mask_miou_all_gt_at50"],
    )
    result["comprehensive_score"] = float(np.mean(comprehensive_components))
    result["mean_inference_seconds"] = float(np.mean([s["inference_seconds"] for s in scenes]))
    return result


def main():
    args = parse_args()
    thresholds = parse_thresholds(args)
    config = load_config(args.config)
    dataset = get_dataset(config["data"], split=args.split)
    selection, quotas = balanced_selection(dataset, args.num_scenes, args.seed)

    checkpoint = None if args.selection_only else resolve_checkpoint(args)
    step = checkpoint_step(checkpoint) if checkpoint else None
    root = args.output_dir
    if root is None:
        root = os.path.join(args.run_dir or "results/detseg_val_benchmark", "eval")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_name = f"step_{step:07d}" if step is not None else "selection"
    eval_dir = Path(root) / step_name / f"{args.split}_{stamp}_{args.eval_name}"
    eval_dir.mkdir(parents=True, exist_ok=True)
    with open(eval_dir / "selection_manifest.json", "w") as f:
        json.dump(json_safe({"seed": args.seed, "quotas": quotas, "scenes": selection}), f, indent=2)
    if args.selection_only:
        print(f"Wrote deterministic selection to {eval_dir}")
        return

    model_config = dict(config["model"])
    model_config["rotation_symmetry"] = config.get("rotation_symmetry", {})
    model = ObjectPoseWSegVoxelize(model_config)
    ckpt = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    incompatible = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    device = torch.device(args.device)
    model = model.to(device).eval()
    model.get_voxelizer_inference()

    successes, failures = [], []
    successes_by_threshold = {threshold_key(v): [] for v in thresholds}
    for order, selected in enumerate(tqdm(selection, desc="Det/seg val benchmark")):
        try:
            item = dataset[selected["dataset_index"]]
            points = item["points"].unsqueeze(0).to(device)
            rgbs = item["rgbs"].unsqueeze(0).to(device)
            valid_mask = item.get("valid_point_masks")
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                feats, _ = model.get_dino_feats(rgbs[0])
                if valid_mask is not None:
                    feats = feats[valid_mask.to(device)]
                t0 = time.perf_counter()
                results = model.forward_inference(points=points, colors=feats.unsqueeze(0))
                inference_seconds = time.perf_counter() - t0
            points_np = points[0].float().cpu().numpy()
            feats_np = feats.float().cpu().numpy()
            batch = {
                "object_translations": item["object_translations"].unsqueeze(0),
                "object_angles": item["object_angles"].unsqueeze(0),
                "object_scales": item["object_scales"].unsqueeze(0),
                "instance_ids": item["instance_ids"].unsqueeze(0),
            }
            gt_obbs, gt_labels = build_gt(batch)
            threshold_metrics = {}
            # The full postprocess depends on threshold only through the query
            # validity mask. Cache identical masks so broad sweeps remain exact
            # without repeating NMS/scale filtering unnecessarily.
            threshold_groups = defaultdict(list)
            valid_scores = results["valid_logits"].sigmoid().detach().float().cpu().numpy()
            for threshold in thresholds:
                mask_key = np.asarray(valid_scores > threshold, dtype=np.bool_).tobytes()
                threshold_groups[mask_key].append(threshold)
            for grouped_thresholds in threshold_groups.values():
                threshold = grouped_thresholds[0]
                pred_obbs, pred_labels = postprocess_predictions(
                    results,
                    points_np,
                    feats_np,
                    threshold,
                    args.nms_iou_threshold,
                )
                metrics = scene_metrics(pred_obbs, gt_obbs, pred_labels, gt_labels)
                for grouped_threshold in grouped_thresholds:
                    key = threshold_key(grouped_threshold)
                    threshold_metrics[key] = metrics
                    successes_by_threshold[key].append(
                        {**selected, "order": order, "inference_seconds": inference_seconds, **metrics}
                    )
            primary_key = threshold_key(thresholds[0])
            record = {
                **selected,
                "order": order,
                "inference_seconds": inference_seconds,
                "threshold_metrics": threshold_metrics,
                **threshold_metrics[primary_key],
            }
            successes.append(record)
            scene_dir = eval_dir / f"{order:03d}_{selected['dataset_name']}_{selected['data_name']}"
            scene_dir.mkdir(parents=True, exist_ok=True)
            with open(scene_dir / "metrics.json", "w") as f:
                json.dump(json_safe(record), f, indent=2)
        except Exception as exc:
            failures.append({**selected, "order": order, "error": repr(exc), "traceback": traceback.format_exc()})
            print(f"[fail] {selected['data_name']}: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    threshold_summaries = {
        key: aggregate(records) for key, records in successes_by_threshold.items()
    }
    per_dataset_by_threshold = {}
    for key, records in successes_by_threshold.items():
        per_dataset_by_threshold[key] = {
            name: aggregate([s for s in records if s["dataset_name"] == name])
            for name in sorted({s["dataset_name"] for s in records})
        }
    metric_objectives = (
        "micro_f1_25",
        "micro_f1_50",
        "micro_accuracy25",
        "micro_accuracy50",
        "comprehensive_score",
        "macro_instance_miou_fg_at25",
        "dataset_instance_mask_miou_all_gt_at25",
        "dataset_instance_mask_miou_matched_at25",
        "dataset_instance_mask_miou_all_gt_at50",
        "dataset_instance_mask_miou_matched_at50",
    )
    best_thresholds = {
        metric: max(
            threshold_summaries,
            key=lambda key: threshold_summaries[key].get(metric) or float("-inf"),
        )
        for metric in metric_objectives
    }
    primary_key = best_thresholds["micro_f1_25"]
    summary = {
        "config": args.config,
        "run_dir": args.run_dir,
        "checkpoint": checkpoint,
        "checkpoint_step": step,
        "split": args.split,
        "seed": args.seed,
        "num_requested_scenes": args.num_scenes,
        "num_success_scenes": len(successes),
        "num_failures": len(failures),
        "selection_quotas": quotas,
        "valid_score_threshold": float(primary_key),
        "valid_score_thresholds": thresholds,
        "nms_iou_threshold": args.nms_iou_threshold,
        "metric_definition": {
            "detection": (
                "greedy one-to-one oriented-box IoU matching; micro and macro "
                "precision/recall/F1/set-accuracy at 0.25 and 0.50; set accuracy "
                "is TP/(TP+FP+FN), since detection has no meaningful TN count"
            ),
            "segmentation": (
                "point-mask IoU after OBB association at 0.25 and 0.50; reports "
                "all-GT mIoU with unmatched GT instances scored as zero, and "
                "matched-only mIoU to isolate mask quality"
            ),
            "background": "layout/background token 0 is excluded from detection",
            "comprehensive_score": (
                "mean of micro F1@0.25, micro F1@0.50, dataset-instance all-GT "
                "mask mIoU after OBB@0.25 association, and the same after "
                "OBB@0.50 association"
            ),
        },
        "best_thresholds": best_thresholds,
        "overall_metrics": threshold_summaries[primary_key],
        "per_dataset_metrics": per_dataset_by_threshold[primary_key],
        "threshold_summaries": threshold_summaries,
        "per_dataset_by_threshold": per_dataset_by_threshold,
        "scenes": successes,
        "failures": failures,
    }
    with open(eval_dir / "eval_summary.json", "w") as f:
        json.dump(json_safe(summary), f, indent=2)
    print(json.dumps(json_safe({
        "best_thresholds": best_thresholds,
        "threshold_summaries": threshold_summaries,
    }), indent=2))
    print(f"Wrote benchmark to {eval_dir}")


if __name__ == "__main__":
    main()
