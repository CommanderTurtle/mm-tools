#!/usr/bin/env python3
"""Render and score LC64 PBR predictions on iTHOR or Imaginarium RGB views."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.geometry_protocol import (  # noqa: E402
    jsonable,
    load_manifest,
    load_scene_transforms,
    object_world_transform,
)
from eval.rendering.pbr_render_metrics import (  # noqa: E402
    METRIC_KEYS,
    aggregate_metric_rows,
    aggregate_objects_two_level,
    image_metrics,
    load_rgba_composited,
)

BLENDER_RENDERER = REPO_ROOT / "benchmarks/scene_reconstruction/render_appearance_blender.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", action="append", default=[])
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--views-per-object", type=int, default=1)
    parser.add_argument("--recipe", default="canonical_pbr")
    parser.add_argument("--blender", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    if args.views_per_object <= 0:
        parser.error("--views-per-object must be positive")
    return args


def resolve_blender(explicit: Path | None) -> Path:
    candidates = [
        explicit,
        Path(shutil.which("blender")) if shutil.which("blender") else None,
        REPO_ROOT / "blender/blender-4.5.1-linux-x64/blender",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Blender was not found; pass --blender")


def select_scenes(manifest: dict[str, Any], requested: list[str], maximum: int | None) -> list[dict[str, Any]]:
    scenes = manifest["scenes"]
    if requested:
        names = set(requested)
        missing = names - {scene["scene_id"] for scene in scenes}
        if missing:
            raise ValueError(f"Scenes not in manifest: {sorted(missing)}")
        scenes = [scene for scene in scenes if scene["scene_id"] in names]
    if maximum is not None:
        scenes = scenes[:maximum]
    if not scenes:
        raise ValueError("No scenes selected")
    return scenes


def load_mask(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        key = "arr_0" if "arr_0" in archive else archive.files[0]
        return np.asarray(archive[key])


def most_visible_views(mask_dir: Path, object_id: int, count: int, num_frames: int) -> list[tuple[int, int]]:
    ranked = []
    for index in range(num_frames):
        pixels = int(np.count_nonzero(load_mask(mask_dir / f"mask_{index:04d}.npz") == int(object_id)))
        if pixels:
            ranked.append((index, pixels))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return ranked[:count]


def make_target(source: Path, mask_path: Path, object_id: int, output: Path) -> None:
    rgb = np.asarray(Image.open(source).convert("RGB"), dtype=np.uint8)
    mask = load_mask(mask_path) == int(object_id)
    if mask.shape != rgb.shape[:2]:
        raise ValueError(f"Mask/image mismatch: {mask.shape} != {rgb.shape[:2]}")
    rgba = np.concatenate([rgb, (mask[..., None].astype(np.uint8) * 255)], axis=-1)
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgba, "RGBA").save(output)


def prepare_scene_tasks(
    *,
    manifest: dict[str, Any],
    scene: dict[str, Any],
    dataset_root: Path,
    prediction_root: Path,
    output_root: Path,
    views_per_object: int,
) -> tuple[Path, list[dict[str, Any]], int]:
    scene_id = scene["scene_id"]
    inference_path = prediction_root / scene_id / "inference_summary.json"
    inference = json.loads(inference_path.read_text())
    predictions = {
        int(record["object_id"]): record
        for record in inference.get("objects", [])
        if record.get("status") == "decoded" and record.get("textured_glb")
    }
    transforms = load_scene_transforms(dataset_root / scene["transforms_path"])
    camera_path = dataset_root / scene["camera_path"]
    camera = json.loads(camera_path.read_text())
    frame_dir = dataset_root / scene["frames_dir"]
    mask_dir = dataset_root / scene["masks_dir"]
    tasks = []
    for object_id in scene["visible_object_ids"]:
        object_id = int(object_id)
        prediction = predictions.get(object_id)
        if prediction is None:
            continue
        object_name = f"object_{object_id:04d}"
        views = []
        for view_index, visible_pixels in most_visible_views(
            mask_dir, object_id, views_per_object, len(camera["frames"])
        ):
            view_dir = output_root / scene_id / object_name / f"view_{view_index:04d}"
            target = view_dir / "target_rgba.png"
            prediction_rgba = view_dir / "prediction_rgba.png"
            source = frame_dir / f"frame_{view_index:04d}.jpg"
            make_target(source, mask_dir / f"mask_{view_index:04d}.npz", object_id, target)
            views.append(
                {
                    "view_index": view_index,
                    "visible_pixels": visible_pixels,
                    "source_rgb": str(source),
                    "target_rgba": str(target),
                    "prediction_rgba": str(prediction_rgba),
                }
            )
        if views:
            tasks.append(
                {
                    "object_id": object_id,
                    "object_name": object_name,
                    "textured_glb": prediction["textured_glb"],
                    "object_to_world": object_world_transform(transforms[object_name]).tolist(),
                    "views": views,
                }
            )
    payload = {
        "schema": "ff_scene_appearance_exact_camera_tasks_v1",
        "dataset": manifest["dataset"],
        "scene_id": scene_id,
        "camera_path": str(camera_path),
        "frames_dir": str(frame_dir),
        "masks_dir": str(mask_dir),
        "objects": tasks,
    }
    path = output_root / scene_id / "render_tasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path, tasks, len(scene["visible_object_ids"])


def font(size: int, bold: bool = False):
    candidates = [
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def comparison_image(task: dict[str, Any], view: dict[str, Any], output: Path, metrics: dict[str, float], tile: int = 256) -> None:
    target_rgb, target_mask = load_rgba_composited(view["target_rgba"])
    pred_rgb, pred_mask = load_rgba_composited(view["prediction_rgba"])
    union = target_mask | pred_mask
    ys, xs = np.nonzero(union)
    if len(xs):
        pad = 8
        box = (
            max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
            min(union.shape[1], int(xs.max()) + pad + 1), min(union.shape[0], int(ys.max()) + pad + 1),
        )
    else:
        box = (0, 0, union.shape[1], union.shape[0])
    source = Image.open(view["source_rgb"]).convert("RGB")
    source_draw = ImageDraw.Draw(source)
    source_draw.rectangle(box, outline=(255, 40, 40), width=3)
    target = Image.fromarray(np.clip(target_rgb * 255, 0, 255).astype(np.uint8)).crop(box)
    pred = Image.fromarray(np.clip(pred_rgb * 255, 0, 255).astype(np.uint8)).crop(box)
    error = Image.fromarray(
        np.clip(np.abs(pred_rgb - target_rgb) * 4.0 * 255, 0, 255).astype(np.uint8)
    ).crop(box)
    images = [source, target, pred, error]
    labels = ["Input frame", "Masked reference", "Predicted PBR", "|Pred-reference| x4"]
    header = 72
    canvas = Image.new("RGB", (4 * tile, header + tile), (244, 246, 250))
    draw = ImageDraw.Draw(canvas)
    title = (
        f"{task['object_name']} view {int(view['view_index'])} | "
        f"crop PSNR {metrics['crop_psnr']:.2f} | crop SSIM {metrics['crop_ssim']:.3f}"
    )
    draw.text((10, 8), title, fill=(20, 28, 40), font=font(16, True))
    for index, (image, label) in enumerate(zip(images, labels)):
        draw.text((index * tile + 8, 42), label, fill=(38, 48, 62), font=font(13, True))
        fitted = image.resize((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(fitted, (index * tile, header))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def scene_overview(comparisons: list[Path], output: Path, maximum: int = 8) -> None:
    comparisons = comparisons[:maximum]
    if not comparisons:
        return
    images = [Image.open(path).convert("RGB") for path in comparisons]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    canvas = Image.new("RGB", (width, height), "white")
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def write_csv(path: Path, objects: list[dict[str, Any]]) -> None:
    fields = ["dataset", "scene_id", "object_id", "num_views", *METRIC_KEYS]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in objects:
            writer.writerow(
                {
                    "dataset": record["dataset"],
                    "scene_id": record["scene_id"],
                    "object_id": record["object_id"],
                    "num_views": record["num_views"],
                    **record["metrics"],
                }
            )


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest.resolve())
    scenes = select_scenes(manifest, args.scene, args.max_scenes)
    dataset_root = args.fire3d_test_root.resolve() / manifest["dataset_subdir"]
    output_root = args.output.resolve()
    blender = resolve_blender(args.blender)
    object_rows = []
    scene_coverage = {}
    for scene in scenes:
        tasks_path, tasks, expected = prepare_scene_tasks(
            manifest=manifest,
            scene=scene,
            dataset_root=dataset_root,
            prediction_root=args.prediction_root.resolve(),
            output_root=output_root,
            views_per_object=args.views_per_object,
        )
        command = [
            str(blender), "-b", "--factory-startup", "--python", str(BLENDER_RENDERER),
            "--", "--tasks", str(tasks_path), "--recipe", args.recipe,
        ]
        if args.overwrite:
            command.append("--overwrite")
        subprocess.run(command, check=True)
        missing_renders = [
            view["prediction_rgba"]
            for task in tasks
            for view in task["views"]
            if not Path(view["prediction_rgba"]).is_file()
        ]
        if missing_renders:
            preview = "\n".join(missing_renders[:8])
            raise RuntimeError(
                f"Blender did not produce {len(missing_renders)} requested appearance renders. "
                "Its executable can return zero after a Python exception. Missing outputs include:\n"
                f"{preview}"
            )
        comparisons = []
        for task in tasks:
            views = []
            for view in task["views"]:
                pred, pred_mask = load_rgba_composited(view["prediction_rgba"])
                target, target_mask = load_rgba_composited(view["target_rgba"])
                metrics = image_metrics(pred, target, pred_mask, target_mask)
                view_record = {**view, "metrics": metrics}
                comparison = Path(view["prediction_rgba"]).parent / "comparison.png"
                comparison_image(task, view, comparison, metrics)
                view_record["comparison"] = str(comparison)
                comparisons.append(comparison)
                views.append(view_record)
            object_rows.append(
                {
                    "dataset": manifest["dataset"],
                    "scene_id": scene["scene_id"],
                    "sample_id": scene["scene_id"],
                    "object_id": int(task["object_id"]),
                    "num_views": len(views),
                    "metrics": aggregate_metric_rows([row["metrics"] for row in views]),
                    "views": views,
                }
            )
        scene_coverage[scene["scene_id"]] = {
            "num_expected_objects": expected,
            "num_rendered_objects": len(tasks),
            "coverage": len(tasks) / expected if expected else 1.0,
        }
        scene_overview(
            comparisons,
            output_root / scene["scene_id"] / "appearance_overview.png",
        )

    aggregation = aggregate_objects_two_level(object_rows, sample_key="scene_id")
    by_scene = {
        scene["scene_id"]: aggregate_metric_rows(
            [row["metrics"] for row in object_rows if row["scene_id"] == scene["scene_id"]]
        )
        for scene in scenes
    }
    complete = all(item["coverage"] == 1.0 for item in scene_coverage.values())
    summary = {
        "schema": "ff_scene_appearance_benchmark_v1",
        "protocol": {
            "dataset": manifest["dataset"],
            "manifest": str(args.manifest.resolve()),
            "scope": "appearance_oracle_object_id_and_pose",
            "reference": "source RGB masked by GT instance ID and composited white",
            "prediction": "isolated canonical textured GLB at GT pose and exact source camera",
            "decoded_glb_frame": (
                "Blender glTF import followed by X-minus-90 back to FF canonical, "
                "then GT object-to-world pose"
            ),
            "view_selection": "deterministic highest visible-pixel source views",
            "views_per_object": args.views_per_object,
            "recipe": args.recipe,
            "metric_source": "eval/rendering/pbr_render_metrics.py",
            "aggregation": "views-to-object, objects-to-scene, then scene macro",
        },
        "complete": complete,
        "coverage": scene_coverage,
        "scene_metrics": by_scene,
        "object_weighted_metrics": aggregation["object_weighted"],
        "scene_macro_metrics": aggregation["sample_weighted"],
        "num_objects": aggregation["num_objects"],
        "num_scenes": aggregation["num_samples"],
        "objects": object_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "appearance_metrics.json").write_text(
        json.dumps(jsonable(summary), indent=2) + "\n"
    )
    write_csv(output_root / "appearance_metrics.csv", object_rows)
    print(
        json.dumps(
            {
                "complete": complete,
                "num_scenes": aggregation["num_samples"],
                "num_objects": aggregation["num_objects"],
                "scene_macro_metrics": aggregation["sample_weighted"],
            },
            indent=2,
        )
    )
    if not complete and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
