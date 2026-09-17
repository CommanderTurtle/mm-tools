from __future__ import annotations

"""Build frozen geometry-only manifests for iTHOR and Imaginarium."""

import argparse
import json
import pickle
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.scene_reconstruction.geometry_protocol import (  # noqa: E402
    OBJECT_MESH_RE,
    jsonable,
    sha256_file,
)


DATASET_SPECS = {
    "ithor": {
        "dataset_subdir": "ithor",
        "selection_file": "whitelist.txt",
        "selection_mode": "include",
        "legacy_index_space": "selected",
    },
    "imaginarium": {
        "dataset_subdir": "Imaginarium",
        "selection_file": "blacklist.txt",
        "selection_mode": "exclude",
        "legacy_index_space": "raw",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
        help="Directory containing the ithor and Imaginarium dataset roots.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "manifests",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        choices=tuple(DATASET_SPECS),
        help="Dataset(s) to build; defaults to both.",
    )
    parser.add_argument("--video-id", type=int, default=0)
    parser.add_argument("--min-visible-pixels", type=int, default=1)
    parser.add_argument("--mask-workers", type=int, default=16)
    return parser.parse_args()


def read_mask_counts(path: Path) -> Counter[int]:
    with np.load(path) as archive:
        if "mask" in archive:
            mask = archive["mask"]
        elif "arr_0" in archive:
            mask = archive["arr_0"]
        else:
            raise KeyError(f"Unsupported mask archive keys in {path}: {list(archive.keys())}")
    ids, counts = np.unique(np.asarray(mask, dtype=np.int64), return_counts=True)
    return Counter({int(object_id): int(count) for object_id, count in zip(ids, counts) if object_id > 0})


def visible_pixel_counts(mask_paths: list[Path], workers: int) -> Counter[int]:
    counts: Counter[int] = Counter()
    if workers <= 1:
        per_frame = map(read_mask_counts, mask_paths)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        per_frame = executor.map(read_mask_counts, mask_paths)
    try:
        for frame_counts in per_frame:
            counts.update(frame_counts)
    finally:
        if workers > 1:
            executor.shutdown(wait=True)
    return counts


def sorted_files(path: Path, suffix: str) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == suffix)


def load_transform_object_ids(path: Path) -> list[int]:
    # The benchmark annotations are trusted local files.
    with path.open("rb") as handle:
        transforms = pickle.load(handle)
    if not isinstance(transforms, dict):
        raise TypeError(f"Expected transform dictionary in {path}")
    ids = []
    for name in transforms:
        match = re.fullmatch(r"object_(\d{4})", str(name))
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def numbered_mesh_ids(mesh_dir: Path) -> list[int]:
    ids = []
    for path in mesh_dir.iterdir():
        match = OBJECT_MESH_RE.fullmatch(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def selected_scene_ids(dataset_root: Path, spec: dict[str, str]) -> tuple[list[str], list[str], set[str]]:
    renders_root = dataset_root / "renders"
    raw_scene_ids = sorted(path.name for path in renders_root.iterdir() if path.is_dir())
    selection_path = dataset_root / spec["selection_file"]
    selection = {line.strip() for line in selection_path.read_text().splitlines() if line.strip()}
    if spec["selection_mode"] == "include":
        selected = [scene_id for scene_id in raw_scene_ids if scene_id in selection]
    else:
        selected = [scene_id for scene_id in raw_scene_ids if scene_id not in selection]
    return raw_scene_ids, selected, selection


def build_scene_record(
    *,
    dataset_root: Path,
    scene_id: str,
    raw_index: int,
    benchmark_index: int,
    legacy_eval_index: int,
    video_id: int,
    min_visible_pixels: int,
    mask_workers: int,
) -> dict[str, Any]:
    render_dir = dataset_root / "renders" / scene_id
    frames_dir = render_dir / f"{video_id}_frames"
    depths_dir = render_dir / f"{video_id}_depth"
    masks_dir = render_dir / f"{video_id}_masks"
    camera_path = render_dir / f"{video_id}.json"
    transform_path = dataset_root / "transforms" / f"{scene_id}.pkl"
    mesh_dir = dataset_root / "scenes" / scene_id

    required = [frames_dir, depths_dir, masks_dir, camera_path, transform_path, mesh_dir]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"{scene_id} is missing required paths: {missing}")

    frame_paths = sorted_files(frames_dir, ".jpg")
    depth_paths = sorted_files(depths_dir, ".npz")
    mask_paths = sorted_files(masks_dir, ".npz")
    if not frame_paths or not (len(frame_paths) == len(depth_paths) == len(mask_paths)):
        raise ValueError(
            f"{scene_id} RGB/depth/mask count mismatch: "
            f"{len(frame_paths)}/{len(depth_paths)}/{len(mask_paths)}"
        )

    gt_object_ids = load_transform_object_ids(transform_path)
    mesh_object_ids = numbered_mesh_ids(mesh_dir)
    if gt_object_ids != mesh_object_ids:
        missing_meshes = sorted(set(gt_object_ids) - set(mesh_object_ids))
        extra_meshes = sorted(set(mesh_object_ids) - set(gt_object_ids))
        raise ValueError(
            f"{scene_id} transform/mesh mismatch; missing={missing_meshes}, extra={extra_meshes}"
        )

    pixel_counts = visible_pixel_counts(mask_paths, mask_workers)
    unknown_mask_ids = sorted(set(pixel_counts) - set(gt_object_ids))
    if unknown_mask_ids:
        raise ValueError(f"{scene_id} masks contain unknown positive object IDs: {unknown_mask_ids}")
    visible_ids = [
        object_id
        for object_id in gt_object_ids
        if pixel_counts.get(object_id, 0) >= min_visible_pixels
    ]

    def relative(path: Path) -> str:
        return str(path.relative_to(dataset_root))

    return {
        "scene_id": scene_id,
        "benchmark_index": benchmark_index,
        "raw_index": raw_index,
        "legacy_eval_index": legacy_eval_index,
        "video_id": video_id,
        "camera_path": relative(camera_path),
        "frames_dir": relative(frames_dir),
        "depths_dir": relative(depths_dir),
        "masks_dir": relative(masks_dir),
        "transforms_path": relative(transform_path),
        "mesh_dir": relative(mesh_dir),
        "num_frames": len(frame_paths),
        "num_gt_objects": len(gt_object_ids),
        "num_visible_objects": len(visible_ids),
        "gt_object_ids": gt_object_ids,
        "visible_object_ids": visible_ids,
        "visible_pixel_counts": {
            str(object_id): int(pixel_counts[object_id]) for object_id in visible_ids
        },
    }


def build_manifest(
    *,
    fire3d_test_root: Path,
    dataset: str,
    video_id: int,
    min_visible_pixels: int,
    mask_workers: int,
) -> dict[str, Any]:
    spec = DATASET_SPECS[dataset]
    dataset_root = fire3d_test_root / spec["dataset_subdir"]
    raw_scene_ids, selected_ids, selection = selected_scene_ids(dataset_root, spec)
    raw_index = {scene_id: index for index, scene_id in enumerate(raw_scene_ids)}

    scenes = []
    for benchmark_index, scene_id in enumerate(selected_ids):
        legacy_eval_index = (
            benchmark_index if spec["legacy_index_space"] == "selected" else raw_index[scene_id]
        )
        print(f"[{dataset}] {benchmark_index + 1}/{len(selected_ids)} {scene_id}", flush=True)
        scenes.append(
            build_scene_record(
                dataset_root=dataset_root,
                scene_id=scene_id,
                raw_index=raw_index[scene_id],
                benchmark_index=benchmark_index,
                legacy_eval_index=legacy_eval_index,
                video_id=video_id,
                min_visible_pixels=min_visible_pixels,
                mask_workers=mask_workers,
            )
        )

    selection_path = dataset_root / spec["selection_file"]
    manifest = {
        "schema": "ff_scene_geometry_manifest_v1",
        "protocol": "ff_ithor_imaginarium_geometry_v1",
        "dataset": dataset,
        "dataset_subdir": spec["dataset_subdir"],
        "video_id": video_id,
        "selection": {
            "mode": spec["selection_mode"],
            "file": spec["selection_file"],
            "sha256": sha256_file(selection_path),
            "num_entries": len(selection),
            "legacy_index_space": spec["legacy_index_space"],
        },
        "object_set": {
            "primary": "visible",
            "visibility_source": "positive GT instance-mask pixels across the selected video",
            "min_visible_pixels": min_visible_pixels,
        },
        "totals": {
            "num_raw_scenes": len(raw_scene_ids),
            "num_selected_scenes": len(scenes),
            "num_gt_objects": sum(scene["num_gt_objects"] for scene in scenes),
            "num_visible_objects": sum(scene["num_visible_objects"] for scene in scenes),
            "num_frames": sum(scene["num_frames"] for scene in scenes),
        },
        "scenes": scenes,
    }
    return manifest


def main() -> None:
    args = parse_args()
    if args.video_id < 0:
        raise SystemExit("--video-id must be non-negative")
    if args.min_visible_pixels <= 0:
        raise SystemExit("--min-visible-pixels must be positive")
    if args.mask_workers <= 0:
        raise SystemExit("--mask-workers must be positive")

    datasets = args.dataset or list(DATASET_SPECS)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        manifest = build_manifest(
            fire3d_test_root=args.fire3d_test_root.resolve(),
            dataset=dataset,
            video_id=args.video_id,
            min_visible_pixels=args.min_visible_pixels,
            mask_workers=args.mask_workers,
        )
        output_path = args.output_dir / f"{dataset}_v1.json"
        output_path.write_text(json.dumps(jsonable(manifest), indent=2) + "\n")
        print(json.dumps(manifest["totals"], indent=2))
        print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
