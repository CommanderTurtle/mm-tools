#!/usr/bin/env python3
"""Unified render-view sampling: emit a camera config for the unified renderer.

One sampler for all datasets whose ONLY output is a YAML in exactly the schema
`eval/unified_render.py --config` consumes (`cameras.source: explicit` with
eye/lookat/up frames + intrinsics + resolution), plus a `sampling:` provenance
block the renderer ignores. Sample once per scene, reuse the same YAML across
every source (ours predicted / ours GT / GT scene / baselines) so all sheets
are row-aligned on identical views.

Profiles:
  dataset-random   random subset of the dataset's own recorded cameras
                   (needs nothing but the camera file; deterministic by seed).
  balanced         the validated Aug-28 sampled-grid policy (source cameras +
                   GT visibility statistics).
  high-overview    the validated Aug-28 elevated-room-overview policy
                   (20260828_213614): >=55th height percentile anchors,
                   coverage/dominance scoring, azimuth diversity, pitch
                   clamped to [30, 68] degrees, GT-median room framing.
                   Selection never inspects predicted renders.

balanced/high-overview reuse the validated implementation in
eval/rendering/visualize_percept_recon_mesh_comparison.py (imported, not reimplemented)
and are available for ithor/imaginarium, whose GT masks feed the visibility
statistics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.unified_protocol import (  # noqa: E402
    load_protocol,
    native_dataset_environment,
    protocol_identity,
    protocol_render_config,
    validate_scope,
)


PROFILES = ("dataset-random", "balanced", "high-overview", "protocol")
DATASETS = ("ithor", "imaginarium", "scannetpp", "single_image")
FIRE3D_SUBDIRS = {"ithor": "ithor", "imaginarium": "Imaginarium"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--profile", choices=PROFILES, default="dataset-random")
    parser.add_argument(
        "--protocol",
        default=None,
        help="Frozen protocol name or JSON path; required by --profile protocol.",
    )
    parser.add_argument("--num-views", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fire3d-test-root", type=Path, default=REPO_ROOT / "data"
    )
    parser.add_argument("--video-id", type=int, default=0)
    parser.add_argument(
        "--obb-source",
        type=Path,
        default=None,
        help=(
            "Reconstruction scene directory (the one holding "
            "appearance/appearance_summary.json). Required for scannetpp "
            "high-overview: that dataset has no GT instance masks, so the "
            "per-frame view scores are derived from the predicted objects "
            "instead."
        ),
    )
    parser.add_argument(
        "--render-width", type=int, default=None,
        help="Output render width; K is adapted preserving the vertical FOV. "
        "Default: 960 for multi-view datasets, NATIVE for single_image.",
    )
    parser.add_argument(
        "--render-height", type=int, default=None,
        help="Output render height. Default: 540 (16:9 figure protocol) for "
        "multi-view datasets, NATIVE for single_image.",
    )
    args = parser.parse_args()
    if args.dataset == "single_image":
        # The single_image protocol IS the input-image view: its GT mesh,
        # point cloud, and our percept+recon all live in the loader's exact
        # cropped camera frame. Retargeting to 16:9 would change the aspect,
        # intrinsics, and framing of the very image the scene was built from,
        # so native is the default and must be requested explicitly to change.
        if (args.render_width is None) != (args.render_height is None):
            parser.error("pass both --render-width and --render-height, or neither")
    else:
        if args.render_width is None:
            args.render_width = 960
        if args.render_height is None:
            args.render_height = 540
    if (
        args.profile == "high-overview"
        and args.dataset == "scannetpp"
        and args.obb_source is None
    ):
        parser.error(
            "scannetpp has no GT instance masks, so high-overview needs "
            "--obb-source <reconstruction scene dir> to score views from the "
            "predicted objects"
        )
    if (
        args.profile != "dataset-random"
        and args.profile != "protocol"
        and args.dataset not in FIRE3D_SUBDIRS
        and not (args.dataset == "scannetpp" and args.profile == "high-overview")
    ):
        parser.error(
            f"profile {args.profile} needs GT visibility statistics; "
            f"dataset '{args.dataset}' supports dataset-random only (for now)"
        )
    if args.profile == "protocol" and args.protocol is None:
        parser.error("--profile protocol requires --protocol")
    return args


def _frames_from_c2ws(c2ws, indices, lookat_distance: float = 1.0):
    frames = []
    for index in indices:
        pose = np.asarray(c2ws[index], dtype=np.float64)
        eye, forward, up = pose[:3, 3], pose[:3, 2], -pose[:3, 1]
        frames.append({
            "eye": eye.tolist(),
            "lookat": (eye + float(lookat_distance) * forward).tolist(),
            "up": up.tolist(),
            "source_frame_index": int(index),
        })
    return frames


def _plain_floats(value):
    """Recursively convert numpy scalars/arrays to plain Python for yaml."""
    import numpy as _np

    if isinstance(value, _np.generic):
        return value.item()
    if isinstance(value, _np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain_floats(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_floats(item) for item in value]
    return value


def sample_dataset_random(args) -> tuple[list, list, int, int, dict]:
    if args.dataset in FIRE3D_SUBDIRS:
        from utils.read_frames import read_cameras

        scene_dir = (
            args.fire3d_test_root / FIRE3D_SUBDIRS[args.dataset]
            / "renders" / args.scene_id
        )
        camera_path = sorted(scene_dir.glob("*.json"))[args.video_id]
        intrinsics, c2ws, (height, width) = read_cameras(str(camera_path))
        rng = np.random.default_rng(args.seed)
        total = len(c2ws)
        indices = (
            list(range(total))
            if total <= args.num_views
            else sorted(rng.choice(total, size=args.num_views, replace=False).tolist())
        )
        frames = _frames_from_c2ws(np.asarray(c2ws), indices)
        diag = {"camera_file": str(camera_path), "frame_indices": indices}
        return np.asarray(intrinsics[0]).tolist(), frames, int(width), int(height), diag

    if args.dataset == "scannetpp":
        import os

        os.environ.setdefault("FF_SCANNETPP_WHITELIST", "0")
        from eval.rendering.render_scannetpp_recon_views import camera_frames, scene_alignment
        from utils.data_scannetpp import load_scannetpp_data, load_scene_frames

        _, intrinsics, c2ws, center, angle, alignment = scene_alignment(args.scene_id)
        rng = np.random.default_rng(args.seed)
        total = len(c2ws)
        indices = (
            list(range(total))
            if total <= args.num_views
            else sorted(rng.choice(total, size=args.num_views, replace=False).tolist())
        )
        frames = [
            {
                **{key: frame[key] for key in ("eye", "lookat", "up")},
                "source_frame_index": int(source_index),
            }
            for source_index, frame in zip(
                indices, camera_frames(c2ws, center, angle, indices)
            )
        ]
        entry = next(
            d for d in load_scannetpp_data() if d["scene_id"] == args.scene_id
        )
        rgbs, *_ = load_scene_frames(entry, max_num_frames=1)
        diag = {
            "frame_indices": indices,
            "max_frames": int(os.environ.get("FF_SCANNETPP_MAX_FRAMES", "300")),
            "wall_aligned": bool(alignment.get("enabled", True)),
            "scene_alignment": alignment,
        }
        return (
            np.asarray(intrinsics[0]).tolist(),
            frames,
            int(rgbs.shape[2]),
            int(rgbs.shape[1]),
            diag,
        )

    if args.dataset == "single_image":
        from utils.data_single_image import scene_camera

        cam = scene_camera(args.scene_id, native_resolution=True)
        frame = {
            **{key: cam["frame"][key] for key in ("eye", "lookat", "up")},
            "source_frame_index": 0,
        }
        return cam["K"], [frame], cam["width"], cam["height"], {"native_camera": True}

    raise ValueError(args.dataset)


def sample_protocol(args) -> tuple[list, list, int, int, dict, dict]:
    protocol, source = load_protocol(args.protocol)
    if protocol is None:
        raise ValueError("current_unified_v1 has no frozen source-view selection")
    validate_scope(protocol, args.dataset, [args.scene_id])
    os.environ.update(native_dataset_environment(protocol, args.dataset))
    view_config = protocol["views"]
    if args.dataset not in FIRE3D_SUBDIRS:
        expected_profile = {
            "single_image": "native-input-camera",
            "scannetpp": "dataset-random",
        }.get(args.dataset)
        if view_config.get("profile") != expected_profile:
            raise ValueError(
                f"Protocol {protocol['name']} view profile must be "
                f"{expected_profile!r} for {args.dataset}"
            )
        args.num_views = int(view_config["num_views"])
        args.seed = int(view_config.get("seed", args.seed))
        K, frames, width, height, diagnostics = sample_dataset_random(args)
        if len(frames) != args.num_views:
            raise ValueError(
                f"Protocol {protocol['name']} resolved {len(frames)} views, "
                f"expected {args.num_views}"
            )
        diagnostics.update(
            {
                "selection": view_config["profile"],
                "protocol": protocol_identity(args.protocol, protocol, source),
            }
        )
        return K, frames, width, height, diagnostics, protocol
    if args.dataset not in FIRE3D_SUBDIRS:
        raise ValueError("Frozen source-view protocols currently require Fire3D data")

    from utils.read_frames import read_cameras

    scene_dir = (
        args.fire3d_test_root
        / FIRE3D_SUBDIRS[args.dataset]
        / "renders"
        / args.scene_id
    )
    camera_path = sorted(scene_dir.glob("*.json"))[args.video_id]
    intrinsics, c2ws, (height, width) = read_cameras(str(camera_path))
    selection_diagnostics = None
    if "source_frame_indices" in view_config:
        indices = [int(value) for value in view_config["source_frame_indices"]]
        selection = "frozen_source_order"
    elif view_config.get("profile") == "source_coverage_diversity":
        from eval.rendering.visualize_percept_recon_mesh_comparison import select_source_views

        manifest_value = (protocol.get("scope") or {}).get(
            "scene_manifests", {}
        ).get(args.dataset)
        manifest_path = (
            Path(manifest_value)
            if manifest_value
            else Path("benchmarks/scene_reconstruction/manifests")
            / f"{args.dataset}_v1.json"
        )
        if not manifest_path.is_absolute():
            manifest_path = REPO_ROOT / manifest_path
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entry = next(
            (row for row in manifest["scenes"] if row["scene_id"] == args.scene_id),
            None,
        )
        if entry is None:
            raise ValueError(
                f"{args.scene_id} is not covered by {manifest_path.name}"
            )
        dataset_root = args.fire3d_test_root / FIRE3D_SUBDIRS[args.dataset]
        camera_payload = json.loads(camera_path.read_text(encoding="utf-8"))
        mask_paths = sorted((dataset_root / entry["masks_dir"]).iterdir())
        indices, selection_diagnostics = select_source_views(
            camera_payload,
            mask_paths,
            entry["visible_object_ids"],
            count=int(view_config["num_views"]),
        )
        selection = "source_coverage_diversity"
    else:
        raise ValueError(
            f"Protocol {protocol['name']} has no supported source-view selection"
        )
    if len(indices) != int(protocol["views"]["num_views"]):
        raise ValueError("Protocol view count does not match source_frame_indices")
    invalid = [index for index in indices if not 0 <= index < len(c2ws)]
    if invalid:
        raise ValueError(f"Protocol source frame indices out of range: {invalid}")
    frames = _frames_from_c2ws(
        np.asarray(c2ws),
        indices,
        lookat_distance=float(protocol["views"].get("lookat_distance", 1.0)),
    )
    diagnostics = {
        "camera_file": str(camera_path),
        "frame_indices": indices,
        "selection": selection,
        "selection_diagnostics": selection_diagnostics,
        "protocol": protocol_identity(args.protocol, protocol, source),
    }
    return (
        np.asarray(intrinsics[0]).tolist(),
        frames,
        int(width),
        int(height),
        diagnostics,
        protocol,
    )


def predicted_view_statistics(frames, K, width, height, centres, radii):
    """The three scorer quantities, from projected predicted objects.

    `build_view_statistics` measures visible_objects, visible_pixel_fraction and
    largest_object_pixel_fraction off GT instance masks. Here each predicted
    object is treated as a sphere at its `object_to_world` translation, and the
    same quantities come from projecting those spheres: an object counts as
    visible when its centre is in front of the camera and inside the frame, and
    its screen area is that of the projected disc. Approximating the union of
    discs by a capped sum is fine because the scorer only ranks frames.
    """

    import numpy as np

    fx, fy = float(K[0][0]), float(K[1][1])
    cx, cy = float(K[0][2]), float(K[1][2])
    frame_area = float(width * height)
    statistics = []
    for index, frame in enumerate(frames):
        eye = np.asarray(frame["eye"], dtype=np.float64)
        forward = np.asarray(frame["lookat"], dtype=np.float64) - eye
        forward /= max(np.linalg.norm(forward), 1e-9)
        up = np.asarray(frame["up"], dtype=np.float64)
        right = np.cross(forward, up)
        right /= max(np.linalg.norm(right), 1e-9)
        true_up = np.cross(right, forward)
        rotation = np.stack([right, -true_up, forward], axis=0)  # world -> OpenCV

        local = (centres - eye) @ rotation.T
        depth = local[:, 2]
        in_front = depth > 1e-6
        u = np.where(in_front, fx * local[:, 0] / np.where(in_front, depth, 1) + cx, -1e9)
        v = np.where(in_front, fy * local[:, 1] / np.where(in_front, depth, 1) + cy, -1e9)
        inside = in_front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        pixel_radius = np.where(
            in_front, radii * fx / np.where(in_front, depth, 1), 0.0
        )
        areas = np.pi * pixel_radius ** 2
        areas = np.where(inside, np.minimum(areas, frame_area), 0.0)
        visible_pixels = float(min(areas.sum(), frame_area))
        statistics.append(
            {
                "index": index,
                "visible_objects": int(inside.sum()),
                "visible_pixels": int(visible_pixels),
                "visible_pixel_fraction": visible_pixels / frame_area,
                "largest_object_pixel_fraction": float(areas.max(initial=0.0))
                / frame_area,
                "object_pixel_balance": (
                    1.0 - float(areas.max(initial=0.0)) / visible_pixels
                    if visible_pixels > 0
                    else 0.0
                ),
                "forward": forward,
                "eye": eye,
            }
        )
    return statistics


def sample_scannetpp_high_overview(args) -> tuple[list, list, int, int, dict]:
    """Aug-28 high-overview anchors for scannetpp, scored on predicted objects."""

    import numpy as np

    os.environ.setdefault("FF_SCANNETPP_WHITELIST", "0")
    from eval.rendering.render_scannetpp_recon_views import camera_frames, scene_alignment
    from eval.rendering.visualize_percept_recon_mesh_comparison import sample_novel_views
    from utils.data_scannetpp import load_scannetpp_data, load_scene_frames

    summary_path = args.obb_source / "appearance/appearance_summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"no appearance summary under --obb-source: {summary_path}")
    records = json.loads(summary_path.read_text())["composed_world_scene"]["objects"]
    centres, radii = [], []
    for record in records:
        transform = np.asarray(record["object_to_world"], dtype=np.float64)
        centres.append(transform[:3, 3])
        # the composed transform carries the object's normalised-cube scale, so
        # half its linear norm is a serviceable radius for screen-area scoring
        radii.append(0.5 * float(np.mean(np.linalg.norm(transform[:3, :3], axis=0))))
    if not centres:
        raise SystemExit(f"no composed objects in {summary_path}")
    centres = np.stack(centres)
    radii = np.asarray(radii)

    _, intrinsics, c2ws, center, angle, _ = scene_alignment(args.scene_id)
    frames = camera_frames(c2ws, center, angle, list(range(len(c2ws))))
    K = np.asarray(intrinsics[0], dtype=np.float64).tolist()
    entry = next(d for d in load_scannetpp_data() if d["scene_id"] == args.scene_id)
    rgbs, *_ = load_scene_frames(entry, max_num_frames=1)
    height, width = int(rgbs.shape[1]), int(rgbs.shape[2])

    statistics = predicted_view_statistics(
        frames, K, width, height, centres, radii
    )
    views = sample_novel_views(
        {"frames": frames},
        [],
        [],
        count=args.num_views,
        seed=args.seed,
        profile="high-overview",
        overview_target=np.median(centres, axis=0),
        statistics=statistics,
        total_objects=len(centres),
    )
    out_frames = [
        {key: list(map(float, view["frame"][key])) for key in ("eye", "lookat", "up")}
        for view in views
    ]
    diagnostics = {
        "statistics_source": "predicted_object_projection",
        "num_predicted_objects": int(len(centres)),
        "num_source_cameras": int(len(frames)),
        "overview_target": [float(v) for v in np.median(centres, axis=0)],
        "obb_source": str(args.obb_source),
        "note": (
            "scannetpp has no GT instance masks; view scores come from the "
            "predicted objects, so this is NOT the GT-scored high-overview used "
            "on ithor/imaginarium"
        ),
    }
    return K, out_frames, width, height, diagnostics


def sample_profiled(args) -> tuple[list, list, int, int, dict]:
    """balanced / high-overview via the validated Aug-28 implementation."""
    import pickle

    from eval.rendering.scene_assets import (
        load_scene_transforms,
    )
    from eval.rendering.rerender_imaginarium_exact_camera_rgb import compose_matrix
    from eval.rendering.visualize_percept_recon_mesh_comparison import sample_novel_views
    from utils.read_frames import read_cameras

    subdir = FIRE3D_SUBDIRS[args.dataset]
    manifest = json.loads(
        (
            REPO_ROOT
            / "benchmarks/scene_reconstruction/manifests"
            / f"{args.dataset}_v1.json"
        ).read_text()
    )
    scene = next(
        (row for row in manifest["scenes"] if row["scene_id"] == args.scene_id), None
    )
    if scene is None:
        raise SystemExit(f"{args.scene_id} not in {args.dataset}_v1.json")
    dataset_root = args.fire3d_test_root / subdir
    camera_path = dataset_root / scene["camera_path"]
    intrinsics, c2ws, (height, width) = read_cameras(str(camera_path))
    camera_payload = json.loads(camera_path.read_text())
    if "frames" not in camera_payload:
        raise SystemExit(f"camera file has no frames block: {camera_path}")

    visible_ids = [int(v) for v in scene["visible_object_ids"]]
    masks_dir = dataset_root / scene["masks_dir"]
    mask_paths = sorted(masks_dir.iterdir())

    transforms = load_scene_transforms(dataset_root / scene["transforms_path"])
    translations = [
        np.asarray(compose_matrix(transforms[f"object_{oid:04d}"]))[:3, 3]
        for oid in visible_ids
        if f"object_{oid:04d}" in transforms
    ]
    overview_target = (
        np.median(np.stack(translations), axis=0) if translations else None
    )

    views = sample_novel_views(
        camera_payload,
        mask_paths,
        visible_ids,
        count=args.num_views,
        seed=args.seed,
        profile=args.profile,
        overview_target=overview_target,
    )
    frames = [
        {key: list(map(float, view["frame"][key])) for key in ("eye", "lookat", "up")}
        for view in views
    ]
    diag = {
        "overview_target": None
        if overview_target is None
        else [float(v) for v in overview_target],
        "anchors": [view.get("anchor_frame_index") for view in views],
        "camera_file": str(camera_path),
    }
    return np.asarray(intrinsics[0]).tolist(), frames, int(width), int(height), diag


def _adapt_intrinsics_16x9(K, width, height, out_width, out_height):
    """Retarget K to a new viewport, preserving the VERTICAL field of view.

    The bg16x9 figure protocol: fy scales with the height ratio, fx keeps
    square pixels (widening the horizontal FOV instead of stretching), and the
    principal point recenters. Matches the vertical-FOV viewport used by the
    final-comparison sheets.
    """
    K = np.asarray(K, dtype=np.float64)
    scale = out_height / float(height)
    fy = float(K[1, 1] * scale)
    return [
        [fy, 0.0, float(out_width) / 2.0],
        [0.0, fy, float(out_height) / 2.0],
        [0.0, 0.0, 1.0],
    ]


def main() -> None:
    args = parse_args()
    protocol = None
    if args.profile == "protocol":
        K, frames, width, height, diag, protocol = sample_protocol(args)
        target_width = protocol["views"].get("width")
        target_height = protocol["views"].get("height")
        if (target_width is None) != (target_height is None):
            raise ValueError("Protocol views must set both width and height, or neither")
        if target_width is not None:
            args.render_width = int(target_width)
            args.render_height = int(target_height)
    elif args.profile == "dataset-random":
        K, frames, width, height, diag = sample_dataset_random(args)
    elif args.dataset == "scannetpp":
        # no GT masks there; scores come from the predicted objects
        K, frames, width, height, diag = sample_scannetpp_high_overview(args)
    else:
        K, frames, width, height, diag = sample_profiled(args)

    if (
        args.render_width is not None
        and args.render_height is not None
        and (args.render_width, args.render_height) != (width, height)
    ):
        K = _adapt_intrinsics_16x9(K, width, height, args.render_width, args.render_height)
        diag["native_resolution"] = [width, height]
        diag["intrinsics_policy"] = "vertical_fov_preserved_square_pixels"
        width, height = args.render_width, args.render_height

    config = {
        "resolution": {"width": width, "height": height},
        "intrinsics": {"K": K},
        "cameras": {"source": "explicit", "frames": frames},
        "sampling": {
            "schema": "ff_unified_view_sampling_v1",
            "dataset": args.dataset,
            "scene_id": args.scene_id,
            "profile": args.profile,
            "num_views": len(frames),
            "seed": args.seed,
            "camera_contract": (
                "native_input_camera_exact"
                if args.dataset == "single_image"
                and diag.get("native_resolution") is None
                else "retargeted_vertical_fov_preserved"
            ),
            "diagnostics": diag,
        },
    }
    if protocol is not None:
        protocol_config = protocol_render_config(protocol)
        config["protocols"] = protocol_config["protocols"]
        config["render"] = protocol_config["render"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import yaml

    args.output.write_text(yaml.safe_dump(_plain_floats(config), sort_keys=False))
    print(f"[views] {args.profile}: {len(frames)} frames -> {args.output}", flush=True)


if __name__ == "__main__":
    main()
