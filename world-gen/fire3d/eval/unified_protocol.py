"""Versioned protocol loading and argv construction for unified inference."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_DIR = REPO_ROOT / "configs/inference"
CURRENT_PROTOCOL = "current_unified_v1"
REPRODUCTION_PROTOCOL = "floorplan312_aug29_teacher12_reproduction_v1"


def available_protocols() -> tuple[str, ...]:
    names = [CURRENT_PROTOCOL]
    names.extend(path.stem for path in sorted(PROTOCOL_DIR.glob("*.json")))
    return tuple(dict.fromkeys(names))


def load_protocol(identifier: str | Path) -> tuple[dict[str, Any] | None, Path | None]:
    """Load a frozen protocol; current_unified_v1 preserves pre-profile defaults."""

    value = str(identifier)
    if value == CURRENT_PROTOCOL:
        return None, None
    candidate = Path(value)
    if not candidate.is_file():
        candidate = PROTOCOL_DIR / f"{value}.json"
    if not candidate.is_file():
        raise ValueError(
            f"Unknown protocol {value!r}; choose one of {available_protocols()} "
            "or pass a JSON path"
        )
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    if payload.get("schema") != "ff_unified_pipeline_protocol_v1":
        raise ValueError(f"Unsupported protocol schema in {candidate}")
    if not payload.get("name"):
        raise ValueError(f"Protocol has no name: {candidate}")
    return payload, candidate.resolve()


def validate_scope(
    protocol: dict[str, Any] | None,
    dataset: str,
    scene_ids: list[str],
) -> None:
    if protocol is None:
        return
    scope = protocol.get("scope") or {}
    expected_datasets = scope.get("datasets")
    if expected_datasets and dataset not in expected_datasets:
        raise ValueError(
            f"Protocol {protocol['name']} supports datasets "
            f"{expected_datasets!r}, not {dataset!r}"
        )
    expected_dataset = scope.get("dataset")
    if expected_dataset and dataset != expected_dataset:
        raise ValueError(
            f"Protocol {protocol['name']} is frozen for dataset "
            f"{expected_dataset!r}, not {dataset!r}"
        )
    expected_scene = scope.get("scene_id")
    if expected_scene and scene_ids != [expected_scene]:
        raise ValueError(
            f"Protocol {protocol['name']} is frozen for exactly scene "
            f"{expected_scene!r}; received {scene_ids!r}"
        )


def native_loader_option_environment(
    input_config: dict[str, Any], dataset: str
) -> dict[str, str]:
    """Translate structured native-loader options into loader environment values.

    The underlying dataset modules are also used by legacy entrypoints, where
    environment variables remain the compatibility boundary. Protocols use
    typed JSON options so their data processing is visible and validated.
    """

    options = input_config.get("loader_options")
    if options is None:
        return {}
    if not isinstance(options, dict):
        raise ValueError("input_rgb.loader_options must be an object")

    allowed = {
        "single_image": {"wall_alignment"},
        "scannetpp": {
            "depth_confidence_threshold",
            "depth_invalid_policy",
            "depth_subdir",
            "max_frames",
            "wall_alignment",
        },
    }.get(dataset, set())
    unknown = sorted(set(options) - allowed)
    if unknown:
        raise ValueError(
            f"Unsupported {dataset} loader options: {unknown}; "
            f"expected only {sorted(allowed)}"
        )

    resolved: dict[str, str] = {}
    if "wall_alignment" in options:
        enabled = options["wall_alignment"]
        if not isinstance(enabled, bool):
            raise ValueError(
                f"{dataset} loader option wall_alignment must be boolean"
            )
        variable = {
            "single_image": "FF_SINGLE_IMAGE_WALL_ALIGN",
            "scannetpp": "FF_SCANNETPP_WALL_ALIGN",
        }[dataset]
        resolved[variable] = "1" if enabled else "0"

    if "max_frames" in options:
        max_frames = options["max_frames"]
        if isinstance(max_frames, bool) or not isinstance(max_frames, int):
            raise ValueError("scannetpp loader option max_frames must be an integer")
        if max_frames <= 0:
            raise ValueError("scannetpp loader option max_frames must be positive")
        resolved["FF_SCANNETPP_MAX_FRAMES"] = str(max_frames)

    if "depth_subdir" in options:
        depth_subdir = options["depth_subdir"]
        if not isinstance(depth_subdir, str):
            raise ValueError("scannetpp loader option depth_subdir must be a string")
        if (
            depth_subdir in {"", ".", ".."}
            or os.path.isabs(depth_subdir)
            or os.path.basename(depth_subdir) != depth_subdir
        ):
            raise ValueError(
                "scannetpp loader option depth_subdir must be one relative "
                "directory name"
            )
        resolved["FF_SCANNETPP_DEPTH_SUBDIR"] = depth_subdir

    if "depth_invalid_policy" in options:
        invalid_policy = options["depth_invalid_policy"]
        if invalid_policy not in {"nearest_fill", "drop"}:
            raise ValueError(
                "scannetpp loader option depth_invalid_policy must be "
                "'nearest_fill' or 'drop'"
            )
        resolved["FF_SCANNETPP_DEPTH_INVALID_POLICY"] = invalid_policy

    if "depth_confidence_threshold" in options:
        threshold = options["depth_confidence_threshold"]
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(
                "scannetpp loader option depth_confidence_threshold must be a "
                "number in [0,1]"
            )
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "scannetpp loader option depth_confidence_threshold must be in [0,1]"
            )
        resolved["FF_SCANNETPP_DEPTH_CONFIDENCE_THRESHOLD"] = str(threshold)
    return resolved


def native_dataset_environment(
    protocol: dict[str, Any] | None,
    dataset: str,
    root_override: Path | None = None,
) -> dict[str, str]:
    """Resolve a frozen native-loader input contract into environment values.

    iTHOR and Imaginarium protocols prepare an exact-camera RGB overlay under a
    Fire3D test root.  Single-image and ScanNet++ instead own their complete
    RGB-D/point-cloud layouts, so their existing loaders are configured through
    dataset-specific environment variables.  Keeping that resolution here lets
    the driver, standalone view sampler, and standalone renderer share the same
    contract.
    """

    if protocol is None:
        return {}
    input_config = protocol.get("input_rgb") or {}
    if input_config.get("mode") != "dataset_native":
        return {}
    configured_dataset = input_config.get("dataset")
    if configured_dataset != dataset:
        raise ValueError(
            f"Protocol {protocol['name']} native input is for "
            f"{configured_dataset!r}, not {dataset!r}"
        )
    root_env = str(input_config["root_env"])
    root = (
        Path(root_override)
        if root_override is not None
        else Path(os.environ.get(root_env, input_config["default_root"]))
    )
    resolved = {
        str(key): str(value)
        for key, value in (input_config.get("environment") or {}).items()
    }
    resolved.update(native_loader_option_environment(input_config, dataset))
    resolved[root_env] = str(root.resolve())
    return resolved


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def protocol_identity(
    name: str,
    protocol: dict[str, Any] | None,
    source: Path | None,
) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "requested_name": name,
        "resolved_name": protocol["name"] if protocol else CURRENT_PROTOCOL,
        "source": str(source) if source else None,
        "source_sha256": file_sha256(source) if source else None,
    }
    if protocol:
        identity["schema"] = protocol["schema"]
        identity["version"] = protocol.get("version")
        identity["status"] = protocol.get("status")
    else:
        identity["baseline_git_commit"] = "7962026"
    return identity


def protocol_perception_mode(protocol: dict[str, Any] | None) -> str:
    """Return the protocol's perception source; legacy protocols are predicted."""

    mode = "predicted" if protocol is None else (protocol.get("perception") or {}).get(
        "mode", "predicted"
    )
    if mode not in {"predicted", "gt"}:
        name = protocol.get("name", "<unnamed>") if protocol else CURRENT_PROTOCOL
        raise ValueError(f"Protocol {name} has unsupported perception mode {mode!r}")
    return mode


def _bool_flag(flag: str, enabled: bool) -> str:
    return flag if enabled else f"--no-{flag.removeprefix('--')}"


def _bundle_path(bundle: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else bundle / path


def reproduction_perception_args(protocol: dict[str, Any]) -> list[str]:
    config = protocol["perception"]
    args = [
        "--nms_iou_threshold",
        str(config["nms_iou_threshold"]),
        "--valid_score_threshold",
        str(config["valid_score_threshold"]),
        (
            "--clean_instance_labels_by_obbs"
            if config["clean_instance_labels_by_obbs"]
            else "--no_clean_instance_labels_by_obbs"
        ),
    ]
    feature = config.get("feature_extraction")
    if feature is not None:
        args.extend(
            [
                "--dino_upsample",
                str(feature["dino_upsample"]),
                "--anyup_frame_batch_size",
                str(feature["anyup_frame_batch_size"]),
                "--rgb_upsample",
                str(feature["rgb_upsample"]),
                "--rgb_upsampler",
                str(feature["rgb_upsampler"]),
                "--feature_subsample",
                str(feature["feature_subsample"]),
            ]
        )
        if feature.get("dino_frame_batch_size") is not None:
            args.extend(
                [
                    "--dino_frame_batch_size",
                    str(feature["dino_frame_batch_size"]),
                ]
            )
    return args


def reproduction_geometry_args(
    protocol: dict[str, Any], repo_root: Path = REPO_ROOT
) -> list[str]:
    """Return all output-affecting reconstruction flags for the frozen run."""

    recon = protocol["reconstruction"]
    flow = recon["flow"]
    vae = recon["vae"]
    conditioning = recon["conditioning"]
    occupancy = recon["occupancy"]
    background = recon["background"]
    geometry = recon["geometry_stage"]
    decode = recon["appearance_decode"]
    mesh = recon["mesh_postprocess"]
    execution = recon["execution"]
    bundle = _bundle_path(repo_root, recon["bundle"])

    args: list[str] = [
        "--perception-label-transfer",
        conditioning["perception_label_transfer"],
        "--perception-voxel-resolution",
        str(conditioning["perception_voxel_resolution"]),
        "--perception-scene-scale",
        str(conditioning["perception_scene_scale"]),
        _bool_flag("--background-room-box-prior", background["room_box_prior"]),
        _bool_flag(
            "--background-unit-box-prune",
            background.get("unit_box_prune", True),
        ),
        "--background-canonical-transform",
        background.get("canonical_transform", "perception_obb"),
        "--background-canonical-margin",
        str(background.get("canonical_margin", 0.01)),
        "--background-room-box-distance",
        str(background["distance"]),
        "--background-room-box-adaptive-max-distance",
        str(background["adaptive_max_distance"]),
        "--background-room-box-trim-quantile",
        str(background["trim_quantile"]),
        "--background-room-box-yaw-samples",
        str(background["yaw_samples"]),
        "--background-room-box-min-points",
        str(background["min_points"]),
    ]
    if "prune_oversized_instances" in background:
        args.extend(
            [
                _bool_flag(
                    "--prune-oversized-instances",
                    background["prune_oversized_instances"],
                ),
                "--oversized-instance-volume-ratio",
                str(background["oversized_instance_volume_ratio"]),
            ]
        )

    for stage in ("ss", "shape", "pbr"):
        stage_config = flow[stage]
        args.extend(
            [
                f"--{stage}-run-dir",
                str(bundle / "flows" / stage),
                f"--{stage}-checkpoint",
                str(_bundle_path(bundle, stage_config["checkpoint"])),
                f"--{stage}-model-family",
                stage_config["model_family"],
                _bool_flag(f"--{stage}-use-ema", stage_config["use_ema"]),
            ]
        )

    args.extend(
        [
            "--predict-appearance",
            _bool_flag("--geometry-stage-mesh-decode", geometry["mesh_decode"]),
            _bool_flag(
                "--export-world-object-meshes",
                geometry["export_world_object_meshes"],
            ),
        ]
    )
    for stage in ("ss", "shape", "pbr"):
        stage_config = vae[stage]
        args.extend(
            [
                f"--{stage}-vae-root",
                str(_bundle_path(bundle, stage_config["root"])),
                f"--{stage}-vae-checkpoint",
                stage_config["checkpoint"],
                _bool_flag(f"--{stage}-vae-use-ema", stage_config["use_ema"]),
            ]
        )

    args.extend(
        [
            "--shape-decoder-pretrained",
            str(_bundle_path(repo_root, recon["decoders"]["shape"])),
            "--pbr-decoder-pretrained",
            str(_bundle_path(repo_root, recon["decoders"]["pbr"])),
            "--dino-repo-dir",
            str(_bundle_path(repo_root, protocol["perception"]["dino_repo"])),
            "--dino-model-path",
            str(_bundle_path(repo_root, protocol["perception"]["dino_model"])),
            "--anyup-repo-dir",
            str(repo_root / "third_party/anyup"),
            "--max-cond-len",
            str(flow["max_cond_len"]),
            "--pbr-max-cond-len",
            str(flow["pbr_max_cond_len"]),
            "--sample-method",
            flow["sample_method"],
            "--pbr-sample-method",
            flow["pbr_sample_method"],
            "--dino-upsample",
            str(conditioning["dino_upsample"]),
            "--anyup-frame-batch-size",
            str(conditioning["anyup_frame_batch_size"]),
            "--inference-num-steps",
            str(flow["inference_num_steps"]),
            "--pbr-guidance-strength",
            str(flow["pbr_guidance_strength"]),
            "--object-batch-size",
            str(flow["object_batch_size"]),
            "--seed",
            str(flow["seed"]),
            _bool_flag("--pbr-condition-snap", conditioning["pbr_condition_snap"]),
            "--occupancy-threshold",
            str(occupancy["threshold"]),
            "--empty-occupancy-policy",
            occupancy["empty_policy"],
            "--mesh-decode-batch-size",
            str(geometry["mesh_decode_batch_size"]),
            "--shape-decode-resolution",
            str(geometry["shape_decode_resolution"]),
            "--shape-mesh-decimation-target",
            str(geometry["shape_mesh_decimation_target"]),
            _bool_flag("--shape-mesh-remesh", geometry["shape_mesh_remesh"]),
            "--shape-mesh-remesh-band",
            str(geometry["shape_mesh_remesh_band"]),
            "--shape-mesh-remesh-project",
            str(geometry["shape_mesh_remesh_project"]),
            "--appearance-texture-size",
            str(mesh["texture_size"]),
            "--appearance-decimation-target",
            str(mesh["decimation_target"]),
            "--appearance-background-decimation-multiplier",
            str(mesh["background_decimation_multiplier"]),
            "--appearance-decode-object-chunk-size",
            str(decode["object_chunk_size"]),
            _bool_flag(
                "--appearance-persist-raw-decoded",
                decode["persist_raw_decoded"],
            ),
            "--appearance-mesh-batch-size",
            str(mesh["object_batch_size"]),
            "--appearance-mesh-backend",
            mesh["backend"],
            "--appearance-topology-initial-target-multiplier",
            str(mesh["topology_initial_target_multiplier"]),
            "--appearance-topology-simplify-threshold",
            str(mesh["topology_simplify_threshold"]),
            "--appearance-topology-chart-area-penalty",
            str(mesh["topology_chart_area_penalty"]),
            _bool_flag("--appearance-topology-remesh", mesh["topology_remesh"]),
            "--appearance-topology-remesh-band",
            str(mesh["topology_remesh_band"]),
            "--appearance-topology-remesh-project",
            str(mesh["topology_remesh_project"]),
            "--appearance-surface-mapping",
            mesh["surface_mapping"],
            "--appearance-sparse-query-mode",
            mesh["sparse_query_mode"],
            "--appearance-texture-fill-mode",
            mesh["texture_fill_mode"],
            "--appearance-texture-dilation-pixels",
            str(mesh["texture_dilation_pixels"]),
            "--appearance-texture-erode-iterations",
            str(mesh["texture_erode_iterations"]),
            "--appearance-projection-raster-instance-batch-size",
            str(mesh["projection_raster_instance_batch_size"]),
            _bool_flag("--resume-existing", execution["resume_existing"]),
            _bool_flag("--continue-on-error", execution["continue_on_error"]),
            _bool_flag("--model-cache", execution["model_cache"]),
            "--object-selection",
            execution.get("object_selection", "all"),
        ]
    )
    if "ss_shape_object_batch_size" in flow:
        args.extend(
            [
                "--ss-shape-object-batch-size",
                str(flow["ss_shape_object_batch_size"]),
            ]
        )
    if "pbr_object_batch_size" in flow:
        args.extend(
            ["--pbr-object-batch-size", str(flow["pbr_object_batch_size"])]
        )
    if "uv_cpu_workers" in mesh:
        args.extend(
            ["--appearance-uv-cpu-workers", str(mesh["uv_cpu_workers"])]
        )
    if "detailed_profile" in execution:
        args.append(
            _bool_flag(
                "--appearance-detailed-profile",
                execution["detailed_profile"],
            )
        )
    if "xatlas_block_align" in mesh:
        args.append(
            _bool_flag(
                "--appearance-xatlas-block-align",
                mesh["xatlas_block_align"],
            )
        )
    if "read_workers" in execution:
        args.extend(["--read-workers", str(execution["read_workers"])])
    return args


def protocol_render_config(protocol: dict[str, Any]) -> dict[str, Any]:
    render = protocol["render"]
    return {
        "protocols": ["geometry", "texture"],
        "render": {
            "samples": int(render["samples"]),
            "background": bool(render["reconstruction_texture_background"]),
            "gt_geometry_background": bool(render["gt_background"]),
            "gt_texture_background": bool(render["gt_background"]),
            "reconstruction_geometry_background": bool(
                render["reconstruction_geometry_background"]
            ),
            "reconstruction_texture_background": bool(
                render["reconstruction_texture_background"]
            ),
            "geometry_palette": render["geometry_palette"],
            "gray_rgba": [0.5, 0.5, 0.5, 1.0],
            "style": render["style"],
        },
    }
