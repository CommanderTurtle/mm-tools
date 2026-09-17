#!/usr/bin/env python3
"""Unified percept+recon entrypoint for every supported dataset.

One interface, one saving protocol, one source of truth for the pipeline
settings. Datasets differ only in their loader; everything downstream --
detseg perception and LC64 reconstruction -- runs from a complete
versioned protocol under ``configs/inference``. Frozen protocols pass every output-affecting setting
explicitly and write their resolved commands into the run root.

Output protocol (everything keyed by scene_id, never by loader position --
dataset roots grow over time and positional indices move):

    <output-root>/
      manifests/<scene_id>.json         single-scene geometry manifest
      perception/<dataset>/val_<i>/     eval_perception native output
      perception/<dataset>/<scene_id>   symlink to its val_<i>
      reconstruction/<scene_id>/        run_lc64_geometry output
      visualization/                    current-protocol reconstruction renders
      views.yaml + renders/              frozen-protocol comparison renders
      logs/                             one log per stage per scene
      summary.json                      per-scene status keyed by scene_id

Add a dataset by appending a DatasetSpec to DATASETS: name the perception
dataset_type, the loader, and how reconstruction inputs are built ("fire3d"
for the manifest-driven RGB-D layout, "custom" for datasets with their own
branch inside run_lc64_geometry.load_scene_inputs, None while unsupported).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PYTHON = Path(sys.executable)
PERCEPTION = REPO_ROOT / "eval/perception/eval_perception.py"
GEOMETRY = REPO_ROOT / "benchmarks/scene_reconstruction/run_lc64_geometry.py"
EXACT_RGB_RENDERER = REPO_ROOT / "eval/rendering/rerender_imaginarium_exact_camera_rgb.py"
VIEW_SAMPLER = REPO_ROOT / "eval/unified_sample_render_views.py"
UNIFIED_RENDERER = REPO_ROOT / "eval/unified_render.py"

from eval.unified_protocol import (  # noqa: E402
    CURRENT_PROTOCOL,
    available_protocols,
    load_protocol,
    native_dataset_environment,
    protocol_identity,
    protocol_perception_mode,
    reproduction_geometry_args,
    reproduction_perception_args,
    validate_scope,
)

# Protocol values the unified interface pins EXPLICITLY (user directive
# 2026-09-01): the E32 texture-fill recipe from the Aug-27 bake optimization,
# gated on frozen raws (visually identical to nearest_push-32, bake
# 35.9 -> 19.4 s/scene). These match the run_lc64_geometry argparse defaults;
# the parity test keeps the two in agreement.
SHIPPED_GEOMETRY_ARGS = [
    "--appearance-texture-fill-mode", "gpu_dilate",
    "--appearance-texture-dilation-pixels", "32",
]

DEFAULT_PERCEPTION_BUNDLE = REPO_ROOT / (
    "checkpoints/Fire3D/perception"
)
DEFAULT_DINO_REPO = REPO_ROOT / "third_party/dinov3"
DEFAULT_DINO_MODEL = Path(
    REPO_ROOT
    / "checkpoints/Fire3D/external/"
    "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
)
MANIFEST_DIR = REPO_ROOT / "benchmarks/scene_reconstruction/manifests"


@dataclass(frozen=True)
class DatasetSpec:
    perception_type: str
    loader_module: str
    loader_fn: str
    # "fire3d": copy the scene entry from base_manifest (RGB-D dirs resolved
    # under --fire3d-test-root/<dataset_subdir>). "custom": the dataset has its
    # own branch in run_lc64_geometry.load_scene_inputs. None: perception only.
    recon: str | None
    dataset_subdir: str
    base_manifest: Path | None = None
    render: str | None = None  # named render hook, see render_scene()
    num_frames: int = 60


DATASETS: dict[str, DatasetSpec] = {
    "ithor": DatasetSpec(
        perception_type="ithor",
        loader_module="utils.data_ithor",
        loader_fn="load_ithor_data",
        recon="fire3d",
        dataset_subdir="ithor",
        base_manifest=MANIFEST_DIR / "ithor_v1.json",
    ),
    "imaginarium": DatasetSpec(
        perception_type="imaginarium",
        loader_module="utils.data_imaginarium",
        loader_fn="load_imaginarium_data",
        recon="fire3d",
        dataset_subdir="Imaginarium",
        base_manifest=MANIFEST_DIR / "imaginarium_v1.json",
    ),
    "single_image": DatasetSpec(
        perception_type="single_image",
        loader_module="utils.data_single_image",
        loader_fn="load_single_image_data",
        recon="custom",
        dataset_subdir="single_image",
        num_frames=1,
    ),
    "scannetpp": DatasetSpec(
        perception_type="scannetpp",
        loader_module="utils.data_scannetpp",
        loader_fn="load_scannetpp_data",
        recon="custom",
        dataset_subdir="scannetpp",
        render="scannetpp",
        num_frames=300,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Repeatable. Omit together with --all to list available scenes.",
    )
    parser.add_argument("--all", action="store_true", help="Run every scene.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument(
        "--protocol",
        default=CURRENT_PROTOCOL,
        help=(
            "Versioned protocol name under configs/inference or a JSON path. "
            f"Known protocols: {', '.join(available_protocols())}. "
            f"Default {CURRENT_PROTOCOL!r} preserves commit 7962026 behavior."
        ),
    )
    parser.add_argument(
        "--protocol-input-root",
        type=Path,
        default=None,
        help=(
            "Existing Fire3D test root containing the protocol's exact RGB "
            "overlay. For a run-local-rerender protocol, omission creates the "
            "overlay under <output-root>/source/dataset. For a dataset-native "
            "single_image or scannetpp protocol, this overrides that dataset's "
            "native root instead."
        ),
    )
    parser.add_argument(
        "--allow-protocol-overrides",
        action="store_true",
        help=(
            "Allow --geometry-arg or perception feature overrides with a "
            "frozen protocol. Such a run is recorded as modified and is no "
            "longer an exact protocol reproduction."
        ),
    )
    parser.add_argument(
        "--perception-bundle", type=Path, default=DEFAULT_PERCEPTION_BUNDLE
    )
    parser.add_argument("--perception-checkpoint", default="model.pt")
    parser.add_argument("--perception-rgb-upsample", type=int, default=None)
    parser.add_argument(
        "--perception-rgb-upsampler", choices=("swin2sr", "bicubic"), default=None
    )
    parser.add_argument(
        "--perception-feature-subsample",
        type=int,
        default=None,
        help=(
            "Keep every K-th feature-lattice cell after upsampling, to hold the "
            "point density fixed while changing the feature type."
        ),
    )
    parser.add_argument(
        "--perception-dino-upsample",
        type=int,
        default=None,
        help=(
            "AnyUp factor for perception's DINO features, which also sets its "
            "conditioning-cloud stride (16 // upsample). Default 1 = raw ViT "
            "patch grid at stride 16."
        ),
    )
    parser.add_argument("--dino-repo-dir", type=Path, default=DEFAULT_DINO_REPO)
    parser.add_argument("--dino-model-path", type=Path, default=DEFAULT_DINO_MODEL)
    parser.add_argument(
        "--fire3d-test-root",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument(
        "--perception",
        choices=("predicted", "gt"),
        default=None,
        help=(
            "Override the source recorded by the selected protocol. Protocols "
            "without an explicit source retain the legacy predicted default. "
            "predicted: run detseg perception and condition on it. "
            "gt: skip perception entirely and reconstruct from the dataset's "
            "ground-truth instance annotations (oracle IDs/poses) -- only "
            "valid for datasets that ship GT masks (ithor, imaginarium). "
            "Output format and dataset-world coordinates are identical, so "
            "the unified renderer applies unchanged."
        ),
    )
    parser.add_argument("--skip-perception", action="store_true")
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument(
        "--evaluate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run geometry evaluation. Frozen protocols use their recorded value.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip stages whose outputs already exist and are complete.",
    )
    parser.add_argument(
        "--geometry-arg",
        action="append",
        default=None,
        metavar="ARGS",
        help=(
            "Extra arguments appended to the reconstruction argv, shell-split "
            "and repeatable. ALWAYS use the equals form, e.g. "
            "--geometry-arg=--sample-method=random or "
            "--geometry-arg='--sample-method random': argparse accepts a "
            "quoted value only when it contains a space, so a space-free "
            "--token passed as a separate word is read as an option and fails "
            "with \"expected one argument\". Appended after the pinned "
            "protocol args so an ablation can override them; the shipped "
            "protocol is what you get when this is absent."
        ),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if not args.all and not args.scene_id:
        parser.error("Pass --scene-id (repeatable) or --all")
    return args


def resolve_perception_mode(
    requested: str | None,
    protocol: dict[str, Any] | None,
    dataset: str,
) -> str:
    recorded = protocol_perception_mode(protocol)
    effective = recorded if requested is None else requested
    if protocol is not None and effective != recorded:
        raise ValueError(
            f"Protocol {protocol['name']} records {recorded} perception, not "
            f"--perception {effective}"
        )
    if effective == "gt" and dataset not in ("ithor", "imaginarium"):
        raise ValueError(
            "GT perception requires ground-truth instance annotations; "
            f"dataset {dataset!r} has none"
        )
    return effective


def load_scene_index(args, spec: DatasetSpec) -> tuple[list, dict[str, int]]:
    module = importlib.import_module(spec.loader_module)
    loader = getattr(module, spec.loader_fn)
    if spec.recon == "fire3d":
        # The loaders' baked-in default roots point at the original training
        # machine; the fire3d test copy is the canonical root here.
        data_list = loader(data_root=str(args.fire3d_test_root / spec.dataset_subdir))
    else:
        data_list = loader()
    lookup: dict[str, int] = {}
    for index, entry in enumerate(data_list):
        scene_id = entry["scene_id"] if isinstance(entry, dict) else str(entry)
        # First occurrence wins for datasets with several videos per scene.
        lookup.setdefault(str(scene_id), index)
    return data_list, lookup


def run_logged(
    cmd: list[str],
    log_path: Path,
    env: dict,
    *,
    commands: list[dict[str, Any]] | None = None,
    stage: str | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(" ".join(cmd) + "\n\n")
        handle.flush()
        code = subprocess.call(
            cmd, stdout=handle, stderr=subprocess.STDOUT, env=env,
            cwd=str(REPO_ROOT),
        )
    wall_seconds = float(time.perf_counter() - started)
    if commands is not None:
        commands.append(
            {
                "stage": stage or log_path.stem,
                "argv": cmd,
                "log": str(log_path.resolve()),
                "exit_code": int(code),
                "wall_seconds": wall_seconds,
            }
        )
    return code


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def exact_rgb_frame_dir(root: Path, spec: DatasetSpec, scene_id: str) -> Path:
    return root / spec.dataset_subdir / "renders" / scene_id / "0_frames"


def validate_exact_rgb_input(
    root: Path, spec: DatasetSpec, scene_ids: list[str], protocol: dict[str, Any]
) -> None:
    from PIL import Image

    rgb = protocol["input_rgb"]
    expected_count = int(rgb["expected_frames"])
    expected_size = (int(rgb["expected_width"]), int(rgb["expected_height"]))
    for scene_id in scene_ids:
        frame_dir = exact_rgb_frame_dir(root, spec, scene_id)
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        if len(frames) != expected_count:
            raise RuntimeError(
                f"Protocol RGB input {frame_dir} has {len(frames)} frames; "
                f"expected {expected_count}"
            )
        sizes = {Image.open(path).size for path in frames}
        if sizes != {expected_size}:
            raise RuntimeError(
                f"Protocol RGB input {frame_dir} has sizes {sorted(sizes)}; "
                f"expected only {expected_size}"
            )


def canonical_exact_rgb_ready(
    root: Path,
    spec: DatasetSpec,
    scene_ids: list[str],
) -> bool:
    from utils.exact_camera_rgb import canonical_frames_dir, canonical_is_complete

    for scene_id in scene_ids:
        original = exact_rgb_frame_dir(root, spec, scene_id)
        candidate = canonical_frames_dir(original, scene_id)
        if not canonical_is_complete(
            candidate,
            original,
            dataset_subdir=spec.dataset_subdir,
            scene_id=scene_id,
            video_id=0,
        ):
            return False
    return True


def prepare_protocol_input(
    args: argparse.Namespace,
    spec: DatasetSpec,
    scene_ids: list[str],
    protocol: dict[str, Any] | None,
    env: dict[str, str],
    commands: list[dict[str, Any]],
) -> Path:
    if protocol is None:
        if args.protocol_input_root is not None:
            raise ValueError("--protocol-input-root requires a frozen --protocol")
        return args.fire3d_test_root
    rgb = protocol.get("input_rgb") or {}
    mode = rgb.get("mode")
    if mode == "dataset_native":
        native_env = native_dataset_environment(
            protocol, args.dataset, args.protocol_input_root
        )
        root = Path(native_env[str(rgb["root_env"])])
        missing = [
            str(root / relative)
            for relative in rgb.get("required_relative_paths", [])
            if not (root / relative).exists()
        ]
        if missing:
            raise FileNotFoundError(
                f"Protocol native dataset root {root} is incomplete; missing {missing}"
            )
        env.update(native_env)
        os.environ.update(native_env)
        commands.append(
            {
                "stage": "protocol_native_input",
                "dataset": args.dataset,
                "root": str(root),
                "environment": native_env,
                "exit_code": 0,
            }
        )
        return args.fire3d_test_root
    supported_modes = {
        "canonical_exact_camera_preferred",
        "run_local_exact_camera_rerender",
    }
    if mode not in supported_modes:
        return args.protocol_input_root or args.fire3d_test_root
    if spec.recon != "fire3d":
        raise ValueError(f"Protocol exact-RGB input is unsupported for {args.dataset}")

    if (
        mode == "canonical_exact_camera_preferred"
        and args.protocol_input_root is None
        and canonical_exact_rgb_ready(args.fire3d_test_root, spec, scene_ids)
    ):
        env["FF_EXACT_RGB"] = "1"
        os.environ["FF_EXACT_RGB"] = "1"
        commands.append(
            {
                "stage": "protocol_exact_rgb",
                "resolution": "canonical",
                "root": str(args.fire3d_test_root.resolve()),
                "scene_ids": list(scene_ids),
                "exit_code": 0,
            }
        )
        return args.fire3d_test_root

    input_root = (
        args.protocol_input_root.resolve()
        if args.protocol_input_root is not None
        else (args.output_root / rgb["overlay_relative_root"]).resolve()
    )
    try:
        validate_exact_rgb_input(input_root, spec, scene_ids, protocol)
    except RuntimeError:
        if args.protocol_input_root is not None:
            raise
        cmd = [
            str(PYTHON),
            str(EXACT_RGB_RENDERER),
            "--dataset",
            args.dataset,
            "--source-fire3d-test-root",
            str(
                args.fire3d_test_root
                if mode == "canonical_exact_camera_preferred"
                else rgb["source_fire3d_test_root"]
            ),
            "--overlay-fire3d-test-root",
            str(input_root),
            "--samples",
            str(rgb["samples"]),
            "--recipe",
            rgb["recipe"],
            "--jpeg-quality",
            str(rgb["jpeg_quality"]),
        ]
        for scene_id in scene_ids:
            cmd.extend(["--scene", scene_id])
        code = run_logged(
            cmd,
            args.output_root / "logs" / "protocol_exact_rgb.log",
            env,
            commands=commands,
            stage="protocol_exact_rgb",
        )
        if code != 0:
            raise RuntimeError("exact-camera RGB preparation failed")
        validate_exact_rgb_input(input_root, spec, scene_ids, protocol)

    # The returned root already contains the exact frames under renders/. Do
    # not allow another global overlay to redirect this validated fallback.
    env["FF_EXACT_RGB"] = "0"
    os.environ["FF_EXACT_RGB"] = "0"
    return input_root


def extra_geometry_args(args) -> list[str]:
    """Shell-split the repeatable --geometry-arg strings into argv tokens."""

    import shlex

    tokens: list[str] = []
    for chunk in args.geometry_arg or []:
        tokens.extend(shlex.split(chunk))
    return tokens


def perception_dir(args, spec: DatasetSpec, index: int) -> Path:
    return args.output_root / "perception" / spec.perception_type / f"val_{index}"


def perception_required_outputs(protocol: dict[str, Any] | None) -> tuple[str, ...]:
    if protocol is None:
        return ("oriented_bboxes.json", "raw_oriented_bboxes.json", "point_instance_masks.pkl")
    return tuple(protocol["perception"]["required_outputs"])


def perception_ready(
    args: argparse.Namespace,
    spec: DatasetSpec,
    index: int,
    protocol: dict[str, Any] | None,
) -> bool:
    root = perception_dir(args, spec, index)
    return all((root / name).is_file() for name in perception_required_outputs(protocol))


def scene_symlink(args, spec: DatasetSpec, scene_id: str, index: int) -> None:
    link = args.output_root / "perception" / spec.perception_type / scene_id
    target = Path(f"val_{index}")
    if link.is_symlink():
        link.unlink()
    if not link.exists():
        link.symlink_to(target)


def build_perception_command(
    args: argparse.Namespace,
    spec: DatasetSpec,
    pending: list[tuple[str, int]],
    protocol: dict[str, Any] | None,
) -> list[str]:
    indices = ",".join(str(index) for _, index in pending)
    out_dir = args.output_root / "perception" / spec.perception_type
    if protocol is None:
        bundle = args.perception_bundle
        checkpoint = args.perception_checkpoint
        dino_repo = args.dino_repo_dir
        dino_model = args.dino_model_path
    else:
        config = protocol["perception"]
        bundle = _repo_path(config["bundle"])
        checkpoint = config["checkpoint"]
        dino_repo = _repo_path(config["dino_repo"])
        dino_model = _repo_path(config["dino_model"])
    cmd = [
        str(PYTHON), str(PERCEPTION),
        "--run_dir_seg", str(bundle),
        "--config_path", str(bundle / "config.yaml"),
        "--checkpoint_path", str(bundle / checkpoint),
        "--dino_repo_dir", str(dino_repo),
        "--dino_model_path", str(dino_model),
        "--output_dir", str(out_dir),
        "--dataset_type", spec.perception_type,
    ]
    if spec.recon == "fire3d":
        cmd += ["--dataset_root", str(args.fire3d_test_root / spec.dataset_subdir)]
    cmd += ["--eval_indices", indices]
    if args.skip_existing:
        cmd.append("--skip_existing")
    if protocol is not None:
        cmd.extend(reproduction_perception_args(protocol))
    if args.perception_dino_upsample is not None:
        cmd += ["--dino_upsample", str(int(args.perception_dino_upsample))]
    if args.perception_feature_subsample is not None:
        cmd += ["--feature_subsample", str(int(args.perception_feature_subsample))]
    if args.perception_rgb_upsample is not None:
        cmd += ["--rgb_upsample", str(int(args.perception_rgb_upsample))]
    if args.perception_rgb_upsampler is not None:
        cmd += ["--rgb_upsampler", args.perception_rgb_upsampler]
    return cmd


def validate_perception_reference(
    args: argparse.Namespace,
    spec: DatasetSpec,
    scene_id: str,
    index: int,
    protocol: dict[str, Any] | None,
) -> None:
    if protocol is None or scene_id != (protocol.get("scope") or {}).get("scene_id"):
        return
    gate = (protocol.get("perception") or {}).get("reference_gate")
    if not gate:
        return
    root = perception_dir(args, spec, index)
    boxes = json.loads((root / "oriented_bboxes.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "det_seg_metrics.json").read_text(encoding="utf-8"))
    expected_objects = int(gate["num_objects"])
    if len(boxes["objects"]) != expected_objects:
        raise RuntimeError(
            f"Perception reference gate failed: {len(boxes['objects'])} objects, "
            f"expected {expected_objects}"
        )
    tolerance = float(gate["absolute_tolerance"])
    for key in ("mAP", "mIoU"):
        actual, expected = float(metrics[key]), float(gate[key])
        if abs(actual - expected) > tolerance:
            raise RuntimeError(
                f"Perception reference gate failed: {key}={actual}, expected "
                f"{expected} +/- {tolerance}"
            )


def run_perception(
    args,
    spec: DatasetSpec,
    pairs: list[tuple[str, int]],
    env,
    protocol: dict[str, Any] | None,
    commands: list[dict[str, Any]],
) -> bool:
    pending = [
        (scene_id, index)
        for scene_id, index in pairs
        if not (args.skip_existing and perception_ready(args, spec, index, protocol))
    ]
    for scene_id, index in pairs:
        if perception_ready(args, spec, index, protocol):
            scene_symlink(args, spec, scene_id, index)
    if not pending:
        for scene_id, index in pairs:
            validate_perception_reference(args, spec, scene_id, index, protocol)
        return True
    out_dir = args.output_root / "perception" / spec.perception_type
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_perception_command(args, spec, pending, protocol)
    code = run_logged(
        cmd,
        args.output_root / "logs" / "perception.log",
        env,
        commands=commands,
        stage="perception",
    )
    if code != 0:
        return False
    for scene_id, index in pairs:
        if not perception_ready(args, spec, index, protocol):
            return False
        scene_symlink(args, spec, scene_id, index)
        validate_perception_reference(args, spec, scene_id, index, protocol)
    return True


def write_scene_manifest(args, spec: DatasetSpec, scene_id: str, index: int) -> Path:
    path = args.output_root / "manifests" / f"{scene_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    if spec.recon == "fire3d":
        base = json.loads(spec.base_manifest.read_text(encoding="utf-8"))
        matches = [s for s in base["scenes"] if s["scene_id"] == scene_id]
        if not matches:
            raise ValueError(
                f"{scene_id} is not in {spec.base_manifest.name}; "
                "regenerate the base manifest first"
            )
        payload = dict(base)
        payload["selection"] = {"mode": "explicit_scene", "num_entries": 1}
        payload["scenes"] = matches
    else:
        num_frames = (
            int(os.environ.get("FF_SCANNETPP_MAX_FRAMES", spec.num_frames))
            if args.dataset == "scannetpp"
            else spec.num_frames
        )
        payload = {
            "schema": "ff_scene_geometry_manifest_v1",
            "protocol": f"ff_{args.dataset}_percept_reconstruction_v1",
            "dataset": args.dataset,
            "dataset_subdir": spec.dataset_subdir,
            "selection": {"mode": "explicit_scene", "num_entries": 1},
            "object_set": {
                "primary": (
                    "perception_predictions"
                    if args.perception == "predicted"
                    else "gt_annotations"
                ),
                "visibility_source": "multi-view RGB-D"
                if num_frames > 1
                else "single aligned RGB-D image",
            },
            "scenes": [
                {
                    "scene_id": scene_id,
                    "scene_dir": scene_id,
                    "benchmark_index": index,
                    "raw_index": index,
                    "legacy_eval_index": index,
                    "num_frames": num_frames,
                }
            ],
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def recon_status(recon_dir: Path) -> tuple[str, int, int]:
    summary = recon_dir / "inference_summary.json"
    if not summary.is_file():
        return "missing", 0, 0
    payload = json.loads(summary.read_text(encoding="utf-8"))
    return (
        payload.get("status", "unknown"),
        int(payload.get("num_decoded_objects", 0)),
        int(payload.get("num_expected_objects", 0)),
    )


def build_reconstruction_batch_command(
    args: argparse.Namespace,
    batch_path: Path,
    protocol: dict[str, Any] | None,
) -> list[str]:
    cmd = [
        str(PYTHON),
        str(GEOMETRY),
        "--batch-scenes",
        str(batch_path),
        "--fire3d-test-root",
        str(args.fire3d_test_root),
        "--output-root",
        str(args.output_root / "reconstruction"),
    ]
    cmd.extend(
        SHIPPED_GEOMETRY_ARGS
        if protocol is None
        else reproduction_geometry_args(protocol, REPO_ROOT)
    )
    cmd.extend(extra_geometry_args(args))
    return cmd


def evaluation_enabled(
    args: argparse.Namespace, protocol: dict[str, Any] | None
) -> bool:
    if args.evaluate is not None:
        return bool(args.evaluate)
    return bool(protocol and (protocol.get("evaluation") or {}).get("enabled"))


def evaluation_result_path(args: argparse.Namespace, scene_id: str) -> Path:
    return (
        args.output_root
        / "evaluation"
        / args.dataset
        / scene_id
        / "eval_results.json"
    )


def run_evaluation(
    args: argparse.Namespace,
    spec: DatasetSpec,
    scene_id: str,
    env: dict[str, str],
    protocol: dict[str, Any] | None,
    commands: list[dict[str, Any]],
) -> bool:
    if args.dataset not in ("ithor", "imaginarium"):
        return False
    config = (protocol or {}).get("evaluation") or {}
    entrypoint = (
        config.get("entrypoint")
        or (config.get("entrypoint_by_dataset") or {}).get(args.dataset)
        or (
            "benchmarks/scene_reconstruction/evaluators/fire3d/"
            f"eval_{args.dataset}.py"
        )
    )
    evaluator = REPO_ROOT / entrypoint
    output_dir = evaluation_result_path(args, scene_id).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_env = dict(env)
    eval_env[
        "FF_ITHOR_ROOT" if args.dataset == "ithor" else "FF_IMAGINARIUM_ROOT"
    ] = str(args.fire3d_test_root / spec.dataset_subdir)
    cmd = [
        str(PYTHON),
        str(evaluator),
        "--scene_name",
        scene_id,
        "--pred_results_root_dir",
        str(args.output_root / "reconstruction"),
        "--output_dir",
        str(output_dir),
        "--point_downsample",
        str(config.get("point_downsample", 8)),
        "--geometry_samples",
        str(config.get("sample_points", 100000)),
    ]
    code = run_logged(
        cmd,
        args.output_root / "logs" / f"evaluation_{scene_id}.log",
        eval_env,
        commands=commands,
        stage=f"evaluation:{scene_id}",
    )
    return code == 0 and (output_dir / "eval_results.json").is_file()


def write_protocol_record(
    args: argparse.Namespace,
    identity: dict[str, Any],
    protocol: dict[str, Any] | None,
    commands: list[dict[str, Any]],
) -> Path:
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    payload = {
        "schema": "ff_unified_resolved_protocol_v1",
        "identity": identity,
        "git_commit": git_commit,
        "dataset": args.dataset,
        "perception_mode": args.perception,
        "scene_ids": list(args.scene_id),
        "effective_fire3d_test_root": str(args.fire3d_test_root.resolve()),
        "effective_native_dataset_environment": native_dataset_environment(
            protocol, args.dataset, args.protocol_input_root
        ),
        "protocol_modified": bool(args.allow_protocol_overrides),
        "geometry_overrides": list(args.geometry_arg or []),
        "protocol": protocol,
        "commands": commands,
    }
    path = args.output_root / "resolved_protocol.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def run_reconstruction(args, spec, scene_id: str, index: int, env) -> dict:
    recon_dir = args.output_root / "reconstruction" / scene_id
    status, decoded, expected = recon_status(recon_dir)
    if args.skip_existing and status == "complete":
        return {"status": "skipped", "objects": decoded}
    # eval_perception writes its index for fire3d datasets by legacy order; the
    # perception stage above always uses the loader index, so both agree here.
    manifest = write_scene_manifest(args, spec, scene_id, index)
    cmd = [
        str(PYTHON), str(GEOMETRY),
        "--manifest", str(manifest),
        "--scene-id", scene_id,
        "--fire3d-test-root", str(args.fire3d_test_root),
        "--output-root", str(args.output_root / "reconstruction"),
        "--perception-result-dir", str(perception_dir(args, spec, index)),
    ]
    code = run_logged(
        cmd, args.output_root / "logs" / f"reconstruction_{scene_id}.log", env
    )
    status, decoded, expected = recon_status(recon_dir)
    if code != 0 or status != "complete":
        return {"status": "failed", "exit_code": code, "recon_status": status}
    return {"status": "complete", "objects": decoded, "expected": expected}


def render_scene(
    args,
    spec,
    scene_id: str,
    env,
    protocol: dict[str, Any] | None,
    commands: list[dict[str, Any]],
) -> bool:
    """Delegate to the unified two-protocol rendering interface."""
    from eval.unified_render import DATASETS as RENDER_DATASETS

    if args.dataset not in RENDER_DATASETS:
        return False
    if protocol is not None:
        views = args.output_root / "views.yaml"
        sample_cmd = [
            str(PYTHON),
            str(VIEW_SAMPLER),
            "--dataset",
            args.dataset,
            "--scene-id",
            scene_id,
            "--profile",
            "protocol",
            "--protocol",
            str(args.protocol),
            "--num-views",
            str(protocol["views"]["num_views"]),
            "--fire3d-test-root",
            str(args.fire3d_test_root),
            "--output",
            str(views),
        ]
        if run_logged(
            sample_cmd,
            args.output_root / "logs" / f"render_views_{scene_id}.log",
            env,
            commands=commands,
            stage=f"render_views:{scene_id}",
        ) != 0:
            return False
        render_source = "comparison" if spec.recon == "fire3d" else "recon"
        cmd = [
            str(PYTHON),
            str(UNIFIED_RENDERER),
            "--dataset",
            args.dataset,
            "--source",
            render_source,
            "--run-root",
            str(args.output_root),
            "--scene-id",
            scene_id,
            "--config",
            str(views),
            "--output-root",
            str(args.output_root / "renders"),
            "--protocol",
            str(args.protocol),
        ]
        if args.gpu is not None:
            cmd.extend(["--gpu", str(args.gpu)])
        if getattr(args, "skip_existing", False):
            cmd.append("--skip-existing")
    else:
        cmd = [
            str(PYTHON),
            str(UNIFIED_RENDERER),
            "--dataset",
            args.dataset,
            "--run-root",
            str(args.output_root),
            "--scene-id",
            scene_id,
        ]
    command_succeeded = run_logged(
        cmd,
        args.output_root / "logs" / f"render_{scene_id}.log",
        env,
        commands=commands,
        stage=f"render:{scene_id}",
    ) == 0
    return command_succeeded and render_result_ready(args, scene_id)


def render_result_ready(args: argparse.Namespace, scene_id: str) -> bool:
    index_path = args.output_root / "renders" / "render_index.json"
    if not index_path.is_file():
        return False
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        (payload.get("scenes") or {}).get(scene_id, {}).get("status")
        == "complete"
    )


def run_downstream_stages(
    args: argparse.Namespace,
    spec: DatasetSpec,
    scene_id: str,
    record: dict[str, Any],
    env: dict[str, str],
    protocol: dict[str, Any] | None,
    commands: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run or reuse evaluation and rendering after a valid reconstruction."""
    if record.get("status") not in ("complete", "skipped"):
        return record

    if evaluation_enabled(args, protocol):
        evaluation_exists = evaluation_result_path(args, scene_id).is_file()
        if getattr(args, "skip_existing", False) and evaluation_exists:
            record["evaluated"] = True
            record["evaluation_skipped"] = True
        else:
            record["evaluated"] = run_evaluation(
                args, spec, scene_id, env, protocol, commands
            )
        if not record["evaluated"]:
            record["status"] = "failed"
            record["evaluation_error"] = "evaluation command/output failed"
            return record

    if not args.skip_render:
        render_exists = render_result_ready(args, scene_id)
        if getattr(args, "skip_existing", False) and render_exists:
            record["rendered"] = True
            record["render_skipped"] = True
        else:
            record["rendered"] = render_scene(
                args, spec, scene_id, env, protocol, commands
            )
        if not record["rendered"]:
            record["status"] = "failed"
            record["render_error"] = "unified renderer failed"
    return record


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    try:
        protocol, protocol_source = load_protocol(args.protocol)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if protocol is not None:
        if args.all:
            raise SystemExit("A frozen scene protocol cannot be combined with --all")
        try:
            validate_scope(protocol, args.dataset, list(args.scene_id))
        except ValueError as error:
            raise SystemExit(str(error)) from error
        protocol_overrides = bool(
            args.geometry_arg
            or args.perception_dino_upsample is not None
            or args.perception_feature_subsample is not None
            or args.perception_rgb_upsample is not None
            or args.perception_rgb_upsampler is not None
        )
        if protocol_overrides and not args.allow_protocol_overrides:
            raise SystemExit(
                "Frozen protocols reject geometry/perception overrides. Pass "
                "--allow-protocol-overrides to run a recorded non-reference ablation."
            )
        args.allow_protocol_overrides = bool(
            args.allow_protocol_overrides and protocol_overrides
        )

    try:
        args.perception = resolve_perception_mode(
            args.perception, protocol, args.dataset
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if args.dataset == "scannetpp":
        # The shared root is actively being extended; a partial whitelist
        # appeared mid-experiment and dropped scenes. Enumerate everything.
        env.setdefault("FF_SCANNETPP_WHITELIST", "0")
        os.environ.setdefault("FF_SCANNETPP_WHITELIST", "0")
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    commands: list[dict[str, Any]] = []
    try:
        args.fire3d_test_root = prepare_protocol_input(
            args, spec, list(args.scene_id), protocol, env, commands
        )
    except (OSError, RuntimeError, ValueError) as error:
        write_protocol_record(
            args,
            protocol_identity(args.protocol, protocol, protocol_source),
            protocol,
            commands,
        )
        raise SystemExit(str(error)) from error

    data_list, lookup = load_scene_index(args, spec)
    scene_ids = sorted(lookup) if args.all else args.scene_id
    missing = [scene_id for scene_id in scene_ids if scene_id not in lookup]
    if missing:
        raise SystemExit(f"scenes not loadable for {args.dataset}: {missing}")
    pairs = [(scene_id, lookup[scene_id]) for scene_id in scene_ids]
    if args.num_shards > 1:
        pairs = [
            pair for n, pair in enumerate(pairs)
            if n % args.num_shards == args.shard_index
        ]
    print(
        f"[driver] {args.dataset}: {len(pairs)} scenes "
        f"{[scene_id for scene_id, _ in pairs]}",
        flush=True,
    )

    identity = protocol_identity(args.protocol, protocol, protocol_source)
    print(f"[driver] protocol: {identity['resolved_name']}", flush=True)
    write_protocol_record(args, identity, protocol, commands)
    summary: dict[str, dict] = {}
    if args.perception == "predicted" and not args.skip_perception:
        if not run_perception(args, spec, pairs, env, protocol, commands):
            write_protocol_record(args, identity, protocol, commands)
            raise SystemExit("perception failed; see logs/perception.log")
    elif args.perception == "predicted":
        for scene_id, index in pairs:
            if not perception_ready(args, spec, index, protocol):
                required = perception_required_outputs(protocol)
                raise SystemExit(
                    f"--skip-perception requested but {scene_id} is missing one "
                    f"of {required} under {perception_dir(args, spec, index)}"
                )
            scene_symlink(args, spec, scene_id, index)
            validate_perception_reference(args, spec, scene_id, index, protocol)

    # Reconstruction runs as ONE process over all pending scenes: models load
    # once and stay resident (~60 s saved per scene after the first).
    batch_entries = []
    for scene_id, index in pairs:
        if args.skip_reconstruction or spec.recon is None:
            summary[scene_id] = {
                "status": "perception_only" if spec.recon is None else "recon_skipped"
            }
            continue
        recon_dir = args.output_root / "reconstruction" / scene_id
        status, decoded, expected = recon_status(recon_dir)
        if args.skip_existing and status == "complete":
            summary[scene_id] = {
                "status": "skipped",
                "objects": decoded,
                "expected": expected,
                "reconstruction_skipped": True,
            }
            continue
        manifest = write_scene_manifest(args, spec, scene_id, index)
        entry = {"scene_id": scene_id, "manifest": str(manifest)}
        if args.perception == "predicted":
            entry["perception_result_dir"] = str(perception_dir(args, spec, index))
        batch_entries.append(entry)

    if batch_entries:
        batch_path = args.output_root / "manifests" / "batch_scenes.json"
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        batch_path.write_text(json.dumps(batch_entries, indent=2) + "\n")
        started = time.time()
        reconstruction_code = run_logged(
            build_reconstruction_batch_command(args, batch_path, protocol),
            args.output_root / "logs" / "reconstruction_batch.log",
            env,
            commands=commands,
            stage="reconstruction_batch",
        )
        batch_elapsed = time.time() - started
        for entry in batch_entries:
            scene_id = entry["scene_id"]
            status, decoded, expected = recon_status(
                args.output_root / "reconstruction" / scene_id
            )
            record: dict = (
                {"status": "complete", "objects": decoded, "expected": expected}
                if reconstruction_code == 0 and status == "complete"
                else {
                    "status": "failed",
                    "recon_status": status,
                    "reconstruction_exit_code": reconstruction_code,
                }
            )
            summary[scene_id] = record
        print(f"[driver] batch reconstruction took {batch_elapsed:.0f}s "
              f"for {len(batch_entries)} scenes", flush=True)

    for n, (scene_id, _) in enumerate(pairs, start=1):
        record = run_downstream_stages(
            args,
            spec,
            scene_id,
            summary[scene_id],
            env,
            protocol,
            commands,
        )
        summary[scene_id] = record
        print(f"[driver] ({n}/{len(pairs)}) {scene_id}: {record}", flush=True)

    out = args.output_root / (
        "summary.json" if args.num_shards == 1
        else f"summary_shard{args.shard_index}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "perception_mode": args.perception,
                "protocol": identity,
                "scenes": summary,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_protocol_record(args, identity, protocol, commands)
    done = sum(
        1 for record in summary.values()
        if record.get("status") in ("complete", "skipped", "perception_only")
    )
    print(f"\n[driver] {done}/{len(pairs)} ok; wrote {out}", flush=True)
    if done != len(pairs):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
