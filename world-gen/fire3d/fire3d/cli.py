"""Command-line interface for reproducible Fire3D inference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from fire3d.download import (
    DATASET_SUBDIRS,
    EVALUATION_DATASETS,
    download_data,
    download_evaluation_data,
    download_models,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_PATH = REPO_ROOT / "configs/examples.json"
UNIFIED_RUNNER = REPO_ROOT / "eval/unified_percept_recon.py"
VIEW_SAMPLER = REPO_ROOT / "eval/unified_sample_render_views.py"
UNIFIED_RENDERER = REPO_ROOT / "eval/unified_render.py"
DEFAULT_MODEL_ROOT = REPO_ROOT / "checkpoints/Fire3D"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results"


def examples() -> dict[str, dict[str, str]]:
    payload = json.loads(EXAMPLES_PATH.read_text(encoding="utf-8"))
    return payload["examples"]


def protocol_for(dataset: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return REPO_ROOT / examples()[dataset]["protocol"]


def validate_release_inputs(dataset: str, data_root: Path) -> None:
    required_model_paths = [
        DEFAULT_MODEL_ROOT / "perception/config.yaml",
        DEFAULT_MODEL_ROOT / "perception/model.pt",
        DEFAULT_MODEL_ROOT / "reconstruction/flows/ss/config.yaml",
        DEFAULT_MODEL_ROOT / "external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        DEFAULT_MODEL_ROOT
        / "external/trellis2/shape_dec_next_dc_f16c32_fp16.safetensors",
        DEFAULT_MODEL_ROOT
        / "external/trellis2/tex_dec_next_dc_f16c32_fp16.safetensors",
    ]
    missing_models = [str(path) for path in required_model_paths if not path.is_file()]
    if missing_models:
        raise FileNotFoundError(
            "Fire3D model files are missing. Run `fire3d download --models`. "
            f"First missing path: {missing_models[0]}"
        )
    dataset_root = data_root / DATASET_SUBDIRS[dataset]
    if not dataset_root.is_dir():
        raise FileNotFoundError(
            f"Missing {dataset} data at {dataset_root}. Run "
            f"`fire3d download --data --dataset {dataset}`."
        )
    if not (REPO_ROOT / "third_party/dinov3/dinov3").is_dir():
        raise FileNotFoundError(
            "Missing third_party/dinov3. Run `scripts/install.sh` to install "
            "the pinned DINOv3 source tree."
        )


def infer_command(args: argparse.Namespace) -> list[str]:
    data_root = args.data_root.expanduser().resolve()
    scene_ids = args.scene_id or [examples()[args.dataset]["scene_id"]]
    protocol = protocol_for(args.dataset, args.protocol)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else (
            DEFAULT_OUTPUT_ROOT / args.dataset / scene_ids[0]
            if len(scene_ids) == 1
            else DEFAULT_OUTPUT_ROOT / args.dataset / "fire3d_release"
        )
    )
    validate_release_inputs(args.dataset, data_root)
    command = [
        sys.executable,
        str(UNIFIED_RUNNER),
        "--dataset",
        args.dataset,
        "--output-root",
        str(output_root),
        "--protocol",
        str(protocol),
        "--gpu",
        str(args.gpu),
    ]
    for scene_id in scene_ids:
        command.extend(["--scene-id", scene_id])
    if args.dataset in {"ithor", "imaginarium"}:
        command.extend(["--fire3d-test-root", str(data_root)])
    else:
        command.extend(
            ["--protocol-input-root", str(data_root / DATASET_SUBDIRS[args.dataset])]
        )
    if args.skip_render:
        command.append("--skip-render")
    if args.skip_existing:
        command.append("--skip-existing")
    return command


def run_infer(args: argparse.Namespace) -> int:
    command = infer_command(args)
    print("[fire3d]", " ".join(command), flush=True)
    environment = os.environ.copy()
    environment.setdefault("PYTHONUNBUFFERED", "1")
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode


def native_dataset_environment(dataset: str, data_root: Path) -> dict[str, str]:
    variable = {
        "single_image": "FF_SINGLE_IMAGE_ROOT",
        "scannetpp": "FF_SCANNETPP_ROOT",
    }.get(dataset)
    if variable is None:
        return {}
    return {variable: str((data_root / DATASET_SUBDIRS[dataset]).resolve())}


def run_tool(command: list[str], dataset: str, data_root: Path) -> int:
    environment = os.environ.copy()
    environment.update(native_dataset_environment(dataset, data_root))
    environment.setdefault("PYTHONUNBUFFERED", "1")
    print("[fire3d]", " ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=environment, check=False).returncode


def run_sample_views(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    scene_id = args.scene_id or examples()[args.dataset]["scene_id"]
    validate_release_inputs(args.dataset, data_root)
    command = [
        sys.executable,
        str(VIEW_SAMPLER),
        "--dataset",
        args.dataset,
        "--scene-id",
        scene_id,
        "--profile",
        "protocol",
        "--protocol",
        str(protocol_for(args.dataset, args.protocol)),
        "--num-views",
        str(args.num_views),
        "--output",
        str(args.output.expanduser().resolve()),
        "--fire3d-test-root",
        str(data_root),
    ]
    return run_tool(command, args.dataset, data_root)


def run_render(args: argparse.Namespace) -> int:
    data_root = args.data_root.expanduser().resolve()
    scene_id = args.scene_id or examples()[args.dataset]["scene_id"]
    validate_release_inputs(args.dataset, data_root)
    command = [
        sys.executable,
        str(UNIFIED_RENDERER),
        "--dataset",
        args.dataset,
        "--run-root",
        str(args.run_root.expanduser().resolve()),
        "--scene-id",
        scene_id,
        "--config",
        str(args.config.expanduser().resolve()),
        "--protocol",
        str(protocol_for(args.dataset, args.protocol)),
        "--output-root",
        str(args.output_root.expanduser().resolve()),
        "--fire3d-test-root",
        str(data_root),
        "--gpu",
        str(args.gpu),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    return run_tool(command, args.dataset, data_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fire3d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download = subparsers.add_parser("download", help="Download release artifacts")
    download.add_argument("--models", action="store_true")
    download.add_argument("--data", action="store_true")
    download.add_argument(
        "--evaluation",
        action="append",
        choices=EVALUATION_DATASETS,
        default=[],
        help="Download a published evaluation GT bundle.",
    )
    download.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_SUBDIRS),
        default=[],
        help="Repeat to download selected datasets; default is all four.",
    )
    download.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Download only selected whitelisted scenes for one dataset.",
    )
    download.add_argument("--keep-archives", action="store_true")
    download.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    download.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    download.add_argument("--model-revision", default=None)
    download.add_argument("--data-revision", default=None)

    infer = subparsers.add_parser("infer", help="Run perception and reconstruction")
    infer.add_argument("--dataset", choices=sorted(DATASET_SUBDIRS), required=True)
    infer.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Repeat for batched multi-scene inference; defaults to the release example.",
    )
    infer.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    infer.add_argument("--output-root", type=Path, default=None)
    infer.add_argument("--protocol", type=Path, default=None)
    infer.add_argument("--gpu", default="0")
    infer.add_argument("--skip-render", action="store_true")
    infer.add_argument("--skip-existing", action="store_true")

    sample = subparsers.add_parser(
        "sample-views", help="Sample deterministic cameras for a reconstructed scene"
    )
    sample.add_argument("--dataset", choices=sorted(DATASET_SUBDIRS), required=True)
    sample.add_argument("--scene-id", default=None)
    sample.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    sample.add_argument("--protocol", type=Path, default=None)
    sample.add_argument("--num-views", type=int, default=16)
    sample.add_argument("--output", type=Path, required=True)

    render = subparsers.add_parser("render", help="Render a reconstructed scene")
    render.add_argument("--dataset", choices=sorted(DATASET_SUBDIRS), required=True)
    render.add_argument("--scene-id", default=None)
    render.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    render.add_argument("--run-root", type=Path, required=True)
    render.add_argument("--config", type=Path, required=True)
    render.add_argument("--output-root", type=Path, required=True)
    render.add_argument("--protocol", type=Path, default=None)
    render.add_argument("--gpu", default="0")
    render.add_argument("--skip-existing", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "download":
        download_both = not args.models and not args.data and not args.evaluation
        if args.models or download_both:
            print(f"[fire3d] downloading models to {args.model_root}", flush=True)
            download_models(args.model_root, revision=args.model_revision)
        if args.data or download_both:
            selected = args.dataset or sorted(DATASET_SUBDIRS)
            print(
                f"[fire3d] downloading {', '.join(selected)} to {args.data_root}",
                flush=True,
            )
            download_data(
                args.data_root,
                selected,
                scene_ids=args.scene_id,
                revision=args.data_revision,
                keep_archives=args.keep_archives,
            )
        if args.evaluation:
            print(
                f"[fire3d] downloading {', '.join(args.evaluation)} evaluation data",
                flush=True,
            )
            download_evaluation_data(
                args.data_root,
                args.evaluation,
                revision=args.data_revision,
                keep_archives=args.keep_archives,
            )
        return
    if args.command == "infer":
        raise SystemExit(run_infer(args))
    if args.command == "sample-views":
        raise SystemExit(run_sample_views(args))
    if args.command == "render":
        raise SystemExit(run_render(args))
    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
