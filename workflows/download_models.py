from __future__ import annotations

import argparse
import filecmp
import inspect
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# mm-tools setup commands deliberately keep hf_transfer canonical. Transport
# negotiation itself stays with the installed Hub client for version safety.
CPU_DEFAULT_WORKERS = min(16, max(1, os.cpu_count() or 4))
try:
    ENV_DEFAULT_WORKERS = int(
        os.environ.get("MMTOOLS_DOWNLOAD_WORKERS", str(CPU_DEFAULT_WORKERS))
    )
except ValueError:
    ENV_DEFAULT_WORKERS = CPU_DEFAULT_WORKERS
DEFAULT_WORKERS = (
    ENV_DEFAULT_WORKERS
    if 1 <= ENV_DEFAULT_WORKERS <= 256
    else CPU_DEFAULT_WORKERS
)
from huggingface_hub import logging, snapshot_download


# This file is tracked inside <mm-tools>/workflows. Resolving from the file
# rather than the caller's working directory keeps fresh clones and `uv run`
# identical.
WORKFLOWS = Path(__file__).resolve().parent
MODELS = WORKFLOWS / "models"
ROOT = WORKFLOWS.parent
SNAPSHOT_SUPPORTS_SYMLINK_FLAG = "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters


@dataclass(frozen=True)
class Artifact:
    repo_id: str
    destination: Path
    allow_patterns: tuple[str, ...] = ()
    cache_layout: bool = False
    revision: str | None = None
    placements: tuple[tuple[str, Path], ...] = ()
    direct_url: str | None = None
    direct_filename: str | None = None
    sha256: str | None = None
    expected_size: int | None = None


@dataclass(frozen=True)
class Bundle:
    key: str
    title: str
    inventory: str
    artifacts: tuple[Artifact, ...]


def model_path(relative: str) -> Path:
    return MODELS / relative


def comfy_model_path(relative: str) -> Path:
    # Runtime placement target: the shared pinned Comfy checkout's models
    # directory, exactly where the video studios resolve their weights.
    return ROOT / "workflows" / "V2V" / "ComfyUI" / "models" / relative


A = Artifact

# Inventory identifiers 1-8 are owned by this file alone (see the top of
# which-ones.txt in models/ for the main downloader's numbering). Each bundle
# below downloads resumable snapshots into models/imports/ and hard-links the
# selected files into the native models folders the exported pipelines read.
WAN_ANIMATE_2 = A(
    "Comfy-Org/Wan-Animate-2",
    model_path("imports/Comfy-Org--Wan-Animate-2"),
    (
        "clip_vision/clip_vision_h.safetensors",
        "diffusion_models/wan_animate_2_int8_convrot.safetensors",
        "diffusion_models/wan_animate_2_distill_int8_convrot.safetensors",
        "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "vae/Wan2_1_VAE_bf16.safetensors",
    ),
    revision="ed158470869ff31fa51cf56012dac33fb00f494b",
    placements=(
        (
            "clip_vision/clip_vision_h.safetensors",
            comfy_model_path("clip_vision/clip_vision_h.safetensors"),
        ),
        (
            "diffusion_models/wan_animate_2_int8_convrot.safetensors",
            comfy_model_path("diffusion_models/wan_animate_2_int8_convrot.safetensors"),
        ),
        (
            "diffusion_models/wan_animate_2_distill_int8_convrot.safetensors",
            comfy_model_path("diffusion_models/wan_animate_2_distill_int8_convrot.safetensors"),
        ),
        (
            "loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
            comfy_model_path("loras/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
        ),
        (
            "text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            comfy_model_path("text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        ),
        (
            "vae/Wan2_1_VAE_bf16.safetensors",
            comfy_model_path("vae/Wan2_1_VAE_bf16.safetensors"),
        ),
    ),
)
WAN22_REPLACE = A(
    "Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
    model_path("imports/Comfy-Org--Wan_2.2_ComfyUI_Repackaged-animate"),
    ("split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors",),
    revision="fb1388adc906ab39ffc26ee40e96b22886b56bc4",
    placements=((
        "split_files/diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors",
        comfy_model_path("diffusion_models/wan2.2_animate_14B_int8_convrot.safetensors"),
    ),),
)
# Blackwell NVFP4 drop-ins for the INT8 ConvRot pair above: the same Wan 2.2
# Animate 14B DiT and UMT5-XXL text encoder in fp16/NVFP4 mixed quantization.
# The exported replacement graph offers them as two independent loader
# switches; the distilled lane stays INT8 for the DiT.
WAN22_NVFP4_DIT = A(
    "LHQAQ-Li/wan2.2_animate_14b_fp16_nvfp4_comfy_V2",
    model_path("imports/LHQAQ-Li--wan2.2_animate_14b_fp16_nvfp4_comfy_V2"),
    ("wan2.2_animate_14b_fp16_nvfp4_comfy_V2.safetensors",),
    revision="955c2d9ade00382be464a4796a5ec2bee80fc8c0",
    placements=(
        (
            "wan2.2_animate_14b_fp16_nvfp4_comfy_V2.safetensors",
            comfy_model_path("diffusion_models/wan2.2_animate_14b_fp16_nvfp4_comfy_V2.safetensors"),
        ),
    ),
)
WAN22_NVFP4_ENCODER = A(
    "rst220/UMT5_XXL_NVFP4",
    model_path("imports/rst220--UMT5_XXL_NVFP4"),
    ("UMT5_XXL_NVFP4.safetensors",),
    revision="2ae584c8370d19190c56a00112c0360df34d8483",
    placements=(
        (
            "UMT5_XXL_NVFP4.safetensors",
            comfy_model_path("text_encoders/UMT5_XXL_NVFP4.safetensors"),
        ),
    ),
)
ANIMATE_YOLO = A(
    "Wan-AI/Wan2.2-Animate-14B",
    model_path("imports/Wan-AI--Wan2.2-Animate-14B-preprocess"),
    ("process_checkpoint/det/yolov10m.onnx",),
    revision="cb93a225fbaf1ca100f54e79da8f994995b689b3",
    placements=(
        (
            "process_checkpoint/det/yolov10m.onnx",
            comfy_model_path("detection/yolov10m.onnx"),
        ),
    ),
)
ANIMATE_VITPOSE = A(
    "Kijai/vitpose_comfy",
    model_path("imports/Kijai--vitpose_comfy"),
    (
        "onnx/vitpose_h_wholebody_data.bin",
        "onnx/vitpose_h_wholebody_model.onnx",
    ),
    revision="ae68f4e542151cebec0995b8469c70b07b8c3df4",
    placements=(
        (
            "onnx/vitpose_h_wholebody_data.bin",
            comfy_model_path("detection/vitpose_h_wholebody_data.bin"),
        ),
        (
            "onnx/vitpose_h_wholebody_model.onnx",
            comfy_model_path("detection/vitpose_h_wholebody_model.onnx"),
        ),
    ),
)
# Ungated community mirror of Lightricks/LTX-2.5 with the identical
# file tree; no Hub license acceptance or authentication is needed.
# The weights remain under the LTX-2.x Community License (upstream
# Lightricks/LTX-2.5). The Comfy int8/convrot files are the exact
# checkpoints ComfyUI's official LTX-2.5 templates name. The mirror
# re-commits each file under its own SHAs, so this entry tracks main
# without a revision pin.
LTX_25 = A(
    "comfyicu/LTX-2.5",
    model_path("imports/comfyicu--LTX-2.5"),
    (
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
        "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
        "model_patches/ltx-2.5-duration-head-bf16.safetensors",
    ),
    placements=(
        (
            "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
            comfy_model_path("diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"),
        ),
        (
            "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
            comfy_model_path("diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"),
        ),
        (
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
            comfy_model_path("text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"),
        ),
        (
            "vae/ltx-2.5-video-vae-bf16.safetensors",
            comfy_model_path("vae/ltx-2.5-video-vae-bf16.safetensors"),
        ),
        (
            "vae/ltx-2.5-audio-vae-bf16.safetensors",
            comfy_model_path("vae/ltx-2.5-audio-vae-bf16.safetensors"),
        ),
        (
            "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
            comfy_model_path("latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"),
        ),
        (
            "latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors",
            comfy_model_path("latent_upscale_models/ltx-2.5-latent-temporal-upscaler-x2-bf16-1.0.safetensors"),
        ),
        (
            "model_patches/ltx-2.5-duration-head-bf16.safetensors",
            comfy_model_path("model_patches/ltx-2.5-duration-head-bf16.safetensors"),
        ),
    ),
)
LTX_ENHANCER = A(
    "Comfy-Org/gemma-4",
    model_path("imports/Comfy-Org--gemma-4"),
    ("text_encoders/gemma4_e2b_it_int8_convrot.safetensors",),
    revision="63d0f7c476756b88910170c1df75e2384ea1af31",
    placements=((
        "text_encoders/gemma4_e2b_it_int8_convrot.safetensors",
        comfy_model_path("text_encoders/gemma4_e2b_it_int8_convrot.safetensors"),
    ),),
)

BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        "animate",
        "Wan Animate 2 motion transfer, scene replacement, and ONNX pose inputs",
        "1-6",
        (
            WAN_ANIMATE_2,
            WAN22_REPLACE,
            WAN22_NVFP4_DIT,
            WAN22_NVFP4_ENCODER,
            ANIMATE_YOLO,
            ANIMATE_VITPOSE,
        ),
    ),
    Bundle(
        "ltx",
        "LTX-2.5 native video and audio generation with the Blackwell NVFP4 twin",
        "7-8",
        (
            LTX_25,
            LTX_ENHANCER,
        ),
    ),
)


def print_menu() -> None:
    print("\nMM Tools workflow model downloader")
    print(f"Destination root: {MODELS}")
    print("Bundle numbers select complete pipeline lanes; inventory numbers refer to which-ones.txt (workflows section).\n")
    for number, bundle in enumerate(BUNDLES, start=1):
        count = len(bundle.artifacts)
        suffix = "s" if count != 1 else ""
        print(f"  {number:>2}. {bundle.title} [{bundle.key}] — {count} snapshot{suffix}; inventory {bundle.inventory}")
    print("\n  all. Every bundle (the original downloader behavior)")
    print("  q.   Exit without downloading")


def _selection_tokens(values: Sequence[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def resolve_selection(values: Sequence[str]) -> tuple[Bundle, ...]:
    tokens = _selection_tokens(values)
    if not tokens:
        raise ValueError("No model bundle was selected")
    if any(token.lower() in {"all", "a", "*"} for token in tokens):
        return BUNDLES

    by_key = {bundle.key: bundle for bundle in BUNDLES}
    selected: list[Bundle] = []
    seen: set[str] = set()
    for token in tokens:
        lowered = token.lower()
        if lowered in {"q", "quit", "exit"}:
            return ()
        if token.isdecimal() and 1 <= int(token) <= len(BUNDLES):
            bundle = BUNDLES[int(token) - 1]
        elif lowered in by_key:
            bundle = by_key[lowered]
        else:
            raise ValueError(f"Unknown selection: {token!r}")
        if bundle.key not in seen:
            selected.append(bundle)
            seen.add(bundle.key)
    return tuple(selected)


def interactive_selection() -> tuple[Bundle, ...]:
    print_menu()
    while True:
        try:
            raw = input("\nSelect numbers or names, comma-separated [all]: ").strip() or "all"
            return resolve_selection((raw,))
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def worker_count(value: str) -> int:
    try:
        workers = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be an integer") from exc
    if not 1 <= workers <= 256:
        raise argparse.ArgumentTypeError("workers must be between 1 and 256")
    return workers


def interactive_worker_count(default: int) -> int:
    while True:
        raw = input(
            f"Parallel Hugging Face file workers [{default}]: "
        ).strip()
        if not raw:
            return default
        try:
            return worker_count(raw)
        except argparse.ArgumentTypeError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def download_artifact(
    artifact: Artifact,
    number: int,
    total: int,
    workers: int,
) -> None:
    # snapshot_download normally creates local_dir/cache_dir, but doing it here
    # guarantees that fresh nested roots such as models/imports/ exist.
    artifact.destination.mkdir(parents=True, exist_ok=True)
    mode = "cache_dir" if artifact.cache_layout else "local_dir"
    print(f"\n[{number}/{total}] {artifact.repo_id}")
    print(f"  {mode}: {artifact.destination}")
    kwargs: dict[str, object] = {
        "repo_id": artifact.repo_id,
        mode: artifact.destination,
        "max_workers": workers,
    }
    if artifact.revision:
        kwargs["revision"] = artifact.revision
    # Older Hub releases need this to materialize local files. Current Hub
    # removed the argument and already uses real files for local_dir snapshots.
    if SNAPSHOT_SUPPORTS_SYMLINK_FLAG:
        kwargs["local_dir_use_symlinks"] = False
    if artifact.allow_patterns:
        kwargs["allow_patterns"] = list(artifact.allow_patterns)
    snapshot_download(**kwargs)
    materialize_placements(artifact)


def materialize_placements(artifact: Artifact) -> None:
    """Expose selected snapshot files at native runtime paths without copies.

    Hugging Face's resumable local snapshot remains intact. On the same
    filesystem, the runtime path is a hard link to those verified bytes; the
    copy fallback is only for an unusual cross-filesystem destination.
    """
    for relative_source, target in artifact.placements:
        source = artifact.destination / relative_source
        if not source.is_file():
            raise FileNotFoundError(
                f"Downloaded snapshot did not contain placement source: {source}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            try:
                if source.samefile(target):
                    print(f"  ready: {target}")
                    continue
            except OSError:
                pass
            if (
                target.is_file()
                and target.stat().st_size == source.stat().st_size
                and filecmp.cmp(source, target, shallow=False)
            ):
                print(f"  ready: {target} (verified existing copy)")
                continue
            raise FileExistsError(
                f"Refusing to replace a different runtime artifact: {target}"
            )
        try:
            os.link(source, target)
            method = "hard link"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"
        print(f"  placed ({method}): {target}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download the model artifacts for the committed ComfyUI pipeline exports.",
        epilog=(
            "Examples: uv run download_models.py | uv run download_models.py all | "
            "uv run download_models.py ltx | uv run download_models.py animate --yes --workers 24"
        ),
    )
    parser.add_argument("selection", nargs="*", help="bundle numbers, bundle names, or 'all'")
    parser.add_argument("--all", action="store_true", dest="download_all", help="download every bundle")
    parser.add_argument("--list", action="store_true", help="show bundles and exit")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    parser.add_argument(
        "--workers",
        type=worker_count,
        help="parallel Hugging Face snapshot file workers (1-256)",
    )
    parser.add_argument("--debug", action="store_true", help="enable verbose huggingface_hub logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print_menu()
        return 0

    MODELS.mkdir(parents=True, exist_ok=True)
    workers = args.workers or DEFAULT_WORKERS
    if args.workers is None and not args.yes and sys.stdin.isatty():
        workers = interactive_worker_count(DEFAULT_WORKERS)
    if args.download_all:
        bundles = BUNDLES
    elif args.selection:
        try:
            bundles = resolve_selection(args.selection)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    elif sys.stdin.isatty():
        bundles = interactive_selection()
    else:
        print("No selection supplied in a non-interactive shell; pass 'all' or bundle numbers.", file=sys.stderr)
        return 2

    if not bundles:
        print("No downloads selected.")
        return 0

    artifacts = tuple(dict.fromkeys(artifact for bundle in bundles for artifact in bundle.artifacts))
    print("\nSelected bundles:")
    for bundle in bundles:
        count = len(bundle.artifacts)
        suffix = "s" if count != 1 else ""
        print(f"  - {bundle.title} ({count} snapshot{suffix})")
    print(f"Total snapshots: {len(artifacts)}")
    print(f"Parallel workers: {workers}")

    if not args.yes and sys.stdin.isatty():
        answer = input("Continue? [Y/n]: ").strip().lower()
        if answer not in {"", "y", "yes"}:
            print("Cancelled.")
            return 0

    if args.debug:
        logging.set_verbosity_debug()
    else:
        logging.set_verbosity_info()
    for number, artifact in enumerate(artifacts, start=1):
        download_artifact(artifact, number, len(artifacts), workers)

    print(f"\nCompleted {len(artifacts)} snapshot downloads.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
