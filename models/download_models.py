from __future__ import annotations

import argparse
import inspect
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

# Current huggingface_hub uses its Xet transport for high-speed snapshots.
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"
CPU_DEFAULT_WORKERS = min(16, max(1, os.cpu_count() or 4))
try:
    ENV_DEFAULT_WORKERS = int(
        os.environ.get("TOKIO_WORKER_THREADS", str(CPU_DEFAULT_WORKERS))
    )
except ValueError:
    ENV_DEFAULT_WORKERS = CPU_DEFAULT_WORKERS
DEFAULT_WORKERS = (
    ENV_DEFAULT_WORKERS
    if 1 <= ENV_DEFAULT_WORKERS <= 256
    else CPU_DEFAULT_WORKERS
)
os.environ["TOKIO_WORKER_THREADS"] = str(DEFAULT_WORKERS)

from huggingface_hub import logging, snapshot_download


# This file is tracked inside <mm-tools>/models. Resolving from the file rather
# than the caller's working directory keeps fresh clones and `uv run` identical.
MODELS = Path(__file__).resolve().parent
ROOT = MODELS.parent
SNAPSHOT_SUPPORTS_SYMLINK_FLAG = "local_dir_use_symlinks" in inspect.signature(snapshot_download).parameters
LEGACY_TRANSLATOR_DIR = MODELS / "text-only/anhbn--raX-Translator-V1.0-GGUF"
TRANSLATOR_DIR = MODELS / "text-only/mradermacher--EraX-Translator-V1.0-GGUF"


@dataclass(frozen=True)
class Artifact:
    repo_id: str
    destination: Path
    allow_patterns: tuple[str, ...] = ()
    cache_layout: bool = False


@dataclass(frozen=True)
class Bundle:
    key: str
    title: str
    inventory: str
    artifacts: tuple[Artifact, ...]


def model_path(relative: str) -> Path:
    return MODELS / relative


def repo_path(relative: str) -> Path:
    return ROOT / relative


A = Artifact
QWEN_GUIDE = A(
    "SergiusFlavius/Qwen3-VL-4B-Instruct-heretic-NVFP4",
    model_path("qwen/text-encoder-vl-nvfp4"),
    ("qwen3_vl_4b_nvfp4_full.safetensors",),
)
BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        "ideogram",
        "Ideogram generation and editing",
        "1-3, 40",
        (
            A("ideogram-ai/ideogram-4-fp8", model_path("ideogram-ai--ideogram-4-fp8")),
            A("jixin0101/ObjectClear", model_path("jixin0101--ObjectClear")),
            A("ZhengPeng7/BiRefNet", model_path("ZhengPeng7--BiRefNet")),
            QWEN_GUIDE,
        ),
    ),
    Bundle(
        "longcat",
        "LongCat multilingual TTS",
        "4-5",
        (
            A(
                "meituan-longcat/LongCat-AudioDiT-3.5B",
                model_path("meituan-longcat--AudioDiT-3.5B-tts-text-to-speech-SOTA"),
            ),
            A(
                "google/umt5-base",
                model_path("google--umt5-base-tokenizer"),
                (
                    "special_tokens_map.json",
                    "spiece.model",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
            ),
        ),
    ),
    Bundle(
        "muscriptor",
        "MuScriptor transcription and soundfonts",
        "6-7",
        (
            A("MuScriptor/muscriptor-large", model_path("MuScriptor--muscriptor-large")),
            A("MuScriptor/assets", model_path("MuScriptor--assets")),
        ),
    ),
    Bundle(
        "musvit",
        "MusVIT score understanding",
        "8-9",
        (
            A("PRAIG/musvit", model_path("PRAIG--musvit")),
            A("PRAIG/smt-fp-grandstaff", model_path("PRAIG--smt-fp-grandstaff")),
        ),
    ),
    Bundle(
        "whisper",
        "CrisperWhisper speech recognition",
        "10",
        (A("nyralabs/CrisperWhisper2.0_large", model_path("nyralabs--CrisperWhisper2.0_large")),),
    ),
    Bundle(
        "translate",
        "Translation, language detection, and arbitration",
        "11-13",
        (
            A(
                "mradermacher/EraX-Translator-V1.0-GGUF",
                TRANSLATOR_DIR,
                ("EraX-Translator-V1.0.Q8_0.gguf",),
            ),
            A(
                "papluca/xlm-roberta-base-language-detection",
                model_path("text-only/papluca--xlm-roberta-base-language-detection"),
                (
                    "config.json",
                    "model.safetensors",
                    "sentencepiece.bpe.model",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
            ),
            A(
                "anhbn/EraX-VL-7B-V1.5-Openvino-INT4",
                model_path("text-only/anhbn--EraX-VL-7B-V1.5-Openvino-INT4"),
            ),
        ),
    ),
    Bundle(
        "redesign",
        "ReDesign layered editing, tools, and optional Comfy assets",
        "14-19, 24-31",
        (
            A(
                "T5B/Qwen-Image-Layered-FP8",
                model_path("qwen/T5B--qwen-image-layered-fp8"),
                ("qwen_image_layered_fp8_e4m3fn.safetensors",),
            ),
            A(
                "diffusers/Qwen-Image-Layered-modular",
                model_path("qwen/diffusers--hfstaff--Qwen-Image-Layered-modular"),
            ),
            A(
                "suzukimain/Qwen-Image-Layered-Control-SDNQ-int4",
                model_path("qwen/suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4"),
            ),
            A(
                "appmana/qwen-image-layered-int8convrot",
                model_path("qwen/appmana--diffusion--qwen-image-layered-int8convrot"),
                ("qwen_image_layered_int8convrot.safetensors",),
            ),
            A(
                "Comfy-Org/HunyuanVideo_1.5_repackaged",
                model_path("qwen/comfy-org--text--qwen_2.5_vl_7b_fp8_scaled.safetensors"),
                ("split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",),
            ),
            A(
                "Comfy-Org/Qwen-Image-Layered_ComfyUI",
                model_path("qwen/comfy-org--vae--qwen_image_layered_vae.safetensors"),
                ("split_files/vae/qwen_image_layered_vae.safetensors",),
            ),
            A(
                "benjiaiplayground/GroundingDINO_SwinB",
                repo_path("redesign/weights"),
                ("groundingdino_swinb_cogcoor.pth",),
            ),
            A(
                "facebook/sam2.1-hiera-large",
                repo_path("redesign/weights"),
                ("sam2.1_hiera_large.pt",),
            ),
            A(
                "GoGiants1/Hi-SAM",
                repo_path("redesign/weights"),
                ("sam_tss_h_textseg.pth", "sam_vit_h_4b8939.pth"),
            ),
            A("iimate/big-lama-pt", repo_path("redesign/weights"), ("big-lama.pt",)),
            A("jixin0101/ObjectClear", repo_path("redesign/weights"), cache_layout=True),
            A("google-bert/bert-base-uncased", repo_path("redesign/weights"), cache_layout=True),
            A(
                "PaddlePaddle/PP-OCRv5_server_det",
                Path.home() / ".paddlex/official_models/PP-OCRv5_server_det",
            ),
            A(
                "PaddlePaddle/PP-OCRv5_server_rec",
                Path.home() / ".paddlex/official_models/PP-OCRv5_server_rec",
            ),
        ),
    ),
    Bundle(
        "kijai",
        "Standalone Kijai Wan V2V assets",
        "20-23",
        (
            A(
                "Kijai/Wan_ID_V2V_comfy",
                model_path("kijai/Kijai--Wan_ID_V2V_comfy"),
                ("wan_2.1_idv2v_int8_convrot.safetensors",),
            ),
            A(
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                model_path("kijai/Comfy-Org--Wan_2.1_ComfyUI_repackaged"),
                (
                    "split_files/clip_vision/clip_vision_h.safetensors",
                    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "split_files/vae/wan_2.1_vae.safetensors",
                ),
            ),
            A(
                "denisbalon/lightx2v-i2v-14b-480p-cfg-step-distill-rank64-bf16.safetensors",
                model_path("kijai/denisbalon--lightx2v-i2v-14b-480p-cfg-step-distill-rank64-bf16"),
                ("lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",),
            ),
            A(
                "Kijai/WanVideo_comfy",
                model_path("kijai/Kijai--WanVideo_comfy"),
                ("Wan2_1_VAE_bf16.safetensors",),
            ),
        ),
    ),
    Bundle(
        "acestep",
        "ACE-Step 1.5 music generation",
        "32-34",
        (
            A(
                "ACE-Step/Ace-Step1.5",
                model_path("Ace-Step--Ace-Step1.5"),
                ("config.json", "Qwen3-Embedding-0.6B/*", "vae/*"),
            ),
            A(
                "ACE-Step/acestep-v15-xl-sft",
                model_path("Ace-Step--Ace-Step1.5/acestep-v15-xl-sft"),
                (
                    "apg_guidance.py",
                    "config.json",
                    "configuration_acestep_v15.py",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                    "modeling_acestep_v15_xl_base.py",
                    "silence_latent.pt",
                ),
            ),
            A(
                "ACE-Step/acestep-5Hz-lm-4B",
                model_path("Ace-Step--Ace-Step1.5/acestep-5Hz-lm-4B"),
                (
                    "added_tokens.json",
                    "chat_template.jinja",
                    "config.json",
                    "merges.txt",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                    "special_tokens_map.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
            ),
        ),
    ),
    Bundle(
        "minimax",
        "MiniMax Music 3 and optional Qwen Prompt Guide",
        "35, 40",
        (
            A(
                "Comfy-Org/MiniMax-Music-3",
                model_path("Comfy-Org--Minimax-Music-3"),
                (
                    "diffusion_models/minimax_music3_dit_fp16.safetensors",
                    "text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
                    "vae/minimax_music3_dav.safetensors",
                ),
            ),
            QWEN_GUIDE,
        ),
    ),
    Bundle(
        "stableaudio",
        "Stable Audio Foundation-1 looping",
        "36-37",
        (
            A(
                "RoyalCities/Foundation-1",
                model_path("RoyalCities--Foundation-1"),
                ("Foundation_1.safetensors", "model_config.json"),
            ),
            A(
                "google-t5/t5-base",
                model_path("google-t5--t5-base"),
                ("config.json", "generation_config.json", "model.safetensors", "spiece.model", "tokenizer.json"),
            ),
        ),
    ),
    Bundle(
        "symphony",
        "SymphonyGen symbolic composition",
        "38",
        (
            A(
                "SymphonyGen/SymphonyGen",
                model_path("SymphonyGen--SymphonyGen"),
                (
                    "stage_one_pretrained.pt",
                    "stage_two_pretrained.pt",
                    "grpo_clamp_epoch_10.pt",
                    "grpo_clamp+track_epoch_6.pt",
                ),
            ),
        ),
    ),
    Bundle(
        "vocalrender",
        "VocalRender Pro singing synthesis",
        "39",
        (
            A("pymaster/VocalRender", model_path("pymaster--VocalRender"), ("VocalRender-Pro/*",)),
        ),
    ),
)


def print_menu() -> None:
    print("\nMM Tools model downloader")
    print(f"Destination root: {MODELS}")
    print("Bundle numbers select complete runnable projects; inventory numbers refer to which-ones.txt.\n")
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
            f"Parallel download workers / CPU threads [{default}]: "
        ).strip()
        if not raw:
            return default
        try:
            return worker_count(raw)
        except argparse.ArgumentTypeError as exc:
            print(f"Error: {exc}", file=sys.stderr)


def migrate_legacy_layout(bundles: Sequence[Bundle]) -> None:
    if not any(bundle.key == "translate" for bundle in bundles):
        return
    if LEGACY_TRANSLATOR_DIR.exists() and not TRANSLATOR_DIR.exists():
        TRANSLATOR_DIR.parent.mkdir(parents=True, exist_ok=True)
        LEGACY_TRANSLATOR_DIR.rename(TRANSLATOR_DIR)
        print(
            "Migrated historical translator folder:\n"
            f"  {LEGACY_TRANSLATOR_DIR}\n"
            f"  -> {TRANSLATOR_DIR}"
        )
    elif LEGACY_TRANSLATOR_DIR.exists() and TRANSLATOR_DIR.exists():
        print(
            "Warning: both historical and canonical translator folders exist; "
            "neither was moved.",
            file=sys.stderr,
        )


def download_artifact(
    artifact: Artifact,
    number: int,
    total: int,
    workers: int,
) -> None:
    # snapshot_download normally creates local_dir/cache_dir, but doing it here
    # guarantees that fresh nested roots such as qwen/ and text-only/ exist.
    artifact.destination.mkdir(parents=True, exist_ok=True)
    mode = "cache_dir" if artifact.cache_layout else "local_dir"
    print(f"\n[{number}/{total}] {artifact.repo_id}")
    print(f"  {mode}: {artifact.destination}")
    kwargs: dict[str, object] = {
        "repo_id": artifact.repo_id,
        mode: artifact.destination,
        "max_workers": workers,
    }
    # Older Hub releases need this to materialize local files. Current Hub
    # removed the argument and already uses real files for local_dir snapshots.
    if SNAPSHOT_SUPPORTS_SYMLINK_FLAG:
        kwargs["local_dir_use_symlinks"] = False
    if artifact.allow_patterns:
        kwargs["allow_patterns"] = list(artifact.allow_patterns)
    snapshot_download(**kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all mm-tools model artifacts or selected project bundles.",
        epilog=(
            "Examples: uv run download_models.py | uv run download_models.py all | "
            "uv run download_models.py 2,6,10 | uv run download_models.py minimax stableaudio"
        ),
    )
    parser.add_argument("selection", nargs="*", help="bundle numbers, bundle names, or 'all'")
    parser.add_argument("--all", action="store_true", dest="download_all", help="download every bundle")
    parser.add_argument("--list", action="store_true", help="show bundles and exit")
    parser.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    parser.add_argument(
        "--workers",
        type=worker_count,
        help="parallel Hub file workers and Xet Tokio worker threads (1-256)",
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
    os.environ["TOKIO_WORKER_THREADS"] = str(workers)
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

    # Do not mutate the model store until the user has accepted the download.
    # Renaming the complete local_dir also preserves Hugging Face's resumable
    # metadata and any interrupted chunks beneath that directory.
    migrate_legacy_layout(bundles)

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
