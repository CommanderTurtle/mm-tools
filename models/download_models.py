from __future__ import annotations

import argparse
import filecmp
import hashlib
import inspect
import os
import shutil
import sys
import urllib.request
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


def repo_path(relative: str) -> Path:
    return ROOT / relative


A = Artifact
QWEN_GUIDE = A(
    "SergiusFlavius/Qwen3-VL-4B-Instruct-heretic-NVFP4",
    model_path("qwen/text-encoder-vl-nvfp4"),
    ("qwen3_vl_4b_nvfp4_full.safetensors",),
)

# TRELLIS.2 and Pixal3D name these repositories from their pipeline JSON.  Keep
# the dependencies local and pinned so neither sculpting runtime can silently
# contact the Hub during model construction.  Only the one legacy sparse-
# structure decoder absent from TRELLIS.2-4B is placed into that checkpoint
# tree; the source snapshot remains intact for resume/checksum purposes.
SCULPT_SPARSE_DECODER = A(
    "microsoft/TRELLIS-image-large",
    repo_path("sculpting/pretrained/deps/microsoft--TRELLIS-image-large"),
    (
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",
    placements=(
        (
            "ckpts/ss_dec_conv3d_16l8_fp16.json",
            repo_path("sculpting/pretrained/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.json"),
        ),
        (
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            repo_path("sculpting/pretrained/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
        ),
    ),
)
SCULPT_DINO = A(
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    repo_path("sculpting/pretrained/deps/camenduru--dinov3-vitl16-pretrain-lvd1689m"),
    (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ),
    revision="3c276edd87d6f6e569ff0c4400e086807d0f3881",
)
SCULPT_RMBG = A(
    # TRELLIS.2's BiRefNet wrapper accepts this checkpoint through the same
    # Transformers remote-code contract.  Unlike the pipeline's briaai alias,
    # the official ZhengPeng7 model is ungated, so unattended setup stays
    # deterministic instead of stalling on a click-through approval.
    "ZhengPeng7/BiRefNet",
    repo_path("sculpting/pretrained/deps/ZhengPeng7--BiRefNet"),
    (
        "BiRefNet_config.py",
        "birefnet.py",
        "config.json",
        "model.safetensors",
    ),
    revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
)
SCULPT_MOGE = A(
    "Ruicheng/moge-2-vitl",
    repo_path("sculpting/pretrained/deps/Ruicheng--moge-2-vitl"),
    ("model.pt",),
    revision="39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
)
SCULPT_NAF = A(
    "valeoai/NAF release checkpoint",
    repo_path("sculpting/pretrained/deps/valeoai--NAF"),
    direct_url="https://github.com/valeoai/NAF/releases/download/model/naf_release.pth",
    direct_filename="naf_release.pth",
    sha256="c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f",
    expected_size=2664431,
)
FIRE3D_ANYUP = A(
    "wimmerth/AnyUp multi-backbone checkpoint",
    repo_path("world-gen/fire3d/checkpoints/Fire3D/external"),
    direct_url="https://github.com/wimmerth/anyup/releases/download/checkpoint_v2/anyup_multi_backbone.pth",
    direct_filename="anyup_multi_backbone.pth",
    sha256="b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59",
    expected_size=3541624,
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
        "v2v",
        "ID-V2V with its complete local Comfy support set",
        "20-21",
        (
            A(
                "Kijai/Wan_ID_V2V_comfy",
                model_path("imports/Kijai--Wan_ID_V2V_comfy"),
                (
                    "wan_2.1_idv2v_int8_convrot.safetensors",
                    "wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors",
                ),
                revision="72a0683760887daf321a15ffe1d0ffe186ec9fbe",
                placements=(
                    (
                        "wan_2.1_idv2v_int8_convrot.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/diffusion_models/wan_2.1_idv2v_int8_convrot.safetensors"),
                    ),
                    (
                        "wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/diffusion_models/wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors"),
                    ),
                ),
            ),
            A(
                "Comfy-Org/Wan_2.1_ComfyUI_repackaged",
                model_path("imports/Comfy-Org--Wan_2.1_ComfyUI_repackaged"),
                (
                    "split_files/clip_vision/clip_vision_h.safetensors",
                    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                    "split_files/vae/wan_2.1_vae.safetensors",
                ),
                revision="617a7633e636506f850e043bc4605f290a466a8e",
                placements=(
                    (
                        "split_files/clip_vision/clip_vision_h.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/clip_vision/clip_vision_h.safetensors"),
                    ),
                    (
                        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    ),
                    (
                        "split_files/vae/wan_2.1_vae.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/vae/wan_2.1_vae.safetensors"),
                    ),
                ),
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
    Bundle(
        "4d",
        "4DAnyone reconstruction and foreground extraction",
        "41-42",
        (
            A(
                "AntResearch/4DAnyone",
                repo_path("4d-ify/models"),
                (
                    "4danyone/*",
                    "gvhmr/*",
                    "perceptual/*",
                ),
                revision="4c80e87b805a5f8461cf339cdbe2fb4249e585aa",
            ),
            A(
                "ZhengPeng7/BiRefNet",
                repo_path("4d-ify/models/birefnet"),
                (
                    "BiRefNet_config.py",
                    "birefnet.py",
                    "config.json",
                    "model.safetensors",
                    "requirements.txt",
                ),
                revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
            ),
            A(
                "camenduru/GVHMR",
                repo_path("4d-ify/models/.sources/camenduru--GVHMR"),
                ("dpvo/dpvo.pth",),
                revision="21b32d5389e2e59c0737d4c4095bbc0b8c23f66b",
                placements=(("dpvo/dpvo.pth", repo_path("4d-ify/models/gvhmr/dpvo.pth")),),
            ),
        ),
    ),
    Bundle(
        "lingbot",
        "LingBot-World-V2 shared encoder, tokenizer, and VAE",
        "43",
        (
            A(
                "robbyant/lingbot-world-v2-14b-causal-fast",
                repo_path("world-gen/lingbot/lingbot-world-v2-14b-causal-fast"),
                (
                    "Wan2.1_VAE.pth",
                    "config.json",
                    "google/umt5-xxl/*",
                    "models_t5_umt5-xxl-enc-bf16.pth",
                ),
                revision="5c33dd40b213598c418fd25bff30fdbd23fd38a7",
            ),
        ),
    ),
    Bundle(
        "lingbot-5090",
        "LingBot-World-V2 official 1.3B single-5090 DiT",
        "43a",
        (
            A(
                "robbyant/lingbot-world-v2-1.3b-causal-fast",
                repo_path("world-gen/lingbot/lingbot-world-v2-1.3b-causal-fast"),
                (
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                ),
                revision="7e36a5f919f86cb4255cc9bfc30adb44963fbde1",
            ),
        ),
    ),
    Bundle(
        "fire3d",
        "Fire3D single-image scene reconstruction",
        "44",
        (
            A(
                "hongchi/Fire3D",
                repo_path("world-gen/fire3d/checkpoints/Fire3D"),
                (
                    "checksums.sha256",
                    "config.json",
                    "manifest.json",
                    "external/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
                    "external/trellis2/*",
                    "perception/*",
                    "reconstruction/flows/*/*",
                    "reconstruction/stats/*.json",
                    "reconstruction/vae/*/config.json",
                    "reconstruction/vae/*/ckpts/*.pt",
                ),
                revision="84d4246ba2a0f47c8b93fd9dd65a67d354afd668",
            ),
            FIRE3D_ANYUP,
        ),
    ),
    Bundle(
        "aukspeech",
        "AuK base speech generation/editing and Qwen encoder",
        "45-46",
        (
            A(
                "tencent/AuK",
                repo_path("aukspeech/ckpts/AuK"),
                (
                    "auk_base.safetensors",
                    "config.yaml",
                    "vae.safetensors",
                ),
                revision="790742b71a4430120daf2b2099192abae449eb9f",
            ),
            A(
                "Qwen/Qwen2.5-Omni-3B",
                repo_path("aukspeech/ckpts/Qwen2.5-Omni-3B"),
                (
                    "added_tokens.json",
                    "chat_template.json",
                    "config.json",
                    "generation_config.json",
                    "merges.txt",
                    "model-*.safetensors",
                    "model.safetensors.index.json",
                    "preprocessor_config.json",
                    "special_tokens_map.json",
                    "spk_dict.pt",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "vocab.json",
                ),
                revision="f75b40e3da2003cdd6e1829b1f420ca70797c34e",
            ),
        ),
    ),
    Bundle(
        "sculpting",
        "TRELLIS.2 and Pixal3D single-object sculpting",
        "47-48, 64-68",
        (
            A(
                "microsoft/TRELLIS.2-4B",
                repo_path("sculpting/pretrained/TRELLIS.2-4B"),
                ("pipeline.json", "texturing_pipeline.json", "ckpts/*"),
                revision="af44b45f2e35a493886929c6d786e563ec68364d",
            ),
            A(
                "TencentARC/Pixal3D",
                repo_path("sculpting/world/pretrained/Pixal3D"),
                (
                    "pipeline.json",
                    "pipeline_mv.json",
                    "ckpts/*",
                ),
                revision="b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
            ),
            SCULPT_SPARSE_DECODER,
            SCULPT_DINO,
            SCULPT_RMBG,
            SCULPT_MOGE,
            SCULPT_NAF,
        ),
    ),
    Bundle(
        "worldsculpt",
        "WorldSculpt multi-object scene reconstruction",
        "48-49, 64-68",
        (
            A(
                "TencentARC/Pixal3D",
                repo_path("sculpting/world/pretrained/Pixal3D"),
                (
                    "pipeline.json",
                    "pipeline_mv.json",
                    "ckpts/*",
                ),
                revision="b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
            ),
            A(
                "AlayaLab/WorldSculpt",
                repo_path("sculpting/world/pretrained"),
                (
                    "shape_ft1024_mv_lora_ibr_texverse_fixedmem05/config.json",
                    "shape_ft1024_mv_lora_ibr_texverse_fixedmem05/ckpts/*.pt",
                    "ss_ft64_mv_lora_ibr_texverse/config.json",
                    "ss_ft64_mv_lora_ibr_texverse/ckpts/*.pt",
                ),
                revision="8cb81056d803c61371dd84ef18a14142a738610e",
            ),
            SCULPT_SPARSE_DECODER,
            SCULPT_DINO,
            SCULPT_RMBG,
            SCULPT_MOGE,
            SCULPT_NAF,
        ),
    ),
    Bundle(
        "nvidia-sim",
        "ARDY motion generation and SOMA-X character assets",
        "53-58",
        (
            A(
                "nvidia/ARDY-Core-RP-20FPS-Horizon40",
                repo_path("nvidia-sim/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="abe6c43beb28c867c950acb824b9c4ef3d63fb76",
            ),
            A(
                "nvidia/ARDY-Core-RP-20FPS-Horizon8",
                repo_path("nvidia-sim/checkpoints/ARDY-Core-RP-20FPS-Horizon8"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="257a0843a10bf5201065963d7bca2791e9393a7a",
            ),
            A(
                "nvidia/ARDY-G1-RP-25FPS-Horizon52",
                repo_path("nvidia-sim/checkpoints/ARDY-G1-RP-25FPS-Horizon52"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="059b8007df0ba194a006a877b59a563955ac7b70",
            ),
            A(
                "nvidia/ARDY-G1-RP-25FPS-Horizon8",
                repo_path("nvidia-sim/checkpoints/ARDY-G1-RP-25FPS-Horizon8"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="334a8a9cdbafc962dcd304c26f31e6b17a869355",
            ),
            A(
                "voxta/Llama-3-8B-LLM2Vec-ARDY-INT8",
                repo_path("nvidia-sim/text-encoders/voxta/Llama-3-8B-LLM2Vec-ARDY-INT8"),
                (
                    "chat_template.jinja",
                    "config.json",
                    "llm2vec_config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
                revision="96c7eb1cc9100cd3925662d1093926c00b64a698",
            ),
            A(
                "nvidia/SOMA-X",
                repo_path("nvidia-sim/SOMA-X/assets"),
                (
                    "Anny/*",
                    "GarmentMeasurements/*",
                    "MANO/*",
                    "MHR/*",
                    "SMPL/*",
                    "SMPLX/*",
                    "SOMA-X-HF-MANIFEST.json",
                    "SOMAHand.npz",
                    "SOMA_neutral.npz",
                    "SOMA_procedural_transforms.json",
                    "SOMA_template_rig.usda",
                    "correctives_model.pt",
                    "example_animation.npy",
                ),
                revision="32f0ab41a0db0f2710d6542a435f90ad452c3b73",
            ),
        ),
    ),
    Bundle(
        "yue2",
        "YuE2 native and Comfy music generation/cover stack",
        "59-63",
        (
            A(
                "m-a-p/YuE2-3B",
                repo_path("music/yue2/models/YuE2-3B"),
                (
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "modeling_yue2.py",
                    "qwen.tiktoken",
                    "weights_manifest.json",
                    "yue2_generation_config.json",
                ),
                revision="14fc6c6f146441b1dd6363fcb2e01e82a6914cb7",
            ),
            A(
                "m-a-p/YuE2-Vae",
                repo_path("music/yue2/models/YuE2-Vae"),
                (
                    "config.json",
                    "model.safetensors",
                    "modeling_vae.py",
                    "weights_manifest.json",
                ),
                revision="9a94e1d0ea9f8087e98f77fa88df4a4068104d2a",
            ),
            A(
                "m-a-p/SheetSage2",
                repo_path("music/yue2/models/SheetSage2"),
                (
                    "*.py",
                    "config.json",
                    "model.safetensors",
                    "processor_config.json",
                    "render_assets/**",
                    "requirements-render.txt",
                    "requirements.txt",
                ),
                revision="80af707174fc7ee521c25925d5f014729f0e61ae",
            ),
            A(
                "m-a-p/MERT-v2-FullSong",
                repo_path("music/yue2/models/MERT-v2-FullSong"),
                (
                    "config.json",
                    "configuration_mert2.py",
                    "model.safetensors",
                    "modeling_mert2.py",
                    "preprocessor_config.json",
                    "weights_manifest.json",
                ),
                revision="d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
            ),
            QWEN_GUIDE,
            A(
                "Comfy-Org/YuE2",
                model_path("imports/Comfy-Org--YuE2"),
                (
                    "audio_encoders/sheetsage2_bf16.safetensors",
                    "checkpoints/yue2_3b_int8_convrot.safetensors",
                ),
                revision="8e6fcf0f23252ed188b634bd50d44f4b01fba890",
                placements=(
                    (
                        "audio_encoders/sheetsage2_bf16.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/audio_encoders/sheetsage2_bf16.safetensors"),
                    ),
                    (
                        "checkpoints/yue2_3b_int8_convrot.safetensors",
                        repo_path("workflows/V2V/ComfyUI/models/checkpoints/yue2_3b_int8_convrot.safetensors"),
                    ),
                ),
            ),
        ),
    ),
    Bundle(
        "stemkit",
        "StemKit studio-quality Roformer vocals checkpoint",
        "73",
        (
            A(
                "KimberleyJSN/melbandroformer",
                model_path("stemkit/roformer"),
                ("MelBandRoformer.ckpt",),
            ),
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
    # Preserve the former command-line name while presenting the canonical
    # project name in the menu and documentation.
    by_key["kijai"] = by_key["v2v"]
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
    if artifact.direct_url:
        download_direct_artifact(artifact, number, total)
        materialize_placements(artifact)
        return
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_direct_artifact(artifact: Artifact, number: int, total: int) -> None:
    """Download one upstream release asset with resume and a pinned digest."""
    if not artifact.direct_url or not artifact.direct_filename or not artifact.sha256:
        raise ValueError(f"Incomplete direct artifact declaration: {artifact.repo_id}")
    destination = artifact.destination / artifact.direct_filename
    if destination.is_file():
        if (artifact.expected_size is None or destination.stat().st_size == artifact.expected_size) and _sha256(destination) == artifact.sha256:
            print(f"\n[{number}/{total}] {artifact.repo_id}")
            print(f"  ready: {destination} (SHA-256 verified)")
            return
        raise FileExistsError(f"Refusing to replace an unverified direct artifact: {destination}")
    partial = destination.with_name(destination.name + ".partial")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "mm-tools-model-downloader/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    print(f"\n[{number}/{total}] {artifact.repo_id}")
    print(f"  release: {artifact.direct_url}")
    print(f"  destination: {destination}")
    with urllib.request.urlopen(urllib.request.Request(artifact.direct_url, headers=headers), timeout=60) as response:
        resumed = offset > 0 and getattr(response, "status", 200) == 206
        mode = "ab" if resumed else "wb"
        with partial.open(mode) as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
    if artifact.expected_size is not None and partial.stat().st_size != artifact.expected_size:
        raise RuntimeError(
            f"Direct artifact has {partial.stat().st_size} bytes; expected {artifact.expected_size}: {partial}"
        )
    actual = _sha256(partial)
    if actual != artifact.sha256:
        raise RuntimeError(f"SHA-256 mismatch for {partial.name}: {actual}")
    os.replace(partial, destination)
    print(f"  ready: {destination} (SHA-256 verified)")


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
