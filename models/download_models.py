from __future__ import annotations

import argparse
import filecmp
import fnmatch
import hashlib
import inspect
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import urllib.request
import zipfile
from multiprocessing import get_context
from urllib.parse import quote
from dataclasses import dataclass, replace
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
    dir_links: tuple[Path, ...] = ()
    file_links: tuple[tuple[str, Path], ...] = ()
    direct_url: str | None = None
    direct_filename: str | None = None
    sha256: str | None = None
    expected_size: int | None = None
    repo_type: str = "model"


@dataclass(frozen=True)
class Bundle:
    key: str
    title: str
    inventory: str
    artifacts: tuple[Artifact, ...]
    # (tier title, artifacts) pairs; interactive selection re-prompts for one
    # or more tiers, non-interactive selection uses default_tiers when set and
    # every tier otherwise.
    variants: tuple[tuple[str, tuple[Artifact, ...]], ...] = ()
    default_tiers: tuple[int, ...] | None = None
    prompt_note: str = ""


def model_path(relative: str) -> Path:
    return MODELS / relative


def repo_path(relative: str) -> Path:
    return ROOT / relative


A = Artifact


def bria_artifacts(tier: str | None = None) -> tuple[Artifact, ...]:
    manifest = json.loads((MODELS / "manifests/bria-rmbg3.json").read_text())
    return tuple(
        A(group["repository"], model_path(f"bria-rmbg3/{dest}") / Path(row["path"]).parent,
          revision=row["revision"], direct_url=(
              f"https://modelscope.cn/models/{group['repository']}/resolve/{row['revision']}/{quote(row['path'])}"
          ), direct_filename=Path(row["path"]).name, sha256=row["sha256"], expected_size=row["size"])
        for dest, group in manifest.items() if tier is None or dest == tier
        for row in group["files"]
    )


def motionbricks_artifacts() -> tuple[Artifact, ...]:
    manifest = json.loads((MODELS / "manifests/motionbricks.json").read_text())
    repository, revision = manifest["repository"], manifest["revision"]
    artifacts = []
    for file in manifest["files"]:
        path = Path(file["path"])
        host = "media.githubusercontent.com/media" if file["lfs"] else "raw.githubusercontent.com"
        artifacts.append(A(
            repository,
            model_path(f"motionbricks/{revision}") / path.parent,
            revision=revision,
            direct_url=f"https://{host}/{repository}/{revision}/{quote(file['path'])}",
            direct_filename=path.name,
            sha256=file["sha256"],
            expected_size=file["size"],
        ))
    return tuple(artifacts)


QWEN_GUIDE = A(
    "SergiusFlavius/Qwen3-VL-4B-Instruct-heretic-NVFP4",
    model_path("qwen/text-encoder-vl-nvfp4"),
    ("qwen3_vl_4b_nvfp4_full.safetensors",),
)


def scenebloom_artifacts() -> tuple[Artifact, ...]:
    manifest = json.loads((MODELS / "manifests/scenebloom.json").read_text())
    root = model_path("scenebloom")
    result = [A(row["repo"], root / row["dest"], tuple(row["patterns"]),
                revision=row["revision"], cache_layout=row.get("cache", False),
                repo_type=row.get("type", "model")) for row in manifest["snapshots"]]
    result.append(A("facebookresearch/dinov2", root / "checkpoints/dinov2",
                    direct_url=manifest["dinov2"]["url"],
                    direct_filename="dinov2_vitb14_pretrain.pth",
                    sha256=manifest["dinov2"]["sha256"]))
    return tuple(result)

# TRELLIS.2 and Pixal3D name these repositories from their pipeline JSON.  Keep
# the dependencies local and pinned so neither sculpting runtime can silently
# contact the Hub during model construction.  Only the one legacy sparse-
# structure decoder absent from TRELLIS.2-4B is placed into that checkpoint
# tree; the source snapshot remains intact for resume/checksum purposes.
SCULPT_SPARSE_DECODER = A(
    "microsoft/TRELLIS-image-large",
    model_path("sculpting/microsoft--TRELLIS-image-large"),
    (
        "ckpts/ss_dec_conv3d_16l8_fp16.json",
        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
    ),
    revision="25e0d31ffbebe4b5a97464dd851910efc3002d96",
    file_links=(
        (
            "ckpts/ss_dec_conv3d_16l8_fp16.json",
            repo_path("sculpting/trellis/pretrained/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.json"),
        ),
        (
            "ckpts/ss_dec_conv3d_16l8_fp16.safetensors",
            repo_path("sculpting/trellis/pretrained/TRELLIS.2-4B/ckpts/ss_dec_conv3d_16l8_fp16.safetensors"),
        ),
    ),
)
SCULPT_DINO = A(
    "camenduru/dinov3-vitl16-pretrain-lvd1689m",
    model_path("sculpting/camenduru--dinov3-vitl16-pretrain-lvd1689m"),
    (
        "config.json",
        "model.safetensors",
        "preprocessor_config.json",
    ),
    revision="3c276edd87d6f6e569ff0c4400e086807d0f3881",
    dir_links=(
        repo_path("sculpting/trellis/pretrained/deps/camenduru--dinov3-vitl16-pretrain-lvd1689m"),
    ),
)
SCULPT_RMBG = A(
    # TRELLIS.2's BiRefNet wrapper accepts this checkpoint through the same
    # Transformers remote-code contract.  Unlike the pipeline's briaai alias,
    # the official ZhengPeng7 model is ungated, so unattended setup stays
    # deterministic instead of stalling on a click-through approval.
    "ZhengPeng7/BiRefNet",
    model_path("sculpting/ZhengPeng7--BiRefNet"),
    (
        "BiRefNet_config.py",
        "birefnet.py",
        "config.json",
        "model.safetensors",
    ),
    revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
    dir_links=(repo_path("sculpting/trellis/pretrained/deps/ZhengPeng7--BiRefNet"),),
)
SCULPT_MOGE = A(
    "Ruicheng/moge-2-vitl",
    model_path("sculpting/Ruicheng--moge-2-vitl"),
    ("model.pt",),
    revision="39c4d5e957afe587e04eec59dc2bcc3be5ecd968",
    dir_links=(repo_path("sculpting/trellis/pretrained/deps/Ruicheng--moge-2-vitl"),),
)
# sHEL1562/shelling (src/kr3ahardtofinds) mirrors the files upstream keeps off
# the Hub: GitHub release assets, single members of release archives, and the
# off-HF VBench scorer weights. The pinned revision holds exact copies
# (src/MANIFEST.sha256 in that repository); per-file digests stay pinned below.
ARCHIVE_REPO = "sHEL1562/shelling"
ARCHIVE_REV = "4b859a1bfa801727c84087efd455f8c0a3205902"
ARCHIVE_ROOT = "src/kr3ahardtofinds"
ARCHIVE_RAW = f"https://huggingface.co/{ARCHIVE_REPO}/resolve/{ARCHIVE_REV}/{ARCHIVE_ROOT}"

SCULPT_NAF = A(
    "valeoai/NAF release checkpoint",
    model_path("sculpting/valeoai--NAF"),
    direct_url=f"{ARCHIVE_RAW}/trellis/naf_release.pth",
    direct_filename="naf_release.pth",
    sha256="c096c1ab2217a5c3ac136365f721685e2201379cb69d509cfb0261183847c98f",
    expected_size=2664431,
    dir_links=(repo_path("sculpting/trellis/pretrained/deps/valeoai--NAF"),),
)
FIRE3D_ANYUP = A(
    "wimmerth/AnyUp multi-backbone checkpoint",
    model_path("fire3d/wimmerth--AnyUp"),
    direct_url=f"{ARCHIVE_RAW}/Fire3D/anyup_multi_backbone.pth",
    direct_filename="anyup_multi_backbone.pth",
    sha256="b6cc407da8986c7e5c9098e61f7531767a9aca8fff20a1bc6c99d488e61aac59",
    expected_size=3541624,
    file_links=(
        (
            "anyup_multi_backbone.pth",
            repo_path("world-gen/fire3d/checkpoints/Fire3D/external/anyup_multi_backbone.pth"),
        ),
    ),
)
SOMA_X = A(
    "nvidia/SOMA-X",
    model_path("motion-sim/nvidia--SOMA-X"),
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
    dir_links=(repo_path("motion-sim/nvidia-sim/SOMA-X/assets"),),
)

FLOW_SAM_REPO = "jetjodh/sam-3d-body-dinov3"
FLOW_SAM_REV = "1f026b2cc9076fd460243dae553cff9b0dcd199d"
FLOW_SAM_DIR = model_path("flowhmr/ckpts/sam-3d-body-dinov3")

FLOW_SAM_MHR = A(ARCHIVE_REPO, FLOW_SAM_DIR / "assets",
                 direct_url=f"{ARCHIVE_RAW}/FAB/mhr/mhr_model.pt",
                 direct_filename="mhr_model.pt", expected_size=696110248,
                 sha256="352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc")


def fab_body_artifacts() -> tuple[Artifact, ...]:
    body = model_path("fab/body_models")
    maps = (
        ("head_hand_mask.npz", 222040, "60a64a305731dcc1c09f167dfa844246443b3115ae05abebe3813aa64a9ba2ae"),
        ("mhr2smpl_mapping.npz", 170721, "4bfe2fd6abb598f38ba3a3aa5506ac45b4efb3cdaed7fe4757cfbbf20bbe6ac7"),
        ("mhr2smplx_mapping.npz", 249356, "d6d023f9f483be73cd4491d0ddf217450627263405c9bc6bf43795dfff809af1"),
        ("mhr_face_mask.ply", 774669, "db8ed4fca24a744b1e1f4ed9562f55aff837c9f270debbb34a9d1537a3ac62e1"),
        ("smpl2mhr_mapping.npz", 590582, "84f908ad534f470dc5db072a4aeb9ccb4f9aa4484d2bce2558cc1a346df95c98"),
        ("smplx2mhr_mapping.npz", 454393, "94742d378ccbd6058ce8aa0221d16726d1e5def45490685c223b2e87c6070f97"),
        ("subsampled_vertex_indices.npy", 18567, "76008acafd4468dda802e3ee5523b29fef4de8408cc5e1c65351d4145efc01aa"),
    )
    reg = A("nkolot/GraphCMR", body / "smpl",
            direct_url=f"{ARCHIVE_RAW}/FAB/mhr/J_regressor_h36m.npy",
            direct_filename="J_regressor_h36m.npy", expected_size=937168,
            sha256="c655cd7013d7829eb9acbebf0e43f952a3fa0305a53c35880e39192bfb6444a0")
    return (reg, *(A("facebookresearch/MHR", body / "mhr",
                     direct_url=f"{ARCHIVE_RAW}/FAB/mhr/{name}",
                     direct_filename=name, sha256=digest, expected_size=size)
                   for name, size, digest in maps))


def fab_g1_artifacts() -> tuple[Artifact, ...]:
    groups = json.loads((MODELS / "manifests/fab-g1.json").read_text())
    return tuple(
        A(g["repo"], model_path(f"fab/g1-sources/{group}") / Path(name).parent,
          revision=g["revision"],
          direct_url=f"https://raw.githubusercontent.com/{g['repo']}/{g['revision']}/{g['folder']}/{quote(name)}",
          direct_filename=Path(name).name, expected_size=size, sha256=sha)
        for group, g in groups.items() for name, size, sha in g["files"]
    )


BUNDLES: tuple[Bundle, ...] = (
    Bundle(
        "ideogram",
        "Ideogram generation and editing",
        "1-3, 40",
        (
            A("Comfy-Org/Ideogram-4", model_path("comfy--Ideogram-4"),
              ("diffusion_models/ideogram4_int8_convrot.safetensors",
               "diffusion_models/ideogram4_unconditional_nvfp4_mixed.safetensors",
               "text_encoders/qwen3vl_8b_fp8_scaled.safetensors",
               "vae/flux2-vae.safetensors"),
              revision="2aa6c75ce6d5fabded0ca4d0f76abbfaf8edc87d"),
            A("jixin0101/ObjectClear", model_path("jixin0101--ObjectClear"),
              ("model_index.json", "*/config.json", "*/model.safetensors", "*/model.fp16.safetensors",
               "*/diffusion_pytorch_model.safetensors", "*/diffusion_pytorch_model.fp16.safetensors",
               "tokenizer/*", "tokenizer_2/*", "scheduler/*", "README.md")),
            A("ZhengPeng7/BiRefNet", model_path("ZhengPeng7--BiRefNet"),
              ("BiRefNet_config.py", "birefnet.py", "handler.py", "config.json",
               "model.safetensors", "requirements.txt", "README.md")),
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
                ("model.safetensors", "config.json", "LICENSE", "README.md"),
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
            A("MuScriptor/muscriptor-large", model_path("MuScriptor--muscriptor-large"),
              ("model.safetensors", "config.json", "README.md")),
            A("MuScriptor/assets", model_path("MuScriptor--assets"),
              ("MuseScore_General.sf2", "MuseScore_General.sf3", "beat_this-final0.ckpt", "README.md")),
        ),
    ),
    Bundle(
        "musvit",
        "MusVIT score understanding",
        "8-9",
        (
            A("PRAIG/musvit", model_path("PRAIG--musvit"),
              ("model.safetensors", "config.json", "README.md")),
            A("PRAIG/smt-fp-grandstaff", model_path("PRAIG--smt-fp-grandstaff"),
              ("model.safetensors", "config.json", "README.md")),
        ),
    ),
    Bundle(
        "whisper",
        "CrisperWhisper speech recognition",
        "10",
        (A("nyralabs/CrisperWhisper2.0_large", model_path("nyralabs--CrisperWhisper2.0_large"),
           ("model.safetensors", "*.json", "merges.txt", "vocab.json", "tokenizer.json",
            "LICENSE.md", "README.md")),),
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
                ("*.bin", "*.xml", "*.json", "merges.txt", "vocab.json", "tokenizer.json", "README.md"),
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
                ("modular_model_index.json", "mellon_pipeline_config.json", "processor/*",
                 "scheduler/*", "text_encoder/*", "tokenizer/*", "transformer/*", "vae/*", "README.md"),
            ),
            A(
                "suzukimain/Qwen-Image-Layered-Control-SDNQ-int4",
                model_path("qwen/suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4"),
                ("model_index.json", "processor/*", "scheduler/*", "text_encoder/*",
                 "tokenizer/*", "transformer/*", "vae/*"),
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
                model_path("redesign/benjiaiplayground--GroundingDINO_SwinB"),
                ("groundingdino_swinb_cogcoor.pth",),
                file_links=(
                    (
                        "groundingdino_swinb_cogcoor.pth",
                        repo_path("visual-modeling/redesign/weights/groundingdino_swinb_cogcoor.pth"),
                    ),
                ),
            ),
            A("facebook/sam2.1-hiera-large",
              model_path("redesign/facebook--sam2.1-hiera-large"),
              direct_url="https://modelscope.cn/models/facebook/sam2.1-hiera-large/resolve/5d3a27da5fe71a9c4e2c87d8322a73fe6359f2bf/sam2.1_hiera_large.pt",
              direct_filename="sam2.1_hiera_large.pt", expected_size=898083611,
              sha256="2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318",
              file_links=(
                  ("sam2.1_hiera_large.pt",
                   repo_path("visual-modeling/redesign/weights/sam2.1_hiera_large.pt")),
              )),
            A(
                "GoGiants1/Hi-SAM",
                model_path("redesign/GoGiants1--Hi-SAM"),
                ("sam_tss_h_textseg.pth", "sam_vit_h_4b8939.pth"),
                file_links=(
                    ("sam_tss_h_textseg.pth",
                     repo_path("visual-modeling/redesign/weights/sam_tss_h_textseg.pth")),
                    ("sam_vit_h_4b8939.pth",
                     repo_path("visual-modeling/redesign/weights/sam_vit_h_4b8939.pth")),
                ),
            ),
            A("iimate/big-lama-pt", model_path("redesign/iimate--big-lama-pt"), ("big-lama.pt",),
              file_links=(("big-lama.pt", repo_path("visual-modeling/redesign/weights/big-lama.pt")),)),
            A("jixin0101/ObjectClear", model_path("redesign/hf-cache"),
              ("model_index.json", "*/config.json", "*/model.safetensors", "*/model.fp16.safetensors",
               "*/diffusion_pytorch_model.safetensors", "*/diffusion_pytorch_model.fp16.safetensors",
               "tokenizer/*", "tokenizer_2/*", "scheduler/*", "README.md"),
              cache_layout=True,
              file_links=(
                  ("models--jixin0101--ObjectClear",
                   repo_path("visual-modeling/redesign/weights/models--jixin0101--ObjectClear")),
              )),
            A("google-bert/bert-base-uncased", model_path("redesign/hf-cache"), cache_layout=True,
              file_links=(
                  ("models--google-bert--bert-base-uncased",
                   repo_path("visual-modeling/redesign/weights/models--google-bert--bert-base-uncased")),
              )),
            A(
                "PaddlePaddle/PP-OCRv5_server_det",
                model_path("redesign/PaddlePaddle--PP-OCRv5_server_det"),
                ("inference.json", "inference.pdiparams", "inference.yml", "config.json", "README.md"),
                dir_links=(Path.home() / ".paddlex/official_models/PP-OCRv5_server_det",),
            ),
            A(
                "PaddlePaddle/PP-OCRv5_server_rec",
                model_path("redesign/PaddlePaddle--PP-OCRv5_server_rec"),
                ("inference.json", "inference.pdiparams", "inference.yml", "config.json", "README.md"),
                dir_links=(Path.home() / ".paddlex/official_models/PP-OCRv5_server_rec",),
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
                        repo_path("workflows/ComfyUI/models/diffusion_models/wan_2.1_idv2v_int8_convrot.safetensors"),
                    ),
                    (
                        "wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors",
                        repo_path("workflows/ComfyUI/models/diffusion_models/wan_2.1_idv2v_with_normal_depth_int8_convrot.safetensors"),
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
                        repo_path("workflows/ComfyUI/models/clip_vision/clip_vision_h.safetensors"),
                    ),
                    (
                        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                        repo_path("workflows/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
                    ),
                    (
                        "split_files/vae/wan_2.1_vae.safetensors",
                        repo_path("workflows/ComfyUI/models/vae/Wan2_1_VAE_bf16.safetensors"),
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
                model_path("motion-sim/AntResearch--4DAnyone"),
                (
                    "4danyone/*",
                    "gvhmr/*",
                    "perceptual/*",
                ),
                revision="4c80e87b805a5f8461cf339cdbe2fb4249e585aa",
                file_links=(
                    ("4danyone", repo_path("motion-sim/4d-ify/models/4danyone")),
                    ("gvhmr", repo_path("motion-sim/4d-ify/models/gvhmr")),
                    ("perceptual", repo_path("motion-sim/4d-ify/models/perceptual")),
                ),
            ),
            A(
                "ZhengPeng7/BiRefNet",
                model_path("motion-sim/ZhengPeng7--BiRefNet"),
                (
                    "BiRefNet_config.py",
                    "birefnet.py",
                    "config.json",
                    "model.safetensors",
                    "requirements.txt",
                ),
                revision="e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4",
                dir_links=(repo_path("motion-sim/4d-ify/models/birefnet"),),
            ),
            A(
                "camenduru/GVHMR",
                model_path("motion-sim/camenduru--GVHMR"),
                ("dpvo/dpvo.pth",),
                revision="21b32d5389e2e59c0737d4c4095bbc0b8c23f66b",
                file_links=(
                    ("dpvo/dpvo.pth", repo_path("motion-sim/4d-ify/models/gvhmr/dpvo.pth")),
                ),
            ),
        ),
    ),
    Bundle(
        "lingbot",
        "LingBot-World-V2 14B world model (native studio + GGUF tiers)",
        "43",
        (
            # The Rebels GGUF workflow is self-sufficing on the Comfy side: its
            # UMT5-XXL encoder GGUF, VAE, and reference graph ship in this one
            # repository, so every GGUF/NVFP4 tier shares this support set.
            A(
                "realrebelai/LingBot_World_V2_ComfyUI",
                model_path("lingbot/realrebelai--LingBot_World_V2_ComfyUI"),
                (
                    "LINGBOT WORLD GGUF workflow.json",
                    "README.md",
                    "Wan2.1_VAE.pth",
                    "umt5-xxl-encoder-Q4_K_S.gguf",
                ),
                revision="79be5d5f30d63327ac6020ced7f653cdbc301f49",
            ),
        ),
        variants=(
            (
                "14B native BF16 causal-fast DiT with shared UMT5/VAE/tokenizer (studio path; 64 GB-class)",
                (
                    A(
                        "robbyant/lingbot-world-v2-14b-causal-fast",
                        model_path("lingbot/robbyant--lingbot-world-v2-14b-causal-fast"),
                        ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "config.json",
                         "google/umt5-xxl/*", "transformers/*", "README.md"),
                        revision="5c33dd40b213598c418fd25bff30fdbd23fd38a7",
                        dir_links=(
                            repo_path("world-gen/lingbot/lingbot-world-v2-14b-causal-fast"),
                        ),
                    ),
                ),
            ),
            (
                "14B GGUF Q4_K_S (ComfyUI Rebels nodes; 8 GB+ VRAM)",
                (
                    A(
                        "realrebelai/LingBot_World_V2_ComfyUI",
                        model_path("lingbot/realrebelai--LingBot_World_V2_ComfyUI"),
                        ("LingBot-World-14B-Q4_K_S.gguf",),
                        revision="79be5d5f30d63327ac6020ced7f653cdbc301f49",
                    ),
                ),
            ),
            (
                "14B GGUF Q5_K_M (ComfyUI Rebels nodes; 12 GB+ VRAM)",
                (
                    A(
                        "realrebelai/LingBot_World_V2_ComfyUI",
                        model_path("lingbot/realrebelai--LingBot_World_V2_ComfyUI"),
                        ("LingBot-World-14B-Q5_K_M.gguf",),
                        revision="79be5d5f30d63327ac6020ced7f653cdbc301f49",
                    ),
                ),
            ),
            (
                "14B GGUF Q6_K (ComfyUI Rebels nodes; 16 GB+ VRAM)",
                (
                    A(
                        "realrebelai/LingBot_World_V2_ComfyUI",
                        model_path("lingbot/realrebelai--LingBot_World_V2_ComfyUI"),
                        ("LingBot-World-14B-Q6_K.gguf",),
                        revision="79be5d5f30d63327ac6020ced7f653cdbc301f49",
                    ),
                ),
            ),
            (
                "14B GGUF Q8_0 (ComfyUI Rebels nodes; 24 GB+ VRAM)",
                (
                    A(
                        "realrebelai/LingBot_World_V2_ComfyUI",
                        model_path("lingbot/realrebelai--LingBot_World_V2_ComfyUI"),
                        ("LingBot-World-14B-Q8_0.gguf",),
                        revision="79be5d5f30d63327ac6020ced7f653cdbc301f49",
                    ),
                ),
            ),
            (
                "14B hybrid NVFP4 transformer + Diffusers base without its own DiT (SGLang serving; 72 GB+ Blackwell)",
                (
                    A(
                        "igorls/LingBot-World-V2-14B-Causal-Fast-NVFP4",
                        model_path("lingbot/igorls--LingBot-World-V2-14B-Causal-Fast-NVFP4"),
                        ("*.safetensors", "*.json", "README.md", "LICENSE*"),
                        revision="5ae4f45969a224ca8c9fa2072cc49468a2d74b34",
                    ),
                    A(
                        "robbyant/lingbot-world-v2-14b-causal-fast-diffusers",
                        model_path("lingbot/robbyant--lingbot-world-v2-14b-causal-fast-diffusers"),
                        (
                            "README.md",
                            "model_index.json",
                            "scheduler/*",
                            "text_encoder/*",
                            "tokenizer/*",
                            "vae/*",
                        ),
                        revision="59cccf49f2d2dd27418ae7a04b82b10868d455c2",
                    ),
                ),
            ),
        ),
        default_tiers=(5,),
        prompt_note=(
            "Studio tier 1 needs 64 GiB+ VRAM; an RTX 5090 (32 GB) runs the GGUF path, "
            "tiers 2-5, through the bundled ComfyUI Rebels nodes, with tier 5 (Q8_0) as "
            "the quality ceiling there; tier 6 targets 72 GB+ Blackwell under SGLang."
        ),
    ),
    Bundle(
        "fire3d",
        "Fire3D single-image scene reconstruction",
        "44",
        (
            A(
                "hongchi/Fire3D",
                model_path("fire3d/hongchi--Fire3D"),
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
                dir_links=(repo_path("world-gen/fire3d/checkpoints/Fire3D"),),
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
                model_path("aukspeech/tencent--AuK"),
                (
                    "auk_base.safetensors",
                    "config.yaml",
                    "vae.safetensors",
                ),
                revision="790742b71a4430120daf2b2099192abae449eb9f",
                dir_links=(repo_path("vocals/tts/aukspeech/ckpts/AuK"),),
            ),
            A(
                "Qwen/Qwen2.5-Omni-3B",
                model_path("aukspeech/Qwen--Qwen2.5-Omni-3B"),
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
                dir_links=(repo_path("vocals/tts/aukspeech/ckpts/Qwen2.5-Omni-3B"),),
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
                model_path("sculpting/microsoft--TRELLIS.2-4B"),
                ("pipeline.json", "texturing_pipeline.json", "ckpts/*"),
                revision="af44b45f2e35a493886929c6d786e563ec68364d",
                dir_links=(repo_path("sculpting/trellis/pretrained/TRELLIS.2-4B"),),
            ),
            A(
                "TencentARC/Pixal3D",
                model_path("sculpting/TencentARC--Pixal3D"),
                (
                    "pipeline.json",
                    "pipeline_mv.json",
                    "ckpts/*",
                ),
                revision="b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
                dir_links=(repo_path("sculpting/trellis/world/pretrained/Pixal3D"),),
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
                model_path("sculpting/TencentARC--Pixal3D"),
                (
                    "pipeline.json",
                    "pipeline_mv.json",
                    "ckpts/*",
                ),
                revision="b0cb2e1b794cab9aa0ac38a95d794a4d9337437f",
                dir_links=(repo_path("sculpting/trellis/world/pretrained/Pixal3D"),),
            ),
            A(
                "AlayaLab/WorldSculpt",
                model_path("sculpting/AlayaLab--WorldSculpt"),
                (
                    "shape_ft1024_mv_lora_ibr_texverse_fixedmem05/config.json",
                    "shape_ft1024_mv_lora_ibr_texverse_fixedmem05/ckpts/*.pt",
                    "ss_ft64_mv_lora_ibr_texverse/config.json",
                    "ss_ft64_mv_lora_ibr_texverse/ckpts/*.pt",
                ),
                revision="8cb81056d803c61371dd84ef18a14142a738610e",
                file_links=(
                    (
                        "shape_ft1024_mv_lora_ibr_texverse_fixedmem05",
                        repo_path("sculpting/trellis/world/pretrained/shape_ft1024_mv_lora_ibr_texverse_fixedmem05"),
                    ),
                    (
                        "ss_ft64_mv_lora_ibr_texverse",
                        repo_path("sculpting/trellis/world/pretrained/ss_ft64_mv_lora_ibr_texverse"),
                    ),
                ),
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
                model_path("motion-sim/nvidia--ARDY-Core-RP-20FPS-Horizon40"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="abe6c43beb28c867c950acb824b9c4ef3d63fb76",
                dir_links=(repo_path("motion-sim/nvidia-sim/checkpoints/ARDY-Core-RP-20FPS-Horizon40"),),
            ),
            A(
                "nvidia/ARDY-Core-RP-20FPS-Horizon8",
                model_path("motion-sim/nvidia--ARDY-Core-RP-20FPS-Horizon8"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="257a0843a10bf5201065963d7bca2791e9393a7a",
                dir_links=(repo_path("motion-sim/nvidia-sim/checkpoints/ARDY-Core-RP-20FPS-Horizon8"),),
            ),
            A(
                "nvidia/ARDY-G1-RP-25FPS-Horizon52",
                model_path("motion-sim/nvidia--ARDY-G1-RP-25FPS-Horizon52"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="059b8007df0ba194a006a877b59a563955ac7b70",
                dir_links=(repo_path("motion-sim/nvidia-sim/checkpoints/ARDY-G1-RP-25FPS-Horizon52"),),
            ),
            A(
                "nvidia/ARDY-G1-RP-25FPS-Horizon8",
                model_path("motion-sim/nvidia--ARDY-G1-RP-25FPS-Horizon8"),
                ("config.yaml", "denoiser.safetensors", "tokenizer.safetensors", "stats/*/*.npy"),
                revision="334a8a9cdbafc962dcd304c26f31e6b17a869355",
                dir_links=(repo_path("motion-sim/nvidia-sim/checkpoints/ARDY-G1-RP-25FPS-Horizon8"),),
            ),
            A(
                "voxta/Llama-3-8B-LLM2Vec-ARDY-INT8",
                model_path("motion-sim/voxta--Llama-3-8B-LLM2Vec-ARDY-INT8"),
                (
                    "chat_template.jinja",
                    "config.json",
                    "llm2vec_config.json",
                    "model.safetensors",
                    "tokenizer.json",
                    "tokenizer_config.json",
                ),
                revision="96c7eb1cc9100cd3925662d1093926c00b64a698",
                dir_links=(
                    repo_path("motion-sim/nvidia-sim/text-encoders/voxta/Llama-3-8B-LLM2Vec-ARDY-INT8"),
                ),
            ),
            SOMA_X,
        ),
    ),
    Bundle(
        "yue2",
        "YuE2 native and Comfy music generation/cover stack",
        "59-63",
        (
            A(
                "m-a-p/YuE2-3B",
                model_path("yue2/m-a-p--YuE2-3B"),
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
                dir_links=(repo_path("music/yue2/models/YuE2-3B"),),
            ),
            A(
                "m-a-p/YuE2-Vae",
                model_path("yue2/m-a-p--YuE2-Vae"),
                (
                    "config.json",
                    "model.safetensors",
                    "modeling_vae.py",
                    "weights_manifest.json",
                ),
                revision="9a94e1d0ea9f8087e98f77fa88df4a4068104d2a",
                dir_links=(repo_path("music/yue2/models/YuE2-Vae"),),
            ),
            A(
                "m-a-p/SheetSage2",
                model_path("yue2/m-a-p--SheetSage2"),
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
                dir_links=(repo_path("music/yue2/models/SheetSage2"),),
            ),
            A(
                "m-a-p/MERT-v2-FullSong",
                model_path("yue2/m-a-p--MERT-v2-FullSong"),
                (
                    "config.json",
                    "configuration_mert2.py",
                    "model.safetensors",
                    "modeling_mert2.py",
                    "preprocessor_config.json",
                    "weights_manifest.json",
                ),
                revision="d8ba1c745e733b3908ce6ad16ebeb17ac7600a42",
                dir_links=(repo_path("music/yue2/models/MERT-v2-FullSong"),),
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
                        repo_path("workflows/ComfyUI/models/audio_encoders/sheetsage2_bf16.safetensors"),
                    ),
                    (
                        "checkpoints/yue2_3b_int8_convrot.safetensors",
                        repo_path("workflows/ComfyUI/models/checkpoints/yue2_3b_int8_convrot.safetensors"),
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
    Bundle(
        "motionbricks",
        "MotionBricks SOMA / Unitree G1 models, rigs, motions and scenes",
        "74",
        motionbricks_artifacts(),
    ),
    Bundle(
        "flowhmr", "FlowHMR video reconstruction (licensed SMPL-H supplied separately)", "75-76", (),
        variants=(
            ("reconstruction model pack (core)", (
                A("fafsaf1/flowhmr-0.46B", model_path("flowhmr/checkpoints"), ("flowhmr_latest/*",), revision="1eb105e554e0cfc1846ea5feac8c2d516a36d319"),
                A(FLOW_SAM_REPO, FLOW_SAM_DIR, ("model.ckpt", "model_config.yaml"), revision=FLOW_SAM_REV),
                FLOW_SAM_MHR,
                A("facebook/VGGT-Omega", model_path("flowhmr/ckpts/vggt-omega"),
                  direct_url="https://modelscope.cn/models/facebook/VGGT-Omega/resolve/f4fe2040e400351980060e5391e12f00a6c037b2/vggt_omega_1b_512.pt",
                  direct_filename="vggt_omega_1b_512.pt", expected_size=4576706117,
                  sha256="c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934"),
                A("kadirnar/yolox_l-v0.1.1", model_path("flowhmr/ckpts/yolox"), ("yolox_l.pth",), revision="f63e87ffbded0e30713d32dba22501f56fe5007e"),
                A("zju3dv/GVHMR body regressor", model_path("flowhmr/assets/body_models"),
                  direct_url="https://raw.githubusercontent.com/zju3dv/GVHMR/ee960bb6e2ea2d381aa97f08e9b71ef320b624b1/hmr4d/utils/body_model/smpl_neutral_J_regressor.pt",
                  direct_filename="smpl_neutral_J_regressor.pt", expected_size=662187,
                  sha256="70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf"),
            )),
            ("optional PHC tracking (licensed Isaac Gym / SMPL supplied separately)", (
                A("sHEL1562/shelling", model_path("flowhmr/archive/bodymesh"),
                  ("src/kr3ahardtofinds/bodymesh/*",), revision="b7e899fd313b8040f1bbc6d13e4bb0bd6169052a",
                  placements=(
                      ("src/kr3ahardtofinds/bodymesh/Humanoid_00023501.pth",
                       model_path("flowhmr/output/HumanoidIm/phc_comp_3/Humanoid.pth")),
                      ("src/kr3ahardtofinds/bodymesh/Humanoid_00258000.pth",
                       model_path("flowhmr/output/HumanoidIm/phc_3/Humanoid.pth")),
                      ("src/kr3ahardtofinds/bodymesh/amass_isaac_gender_betas.pkl",
                       model_path("flowhmr/assets/example/amass_isaac_gender_betas.pkl")),
                      ("src/kr3ahardtofinds/bodymesh/amass_isaac_gender_betas_unique.pkl",
                       model_path("flowhmr/assets/example/amass_isaac_gender_betas_unique.pkl")),
                  )),
            )),
        ),
        default_tiers=(1,),
    ),
    Bundle(
        "physag3d", "PhysAg3D articulated assets (TRELLIS.2 / DINOv3 / BiRefNet)", "77",
        (
            A("microsoft/TRELLIS.2-4B", model_path("sculpting/microsoft--TRELLIS.2-4B"),
              ("pipeline.json", "texturing_pipeline.json", "ckpts/*"),
              revision="af44b45f2e35a493886929c6d786e563ec68364d",
              dir_links=(repo_path("sculpting/trellis/pretrained/TRELLIS.2-4B"),)),
            SCULPT_SPARSE_DECODER, SCULPT_DINO, SCULPT_RMBG,
        ),
    ),
    Bundle(
        "idol", "Idol image-to-splats (FLUX.2 VAE, Space examples)", "78",
        (
            A("speridlabs/idol", model_path("idol/weights"),
              ("config.json", "*.safetensors", "README.md"),
              revision="2731c2291fc65a797c5cb627a5e5526fc1490902"),
            A("facebook/dinov3-vith16plus-pretrain-lvd1689m", model_path("idol/dinov3"),
              direct_url="https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/a7e110a46be5aab90d0d369a986728826fec06ee/model.safetensors",
              direct_filename="model.safetensors", expected_size=3362432800,
              sha256="3e1d4d18b9bfa9f28fad8e9de6a783f1313532d3460efa4cd0b12521d81d1a4d"),
            A("facebook/dinov3-vith16plus-pretrain-lvd1689m", model_path("idol/dinov3"),
              direct_url="https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/b9b800e3218a3ae46bc4adeabd7b733f47d1332d/config.json",
              direct_filename="config.json", expected_size=744,
              sha256="35770e98e425c9383534bcbfa7f3b7a3cb1d943f6d26a37d7d19fea0735eab57"),
            A("facebook/dinov3-vith16plus-pretrain-lvd1689m", model_path("idol/dinov3"),
              direct_url="https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/b9b800e3218a3ae46bc4adeabd7b733f47d1332d/preprocessor_config.json",
              direct_filename="preprocessor_config.json", expected_size=585,
              sha256="960c41d1f3a7778b936365769a2d90550b318a6c0a53a0296957adacfe5e0dd7"),
            A("facebook/dinov3-vith16plus-pretrain-lvd1689m", model_path("idol/dinov3"),
              direct_url="https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/98b7096b2938406ade58801a0bf75eca3198b5bd/LICENSE.md",
              direct_filename="LICENSE.md", expected_size=7503,
              sha256="25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"),
            A("facebook/dinov3-vith16plus-pretrain-lvd1689m", model_path("idol/dinov3"),
              direct_url="https://modelscope.cn/models/facebook/dinov3-vith16plus-pretrain-lvd1689m/resolve/98b7096b2938406ade58801a0bf75eca3198b5bd/README.md",
              direct_filename="README.md", expected_size=14528,
              sha256="c5b22ecac789a9c2de7c3115b975f9b00351f68a141d5de175fdfd5c2858b99e"),
            A("black-forest-labs/FLUX.2-dev", model_path("idol/flux2/vae"),
              direct_url="https://modelscope.cn/models/black-forest-labs/FLUX.2-dev/resolve/d72c4754e67efcaca88832fc18ea694547a28bd2/vae/diffusion_pytorch_model.safetensors",
              direct_filename="diffusion_pytorch_model.safetensors", expected_size=336213556,
              sha256="d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"),
            A("black-forest-labs/FLUX.2-dev", model_path("idol/flux2/vae"),
              direct_url="https://modelscope.cn/models/black-forest-labs/FLUX.2-dev/resolve/d72c4754e67efcaca88832fc18ea694547a28bd2/vae/config.json",
              direct_filename="config.json", expected_size=770,
              sha256="7948580b2d53da578d49b9fed61b45d94a4c037ded5057ebd3adb4040bb5aef5"),
            A("black-forest-labs/FLUX.2-dev", model_path("idol/flux2"),
              direct_url="https://modelscope.cn/models/black-forest-labs/FLUX.2-dev/resolve/03c3c9bb08cfd1a5122386c77a61fca5ebcdd0a1/LICENSE.md",
              direct_filename="LICENSE.md", expected_size=18158,
              sha256="468d9f4332c0c895e9132c035982a40a12750092603a2e910913cb49aa887d3b"),
            A("black-forest-labs/FLUX.2-dev", model_path("idol/flux2"),
              direct_url="https://modelscope.cn/models/black-forest-labs/FLUX.2-dev/resolve/03c3c9bb08cfd1a5122386c77a61fca5ebcdd0a1/README.md",
              direct_filename="README.md", expected_size=9439,
              sha256="f161ac5822374c29c1df2ba13ea8567a7a5318d3b0ec6ac0b1b9f00c596f300b"),
            A("speridlabs/idol", model_path("idol/demo"),
              ("examples/*", "README.md"), repo_type="space",
              revision="2fd68f6aaa69df7c178503a788ad8deb4c7d3117"),
        ),
    ),
    Bundle(
        "hy3d", "Hi3D mesh refinement / local background removal", "79-80", (),
        variants=(
            ("Hi3D refinement + background removal (core)", (
                A("hi3d-official/Hi3Dv1.0", model_path("hy3d/weights"),
                  ("model_index.json", "scheduler/*.json", "vae/*.json", "vae/*.safetensors",
                   "transformer_*/*.json", "transformer_*/*.safetensors", "LICENSE", "NOTICE",
                   "AUTHORS", "THIRD_PARTY_LICENSES.md", "sparc3d/ops/flexicubes/LICENSE.txt"),
                  revision="d502e938cbedc6e87aba8884614c5ca6d177a4b6"),
                A("facebook/dinov2-large", model_path("hy3d/dinov2"),
                  direct_url="https://modelscope.cn/models/facebook/dinov2-large/resolve/b8c1ab6eb90a82f1fd5f228985cd8b80efffc0fc/model.safetensors",
                  direct_filename="model.safetensors", expected_size=1217522888,
                  sha256="399fba97a95f22c36834418bc69373364a99af3a1153da1c0fb31db567c92e23"),
                A("facebook/dinov2-large", model_path("hy3d/dinov2"),
                  direct_url="https://modelscope.cn/models/facebook/dinov2-large/resolve/5aba9c6dd22a59bf62a21e9193bccd0a70231eaa/config.json",
                  direct_filename="config.json", expected_size=549,
                  sha256="12df51c069a2dc1305e34ba71ef58bc2407ea553b75f4722a1715c1bce3bbed0"),
                A("facebook/dinov2-large", model_path("hy3d/dinov2"),
                  direct_url="https://modelscope.cn/models/facebook/dinov2-large/resolve/5aba9c6dd22a59bf62a21e9193bccd0a70231eaa/preprocessor_config.json",
                  direct_filename="preprocessor_config.json", expected_size=436,
                  sha256="14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828"),
                A("facebook/dinov2-large", model_path("hy3d/dinov2"),
                  direct_url="https://modelscope.cn/models/facebook/dinov2-large/resolve/5aba9c6dd22a59bf62a21e9193bccd0a70231eaa/README.md",
                  direct_filename="README.md", expected_size=3033,
                  sha256="0be39948cdb4f4c364c505c039cffe26ba8097b7300bf55f361952c8b17a5fee"),
                A("openai/clip-vit-large-patch14", model_path("hy3d/clip"),
                  ("config.json", "model.safetensors", "preprocessor_config.json", "README.md"),
                  revision="32bd64288804d66eefd0ccbe215aa642df71cc41"),
                A("danielgatis/rembg", model_path("hy3d/rembg"),
                  direct_url="https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx",
                  direct_filename="u2net.onnx", expected_size=175997641,
                  sha256="8d10d2f3bb75ae3b6d527c77944fc5e7dcd94b29809d47a739a7a728a912b491"),
            )),
            ("optional image-to-mesh coarse stage (Hunyuan3D-2 mini turbo)", (
                A("tencent/Hunyuan3D-2mini", model_path("hy3d/coarse"),
                  ("hunyuan3d-dit-v2-mini-turbo/config.yaml",
                   "hunyuan3d-dit-v2-mini-turbo/model.fp16.safetensors", "LICENSE", "NOTICE", "README.md"),
                  revision="f90a0f7df7d5e6f71109cf333f6a95a0ae3194a6"),
            )),
        ),
        default_tiers=(1,),
    ),
)


BUNDLES += (Bundle("scenebloom", "SceneBloom rooms + cached assets + local vision", "81", scenebloom_artifacts()),)
BUNDLES += (
    Bundle("hynomas", "Hy-Nomas INT8 world studio + adapter preparation", "82-83", (
        A("Qwen/Qwen2.5-VL-7B-Instruct", model_path("hynomas/text-encoder"),
          ("*.safetensors", "*.json", "*.txt", "*.jinja", "LICENSE", "README.md"),
          revision="cc594898137f460bfe9f0759e9844b3ce807cfb5"),
    ), variants=(
        ("INT8 world studio core + local prompt enhancer", (
            A("tencent/Hy-Nomas", model_path("hynomas"),
              ("DiT/distilled_int8/*", "VAE/PixelUMM_FT/*", "VAE/DCAE-V/*", "PE/*", "LICENSE", "README.md"),
              revision="621df1570098aa1ebf916dac3ac72ae3cbe25caa"),
        )),
        ("cinematic + robotics INT8 adapter preparation", (
            A("tencent/Hy-Nomas", model_path("hynomas"),
              ("DiT/distilled/*", "DiT/lora_weights/*", "LICENSE", "README.md"),
              revision="621df1570098aa1ebf916dac3ac72ae3cbe25caa"),
        )),
    ), default_tiers=(1,)),
)


BUNDLES += (Bundle("unlimited-ocr2", "Unlimited OCR 2 document studio", "84", (
    A("baidu/Unlimited-OCR-2", model_path("unlimited-ocr2"),
      ("*.safetensors", "*.json", "*.py", "*.txt", "LICENSE", "README.md"),
      revision="0480966938291445591f69f08a13b041c073a96a"),
)),)


BUNDLES += (Bundle("pixel-wizard", "WeVisOne-VL-4B perception studio", "85", (
    A("tencent/WeVisOne-VL-4B", model_path("wevisone-vl-4b"),
      ("*.safetensors", "*.pt", "*.json", "*.jinja", "LICENSE", "README.md"),
      revision="6b44a45873b4bbc7dca383922504430dc60eca23"),
    A("hugging-apps/wevisone-vl-4b", model_path("wevisone-vl-4b"),
      ("examples/*.jpg",), repo_type="space",
      revision="ef77acac09937039d2d8cf318dc613090d8f7cba"),
)),)


BUNDLES += (Bundle("nemotron", "Nemotron 3.5 Transcribe NVFP4 studio", "86", (
    A("nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-NVFP4", model_path("nemotron-transcribe/checkpoint"),
      ("*.safetensors", "*.json", "*.jinja", "LICENSE", "licenses/*", "THIRD_PARTY_NOTICES.md", "README.md"),
      revision="66c6adc4020179567074d0399cda2a4eeb2bb347"),
    A("facebookresearch/demucs htdemucs", model_path("nemotron-transcribe/stems"),
      direct_url="https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/955717e8-8726e21a.th",
      direct_filename="955717e8-8726e21a.th",
      sha256="8726e21a993978c7ba086d3872e7608d7d5bfca646ca4aca459ffda844faa8b4"),
)),)


BUNDLES += (Bundle("audio8", "Audio8 Realistic Air speech studio", "87", (
    A("Edge0/Audio8-TTS-Realistic-Air", model_path("audio8-realistic-air"),
      ("*.safetensors", "*.pth", "*.json", "*.py", "*.txt", "LICENSE", "NOTICE", "README.md"),
      revision="23c305cc013fde378edb459f538317043c993a74"),
    A("Edge0/Audio8-TTS-Realistic-Air", model_path("audio8-realistic-air"),
      ("examples/sample-10-reference.wav", "examples/sample-18-reference.wav"), repo_type="space",
      revision="0e7d7ed6626f49e7ac5898365cb116321b610a5d"),
)),)


MING3_REVISION = "a39b80660bdc5a4394d5e1d41af6dbfd9c443a54"
MING3_REFERENCES = tuple(
    A("inclusionAI/Ming-3.0-tts", model_path("ming3/references"),
      direct_url=f"https://raw.githubusercontent.com/inclusionAI/Ming-3.0-tts/{MING3_REVISION}/{path}",
      direct_filename=Path(path).name, sha256=sha, expected_size=size)
    for path, sha, size in (
        ("data/voice_presets/voice_audio/lingguang.wav", "c1e7e796f9e7798db98de29625cf1ae8acb5eaa271171f52223e47ea777b2802", 321690),
        ("data/voice_presets/voice_audio/db30.wav", "1611471cc6ebdda5a207802802ad12d2265b21e61e2ca43b98a2605cf981559c", 39094),
        ("data/wavs/emotion_prompt.wav", "0380398fb7055a0fb00080ad3084024969c35441bd11f7c463b148ac5bce8b4d", 139678),
        ("data/wavs/yue_prompt.wav", "766fa80cd16b1488d4ff01f7c8e6fd3ce5775d215266bfbbe807f73ca89aa2f2", 132558),
    )
) + (A("hugging-apps/ming-3-tts-05b", model_path("ming3/references"), ("voices.json",),
       repo_type="space", revision="d4ecae7bc5c59fb0b8e80461e503e53a8a6ba6bd"),)
BUNDLES += (
    Bundle(
        "ming3", "Ming3 speech (0.5B dense or tiny MoE)", "88-89",
        MING3_REFERENCES,
        variants=(
            ("0.5B dense speech (~3 GB)", (
                A("inclusionAI/Ming-3.0-tts-0.5B", model_path("ming3/0.5B"),
                  ("*.safetensors", "*.onnx", "*.json", "*.txt", "*.py", "*.jinja", "LICENSE", "README.md"),
                  revision="7247f4c94a5b2e987404ed28a5cfec41b1a4ca61"),
            )),
            ("tiny MoE speech + podcast (~18 GB)", (
                A("inclusionAI/Ming-3.0-tts-tiny", model_path("ming3/tiny"),
                  ("*.safetensors", "*.onnx", "*.json", "*.txt", "*.py", "*.jinja", "LICENSE", "README.md"),
                  revision="4715ef5a2f02901f75687d538b6920fd0dd11e6b"),
            )),
        ),
    ),
)


UNDERDOG = json.loads((MODELS / "manifests/underdog.json").read_text())
BUNDLES += (
    Bundle(
        "underdog", "Bark 3.0 English STT (base/plus/clean/pro)", "90-93",
        (
            A(UNDERDOG["processor"]["repo"], model_path("underdog/processor"),
              tuple(UNDERDOG["processor"]["files"]), revision=UNDERDOG["processor"]["revision"]),
        ),
        variants=tuple(
            (f"{row['name']} English ({row['download']})", (
                A(row["repo"], model_path(f"underdog/{key}"),
                  tuple(row["files"]), revision=row["revision"]),
            ))
            for key, row in UNDERDOG["variants"].items()
        ),
    ),
)


# Budget3D reuses Idol's frozen encoders, matting, and decoders in place.
_IDOL_CORE = tuple(a for a in next(b for b in BUNDLES if b.key == "idol").artifacts
                   if a.repo_type != "space")
BUNDLES += (
    Bundle("bud3d", "Budget3D Idol / Idol-Turbo image-to-splats", "94-95", _IDOL_CORE,
           variants=(
               ("Idol full pipeline", (
                   A("BudgetAny/Budget3D-Idol", model_path("bud3d/Idol"),
                     ("*.safetensors", "config.json", "README.md"),
                     revision="2dd6d83c57e7853234dbbb24c54b78edc842d074"),
               )),
               ("Idol-Turbo distilled (+ turbo LoRA)", (
                   A("BudgetAny/Budget3D-Idol-Turbo", model_path("bud3d/Idol-Turbo"),
                     ("*.safetensors", "config.json", "README.md"),
                     revision="ba76450efdc888310741659665ec72b8c7d0ebc0"),
                   A("speridlabs/idol-turbo", model_path("bud3d/Idol-Turbo"),
                     ("idol_turbo_lora.safetensors",), revision="45cdd41f1f4d6d70496daa7d619f57ad79307a37"),
               )),
           ), default_tiers=(1,)),
)


BUNDLES += (Bundle("splat-v3", "SplatAnything reconstruction, example scenes, and gallery", "96-96b", (), variants=(
    ("reconstruction weights + example inputs (core)", (
        A("ritianyu/SplatAnything", model_path("splat-v3"), ("*.ckpt", "*.safetensors", "*.onnx", "README.md"),
          revision="cea57139b63da978424b5faca31e06573261215e"),
        A("ritianyu/SplatAnything_Demo", model_path("splat-v3/demo"), ("examples/**", "README.md"),
          revision="360fcaa98af2b9f1b8539d550365b3910d9206ba", repo_type="dataset"),
        A("ritianyu/SplatAnything", model_path("splat-v3/demo"), ("assets/scene-previews/*", "Notice.txt", "License.txt"),
          revision="bcb83eb8e25b3fafbe52ca319b5c6a050796610a", repo_type="space"),
    )),
    ("optional offline fly-through gallery (showcase)", (
        A("ritianyu/SplatAnything_Demo", model_path("splat-v3/showcase"),
          tuple(f"{category}/{number}/*.html" for category in ("object","indoor","street","aerial","rig","pano")
                for number in range(1,5)), revision="4e2f7cc7b3191600ccbd7a7f0de0eb20b6676402", repo_type="dataset"),
    )),
), default_tiers=(1,),),)


# FastFrames worlds/eval bundles merge below, after the VBench file list.


BUNDLES += (Bundle("inworld-isolator","Inworld Voice Isolator • recording and realtime","99",(
    A("inworld/voice-isolator",model_path("inworld-isolator"),("*.safetensors","LICENSE","NOTICE","README.md"),revision="9b0285b3fe8403aa8e2f1dbb2165a80a4294c970"),
)),)


BUNDLES += (Bundle("sophon-agent","Sophon Realtime Tag + NVFP4A16 thinker","100",(), variants=(
    ("realtime talker + voices (core)", (
        A("adventists-ai/Sophon-Realtime-32B-Tag",model_path("sophon-agent"),("config.json","README*","LICENSE*","NOTICE*","talker/**","voices/**"),revision="ee0fb1bd3db4ddb24ec5efbe75bdebac2a7986fd"),
    )),
    ("NVFP4A16 flash thinker", (
        A("caijunqiang/Sophon-Realtime-32B-Flash-Thinker-NVFP4A16",model_path("sophon-agent/thinker"),("*.safetensors","*.json","*.jinja","*.txt","audio/*","README.md","NOTICE","LICENSE","MODIFICATIONS.md"),revision="d324f5c7b8ba51485040b3d405d1cd6aa5dd399c"),
    )),
), default_tiers=(1,),),)


BUNDLES += (Bundle("fab", "FAB motion, SOMA & ego hands", "101", (
    A("noahcao/FAB",model_path("fab/weights"),("fab-l/*","fab-s/*","fab-h/*","shared/*","*LICENSE*"),revision="f2e204f280aa27944b98fbf783f618a4bea2c223"),
    A("noahcao/FAH",model_path("fab/hands"),("fah/*","shared/*","*LICENSE*"),revision="c2e67a5779c104f85c33c147e8d4c6a8ebdc28b4"),
    A(FLOW_SAM_REPO, FLOW_SAM_DIR, ("model_config.yaml",), revision=FLOW_SAM_REV),
    FLOW_SAM_MHR,
    SOMA_X,
    *fab_body_artifacts(),
), variants=(
    ("demo reels (sample motion captures)", (
        A(ARCHIVE_REPO, model_path("fab/demo-archive"), (f"{ARCHIVE_ROOT}/FAB/demo/*",), revision=ARCHIVE_REV,
          placements=tuple((f"{ARCHIVE_ROOT}/FAB/demo/{name}", model_path("fab/demo") / name)
                           for name in ("dancer_street.mp4", "steps_up.mp4", "joggers.jpg", "skater_seated.jpg"))),
    )),
    ("G1 retarget source meshes (mjlab/unitree)", fab_g1_artifacts()),
), default_tiers=(1,2),),)


BUNDLES += (Bundle("jeveration", "Jeveration generate, edit & dense", "102", (
    A("linlinlinlinsdada/Jeveration",model_path("jeveration/weights"),("*.pt","*LICENSE*"),revision="48c66dceebe76fbfa588072547d5360f8029ee29"),
    A("Qwen/Qwen3.5-4B",model_path("jeveration/qwen"),("*.safetensors","*.json","*.txt","*.jinja","LICENSE*"),revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"),
    A("turkeyju/tokenizer_tatitok_bl128_vq",model_path("jeveration/tokens"),("config.json","README.md"),revision="7650d466a955c754265d20e57513518a4e049f00"),
    A("timm/vit_large_patch14_clip_336.openai",model_path("jeveration/clip"),("open_clip_model.safetensors","*LICENSE*"),revision="81e38efc4637de5023b10e75a7f9bd1c6fa6b010"),
    A("lllyasviel/Annotators",model_path("jeveration/hub"),("ControlNetHED.pth",),cache_layout=True,revision="982e7edaec38759d914a963c48c4726685de7d96"),
    A("depth-anything/Depth-Anything-V2-Small-hf",model_path("jeveration/hub"),("config.json","preprocessor_config.json","model.safetensors"),cache_layout=True,revision="5426e4f0f36572d16453bbda7a8389317b1bef99"),
    A("linlinlinlinsdada/Jeveration",model_path("jeveration/space"),("workflow.json","requirements.txt"),repo_type="space",revision="1cc6c16ef4a532863219fcb86f07a90e9487f914"),
)),)


BUNDLES += (Bundle("bria-rmbg3", "BRIA RMBG-3.0 cutouts", "103", (),
                   variants=(
                       ("matte pipeline (RMBG-3.0)", bria_artifacts("matte")),
                       ("Fibo-2.0 edit-turbo components", bria_artifacts("fibo")),
                   ),
                   default_tiers=(1, 2),),)


BUNDLES += (Bundle("wemm-video","WeMM-Video 35B-A3B AWQ","104",(
    A("tencent/WeMM-Video-35BA3B-AWQ",model_path("wemm-video/awq"),("*.safetensors","*.json","*.py","*.txt","README.md","LICENSE*"),revision="205c63375c223610f99e62d432b816b67b258de0"),
)),)


# Optional VBench scorer weights for the FastFrames evaluation harness
# (world-gen/fastframes/fastframes/evaluation/vbench). Nothing in the studio
# start path needs them; run score.sh/filter_temporal.sh with
# VBENCH_CACHE_DIR=<mm-tools>/models/fastframes/vbench (TORCH_HOME defaults to
# its torch/ subdirectory, hence the second DINO placement). The torch.hub clone
# files under dino_model/facebookresearch_dino_main are the exact set the
# pinned scorer imports.
_VBENCH_FILES = (
    "clip_model/ViT-B-32.pt", "clip_model/ViT-L-14.pt",
    "aesthetic_model/emb_reader/sa_0_4_vit_l_14_linear.pth",
    "umt_model/l16_ptk710_ftk710_ftk400_f16_res224.pth",
    "amt_model/amt-s.pth",
    "dino_model/dino_vitbase16_pretrain.pth",
    "dino_model/facebookresearch_dino_main/hubconf.py",
    "dino_model/facebookresearch_dino_main/vision_transformer.py",
    "dino_model/facebookresearch_dino_main/utils.py",
    "pyiqa_model/musiq_spaq_ckpt-358bb6af.pth",
    "ViCLIP/ViClip-InternVid-10M-FLT.pth",
    "ViCLIP/bpe_simple_vocab_16e6.txt.gz",
    "grit_model/grit_b_densecap_objectdet.pth",
    "caption_model/tag2text_swin_14m.pth",
    "raft_model/models/raft-things.pth",
    *(f"bert-base-uncased/{name}" for name in
      ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt")),
)

BUNDLES += (Bundle("fastframes", "FastFrames playable worlds, text-to-video, and eval scorers", "97-98,105", (), variants=(
    ("playable LingBot worlds (live)", (
        A("apple/FastFrames-S", model_path("fastframes"), ("lingbot-1.3b/*","LICENSE*","README.md"), revision="aa099efcedf1b65447106bf9004cde75821f7387"),
        A("robbyant/lingbot-world-v2-1.3b-causal-fast", model_path("fastframes/lingbot-world-v2-1.3b-causal-fast/transformers"), ("model-*.safetensors", "model.safetensors.index.json", "README.md"), revision="7e36a5f919f86cb4255cc9bfc30adb44963fbde1"),
        A("robbyant/lingbot-world-v2-14b-causal-fast", model_path("fastframes/lingbot-world-v2-14b-causal-fast"), ("models_t5_umt5-xxl-enc-bf16.pth","Wan2.1_VAE.pth","google/umt5-xxl/*","LICENSE*"), revision="5c33dd40b213598c418fd25bff30fdbd23fd38a7"),
    )),
    ("FastFrames-L text to video", (
        A("apple/FastFrames-L", model_path("fastframes"), ("fastframes-l/*","README.md","LICENSE*"), revision="d18ec5f054bb03f6b8e561a63d52af3142b5b81d"),
        A("Wan-AI/Wan2.1-T2V-14B", model_path("fastframes/Wan2.1-T2V-14B"), ("models_t5_umt5-xxl-enc-bf16.pth","google/umt5-xxl/*","LICENSE*"), revision="a064a6c71f5be440641209c07bf2a5ce7a2ff5e4"),
    )),
    ("VBench scorer weights (evaluation, optional)", (
        A(ARCHIVE_REPO, model_path("fastframes/vbench-archive"), (f"{ARCHIVE_ROOT}/FastFrames/vbench/**",),
          revision=ARCHIVE_REV,
          placements=tuple((f"{ARCHIVE_ROOT}/FastFrames/vbench/{name}", model_path("fastframes/vbench") / name)
                           for name in _VBENCH_FILES) + (
              (f"{ARCHIVE_ROOT}/FastFrames/vbench/dino_model/dino_vitbase16_pretrain.pth",
               model_path("fastframes/vbench/torch/hub/checkpoints/dino_vitbase16_pretrain.pth")),
          )),
    )),
),),)


def print_menu() -> None:
    print("\nMM Tools model downloader")
    print(f"Destination root: {MODELS}")
    print("Bundle numbers select complete runnable projects; inventory numbers refer to which-ones.txt.\n")
    for number, bundle in enumerate(BUNDLES, start=1):
        count = len(bundle.artifacts)
        suffix = "s" if count != 1 else ""
        tiers = ""
        if bundle.variants:
            tiers = f" · {len(bundle.variants)} selectable tiers"
            if bundle.default_tiers:
                tiers += f" (default {'+'.join(str(t) for t in bundle.default_tiers)})"
        print(f"  {number:>2}. {bundle.title} [{bundle.key}] — {count} snapshot{suffix}{tiers}; inventory {bundle.inventory}")
    print("\n  all. Every bundle (the original downloader behavior)")
    print("  q.   Exit without downloading")


def _selection_tokens(values: Sequence[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


# --- Canonical store layout --------------------------------------------------
# Every model file lives directly under a models/<org>--<model> directory.
# Catalog destinations that predate this rule (project-named trees such as
# scenebloom/ or hynomas/) are migrated once by organize_store(): the bytes
# move into the canonical directory and each old runtime path keeps a relative
# symlink, so no project reference anywhere needs editing ("dummy folders").
# Hub cache roots (cache_layout snapshots) are exempt because their
# models--<org>--<model> trees already satisfy the rule.

def _repo_slug(repo_id: str) -> str:
    org, _, name = repo_id.partition("/")
    slug = repo_id if not name else f"{org}--{name}"
    return re.sub(r"[^\w.-]+", "-", slug).strip("-")


def _name_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def canonical_destination(artifact: Artifact) -> Path:
    """The <org>--<model> directory that must physically hold this artifact."""
    dest = Path(os.path.normpath(artifact.destination))
    try:
        relative = dest.relative_to(MODELS)
    except ValueError:
        return dest  # lane or home-path placement: it follows its owner
    if artifact.cache_layout or any("--" in part for part in relative.parts):
        return dest
    slug = _repo_slug(artifact.repo_id)
    model_name = artifact.repo_id.partition("/")[2] or artifact.repo_id
    if _name_key(relative.parts[-1]) == _name_key(model_name):
        return MODELS / slug
    # Keep the distinguishing tail of the legacy path under the canonical root
    # so different destinations of one repository never collide.
    tail = relative.parts[1:]
    return MODELS.joinpath(slug, *tail) if tail else MODELS / slug


def _defined_artifacts() -> tuple[Artifact, ...]:
    seen: dict[Artifact, None] = {}
    for bundle in BUNDLES:
        for arts in (bundle.artifacts, *(arts for _title, arts in bundle.variants)):
            for artifact in arts:
                seen.setdefault(artifact, None)
    return tuple(seen)


def canonical_artifact(artifact: Artifact) -> Artifact:
    dest = canonical_destination(artifact)
    return artifact if dest == artifact.destination else replace(artifact, destination=dest)


# canonical destination -> legacy runtime paths that must keep resolving
STORE_MIRRORS: dict[Path, list[Path]] = {}
# legacy destination -> artifacts claiming it (multi-repo trees split by allowlist)
LEGACY_OWNERS: dict[Path, list[Artifact]] = {}
for _artifact in _defined_artifacts():
    _canonical = canonical_destination(_artifact)
    _legacy = Path(os.path.normpath(_artifact.destination))
    if _canonical != _legacy:
        _mirrors = STORE_MIRRORS.setdefault(_canonical, [])
        if _legacy not in _mirrors:
            _mirrors.append(_legacy)
        _owners = LEGACY_OWNERS.setdefault(_legacy, [])
        if _artifact not in _owners:
            _owners.append(_artifact)

# Trees the picker writes itself (SceneBloom AmbientCG materials): canonical
# under their upstream source name; the legacy project paths keep a
# whole-directory symlink.
PICKER_MIGRATIONS = {
    MODELS / "scenebloom/materials": MODELS / "ambientcg--materials",
    MODELS / "scenebloom/material-archives": MODELS / "ambientcg--material-archives",
}


def _entry_claimant(legacy: Path, entry: Path) -> Artifact:
    """Which artifact's canonical directory a legacy top-level entry moves to."""
    artifacts = LEGACY_OWNERS[legacy]
    name = entry.name
    for artifact in artifacts:
        for pattern in artifact.allow_patterns:
            if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name, pattern.split("/")[0]):
                return artifact
    return artifacts[0]  # unmatched strays ride with the first definer


def _entry_size(entry: Path) -> int:
    if entry.is_symlink():
        return 0
    if entry.is_dir():
        return _tree_bytes(entry)
    try:
        return entry.stat().st_size
    except OSError:
        return 0


def _tree_bytes(path: Path) -> int:
    """Real bytes under path; compatibility symlinks never re-count the store."""
    total = 0
    for root, _dirs, files in os.walk(path):
        base = Path(root)
        for name in files:
            entry = base / name
            if not entry.is_symlink():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
    return total


def _organize_plan() -> list[tuple[Path, dict[Path, list[Path]]]]:
    """legacy dir -> {canonical dir: entries to move}, for trees on disk."""
    plans: list[tuple[Path, dict[Path, list[Path]]]] = []
    for legacy in LEGACY_OWNERS:
        if not legacy.is_dir() or legacy.is_symlink():
            continue
        entries = [
            entry for entry in sorted(legacy.iterdir(), key=lambda p: p.name)
            if not entry.is_symlink()  # existing links stay put and get inventoried
            and entry.name not in TRANSIENT_NAMES and entry.name not in STATE_NAMES
            and not (entry.is_dir() and (legacy / entry.name) in LEGACY_OWNERS)
            # nested legacy destinations migrate on their own (deepest first)
        ]
        if not entries:
            continue
        groups: dict[Path, list[Path]] = {}
        for entry in entries:
            groups.setdefault(canonical_destination(_entry_claimant(legacy, entry)), []).append(entry)
        plans.append((legacy, groups))
    for legacy, canonical in PICKER_MIGRATIONS.items():
        if legacy.is_dir() and not legacy.is_symlink() and any(legacy.iterdir()):
            plans.append((legacy, {canonical: [legacy]}))  # whole-directory move
    # Nested destinations must land in their own canonical roots before any
    # enclosing legacy tree tries to swallow them.
    plans.sort(key=lambda plan: (-len(plan[0].parts), str(plan[0])))
    return plans


def organize_store(assume_yes: bool) -> None:
    """Move legacy project-named trees into canonical <org>--<model> dirs."""
    plans = _organize_plan()
    if not plans:
        return
    print("\nLegacy store layout found; canonicalizing into <org>--<model> directories:")
    total = 0
    for legacy, groups in plans:
        for canonical, entries in groups.items():
            size = sum(_entry_size(entry) for entry in entries)
            total += size
            print(
                f"  {legacy.relative_to(MODELS)}/ -> {canonical.relative_to(MODELS)}/"
                f"  ({len(entries)} entr{ 'y' if len(entries) == 1 else 'ies'}, {size / 1e9:.2f} GB)"
            )
    print(
        f"\nMoving {total / 1e9:.2f} GB relocates nothing for the projects: each old path"
        " keeps a relative symlink into the canonical directory."
    )
    answer = "y" if assume_yes else _prompt_or_eof(
        "Migrate the store layout now? [Y/n]: ", default="y"
    )
    if answer not in {"", "y", "yes"}:
        print("Layout left as-is; hashes.txt flags the misplaced files until the next run.")
        return
    moved = 0
    for legacy, groups in plans:
        whole_dir = len(groups) == 1 and next(iter(groups.values()))[0] == legacy
        for canonical, entries in groups.items():
            if whole_dir:
                entry = entries[0]
                if not canonical.exists():
                    canonical.parent.mkdir(parents=True, exist_ok=True)
                    entry.rename(canonical)
                    moved += 1
                elif entry.is_dir():
                    for child in sorted(entry.iterdir(), key=lambda p: p.name):
                        if _move_into(child, canonical):
                            moved += 1
                if not entry.is_symlink():
                    try:
                        entry.rmdir()
                    except OSError:
                        pass  # kept strays keep the legacy folder real
                    if not entry.exists():
                        entry.symlink_to(os.path.relpath(canonical, entry.parent))
                continue
            canonical.mkdir(parents=True, exist_ok=True)
            for entry in entries:
                if _move_into(entry, canonical):
                    moved += 1
                link = legacy / entry.name
                if not link.is_symlink() and not link.exists():
                    link.symlink_to(os.path.relpath(canonical / entry.name, legacy))
    print(f"Migrated {moved} legacy trees into canonical directories.")


def _move_into(entry: Path, canonical: Path) -> bool:
    target = canonical / entry.name
    if target.exists():
        try:
            if entry.samefile(target):
                entry.unlink()
                return True
        except OSError:
            pass
        print(f"  warning: {target} already exists; kept {entry} in place", file=sys.stderr)
        return False
    entry.rename(target)
    return True


def _prompt_or_eof(text: str, default: str = "") -> str:
    try:
        return input(text).strip().lower() or default
    except EOFError:
        return "n"


# --- Stateful hash registry ---------------------------------------------------
# models/hashes.txt is the store's state file, in the header-block format of
# the reference hashes-recurse.py (FILE/SIZE/TIMESTAMP/SHA256/SHA1/MD5 blocks
# separated by ---).  Every untracked path under models/ is registered: FILE
# blocks carry the three digests, LINK blocks inventory symlinks with their
# target (contents are never double-hashed), DIR blocks inventory every
# directory (STATUS: empty when it holds nothing, so symlink-only and empty
# folders are on record too), TRANSIENT blocks stand in for .cache/__pycache__
# churn, and OTHER blocks inventory anything not hashable.  Repo-shipped
# files (download_models.py, manifests/, the docs, uv.toml, ...) are mm-tools
# repository state, not store content, and stay out of the registry entirely.
# SIZE+TIMESTAMP
# decide when a digest can be reused without rehashing.  WANT marks each entry
# 1 when this run's selection owns it and 0 when a previous run wanted it but
# this one does not; future runs may freely demote entries, and every WANT 0
# record is an item the end-of-run cleanup offers back for deletion.  The
# file is checked and updated at the start of every run, before the downloader
# touches the network, and refreshed again after a completed pass.

HASHES_NAME = "hashes.txt"
STATE_NAMES = {HASHES_NAME, HASHES_NAME + ".tmp", "store-ledger.tsv"}
# Repository state that ships inside mm-tools/models, plus small local config:
# never store content, so these stay out of the registry and out of cleanup.
REPO_PATHS = {
    "ARCHITECTURE.md", "download_models.py", "download_models_info.txt",
    "fab_assets.py", "studio-child.py", "studio-env.sh", "which-ones.txt",
    "manifests",
}
INFRA_PATHS = STATE_NAMES | {"__pycache__", "uv.toml"}


def _is_repo_path(relative: str) -> bool:
    head = relative.split("/", 1)[0]
    return head in REPO_PATHS or head in INFRA_PATHS
SKIP_TOP = {".venv"}
TRANSIENT_NAMES = {".cache", "__pycache__"}
HASH_CHUNK = 1024 * 1024  # 1 MB streaming chunks, as in hashes-recurse.py


def _registry_path() -> Path:
    return MODELS / HASHES_NAME


def _registry_load() -> dict[str, dict[str, str]]:
    table: dict[str, dict[str, str]] = {}
    registry = _registry_path()
    if not registry.is_file():
        return table
    block: dict[str, str] = {}
    for line in registry.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped == "---":
            for key in ("FILE", "LINK", "OTHER", "TRANSIENT", "DIR"):
                if key in block:
                    table[block[key].rstrip("/")] = block
                    break
            block = {}
        elif not stripped or stripped.startswith("#"):
            continue
        else:
            head, _, value = line.partition(":")
            block[head.strip()] = value.strip()
    return table


def _registry_dump(entries: dict[str, list[str]]) -> None:
    lines = [
        "# mm-tools model store hashes — one header block per entry, separated by ---",
        "# every untracked models/ path except .venv/ and repo-shipped files is listed:",
        "# FILE hashed (reference digest set), LINK/specials inventoried, DIR metadata",
        "# (STATUS: empty marks void folders); WANT 1 = selected by the latest run,",
        "# 0 = deselected and offered for cleanup; digests reused while SIZE+TIMESTAMP hold",
    ]
    for relative in sorted(entries):
        lines += entries[relative] + ["", "---", ""]
    temporary = MODELS / (HASHES_NAME + ".tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(_registry_path())


def _registry_entry_current(registry: dict[str, dict[str, str]], relative: str) -> dict[str, str] | None:
    """Registry block for a path, but only while disk still matches its SIZE/TIMESTAMP."""
    entry = registry.get(relative)
    if not entry or not entry.get("SHA256"):
        return None
    try:
        info = (MODELS / relative).stat()
    except OSError:
        return None
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
    if (entry.get("SIZE", "").split() != [str(info.st_size), "bytes"]
            or entry.get("TIMESTAMP") != stamp):
        return None
    return entry


def _registry_digest_skip(artifact: Artifact, registry: dict[str, dict[str, str]] | None) -> bool:
    """True when a pinned direct artifact is already in hashes.txt, unchanged."""
    if registry is None or not artifact.direct_filename or not artifact.sha256:
        return False
    target = Path(os.path.normpath(str(artifact.destination))) / artifact.direct_filename
    try:
        relative = target.relative_to(MODELS).as_posix()
    except ValueError:
        return False
    entry = _registry_entry_current(registry, relative)
    return bool(entry) and entry["SHA256"].lower() == artifact.sha256.lower()


def _registry_snapshot_files(
    artifact: Artifact, registry: dict[str, dict[str, str]] | None
) -> list[str] | None:
    """Files proven present for a snapshot artifact, or None to hit the network.

    The registry written by the pre-run hash pass is the completion proof: a
    previous finished pass left no `.cache` behind and every file under the
    canonical destination carries a fresh SIZE/TIMESTAMP-verified digest.
    Allowlists must be fully satisfied so shared destinations (several
    repositories in one legacy folder) never skip a missing member.
    """
    if registry is None:
        return None
    dest = Path(os.path.normpath(str(artifact.destination)))
    if not dest.is_dir() or (dest / ".cache").is_dir():
        return None  # fresh or interrupted: let the resumable client verify
    try:
        prefix = dest.relative_to(MODELS).as_posix() + "/"
    except ValueError:
        return None  # home-path destination: outside the registry's scope
    owned = [
        key for key in registry
        if key.startswith(prefix) and _registry_entry_current(registry, key)
    ]
    if artifact.cache_layout:
        # Cache roots hold several repositories side by side; prove this one
        # completed by requiring its snapshot files plus its written refs.
        slug = f"models--{artifact.repo_id.replace('/', '--')}"
        snapshots = [k for k in owned if f"/{slug}/snapshots/" in k]
        refs = [k for k in owned if f"/{slug}/refs/" in k]
        return snapshots + refs if snapshots and refs else None
    if not owned:
        return None
    inside = [key[len(prefix):] for key in owned]
    if not artifact.cache_layout:
        # A flat local_dir snapshot's own files never live under models--*
        # cache subdirectories; sibling cache trees cannot prove it.
        inside = [
            rel for rel in inside
            if not any(part.startswith("models--") for part in rel.split("/")[:-1])
        ]
        if not inside:
            return None
    for pattern in artifact.allow_patterns:
        leaf = pattern.split("/")[-1]
        if not any(
            fnmatch.fnmatch(rel, pattern)
            or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], leaf)
            or fnmatch.fnmatch(rel.rsplit("/", 1)[-1], pattern)
            for rel in inside
        ):
            return None  # an allowlisted member is not in the store yet
    return inside


def _hash_job(absolute: str) -> tuple[str, str, str, str, str, str, str | None]:
    """Stream one file once and produce the reference digest triple."""
    path = Path(absolute)
    try:
        info = path.stat()
        sha256, sha1, md5 = hashlib.sha256(), hashlib.sha1(), hashlib.md5()
        with open(path, "rb") as handle:
            while True:
                data = handle.read(HASH_CHUNK)
                if not data:
                    break
                sha256.update(data)
                sha1.update(data)
                md5.update(data)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
        return (absolute, str(info.st_size), stamp,
                sha256.hexdigest(), sha1.hexdigest(), md5.hexdigest(), None)
    except OSError as exc:
        return (absolute, "", "", "", "", "", f"{exc.__class__.__name__}: {exc}")


def _walk_store():
    """Yield (kind, Path, entry_count) without descending symlinks or churn."""
    stack: list[tuple[Path, int]] = [(MODELS, 0)]
    while stack:
        root, depth = stack.pop()
        try:
            children = sorted(os.scandir(root), key=lambda e: e.name)
        except OSError as exc:
            print(f"  warning: cannot read {root}: {exc}", file=sys.stderr)
            continue
        for child in children:
            path = Path(child.path)
            if child.is_symlink():
                yield "link", path, 0
            elif child.is_dir(follow_symlinks=False):
                if depth == 0 and child.name in SKIP_TOP:
                    continue
                if child.name in TRANSIENT_NAMES:
                    yield "transient", path, 0
                else:
                    try:
                        inner = sum(1 for _ in os.scandir(path))
                    except OSError:
                        inner = 0
                    yield "dir", path, inner
                    stack.append((path, depth + 1))
            elif child.is_file(follow_symlinks=False):
                yield "file", path, 0
            else:
                yield "special", path, 0


def _owner_names(relative: Path, claims: dict[str, set[Path]]) -> set[str]:
    parts = relative.parts
    names: set[str] = set()
    for name, roots in claims.items():
        for root in roots:
            if root == MODELS:
                continue
            try:
                root_rel = root.relative_to(MODELS)
            except ValueError:
                continue
            if tuple(root_rel.parts) == parts[: len(root_rel.parts)]:
                names.add(name)
                break
    return names


def sync_hashes(
    claims: dict[str, set[Path]],
    wanted: set[str],
    workers: int,
    *,
    assume_yes: bool = False,
) -> None:
    """Check models/hashes.txt, hash only new or changed files, update WANT."""
    registry = _registry_load()
    records: dict[str, list[str]] = {}
    pending: list[str] = []
    reused = hashed = links = specials = transients = 0
    dirs = empties = 0
    unowned = misplaced = 0
    total_bytes = 0
    for kind, path, inner in _walk_store():
        relative = path.relative_to(MODELS)
        key = relative.as_posix()
        if len(relative.parts) == 1 and relative.name in STATE_NAMES:
            continue
        if _is_repo_path(key):
            continue  # repo-shipped state, not store content
        names = _owner_names(relative, claims)
        want = "1" if names & wanted else "0"
        if kind == "transient":
            transients += 1
            records[key] = [f"TRANSIENT: {key}/", f"WANT:      {want}"]
        elif kind == "dir":
            dirs += 1
            record = [f"DIR:       {key}/", f"WANT:      {want}"]
            if inner == 0:
                empties += 1
                record.append("STATUS:    empty")
            else:
                record.append(f"ITEMS:     {inner}")
            records[key] = record
        elif kind == "link":
            links += 1
            try:
                target = os.readlink(path)
            except OSError as exc:
                target = f"(unreadable: {exc})"
            record = [f"LINK:      {key}", f"WANT:      {want}", f"TARGET:    {target}"]
            if not path.exists():
                record.append("STATUS:    broken")
            records[key] = record
        elif kind == "special":
            specials += 1
            mode = path.stat().st_mode
            kind_name = "fifo" if stat.S_ISFIFO(mode) else "socket" if stat.S_ISSOCK(mode) else "device"
            records[key] = [f"OTHER:     {key}", f"WANT:      {want}", f"TYPE:      {kind_name}"]
        else:
            try:
                info = path.stat()
            except OSError as exc:
                records[key] = [f"FILE:      {key}", f"WANT:      {want}",
                                f"ERROR:     {exc.__class__.__name__}: {exc}"]
                continue
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))
            known = registry.get(key)
            if (known and known.get("SIZE", "").split() == [str(info.st_size), "bytes"]
                    and known.get("TIMESTAMP") == stamp and known.get("SHA256")):
                reused += 1
                total_bytes += info.st_size
                records[key] = [
                    f"FILE:      {key}", f"WANT:      {want}",
                    f"SIZE:      {info.st_size} bytes", f"TIMESTAMP: {stamp}",
                    f"SHA256:    {known['SHA256']}", f"SHA1:      {known.get('SHA1', '?')}",
                    f"MD5:       {known.get('MD5', '?')}",
                ]
                if not names:
                    unowned += 1
                    records[key].append("UNOWNED:   claimed by no bundle definition")
                if not any("--" in part for part in relative.parts[:-1]):
                    misplaced += 1
                    records[key].append("MISPLACED: outside an <org>--<model> directory")
                continue
            pending.append(str(path))
            records[key] = [f"FILE:      {key}", f"WANT:      {want}"]  # filled below
    if pending and sys.stdin.isatty() and not assume_yes and len(pending) > 500:
        answer = _prompt_or_eof(
            f"\n{len(pending)} files are new or changed since the last run;"
            " hash them now? [Y/n]: ", default="y",
        )
        if answer not in {"", "y", "yes"}:
            pending = []
    digests: dict[str, tuple[str, str, str, str, str, str | None]] = {}
    if pending:
        workers = max(1, min(workers, len(pending)))
        pool = get_context("fork").Pool(workers)
        try:
            for index, result in enumerate(pool.imap_unordered(_hash_job, pending), 1):
                digests[result[0]] = result[1:]
                hashed += 1
                print(f"{100 * index / len(pending):6.2f}%  ({index}/{len(pending)})", end="\r", flush=True)
        finally:
            pool.close()
            pool.join()
        print()
    for key, record in records.items():
        if (record[0].startswith("FILE")
                and not any(line.startswith(("SHA256", "ERROR")) for line in record)):
            absolute = str(MODELS / key)
            result = digests.get(absolute)
            if result is None:
                record.append("SIZE:      ?")
                record.append("ERROR:     not hashed this pass")
                continue
            size, stamp, sha256, sha1, md5, error = result
            if error:
                record.append(f"ERROR:     {error}")
                continue
            total_bytes += int(size)
            relative_parts = Path(key).parts
            if not _owner_names(Path(key), claims):
                unowned += 1
                record.append("UNOWNED:   claimed by no bundle definition")
            if not any("--" in part for part in relative_parts[:-1]):
                misplaced += 1
                record.append("MISPLACED: outside an <org>--<model> directory")
            record.extend([
                f"SIZE:      {size} bytes", f"TIMESTAMP: {stamp}",
                f"SHA256:    {sha256}", f"SHA1:      {sha1}", f"MD5:       {md5}",
            ])
            continue
        # reused record: flags already appended during the walk
    stale = sorted(set(registry) - set(records))
    _registry_dump(records)
    files = sum(1 for record in records.values() if record[0].startswith("FILE"))
    print(f"\nStore registry ({HASHES_NAME}, paths relative to models/):")
    print(
        f"  Total: {total_bytes / 1e9:.2f} GB across {files} files;"
        f" {dirs} directories ({empties} empty), {links} links,"
        f" {specials} specials, {transients} transient trees inventoried"
    )
    print(
        f"  hashed this pass: {hashed}; reused unchanged digests: {reused};"
        f" dropped stale entries: {len(stale)}"
    )
    if unowned or misplaced:
        print(f"  unowned entries: {unowned}; outside <org>--<model> directories: {misplaced}")


# --- Selection state ---------------------------------------------------------
# The registry's WANT lines are the selection state: every completed pass
# rewrites them from the current selection (1) over previously wanted paths
# (0).  Nothing is deleted without an explicit prompt at the end of a pass.
# Helper trees the picker writes itself are claimed by their project bundle.


def _selection_names(
    bundles: Sequence[Bundle], picks: dict[str, tuple[int, ...]]
) -> set[str]:
    names: set[str] = set()
    for bundle in bundles:
        if not bundle.variants:
            names.add(bundle.key)
            continue
        chosen = (picks.get(bundle.key) or bundle.default_tiers
                  or tuple(range(1, len(bundle.variants) + 1)))
        names.update(f"{bundle.key}:{index}" for index in chosen)
    return names


def _definition_claims() -> dict[str, set[Path]]:
    """Selectable name -> every store path that name's definitions own.

    A name owns the canonical <org>--<model> directory of each of its
    artifacts, the legacy runtime paths kept as compatibility mirrors, and the
    declared placement/link targets that live inside the model store.  Lane
    and home-path placements follow their project and never enter claims.
    """
    claims: dict[str, set[Path]] = {}
    for bundle in BUNDLES:
        if not bundle.variants:
            groups = {bundle.key: bundle.artifacts}
        else:
            groups = {
                f"{bundle.key}:{index}": bundle.artifacts + tier_artifacts
                for index, (_title, tier_artifacts) in enumerate(bundle.variants, start=1)
            }
        for name, artifacts in groups.items():
            paths = claims.setdefault(name, set())
            for artifact in artifacts:
                canonical = canonical_destination(artifact)
                paths.add(canonical)
                paths.update(STORE_MIRRORS.get(canonical, ()))
                raw: list[Path] = list(artifact.dir_links)
                raw += [target for _source, target in artifact.file_links]
                raw += [target for _source, target in artifact.placements]
                paths.update(
                    normalized
                    for normalized in (Path(os.path.normpath(path)) for path in raw)
                    if normalized.is_relative_to(MODELS)
                )
    for name, roots in claims.items():
        if name.startswith("scenebloom"):
            for legacy, canonical in PICKER_MIGRATIONS.items():
                roots.update((legacy, canonical))
            roots.add(MODELS / "scenebloom/assets-ready.json")
    return claims


def _tidy_destination(artifact: Artifact) -> None:
    """After one artifact completes, drop resumable bookkeeping and flatten
    Hub cache link trees so the store keeps plain, loadable files only."""
    if not artifact.destination.is_dir():
        return
    shutil.rmtree(artifact.destination / ".cache", ignore_errors=True)
    if not artifact.cache_layout:
        return
    for model_dir in artifact.destination.glob("models--*"):
        snapshots = model_dir / "snapshots"
        if not snapshots.is_dir():
            continue
        for link in list(snapshots.rglob("*")):
            if not link.is_symlink():
                continue
            try:
                target = link.resolve()
                link.unlink()
                os.link(target, link)
            except OSError:
                pass
        # refs/ stays: Transformers still resolves branch names through it.
        shutil.rmtree(model_dir / "blobs", ignore_errors=True)


def _remove_paths(paths: set[Path]) -> None:
    freed = 0
    count = 0
    for path in sorted(paths):
        try:
            size = path.stat().st_size if path.is_file() else _tree_bytes(path)
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            freed += size
            count += 1
        except OSError as exc:
            print(f"  failed to delete {path}: {exc}", file=sys.stderr)
    print(f"Deleted {count} entries, freed {freed / 1e9:.2f} GB.")


def _offer_removal(items: dict[str, tuple[Path, int]], title: str) -> None:
    """List removal candidates and always prompt before deleting them."""
    if not items:
        print(f"\n{title}\n  none")
        return
    print(f"\n{title}")
    for _label, (path, size) in sorted(
        items.items(), key=lambda item: (-item[1][1], str(item[1][0]))
    ):
        print(f"  {size / 1e9:9.2f} GB  {path.relative_to(MODELS)}"
              f"{'' if path.is_file() else '/'}  [{_label}]")
    total = sum(size for _label, (_path, size) in items.items())
    print(f"\nTotal: {total / 1e9:.2f} GB across {len(items)} entries.")
    try:
        answer = input("Delete them now? [y/n/e per entry]: ").strip().lower()
    except EOFError:
        answer = "n"
    if answer in {"y", "yes", "a", "all"}:
        _remove_paths({path for _label, (path, _size) in items.items()})
    elif answer in {"e", "each"}:
        for _label, (path, size) in items.items():
            try:
                verdict = input(
                    f"Delete {path.relative_to(MODELS)} ({size / 1e9:.2f} GB)? [y/N]: "
                ).strip().lower()
            except EOFError:
                verdict = ""
            if verdict in {"y", "yes"}:
                _remove_paths({path})
    else:
        print("Left in place.")


def _wanted_names(bundles: Sequence[Bundle], picks: dict[str, tuple[int, ...]]) -> set[str]:
    """Exactly the bundle/tier names this run selected.  Plain 'all' plus
    per-tier answers still deselects every tier the operator did not pick."""
    return _selection_names(bundles, picks)


def post_download_maintenance(claims: dict[str, set[Path]]) -> None:
    """Offer to delete every store item the selection did not want.

    Reads the registry the finished hash update wrote: each FILE, LINK, or
    DIR record marked WANT 0 (a deselected name owns it, or nothing owns it)
    is a candidate, and transient churn (.cache, __pycache__) is offered
    unconditionally because it regenerates.  Wholly deselected directories
    fold into one entry — a retired tree is a single prompt — while strays,
    empty folders, and symlink-only folders inside still-wanted trees stay
    itemized.  Repo-shipped files, .venv, and the registry state never
    appear here; nothing is deleted without the explicit prompt.
    """
    registry = _registry_load()
    candidates: set[str] = set()
    for rel, entry in registry.items():
        if "TRANSIENT" in entry:
            candidates.add(rel)  # churn clears safely and regenerates
        elif entry.get("WANT", "0") == "0":
            candidates.add(rel)

    def foldable(rel: str) -> bool:
        try:
            entries = list(os.scandir(MODELS / rel))
        except OSError:
            return False
        prefix = rel + "/"
        for child in entries:
            if child.name in TRANSIENT_NAMES:
                continue
            sub = prefix + child.name
            recorded = [k for k in registry if k == sub or k.startswith(sub + "/")]
            if not recorded or not all(k in candidates for k in recorded):
                return False
        return True

    items: dict[str, tuple[Path, int]] = {}
    covered: set[str] = set()
    for rel in sorted(candidates, key=lambda r: (r.count("/"), r)):
        if any(rel.startswith(p + "/") for p in covered):
            continue  # a folded ancestor entry already covers this subtree
        path = MODELS / rel
        churn = "TRANSIENT" in registry.get(rel, {})
        if path.is_symlink():
            size = 0
        elif path.is_dir():
            # Churn trees hold unregistered partials by design; always clearable.
            if not churn and not foldable(rel):
                continue  # mixed folder: the wanted children stay, files itemize below
            size = _tree_bytes(path)
        elif path.is_file():
            size = path.stat().st_size
        else:
            continue
        owners = sorted(_owner_names(Path(rel), claims))
        label = "churn (regenerates)" if churn else (", ".join(owners) if owners else "unowned")
        key, n = label, 2
        while key in items:
            key, n = f"{label} #{n}", n + 1
        items[key] = (path, size)
        if path.is_dir() and not path.is_symlink():
            covered.add(rel)
    _offer_removal(items, "Unselected store items (WANT 0 / unowned / churn)")


def resolve_selection(
    values: Sequence[str],
) -> tuple[tuple[Bundle, ...], dict[str, tuple[int, ...]]]:
    """Map selection tokens to bundles plus per-bundle 1-based tier picks.

    Tokens accept an optional `:1+3` tier suffix (commas separate bundles,
    pluses separate tiers); a bundle with tiers and no suffix selects every
    tier. Returns (bundles, picks) where a missing pick entry means "every
    tier".
    """
    tokens = _selection_tokens(values)
    if not tokens:
        raise ValueError("No model bundle was selected")
    if any(token.lower() in {"all", "a", "*"} for token in tokens):
        return BUNDLES, {}

    by_key = {bundle.key: bundle for bundle in BUNDLES}
    # Preserve the former command-line name while presenting the canonical
    # project name in the menu and documentation.
    by_key["kijai"] = by_key["v2v"]
    selected: list[Bundle] = []
    picks: dict[str, tuple[int, ...]] = {}
    seen: set[str] = set()
    for token in tokens:
        head, _, tier_text = token.partition(":")
        lowered = head.lower()
        if lowered in {"q", "quit", "exit"}:
            return (), {}
        if head.isdecimal() and 1 <= int(head) <= len(BUNDLES):
            bundle = BUNDLES[int(head) - 1]
        elif lowered in by_key:
            bundle = by_key[lowered]
        else:
            raise ValueError(f"Unknown selection: {token!r}")
        if tier_text and not bundle.variants:
            raise ValueError(f"{bundle.key} has no selectable tiers")
        if tier_text:
            numbers: list[int] = []
            for part in (p.strip() for p in tier_text.replace(",", "+").split("+")):
                if not part:
                    continue
                if part.lower() in {"a", "all", "*"}:
                    numbers = list(range(1, len(bundle.variants) + 1))
                    break
                if not part.isdecimal() or not 1 <= int(part) <= len(bundle.variants):
                    raise ValueError(
                        f"Unknown tier for {bundle.key}: {part!r} "
                        f"(1-{len(bundle.variants)} or a)"
                    )
                numbers.append(int(part))
            if not numbers:
                raise ValueError(f"No tiers selected for {bundle.key}")
            picks[bundle.key] = tuple(dict.fromkeys(numbers))
        if bundle.key not in seen:
            selected.append(bundle)
            seen.add(bundle.key)
    return tuple(selected), picks


def bundle_artifacts(
    bundle: Bundle, picks: Sequence[int] | None
) -> tuple[Artifact, ...]:
    """Shared artifacts plus the chosen tiers (default tiers when picks is empty)."""
    if not bundle.variants:
        return bundle.artifacts
    if not picks:
        picks = bundle.default_tiers or tuple(range(1, len(bundle.variants) + 1))
    return bundle.artifacts + tuple(
        artifact for index in picks for artifact in bundle.variants[index - 1][1]
    )


def _prompt_tiers(bundle: Bundle) -> tuple[int, ...]:
    print(f"\n{bundle.title} is selected.")
    if bundle.prompt_note:
        print(f"  {bundle.prompt_note}")
    marks = set(bundle.default_tiers or ())
    for index, (title, _) in enumerate(bundle.variants, start=1):
        print(f"  {index}. {title}{' *' if index in marks else ''}")
    print("  a. every tier")
    if marks:
        print("  * recommended default for this machine.")
    default = (
        ",".join(str(tier) for tier in bundle.default_tiers)
        if bundle.default_tiers else "a"
    )
    while True:
        try:
            raw = input(f"Model tier(s), comma-separated [{default}]: ").strip() or default
            _, picks = resolve_selection((f"{bundle.key}:{raw.replace(',', '+')}",))
            return picks[bundle.key]
        except (ValueError, KeyError) as exc:
            print(f"Error: {exc}", file=sys.stderr)


def interactive_selection() -> tuple[tuple[Bundle, ...], dict[str, tuple[int, ...]]]:
    print_menu()
    while True:
        try:
            raw = input("\nSelect numbers or names, comma-separated [all]: ").strip() or "all"
            bundles, picks = resolve_selection((raw,))
            break
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
    # Every selected bundle with model tiers is asked which tiers it actually
    # wants, including bare 'all'; only a ':N' suffix on the selection line
    # skips the question.
    for bundle in bundles:
        if bundle.variants and bundle.key not in picks:
            picks[bundle.key] = _prompt_tiers(bundle)
    return bundles, picks


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
    registry: dict[str, dict[str, str]] | None = None,
) -> None:
    # snapshot_download normally creates local_dir/cache_dir, but doing it here
    # guarantees that fresh nested roots such as qwen/ and text-only/ exist.
    artifact.destination.mkdir(parents=True, exist_ok=True)
    if artifact.direct_url:
        if _registry_digest_skip(artifact, registry):
            print(f"\n[{number}/{total}] {artifact.repo_id}")
            print(f"  ready: {artifact.destination / artifact.direct_filename}"
                  " (hashes.txt digest)")
        else:
            download_direct_artifact(artifact, number, total)
        materialize_placements(artifact)
        materialize_dir_links(artifact)
        materialize_file_links(artifact)
        materialize_store_mirrors(artifact.destination)
        return
    proven = _registry_snapshot_files(artifact, registry)
    if proven is not None:
        print(f"\n[{number}/{total}] {artifact.repo_id}")
        print(f"  ready: {len(proven)} files verified through hashes.txt"
              " (delete them or the registry to refetch)")
        materialize_placements(artifact)
        materialize_dir_links(artifact)
        materialize_file_links(artifact)
        materialize_store_mirrors(artifact.destination)
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
    if artifact.repo_type != "model":
        kwargs["repo_type"] = artifact.repo_type
    # Older Hub releases need this to materialize local files. Current Hub
    # removed the argument and already uses real files for local_dir snapshots.
    if SNAPSHOT_SUPPORTS_SYMLINK_FLAG:
        kwargs["local_dir_use_symlinks"] = False
    if artifact.allow_patterns:
        kwargs["allow_patterns"] = list(artifact.allow_patterns)
    snapshot_download(**kwargs)
    materialize_placements(artifact)
    materialize_dir_links(artifact)
    materialize_file_links(artifact)
    materialize_store_mirrors(artifact.destination)


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
    if artifact.direct_url.startswith("https://modelscope.cn/"):
        print("  ModelScope direct download; SHA-256 verified on completion.")
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

    Hugging Face's resumable local snapshot stays intact in the canonical
    <org>--<model> directory.  The runtime path becomes a relative symlink to
    those verified bytes (a "dummy folder" reference, never a second set of
    bytes); the copy fallback is only for an unusual cross-filesystem
    destination.
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
            target.symlink_to(os.path.relpath(source, target.parent))
            method = "symlink"
        except OSError:
            shutil.copy2(source, target)
            method = "copy"
        print(f"  placed ({method}): {target}")


def materialize_dir_links(artifact: Artifact) -> None:
    """Point a runtime folder at the snapshot root with one relative symlink.

    A runtime that expects a whole model directory gets the canonical snapshot
    under models/ linked in place; the folder is created where the runtime
    looks for it rather than copying bytes or patching the runtime.
    """
    for link in artifact.dir_links:
        target = artifact.destination
        if link.is_symlink():
            current = (link.parent / os.readlink(link)).resolve()
            if current == target.resolve():
                print(f"  ready: {link}")
            else:
                print(
                    f"  warning: {link} already links elsewhere; left alone",
                    file=sys.stderr,
                )
            continue
        if link.exists():
            print(f"  ready: {link} (existing directory)")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        relative = os.path.relpath(target, link.parent)
        link.symlink_to(relative)
        print(f"  linked: {link} -> {relative}")


def materialize_file_links(artifact: Artifact) -> None:
    """Symlink snapshot files or directories onto traditional runtime paths.

    The canonical bytes stay under models/; the runtime path is a relative
    symlink, which is cheap enough that mixed directories (several
    repositories sharing one runtime weights folder) compose naturally.
    """
    for relative_source, target in artifact.file_links:
        source = artifact.destination / relative_source
        if not source.exists():
            raise FileNotFoundError(
                f"Downloaded snapshot did not contain link source: {source}"
            )
        if target.is_symlink():
            current = (target.parent / os.readlink(target)).resolve()
            if current == source.resolve():
                print(f"  ready: {target}")
            else:
                print(
                    f"  warning: {target} already links elsewhere; left alone",
                    file=sys.stderr,
                )
            continue
        if target.exists():
            print(f"  ready: {target} (existing file)")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(os.path.relpath(source, target.parent))
        print(f"  linked: {target}")


def materialize_store_mirrors(canonical: Path) -> None:
    """Keep every legacy runtime path of a canonical directory resolvable.

    After organize_store() the bytes live only in the canonical directory; a
    fresh clone has never seen the legacy name.  Either way the legacy path
    ends up as a relative symlink: one directory link when the whole tree is
    wanted, per-entry links where several repositories share the legacy folder.
    """
    for legacy in STORE_MIRRORS.get(canonical, ()):
        if legacy.is_symlink():
            continue
        if not legacy.exists():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.symlink_to(os.path.relpath(canonical, legacy.parent))
            print(f"  linked: {legacy} -> {os.path.relpath(canonical, legacy.parent)}")
            continue
        for entry in sorted(canonical.iterdir(), key=lambda p: p.name):
            if entry.name in TRANSIENT_NAMES or entry.name in STATE_NAMES:
                continue
            link = legacy / entry.name
            if link.is_symlink():
                continue
            if link.exists():
                print(
                    f"  warning: {link} is a second copy of the store; "
                    "hashes.txt flags it until removed",
                    file=sys.stderr,
                )
                continue
            link.symlink_to(os.path.relpath(entry, legacy))
        # legacy path exists as a real directory; entries link in place above


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download all mm-tools model artifacts or selected project bundles.",
        epilog=(
            "Examples: uv run download_models.py | uv run download_models.py all | "
            "uv run download_models.py 2,6,10 | uv run download_models.py minimax stableaudio | "
            "uv run download_models.py lingbot:2 | uv run download_models.py fastframes:2. "
            "Every run first migrates the store into canonical <org>--<model> directories "
            "and checks/updates hashes.txt (stateful WANT registry, reference digest set); "
            "artifacts already verified in hashes.txt skip the network entirely; "
            "after a completed pass it offers to remove deselected or unclaimed paths "
            "under models/ (always y/n first)."
        ),
    )
    parser.add_argument(
        "selection", nargs="*",
        help="bundle numbers, bundle names, 'name:tier[+tier]', or 'all'",
    )
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
        bundles, picks = BUNDLES, {}
    elif args.selection:
        try:
            bundles, picks = resolve_selection(args.selection)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    elif sys.stdin.isatty():
        bundles, picks = interactive_selection()
    else:
        print("No selection supplied in a non-interactive shell; pass 'all' or bundle numbers.", file=sys.stderr)
        return 2

    if not bundles:
        print("No downloads selected.")
        return 0

    ledger = MODELS / "store-ledger.tsv"
    if ledger.exists():
        ledger.unlink()
        print("store-ledger.tsv retired: selection state now lives in the WANT lines of hashes.txt.")

    # The first thing a run does: canonicalize the layout, then check/update
    # hashes.txt against the selection (WANT 0/1).  Only the download loop
    # below touches the network, and only after the Continue prompt.
    organize_store(args.yes)
    claims = _definition_claims()
    sync_hashes(claims, _wanted_names(bundles, picks), workers, assume_yes=args.yes)
    # The registry doubles as the download skip-list: artifacts whose pinned
    # digests or complete snapshot files already sit in hashes.txt need no
    # network pass at all.
    registry = _registry_load()

    artifacts = tuple(dict.fromkeys(
        canonical_artifact(artifact)
        for bundle in bundles
        for artifact in bundle_artifacts(bundle, picks.get(bundle.key))
    ))
    if any(b.key == "fab" for b in bundles):
        from fab_assets import find_smpl
        installed = find_smpl()
        if installed:
            print(f"FAB: using installed neutral body at {installed}")
        else:
            # This mirror has the exact checksum published in FAB's reproduction guide.
            rev = "01543c84f2af7a1a3763cbc96c8f90950563c04b"
            artifacts += (A("vsadhu1/MotionGPT-smpl-models", SOMA_X.destination / "SMPL", revision=rev,
                            direct_url=f"https://huggingface.co/vsadhu1/MotionGPT-smpl-models/resolve/{rev}/smpl/SMPL_NEUTRAL.pkl",
                            direct_filename="SMPL_NEUTRAL.pkl", expected_size=39001280,
                            sha256="98e65c74ad9b998783132f00880d1025a8d64b158e040e6ef13a557e5098bc42"),)
    print("\nSelected bundles:")
    for bundle in bundles:
        chosen = picks.get(bundle.key) or tuple(range(1, len(bundle.variants) + 1))
        suffix = (
            " · " + "; ".join(bundle.variants[index - 1][0] for index in chosen)
            if bundle.variants else ""
        )
        count = len(bundle_artifacts(bundle, picks.get(bundle.key)))
        plural = "s" if count != 1 else ""
        print(f"  - {bundle.title} ({count} snapshot{plural}){suffix}")
    print(f"Total snapshots: {len(artifacts)}")
    print(f"Parallel workers: {workers}")

    if not args.yes and sys.stdin.isatty():
        answer = input("Continue? [Y/n]: ").strip().lower()
        if answer not in {"", "y", "yes"}:
            print("Cancelled.")
            return 0

    # Do not start any download until the user has accepted it.  Renaming a
    # complete local_dir (organize_store) preserves Hugging Face's resumable
    # metadata and any interrupted chunks beneath that directory.
    migrate_legacy_layout(bundles)

    if args.debug:
        logging.set_verbosity_debug()
    else:
        logging.set_verbosity_info()
    for number, artifact in enumerate(artifacts, start=1):
        download_artifact(artifact, number, len(artifacts), workers, registry)
        _tidy_destination(artifact)

    if any(bundle.key == "scenebloom" for bundle in bundles):
        prepare_scenebloom_assets(MODELS)

    print(f"\nCompleted {len(artifacts)} snapshot downloads.")
    # The completed pass first updates hashes.txt (prompted once for anything
    # new), then offers the destructive cleanup: WANT 0 trees and unclaimed
    # folders, i.e. everything models/ holds outside this run's selection.
    sync_hashes(claims, _wanted_names(bundles, picks), workers, assume_yes=args.yes)
    post_download_maintenance(claims)
    return 0

# SceneBloom ships two large published archives plus AmbientCG material packs; the
# picker verifies and unpacks them, and serving stays offline afterwards.
def asset_digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def asset_unpack(archive, destination, checksum):
    receipt = destination / ".complete.json"
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved.get("sha256") == checksum and all(
            (destination / name).is_file()
            and (destination / name).stat().st_size == size
            for name, size in saved["files"].items()
        ):
            return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".unpack-") as work:
        stage = Path(work)
        files = {}
        with zipfile.ZipFile(archive) as source:
            for item in source.infolist():
                target = stage / item.filename
                if (
                    not target.resolve().is_relative_to(stage.resolve())
                    or "\\" in item.filename
                ):
                    raise ValueError(f"Unsafe archive member: {item.filename}")
                if stat.S_ISLNK(item.external_attr >> 16):
                    raise ValueError("Archive links are unsupported")
                if item.is_dir():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                files[item.filename] = item.file_size
        destination.mkdir(exist_ok=True)
        for name in files:
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError("Existing extraction link escapes asset directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / name, target)
        temporary = stage / "receipt.json"
        temporary.write_text(json.dumps(dict(sha256=checksum, files=files)))
        temporary.replace(receipt)


def prepare_scenebloom_assets(models: Path) -> None:
    manifest = json.loads((models / "manifests/scenebloom.json").read_text())
    root = models / "scenebloom"
    # AmbientCG trees are canonicalized under their source name; the project
    # paths below stay resolvable as symlink mirrors (organize_store has moved
    # any existing content into the canonical directories by now).
    material_dirs = {
        "materials": models / "ambientcg--materials",
        "material-archives": models / "ambientcg--material-archives",
    }
    for legacy_name, target in material_dirs.items():
        legacy = root / legacy_name
        if not legacy.is_symlink() and not legacy.exists():
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.symlink_to(os.path.relpath(target, legacy.parent))
    for entry in manifest["archives"]:
        archive = root / "m3dlayout" / entry["path"]
        checksum = asset_digest(archive)
        if checksum != entry["sha256"]:
            raise ValueError(f"Checksum mismatch: {archive}")
        asset_unpack(archive, root / "m3dlayout/extracted" / entry["dest"], checksum)
    cache = material_dirs["material-archives"]
    cache.mkdir(exist_ok=True)
    for material in manifest["materials"]:
        archive = cache / f"{material}_2K-JPG.zip"
        receipt = archive.with_suffix(".json")
        # AmbientCG publishes versioned filenames, not cryptographic digests.
        # Record our completed transfer's checksum; incomplete files never count.
        valid = archive.is_file() and receipt.is_file()
        if valid:
            valid = asset_digest(archive) == json.loads(receipt.read_text())["sha256"]
        if not valid:
            url = f"https://ambientcg.com/get?file={archive.name}"
            request = urllib.request.Request(
                url, headers={"User-Agent": "mm-tools SceneBloom asset picker"}
            )
            temporary = archive.with_suffix(".partial")
            try:
                with (
                    urllib.request.urlopen(request, timeout=120) as src,
                    temporary.open("wb") as dst,
                ):
                    shutil.copyfileobj(src, dst)
                with zipfile.ZipFile(temporary) as source:
                    if source.testzip() is not None:
                        raise ValueError(f"Damaged material archive: {material}")
                temporary.replace(archive)
                receipt.write_text(json.dumps(dict(url=url, sha256=asset_digest(archive))))
            finally:
                temporary.unlink(missing_ok=True)
        asset_unpack(archive, material_dirs["materials"] / material, asset_digest(archive))
        if not (
            material_dirs["materials"] / material / f"{material}_2K-JPG_Color.jpg"
        ).is_file():
            raise ValueError(f"Missing material colour map: {material}")
    # Grounding-DINO-lineage code asks for the legacy bare "bert-base-uncased" id, but only
    # the canonical google-bert/ id resolves xet download tokens. Link the legacy cache
    # directory name onto the downloaded canonical snapshot so offline lookups hit it.
    hf_cache = root / "checkpoints/huggingface"
    canonical = hf_cache / "models--google-bert--bert-base-uncased"
    legacy = hf_cache / "models--bert-base-uncased"
    if not canonical.is_dir():
        raise ValueError(f"Missing canonical bert cache: {canonical}")
    if not legacy.exists():
        legacy.symlink_to(canonical.name)
    (root / "assets-ready.json").write_text(json.dumps(dict(source=manifest["source"])))


if __name__ == "__main__":
    raise SystemExit(main())
