import os
from pathlib import Path

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["TOKIO_WORKER_THREADS"] = "16"

from huggingface_hub import logging, snapshot_download

logging.set_verbosity_debug()

MODELS = Path(__file__).resolve().parent
ROOT = MODELS.parent

snapshot_download(
    repo_id="ideogram-ai/ideogram-4-fp8",
    local_dir=MODELS / "ideogram-ai--ideogram-4-fp8",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="jixin0101/ObjectClear",
    local_dir=MODELS / "jixin0101--ObjectClear",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="ZhengPeng7/BiRefNet",
    local_dir=MODELS / "ZhengPeng7--BiRefNet",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="meituan-longcat/LongCat-AudioDiT-3.5B",
    local_dir=MODELS / "meituan-longcat--AudioDiT-3.5B-tts-text-to-speech-SOTA",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="google/umt5-base",
    local_dir=MODELS / "google--umt5-base-tokenizer",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
)
snapshot_download(
    repo_id="ACE-Step/Ace-Step1.5",
    local_dir=MODELS / "Ace-Step--Ace-Step1.5",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "Qwen3-Embedding-0.6B/*",
        "acestep-5Hz-lm-1.7B/*",
        "acestep-v15-turbo/*",
        "vae/*",
    ],
)
snapshot_download(
    repo_id="RoyalCities/Foundation-1",
    local_dir=MODELS / "RoyalCities--Foundation-1",
    local_dir_use_symlinks=False,
    allow_patterns=["Foundation_1.safetensors", "model_config.json"],
)
snapshot_download(
    repo_id="google-t5/t5-base",
    local_dir=MODELS / "google-t5--t5-base",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "spiece.model",
        "tokenizer.json",
    ],
)
snapshot_download(
    repo_id="SymphonyGen/SymphonyGen",
    local_dir=MODELS / "SymphonyGen--SymphonyGen",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "stage_one_pretrained.pt",
        "stage_two_pretrained.pt",
        "grpo_clamp_epoch_10.pt",
        "grpo_clamp+track_epoch_6.pt",
    ],
)
snapshot_download(
    repo_id="pymaster/VocalRender",
    local_dir=MODELS / "pymaster--VocalRender",
    local_dir_use_symlinks=False,
    allow_patterns=["VocalRender/*", "VocalRender-Pro/*"],
)
snapshot_download(
    repo_id="MuScriptor/muscriptor-large",
    local_dir=MODELS / "MuScriptor--muscriptor-large",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="MuScriptor/assets",
    local_dir=MODELS / "MuScriptor--assets",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="PRAIG/musvit",
    local_dir=MODELS / "PRAIG--musvit",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="PRAIG/smt-fp-grandstaff",
    local_dir=MODELS / "PRAIG--smt-fp-grandstaff",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="nyralabs/CrisperWhisper2.0_large",
    local_dir=MODELS / "nyralabs--CrisperWhisper2.0_large",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="mradermacher/EraX-Translator-V1.0-GGUF",
    local_dir=MODELS / "text-only/anhbn--raX-Translator-V1.0-GGUF",
    local_dir_use_symlinks=False,
    allow_patterns=["EraX-Translator-V1.0.Q6_K.gguf"],
)
snapshot_download(
    repo_id="papluca/xlm-roberta-base-language-detection",
    local_dir=MODELS / "text-only/papluca--xlm-roberta-base-language-detection",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="anhbn/EraX-VL-7B-V1.5-Openvino-INT4",
    local_dir=MODELS / "text-only/anhbn--EraX-VL-7B-V1.5-Openvino-INT4",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="T5B/Qwen-Image-Layered-FP8",
    local_dir=MODELS / "qwen/T5B--qwen-image-layered-fp8",
    local_dir_use_symlinks=False,
    allow_patterns=["qwen_image_layered_fp8_e4m3fn.safetensors"],
)
snapshot_download(
    repo_id="diffusers/Qwen-Image-Layered-modular",
    local_dir=MODELS / "qwen/diffusers--hfstaff--Qwen-Image-Layered-modular",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="suzukimain/Qwen-Image-Layered-Control-SDNQ-int4",
    local_dir=MODELS / "qwen/suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="appmana/qwen-image-layered-int8convrot",
    local_dir=MODELS / "qwen/appmana--diffusion--qwen-image-layered-int8convrot",
    local_dir_use_symlinks=False,
    allow_patterns=["qwen_image_layered_int8convrot.safetensors"],
)
snapshot_download(
    repo_id="Comfy-Org/HunyuanVideo_1.5_repackaged",
    local_dir=MODELS / "qwen/comfy-org--text--qwen_2.5_vl_7b_fp8_scaled.safetensors",
    local_dir_use_symlinks=False,
    allow_patterns=["split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"],
)
snapshot_download(
    repo_id="Comfy-Org/Qwen-Image-Layered_ComfyUI",
    local_dir=MODELS / "qwen/comfy-org--vae--qwen_image_layered_vae.safetensors",
    local_dir_use_symlinks=False,
    allow_patterns=["split_files/vae/qwen_image_layered_vae.safetensors"],
)
snapshot_download(
    repo_id="Kijai/Wan_ID_V2V_comfy",
    local_dir=MODELS / "kijai/Kijai--Wan_ID_V2V_comfy",
    local_dir_use_symlinks=False,
    allow_patterns=["wan_2.1_idv2v_int8_convrot.safetensors"],
)
snapshot_download(
    repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
    local_dir=MODELS / "kijai/Comfy-Org--Wan_2.1_ComfyUI_repackaged",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "split_files/clip_vision/clip_vision_h.safetensors",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "split_files/vae/wan_2.1_vae.safetensors",
    ],
)
snapshot_download(
    repo_id="denisbalon/lightx2v-i2v-14b-480p-cfg-step-distill-rank64-bf16.safetensors",
    local_dir=MODELS / "kijai/denisbalon--lightx2v-i2v-14b-480p-cfg-step-distill-rank64-bf16",
    local_dir_use_symlinks=False,
    allow_patterns=["lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"],
)
snapshot_download(
    repo_id="Kijai/WanVideo_comfy",
    local_dir=MODELS / "kijai/Kijai--WanVideo_comfy",
    local_dir_use_symlinks=False,
    allow_patterns=["Wan2_1_VAE_bf16.safetensors"],
)
snapshot_download(
    repo_id="benjiaiplayground/GroundingDINO_SwinB",
    local_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
    allow_patterns=["groundingdino_swinb_cogcoor.pth"],
)
snapshot_download(
    repo_id="facebook/sam2.1-hiera-large",
    local_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
    allow_patterns=["sam2.1_hiera_large.pt"],
)
snapshot_download(
    repo_id="GoGiants1/Hi-SAM",
    local_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
    allow_patterns=["sam_tss_h_textseg.pth", "sam_vit_h_4b8939.pth"],
)
snapshot_download(
    repo_id="iimate/big-lama-pt",
    local_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
    allow_patterns=["big-lama.pt"],
)
snapshot_download(
    repo_id="jixin0101/ObjectClear",
    cache_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="google-bert/bert-base-uncased",
    cache_dir=ROOT / "redesign/weights",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="PaddlePaddle/PP-OCRv5_server_det",
    local_dir=Path.home() / ".paddlex/official_models/PP-OCRv5_server_det",
    local_dir_use_symlinks=False,
)
snapshot_download(
    repo_id="PaddlePaddle/PP-OCRv5_server_rec",
    local_dir=Path.home() / ".paddlex/official_models/PP-OCRv5_server_rec",
    local_dir_use_symlinks=False,
)
