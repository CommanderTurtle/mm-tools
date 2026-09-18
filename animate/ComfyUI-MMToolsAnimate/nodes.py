from __future__ import annotations

import gc
from typing import Any

import torch

import comfy.model_management as model_management
import comfy.sd
import folder_paths


def _discard_encoder(value: Any) -> None:
    """Release a consumed encoder without moving any model to system RAM."""

    patcher = getattr(value, "patcher", None)
    if patcher is not None:
        model_management.unload_model_and_clones(patcher, all_devices=True)

    model = getattr(value, "cond_stage_model", None)
    if model is None:
        model = getattr(value, "model", None)
    if model is not None:
        model.to(torch.device("meta"))

    if patcher is not None:
        patcher.load_device = torch.device("meta")
        patcher.offload_device = torch.device("meta")


class MMToolsGpuOnlyWanLoader:
    """Load WAN only after text and vision embeddings have been materialized."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "positive_pose": ("CONDITIONING",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "clip_vision_output_pose": ("CLIP_VISION_OUTPUT",),
                "text_encoder": ("CLIP",),
                "vision_encoder": ("CLIP_VISION",),
            }
        }

    RETURN_TYPES = (
        "MODEL",
        "CONDITIONING",
        "CONDITIONING",
        "CONDITIONING",
        "CLIP_VISION_OUTPUT",
        "CLIP_VISION_OUTPUT",
    )
    RETURN_NAMES = (
        "model",
        "positive",
        "negative",
        "positive_pose",
        "clip_vision_output",
        "clip_vision_output_pose",
    )
    FUNCTION = "load_after_encoders"
    CATEGORY = "mm-tools/animate"

    def load_after_encoders(
        self,
        unet_name,
        positive,
        negative,
        positive_pose,
        clip_vision_output,
        clip_vision_output_pose,
        text_encoder,
        vision_encoder,
    ):
        _discard_encoder(text_encoder)
        _discard_encoder(vision_encoder)
        gc.collect()
        model_management.soft_empty_cache(force=True)

        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        model = comfy.sd.load_diffusion_model(unet_path, model_options={})
        return (
            model,
            positive,
            negative,
            positive_pose,
            clip_vision_output,
            clip_vision_output_pose,
        )


NODE_CLASS_MAPPINGS = {"MMToolsGpuOnlyWanLoader": MMToolsGpuOnlyWanLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"MMToolsGpuOnlyWanLoader": "WAN Loader · GPU-only staged"}
