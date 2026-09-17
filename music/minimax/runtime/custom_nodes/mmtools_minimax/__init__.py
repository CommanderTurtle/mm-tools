from __future__ import annotations

import comfy.model_management


class MMToolsMiniMaxPreload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "preload"
    CATEGORY = "mm-tools/minimax"
    OUTPUT_NODE = True

    def preload(self, model, clip, vae):
        patchers = [model]
        for value in (clip, vae):
            patcher = getattr(value, "patcher", None)
            if patcher is not None:
                patchers.append(patcher)
        comfy.model_management.load_models_gpu(patchers)
        return {"ui": {"text": ["MiniMax Music 3 models are resident"]}, "result": ()}


NODE_CLASS_MAPPINGS = {"MMToolsMiniMaxPreload": MMToolsMiniMaxPreload}
NODE_DISPLAY_NAME_MAPPINGS = {"MMToolsMiniMaxPreload": "MM-Tools MiniMax Preload"}
