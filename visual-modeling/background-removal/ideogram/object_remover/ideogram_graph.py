"""Small native-Comfy masked Ideogram graph; no hosted partner nodes."""
from __future__ import annotations

# Pinned Comfy-Org/Ideogram-4 single-file set (models/which-ones.txt inventory 1).
# Kijai's optimized path: int8_convrot main + nvfp4 unconditional, stock loaders.
MAIN_MODEL = "ideogram4_int8_convrot.safetensors"
UNCOND_MODEL = "ideogram4_unconditional_nvfp4_mixed.safetensors"
CLIP_MODEL = "qwen3vl_8b_fp8_scaled.safetensors"
VAE_MODEL = "flux2-vae.safetensors"
# "CFG override start from 0.7": fixed tail cfg over the last 30% of the sigma range.
CFG_TAIL, CFG_TAIL_START = 1.0, .7


def caption_graph(image: str, model: str, instruction: str, schema: str, seed: int | None = None) -> dict:
    prompt = (schema + "\nDescribe the supplied image AFTER the requested edit. Preserve its camera, "
              "lighting and all unedited objects. Omit objects requested for removal; describe the "
              "background that should fill their location. Return only caption JSON, no explanation.\n"
              "Requested edit: " + instruction)
    graph = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": model, "type": "krea2", "device": "default"}},
        "3": {"class_type": "TextGenerate", "inputs": {"clip": ["2", 0], "image": ["1", 0], "prompt": prompt,
                "max_length": 2048, "sampling_mode": "off", "thinking": False, "use_default_template": True}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
    }
    if seed is not None:
        graph["3"]["inputs"].update({"sampling_mode": "on", "sampling_mode.seed": seed,
            "sampling_mode.temperature": .25, "sampling_mode.top_k": 40, "sampling_mode.top_p": .9,
            "sampling_mode.min_p": .05, "sampling_mode.repetition_penalty": 1.05})
    return graph


def inpaint_graph(image: str, mask: str, caption: str, width: int, height: int,
                  steps: int, seed: int, guidance: float, strength: float) -> dict:
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "1": node("LoadImage", image=image),
        "2": node("LoadImage", image=mask),
        "3": node("ImageToMask", image=["2", 0], channel="red"),
        "4": node("UNETLoader", unet_name=MAIN_MODEL, weight_dtype="default"),
        "5": node("UNETLoader", unet_name=UNCOND_MODEL, weight_dtype="default"),
        "6": node("CLIPLoader", clip_name=CLIP_MODEL, type="ideogram4", device="default"),
        "7": node("VAELoader", vae_name=VAE_MODEL),
        "8": node("CLIPTextEncode", text=caption, clip=["6", 0]),
        "9": node("VAEEncodeTiled", pixels=["1", 0], vae=["7", 0], tile_size=512, overlap=64,
                  temporal_size=64, temporal_overlap=8),
        "10": node("SetLatentNoiseMask", samples=["9", 0], mask=["3", 0]),
        "11": node("DifferentialDiffusion", model=["4", 0], strength=1.0),
        "12": node("CFGOverride", model=["11", 0], cfg=CFG_TAIL,
                   start_percent=CFG_TAIL_START, end_percent=1.0),
        "13": node("DualModelGuider", model=["12", 0], model_negative=["5", 0], positive=["8", 0], cfg=guidance),
        "14": node("Ideogram4Scheduler", steps=steps, width=width, height=height, mu=0.0, std=1.75),
        "15": node("SplitSigmasDenoise", sigmas=["14", 0], denoise=strength),
        "16": node("RandomNoise", noise_seed=seed),
        "17": node("KSamplerSelect", sampler_name="euler"),
        "18": node("SamplerCustomAdvanced", noise=["16", 0], guider=["13", 0], sampler=["17", 0],
                   sigmas=["15", 1], latent_image=["10", 0]),
        "19": node("VAEDecodeTiled", samples=["18", 0], vae=["7", 0], tile_size=512, overlap=64,
                   temporal_size=64, temporal_overlap=8),
        "20": node("SaveImage", images=["19", 0], filename_prefix="ideogram-edit"),
    }
