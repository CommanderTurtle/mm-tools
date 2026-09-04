"""Small native-Comfy masked Ideogram graph; no hosted partner nodes."""
from __future__ import annotations


def caption_graph(image: str, model: str, instruction: str, schema: str) -> dict:
    prompt = (schema + "\nDescribe the supplied image AFTER the requested edit. Preserve its camera, "
              "lighting and all unedited objects. Omit objects requested for removal; describe the "
              "background that should fill their location. Return only caption JSON, no explanation.\n"
              "Requested edit: " + instruction)
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": image}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": model, "type": "krea2", "device": "default"}},
        "3": {"class_type": "TextGenerate", "inputs": {"clip": ["2", 0], "image": ["1", 0], "prompt": prompt,
                "max_length": 2048, "sampling_mode": "off", "thinking": False, "use_default_template": True}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["3", 0]}},
    }


def inpaint_graph(image: str, mask: str, caption: str, width: int, height: int,
                  steps: int, seed: int, guidance: float, strength: float) -> dict:
    def node(kind, **inputs):
        return {"class_type": kind, "inputs": inputs}
    return {
        "1": node("LoadImage", image=image),
        "2": node("LoadImage", image=mask),
        "3": node("ImageToMask", image=["2", 0], channel="red"),
        "4": node("MMToolsIdeogramLocal", component="conditional"),
        "5": node("MMToolsIdeogramLocal", component="unconditional"),
        "6": node("MMToolsIdeogramText"),
        "7": node("MMToolsIdeogramVAE"),
        "8": node("CLIPTextEncode", text=caption, clip=["6", 0]),
        "9": node("VAEEncodeTiled", pixels=["1", 0], vae=["7", 0], tile_size=512, overlap=64,
                  temporal_size=64, temporal_overlap=8),
        "10": node("SetLatentNoiseMask", samples=["9", 0], mask=["3", 0]),
        "11": node("DifferentialDiffusion", model=["4", 0], strength=1.0),
        "12": node("DualModelGuider", model=["11", 0], model_negative=["5", 0], positive=["8", 0], cfg=guidance),
        "13": node("Ideogram4Scheduler", steps=steps, width=width, height=height, mu=0.0, std=1.75),
        "14": node("SplitSigmasDenoise", sigmas=["13", 0], denoise=strength),
        "15": node("RandomNoise", noise_seed=seed),
        "16": node("KSamplerSelect", sampler_name="euler"),
        "17": node("SamplerCustomAdvanced", noise=["15", 0], guider=["12", 0], sampler=["16", 0],
                   sigmas=["14", 1], latent_image=["10", 0]),
        "18": node("VAEDecodeTiled", samples=["17", 0], vae=["7", 0], tile_size=512, overlap=64,
                   temporal_size=64, temporal_overlap=8),
        "19": node("SaveImage", images=["18", 0], filename_prefix="ideogram-edit"),
    }
