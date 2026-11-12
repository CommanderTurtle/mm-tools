"""Ephemeral shared NVFP4 vision encoder, using the committed Comfy runtime."""

import base64
import io
import json
from pathlib import Path
import sys
from lifecycle import own, release
from state import REPO, VLM, atomic


def main(request, response):
    sys.argv = [sys.argv[0]]
    sys.path.insert(0, str(REPO / "workflows/ComfyUI"))
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.sd
    import numpy as np
    import torch
    from PIL import Image, ImageOps

    data = json.loads(request.read_text())
    images, turns = [], []
    for message in data["messages"]:
        content = message["content"]
        parts = (
            [{"type": "text", "text": content}] if isinstance(content, str) else content
        )
        text = ""
        for part in parts:
            if part["type"] == "text":
                text += part["text"]
            elif part["type"] == "image_url":
                url = part["image_url"]["url"]
                if not url.startswith("data:image/") or ";base64," not in url:
                    raise ValueError(
                        "Native vision messages must contain inline images"
                    )
                with Image.open(
                    io.BytesIO(base64.b64decode(url.split(",", 1)[1], validate=True))
                ) as source:
                    array = (
                        np.asarray(
                            ImageOps.exif_transpose(source).convert("RGB"),
                            dtype=np.float32,
                        )
                        / 255
                    )
                images.append(torch.from_numpy(array.copy()).unsqueeze(0))
                text += "<|vision_start|><|image_pad|><|vision_end|>"
            else:
                raise ValueError(f"Unsupported vision content: {part['type']}")
        turns.append(f"<|im_start|>{message['role']}\n{text}<|im_end|>\n")
    prompt = "".join(turns) + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    clip = comfy.sd.load_clip([str(VLM)])
    tokens = clip.tokenize(
        prompt, images=images, skip_template=True, thinking=False, min_length=1
    )
    temperature = float(data["temperature"])
    with torch.inference_mode():
        result = clip.generate(
            tokens,
            do_sample=temperature > 0,
            max_length=int(data["max_tokens"]),
            temperature=max(temperature, 0.0001),
            top_k=0,
            top_p=1.0,
        )
    atomic(response, {"text": clip.decode(result)})


if __name__ == "__main__":
    request, response, parent = sys.argv[1:]
    own(int(parent))
    try:
        main(Path(request), Path(response))
    finally:
        release()
