import base64
import io
import json
import sys

from paths import MODELS, parent_guard


def main():
    parent_guard()
    request = json.load(sys.stdin)
    # Keep the model's native prompt/vision evidence; only adapt OpenAI image
    # parts to the processor's input representation. Never retrieve remote URLs.
    with __import__("contextlib").redirect_stdout(sys.stderr):
        import torch
        from PIL import Image
        from transformers import (
            AutoProcessor,
            Qwen3_5ForConditionalGeneration,
            set_seed,
        )

        messages = []
        for message in request["messages"]:
            content = message["content"]
            if isinstance(content, list):
                parts = []
                for part in content:
                    if part.get("type") == "image_url":
                        data = part["image_url"]["url"]
                        if not data.startswith("data:image/") or ";base64," not in data:
                            raise ValueError("PE accepts local image evidence only")
                        image = Image.open(
                            io.BytesIO(
                                base64.b64decode(data.split(",", 1)[1], validate=True)
                            )
                        )
                        parts.append({"type": "image", "image": image.convert("RGB")})
                    elif part.get("type") == "text":
                        parts.append(part)
                    else:
                        raise ValueError("Unsupported PE content part")
                content = parts
            messages.append({"role": message["role"], "content": content})
        if request.get("seed") is not None:
            set_seed(int(request["seed"]))
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            MODELS / "PE",
            dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True,
            attn_implementation="sdpa",
        ).eval()
        processor = AutoProcessor.from_pretrained(MODELS / "PE", local_files_only=True)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=4096)
        text = processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
    print(json.dumps({"content": text}))


if __name__ == "__main__":
    main()
