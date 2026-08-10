from __future__ import annotations

import argparse
import json
import os
import socket
from ipaddress import ip_address
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig, PRESETS
from ideogram4.caption_verifier import CaptionVerifier
from ideogram4.magic_prompt import aspect_ratio_from_size, build_messages, strip_aspect_ratio_and_bboxes


MODEL_ENV = {
    "fp8": "IDEOGRAM4_FP8_MODEL",
    "nf4": "IDEOGRAM4_NF4_MODEL",
}


def local_expand(prompt: str, width: int, height: int) -> str:
    base = os.getenv("IDEOGRAM_LOCAL_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.getenv("IDEOGRAM_LOCAL_MODEL", "").strip()
    host = urlparse(base).hostname
    if not host or not model:
        raise ValueError("IDEOGRAM_LOCAL_BASE_URL and IDEOGRAM_LOCAL_MODEL are required for --expand.")
    try:
        addresses = {host} if ip_address(host) else set()
    except ValueError:
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
        except OSError as exc:
            raise ValueError(f"Local VLM hostname does not resolve: {host}") from exc
    if not addresses or not all(ip_address(item).is_private or ip_address(item).is_loopback for item in addresses):
        raise ValueError("IDEOGRAM_LOCAL_BASE_URL must resolve only to private/loopback addresses.")
    content = ""
    for _ in range(2):
        response = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": "Bearer local-vllm"},
            json={
                "model": model,
                "messages": build_messages("v1.txt", prompt, aspect_ratio_from_size(width, height)),
                "temperature": float(os.getenv("IDEOGRAM_LOCAL_TEMPERATURE", "0.2")),
                "max_tokens": 8192,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=float(os.getenv("IDEOGRAM_LOCAL_TIMEOUT", "240")),
        )
        response.raise_for_status()
        value = response.json()["choices"][0]["message"].get("content")
        if isinstance(value, str) and value.strip():
            content = value.strip()
            break
    if not content:
        raise ValueError("The local VLM returned no structured Ideogram caption after two attempts.")
    if content.startswith("```"):
        lines = content.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        content = "\n".join(lines).strip()
    return strip_aspect_ratio_and_bboxes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entirely local Ideogram 4 generation")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="Structured JSON, or plain text with --expand")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 structured caption file")
    parser.add_argument("--expand", action="store_true", help="Expand plain text with the private local VLM")
    parser.add_argument(
        "--quantization",
        choices=sorted(MODEL_ENV),
        default=os.getenv("IDEOGRAM4_QUANTIZATION", "fp8").strip().lower(),
        help="Local checkpoint format (default: IDEOGRAM4_QUANTIZATION or fp8)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="Override the local checkpoint selected by the quantization-specific environment variable",
    )
    parser.add_argument("--output", type=Path, default=Path("output.png"))
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--preset", choices=sorted(PRESETS), default="V4_DEFAULT_20")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_value = args.model or os.getenv(MODEL_ENV[args.quantization], "").strip()
    if not model_value:
        raise SystemExit(f"{MODEL_ENV[args.quantization]} is unset and --model was not supplied.")
    model = Path(model_value).expanduser().resolve()
    if not (model / "model_index.json").is_file():
        raise SystemExit(f"Incomplete local Ideogram checkpoint: {model}")
    if args.width % 16 or args.height % 16 or not (256 <= args.width <= 2048 and 256 <= args.height <= 2048):
        raise SystemExit("Width and height must be 256–2048 and divisible by 16.")
    caption = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else args.prompt
    if args.expand:
        caption = local_expand(caption, args.width, args.height)
    issues = CaptionVerifier().verify_raw(caption)
    if issues:
        raise SystemExit("Caption validation failed:\n" + "\n".join(issues))

    pipeline = Ideogram4Pipeline.from_pretrained(
        config=Ideogram4PipelineConfig(weights_repo=str(model)),
        device="cuda",
        dtype=torch.bfloat16,
    )
    preset = PRESETS[args.preset]
    image = pipeline(
        caption,
        width=args.width,
        height=args.height,
        num_steps=preset.num_steps,
        guidance_schedule=preset.guidance_schedule,
        mu=preset.mu,
        std=preset.std,
        seed=args.seed,
    )[0]
    args.output.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    image.save(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "model": str(model),
                "quantization": args.quantization,
                "width": args.width,
                "height": args.height,
                "seed": args.seed,
            }
        )
    )


if __name__ == "__main__":
    main()
