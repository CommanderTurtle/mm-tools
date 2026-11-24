# Copyright 2026 Adventists AI. Apache-2.0.
"""Convert thinker LoRA adapters between the vLLM layout (what Sophon-Realtime serves) and ms-swift / PEFT.

Both use the same tensor names (`base_model.model.language_model.model.layers.N...`). What differs is
`target_modules`: vLLM wants plain module names, while PEFT on the full model must not touch the audio
encoder (its attention also has q_proj / k_proj / v_proj), so the PEFT copy uses a regex limited to the LLM.

    python -m sophon_swift.adapters to-swift  Sophon-Realtime-v1/adapters/tag  tag_swift
    python -m sophon_swift.adapters to-vllm   output/vx-xxx/checkpoint-yyy      my_lora_vllm
"""
import argparse
import dataclasses
import json
import os
import shutil

LLM_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
LLM_REGEX = r"^language_model\.model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$"
PREFIX = "base_model.model.language_model.model.layers."


def _check_weights(path):
    from safetensors import safe_open
    with safe_open(path, "pt") as f:
        keys = list(f.keys())
    bad = [k for k in keys if not k.startswith(PREFIX)]
    if bad:
        raise ValueError(f"{len(bad)} tensors outside the LLM (e.g. {bad[0]}); only LLM LoRA is supported")
    return len(keys)


def _known_lora_fields():
    try:
        from peft import LoraConfig
        return {f.name for f in dataclasses.fields(LoraConfig)}
    except Exception:
        return None


def _write(src, dst, cfg):
    os.makedirs(dst, exist_ok=True)
    n = _check_weights(os.path.join(src, "adapter_model.safetensors"))
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copy2(os.path.join(src, "adapter_model.safetensors"), os.path.join(dst, "adapter_model.safetensors"))
    json.dump(cfg, open(os.path.join(dst, "adapter_config.json"), "w"), indent=2)
    return n


def to_swift(src, dst):
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    known = _known_lora_fields()
    if known:
        cfg = {k: v for k, v in cfg.items() if k in known or k in ("peft_type", "auto_mapping")}
    cfg.update(target_modules=LLM_REGEX, base_model_name_or_path="Sophon-Realtime-v1", inference_mode=False,
               auto_mapping=None, modules_to_save=None)
    n = _write(src, dst, cfg)
    print(f"ms-swift adapter: {dst} ({n} tensors)")


def to_vllm(src, dst):
    cfg = json.load(open(os.path.join(src, "adapter_config.json")))
    if cfg.get("modules_to_save"):
        raise ValueError("modules_to_save is not supported by the vLLM thinker; train LoRA only")
    cfg.update(target_modules=LLM_TARGETS, base_model_name_or_path="Sophon-Realtime-v1", inference_mode=True)
    n = _write(src, dst, cfg)
    for f in ("additional_config.json", "args.json"):
        if os.path.exists(os.path.join(src, f)) and os.path.abspath(src) != os.path.abspath(dst):
            shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
    print(f"vLLM adapter: {dst} ({n} tensors, r={cfg.get('r')})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("direction", choices=["to-swift", "to-vllm"])
    ap.add_argument("src")
    ap.add_argument("dst")
    a = ap.parse_args()
    (to_swift if a.direction == "to-swift" else to_vllm)(a.src, a.dst)


if __name__ == "__main__":
    main()
