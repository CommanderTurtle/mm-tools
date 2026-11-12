#!/usr/bin/env python3
# Copyright (c) Ant Group. All rights reserved.
"""Convert a deployment ``talker`` checkpoint into a self-contained release.

The deployment format splits the model across ``talker/``, ``talker/vae/`` and
``talker/llm/`` with its own key names. The released open-source format is a
single ``BailingMMNativeForConditionalGeneration`` directory. This is the
inverse of the deployment-side ``convert_talker_keys.py``:

    talker/model.safetensors
        cfm.*           -> flowloss.cfm.*
        aggregator.*    -> linear_proj_audio.*
        model.*         -> model.model.*
        spk_head.*      -> spk_head.*         (unchanged)
        stop_head.*     -> stop_head.*        (unchanged)
    talker/vae/model.safetensors
        encoder.*       -> audio.encoder.*
        decoder.*       -> audio.decoder.*

``model.lm_head.weight`` is intentionally absent: the LLM ties its output
projection to ``embed_tokens``, so transformers recreates it on load.

The output directory is self-contained — weights, config, tokenizer and the
speaker encoder — so that inference needs no second download.

Usage:
    python tools/convert_deploy_to_opensource.py \
        --talker_dir /path/to/deploy/talker \
        --tokenizer_dir /path/to/dense_tokenizer \
        --output_dir ./Ming-3.0-tts-0.5b
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ordered longest-prefix-first; ``model.`` is the catch-all and must come last
# so it does not swallow the module-specific prefixes.
TALKER_PREFIX_MAP = (
    ("cfm.", "flowloss.cfm."),
    ("aggregator.", "linear_proj_audio."),
    ("spk_head.", "spk_head."),
    ("stop_head.", "stop_head."),
    ("model.", "model.model."),
)
VAE_PREFIX_MAP = (
    ("encoder.", "audio.encoder."),
    ("decoder.", "audio.decoder."),
)

# Files the runtime needs next to the weights.
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)


def remap(state_dict, prefix_map, *, source):
    """Rename keys by longest matching prefix, refusing to drop anything."""
    renamed, unmapped = {}, []
    for key, tensor in state_dict.items():
        for old, new in prefix_map:
            if key.startswith(old):
                target = new + key[len(old):]
                if target in renamed:
                    raise ValueError(
                        f"{source}: {key!r} collides with an earlier key at "
                        f"{target!r}")
                renamed[target] = tensor
                break
        else:
            unmapped.append(key)
    if unmapped:
        raise ValueError(
            f"{source}: {len(unmapped)} key(s) match no known prefix, so the "
            f"converted checkpoint would be incomplete: {unmapped[:8]}")
    return renamed


# Keys that pin an attention backend. The deployment configs hard-code
# ``flash_attention_2``, which turns an optional speedup into a hard
# requirement: transformers raises ImportError at construction time if
# flash-attn is absent, so the checkpoint would not load at all on a machine
# without it. Dropping the pin lets transformers pick (sdpa when available)
# and still lets a caller ask for flash explicitly via
# ``from_pretrained(..., attn_implementation="flash_attention_2")``.
#
# This is safe for the audio VAE even though it is trained with a 64-token
# sliding window: transformers warns that sliding-window attention is "not
# implemented" outside flash_attention_2, but that warning only inspects the
# config flag. The window is in fact applied through the 4D attention mask, so
# sdpa and eager honour it.
ATTENTION_PIN_KEYS = ("_attn_implementation", "attn_implementation")


def drop_attention_pins(node):
    """Recursively remove hard attention-backend pins from a config tree."""
    if isinstance(node, dict):
        return {
            key: drop_attention_pins(value)
            for key, value in node.items()
            if key not in ATTENTION_PIN_KEYS
        }
    if isinstance(node, list):
        return [drop_attention_pins(value) for value in node]
    return node


def build_config(talker_config, llm_config, vae_config):
    """Reassemble the open-source config from the three deployment configs."""
    llm_config = drop_attention_pins(llm_config)
    vae_config = drop_attention_pins(vae_config)
    flowmodel = dict(talker_config.get("flowmodel", {}))
    ditar = {
        "attn_backend": flowmodel.get("attn_backend", "torch"),
        "attn_mask_enabled": flowmodel.get("attn_mask_enabled", False),
        "depth": flowmodel["depth"],
        "dropout": flowmodel.get("dropout", 0.0),
        "hidden_size": flowmodel["hidden_size"],
        "history_patch_size": talker_config["history_patch_size"],
        "mlp_ratio": flowmodel.get("mlp_ratio", 4),
        "num_heads": flowmodel["num_heads"],
        "patch_size": talker_config["patch_size"],
    }
    aggregator = dict(talker_config.get("aggregator", {}))
    return {
        "aggregator_config": {
            "attn_mask_enabled": aggregator.get("attn_mask_enabled", False),
            "depth": aggregator["depth"],
            "dropout": aggregator.get("dropout", 0.0),
            "hidden_size": aggregator["hidden_size"],
            "mlp_ratio": aggregator.get("mlp_ratio", 4),
            "num_heads": aggregator["num_heads"],
        },
        "architectures": ["BailingMMNativeForConditionalGeneration"],
        "audio_tokenizer_config": vae_config,
        "auto_map": {"AutoConfig": "configuration_bailingmm.BailingMMConfig"},
        "ditar_config": ditar,
        "llm_config": llm_config,
        "mlp_depth": talker_config.get("mlp_depth", 2),
        "model_type": "dense",
        "torch_dtype": "bfloat16",
        "transformers_version": "4.52.4",
    }


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def copy_tokenizer(tokenizer_dir, output_dir):
    """Copy the tokenizer, and fail if the essential pieces are missing.

    A checkpoint whose tokenizer has to be fetched from somewhere else is
    exactly the failure this conversion exists to prevent.
    """
    copied = []
    for name in TOKENIZER_FILES:
        source = os.path.join(tokenizer_dir, name)
        if os.path.exists(source):
            shutil.copy2(source, os.path.join(output_dir, name))
            copied.append(name)
    if "tokenizer.json" not in copied and "vocab.json" not in copied:
        raise FileNotFoundError(
            f"{tokenizer_dir} holds neither tokenizer.json nor vocab.json; "
            "the release would not be self-contained")
    return copied


def verify_state_dict(config, converted_keys):
    """Hold the converted keys against what the model actually asks for.

    Builds the model on the meta device, so this costs no memory and needs no
    weights. A prefix that silently maps to the wrong name would otherwise only
    surface as a wall of missing-key warnings at first load, or worse, as a
    randomly initialised submodule that still produces audio.
    """
    import torch

    from configuration_bailingmm import BailingMMConfig
    from modeling_bailingmm import BailingMMNativeForConditionalGeneration

    skip = ("architectures", "auto_map", "transformers_version", "torch_dtype")
    with torch.device("meta"):
        model = BailingMMNativeForConditionalGeneration(
            BailingMMConfig(**{k: v for k, v in config.items() if k not in skip})
        )

    # Tied weights are rebuilt from the embedding table on load, so the
    # checkpoint is expected not to carry them.
    tied = set(getattr(model, "_tied_weights_keys", None) or ())
    wanted = set(model.state_dict()) - tied
    got = set(converted_keys) - tied

    if wanted != got:
        missing, extra = sorted(wanted - got), sorted(got - wanted)
        raise ValueError(
            f"converted checkpoint does not match the model: "
            f"{len(missing)} missing, {len(extra)} unexpected.\n"
            f"  missing: {missing[:8]}\n  unexpected: {extra[:8]}"
        )
    return len(wanted), sorted(tied)


def verify_tokenizer(output_dir):
    """Load the copied tokenizer and hold it to the trained token contract."""
    from transformers import AutoTokenizer

    import prompt_contract

    tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    prompt_contract.validate_tokenizer(tokenizer, "dense")
    return tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--talker_dir", required=True,
                        help="deployment talker/ directory")
    parser.add_argument("--tokenizer_dir",
                        help="dense tokenizer directory; defaults to "
                             "<talker_dir>/llm")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--skip_verify", action="store_true",
                        help="skip loading the tokenizer after conversion")
    return parser.parse_args()


def main():
    from safetensors.torch import load_file, save_file

    args = parse_args()
    talker_dir = args.talker_dir
    vae_dir = os.path.join(talker_dir, "vae")
    llm_dir = os.path.join(talker_dir, "llm")
    tokenizer_dir = args.tokenizer_dir or llm_dir

    talker_weights = load_file(os.path.join(talker_dir, "model.safetensors"))
    vae_weights = load_file(os.path.join(vae_dir, "model.safetensors"))
    print(f"loaded talker={len(talker_weights)} keys vae={len(vae_weights)} keys")

    state_dict = remap(talker_weights, TALKER_PREFIX_MAP, source="talker")
    state_dict.update(remap(vae_weights, VAE_PREFIX_MAP, source="vae"))
    print(f"converted to {len(state_dict)} keys")

    os.makedirs(args.output_dir, exist_ok=True)
    save_file(
        state_dict,
        os.path.join(args.output_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )

    config = build_config(
        read_json(os.path.join(talker_dir, "config.json")),
        read_json(os.path.join(llm_dir, "config.json")),
        read_json(os.path.join(vae_dir, "config.json")),
    )
    with open(os.path.join(args.output_dir, "config.json"), "w") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    copied = copy_tokenizer(tokenizer_dir, args.output_dir)
    print(f"copied tokenizer files: {', '.join(copied)}")

    campplus = os.path.join(talker_dir, "campplus.onnx")
    if os.path.exists(campplus):
        shutil.copy2(campplus, os.path.join(args.output_dir, "campplus.onnx"))
        print("copied campplus.onnx")
    else:
        print("WARNING: campplus.onnx not found; speaker embedding will fail")

    with open(os.path.join(args.output_dir, "configuration.json"), "w") as handle:
        json.dump({"framework": "Pytorch", "task": "text-to-speech"}, handle)
        handle.write("\n")

    if not args.skip_verify:
        matched, tied = verify_state_dict(config, state_dict)
        print(f"weights match the model: {matched} tensors"
              + (f" (+{len(tied)} rebuilt from tied embeddings: "
                 f"{', '.join(tied)})" if tied else ""))
        verify_tokenizer(args.output_dir)
        print("tokenizer matches the Ming-3.0-tts dense contract")

    print(f"self-contained release written to {args.output_dir}")


if __name__ == "__main__":
    main()
