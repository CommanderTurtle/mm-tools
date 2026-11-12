import argparse
import gc
import json
import sys
import tempfile
from pathlib import Path

from paths import ADAPTERS, MODELS, NATIVE, WORK, model_identity, parent_guard

sys.path.insert(0, str(NATIVE))


def main():
    parent_guard()
    parser = argparse.ArgumentParser()
    parser.add_argument("adapter", choices=ADAPTERS[1:])
    args = parser.parse_args()
    import torch
    from apps.seg_ar_demo_config import load_demo_config
    from hymm.models.dit.modules import load_model
    from scripts.quantization.convert_dit_w8_bf16 import convert_checkpoint
    from scripts.train.merge_lora import merge

    destination = WORK / "adapters" / args.adapter
    destination.mkdir(parents=True, exist_ok=True)
    receipt = destination / "ready.json"
    if receipt.is_file():
        record = json.loads(receipt.read_text())
        if record == model_identity(args.adapter):
            return
        raise ValueError("Cached adapter uses a different source pin")
    if (destination / "int8").exists():
        raise ValueError(
            "An incomplete adapter cache exists; preserve/move it before preparing again"
        )
    config = load_demo_config(
        [
            "+experiment=single_gpu_low_mem_inference",
            "parallelism.tp_size=1",
            "inference.dit_weight_quantization=none",
        ]
    )
    with tempfile.TemporaryDirectory(prefix="merge-", dir=destination) as staging:
        merged = Path(staging) / "merged.safetensors"
        merge(
            MODELS / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors",
            MODELS / "DiT/lora_weights" / args.adapter,
            merged,
            "cpu",
        )
        gc.collect()

        def factory():
            return load_model(
                config.dit.model_type,
                config,
                {"device": torch.device("meta"), "dtype": torch.bfloat16},
            )

        convert_checkpoint(merged, destination / "int8", factory, config.dit.model_type)
    receipt.write_text(json.dumps(model_identity(args.adapter)), encoding="utf-8")


if __name__ == "__main__":
    main()
