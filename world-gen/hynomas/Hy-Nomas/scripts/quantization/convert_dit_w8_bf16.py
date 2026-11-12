"""Convert a floating safetensors DiT to validated INT8 stored / BF16 compute."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

import torch
from torch import nn
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hymm.models.dit.modules.int8_stored_bf16_linear import (
    FORMAT, GROUP_SIZE, MANIFEST, quantize_weight, _sha256, load_w8_bf16_model,
)


def normalize_key(key):
    key = key.replace("._checkpoint_wrapped_module.", ".").replace("_checkpoint_wrapped_module.", "")
    return re.sub(r"^double_blocks\.(\d+)\.action_module\.", r"action_modules.\1.", key)


def convert_checkpoint(source, destination, model_factory, model_type, *, shard_bytes=512 << 20):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source.suffix != ".safetensors" or not source.is_file():
        raise ValueError("Source must be a floating safetensors checkpoint")
    if destination.exists():
        raise FileExistsError(f"Output directory already exists: {destination}")
    if type(shard_bytes) is not int or shard_bytes <= 0:
        raise ValueError("shard_bytes must be positive")
    with torch.device("meta"):
        model = model_factory()
    expected = {key:(tuple(value.shape), value.dtype) for key,value in model.state_dict().items()}
    if any(not p.is_meta for p in model.parameters()):
        raise ValueError("Conversion model must be constructed on meta")
    layers = {name:list(module.weight.shape) for name,module in model.named_modules() if isinstance(module,nn.Linear)}
    if not layers:
        raise ValueError("No Linear modules found")
    del model
    stat = source.stat()
    identity = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": _sha256(source)}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=destination.name+".tmp-", dir=destination.parent))
    try:
        shards, batch, batch_size = [], {}, 0
        def flush():
            nonlocal batch, batch_size
            if not batch:
                return
            name = f"model-{len(shards)+1:05d}.safetensors"
            path = temporary/name
            save_file(batch, str(path))
            shards.append({"file":name,"bytes":path.stat().st_size,"sha256":_sha256(path)})
            batch, batch_size = {}, 0
        def append(key, tensor):
            nonlocal batch_size
            size=tensor.numel()*tensor.element_size()
            if batch and batch_size+size>shard_bytes:
                flush()
            batch[key]=tensor.contiguous()
            batch_size+=size
        with safe_open(str(source),framework="pt",device="cpu") as reader:
            names={}
            for original in reader.keys():
                key=normalize_key(original)
                if key in names:
                    raise ValueError(f"Duplicate source key after normalization: {key}")
                names[key]=original
            if set(names)!=set(expected):
                raise ValueError(f"Source keys mismatch: missing={sorted(expected.keys()-names.keys())[:5]}, unexpected={sorted(names.keys()-expected.keys())[:5]}")
            for key in sorted(names):
                tensor=reader.get_tensor(names[key])
                shape,dtype=expected[key]
                if tuple(tensor.shape)!=shape:
                    raise ValueError(f"Source tensor shape mismatch: {key}")
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    raise ValueError(f"Nonfinite source tensor: {key}")
                name=key.removesuffix(".weight")
                if key.endswith(".weight") and name in layers:
                    if tensor.dtype not in {torch.bfloat16,torch.float16,torch.float32}:
                        raise ValueError(f"Expected floating Linear weight: {key}")
                    q,scale=quantize_weight(tensor)
                    append(name+".weight_q",q)
                    append(name+".weight_scale",scale)
                else:
                    if dtype.is_floating_point and not tensor.is_floating_point():
                        raise ValueError(f"Expected floating source tensor: {key}")
                    if not dtype.is_floating_point and tensor.dtype!=dtype:
                        raise ValueError(f"Source dtype mismatch: {key}")
                    append(key,tensor.to(torch.bfloat16 if dtype.is_floating_point else dtype))
                del tensor
            flush()
        current=source.stat()
        if current.st_size!=identity["bytes"] or current.st_mtime_ns!=identity["mtime_ns"]:
            raise RuntimeError("Source changed during conversion")
        manifest={"format":FORMAT,"group_size":GROUP_SIZE,"model_type":model_type,
                  "layers":layers,"shards":shards,"source":identity}
        (temporary/MANIFEST).write_text(json.dumps(manifest,indent=2)+"\n")
        # Verify the output using the production loader before publishing.
        verified=load_w8_bf16_model(model_factory,model_type,temporary,"cpu")
        del verified
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",required=True,type=Path)
    parser.add_argument("--output-dir",required=True,type=Path)
    parser.add_argument("--experiment",default="distill")
    parser.add_argument("--shard-mib",type=int,default=512)
    options,overrides=parser.parse_known_args()
    from configs.config import parse_args
    from hymm.models.dit.modules import load_model
    sys.argv=[sys.argv[0],f"+experiment={options.experiment}",*overrides,
              "inference.dit_weight_quantization=none","model.dit.seg_ar_attention_backend=magi"]
    args=parse_args(mode="inference")
    factory=lambda:load_model(args.dit.model_type,args,{"device":torch.device("meta"),"dtype":torch.bfloat16})
    manifest=convert_checkpoint(options.source,options.output_dir,factory,args.dit.model_type,
                                shard_bytes=options.shard_mib<<20)
    print(json.dumps({"output":str(options.output_dir),"layers":len(manifest["layers"]),
                      "shards":len(manifest["shards"]),"bytes":sum(x["bytes"] for x in manifest["shards"])},indent=2))


if __name__=="__main__":
    main()
