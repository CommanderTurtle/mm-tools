"""Optional verified local copy of a W8 checkpoint; never modifies its source."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

MANIFEST = "w8_bf16_manifest.json"


def cached_checkpoint(source, cache_dir=None):
    source = Path(source).resolve()
    cache_dir = cache_dir or os.environ.get("INFERENCE_W8_CACHE_DIR")
    if not cache_dir:
        return source
    raw = (source / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("format") != "nomas-dit-w8-bf16-v1" or manifest.get("group_size") != 128:
        raise ValueError("Unsupported W8 cache manifest")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("W8 cache requires shards")
    names = set()
    for shard in shards:
        name = shard.get("file") if isinstance(shard, dict) else None
        if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("Unsafe W8 cache shard path")
        if name in names or not (source / name).resolve().is_relative_to(source):
            raise ValueError("Duplicate or escaping W8 cache shard")
        names.add(name)
        checksum = shard.get("sha256")
        if type(shard.get("bytes")) is not int or shard["bytes"] <= 0:
            raise ValueError("Invalid W8 cache shard size")
        if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
            raise ValueError("Invalid W8 cache checksum")
    digest = hashlib.sha256(raw).hexdigest()
    cache = Path(cache_dir).expanduser().resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / digest
    # One process copies; the others wait and then validate/load local bytes.
    with (cache / (digest + ".lock")).open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            if (target / MANIFEST).read_bytes() != raw:
                raise ValueError("Local W8 manifest changed")
            return target
        temporary = Path(tempfile.mkdtemp(prefix=digest+".tmp-", dir=cache))
        try:
            def copy(shard):
                original = source / shard["file"]
                if original.stat().st_size != shard["bytes"]:
                    raise ValueError(f"Source W8 shard size mismatch: {shard['file']}")
                checksum = hashlib.sha256()
                written = 0
                with original.open("rb") as src, (temporary / shard["file"]).open("wb") as dst:
                    for block in iter(lambda: src.read(8 << 20), b""):
                        checksum.update(block)
                        written += len(block)
                        dst.write(block)
                if written != shard["bytes"] or checksum.hexdigest() != shard["sha256"]:
                    raise ValueError(f"Source W8 shard checksum mismatch: {shard['file']}")
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(copy, shards))
            if (source / MANIFEST).read_bytes() != raw:
                raise RuntimeError("W8 source manifest changed during copying")
            (temporary / MANIFEST).write_bytes(raw)
            (temporary / "cache_origin.json").write_text(
                json.dumps({"source":str(source),"manifest_sha256":digest,
                            "verified_shards":len(shards)}, indent=2)+"\n")
            temporary.rename(target)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
    return target
