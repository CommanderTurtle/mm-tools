"""QVG 0.1.0 PRQ storage with independently selectable token payloads.

Encoding uses the upstream codec. The decoder writes the same packed format
directly into BSHD attention buffers, without a full BHSD temporary or cat.
Already-rotated keys are preserved; selection never requantizes old history.
"""

from dataclasses import dataclass, fields
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError

import torch
import triton
import triton.language as tl

from hymm.utils.inference_metrics import profiled, stage
from .kv_cache_storage import record_qvg_working_peak, storage_bytes


UPSTREAM_VERSION = "0.1.0"


@dataclass(frozen=True)
class QVGOptions:
    preset: str = "upstream_default"
    text_storage: str = "int8"
    seed: int = 42
    num_bits: int = 2
    num_stages: int = 4
    k_centroids: int = 256
    v_centroids: int = 256
    block_size: int = 64
    max_iters: int = 100

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        unknown = set(config) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown QVG options: {sorted(unknown)}")
        result = cls(**config)
        if result.preset != "upstream_default":
            raise ValueError("QVG preset must be upstream_default")
        if result.text_storage not in ("int8", "bf16"):
            raise ValueError("QVG text_storage must be int8 or bf16")
        for name in ("seed", "num_bits", "num_stages", "k_centroids",
                     "v_centroids", "block_size", "max_iters"):
            value = getattr(result, name)
            if type(value) is not int or value < (0 if name == "seed" else 1):
                raise ValueError(f"QVG {name} must be a valid integer")
        if (result.num_bits not in (2, 4) or not 1 <= result.num_stages <= 4
                or not 1 <= result.k_centroids <= 256
                or not 1 <= result.v_centroids <= 256
                or result.block_size not in (16, 32, 64, 128)):
            raise ValueError("Unsupported QVG bits, stages, centroids, or block size")
        return result


@lru_cache(maxsize=1)
def upstream_encoder():
    try:
        installed = version("quant-videogen")
        if installed != UPSTREAM_VERSION:
            raise RuntimeError(f"QVG requires quant-videogen=={UPSTREAM_VERSION}, "
                               f"found {installed}")
        from quant_videogen.real.prq import prq_quant
    except (PackageNotFoundError, ImportError) as error:
        raise RuntimeError("QVG requires quant-videogen==0.1.0 and its codec dependencies. "
                           "See scripts/inference/QVG_KV_CACHE.md; no fallback was selected.") from error
    return prq_quant


def _select(tensor, dim, indices):
    # CPU index_select does not implement all float8 dtypes.
    if tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.uint8).index_select(dim, indices).view(tensor.dtype)
    return tensor.index_select(dim, indices)


@dataclass(frozen=True)
class QVGTensor:
    centroids: torch.Tensor
    cluster_ids: torch.Tensor
    residual: torch.Tensor
    scales: torch.Tensor
    num_bits: int
    block_size: int

    def tensors(self):
        return iter((self.centroids, self.cluster_ids, self.residual, self.scales))

    @property
    def tokens(self):
        return self.residual.shape[2]

    def select(self, indices):
        indices = indices.to(device=self.residual.device, dtype=torch.long)
        return QVGTensor(
            self.centroids, self.cluster_ids.index_select(3, indices).contiguous(),
            self.residual.index_select(2, indices).contiguous(),
            _select(self.scales, 2, indices).contiguous(), self.num_bits, self.block_size,
        )

    def to(self, device):
        return QVGTensor(
            *(tensor.to(device=device, copy=True, non_blocking=False).contiguous()
              for tensor in self.tensors()), self.num_bits, self.block_size,
        )


@dataclass(frozen=True)
class QVGBlock:
    k: QVGTensor
    v: QVGTensor
    text: dict

    @property
    def video_tokens(self):
        return self.k.tokens

    @property
    def text_tokens(self):
        return self.text["k"].shape[1]

    @property
    def tokens(self):
        return self.video_tokens + self.text_tokens

    def tensors(self):
        yield from self.k.tensors()
        yield from self.v.tensors()
        yield from self.text.values()

    def select(self, indices):
        return QVGBlock(self.k.select(indices), self.v.select(indices), self.text)

    def to(self, device):
        return QVGBlock(self.k.to(device), self.v.to(device), {
            key: tensor.to(device=device, copy=True, non_blocking=False).contiguous()
            for key, tensor in self.text.items()
        })


def make_qvg_kv_block(batch, capacity, heads, dim, *, device, options=None, layer=0):
    options = QVGOptions.from_config(vars(options or QVGOptions()))
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if (device.type != "cuda" or dim != 128 or batch != 1
            or min(capacity, heads) < 1
            or torch.cuda.get_device_capability(device) < (8, 9)):
        raise ValueError("QVG currently requires CUDA SM89+, B=1, D=128, and positive capacity")
    upstream_encoder()
    return {
        "storage_format": "qvg", "shape": (batch, capacity, heads, dim),
        "device": device, "qvg_options": options, "qvg_layer": layer,
        "qvg_working": [], "qvg_stats": {},
    }


def _pack_upstream(values, clusters, options, encoder):
    centers, labels, residual, scales, zeros = encoder(
        values.permute(0, 2, 1, 3).contiguous(),
        n_stages=options.num_stages, n_clusters=clusters,
        block_size=options.block_size, num_bits=options.num_bits,
        scale_precision=torch.float8_e4m3fn, max_iters=options.max_iters,
        PACK_OUTPUT_INT8=True, CLUSTER_ID_INT8=True,
    )
    if zeros is not None:
        raise RuntimeError("Unexpected asymmetric QVG payload")
    if (not all(bool(torch.isfinite(center).all()) for center in centers)
            or not bool(torch.isfinite(scales.float()).all())):
        raise ValueError("QVG codebook or FP8 scales are non-finite; no cache was committed")
    return QVGTensor(
        torch.stack(centers, 2).contiguous(), torch.stack(labels, 2).contiguous(),
        residual.contiguous(), scales.contiguous(), options.num_bits, options.block_size,
    )


@profiled("kv_encode")
def encode_segment(cache, k, v, image_tokens):
    options = cache["qvg_options"]
    shape = cache["shape"]
    if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != shape[0]
            or k.shape[2:] != shape[2:] or k.dtype != torch.bfloat16
            or v.dtype != k.dtype or k.device != v.device
            or k.device != cache["device"] or not 0 < image_tokens <= k.shape[1]):
        raise ValueError("QVG expects matching BF16 [B,L,H,128] segment KV on the cache device")
    if not bool(torch.isfinite(k).all()) or not bool(torch.isfinite(v).all()):
        raise ValueError("Cannot quantize non-finite QVG KV")
    stats = cache["qvg_stats"]
    device_index = k.device.index if k.device.index is not None else torch.cuda.current_device()
    seed = (options.seed + cache["qvg_layer"] * 1000003
            + stats.get("encode_calls", 0)) % (2**63)
    with torch.random.fork_rng(devices=[device_index]):
        with torch.cuda.device(device_index):
            torch.cuda.manual_seed(seed)
        encoder = upstream_encoder()
        encoded_k = _pack_upstream(k[:, :image_tokens], options.k_centroids, options, encoder)
        encoded_v = _pack_upstream(v[:, :image_tokens], options.v_centroids, options, encoder)
        text_k, text_v = k[:, image_tokens:], v[:, image_tokens:]
        if options.text_storage == "int8":
            from .int8_kv_cache import make_int8_kv_block, quantize_kv_pair_into
            text = make_int8_kv_block(*text_k.shape, device=k.device)
            quantize_kv_pair_into(text, text_k, text_v, 0)
        else:
            text = {"k": text_k.clone(), "v": text_v.clone()}
    stats["encode_calls"] = stats.get("encode_calls", 0) + 1
    return QVGBlock(encoded_k, encoded_v, text)


@triton.jit
def _decode_into(C, I, R, S, OUT, L: tl.constexpr, H: tl.constexpr,
                 D: tl.constexpr, K: tl.constexpr, STAGES: tl.constexpr,
                 BITS: tl.constexpr, GROUP: tl.constexpr,
                 OS0: tl.constexpr, OS1: tl.constexpr, OS2: tl.constexpr,
                 OS3: tl.constexpr, START: tl.constexpr, ROWS: tl.constexpr):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    b = row // (L * H)
    token = row // H % L
    h = row % H
    bh = b * H + h
    mask = row < L * H
    packed = tl.load(R + bh[:, None] * L * (D // (8 // BITS))
                     + token[:, None] * (D // (8 // BITS))
                     + d[None, :] // (8 // BITS), mask=mask[:, None], other=0).to(tl.int32)
    shift = (8 // BITS - 1 - d % (8 // BITS)) * BITS
    code = ((packed >> shift[None, :]) & ((1 << BITS) - 1)) - ((1 << (BITS - 1)) - 1)
    scale = tl.load(S + (bh[:, None] * L + token[:, None]) * (D // GROUP)
                    + d[None, :] // GROUP, mask=mask[:, None], other=0.0).to(tl.float32)
    value = code.to(tl.float32) * scale
    for s in range(STAGES):
        label = tl.load(I + (bh * STAGES + s) * L + token, mask=mask, other=0).to(tl.int32)
        center = tl.load(C + ((bh[:, None] * STAGES + s) * K + label[:, None]) * D
                         + d[None, :], mask=mask[:, None], other=0).to(tl.float32)
        value += center
    tl.store(OUT + b[:, None] * OS0 + (token[:, None] + START) * OS1
             + h[:, None] * OS2 + d[None, :] * OS3, value, mask=mask[:, None])


def decode_tensor_into(payload, output, start=0):
    if (output.ndim != 4 or output.dtype != torch.bfloat16 or not output.is_cuda
            or output.shape[0] != 1 or output.shape[2] != payload.residual.shape[1]
            or output.shape[3] != payload.centroids.shape[-1]
            or start < 0 or start + payload.tokens > output.shape[1]
            or output.device != payload.residual.device):
        raise ValueError("Invalid QVG BF16 decode destination")
    if payload.tokens:
        heads = output.shape[2]
        _decode_into[(triton.cdiv(payload.tokens * heads, 8),)](
            payload.centroids, payload.cluster_ids, payload.residual, payload.scales,
            output, payload.tokens, heads, output.shape[3], payload.centroids.shape[3],
            payload.centroids.shape[2], payload.num_bits, payload.block_size,
            *output.stride(), start, 8,
        )


def decode_history_into(cache, k, v, valid_len):
    from .int8_kv_cache import dequantize_kv

    if (k.shape != v.shape or k.ndim != 4 or k.dtype != v.dtype or k.device != v.device
            or valid_len < 0 or valid_len > k.shape[1]):
        raise ValueError("QVG history K/V destinations must match and cover valid_len")
    if sum(block.tokens for block in cache["qvg_working"]) != valid_len:
        raise ValueError("QVG working set does not cover the valid history")
    with stage("kv_decode"):
        cursor = 0
        for block in cache["qvg_working"]:
            for key, output in (("k", k), ("v", v)):
                decode_tensor_into(getattr(block, key), output, cursor)
                if block.text_tokens:
                    text = (dequantize_kv(block.text, key, block.text_tokens)
                            if block.text["k"].dtype == torch.int8 else block.text[key])
                    output[:, cursor + block.video_tokens:cursor + block.tokens].copy_(text)
            cursor += block.tokens
    cache["qvg_stats"]["decode_calls"] = cache["qvg_stats"].get("decode_calls", 0) + 1
    cache["qvg_stats"]["peak_decode_buffer_bytes"] = max(
        cache["qvg_stats"].get("peak_decode_buffer_bytes", 0), storage_bytes((k, v)),
    )


def validate_qvg_append(cache, mask, k, v, current_mask, valid_len, image_tokens):
    """Check append-only storage before attention or encoding mutates state."""
    shape = cache["shape"]
    if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != shape[0]
            or k.shape[2:] != shape[2:] or k.dtype != torch.bfloat16
            or v.dtype != k.dtype or k.device != v.device
            or k.device != cache["device"] or not 0 < image_tokens <= k.shape[1]
            or current_mask.shape != k.shape[:2] or current_mask.dtype != torch.bool
            or current_mask.device != cache["device"]
            or mask is None or mask.shape != shape[:2] or mask.dtype != torch.bool
            or mask.device != cache["device"] or "swa_capacity" in cache):
        raise ValueError("Invalid QVG full-history geometry or masks")
    if (valid_len < 0 or valid_len != cache.get("qvg_valid_len", 0)
            or sum(block.tokens for block in cache["qvg_working"]) != valid_len):
        raise ValueError("QVG full-history valid length disagrees with the working set")
    new_end = valid_len + k.shape[1]
    if new_end > shape[1]:
        raise ValueError("QVG full history exceeds token capacity")
    return new_end


@profiled("kv_rebuild")
def commit_qvg_append(cache, mask, k, v, current_mask, valid_len, image_tokens):
    """Retain every completed segment on GPU without sampling or reencoding."""
    new_end = validate_qvg_append(
        cache, mask, k, v, current_mask, valid_len, image_tokens,
    )
    encoded = encode_segment(cache, k, v, image_tokens)
    working = [*cache["qvg_working"], encoded]
    record_qvg_working_peak(cache, working)
    mask[:, valid_len:new_end].copy_(current_mask)
    cache["qvg_working"] = working
    cache["qvg_valid_len"] = new_end


def _transfer(cache, block, device, direction):
    with stage(f"kv_{direction}"):
        result = block.to(device)
    key = f"{direction}_bytes"
    cache["qvg_stats"][key] = cache["qvg_stats"].get(key, 0) + storage_bytes(result)
    return result


def commit_qvg_history(cache, mask, k, v, current_mask, image_tokens,
                       text_tokens, tokens_per_frame):
    from .uniform_history_cache import ArchivedSegment, history_frame_indices

    if (tokens_per_frame <= 0 or image_tokens <= 0 or image_tokens % tokens_per_frame
            or text_tokens < 0
            or k.shape[1] != image_tokens + text_tokens
            or current_mask.shape != k.shape[:2] or current_mask.dtype != torch.bool
            or current_mask.device != cache["device"]
            or mask.shape != cache["shape"][:2] or mask.dtype != torch.bool
            or mask.device != cache["device"]):
        raise ValueError("Invalid QVG archive geometry or masks")
    history = cache.get("swa_archive", [])
    index = cache["swa_segment_index"]
    if history and index <= history[-1].segment_index:
        raise ValueError("QVG requires one chronological write per segment")
    frame_start = cache.get("swa_total_frames", 0)
    frames = image_tokens // tokens_per_frame
    frame_ids = history_frame_indices(
        frame_start + frames, cache["swa_past_frames"], cache["swa_sampling"],
        gamma=cache.get("swa_gamma", 2.0),
    )
    action = int(cache.get("swa_action_start", -1))
    if not -1 <= action <= text_tokens:
        raise ValueError("QVG action offset is outside the text block")
    encoded = encode_segment(cache, k, v, image_tokens)
    archived = _transfer(cache, encoded, "cpu", "d2h")
    del encoded
    record = ArchivedSegment(
        index, frame_start, frames, image_tokens, text_tokens, action,
        None, None, current_mask.detach().to("cpu", copy=True).contiguous(),
        qvg_payload=archived,
    )
    records = [*history, record]
    working, masks, layout = [], [], []
    cursor = 0
    with stage("kv_select"):
        for old in records:
            selected = [f - old.frame_start for f in frame_ids
                        if old.frame_start <= f < old.frame_start + old.frame_count]
            if not selected:
                continue
            indices = (torch.tensor(selected)[:, None] * tokens_per_frame
                       + torch.arange(tokens_per_frame)[None]).flatten().long()
            payload = old.qvg_payload.select(indices)
            working.append(_transfer(cache, payload, cache["device"], "h2d"))
            all_indices = torch.cat((indices, torch.arange(old.image_tokens,
                                                           old.image_tokens + old.text_tokens)))
            masks.append(old.mask.index_select(1, all_indices))
            video_end = cursor + len(selected) * tokens_per_frame
            text_end = video_end + old.text_tokens
            layout.append((old.segment_index, cursor, video_end, video_end, text_end, old.action_start))
            cursor = text_end
        if cursor > cache["swa_capacity"]:
            raise ValueError("QVG history exceeds GPU token capacity")
        mask[:, :cursor].copy_(torch.cat(masks, 1))
        mask[:, cursor:] = False
    record_qvg_working_peak(cache, working)
    cache["qvg_working"] = working
    cache["swa_archive"] = records
    cache["swa_total_frames"] = frame_start + frames
    cache["swa_frame_ids"] = frame_ids
    cache["swa_uniform_layout"] = tuple(layout)
    cache["swa_valid_len"] = cursor
    cache["swa_archive_bytes"] = cache.get("swa_archive_bytes", 0) + storage_bytes(archived) + storage_bytes(record.mask)
    mask._action_tail_ranges = []


def commit_qvg_window(cache, mask, k, v, current_mask, valid_len):
    from .swa_cache import eviction_cut, trim_records

    shape = cache["shape"]
    if (k.ndim != 4 or k.shape != v.shape or k.shape[0] != shape[0]
            or k.shape[2:] != shape[2:] or current_mask.shape != k.shape[:2]
            or current_mask.dtype != torch.bool or current_mask.device != cache["device"]
            or mask.shape != shape[:2] or mask.dtype != torch.bool
            or mask.device != cache["device"] or valid_len < 0
            or valid_len > cache["swa_capacity"]):
        raise ValueError("Invalid QVG recent geometry or masks")
    records = cache["swa_video"]
    total = valid_len + k.shape[1]
    cut = eviction_cut(records, total, cache["swa_capacity"])
    old_records = records[:-1]
    if len(old_records) != len(cache["qvg_working"]):
        raise ValueError("QVG recent records disagree with the working set")
    working = []
    for record, block in zip(old_records, cache["qvg_working"]):
        if record.end <= cut:
            continue
        removed = max(0, cut - record.start)
        if removed:
            block = block.select(torch.arange(removed, block.video_tokens, device=k.device))
        working.append(block)
    current_start = max(0, cut - valid_len)
    if current_start < k.shape[1]:
        working.append(encode_segment(
            cache, k[:, current_start:], v[:, current_start:],
            records[-1].image_tokens - current_start,
        ))
    new_len = total - cut
    history_start = min(cut, valid_len)
    remaining = valid_len - history_start
    if remaining and history_start:
        mask[:, :remaining].copy_(mask[:, history_start:valid_len].clone())
    mask[:, remaining:new_len].copy_(current_mask[:, current_start:])
    mask[:, new_len:] = False
    mask._action_tail_ranges = []
    record_qvg_working_peak(cache, working)
    cache["qvg_working"] = working
    cache["swa_video"] = trim_records(records, cut)
    cache["swa_valid_len"] = new_len
    cache["swa_evicted"] = cut
