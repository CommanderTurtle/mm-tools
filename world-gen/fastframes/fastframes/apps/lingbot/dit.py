#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Optimized LingBot DiT evaluation for one four-latent chunk (Sec. 3.4 of the paper).

The parameters are those of the upstream ``WanModelFast`` (image-to-video DiT with Pluecker
camera injection); only its RoPE frequency helper is used at run time. Per chunk,
``begin_chunk`` stages the image condition, camera rays and absolute RoPE table and
evaluates the camera branch of every block once for all denoising steps. ``forward`` then evaluates one denoising step
(or the clean ``t = 0`` pass of the first chunk):

* fused QKV GEMM; the text K/V and the time modulation of every timestep are precomputed;
* the chunk's keys and values are written into the rolling cache (RoPE applied in FP32)
  and attended with cuDNN attention (optionally split-KV with a log-sum-exp merge);
* per-block ``torch.compile`` (``max-autotune-no-cudagraphs``) and CUDA graphs per cache length
  (denoising steps and cache-writing pass) plus one for the camera branch;
* context parallel under ``world > 1``: each rank owns a contiguous slice of the chunk's
  6032 query tokens; the new K/V of every layer are all-gathered through NVLink symmetric
  memory and every rank keeps the full cache.

At the final denoising timestep the pre-head hidden states, the head modulation and the
canonical (pre-RoPE, RMS-normalized) keys and values of the chunk are kept for the pixel
decoder and the KV refinement network.

Precision follows the upstream model under BF16 autocast: modulation, residual stream,
layer norms and the output head in FP32, the time MLP in FP32.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from .conditioning import CHUNK, LATENT_HEIGHT, LATENT_WIDTH

__all__ = ["LingBotDiT", "DiTResult", "rope_real", "rope_table", "rope_frequencies", "all_gather_into"]

GRID_H, GRID_W = LATENT_HEIGHT // 2, LATENT_WIDTH // 2
FRAME_TOKENS = GRID_H * GRID_W          # 1508
TOKENS = CHUNK * FRAME_TOKENS           # 6032
CACHE_FRAMES = 18                       # rolling cache, including the current chunk
READ_FRAMES = (4, 8, 12, 16, 18)


@torch.library.custom_op("fastframes_lingbot::all_gather_into", mutates_args=("out",))
def all_gather_into(value: torch.Tensor, group_name: str, out: torch.Tensor) -> None:
    """Symmetric-memory multicast all-gather, opaque to ``torch.compile(fullgraph=True)``."""
    torch.ops.symm_mem.multimem_all_gather_out(value, group_name, out)


@all_gather_into.register_fake
def _(value, group_name, out):
    return None


def rope_frequencies(head_dim, device):
    """Upstream complex RoPE frequencies ``[1024, head_dim / 2]`` (time, height, width)."""
    from wan.modules.model_fast import rope_params

    d = head_dim
    return torch.cat([rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)),
                      rope_params(1024, 2 * (d // 6))], dim=1).to(device)


def rope_table(freqs, start_frame):
    """FP32 ``(cos, sin)`` table ``[6032, 1, D/2, 2]`` of latent frames ``start_frame + [0, 4)``."""
    half = freqs.shape[1]
    t, h, w = freqs.split([half - 2 * (half // 3), half // 3, half // 3], dim=1)
    f = CHUNK
    table = torch.cat([t[start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, GRID_H, GRID_W, -1),
                       h[:GRID_H].view(1, GRID_H, 1, -1).expand(f, GRID_H, GRID_W, -1),
                       w[:GRID_W].view(1, 1, GRID_W, -1).expand(f, GRID_H, GRID_W, -1)], dim=-1)
    return torch.view_as_real(table.reshape(TOKENS, 1, half)).float()


def rope_real(value, table):
    """Absolute RoPE as FP32 real rotations; ``table`` from ``rope_table``."""
    pairs = value.float().reshape(*value.shape[:-1], value.shape[-1] // 2, 2)
    xr, xi = pairs[..., 0], pairs[..., 1]
    fr, fi = table[..., 0], table[..., 1]
    return torch.stack((xr * fr - xi * fi, xr * fi + xi * fr), -1).flatten(-2).to(value.dtype)


def rms_norm(value, module):
    normed = value.float() * torch.rsqrt(value.float().square().mean(-1, keepdim=True) + module.eps)
    return normed.to(value.dtype) * module.weight


def layer_norm(value, module):
    """Upstream ``WanLayerNorm``: FP32 statistics, result in the input dtype."""
    weight = None if module.weight is None else module.weight.float()
    bias = None if module.bias is None else module.bias.float()
    return F.layer_norm(value.float(), (value.shape[-1],), weight, bias, module.eps).to(value.dtype)


def attention(query, key, value, split=1):
    """``query [1, Q, H, D]`` over ``key, value [1, K, H, D]`` with cuDNN; exact split-KV merge."""
    q = query.transpose(1, 2)
    if split == 1:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return F.scaled_dot_product_attention(q, key.transpose(1, 2), value.transpose(1, 2)).transpose(1, 2)
    _, length, heads, dim = key.shape
    k = key.reshape(split, length // split, heads, dim).transpose(1, 2)
    v = value.reshape(split, length // split, heads, dim).transpose(1, 2)
    out, lse, *_ = torch.ops.aten._scaled_dot_product_cudnn_attention(
        q.expand(split, -1, -1, -1).contiguous(), k, v, None, True, 0.0, False, False, scale=dim ** -0.5)
    if lse.ndim == 3:
        lse = lse.unsqueeze(-1)
    weights = torch.exp(lse - lse.amax(0, keepdim=True))
    merged = (out.float() * weights).sum(0, keepdim=True) / weights.sum(0, keepdim=True)
    return merged.to(query.dtype).transpose(1, 2)


@dataclass
class DiTResult:
    """Persistent outputs, overwritten by the next ``forward``."""

    prediction: torch.Tensor  # [16, 4, 58, 104] FP32 flow prediction
    features: torch.Tensor    # [1, 6032, dim] FP32 pre-head hidden states (final timestep)
    embedding: torch.Tensor   # [1, 4, 1, dim] FP32 head modulation (final timestep)
    key: torch.Tensor         # [layers, 1, 6032, heads, head_dim] BF16 canonical keys (final timestep)
    value: torch.Tensor       # same geometry, BF16 values


class LingBotDiT:
    """Chunk-wise evaluation of a ``WanModelFast`` against the rolling KV cache.

    ``cache_k`` / ``cache_v``: ``[layers, 1, 18 * 1508, heads, head_dim]`` BF16, owned by the
    caller (who also evicts frames between chunks). ``timesteps``: the denoising timesteps; the
    last one is the one whose features and K/V are captured.
    """

    def __init__(self, model, cache_k, cache_v, context, timesteps, *, world=1, rank=0, split_kv=1,
                 compile=True, graphs=True):
        self.model, self.cache_k, self.cache_v = model, cache_k, cache_v
        self.world, self.rank = int(world), int(rank)
        self.dim, self.heads = model.dim, model.num_heads
        self.head_dim, self.layers = self.dim // self.heads, len(model.blocks)
        self.device = model.patch_embedding.weight.device
        self.split_kv, self.use_graphs = int(split_kv), bool(graphs)
        self.timesteps = tuple(int(t) for t in timesteps) + (0,)
        self.final_timestep = self.timesteps[-2]
        if TOKENS % self.world or (CACHE_FRAMES * FRAME_TOKENS) % self.split_kv:
            raise ValueError("world size must divide 6032 tokens and split_kv the cache length")
        if self.world > 1 and dist.get_world_size() != self.world:
            raise ValueError("initialize a matching NCCL process group first")
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 128)
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit, 4096)
        import torch._inductor.config as inductor_config
        inductor_config.emulate_precision_casts = True

        dim, device = self.dim, self.device
        self.local_tokens = TOKENS // self.world
        self.start, self.end = self.rank * self.local_tokens, (self.rank + 1) * self.local_tokens
        self.x = torch.zeros((16, CHUNK, LATENT_HEIGHT, LATENT_WIDTH), device=device)
        self.y = torch.empty((20, CHUNK, LATENT_HEIGHT, LATENT_WIDTH), device=device)
        self.camera = torch.empty((1, 384, CHUNK, LATENT_HEIGHT, LATENT_WIDTH), device=device, dtype=torch.bfloat16)
        self.rope = torch.empty((TOKENS, 1, self.head_dim // 2, 2), device=device)
        self.time_e = torch.empty((1, 1, dim), device=device)
        self.time_e0 = torch.empty((1, 1, 6, dim), device=device)
        self.camera_params = torch.empty((self.layers, 2, 1, self.local_tokens, dim), device=device,
                                         dtype=torch.bfloat16)
        shape = (self.layers, 1, TOKENS, self.heads, self.head_dim)
        self.result = DiTResult(
            torch.empty_like(self.x), torch.empty((1, TOKENS, dim), device=device),
            torch.empty((1, CHUNK, 1, dim), device=device),
            torch.empty(shape, device=device, dtype=torch.bfloat16),
            torch.empty(shape, device=device, dtype=torch.bfloat16))
        self.patch_weight = model.patch_embedding.weight.reshape(dim, -1).contiguous()
        self.qkv = [(torch.cat([b.self_attn.q.weight, b.self_attn.k.weight, b.self_attn.v.weight]).contiguous(),
                     torch.cat([b.self_attn.q.bias, b.self_attn.k.bias, b.self_attn.v.bias]).contiguous())
                    for b in model.blocks]
        self.cam = [(torch.cat([b.cam_scale_layer.weight, b.cam_shift_layer.weight]).contiguous(),
                     torch.cat([b.cam_scale_layer.bias, b.cam_shift_layer.bias]).contiguous())
                    for b in model.blocks]
        self.gather = [None] * self.layers
        self.feature_gather = self.output_gather = None
        if self.world > 1:
            import torch.distributed._symmetric_memory as symm_mem
            self.group_name = dist.group.WORLD.group_name

            def buffers(width, dtype):
                shard = symm_mem.empty(self.local_tokens, width, dtype=dtype, device=device)
                full = symm_mem.empty(TOKENS, width, dtype=dtype, device=device)
                symm_mem.rendezvous(shard, self.group_name)
                symm_mem.rendezvous(full, self.group_name)
                return shard, full

            self.gather = [buffers(2 * dim, torch.bfloat16) for _ in model.blocks]
            self.feature_gather = buffers(dim, torch.float32)
            self.output_gather = buffers(64, torch.bfloat16)
        self.text_k = torch.empty((self.layers, 1, 512, self.heads, self.head_dim), device=device, dtype=torch.bfloat16)
        self.text_v = torch.empty_like(self.text_k)
        self.set_context(context)
        self.time_bank = self._time_bank()
        self.time_e.copy_(self.time_bank[self.final_timestep][0])
        self.time_e0.copy_(self.time_bank[self.final_timestep][1])
        self.graphs = {}
        self.pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
        self._camera_graph, self._chunk_staged = None, False

        def compiled(fn):
            if not compile:
                return fn
            return torch.compile(fn, dynamic=False, fullgraph=True, mode="max-autotune-no-cudagraphs")
        self._camera_run = compiled(self._prepare_camera)
        self._patch_run = compiled(self._patch)
        self._block_run = compiled(self._block)
        self._post_run = compiled(self._post)

    # ------------------------------------------------------------------ constants
    @torch.no_grad()
    def set_context(self, context):
        """Text K/V of every block for a ``[512, 4096]`` prompt embedding (written in place)."""
        m = self.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            projected = m.text_embedding(context.reshape(1, 512, 4096))
            for i, block in enumerate(m.blocks):
                cross = block.cross_attn
                self.text_k[i].copy_(rms_norm(cross.k(projected), cross.norm_k).view(1, 512, self.heads, self.head_dim))
                self.text_v[i].copy_(cross.v(projected).view(1, 512, self.heads, self.head_dim))

    @torch.no_grad()
    def _time_bank(self):
        m, bank = self.model, {}
        with torch.autocast("cuda", enabled=False):
            half = m.freq_dim // 2
            inv = 10000 ** (-torch.arange(half, device=self.device, dtype=torch.float64) / half)
            for timestep in self.timesteps:
                phases = timestep * inv
                sinusoid = torch.cat([phases.cos(), phases.sin()])[None, None].float()
                e = m.time_embedding(sinusoid)
                bank[timestep] = e, m.time_projection(e).unflatten(-1, (6, self.dim))
        return bank

    # ------------------------------------------------------------------ compiled pieces
    def _patch(self):
        inputs = torch.cat([self.x, self.y], 0)[None]
        tokens = inputs.reshape(1, 36, CHUNK, GRID_H, 2, GRID_W, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(1, TOKENS, 144)
        return F.linear(tokens[:, self.start:self.end], self.patch_weight, self.model.patch_embedding.bias)

    def _prepare_camera(self):
        m = self.model
        camera = self.camera.reshape(1, 384, CHUNK, GRID_H, 2, GRID_W, 2).permute(0, 2, 3, 5, 1, 4, 6)
        camera = camera.reshape(1, TOKENS, 384 * 4)[:, self.start:self.end]
        camera = m.patch_embedding_wancamctrl(camera)
        camera = camera + m.c2ws_hidden_states_layer2(F.silu(m.c2ws_hidden_states_layer1(camera)))
        for i, block in enumerate(m.blocks):
            hidden = block.cam_injector_layer2(F.silu(block.cam_injector_layer1(camera))) + camera
            scale, shift = F.linear(hidden, *self.cam[i]).chunk(2, -1)
            self.camera_params[i, 0].copy_(scale)
            self.camera_params[i, 1].copy_(shift)

    def _block(self, block, value, cache_k, cache_v, text_k, text_v, cam, qkv_weight, qkv_bias, gathered,
               captured_k, captured_v, read_frames: int, capture: bool):
        heads, head_dim, tokens = self.heads, self.head_dim, self.local_tokens
        e = (block.modulation.unsqueeze(0) + self.time_e0).unbind(2)
        hidden = layer_norm(value, block.norm1).float() * (1 + e[1]) + e[0]
        a = block.self_attn
        query, key, val = F.linear(hidden, qkv_weight, qkv_bias).chunk(3, -1)
        query = rms_norm(query, a.norm_q).view(1, tokens, heads, head_dim)
        key = rms_norm(key, a.norm_k).view(1, tokens, heads, head_dim)
        val = val.view(1, tokens, heads, head_dim)
        query = rope_real(query, self.rope[self.start:self.end])
        if self.world > 1:
            shard, full = gathered
            shard.copy_(torch.cat([key.flatten(2), val.flatten(2)], -1).reshape(tokens, 2 * self.dim))
            all_gather_into(shard, self.group_name, full)
            key, val = full.split(self.dim, -1)
            key = key.reshape(1, TOKENS, heads, head_dim)
            val = val.reshape(1, TOKENS, heads, head_dim)
        physical_key = rope_real(key, self.rope)
        if capture:
            captured_k.copy_(key)
            captured_v.copy_(val)
        end = read_frames * FRAME_TOKENS
        cache_k[:, end - TOKENS:end].copy_(physical_key)
        cache_v[:, end - TOKENS:end].copy_(val)
        split = self.split_kv if end % self.split_kv == 0 else 1
        attended = attention(query, cache_k[:, :end], cache_v[:, :end], split)
        value = value + F.linear(attended.reshape(1, tokens, self.dim), a.o.weight, a.o.bias) * e[2]
        # Upstream forms 1 + camera_scale in BF16 before the FP32 residual product.
        value = (1.0 + cam[0]) * value + cam[1]
        cross = block.cross_attn
        q = rms_norm(cross.q(layer_norm(value, block.norm3)), cross.norm_q).view(1, tokens, heads, head_dim)
        value = value + cross.o(attention(q, text_k, text_v).reshape(1, tokens, self.dim))
        hidden = layer_norm(value, block.norm2).float() * (1 + e[4]) + e[3]
        return value + block.ffn(hidden) * e[5]

    def _post(self, value, capture: bool):
        if capture:
            if self.world > 1:
                shard, full = self.feature_gather
                shard.copy_(value.reshape(self.local_tokens, self.dim))
                all_gather_into(shard, self.group_name, full)
                self.result.features.copy_(full[None])
            else:
                self.result.features.copy_(value)
            self.result.embedding.copy_(self.time_e[:, None].expand(1, CHUNK, 1, self.dim))
        head = self.model.head
        e = (head.modulation.unsqueeze(0) + self.time_e.unsqueeze(2)).unbind(2)
        hidden = layer_norm(value, head.norm) * (1 + e[1]) + e[0]
        with torch.autocast("cuda", enabled=False):  # the upstream head runs in FP32
            output = F.linear(hidden.float(), head.head.weight.float(), head.head.bias.float())
        # Upstream unpatchifies under BF16 autocast, which rounds the prediction to BF16.
        output = output.to(torch.bfloat16).reshape(self.local_tokens, 64)
        if self.world > 1:
            shard, full = self.output_gather
            shard.copy_(output)
            all_gather_into(shard, self.group_name, full)
            output = full
        output = output.reshape(CHUNK, GRID_H, GRID_W, 1, 2, 2, 16).permute(6, 0, 3, 1, 4, 2, 5)
        self.result.prediction.copy_(output.reshape(16, CHUNK, LATENT_HEIGHT, LATENT_WIDTH).float())

    def _run(self, read_frames, capture):
        value = self._patch_run()
        for i, block in enumerate(self.model.blocks):
            value = self._block_run(block, value, self.cache_k[i], self.cache_v[i], self.text_k[i], self.text_v[i],
                                    self.camera_params[i], *self.qkv[i], self.gather[i],
                                    self.result.key[i], self.result.value[i], read_frames, capture)
        self._post_run(value, capture)

    # ------------------------------------------------------------------ public API
    @torch.no_grad()
    def begin_chunk(self, y, camera, rope):
        """Stage the chunk's image condition, rays and RoPE table; evaluate the camera branch."""
        self.y.copy_(y)
        self.camera.copy_(camera)
        self.rope.copy_(rope)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not self.use_graphs:
                self._camera_run()
            else:
                if self._camera_graph is None:
                    self._camera_graph = self._capture(self._camera_run)
                self._camera_graph.replay()
        self._chunk_staged = True

    def _capture(self, fn):
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.current_stream(self.device).wait_stream(side)
        if self.world > 1:
            dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            fn()
        return graph

    @torch.no_grad()
    def forward(self, x, timestep, read_frames, *, capture=None):
        """One evaluation on the staged chunk; ``capture`` defaults to ``timestep == final``."""
        if not self._chunk_staged:
            raise RuntimeError("call begin_chunk first")
        timestep = int(timestep)
        capture = timestep == self.final_timestep if capture is None else bool(capture)
        key = (int(read_frames), capture)
        if key[0] not in READ_FRAMES:
            raise ValueError(f"unsupported cache length {read_frames}")
        self.x.copy_(x)
        self.time_e.copy_(self.time_bank[timestep][0])
        self.time_e0.copy_(self.time_bank[timestep][1])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if not self.use_graphs:
                self._run(*key)
            else:
                if key not in self.graphs:
                    self.graphs[key] = self._capture(lambda: self._run(*key))
                self.graphs[key].replay()
        return self.result

    __call__ = forward

    def close(self):
        """Release graphs before their symmetric-memory buffers (call on every rank)."""
        torch.cuda.synchronize(self.device)
        self.graphs.clear()
        self._camera_graph = None
        self.gather = [None] * self.layers
        self.feature_gather = self.output_gather = None
