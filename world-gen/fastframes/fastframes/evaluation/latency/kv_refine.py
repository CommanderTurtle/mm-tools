#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""KV refinement network latency (Sec. 3.3 of the paper).

One call repairs the live-frame cache slot of every DiT layer: stage the clean latent,
patch-embed it with the DiT's patch projection, run the shared 4-block stem, run all
per-layer repair heads on the noisy live K/V read from the cache, and write the repaired
K/V back into the cache. The fixed repair-timestep embedding is precomputed.

Implementation (one GPU): the patch projection is an exact channel-major GEMM (the
Conv3d kernel for this shape launches one copy per output channel), the stem runs
natively, the repair heads plus cache reads/writes are compiled with
``torch.compile`` (max-autotune), and the whole call is captured in a CUDA graph.
``--streams N`` distributes the heads over ``N`` CUDA streams. With ``world > 1`` every
rank computes the stem, repairs its partition of the heads (``--shard tokens``: 1/world of
the tokens of every layer; ``--shard layers``: all tokens of layers/world layers),
the repaired BF16 values are all-gathered and written into every rank's full cache by one
Triton kernel (a per-layer copy loop costs 2 x layers launches).

Timed loops read a fixed noisy source cache and write a separate destination cache of
identical layout, so thousands of calls never re-repair already repaired values.
"""
from __future__ import annotations

import contextlib
import time

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from dit import Geometry
from models import MODEL_DEFAULTS, load_refiner
from timing import capture_graph, compare

from fastframes.checkpoints import load_state_dict
from fastframes.wan.model import WAN_CONFIGS

__all__ = ["KVRefineRuntime", "profile_kv_refine"]

COMMIT_BLOCK = 4096


@triton.jit
def _commit_kernel(gathered, destinations, LAYERS: tl.constexpr, LOCAL: tl.constexpr, TOKENS: tl.constexpr,
                   DIM: tl.constexpr, START: tl.constexpr, BY_TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    """Copy one (K|V, layer) live slot from the all-gather buffer into its cache.

    ``gathered`` is ``[world, 2, layers, LOCAL, DIM]`` (token shards) or ``[world, 2, LOCAL, TOKENS, DIM]``
    (layer shards), viewed as int16 so the copy is bit-exact; ``destinations`` holds the data pointers
    of the contiguous ``[1, kv_len, heads, head_dim]`` caches K[0..LAYERS) followed by V[0..LAYERS).
    """
    kv_layer = tl.program_id(1).to(tl.int64)
    kv, layer = kv_layer // LAYERS, kv_layer % LAYERS
    element = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = element < TOKENS * DIM
    if BY_TOKENS:
        token = element // DIM
        source = (((token // LOCAL * 2 + kv) * LAYERS + layer) * LOCAL + token % LOCAL) * DIM + element % DIM
    else:
        source = ((layer // LOCAL * 2 + kv) * LOCAL + layer % LOCAL) * (TOKENS * DIM) + element
    destination = tl.cast(tl.load(destinations + kv_layer), tl.pointer_type(tl.int16))
    tl.store(destination + START * DIM + element, tl.load(gathered + source, mask=mask), mask=mask)


def load_patch_embedding(model, path=None, device="cuda"):
    """The DiT's ``patch_embedding`` Conv3d (only these two tensors are read from ``path``)."""
    dim = WAN_CONFIGS[MODEL_DEFAULTS[model]["wan"]].dim
    conv = torch.nn.Conv3d(16, dim, (1, 2, 2), stride=(1, 2, 2))
    if path is not None:
        if str(path).endswith(".safetensors"):
            from safetensors import safe_open
            with safe_open(str(path), "pt") as f:
                conv.weight.data = f.get_tensor("patch_embedding.weight")
                conv.bias.data = f.get_tensor("patch_embedding.bias")
        else:
            state = load_state_dict(path, "generator_ema")
            conv.load_state_dict({k.split("patch_embedding.")[1]: v for k, v in state.items()
                                  if "patch_embedding." in k}, strict=True)
    return conv.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)


class KVRefineRuntime:
    """Static-buffer KV refinement of the last frame slot of a full cache."""

    def __init__(self, net, patch, geometry: Geometry, timestep, *, autocast=False, contiguous_stem=False,
                 rank=0, world=1, shard="tokens", streams=1, compile_heads=True, cuda_graph=True):
        self.net, self.patch, self.geometry = net, patch, geometry
        self.rank, self.world, self.shard = rank, world, shard
        self.autocast, self.contiguous_stem = autocast, contiguous_stem
        device = patch.weight.device
        self.device = device
        self.dim, self.layers = net.dim, net.num_blocks
        tokens = geometry.tokens
        self.tokens = tokens
        self.live = slice(geometry.kv_len - tokens, geometry.kv_len)
        with self._autocast():
            self.emb = net.embed_timestep(timestep, 1, device, torch.bfloat16)
        # Noisy source cache (never modified by timed calls) and destination cache, laid out as
        # ``allocate_kv_cache``: per layer ``[1, frames * tokens, heads, head_dim]``.
        shape = (1, geometry.kv_len, net.num_heads, net.head_dim)
        self.source, self.dest = ([{"k": torch.zeros(shape, device=device, dtype=torch.bfloat16),
                                    "v": torch.zeros(shape, device=device, dtype=torch.bfloat16)}
                                   for _ in range(self.layers)] for _ in range(2))
        self.latent = torch.zeros(1, 1, 16, geometry.height, geometry.width, device=device, dtype=torch.bfloat16)
        self.staged = torch.empty_like(self.latent)
        if world > 1:
            if shard == "tokens":
                assert tokens % world == 0, (tokens, world)
                self.local = tokens // world
                self.gathered = torch.empty(world, 2, self.layers, self.local, self.dim, device=device,
                                            dtype=torch.bfloat16)
            else:
                assert self.layers % world == 0, (self.layers, world)
                self.local = self.layers // world
                self.gathered = torch.empty(world, 2, self.local, tokens, self.dim, device=device,
                                            dtype=torch.bfloat16)
        self.streams = [torch.cuda.Stream(device) for _ in range(streams)] if streams > 1 else []
        assert streams == 1 or (world == 1 and self.layers % streams == 0), "streams need 1 GPU and equal groups"
        opts = {"max_autotune": True, "coordinate_descent_tuning": True, "triton.cudagraphs": False,
                "emulate_precision_casts": True}
        self._heads = torch.compile(self._repair, fullgraph=True, dynamic=False, options=opts) \
            if compile_heads else self._repair
        self.cuda_graph = cuda_graph
        self.graph = None
        self._destinations = {}

    def _autocast(self):
        if self.autocast:
            return torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ stages
    def patch_tokens(self, x):
        """Exact channel-major GEMM form of the Conv3d patch embedding -> ``[1, tokens, dim]``."""
        h, w = self.geometry.grid
        cols = x[:, 0].reshape(1, 16, h, 2, w, 2).permute(0, 2, 4, 1, 3, 5).reshape(self.tokens, 64).T.contiguous()
        value = torch.mm(self.patch.weight.flatten(1), cols) + self.patch.bias[:, None]
        value = value.unsqueeze(0).transpose(1, 2)  # same values and strides as the Conv3d output view
        return value.contiguous() if self.contiguous_stem else value

    def _repair(self, phi, emb, layers, tokens, cache, out):
        """Repair ``layers x tokens`` of the live slot; write into ``out`` (a cache) or return BF16 K/V."""
        lo, hi = tokens.start + self.live.start, tokens.stop + self.live.start
        k = torch.stack([cache[i]["k"][:, lo:hi].reshape(hi - lo, self.dim) for i in range(layers.start, layers.stop)])
        v = torch.stack([cache[i]["v"][:, lo:hi].reshape(hi - lo, self.dim) for i in range(layers.start, layers.stop)])
        heads = self.net.heads_grouped
        gemm = heads.gemms[0]
        phi_v = heads.phi_proj[str(gemm.d_v)](phi[0, tokens]).unsqueeze(0).expand(k.shape[0], -1, -1)
        kr, vr = gemm(k, v, phi_v, emb, layers)
        if out is None:
            return torch.stack((kr.to(torch.bfloat16), vr.to(torch.bfloat16)))
        for j, i in enumerate(range(layers.start, layers.stop)):
            out[i]["k"][:, lo:hi].copy_(kr[j].view_as(out[i]["k"][:, lo:hi]))
            out[i]["v"][:, lo:hi].copy_(vr[j].view_as(out[i]["v"][:, lo:hi]))
        return None

    def forward(self, out=None):
        """One refinement; writes the repaired live slot into ``out`` (default: destination cache)."""
        out = self.dest if out is None else out
        all_layers, all_tokens = slice(0, self.layers), slice(0, self.tokens)
        with self._autocast():
            self.staged.copy_(self.latent)
            phi = self.net.compute_phi(self.patch_tokens(self.staged), self.emb)
            if self.world == 1 and not self.streams:
                self._heads(phi, self.emb, all_layers, all_tokens, self.source, out)
            elif self.world == 1:
                current, size = torch.cuda.current_stream(), self.layers // len(self.streams)
                for s, stream in enumerate(self.streams):
                    stream.wait_stream(current)
                    with torch.cuda.stream(stream):
                        self._heads(phi, self.emb, slice(s * size, (s + 1) * size), all_tokens, self.source, out)
                for stream in self.streams:
                    current.wait_stream(stream)
            else:
                r, n = self.rank, self.local
                if self.shard == "tokens":
                    local = self._heads(phi, self.emb, all_layers, slice(r * n, (r + 1) * n), self.source, None)
                else:
                    local = self._heads(phi, self.emb, slice(r * n, (r + 1) * n), all_tokens, self.source, None)
                dist.all_gather_into_tensor(self.gathered.view(-1), local.contiguous().view(-1))
                self._commit(out)

    def _commit(self, out):
        """Write the all-gathered repaired K/V into every layer's live slot of ``out``."""
        if id(out) not in self._destinations:  # built by the eager warm-up calls, before graph capture
            tensors = [c["k"] for c in out] + [c["v"] for c in out]
            assert all(t.is_contiguous() and t.dtype == torch.bfloat16 for t in tensors)
            table = torch.tensor([t.data_ptr() for t in tensors], dtype=torch.int64, device=self.device)
            self._destinations[id(out)] = (out, table)
        grid = (triton.cdiv(self.tokens * self.dim, COMMIT_BLOCK), 2 * self.layers)
        _commit_kernel[grid](self.gathered.view(torch.int16), self._destinations[id(out)][1], LAYERS=self.layers,
                             LOCAL=self.local, TOKENS=self.tokens, DIM=self.dim, START=self.live.start,
                             BY_TOKENS=self.shard == "tokens", BLOCK=COMMIT_BLOCK, num_warps=4)

    # ------------------------------------------------------------------ reference
    def reference(self):
        """The release pipeline's refinement (``FastFrames._refine``): Conv3d patch, all heads at once."""
        with self._autocast():
            stem = self.patch(self.latent.permute(0, 2, 1, 3, 4).contiguous()).flatten(2).transpose(1, 2)
            stem = stem.contiguous() if self.contiguous_stem else stem
            phi = self.net.compute_phi(stem.to(torch.bfloat16), self.emb)
            keys = torch.stack([c["k"][:, self.live].reshape(self.tokens, -1) for c in self.source])
            values = torch.stack([c["v"][:, self.live].reshape(self.tokens, -1) for c in self.source])
            k, v = self.net.repair_all(keys, values, phi, self.emb)
        return k.to(torch.bfloat16), v.to(torch.bfloat16)

    def committed(self, out=None):
        out = self.dest if out is None else out
        k = torch.stack([c["k"][:, self.live].reshape(self.tokens, -1) for c in out])
        v = torch.stack([c["v"][:, self.live].reshape(self.tokens, -1) for c in out])
        return k.clone(), v.clone()

    # ------------------------------------------------------------------ execution
    def fill(self, seed=0):
        g = torch.Generator(device=self.device).manual_seed(seed)
        self.latent.copy_(torch.randn(self.latent.shape, generator=g, device=self.device, dtype=torch.bfloat16))
        for c in self.source:
            for t in (c["k"], c["v"]):
                t.copy_(torch.randn(t.shape, generator=g, device=self.device, dtype=t.dtype))

    def prepare(self, barrier=lambda: None):
        for _ in range(3):
            self.forward()
        torch.cuda.synchronize()
        barrier()
        if self.cuda_graph:
            self.graph, _ = capture_graph(self.forward, before_capture=barrier)
            barrier()

    def __call__(self):
        if self.graph is not None:
            self.graph.replay()
        else:
            self.forward()


def profile_kv_refine(args, ctx, record):
    torch.backends.cudnn.benchmark = True
    defaults = MODEL_DEFAULTS[args.model]
    net = load_refiner(args.model, args.refiner, ctx.device)
    patch = load_patch_embedding(args.model, args.dit, ctx.device)
    geometry = Geometry.from_resolution(args.resolution, args.frames)
    timestep = defaults["timestep"] if args.timestep is None else args.timestep
    shard = args.shard
    runtime = KVRefineRuntime(net, patch, geometry, timestep, autocast=defaults["refiner_autocast"],
                              contiguous_stem=defaults["contiguous_stem"], rank=ctx.rank, world=ctx.world,
                              shard=shard, streams=args.streams,
                              compile_heads=not args.no_compile, cuda_graph=not args.no_cuda_graph)
    runtime.fill(seed=args.seed)
    started = time.perf_counter()
    runtime.prepare(barrier=ctx.barrier)
    record["prepare_seconds"] = time.perf_counter() - started
    # Numerical check of the timed path (graph replay) against the release pipeline's refinement.
    for c in runtime.dest:
        c["k"][:, runtime.live].fill_(float("nan"))
        c["v"][:, runtime.live].fill_(float("nan"))
    runtime()
    torch.cuda.synchronize()
    got = runtime.committed()
    ref = runtime.reference()
    checks = {"key": compare(got[0], ref[0]), "value": compare(got[1], ref[1]),
              "patch_gemm_vs_conv3d": compare(
                  runtime.patch_tokens(runtime.latent).contiguous(),
                  patch(runtime.latent.permute(0, 2, 1, 3, 4).contiguous()).flatten(2).transpose(1, 2).contiguous())}
    checks = ctx.gather(checks)
    record["settings"] = dict(timestep=timestep, autocast=defaults["refiner_autocast"], shard=shard if ctx.world > 1
                              else None, streams=args.streams, compiled_heads=not args.no_compile,
                              cuda_graph=runtime.graph is not None, tokens=geometry.tokens, kv_len=geometry.kv_len,
                              refiner_weights=args.refiner or "random", patch_weights=args.dit or "random",
                              parameters=sum(p.numel() for p in net.parameters()))
    record["check_vs_release_refinement"] = checks
    return runtime
