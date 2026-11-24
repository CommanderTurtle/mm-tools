#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Training of the KV refinement network (Sec. 3.3 and Appendix D of the paper).

The generator stays frozen. Every rollout first runs its *primed* autoregressive
rollout (the baseline, which re-runs the DiT on each frame's clean latent at
``t = 0``) and records, per latent frame and DiT layer,

* the noisy K/V written by the frame's final denoising step (network input),
* the clean K/V written by the ``t = 0`` pass (target),
* the post-RoPE queries of the final denoising step, and
* the patch-embedded clean latent (stem input).

Every frame slot ``1 .. F-2`` then gives one optimizer update (teacher forcing,
random slot order). The refined slot is compared with the clean one directly,
``lambda_k [(1 - cos) + relL2]_K + lambda_v [(1 - cos) + relL2]_V``, and through
attention: the outputs of the next frame and of one random later frame when they
read the refined instead of the clean slot (relative L2, averaged over layers and
queries). Frame 0 keeps its real clean pass at inference and is never refined.

The network is replicated on every rank. An FP32 network is optimised directly;
a BF16 network keeps FP32 master weights and AdamW moments (gradients are averaged
and clipped in FP32, the masters are rounded back after every update), so the
network trains in the dtype it is deployed in.

Output directory: ``metrics.jsonl``, resumable ``step_<N>/`` generations (same world
size required) and ``exports/step_<N>/refiner.safetensors`` (inference weights in the
network's dtype, loadable as ``FastFrames.from_pretrained(..., refiner=<file>)``).
"""
from __future__ import annotations

import contextlib
import copy
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from ..checkpoints import build_wan, load_state_dict
from ..models.kv_refinement import KVRefinementNetwork
from ..wan.attention import attention
from ..wan.causal_model import allocate_kv_cache
from ..wan.model import WAN_CONFIGS, causal_rope_apply
from ..wan.scheduler import FlowMatchScheduler
from ..wan.text_encoder import TextEncoder
from . import checkpoint
from .data import PromptStream, read_prompts
from .distributed import barrier, init_distributed, raise_if_any_failed
from .logging import MetricLogger

__all__ = ["Capture", "Rollout", "MasterAdamW", "build_refiner", "refine_cache", "slot_loss", "train"]

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
EVAL_FRAMES = (5, 10, 15)


def _dtype(module):
    return next(module.parameters()).dtype


# ---------------------------------------------------------------------------------------------- rollout
@dataclass
class Capture:
    """Teacher targets of one rollout. K/V/queries: ``[F, layers, tokens, heads, head_dim]`` (BF16)."""

    noisy_k: torch.Tensor
    noisy_v: torch.Tensor
    clean_k: torch.Tensor
    clean_v: torch.Tensor
    query: torch.Tensor  # frame 0 is zero: it reads no earlier slot
    stem: torch.Tensor   # [F, tokens, dim] patch-embedded clean latents

    @classmethod
    def empty(cls, num_frames, num_layers, frame_tokens, num_heads, head_dim, device):
        def kv():
            return torch.empty(num_frames, num_layers, frame_tokens, num_heads, head_dim,
                               dtype=torch.bfloat16, device=device)
        query = kv()
        query[0].zero_()
        return cls(kv(), kv(), kv(), kv(), query,
                   torch.empty(num_frames, frame_tokens, num_heads * head_dim, dtype=torch.bfloat16, device=device))


class Rollout:
    """Frame-wise autoregressive rollout of the frozen generator.

    Transition noise between the denoising steps of a frame is drawn from the global
    CUDA generator (``torch.randn_like``).
    """

    def __init__(self, dit, first_frame_timesteps, timesteps, refine_timestep, latent_size=(60, 104),
                 num_frames=21):
        self.dit = dit
        self.first_frame_timesteps = [float(t) for t in first_frame_timesteps]
        self.timesteps = [float(t) for t in timesteps]
        self.refine_timestep = float(refine_timestep)
        self.height, self.width = latent_size
        self.num_frames = num_frames
        self.frame_tokens = self.height * self.width // 4
        self.device = dit.patch_embedding.weight.device
        self.scheduler = FlowMatchScheduler(shift=5.0)

    def _predict(self, x, timestep, frame, text_cache, kv_cache):
        t = torch.full((1, 1), timestep, device=self.device, dtype=torch.float32)
        e0, e_head = self.dit.time_conditioning(t, torch.bfloat16)
        flow, _ = self.dit.forward_frame(x, e0, e_head, text_cache, kv_cache, frame)
        return self.scheduler.flow_to_x0(flow.flatten(0, 1), x.flatten(0, 1), t.flatten()).unflatten(0, flow.shape[:2])

    def stem_tokens(self, x0):
        """Clean latent ``[1, 1, 16, H, W]`` -> contiguous patch tokens ``[1, H*W/4, dim]``."""
        tokens = self.dit.patch_embedding(x0.permute(0, 2, 1, 3, 4).contiguous())
        return tokens.flatten(2).transpose(1, 2).contiguous()

    @contextlib.contextmanager
    def _record_queries(self, out, frame):
        """Store every layer's post-RoPE self-attention queries of this forward in ``out[layer]``."""
        if out is None:
            yield
            return
        grid = torch.tensor([[1, self.height // 2, self.width // 2]], dtype=torch.long)
        heads, head_dim = self.dit.num_heads, self.dit.dim // self.dit.num_heads

        def hook(layer):
            def store(module, inputs, q):
                q = q.view(1, self.frame_tokens, heads, head_dim)
                out[layer].copy_(causal_rope_apply(q, grid, self.dit.freqs, start_frame=frame)[0])
            return store

        handles = [block.self_attn.norm_q.register_forward_hook(hook(i)) for i, block in enumerate(self.dit.blocks)]
        try:
            yield
        finally:
            for handle in handles:
                handle.remove()

    @torch.no_grad()
    def run(self, noise, prompt_embeds, refiner=None, prime=True, record=None):
        """Latents ``[1, F, 16, H, W]``.

        After each frame its cache slot is overwritten by the clean ``t = 0`` pass
        (``prime``), except that frames ``>= 1`` are refined by ``refiner`` when given;
        ``prime=False`` without a refiner leaves every slot noisy. ``record`` (a
        ``Capture``) receives the training targets of a primed rollout.
        """
        text_cache = self.dit.text_cache(prompt_embeds)
        kv_cache = allocate_kv_cache(self.dit, self.num_frames, self.frame_tokens, device=self.device)
        latents = torch.empty_like(noise)
        for frame in range(self.num_frames):
            schedule = self.first_frame_timesteps if frame == 0 else self.timesteps
            x = noise[:, frame:frame + 1]
            for step, timestep in enumerate(schedule):
                last = step + 1 == len(schedule)
                queries = record.query[frame] if record is not None and last and frame > 0 else None
                with self._record_queries(queries, frame):
                    x0 = self._predict(x, timestep, frame, text_cache, kv_cache)
                if not last:
                    next_t = torch.full((1,), schedule[step + 1], device=self.device, dtype=torch.float32)
                    flat = x0.flatten(0, 1)
                    x = self.scheduler.add_noise(flat, torch.randn_like(flat), next_t).unflatten(0, x0.shape[:2])
            latents[:, frame:frame + 1] = x0
            start, end = frame * self.frame_tokens, (frame + 1) * self.frame_tokens
            if record is not None:
                for layer, cache in enumerate(kv_cache):
                    record.noisy_k[frame, layer] = cache["k"][0, start:end]
                    record.noisy_v[frame, layer] = cache["v"][0, start:end]
                record.stem[frame] = self.stem_tokens(x0)[0]
            if refiner is not None and frame > 0:
                refine_cache(refiner, self.refine_timestep, self.stem_tokens(x0), kv_cache, start, end)
            elif prime:
                self._predict(x0, 0.0, frame, text_cache, kv_cache)
            if record is not None:
                for layer, cache in enumerate(kv_cache):
                    record.clean_k[frame, layer] = cache["k"][0, start:end]
                    record.clean_v[frame, layer] = cache["v"][0, start:end]
        return latents

    def capture(self, noise, prompt_embeds):
        cfg = self.dit.config
        record = Capture.empty(self.num_frames, cfg.num_layers, self.frame_tokens, cfg.num_heads,
                               cfg.dim // cfg.num_heads, self.device)
        self.run(noise, prompt_embeds, record=record)
        return record


# ---------------------------------------------------------------------------------------------- network
def build_refiner(wan_name, settings, device):
    """A from-scratch refinement network (``torch.manual_seed`` before calling fixes its init)."""
    wan = WAN_CONFIGS[wan_name]
    network = KVRefinementNetwork(
        dim=wan.dim, num_heads=wan.num_heads, head_dim=wan.dim // wan.num_heads, num_blocks=wan.num_layers,
        d_a=settings.d_a, n_stem=settings.n_stem, stem_n_heads=settings.stem_n_heads,
        stem_mlp_mult=settings.stem_mlp_mult, head_mlp_mult=settings.head_mlp_mult)
    return network.to(device, DTYPES[settings.dtype])


def refine_cache(refiner, timestep, stem, kv_cache, start, end):
    """Replace cache slot ``[start, end)`` of every layer by its refinement (deployed path)."""
    dtype = _dtype(refiner)
    e = refiner.embed_timestep(timestep, 1, stem.device, dtype)
    phi = refiner.compute_phi(stem.to(dtype), e)
    tokens = end - start
    keys = torch.stack([c["k"][:, start:end].reshape(tokens, -1) for c in kv_cache]).to(dtype)
    values = torch.stack([c["v"][:, start:end].reshape(tokens, -1) for c in kv_cache]).to(dtype)
    keys, values = refiner.repair_all(keys, values, phi, e)
    for layer in refiner.repaired_block_indices:
        slot = kv_cache[layer]["k"][:, start:end]
        kv_cache[layer]["k"][:, start:end] = keys[layer].reshape(slot.shape).to(slot.dtype)
        kv_cache[layer]["v"][:, start:end] = values[layer].reshape(slot.shape).to(slot.dtype)


# ---------------------------------------------------------------------------------------------- losses
def _relative_l2(x, ref, bf16_residual=False):
    residual = (x - ref).float() if bf16_residual else x.float() - ref.float()
    return residual.norm() / (ref.float().norm() + 1e-8)


def refine_slot(refiner, timestep, cap, frame):
    """``{layer: (K, V)}`` of the refined slot ``frame``, each ``[1, tokens, heads, head_dim]``."""
    dtype = _dtype(refiner)
    num_layers, tokens, heads, head_dim = cap.noisy_k.shape[1:]
    e = refiner.embed_timestep(timestep, 1, cap.stem.device, dtype)
    phi = refiner.compute_phi(cap.stem[frame].unsqueeze(0).to(dtype), e)
    keys = cap.noisy_k[frame].to(dtype).reshape(num_layers, tokens, -1)
    values = cap.noisy_v[frame].to(dtype).reshape(num_layers, tokens, -1)
    keys, values = refiner.repair_all(keys, values, phi, e)
    return {layer: (keys[layer].reshape(1, tokens, heads, head_dim), values[layer].reshape(1, tokens, heads, head_dim))
            for layer in refiner.repaired_block_indices}


def _read(cap, layer, query_frame, dtype, slot=None, k=None, v=None):
    """K/V read by ``query_frame``: clean frames before it (slot ``slot`` replaced) and its own noisy slot."""
    keys = [cap.clean_k[:query_frame, layer], cap.noisy_k[query_frame:query_frame + 1, layer]]
    values = [cap.clean_v[:query_frame, layer], cap.noisy_v[query_frame:query_frame + 1, layer]]
    if slot is not None:
        keys[0:1] = [cap.clean_k[:slot, layer], k, cap.clean_k[slot + 1:query_frame, layer]]
        values[0:1] = [cap.clean_v[:slot, layer], v, cap.clean_v[slot + 1:query_frame, layer]]
    heads, head_dim = cap.clean_k.shape[-2:]
    return (torch.cat([u.to(dtype) for u in keys]).reshape(1, -1, heads, head_dim),
            torch.cat([u.to(dtype) for u in values]).reshape(1, -1, heads, head_dim))


def attention_loss(reps, cap, frame, query_frames, bf16_residual=False):
    """Relative L2 between the attention outputs over the refined and the clean slot ``frame``."""
    total, count = 0.0, 0
    for layer, (k, v) in reps.items():
        for q in query_frames:
            query = cap.query[q:q + 1, layer]
            target = attention(query, *_read(cap, layer, q, k.dtype))
            pred = attention(query, *_read(cap, layer, q, k.dtype, frame, k, v))
            total = total + _relative_l2(pred, target, bf16_residual)
            count += 1
    return total / count


def kv_loss(reps, cap, frame, lambda_k, lambda_v):
    """``lambda_k [(1 - cos) + relL2]_K + lambda_v [(1 - cos) + relL2]_V`` averaged over layers (FP32)."""
    k_cos = k_l2 = v_cos = v_l2 = 0.0
    for layer, (k, v) in reps.items():
        k, v = k.float(), v.float()
        target_k = cap.clean_k[frame, layer].unsqueeze(0).float()
        target_v = cap.clean_v[frame, layer].unsqueeze(0).float()
        kc = 1.0 - F.cosine_similarity(k.flatten(0, 2), target_k.flatten(0, 2), dim=-1).mean()
        vc = 1.0 - F.cosine_similarity(v.flatten(0, 2), target_v.flatten(0, 2), dim=-1).mean()
        kl, vl = _relative_l2(k, target_k), _relative_l2(v, target_v)
        k_cos, k_l2, v_cos, v_l2 = k_cos + kc, k_l2 + kl, v_cos + vc, v_l2 + vl
    k_term, v_term = (k_cos + k_l2) / len(reps), (v_cos + v_l2) / len(reps)
    return lambda_k * k_term + lambda_v * v_term, k_term, v_term


def slot_loss(refiner, timestep, cap, frame, query_frames, *, kv_weight, lambda_k, lambda_v,
              bf16_attention_residual=False):
    """``L_attn + kv_weight * L_KV`` for refining ``frame``; returns ``(loss, metrics)``."""
    reps = refine_slot(refiner, timestep, cap, frame)
    attn = attention_loss(reps, cap, frame, query_frames, bf16_attention_residual)
    kv, k_term, v_term = kv_loss(reps, cap, frame, lambda_k, lambda_v)
    loss = attn + kv_weight * kv
    return loss, {"loss": loss.detach(), "attn_relL2": attn.detach(), "kv_K": k_term.detach(),
                  "kv_V": v_term.detach()}


# ---------------------------------------------------------------------------------------------- optimizer
class MasterAdamW:
    """AdamW on FP32 masters of a replicated network (the network itself when it is FP32).

    ``step`` averages the gradients over ranks and clips their global norm in FP32;
    a BF16 network then receives the rounded masters.
    """

    def __init__(self, network, *, lr, betas, weight_decay, max_grad_norm):
        self.network = network
        self.master = network if _dtype(network) == torch.float32 else copy.deepcopy(network).float()
        self.max_grad_norm = float(max_grad_norm)
        self.optimizer = torch.optim.AdamW(self.master.parameters(), lr=float(lr), betas=tuple(betas),
                                           weight_decay=float(weight_decay))

    def zero_grad(self):
        self.network.zero_grad(set_to_none=True)
        self.master.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self):
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        live, masters = list(self.network.parameters()), list(self.master.parameters())
        grads = [p.grad.float() for p in live]  # the gradients themselves for an FP32 network
        if world_size > 1:
            for g in grads:
                dist.all_reduce(g)
                g.div_(world_size)
        if self.master is self.network:
            norm = torch.nn.utils.clip_grad_norm_(masters, self.max_grad_norm)
        else:
            norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads]))
            coef = torch.clamp(torch.as_tensor(self.max_grad_norm, dtype=torch.float32, device=norm.device)
                               / (norm + 1e-6), max=1.0)
            for master, g in zip(masters, grads):
                master.grad = g.mul_(coef)
        self.optimizer.step()
        self.sync_live()
        return norm

    @torch.no_grad()
    def sync_live(self):
        if self.master is not self.network:
            for p, master in zip(self.network.parameters(), self.master.parameters()):
                p.copy_(master.to(p.dtype))


# ---------------------------------------------------------------------------------------------- training
def _save_safetensors(path, state):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: v.detach().cpu().contiguous() for k, v in state.items()}, str(tmp))
    os.replace(tmp, path)


class Trainer:

    def __init__(self, config):
        self.config = config
        self.ctx = init_distributed()
        self.device = self.ctx.device
        torch.backends.cuda.matmul.allow_tf32 = bool(config.train.allow_tf32)
        torch.backends.cudnn.allow_tf32 = True
        self.seed = int(config.seed)
        self.out = Path(config.output_dir)
        loss = config.loss
        self.loss_args = dict(kv_weight=float(loss.kv_weight), lambda_k=float(loss.lambda_k),
                              lambda_v=float(loss.lambda_v),
                              bf16_attention_residual=bool(loss.get("bf16_attention_residual", False)))
        self.far_query = bool(loss.far_query)

        model, rollout = config.model, config.rollout
        dit = build_wan(model.name, load_state_dict(model.generator, model.get("generator_key")), device=self.device)
        self.rollout = Rollout(dit, rollout.first_frame_timesteps, rollout.timesteps, rollout.refine_timestep,
                               tuple(rollout.latent_size), int(rollout.num_frames))
        self.num_frames = self.rollout.num_frames
        self.text_encoder = TextEncoder(Path(model.checkpoint_root) / model.name).to(self.device, torch.bfloat16)
        self.offload_text_encoder = bool(config.train.get("offload_text_encoder", False))

        torch.manual_seed(self.seed)
        self.refiner = build_refiner(model.name, config.refiner, self.device)
        if config.refiner.get("init_from"):  # weights-only warm start; the update counter starts at 0
            self.refiner.load_state_dict(load_state_dict(config.refiner.init_from, "adapter"), strict=True)
        if dist.is_initialized():
            for p in self.refiner.parameters():
                dist.broadcast(p.data, src=0)
        optim = config.optimizer
        self.optimizer = MasterAdamW(self.refiner, lr=optim.lr, betas=optim.betas,
                                     weight_decay=optim.weight_decay, max_grad_norm=optim.max_grad_norm)
        torch.manual_seed(self.seed + self.ctx.rank)  # transition noise of the rollouts

        data = config.data
        self.prompts = PromptStream(read_prompts(data.prompts), self.ctx.rank, self.ctx.world_size,
                                    seed=int(data.get("shuffle_seed", 0)))
        self.eval_prompts = read_prompts(data.eval_prompts)[:int(data.num_eval_prompts)]
        self.eval_seed = int(data.eval_seed)
        self.step = self.saved_step = 0
        step, rank_state = checkpoint.load(self.out, *self._state())
        if step is not None:
            self.step = self.saved_step = step
            self.prompts.load_state_dict(rank_state["prompts"])
            torch.set_rng_state(rank_state["cpu_rng"])
            torch.cuda.set_rng_state(rank_state["cuda_rng"], self.device)
            self.optimizer.sync_live()
        self.logger = MetricLogger(config, self.out, self.ctx.is_main)
        if self.ctx.is_main:
            params = sum(p.numel() for p in self.refiner.parameters())
            print(f"[kv_refinement] {params:,} parameters ({_dtype(self.refiner)}), world size "
                  f"{self.ctx.world_size}, starting at update {self.step}", flush=True)

    def _state(self):
        master, optimizer = self.optimizer.master, self.optimizer.optimizer
        return {"refiner": master}, {"refiner": (master, optimizer)}

    @torch.no_grad()
    def encode(self, prompts):
        if self.offload_text_encoder:
            self.text_encoder.to(self.device)
        embeds = self.text_encoder(prompts)["prompt_embeds"]
        if self.offload_text_encoder:
            self.text_encoder.to("cpu")
            torch.cuda.empty_cache()
        return embeds

    def noise(self, seed):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        shape = (1, self.num_frames, 16, self.rollout.height, self.rollout.width)
        return torch.randn(shape, generator=generator, device=self.device, dtype=torch.bfloat16)

    def train_rollout(self, max_updates):
        """One prompt: capture its targets, then one update per frame slot in random order."""
        start = time.time()
        rank = self.ctx.rank
        error = cap = None
        try:
            embeds = self.encode(next(self.prompts))
            cap = self.rollout.capture(self.noise(self.seed + rank * 7919 + self.step), embeds)
            torch.cuda.empty_cache()  # return the rollout's KV cache before the updates
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"rollout before update {self.step}")
        frames = list(range(1, self.num_frames - 1))
        order = random.Random(self.step * 100003 + rank)
        order.shuffle(frames)
        for frame in frames:
            if self.step >= max_updates:
                break
            queries = [frame + 1]  # the next frame, plus one random later frame
            if self.far_query and frame + 2 <= self.num_frames - 1:
                queries.append(order.randint(frame + 2, self.num_frames - 1))
            loss, metrics = slot_loss(self.refiner, self.rollout.refine_timestep, cap, frame, queries,
                                      **self.loss_args)
            self.optimizer.zero_grad()
            loss.backward()
            metrics["grad_norm"] = self.optimizer.step()
            metrics["frame"] = frame
            self.logger.log(metrics, self.step)
            if self.ctx.is_main and self.step % int(self.config.train.print_every) == 0:
                print(f"[update {self.step}] frame {frame} loss {metrics['loss'].item():.4f} "
                      f"attn {metrics['attn_relL2'].item():.4f} kv_V {metrics['kv_V'].item():.3f} "
                      f"grad_norm {metrics['grad_norm'].item():.3f}", flush=True)
            self.step += 1
        del cap
        torch.cuda.empty_cache()
        if self.ctx.is_main:
            print(f"[update {self.step}] rollout took {time.time() - start:.1f}s, peak memory "
                  f"{torch.cuda.max_memory_reserved(self.device) / 2**30:.1f} GiB", flush=True)

    @torch.no_grad()
    def _eval_prompt(self, index):
        embeds = self.encode([self.eval_prompts[index]])
        noise = self.noise(self.eval_seed + index)
        cap = self.rollout.capture(noise, embeds)
        frames = [f for f in EVAL_FRAMES if f < self.num_frames - 1]
        t, bf16 = self.rollout.refine_timestep, self.loss_args["bf16_attention_residual"]
        tf = sum(attention_loss(refine_slot(self.refiner, t, cap, f), cap, f, [f + 1], bf16).item()
                 for f in frames) / max(len(frames), 1)
        del cap
        last = []
        for kwargs in ({}, {"prime": False}, {"refiner": self.refiner}):
            # Use the same transition noise in all three variants. This also reseeds
            # the training noise stream of the evaluating ranks.
            torch.manual_seed(self.eval_seed + index * 9176 + 1)
            last.append(self.rollout.run(noise, embeds, **kwargs)[:, -1].float())
        primed, noisy, refined = last
        rel = [((x - primed).norm() / (primed.norm() + 1e-8)).item() for x in (refined, noisy)]
        return torch.tensor([tf, *rel], device=self.device)

    def evaluate(self):
        """Held-out teacher-forced attention relL2 and last-latent relL2 of refined / unrefined rollouts
        against the primed one (one prompt per rank, first ``num_eval_prompts`` prompts)."""
        n_eval = min(len(self.eval_prompts), self.ctx.world_size)
        stats, error = torch.zeros(3, device=self.device), None
        try:
            if self.ctx.rank < n_eval:
                stats = self._eval_prompt(self.ctx.rank)
        except Exception as exc:  # noqa: BLE001 - reported collectively
            error = exc
        raise_if_any_failed(error, f"evaluation at update {self.step}")
        if dist.is_initialized():
            dist.all_reduce(stats)
        tf, refined, unrefined = (stats / n_eval).tolist()
        if self.ctx.is_main:
            print(f"[eval {self.step}] teacher-forced attn relL2 {tf:.4f} | last latent relL2 vs primed: "
                  f"refined {refined:.4f}, unrefined {unrefined:.4f}", flush=True)
        self.logger.log({"eval/tf_attn_relL2": tf, "eval/refined_relL2": refined,
                         "eval/unrefined_relL2": unrefined}, self.step)

    def export(self):
        path = self.out / "exports" / f"step_{self.step:08d}" / "refiner.safetensors"
        error = None
        if self.ctx.is_main:
            try:
                _save_safetensors(path, self.refiner.state_dict())
                print(f"[kv_refinement] exported {path}", flush=True)
            except Exception as exc:  # noqa: BLE001 - reported collectively
                error = exc
        raise_if_any_failed(error, f"exporting weights at update {self.step}")

    def save(self):
        rank_state = {"prompts": self.prompts.state_dict(), "cpu_rng": torch.get_rng_state(),
                      "cuda_rng": torch.cuda.get_rng_state(self.device)}
        checkpoint.save(self.out, self.step, *self._state(), rank_state=rank_state,
                        keep=int(self.config.train.keep_checkpoints), is_main=self.ctx.is_main)
        self.saved_step = self.step

    def fit(self):
        train = self.config.train
        max_updates, export_every, save_every = int(train.max_updates), int(train.export_every), int(train.save_every)
        next_export = 0 if self.step == 0 else (self.step // export_every + 1) * export_every
        next_save = (self.step // save_every + 1) * save_every
        while self.step < max_updates:
            if self.step >= next_export:  # held-out evaluation (also before training) + weight export
                self.evaluate()
                if self.step > 0:
                    self.export()
                next_export = (self.step // export_every + 1) * export_every
            if self.step >= next_save:  # after the evaluation, which reseeds the RNG of evaluating ranks
                self.save()
                next_save = (self.step // save_every + 1) * save_every
            self.train_rollout(max_updates)
            barrier()
        self.evaluate()
        self.export()
        if self.step != self.saved_step:
            self.save()
        self.logger.finish()


def train(config):
    """``scripts/train.py`` entry point."""
    Trainer(config).fit()
