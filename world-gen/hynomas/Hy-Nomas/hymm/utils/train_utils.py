"""Shared utilities for all training modes.

Imported by train.py and hymm/trainers/*.py — kept separate to avoid
circular imports.
"""

import math
import os

import torch
import torch.nn as nn

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    DTensor = None


# ============================================================================
# Adaptive Gradient Clipping
# ============================================================================


class AdaptiveGradClipper:
    """Adaptive gradient clipping based on EMA of gradient norms.

    clip_value = max(min_clip, factor * grad_norm_ema)

    When EMA is not yet initialized (first step), falls back to
    ``fallback_max_grad_norm``.
    """

    def __init__(
        self,
        factor: float = 5.0,
        min_clip: float = 0.5,
        ema_decay: float = 0.99,
    ):
        self.factor = factor
        self.min_clip = min_clip
        self.ema_decay = ema_decay
        self._grad_norm_ema = None

    def get_clip_value(self) -> float:
        """Return the current adaptive clip threshold."""
        if self._grad_norm_ema is None:
            return self.min_clip
        return max(self.min_clip, self.factor * self._grad_norm_ema)

    def update(self, grad_norm: float):
        """Update the EMA with the latest (pre-clip) grad norm."""
        if not (torch.isfinite(torch.tensor(grad_norm)) and grad_norm > 0):
            return
        if self._grad_norm_ema is None:
            self._grad_norm_ema = grad_norm
        else:
            self._grad_norm_ema = (
                self.ema_decay * self._grad_norm_ema + (1 - self.ema_decay) * grad_norm
            )

    @property
    def grad_norm_ema(self):
        return self._grad_norm_ema


# Module-level singleton — shared across all call sites within a training run.
_adaptive_clipper: AdaptiveGradClipper = None


def init_adaptive_grad_clipper(args) -> None:
    """Initialize the global adaptive grad clipper from config.

    Call once at training startup (after config is parsed).
    """
    global _adaptive_clipper
    _adaptive_clipper = AdaptiveGradClipper(
        factor=getattr(args.optimizer, "adaptive_grad_clip_factor", 5.0),
        min_clip=getattr(args.optimizer, "adaptive_grad_clip_min", 0.5),
        ema_decay=getattr(args.optimizer, "adaptive_grad_clip_ema_decay", 0.99),
    )


def get_clip_value(args) -> float:
    """Compute the effective gradient clip value.

    If adaptive clipping is enabled and initialized, returns
    ``max(min_clip, factor * EMA)``.  Otherwise returns the fixed
    ``max_grad_norm``.
    """
    enable_clip = getattr(args.optimizer, "enable_grad_norm_clip", True)
    if not enable_clip:
        return float("inf")

    adaptive = getattr(args.optimizer, "adaptive_grad_clip", False)
    if adaptive and _adaptive_clipper is not None:
        return _adaptive_clipper.get_clip_value()

    return getattr(args.optimizer, "max_grad_norm", 1.0)


def update_adaptive_grad_clipper(grad_norm: float) -> None:
    """Update the adaptive clipper EMA after each step."""
    if _adaptive_clipper is not None:
        _adaptive_clipper.update(grad_norm)


# ============================================================================
# Video / Text Encoding Utilities
# ============================================================================


def encode_video(vae, video_rgb, device):
    """Encode video to latent space (fallback for on-the-fly VAE encoding)."""
    with torch.no_grad():
        video_input = (video_rgb * 2.0 - 1.0).to(device)
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        video_latent = vae.encode(video_input).latent_dist.mode()
        video_latent = video_latent * vae.config.scaling_factor
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
    return video_latent


def encode_text_from_ids(
    text_encoder,
    text_ids,
    text_mask,
    device,
    data_type="video",
    prompt_batch_size=0,
):
    """Encode pre-tokenized text to embeddings (supports 2D and 3D input)."""
    with torch.no_grad():
        text_ids = text_ids.to(device)
        text_mask = text_mask.to(device)

        if text_ids.dim() == 3:
            B, N, L = text_ids.shape
            text_ids_flat = text_ids.reshape(B * N, L)
            text_mask_flat = text_mask.reshape(B * N, L)
            prompt_batch_size = int(prompt_batch_size or 0)

            if prompt_batch_size > 0:
                valid_rows = text_mask_flat.bool().any(dim=-1)
                valid_indices = valid_rows.nonzero(as_tuple=False).flatten()
                if valid_indices.numel() == 0:
                    raise ValueError(
                        "Multi-prompt text encoding found no valid prompt rows."
                    )

                encoded_states = None
                encoded_masks = None
                for start in range(0, valid_indices.numel(), prompt_batch_size):
                    chunk_indices = valid_indices[start : start + prompt_batch_size]
                    text_outputs = text_encoder.encode(
                        {
                            "input_ids": text_ids_flat.index_select(0, chunk_indices),
                            "attention_mask": text_mask_flat.index_select(
                                0, chunk_indices
                            ),
                        },
                        data_type=data_type,
                    )
                    chunk_states = text_outputs.hidden_state.to(dtype=torch.bfloat16)
                    chunk_masks = text_outputs.attention_mask.to(dtype=torch.bfloat16)

                    if encoded_states is None:
                        S, C = chunk_states.shape[1], chunk_states.shape[2]
                        encoded_states = torch.zeros(
                            B * N,
                            S,
                            C,
                            device=chunk_states.device,
                            dtype=chunk_states.dtype,
                        )
                        encoded_masks = torch.zeros(
                            B * N,
                            S,
                            device=chunk_masks.device,
                            dtype=chunk_masks.dtype,
                        )

                    encoded_states.index_copy_(0, chunk_indices, chunk_states)
                    encoded_masks.index_copy_(0, chunk_indices, chunk_masks)

                return (
                    encoded_states.reshape(B, N, S, C),
                    encoded_masks.reshape(B, N, S),
                )

            text_outputs = text_encoder.encode(
                {"input_ids": text_ids_flat, "attention_mask": text_mask_flat},
                data_type=data_type,
            )
            ts = text_outputs.hidden_state.to(dtype=torch.bfloat16)
            tm = text_outputs.attention_mask.to(dtype=torch.bfloat16)
            S, C = ts.shape[1], ts.shape[2]
            return ts.reshape(B, N, S, C), tm.reshape(B, N, S)
        else:
            text_outputs = text_encoder.encode(
                {"input_ids": text_ids, "attention_mask": text_mask},
                data_type=data_type,
            )
            return (
                text_outputs.hidden_state.to(dtype=torch.bfloat16),
                text_outputs.attention_mask.to(dtype=torch.bfloat16),
            )


# ============================================================================
# Backward + Optimizer Step
# ============================================================================


def _dump_grad_snapshot_if_requested(model, global_step, logger):
    """Dump deterministic bounded local-gradient summaries before clipping."""
    output_dir = os.environ.get("HY_DEBUG_GRAD_SNAPSHOT_DIR", "").strip()
    if not output_dir:
        return

    rank = int(os.environ.get("RANK", 0))
    sample_count = max(1, int(os.environ.get("HY_DEBUG_GRAD_SAMPLE_COUNT", "1024")))
    os.makedirs(output_dir, exist_ok=True)

    records = []
    with torch.no_grad():
        for name, param in model.named_parameters():
            grad = param.grad
            if grad is None:
                continue
            if DTensor is not None and isinstance(grad, DTensor):
                grad = grad.to_local()
            flat = grad.detach().reshape(-1)
            numel = flat.numel()
            if numel == 0:
                continue
            grad_f = flat.float()
            grad_d = flat.double()
            count = min(sample_count, numel)
            if count == numel:
                sampled = grad_f
            elif count == 1:
                sampled = grad_f[:1]
            else:
                idx = (
                    torch.arange(count, device=flat.device, dtype=torch.int64)
                    * (numel - 1)
                    // (count - 1)
                )
                sampled = grad_f.index_select(0, idx)
            finite = torch.isfinite(grad_d)
            records.append(
                {
                    "name": name,
                    "shape": tuple(grad.shape),
                    "numel": numel,
                    "dtype": str(grad.dtype),
                    "finite_count": int(finite.sum().item()),
                    "norm_sq": float(torch.sum(grad_d * grad_d).item()),
                    "max_abs": float(torch.max(torch.abs(grad_d)).item()),
                    "sample": sampled.cpu(),
                }
            )

    destination = os.path.join(
        output_dir, f"grad_snapshot_step{global_step}_rank{rank}.pt"
    )
    torch.save(
        {
            "rank": rank,
            "step": int(global_step),
            "sample_count": sample_count,
            "records": records,
        },
        destination,
    )
    logger.info(
        f"[grad-debug] Saved {len(records)} parameter gradient records to {destination}"
    )


def _finish_activation_offload_backward(model, logger, label=None):
    """Release activation-offload leases after one completed backward graph."""
    offload_manager = getattr(model, "_activation_offload_manager", None)
    if offload_manager is None or not offload_manager.enabled:
        return None
    stats = offload_manager.finish_step(assert_all_consumed=True)
    prefix = f"[activation-offload:{label}]" if label else "[activation-offload]"
    logger.info(
        f"{prefix} backward finalized: "
        f"peak_live_handles={stats['peak_live_handles']} "
        f"peak_pinned_gib={stats['peak_pinned_bytes'] / 2**30:.3f} "
        f"peak_active_pinned_gib={stats['peak_active_pinned_bytes'] / 2**30:.3f} "
        f"offloaded_gib={stats['offloaded_bytes_total'] / 2**30:.3f} "
        f"restored_gib={stats['restored_bytes_total'] / 2**30:.3f} "
        f"by_kind={stats['offloaded_bytes_by_kind']} "
        f"handles_by_forward={stats['handles_by_forward']} "
        f"slot_reuse={stats['slot_reuse_count']} "
        f"sync_fallback={stats['sync_fallback_count']}"
    )
    return stats


def common_backward_only(
    loss,
    model,
    args,
    logger,
    *,
    label=None,
    is_last_backward=True,
):
    """Accumulate one loss contribution without clipping or stepping.

    This is used by memory-bounded training modes that complete each independent
    autograd graph immediately, then perform one shared optimizer finalization
    after all contributions have accumulated into ``param.grad``.
    """
    # FSDP2 needs to know whether more independent backward graphs will
    # contribute to the same optimizer step. For non-final contributions,
    # defer gradient synchronization/final cleanup; the final backward reduces
    # the accumulated gradients and returns parameters to their sharded state.
    set_last = getattr(model, "set_is_last_backward", None)
    if callable(set_last):
        set_last(bool(is_last_backward))
    set_sync = getattr(model, "set_requires_gradient_sync", None)
    if callable(set_sync):
        set_sync(bool(is_last_backward), recurse=True)

    gradient_accumulation_steps = int(args.parallelism.gradient_accumulation_steps)
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    loss.backward()
    return _finish_activation_offload_backward(model, logger, label=label)


def common_finalize_backward_and_step(
    model,
    optimizer,
    lr_scheduler,
    global_step,
    args,
    logger,
    pre_clip_callback=None,
    reshard_fsdp_after_backward=False,
):
    """Clip and step after one or more backward calls have accumulated grads."""
    gradient_accumulation_steps = int(args.parallelism.gradient_accumulation_steps)
    is_update_step = (global_step + 1) % gradient_accumulation_steps == 0
    grad_norm = 0.0
    if is_update_step:
        if reshard_fsdp_after_backward:
            # Multiple independent FSDP2 backward graphs may leave some nested
            # modules' parameters unsharded even after the final contribution.
            # Restore the normal post-backward parameter layout before gradient
            # snapshots, clipping, EMA, or the next forward.
            for module in reversed(list(model.modules())):
                reshard = getattr(module, "reshard", None)
                if callable(reshard):
                    reshard()
        _dump_grad_snapshot_if_requested(model, global_step, logger)
        if pre_clip_callback is not None:
            try:
                pre_clip_callback(model)
            except Exception as _cb_err:
                logger.warning(
                    f"pre_clip_callback raised {type(_cb_err).__name__}: {_cb_err}"
                )
        clip_value = get_clip_value(args)
        grad_norm_ = nn.utils.clip_grad_norm_(
            model.parameters(), clip_value, foreach=True
        )
        grad_norm = grad_norm_.item()
        if (
            math.isfinite(clip_value)
            and math.isfinite(grad_norm)
            and grad_norm > clip_value
        ):
            logger.warning(
                f"⚠️  grad_norm clipped at step {global_step}: "
                f"grad_norm={grad_norm:.6f} > clip_value={clip_value:.6f}"
            )
        update_adaptive_grad_clipper(grad_norm)
        skip_optimizer_step = os.environ.get(
            "HY_DEBUG_SKIP_OPTIMIZER_STEP", ""
        ).strip().lower() in ("1", "true", "yes")
        if skip_optimizer_step:
            logger.info(f"[grad-debug] Skipping optimizer.step() at step {global_step}")
        elif not (torch.isnan(grad_norm_) or torch.isinf(grad_norm_)):
            optimizer.step()
        else:
            logger.warning(f"NaN/Inf gradient norm detected at step {global_step}")
        lr_scheduler.step()
        optimizer.zero_grad()
    return grad_norm


def common_backward_and_step(
    loss,
    model,
    optimizer,
    lr_scheduler,
    global_step,
    args,
    logger,
    pre_clip_callback=None,
):
    """Shared one-shot backward, clipping, and optimizer step logic.

    Args:
        pre_clip_callback: optional callable ``fn(model)`` invoked on update
            steps after ``loss.backward()`` and BEFORE ``clip_grad_norm_``.
            Useful for capturing per-subset grad/param stats while grads are
            still alive (they get zeroed at the end of this function).
    """
    common_backward_only(loss, model, args, logger)
    return common_finalize_backward_and_step(
        model,
        optimizer,
        lr_scheduler,
        global_step,
        args,
        logger,
        pre_clip_callback=pre_clip_callback,
    )
