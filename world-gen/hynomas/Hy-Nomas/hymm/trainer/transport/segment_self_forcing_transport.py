"""Segment serial self-forcing transport (non-DMD).

Trains the DiT so that when it denoises video segment ``i``, its attention
context (KV cache) normally contains the model's own previously-generated
latents for segments ``0..i-1``.  With ``df_random_prefix`` enabled, a random
number ``K`` of leading segments are instead prefetched from noisy GT latents,
then segments ``K..N-1`` use the normal self rollout.  This is the hybrid
teacher/self-forcing variant.

Mechanism (per packed sample, serial rollout):
  optionally sample K ~ Uniform{0, ..., num_segments - 1}
  for g in segments 0..K-1:
    0. optionally run a standard flow-matching loss forward on GT segment g
    1. add low-level noise to GT segment g and write it to the KV cache
       (no student rollout)
  for g in segments:
    1. add noise to segment g's GT latent -> xt_g            (grad)
    2. forward WITH cache (history = detached generated 0..g-1) -> v_pred
    3. velocity MSE loss on segment g
    4. construct a low-noise cache latent either from the noise endpoint
       (legacy ``end_noise=True``) or by stepping directly from xt_g, then
       forward update_cache=True to write segment g's self-generated KV into
       the cache                                                        (no_grad)

In hybrid mode the second loop starts at ``g=K``.  Consequently segment K sees
only noisy-GT history, while later segments see a mixture of noisy-GT prefix and
the model's own generated rollout.

Because cache writes are DETACHED (see
``nomas_cpt.causal_attention_with_cache_mp``), the history is a gradient-isolated
constant: each segment's loss backprops only through its own forward. This is
one-step self-forcing (no BPTT), which is why it is memory-safe.

Unlike ``SelfForcingTransport`` (DMD), this creates no critic/teacher and uses a
plain flow-matching velocity loss with the single main model + optimizer.

The transport consumes the packed tensor contract built by the single-file SFT
collator and unpacks it per sample. Its cache path preserves the segment-level action-attention
semantics used by packed diffusion forcing:

* ``cross_segment_text_head_only`` masks every cached previous segment's action
  tail before both loss and cache-write forwards.
* ``intra_segment_action_mask`` passes the current segment's per-action text
  offsets and video windows through every cache-backed forward.
* camera-control batches slice each segment's aligned trajectory, apply the
  Stage-5 per-segment camera keep/drop policy, strip camera text only for
  camera-on segments, and inject camera conditioning in both the loss and
  cache-write forwards.

``segment_dropout`` additionally moves text CFG dropout from the dataset (which
empties every segment of a sample at once) into this transport, one segment at a
time -- the shape the joint bench actually renders, where a single segment runs
text-uncond while its neighbours stay text-conditioned.  A dropped segment runs
two passes over the same cache:

  conditional text   -> velocity used ONLY to roll the KV cache forward (no_grad,
                        since the cache write consumes it detached)
  unconditional text -> velocity used ONLY for the MSE loss (keeps the graph)

so the cached history stays text-conditional (matching seg_ar inference, whose
cache is written from the CFG-combined latent) while the loss still trains the
unconditional branch.  Camera conditioning, when the segment has it, is injected
in both passes.
"""

import contextlib
import hashlib
import json
import os
import random
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.profiler import record_function

from hymm.utils.parallel import sync_object_for_parallel_training, sync_random_states

from hymm.distributed.loss import global_token_count, global_token_mean_loss
from hymm.distributed.parallel_states import nccl_info, get_sequence_parallel_state
from hymm.trainer.transport.segment_sft_helpers import (
    _compute_action_offsets_for_prompt,
    _sample_segment_camera_keep,
    _strip_camera_text_blocks,
    _text_tokens,
)
from hymm.trainer.noise_scheduler import (
    sample_timesteps,
    apply_timestep_shift,
    add_flow_noise,
)
from hymm.utils.train_utils import (
    encode_text_from_ids,
    common_backward_and_step,
    common_backward_only,
    common_finalize_backward_and_step,
)
from hymm.utils.torch_utils import resolve_dit_training_precision


def _resolve_nested_config(args: Any, name: str, default: Any = None) -> Any:
    """Resolve a flattened training option or its common nested config forms."""
    for obj in (
        args,
        getattr(args, "training", None),
        getattr(args, "dataset", None),
    ):
        if obj is not None:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return default


def _loss_data_parallel_group(parallel_dims):
    """Return the data-replica group, excluding sequence-parallel ranks."""
    if parallel_dims is None or not parallel_dims.dp_enabled:
        return None
    return parallel_dims.dp_mesh.get_group()


class SegmentSelfForcingTransport:
    """Serial KV-cache self-forcing with pure velocity loss (no DMD)."""

    def __init__(
        self,
        *,
        timestep_sample_method: str = "uniform",
        timestep_shift: float = 3.0,
        context_low_t_min: float = 0.1,
        context_low_t_max: float = 0.3,
        df_random_prefix: bool = False,
        prefix_flow_matching_loss: bool = False,
        intra_segment_action_mask: bool = False,
        cross_segment_text_head_only: bool = False,
        training_timesteps: int = 1000,
        video_shift: float = 1.0,
        compute_dtype: torch.dtype = torch.bfloat16,
        cache_dtype: torch.dtype = torch.bfloat16,
        streaming_backward: bool = False,
        skip_last_segment_cache_write: bool = True,
        end_noise: bool = True,
        segment_dropout: bool = False,
        segment_dropout_p: float = 0.1,
        logger=None,
    ):
        self.timestep_sample_method = timestep_sample_method
        self.timestep_shift = timestep_shift
        # Cached self-generated context is re-noised to a LOW noise level sampled
        # uniformly in [context_low_t_min, context_low_t_max] (fraction of the
        # noise path; 0=clean, 1=pure noise), following PSE's pse_low_t_min/max.
        # Re-noising reuses the predicted velocity as the noise direction (same
        # trajectory the segment was generated on), NOT fresh noise:
        #   x_t_low = x0 + t_low * v_pred == (1-t_low)*x0 + t_low*noise
        # This brackets the seg_ar inference cache noise level (cache_at_step
        # after flow-shift), instead of denoising all the way to clean.
        self.context_low_t_min = float(context_low_t_min)
        self.context_low_t_max = float(context_low_t_max)
        # Hybrid forcing: prefill K leading segments with low-noise GT latents,
        # then start the ordinary self-generated serial rollout.
        self.df_random_prefix = bool(df_random_prefix)
        # When enabled, prefix segments also receive a normal flow-matching loss
        # forward at the configured training timestep distribution.  Cache
        # prefill remains a separate no-grad low-noise GT forward, because the
        # training timestep/input and cache timestep/input have different roles.
        self.prefix_flow_matching_loss = bool(prefix_flow_matching_loss)
        self.intra_segment_action_mask = bool(intra_segment_action_mask)
        self.cross_segment_text_head_only = bool(cross_segment_text_head_only)
        self.training_timesteps = training_timesteps
        self.video_shift = video_shift
        self.compute_dtype = compute_dtype
        self.cache_dtype = cache_dtype
        self.streaming_backward = bool(streaming_backward)
        # Legacy endpoint reconstruction starts from the known pure-noise
        # endpoint:
        #   x0_hat = noise - v_pred
        #   x_low  = x0_hat + t_low * v_pred
        # When disabled, take one Euler/flow step from the actual model input:
        #   x_low  = x_t + (t_low - t) * v_pred
        # The latter does not use the pure-noise endpoint and remains anchored
        # to the noisy latent that produced v_pred.
        self.end_noise = bool(end_noise)
        # A sample's last segment writes its KV into the cache, but no later
        # segment attends to it and the cache is freed as soon as the sample
        # ends, so both that forward and its cache slots are dead weight.
        self.skip_last_segment_cache_write = bool(skip_last_segment_cache_write)
        # Per-segment text CFG dropout, replacing the dataset's whole-sample
        # dropout. Requires dataset.uncond_p=0 so the conditional text of a
        # dropped segment still reaches this transport (see
        # create_segment_self_forcing_transport).
        self.segment_dropout = bool(segment_dropout)
        self.segment_dropout_p = float(segment_dropout_p)
        if self.segment_dropout and not 0.0 <= self.segment_dropout_p <= 1.0:
            raise ValueError(
                "segment_self_forcing.segment_dropout_p must be in [0, 1], got "
                f"{self.segment_dropout_p}"
            )
        self.logger = logger
        # Advertised so train.py logging (self.transport.enable / chunk_size)
        # does not crash; this transport is neither CPT nor chunked DF.
        self.enable = True
        self.chunk_size = 0
        self.logit_normal_weighting = False

    # ------------------------------------------------------------------
    # KV cache helpers (mirror CPTCausalInferencePipeline._initialize_kv_cache)
    # ------------------------------------------------------------------
    def _init_kv_cache(self, model, batch_size, max_cache_tokens, dtype, device):
        hidden_size = model.hidden_size
        heads_num = model.heads_num
        num_double_blocks = len(model.double_blocks)
        head_dim = hidden_size // heads_num
        cache_heads = (
            heads_num // nccl_info.sp_size
            if get_sequence_parallel_state()
            else heads_num
        )
        kv_cache = [
            {
                "k": torch.zeros(
                    [batch_size, max_cache_tokens, cache_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, max_cache_tokens, cache_heads, head_dim],
                    dtype=dtype,
                    device=device,
                ),
            }
            for _ in range(num_double_blocks)
        ]
        cache_mask_list = [
            torch.zeros([batch_size, max_cache_tokens], dtype=torch.bool, device=device)
            for _ in range(num_double_blocks)
        ]
        return kv_cache, cache_mask_list

    @staticmethod
    def _sample_segment_text_drops(
        *,
        num_segments: int,
        dropout_p: float,
        global_step: int,
        sample_idx: int,
    ) -> List[bool]:
        """Decide which segments of one packed sample run text-uncond.

        Returns a length-``num_segments`` bool list (True = drop that segment's
        text). Like ``_sample_segment_camera_keep``, the draws come from a
        deterministic per-(global_step, sample_idx) RNG so **every SP rank in a
        group derives the identical mask** without an extra collective; a desync
        would leave ranks conditioning the same tokens on different text.

        Each segment is dropped independently with probability ``dropout_p``. No
        constraint forces a sample to keep or drop any segment: unlike camera
        dropout (which teaches within-clip camera/text switching) the point here
        is to match the marginal CFG dropout rate at segment granularity.
        """
        num_segments = max(int(num_segments), 0)
        if num_segments == 0 or dropout_p <= 0.0:
            return [False] * num_segments

        rng = random.Random()
        rng.seed(
            (int(global_step) * 2_654_435_761 + int(sample_idx) * 40_503 + 7_919)
            & 0x7FFFFFFF
        )
        return [rng.random() < dropout_p for _ in range(num_segments)]

    @staticmethod
    def _encode_uncond_text(text_encoder, *, text_len, device, ids_like, mask_like):
        """Encode the empty prompt: the text-uncond block for dropped segments.

        Encoded once per step through the same tokenizer/encoder path the packed
        text uses, so a dropped segment sees exactly the block the dataset would
        have produced for a whole-sample CFG dropout. Returns
        ``([1, S, C], [1, S])``.

        ``text_len`` is the ENCODED block length, i.e. the length the caller
        needs to match, not the tokenized id length: ``text2tokens`` pads to
        ``max_length + crop_start`` and ``encode`` drops the template prefix
        again, so the packed ids are longer than the states they produce.
        """
        ids, mask = _text_tokens(text_encoder, "", int(text_len))
        states, masks = encode_text_from_ids(
            text_encoder,
            ids.to(device=device, dtype=ids_like.dtype).reshape(1, 1, -1),
            mask.to(device=device, dtype=mask_like.dtype).reshape(1, 1, -1),
            device,
        )
        return states[:, 0], masks[:, 0]

    def _sample_context_low_t(self, device, parallel_dims):
        """Draw the noise level a cached-context write is re-noised to.

        Split out so a skipped write can still consume its draw, which keeps
        the sampling stream identical to a run that performs every write.
        """
        low_t_min, low_t_max = (
            self.context_low_t_min,
            self.context_low_t_max,
        )
        if low_t_max < low_t_min:
            low_t_min, low_t_max = low_t_max, low_t_min
        random_fraction = torch.rand((1,), device=device)
        low_t = low_t_min + (low_t_max - low_t_min) * random_fraction
        if parallel_dims is not None:
            low_t = sync_object_for_parallel_training(
                low_t, parallel_dims=parallel_dims
            )
        return low_t

    def _build_self_generated_context(
        self,
        *,
        noisy_segment,
        noise,
        timestep,
        low_t,
        velocity_pred,
    ):
        """Construct the low-noise latent written to the self-forcing cache."""
        low_t_broadcast = low_t
        timestep_broadcast = timestep
        while low_t_broadcast.dim() < noisy_segment.dim():
            low_t_broadcast = low_t_broadcast.unsqueeze(-1)
        while timestep_broadcast.dim() < noisy_segment.dim():
            timestep_broadcast = timestep_broadcast.unsqueeze(-1)

        if self.end_noise:
            generated_clean = noise - velocity_pred
            low_t_broadcast = low_t_broadcast.to(generated_clean.dtype)
            return generated_clean + low_t_broadcast * velocity_pred

        low_t_broadcast = low_t_broadcast.to(noisy_segment.dtype)
        timestep_broadcast = timestep_broadcast.to(noisy_segment.dtype)
        return noisy_segment + (low_t_broadcast - timestep_broadcast) * velocity_pred

    def _cache_forward(
        self,
        model,
        hidden_states,
        timestep,
        text_states,
        text_mask,
        kv_cache,
        cache_valid_len,
        update_cache,
        global_frame_counter,
        cache_mask_list,
        intra_seg_txt_offsets=None,
        intra_seg_vid_windows=None,
        camera_embedding=None,
        camera_has_mask=None,
        camera_inputs=None,
        camera_rope_time_offset=0,
    ):
        """Route a cache forward through model.__call__ (FSDP2-safe).

        Mirrors DMDModelAdapter._forward_with_cache: the KV cache params are
        passed via extra_kwargs['kv_cache_params'] so model.forward() delegates
        to forward_with_cache() and FSDP2 unshard hooks fire at model level.
        Returns (img, new_cache_tokens).
        """
        # Match the autocast compute dtype (bf16) for the input, exactly like
        # DMDModelAdapter (self_forcing_transport.py:46-48). With fp32 master
        # weights the input latent is fp32; feeding it in makes the current
        # chunk's q/k fp32 while the (bf16) cached history k mismatches inside
        # flash-attn ("query and key must have the same dtype"). Cast to the
        # active autocast dtype so q/k/v and the cache are all bf16.
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
            if hidden_states.dtype != target_dtype:
                hidden_states = hidden_states.to(dtype=target_dtype)
            if text_states is not None and text_states.dtype != target_dtype:
                text_states = text_states.to(dtype=target_dtype)
        extra_kwargs = {
            "kv_cache_params": {
                "kv_cache_list": kv_cache,
                "cache_valid_len": cache_valid_len,
                "update_cache": update_cache,
                "global_frame_counter": global_frame_counter,
                "cache_txt": True,
                "cache_mask_list": cache_mask_list,
                "attn_mode_override": None,
                "intra_seg_txt_offsets": intra_seg_txt_offsets,
                "intra_seg_vid_windows": intra_seg_vid_windows,
                "camera_embedding": camera_embedding,
                "camera_has_mask": camera_has_mask,
                "camera_inputs": camera_inputs,
                "camera_rope_time_offset": camera_rope_time_offset,
            }
        }
        return model(
            hidden_states=hidden_states,
            timestep=timestep,
            text_states=text_states,
            encoder_attention_mask=text_mask,
            extra_kwargs=extra_kwargs,
        )

    def _cfg_branch_forward(
        self,
        model,
        *,
        hidden_states,
        timestep,
        segment_text,
        segment_mask,
        uncond_text_states,
        uncond_text_mask,
        kv_cache,
        cache_valid_len,
        global_frame_counter,
        cache_mask_list,
        intra_seg_txt_offsets=None,
        intra_seg_vid_windows=None,
        camera_inputs=None,
        camera_has_mask=None,
    ):
        """Run one text-dropped segment conditionally and unconditionally.

        Returns ``(velocity_cond, velocity_uncond)``, each ``[1, C, T, H, W]``.

        The conditional pass keeps the segment's real text and its per-action
        offsets, and its velocity only ever reaches the cache write, which
        consumes it detached -- so it runs under ``no_grad`` and retains no
        activations. The unconditional pass gets the empty prompt and ``None``
        offsets (mirroring how the seg_ar pipeline builds its CFG-negative row)
        and carries the velocity loss, so it is the one that keeps a graph.
        Camera conditioning goes into BOTH passes, leaving the unconditional
        velocity camera-controlled.

        Both passes read the same batch-1 conditional cache and neither writes
        it; the write happens later from the conditional velocity.
        """

        def _forward(text_states, text_mask, offsets, windows):
            return self._cache_forward(
                model,
                hidden_states=hidden_states,
                timestep=timestep,
                text_states=text_states,
                text_mask=text_mask,
                kv_cache=kv_cache,
                cache_valid_len=cache_valid_len,
                update_cache=False,
                global_frame_counter=global_frame_counter,
                cache_mask_list=cache_mask_list,
                intra_seg_txt_offsets=offsets,
                intra_seg_vid_windows=windows,
                camera_inputs=camera_inputs,
                camera_has_mask=camera_has_mask,
            )

        with torch.no_grad():
            velocity_cond = _forward(
                segment_text,
                segment_mask,
                intra_seg_txt_offsets,
                intra_seg_vid_windows,
            )[0]
        velocity_uncond = _forward(
            uncond_text_states.to(segment_text.dtype),
            uncond_text_mask.to(segment_mask.dtype),
            None,
            None,
        )[0]
        return velocity_cond, velocity_uncond

    @staticmethod
    def _camera_frame_span(frame_start, frame_end, total_camera_frames):
        """Map a latent-frame segment to its aligned raw camera-frame slice."""
        total_camera_frames = max(int(total_camera_frames), 0)
        if total_camera_frames <= 0:
            return 0, 0
        frame_start = max(int(frame_start), 0)
        frame_end = max(int(frame_end), frame_start)
        camera_start = 0 if frame_start == 0 else 4 * frame_start - 3
        camera_end = 0 if frame_end == 0 else 4 * frame_end - 3
        camera_start = max(0, min(camera_start, total_camera_frames))
        camera_end = max(camera_start, min(camera_end, total_camera_frames))
        return camera_start, camera_end

    @classmethod
    def _prepare_camera_inputs(
        cls,
        model,
        *,
        norm_extrinsics,
        norm_intrinsics,
        action_onehot,
        frame_start,
        frame_end,
        device,
    ):
        """Slice raw camera conditioning for one segment.

        The actual parameterized preprocessing stays inside model.__call__ so
        the FSDP2 root unshard hooks cover the camera embedding modules.
        """
        mode = getattr(model, "camera_injection_mode", None)
        if mode not in ("action_attn", "extrinsics_attn"):
            if mode is None:
                return None
            raise NotImplementedError(
                "SegmentSelfForcing camera control only supports cache-compatible "
                f"action_attn/extrinsics_attn injection, got {mode!r}."
            )

        if mode == "action_attn":
            if action_onehot is None:
                return None
            total_camera_frames = int(action_onehot.shape[0])
        else:
            if norm_extrinsics is None:
                return None
            total_camera_frames = int(norm_extrinsics.shape[0])

        camera_start, camera_end = cls._camera_frame_span(
            frame_start, frame_end, total_camera_frames
        )
        if camera_end <= camera_start:
            return None

        if mode == "action_attn":
            return {
                "action_onehot": action_onehot[camera_start:camera_end]
                .unsqueeze(0)
                .to(device=device),
            }
        if norm_intrinsics is None:
            return None
        return {
            "norm_extrinsics": norm_extrinsics[camera_start:camera_end]
            .unsqueeze(0)
            .to(device=device),
            "norm_intrinsics": norm_intrinsics[camera_start:camera_end]
            .unsqueeze(0)
            .to(device=device),
        }

    # ------------------------------------------------------------------
    # Per-sample segment boundaries (latent frames) from the batch kwargs.
    # Mirrors unified.py boundary clamping.
    # ------------------------------------------------------------------
    @staticmethod
    def _segment_boundaries(pvb, num_latent_frames):
        if isinstance(pvb, torch.Tensor):
            pvb = pvb.squeeze().tolist()
        if isinstance(pvb, (list, tuple)) and pvb and isinstance(pvb[0], (list, tuple)):
            pvb = pvb[0]
        if not pvb or not isinstance(pvb, (list, tuple)):
            return [0, num_latent_frames]
        boundaries = [int(x) for x in pvb]
        boundaries[0] = 0
        boundaries[-1] = num_latent_frames
        result = []
        for boundary in boundaries:
            boundary = max(0, min(int(boundary), num_latent_frames))
            if not result or boundary != result[-1]:
                result.append(boundary)
        if len(result) < 2:
            result = [0, num_latent_frames]
        return result

    @staticmethod
    def _per_segment_metadata(raw, num_segments, fill_value=None):
        """Normalize a ragged metadata field to one entry per segment."""
        if raw is None:
            return [fill_value] * num_segments
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().tolist()
        if not isinstance(raw, (list, tuple)):
            return [fill_value] * num_segments
        values = list(raw[:num_segments])
        if len(values) < num_segments:
            values.extend([fill_value] * (num_segments - len(values)))
        return values

    @staticmethod
    def _prepare_intra_segment_payload(
        offsets,
        windows,
        *,
        frame_start,
        frame_end,
        valid_text_tokens,
    ):
        """Build the cache-attention payload for one segment.

        Dataset video windows use sample-level latent-frame coordinates, while
        ``forward_with_cache`` consumes a single segment and therefore expects
        segment-relative coordinates.
        """
        if offsets is None or windows is None:
            return None, None
        if torch.is_tensor(offsets):
            offsets = offsets.detach().cpu().tolist()
        if torch.is_tensor(windows):
            windows = windows.detach().cpu().tolist()
        if not isinstance(offsets, (list, tuple)) or len(offsets) < 2:
            return None, None
        if not isinstance(windows, (list, tuple)) or len(windows) == 0:
            return None, None

        valid = max(int(valid_text_tokens), 0)
        clamped_offsets = [max(0, min(int(value), valid)) for value in offsets]
        if any(
            clamped_offsets[index] > clamped_offsets[index + 1]
            for index in range(len(clamped_offsets) - 1)
        ):
            return None, None

        segment_frames = max(int(frame_end) - int(frame_start), 0)
        action_count = len(clamped_offsets) - 1
        relative_windows = []
        for window in windows:
            if torch.is_tensor(window):
                window = window.detach().cpu().tolist()
            if not isinstance(window, (list, tuple)) or len(window) != 3:
                return None, None
            start, end, action_index = (int(value) for value in window)
            start = max(0, min(start - int(frame_start), segment_frames))
            end = max(0, min(end - int(frame_start), segment_frames))
            if end <= start:
                return None, None
            if action_index < -1 or action_index >= action_count:
                return None, None
            relative_windows.append([start, end, action_index])

        # The range builder emits only the supplied video windows; require an
        # exact contiguous tiling so no current video query can be dropped.
        if (
            relative_windows[0][0] != 0
            or relative_windows[-1][1] != segment_frames
            or any(
                relative_windows[index][1] != relative_windows[index + 1][0]
                for index in range(len(relative_windows) - 1)
            )
        ):
            return None, None

        # The model API is batched. Segment self-forcing unpacks one sample at a
        # time, so wrap this segment's offsets in a batch-size-one list.
        return [clamped_offsets], relative_windows

    @staticmethod
    def _resolve_action_head_start(offsets, fallback, valid_text_tokens):
        """Match Magi's offsets-first cross-segment head boundary selection."""
        valid = max(int(valid_text_tokens), 0)
        if torch.is_tensor(offsets):
            offsets = offsets.detach().cpu().tolist()
        if isinstance(offsets, (list, tuple)) and len(offsets) > 0:
            return max(0, min(int(offsets[0]), valid))
        if torch.is_tensor(fallback):
            fallback = fallback.detach().cpu().item()
        if fallback is None or int(fallback) < 0:
            return -1
        return max(0, min(int(fallback), valid))

    @staticmethod
    def _hide_cached_action_tails(cache_mask_list, cached_action_tails):
        """Hide all previous segments' cached action-tail keys in-place."""
        if not cached_action_tails:
            return
        for tail_start, tail_end in cached_action_tails:
            for cache_mask in cache_mask_list:
                cache_mask[:, tail_start:tail_end] = False

    @staticmethod
    def _record_cached_action_tail(
        cached_action_tails,
        *,
        cache_valid_len,
        new_tokens,
        text_block_len,
        action_start,
        valid_text_tokens,
    ):
        """Record a just-written segment action tail in absolute cache coords."""
        if action_start is None:
            return False
        action_start = int(action_start)
        valid_text_tokens = int(valid_text_tokens)
        if action_start < 0 or action_start >= valid_text_tokens:
            return False
        image_tokens = int(new_tokens) - int(text_block_len)
        if image_tokens < 0:
            raise ValueError(
                "SegmentSelfForcing cache write returned fewer tokens than its "
                f"text block: new_tokens={new_tokens}, text_block_len={text_block_len}"
            )
        text_start = int(cache_valid_len) + image_tokens
        tail_start = text_start + action_start
        tail_end = text_start + int(text_block_len)
        if tail_end <= tail_start:
            return False
        cached_action_tails.append((tail_start, tail_end))
        return True

    @staticmethod
    def _velocity_loss_stats(
        velocity_pred: torch.Tensor,
        target_velocity: torch.Tensor,
        recreation_mask: torch.Tensor = None,
    ):
        """Return ``(loss_sum, valid_element_count, normalized_loss)``.

        ``recreation_mask`` is segment-local ``[1, T, H, W]`` with True meaning
        excluded.  Returning sums/counts lets the caller reduce all packed
        samples and variable-length segments with the same per-element weighting
        as the fused token-packed flow-matching path, rather than giving every
        segment equal weight regardless of its token count.
        """
        if velocity_pred.shape != target_velocity.shape:
            raise ValueError(
                "SegmentSelfForcing velocity/target shape mismatch: "
                f"{tuple(velocity_pred.shape)} vs {tuple(target_velocity.shape)}"
            )

        loss_raw = (velocity_pred.float() - target_velocity.float()).square()
        if recreation_mask is None:
            loss_sum = loss_raw.sum()
            valid_count = loss_raw.new_tensor(float(loss_raw.numel()))
        else:
            expected_mask_shape = (
                loss_raw.shape[0],
                loss_raw.shape[2],
                loss_raw.shape[3],
                loss_raw.shape[4],
            )
            if tuple(recreation_mask.shape) != expected_mask_shape:
                raise ValueError(
                    "SegmentSelfForcing recreation-mask shape mismatch: "
                    f"mask={tuple(recreation_mask.shape)} "
                    f"expected={expected_mask_shape}"
                )
            keep = (~recreation_mask.bool()).unsqueeze(1)
            loss_sum = (loss_raw * keep).sum()
            valid_count = keep.sum(dtype=loss_raw.dtype) * loss_raw.shape[1]

        normalized = loss_sum / valid_count.clamp(min=1.0)
        return loss_sum, valid_count, normalized

    @staticmethod
    def _metadata_loss_count(
        *,
        latent_channels: int,
        frame_start: int,
        frame_end: int,
        height: int,
        width: int,
        recreation_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Compute one segment's valid-element count without building a graph."""
        if recreation_mask is None:
            count = (
                int(latent_channels)
                * (int(frame_end) - int(frame_start))
                * int(height)
                * int(width)
            )
            return torch.tensor(float(count), dtype=torch.float32)
        keep = ~recreation_mask[:, int(frame_start) : int(frame_end)].bool()
        return keep.sum(dtype=torch.float32) * int(latent_channels)

    def forward_step(
        self,
        args,
        model,
        vae,
        text_encoder,
        optimizer,
        lr_scheduler,
        batch,
        device,
        global_step,
        scalar_states,
        logger,
        parallel_dims=None,
    ):
        """Serial self-forcing forward step over a packed tensor batch."""
        logger = logger or self.logger
        profile_active = bool(
            getattr(getattr(args, "profiling", None), "is_profiler", False)
            or getattr(args, "is_profiler", False)
            or os.environ.get("HY_DEBUG_MEMORY_MARKS", "").strip().lower()
            in ("1", "true", "yes")
        )
        profile_rank0 = profile_active and int(os.environ.get("LOCAL_RANK", 0)) == 0

        def _scope(name):
            return record_function(name) if profile_active else contextlib.nullcontext()

        def _mem_mark(name):
            if not profile_rank0:
                return
            torch.cuda.synchronize(device)
            logger.info(
                f"[SSF_MEM] {name} "
                f"allocated_gib={torch.cuda.memory_allocated(device) / 2**30:.3f} "
                f"reserved_gib={torch.cuda.memory_reserved(device) / 2**30:.3f} "
                f"peak_allocated_gib={torch.cuda.max_memory_allocated(device) / 2**30:.3f}"
            )

        debug_fingerprint = os.environ.get(
            "HY_DEBUG_INPUT_FINGERPRINT", ""
        ).strip().lower() in ("1", "true", "yes")

        if not isinstance(batch, dict) or "num_packed_samples" not in batch:
            raise ValueError(
                "SegmentSelfForcingTransport requires the packed SFT "
                "batch (num_packed_samples/sample_boundaries/sample_shapes/"
                "packed_latent). Non-packed batches are not supported in v1."
            )

        num_packed = batch["num_packed_samples"]
        sample_boundaries = batch["sample_boundaries"]
        sample_shapes = batch["sample_shapes"]
        packed_latent = batch["packed_latent"].to(device)
        packed_text_ids = batch["packed_text_ids"].to(device).clone()
        packed_text_mask = batch["packed_text_mask"].to(device).clone()
        kwargs_list = batch.get("kwargs_list", [{}] * num_packed)
        use_camera_control = bool(
            getattr(getattr(args, "camera", None), "use_camera_control", False)
        )
        fingerprint = None
        if debug_fingerprint:

            def _tensor_fp(tensor):
                tensor = tensor.detach()
                cpu = tensor.reshape(-1).cpu()
                raw_view = cpu.view(torch.uint8)
                return {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "sha256": hashlib.sha256(raw_view.numpy().tobytes()).hexdigest(),
                }

            fingerprint = {
                "step": int(global_step),
                "num_packed": int(num_packed),
                "sample_boundaries": [[int(a), int(b)] for a, b in sample_boundaries],
                "sample_shapes": [[int(v) for v in shape] for shape in sample_shapes],
                "packed_latent": _tensor_fp(packed_latent),
                "packed_text_ids": _tensor_fp(packed_text_ids),
                "packed_text_mask": _tensor_fp(packed_text_mask),
                "cpu_rng_sha256": hashlib.sha256(
                    torch.get_rng_state().cpu().numpy().tobytes()
                ).hexdigest(),
                "cuda_rng_sha256": hashlib.sha256(
                    torch.cuda.get_rng_state(device).cpu().numpy().tobytes()
                ).hexdigest(),
            }

        if not len(sample_boundaries) == len(sample_shapes) == num_packed:
            raise ValueError(
                "SegmentSelfForcing packed metadata misalignment: "
                f"num_packed={num_packed}, "
                f"sample_boundaries={len(sample_boundaries)}, "
                f"sample_shapes={len(sample_shapes)}"
            )
        if len(kwargs_list) != num_packed:
            raise ValueError(
                "SegmentSelfForcing kwargs_list misalignment: "
                f"num_packed={num_packed}, kwargs_list={len(kwargs_list)}"
            )

        camera_has_list: List[bool] = [False] * num_packed
        camera_keep_per_segment: List[Optional[List[bool]]] = [None] * num_packed
        camera_diag = {}
        if use_camera_control:
            required_camera_fields = (
                "packed_norm_ext",
                "packed_norm_int",
                "packed_action_oh",
            )
            missing_camera_fields = [
                name for name in required_camera_fields if name not in batch
            ]
            if missing_camera_fields:
                raise ValueError(
                    "SegmentSelfForcing camera control requires token-packed "
                    "camera fields, missing: " + ", ".join(missing_camera_fields)
                )
            for name in required_camera_fields:
                if len(batch[name]) != num_packed:
                    raise ValueError(
                        "SegmentSelfForcing camera metadata misalignment: "
                        f"{name}={len(batch[name])}, num_packed={num_packed}"
                    )

            raw_has_camera = batch.get("packed_has_camera", None)
            for sample_idx in range(num_packed):
                if raw_has_camera is not None and sample_idx < len(raw_has_camera):
                    value = raw_has_camera[sample_idx]
                    camera_has_list[sample_idx] = (
                        bool(value.item())
                        if isinstance(value, torch.Tensor)
                        else bool(value)
                    )
                else:
                    camera_has_list[sample_idx] = True

            dropout_p = float(
                getattr(
                    getattr(args, "camera", None),
                    "per_segment_camera_dropout_p",
                    0.0,
                )
                or 0.0
            )
            min_on = int(
                getattr(
                    getattr(args, "camera", None),
                    "per_segment_camera_dropout_min_on",
                    1,
                )
                or 1
            )
            dropped_samples = 0
            dropped_segments = 0
            for sample_idx in range(num_packed):
                sample_kwargs = (
                    kwargs_list[sample_idx] if sample_idx < len(kwargs_list) else {}
                )
                if not isinstance(sample_kwargs, dict):
                    sample_kwargs = {}
                num_frames = int(sample_shapes[sample_idx][0])
                boundaries = self._segment_boundaries(
                    sample_kwargs.get("prompt_video_boundaries", None),
                    num_frames,
                )
                num_segments = len(boundaries) - 1
                if camera_has_list[sample_idx]:
                    keep = _sample_segment_camera_keep(
                        num_segments=num_segments,
                        dropout_p=dropout_p,
                        min_on=min_on,
                        global_step=global_step,
                        sample_idx=sample_idx,
                    )
                else:
                    keep = [False] * num_segments
                camera_keep_per_segment[sample_idx] = keep
                num_dropped = sum(1 for enabled in keep if not enabled)
                if camera_has_list[sample_idx] and num_dropped:
                    dropped_samples += 1
                    dropped_segments += num_dropped
            if fingerprint is not None:
                fingerprint["camera_has"] = camera_has_list
                fingerprint["camera_keep"] = camera_keep_per_segment

            camera_diag["camera/has_camera_ratio"] = float(
                sum(camera_has_list)
            ) / float(max(num_packed, 1))
            camera_diag["camera/seg_dropout_samples"] = float(dropped_samples)
            camera_diag["camera/seg_dropout_segments"] = float(dropped_segments)
            strip_camera_text = bool(
                getattr(
                    getattr(args, "camera", None),
                    "strip_camera_text_blocks",
                    True,
                )
            )
            if (
                strip_camera_text
                and packed_text_ids.dim() == 3
                and text_encoder is not None
            ):
                text_len = int(getattr(args, "text_len"))
                for sample_idx, keep in enumerate(camera_keep_per_segment):
                    if keep is None:
                        continue
                    sample_kwargs = kwargs_list[sample_idx]
                    if not isinstance(sample_kwargs, dict):
                        continue
                    prompts = sample_kwargs.get("text_encoder_input")
                    if not isinstance(prompts, list):
                        continue
                    actual_segments = min(
                        int(sample_kwargs.get("num_text_segments", len(prompts))),
                        packed_text_ids.shape[1],
                        len(prompts),
                        len(keep),
                    )
                    rewritten = list(prompts)
                    action_starts = sample_kwargs.get("text_action_start_index", None)
                    if action_starts is not None:
                        action_starts = list(action_starts)
                    action_offsets = sample_kwargs.get("text_action_offsets", None)
                    if action_offsets is not None:
                        action_offsets = list(action_offsets)
                    for segment_idx in range(actual_segments):
                        if not keep[segment_idx]:
                            continue
                        cleaned = _strip_camera_text_blocks(rewritten[segment_idx])
                        if cleaned == rewritten[segment_idx]:
                            continue
                        ids, mask = _text_tokens(text_encoder, cleaned, text_len)
                        packed_text_ids[sample_idx, segment_idx] = ids.to(
                            device=device, dtype=packed_text_ids.dtype
                        )
                        packed_text_mask[sample_idx, segment_idx] = mask.to(
                            device=device, dtype=packed_text_mask.dtype
                        )
                        rewritten[segment_idx] = cleaned
                        new_offsets = _compute_action_offsets_for_prompt(
                            text_encoder, cleaned, text_len
                        )
                        if action_starts is not None and segment_idx < len(
                            action_starts
                        ):
                            action_starts[segment_idx] = (
                                int(new_offsets[0]) if new_offsets else -1
                            )
                        if action_offsets is not None and segment_idx < len(
                            action_offsets
                        ):
                            action_offsets[segment_idx] = new_offsets
                    sample_kwargs["text_encoder_input"] = rewritten
                    if action_starts is not None:
                        sample_kwargs["text_action_start_index"] = action_starts
                    if action_offsets is not None:
                        sample_kwargs["text_action_offsets"] = action_offsets

        latent_channels = packed_latent.shape[1]
        scaling_factor = vae.config.scaling_factor
        video_latent = packed_latent * scaling_factor
        total_packed_tokens = int(packed_latent.shape[-1])
        if sample_boundaries:
            expected_start = 0
            for sample_idx, ((token_start, token_end), shape) in enumerate(
                zip(sample_boundaries, sample_shapes)
            ):
                token_start, token_end = int(token_start), int(token_end)
                num_frames, height, width = (int(v) for v in shape)
                expected_tokens = num_frames * height * width
                if (
                    token_start != expected_start
                    or token_end - token_start != expected_tokens
                ):
                    raise ValueError(
                        "SegmentSelfForcing invalid packed boundary/shape at "
                        f"sample {sample_idx}: boundary=({token_start},{token_end}), "
                        f"expected_start={expected_start}, shape={tuple(shape)}, "
                        f"expected_tokens={expected_tokens}"
                    )
                expected_start = token_end
            if expected_start != total_packed_tokens:
                raise ValueError(
                    "SegmentSelfForcing packed token coverage mismatch: "
                    f"metadata covers {expected_start} tokens, tensor has "
                    f"{total_packed_tokens}"
                )

        packed_rec_mask = batch.get("packed_recreation_mask", None)
        use_recreation_mask = bool(
            getattr(args, "enable_recreation_mask", False)
            and batch.get("enable_recreation_mask", False)
            and packed_rec_mask is not None
        )
        if use_recreation_mask:
            packed_rec_mask = packed_rec_mask.to(device=device, dtype=torch.bool)
            if packed_rec_mask.numel() != total_packed_tokens:
                if logger is not None:
                    logger.warning(
                        "SegmentSelfForcing packed_recreation_mask length "
                        f"{packed_rec_mask.numel()} != packed tokens "
                        f"{total_packed_tokens}; disabling recreation masking "
                        "for this batch."
                    )
                packed_rec_mask = None
                use_recreation_mask = False

        # ---- Text: encode all N samples, keep per-segment 4D layout. ----
        # Each packed sample retains per-segment text: [N, max_np, S, hidden].
        all_text_states, all_text_masks = encode_text_from_ids(
            text_encoder,
            packed_text_ids,
            packed_text_mask,
            device,
        )
        if all_text_states.dim() != 4:
            raise ValueError(
                "SegmentSelfForcing expects per-segment text (4D "
                "[N, max_np, S, hidden]). "
                f"Got shape {tuple(all_text_states.shape)}."
            )
        if all_text_states.shape[0] != num_packed:
            raise ValueError(
                "SegmentSelfForcing text batch mismatch: "
                f"text batch={all_text_states.shape[0]}, num_packed={num_packed}"
            )

        # ---- Text-uncond block for per-segment dropout (one encode/step). ----
        # Encoded unconditionally on the dropout outcome: with a sharded text
        # encoder this call carries collectives, so every rank must run it the
        # same number of times regardless of which segments it happens to drop.
        uncond_text_states = None
        uncond_text_mask = None
        if self.segment_dropout:
            if text_encoder is None:
                raise ValueError(
                    "segment_self_forcing.segment_dropout requires a text "
                    "encoder to build the per-segment unconditional block."
                )
            uncond_text_states, uncond_text_mask = self._encode_uncond_text(
                text_encoder,
                text_len=all_text_states.shape[2],
                device=device,
                ids_like=packed_text_ids,
                mask_like=packed_text_mask,
            )
            if uncond_text_states.shape[1] != all_text_states.shape[2]:
                raise ValueError(
                    "SegmentSelfForcing unconditional text length mismatch: "
                    f"uncond={uncond_text_states.shape[1]}, "
                    f"segments={all_text_states.shape[2]}"
                )

        if parallel_dims is not None:
            sync_random_states(parallel_dims)
        if fingerprint is not None and int(os.environ.get("RANK", 0)) == 0:
            logger.info(
                "[precision-input-fingerprint] "
                + json.dumps(fingerprint, sort_keys=True)
            )

        loss_dp_group = _loss_data_parallel_group(parallel_dims)
        streaming_backward = self.streaming_backward
        if (
            streaming_backward
            and int(args.parallelism.gradient_accumulation_steps) != 1
        ):
            raise ValueError(
                "segment_self_forcing.streaming_backward currently requires "
                "parallelism.gradient_accumulation_steps=1 because its final "
                "global loss-denominator scaling must not rescale gradients "
                "from an earlier microstep."
            )
        if streaming_backward and self.df_random_prefix:
            raise ValueError(
                "segment_self_forcing.streaming_backward currently requires "
                "df_random_prefix=false. A random prefix changes which segments "
                "contribute loss, so the exact global denominator is not known "
                "before the first segment backward."
            )

        # Each segment gets dp_size / global_valid_elements: FSDP averages
        # parameter gradients across the DP group after the final backward.
        # Metadata gives the denominator before any segment backward, so DP
        # replicas with different segment counts need no per-segment collective.
        streaming_total_loss_count = None
        streaming_backward_scale = None
        streaming_backward_total = 0
        streaming_backward_index = 0
        if streaming_backward:
            metadata_counts = []
            for sample_idx in range(num_packed):
                num_frames, height, width = (
                    int(value) for value in sample_shapes[sample_idx]
                )
                sample_kwargs = (
                    kwargs_list[sample_idx] if sample_idx < len(kwargs_list) else {}
                )
                if not isinstance(sample_kwargs, dict):
                    sample_kwargs = {}
                boundaries = self._segment_boundaries(
                    sample_kwargs.get("prompt_video_boundaries", None),
                    num_frames,
                )
                sample_rec_mask = None
                if use_recreation_mask:
                    token_start, token_end = sample_boundaries[sample_idx]
                    sample_rec_mask = packed_rec_mask[token_start:token_end].reshape(
                        1, num_frames, height, width
                    )
                for segment_idx in range(len(boundaries) - 1):
                    frame_start = boundaries[segment_idx]
                    frame_end = boundaries[segment_idx + 1]
                    if frame_end <= frame_start:
                        continue
                    metadata_counts.append(
                        self._metadata_loss_count(
                            latent_channels=latent_channels,
                            frame_start=frame_start,
                            frame_end=frame_end,
                            height=height,
                            width=width,
                            recreation_mask=sample_rec_mask,
                        ).to(device=device)
                    )
            if not metadata_counts:
                raise ValueError(
                    "SegmentSelfForcing streaming backward found no valid "
                    "loss-producing segments."
                )
            streaming_total_loss_count = global_token_count(
                torch.stack(metadata_counts).sum(), group=loss_dp_group
            )
            streaming_backward_total = len(metadata_counts)
            if streaming_total_loss_count.item() <= 0:
                raise ValueError(
                    "SegmentSelfForcing recreation mask excluded every loss "
                    "element across all data replicas."
                )
            streaming_backward_scale = (
                dist.get_world_size(loss_dp_group) if loss_dp_group is not None else 1
            ) / streaming_total_loss_count

        # ---- Serial rollout, accumulate per-segment velocity losses. ----
        loss_sums: List[torch.Tensor] = []
        loss_counts: List[torch.Tensor] = []
        seg_losses_log: List[torch.Tensor] = []
        total_segments = 0
        total_rollout_segments = 0
        total_gt_prefix_segments = 0
        total_prefix_loss_segments = 0
        max_cache_tokens_used = 0
        total_intra_masked_segments = 0
        total_cached_action_tails = 0
        total_camera_segments = 0
        total_text_dropped_segments = 0
        total_cfg_branch_segments = 0

        for sample_idx in range(num_packed):
            num_frames, height, width = (
                int(value) for value in sample_shapes[sample_idx]
            )
            tokens_per_frame = height * width
            token_start, token_end = sample_boundaries[sample_idx]
            # [1, C, span] -> [1, C, T, H, W] (token dim is frame-major).
            clean_sample = video_latent[:, :, token_start:token_end]
            clean_sample = clean_sample.reshape(
                1, latent_channels, num_frames, height, width
            )

            sample_kwargs = (
                kwargs_list[sample_idx] if sample_idx < len(kwargs_list) else {}
            )
            if not isinstance(sample_kwargs, dict):
                sample_kwargs = {}
            boundaries = self._segment_boundaries(
                sample_kwargs.get("prompt_video_boundaries", None),
                num_frames,
            )
            num_segments = len(boundaries) - 1
            num_text_segments = int(
                sample_kwargs.get("num_text_segments", num_segments)
            )
            if num_text_segments != num_segments:
                raise ValueError(
                    "SegmentSelfForcing segment/text mismatch at packed sample "
                    f"{sample_idx}: boundaries={boundaries} -> "
                    f"{num_segments} segments, num_text_segments="
                    f"{num_text_segments}"
                )
            if num_segments > all_text_states.shape[1]:
                raise ValueError(
                    "SegmentSelfForcing segment count exceeds padded text blocks "
                    f"at packed sample {sample_idx}: num_segments={num_segments}, "
                    f"text_blocks={all_text_states.shape[1]}"
                )
            action_starts = self._per_segment_metadata(
                sample_kwargs.get("text_action_start_index", None),
                num_segments,
                fill_value=-1,
            )
            action_offsets = self._per_segment_metadata(
                sample_kwargs.get("text_action_offsets", None),
                num_segments,
            )
            action_video_boundaries = self._per_segment_metadata(
                sample_kwargs.get("action_video_boundaries", None),
                num_segments,
            )
            text_drops = (
                self._sample_segment_text_drops(
                    num_segments=num_segments,
                    dropout_p=self.segment_dropout_p,
                    global_step=global_step,
                    sample_idx=sample_idx,
                )
                if self.segment_dropout
                else [False] * num_segments
            )

            sample_rec_mask = None
            if use_recreation_mask:
                sample_rec_mask = packed_rec_mask[token_start:token_end].reshape(
                    1, num_frames, height, width
                )

            # Fresh cache for this sample. Max tokens = all img tokens + one text
            # block per segment (cache_txt=True writes img+txt for each segment).
            # When the final segment's write is skipped, the cache only has to
            # cover segments [0, num_segments - 1), i.e. the frames up to that
            # segment's start plus one text block each.
            text_len = all_text_states.shape[2]
            skip_last_cache_write = (
                self.skip_last_segment_cache_write and num_segments >= 1
            )
            last_cache_write_segment = num_segments - 1 if skip_last_cache_write else -1
            if skip_last_cache_write:
                cached_frames = boundaries[num_segments - 1] - boundaries[0]
                cached_text_blocks = num_segments - 1
            else:
                cached_frames = num_frames
                cached_text_blocks = num_segments
            max_cache_tokens = (
                cached_frames * tokens_per_frame + cached_text_blocks * text_len
            )
            with _scope("ssf/kv_cache_alloc"):
                kv_cache, cache_mask_list = self._init_kv_cache(
                    model,
                    batch_size=1,
                    max_cache_tokens=max_cache_tokens,
                    dtype=self.cache_dtype,
                    device=device,
                )
            _mem_mark(f"sample{sample_idx}/after_kv_cache_alloc")
            cache_valid_len = 0
            global_frame_counter = 0
            cached_action_tails = []
            camera_keep = (
                camera_keep_per_segment[sample_idx] if use_camera_control else None
            )
            sample_norm_extrinsics = (
                batch["packed_norm_ext"][sample_idx] if use_camera_control else None
            )
            sample_norm_intrinsics = (
                batch["packed_norm_int"][sample_idx] if use_camera_control else None
            )
            sample_action_onehot = (
                batch["packed_action_oh"][sample_idx] if use_camera_control else None
            )

            # Hybrid forcing split.  Keep at least one segment for student
            # rollout/loss: K is sampled uniformly from [0, num_segments - 1].
            prefix_segments = 0
            if self.df_random_prefix and num_segments > 1:
                prefix_tensor = torch.randint(0, num_segments, (1,), device=device)
                if parallel_dims is not None:
                    prefix_tensor = sync_object_for_parallel_training(
                        prefix_tensor, parallel_dims=parallel_dims
                    )
                prefix_segments = int(prefix_tensor.item())

            for segment_idx in range(num_segments):
                frame_start = boundaries[segment_idx]
                frame_end = boundaries[segment_idx + 1]
                if frame_end <= frame_start:
                    continue
                clean_segment = clean_sample[:, :, frame_start:frame_end]
                segment_text = all_text_states[sample_idx, segment_idx].unsqueeze(0)
                segment_mask = all_text_masks[sample_idx, segment_idx].unsqueeze(0)
                valid_text_tokens = int(segment_mask.bool().sum().item())
                action_start = self._resolve_action_head_start(
                    action_offsets[segment_idx],
                    action_starts[segment_idx],
                    valid_text_tokens,
                )
                # Text-uncond segment: the loss forward drops this segment's
                # text, while every cache write keeps the conditional block (so
                # cached history, its recorded action tail, and the head/tail
                # visibility rules are unchanged by dropout).
                text_dropped = bool(text_drops[segment_idx])
                if text_dropped:
                    total_text_dropped_segments += 1

                intra_seg_txt_offsets = None
                intra_seg_vid_windows = None
                if self.intra_segment_action_mask:
                    (
                        intra_seg_txt_offsets,
                        intra_seg_vid_windows,
                    ) = self._prepare_intra_segment_payload(
                        action_offsets[segment_idx],
                        action_video_boundaries[segment_idx],
                        frame_start=frame_start,
                        frame_end=frame_end,
                        valid_text_tokens=valid_text_tokens,
                    )
                    if intra_seg_txt_offsets is not None:
                        total_intra_masked_segments += 1

                camera_inputs = None
                camera_has_mask = None
                if use_camera_control:
                    camera_inputs = self._prepare_camera_inputs(
                        model,
                        norm_extrinsics=sample_norm_extrinsics,
                        norm_intrinsics=sample_norm_intrinsics,
                        action_onehot=sample_action_onehot,
                        frame_start=frame_start,
                        frame_end=frame_end,
                        device=device,
                    )
                    camera_enabled = bool(
                        camera_has_list[sample_idx]
                        and camera_keep is not None
                        and segment_idx < len(camera_keep)
                        and camera_keep[segment_idx]
                    )
                    camera_has_mask = torch.tensor(
                        [camera_enabled],
                        dtype=torch.bool,
                        device=device,
                    )
                    if camera_inputs is None:
                        camera_has_mask = torch.zeros_like(camera_has_mask)
                        camera_enabled = False
                    if camera_enabled:
                        total_camera_segments += 1

                # Every cached tail belongs to a previous segment at this point.
                # Mask it before both the gradient-bearing forward and the
                # no-grad cache-write forward so deeper-layer cached K/V states
                # are produced under the same cross-segment visibility rule.
                if self.cross_segment_text_head_only:
                    self._hide_cached_action_tails(cache_mask_list, cached_action_tails)
                segment_rec_mask = (
                    sample_rec_mask[:, frame_start:frame_end]
                    if sample_rec_mask is not None
                    else None
                )

                total_segments += 1

                # ---- Hybrid prefix: optional standard FM loss + GT prefill ----
                if segment_idx < prefix_segments:
                    if self.prefix_flow_matching_loss:
                        # This is deliberately a separate update_cache=False
                        # forward.  The cache input below is constrained to the
                        # low-noise context window, whereas standard FM training
                        # samples from the full configured timestep distribution.
                        prefix_train_t = sample_timesteps(
                            (1,), device, self.timestep_sample_method
                        )
                        prefix_train_t = apply_timestep_shift(
                            prefix_train_t, self.timestep_shift
                        )
                        if parallel_dims is not None:
                            prefix_train_t = sync_object_for_parallel_training(
                                prefix_train_t, parallel_dims=parallel_dims
                            )
                        prefix_noisy, prefix_train_noise = add_flow_noise(
                            clean_segment, prefix_train_t
                        )
                        prefix_model_t = (
                            (prefix_train_t * self.training_timesteps)
                            .expand(1)
                            .to(device)
                        )

                        # A prefix segment's cache entry comes from its GT
                        # latent, not from a prediction, so a dropped segment
                        # needs no conditional row here: run the unconditional
                        # text alone (dropping the per-action payload with it,
                        # since the empty block carries no actions).
                        prefix_text = segment_text
                        prefix_mask = segment_mask
                        prefix_offsets = intra_seg_txt_offsets
                        prefix_windows = intra_seg_vid_windows
                        if text_dropped:
                            prefix_text = uncond_text_states.to(segment_text.dtype)
                            prefix_mask = uncond_text_mask.to(segment_mask.dtype)
                            prefix_offsets = None
                            prefix_windows = None

                        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                            prefix_output = self._cache_forward(
                                model,
                                hidden_states=prefix_noisy,
                                timestep=prefix_model_t,
                                text_states=prefix_text,
                                text_mask=prefix_mask,
                                kv_cache=kv_cache,
                                cache_valid_len=cache_valid_len,
                                update_cache=False,
                                global_frame_counter=global_frame_counter,
                                cache_mask_list=cache_mask_list,
                                intra_seg_txt_offsets=prefix_offsets,
                                intra_seg_vid_windows=prefix_windows,
                                camera_inputs=camera_inputs,
                                camera_has_mask=camera_has_mask,
                            )
                            prefix_velocity_pred = prefix_output[0]

                        prefix_target_velocity = prefix_train_noise - clean_segment
                        (
                            prefix_loss_sum,
                            prefix_loss_count,
                            prefix_loss,
                        ) = self._velocity_loss_stats(
                            prefix_velocity_pred,
                            prefix_target_velocity,
                            segment_rec_mask,
                        )
                        loss_sums.append(
                            prefix_loss_sum.detach()
                            if streaming_backward
                            else prefix_loss_sum
                        )
                        loss_counts.append(prefix_loss_count)
                        seg_losses_log.append(prefix_loss.detach())
                        total_prefix_loss_segments += 1
                        if streaming_backward:
                            with _scope("ssf/segment_streaming_backward"):
                                common_backward_only(
                                    prefix_loss_sum * streaming_backward_scale,
                                    model,
                                    args,
                                    logger,
                                    label=(f"sample{sample_idx}/prefix{segment_idx}"),
                                    is_last_backward=(
                                        streaming_backward_index + 1
                                        == streaming_backward_total
                                    ),
                                )
                                streaming_backward_index += 1
                            _mem_mark(
                                f"sample{sample_idx}/segment{segment_idx}/"
                                "after_streaming_backward"
                            )

                    # Defensive: a prefix segment is never the final one, since
                    # prefix_segments < num_segments. Mirror the draws anyway so
                    # the guard can never desync the sampling stream.
                    if segment_idx == last_cache_write_segment:
                        self._sample_context_low_t(device, parallel_dims)
                        torch.randn_like(clean_segment)
                        total_gt_prefix_segments += 1
                        continue
                    with torch.no_grad():
                        low_t = self._sample_context_low_t(device, parallel_dims)
                        prefix_noise = torch.randn_like(clean_segment)
                        context_latent, _ = add_flow_noise(
                            clean_segment, low_t, noise=prefix_noise
                        )
                        context_latent = context_latent.to(self.cache_dtype)
                        cache_timestep = (low_t * self.training_timesteps).to(device)
                        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                            _, new_tokens = self._cache_forward(
                                model,
                                hidden_states=context_latent,
                                timestep=cache_timestep,
                                text_states=segment_text,
                                text_mask=segment_mask,
                                kv_cache=kv_cache,
                                cache_valid_len=cache_valid_len,
                                update_cache=True,
                                global_frame_counter=global_frame_counter,
                                cache_mask_list=cache_mask_list,
                                intra_seg_txt_offsets=intra_seg_txt_offsets,
                                intra_seg_vid_windows=intra_seg_vid_windows,
                                camera_inputs=camera_inputs,
                                camera_has_mask=camera_has_mask,
                            )
                    if (
                        self.cross_segment_text_head_only
                        and self._record_cached_action_tail(
                            cached_action_tails,
                            cache_valid_len=cache_valid_len,
                            new_tokens=new_tokens,
                            text_block_len=text_len,
                            action_start=action_start,
                            valid_text_tokens=valid_text_tokens,
                        )
                    ):
                        total_cached_action_tails += 1
                    cache_valid_len += int(new_tokens)
                    global_frame_counter += frame_end - frame_start
                    total_gt_prefix_segments += 1
                    continue

                # Per-segment timestep (scalar), SP-synced.
                timestep = sample_timesteps((1,), device, self.timestep_sample_method)
                timestep = apply_timestep_shift(timestep, self.timestep_shift)
                if parallel_dims is not None:
                    timestep = sync_object_for_parallel_training(
                        timestep, parallel_dims=parallel_dims
                    )

                noisy_segment, noise = add_flow_noise(clean_segment, timestep)
                model_timestep = (
                    (timestep * self.training_timesteps).expand(1).to(device)
                )

                # ---- Grad forward (read history + current, no cache write) ----
                # ``velocity_pred`` feeds the loss, ``cache_velocity_pred`` feeds
                # the cache write. They are the same tensor unless this segment's
                # text is dropped, in which case the loss takes the
                # unconditional row and the cache takes the conditional one.
                writes_cache = segment_idx != last_cache_write_segment
                with _scope("ssf/segment_grad_forward"):
                    with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                        if text_dropped and writes_cache:
                            (
                                cache_velocity_pred,
                                velocity_pred,
                            ) = self._cfg_branch_forward(
                                model,
                                hidden_states=noisy_segment,
                                timestep=model_timestep,
                                segment_text=segment_text,
                                segment_mask=segment_mask,
                                uncond_text_states=uncond_text_states,
                                uncond_text_mask=uncond_text_mask,
                                kv_cache=kv_cache,
                                cache_valid_len=cache_valid_len,
                                global_frame_counter=global_frame_counter,
                                cache_mask_list=cache_mask_list,
                                intra_seg_txt_offsets=intra_seg_txt_offsets,
                                intra_seg_vid_windows=intra_seg_vid_windows,
                                camera_inputs=camera_inputs,
                                camera_has_mask=camera_has_mask,
                            )
                            total_cfg_branch_segments += 1
                        else:
                            # Dropped segments whose cache write is skipped
                            # anyway need no conditional row: run the
                            # unconditional text alone.
                            forward_text = segment_text
                            forward_mask = segment_mask
                            forward_offsets = intra_seg_txt_offsets
                            forward_windows = intra_seg_vid_windows
                            if text_dropped:
                                forward_text = uncond_text_states.to(segment_text.dtype)
                                forward_mask = uncond_text_mask.to(segment_mask.dtype)
                                forward_offsets = None
                                forward_windows = None
                            output = self._cache_forward(
                                model,
                                hidden_states=noisy_segment,
                                timestep=model_timestep,
                                text_states=forward_text,
                                text_mask=forward_mask,
                                kv_cache=kv_cache,
                                cache_valid_len=cache_valid_len,
                                update_cache=False,
                                global_frame_counter=global_frame_counter,
                                cache_mask_list=cache_mask_list,
                                intra_seg_txt_offsets=forward_offsets,
                                intra_seg_vid_windows=forward_windows,
                                camera_inputs=camera_inputs,
                                camera_has_mask=camera_has_mask,
                            )
                            velocity_pred = output[0]
                            cache_velocity_pred = velocity_pred
                _mem_mark(f"sample{sample_idx}/segment{segment_idx}/after_grad_forward")

                target_velocity = noise - clean_segment
                (
                    segment_loss_sum,
                    segment_loss_count,
                    segment_loss,
                ) = self._velocity_loss_stats(
                    velocity_pred,
                    target_velocity,
                    segment_rec_mask,
                )
                loss_sums.append(
                    segment_loss_sum.detach()
                    if streaming_backward
                    else segment_loss_sum
                )
                loss_counts.append(segment_loss_count)
                seg_losses_log.append(segment_loss.detach())
                total_rollout_segments += 1

                # Cache tensors and masks are mutable checkpoint-closure inputs.
                # In streaming mode, finish this independent graph before the
                # no-grad cache write or the next segment mutates them. Keep the
                # detached prediction needed for the cache write.
                detached_velocity = cache_velocity_pred.detach()
                if streaming_backward:
                    with _scope("ssf/segment_streaming_backward"):
                        common_backward_only(
                            segment_loss_sum * streaming_backward_scale,
                            model,
                            args,
                            logger,
                            label=(f"sample{sample_idx}/segment{segment_idx}"),
                            is_last_backward=(
                                streaming_backward_index + 1 == streaming_backward_total
                            ),
                        )
                        streaming_backward_index += 1
                    _mem_mark(
                        f"sample{sample_idx}/segment{segment_idx}/"
                        "after_streaming_backward"
                    )

                # ---- Context write (no grad): self-generated latent into cache ----
                # Nothing reads the last segment's cache entry: the loop ends
                # here and the cache is released right after. The draw is still
                # consumed so later samples see the same sampling stream.
                if not writes_cache:
                    self._sample_context_low_t(device, parallel_dims)
                    continue
                with _scope("ssf/cache_write_forward"):
                    with torch.no_grad():
                        low_t = self._sample_context_low_t(device, parallel_dims)
                        context_latent = self._build_self_generated_context(
                            noisy_segment=noisy_segment,
                            noise=noise,
                            timestep=timestep,
                            low_t=low_t,
                            velocity_pred=detached_velocity,
                        ).to(self.cache_dtype)
                        cache_timestep = (low_t * self.training_timesteps).to(device)
                        with torch.amp.autocast("cuda", dtype=self.compute_dtype):
                            _, new_tokens = self._cache_forward(
                                model,
                                hidden_states=context_latent,
                                timestep=cache_timestep,
                                text_states=segment_text,
                                text_mask=segment_mask,
                                kv_cache=kv_cache,
                                cache_valid_len=cache_valid_len,
                                update_cache=True,
                                global_frame_counter=global_frame_counter,
                                cache_mask_list=cache_mask_list,
                                intra_seg_txt_offsets=intra_seg_txt_offsets,
                                intra_seg_vid_windows=intra_seg_vid_windows,
                                camera_inputs=camera_inputs,
                                camera_has_mask=camera_has_mask,
                            )
                _mem_mark(f"sample{sample_idx}/segment{segment_idx}/after_cache_write")
                if (
                    self.cross_segment_text_head_only
                    and self._record_cached_action_tail(
                        cached_action_tails,
                        cache_valid_len=cache_valid_len,
                        new_tokens=new_tokens,
                        text_block_len=text_len,
                        action_start=action_start,
                        valid_text_tokens=valid_text_tokens,
                    )
                ):
                    total_cached_action_tails += 1
                cache_valid_len += int(new_tokens)
                global_frame_counter += frame_end - frame_start

            max_cache_tokens_used = max(max_cache_tokens_used, cache_valid_len)
            if streaming_backward:
                # All graphs that referenced this sample's mutable cache have
                # completed, so release it before allocating the next sample.
                del kv_cache, cache_mask_list

        if total_rollout_segments == 0:
            raise ValueError("SegmentSelfForcing: no valid rollout segments in batch.")
        if not loss_sums:
            raise ValueError("SegmentSelfForcing: no loss-producing segments.")
        if streaming_backward and streaming_backward_index != streaming_backward_total:
            raise RuntimeError(
                "SegmentSelfForcing streaming backward count mismatch: "
                f"completed={streaming_backward_index}, "
                f"expected={streaming_backward_total}"
            )

        total_loss_count = torch.stack(loss_counts).sum()
        global_loss_count = (
            streaming_total_loss_count
            if streaming_backward
            else global_token_count(total_loss_count, group=loss_dp_group)
        )
        if global_loss_count.item() <= 0:
            raise ValueError(
                "SegmentSelfForcing: recreation mask excluded every loss element "
                "across all data replicas."
            )
        loss = global_token_mean_loss(
            torch.stack(loss_sums).sum(),
            total_loss_count,
            group=loss_dp_group,
            global_valid_tokens=global_loss_count,
        )
        sync_object_for_parallel_training(
            loss.detach(),
            parallel_dims=parallel_dims,
            debug_with_check=True,
        )

        _mem_mark("before_backward")
        with _scope("ssf/backward_and_step"):
            if streaming_backward:
                grad_norm = common_finalize_backward_and_step(
                    model,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    args,
                    logger,
                    reshard_fsdp_after_backward=True,
                )
            else:
                grad_norm = common_backward_and_step(
                    loss,
                    model,
                    optimizer,
                    lr_scheduler,
                    global_step,
                    args,
                    logger,
                )
        _mem_mark("after_backward")

        vis_dict = {
            "loss_flow": loss.item(),
            "ssf/num_segments": float(total_segments),
            "ssf/num_rollout_segments": float(total_rollout_segments),
            "ssf/num_gt_prefix_segments": float(total_gt_prefix_segments),
            "ssf/num_prefix_loss_segments": float(total_prefix_loss_segments),
            "ssf/num_loss_segments": float(
                total_rollout_segments + total_prefix_loss_segments
            ),
            "ssf/mean_gt_prefix_segments": float(
                total_gt_prefix_segments / max(num_packed, 1)
            ),
            "ssf/num_packed": float(num_packed),
            "ssf/max_cache_tokens": float(max_cache_tokens_used),
            "ssf/intra_masked_segments": float(total_intra_masked_segments),
            "ssf/cached_action_tails": float(total_cached_action_tails),
            "ssf/text_dropped_segments": float(total_text_dropped_segments),
            "ssf/cfg_branch_segments": float(total_cfg_branch_segments),
            "ssf/recreation_mask_enabled": float(use_recreation_mask),
            "ssf/valid_loss_elements": float(total_loss_count.item()),
            "ssf/global_valid_loss_elements": float(global_loss_count.item()),
            "ssf/mean_seg_loss": float(torch.stack(seg_losses_log).mean().item()),
        }
        if use_camera_control:
            vis_dict["ssf/camera_segments"] = float(total_camera_segments)
            vis_dict.update(camera_diag)
        if logger is not None and global_step < 3:
            logger.info(
                "[SegmentSelfForcing] "
                f"step={global_step} packed={num_packed} "
                f"segments={total_segments} "
                f"gt_prefix={total_gt_prefix_segments} "
                f"prefix_loss={total_prefix_loss_segments} "
                f"self_rollout={total_rollout_segments} "
                f"loss_segments={total_rollout_segments + total_prefix_loss_segments} "
                f"intra_masked={total_intra_masked_segments} "
                f"action_tails={total_cached_action_tails} "
                f"camera_segments={total_camera_segments} "
                f"segment_dropout={self.segment_dropout} "
                f"text_dropped={total_text_dropped_segments} "
                f"cfg_branch={total_cfg_branch_segments} "
                f"end_noise={self.end_noise} "
                f"recreation_mask={use_recreation_mask} "
                f"valid_loss_elements={int(total_loss_count.item())}"
            )
        terms = {
            "loss": loss.detach(),
            "loss_flow": loss.detach(),
            "vis_dict": vis_dict,
        }
        return loss.item(), terms, grad_norm


def create_segment_self_forcing_transport(args, logger=None):
    """Build a SegmentSelfForcingTransport from config args."""
    segment_self_forcing = getattr(args, "segment_self_forcing", args)
    precision_policy = resolve_dit_training_precision(
        getattr(getattr(args, "dit", None), "master_weight_type", "bf16")
    )
    segment_dropout = bool(getattr(segment_self_forcing, "segment_dropout", False))
    if segment_dropout:
        # The dataset empties every segment of a sample at once and tokenizes
        # before this transport sees the batch, so its dropout would destroy the
        # conditional text that the per-segment conditional row needs.
        dataset_uncond_p = float(_resolve_nested_config(args, "uncond_p", 0.0) or 0.0)
        if dataset_uncond_p > 0.0:
            raise ValueError(
                "segment_self_forcing.segment_dropout=true replaces the "
                "dataset's whole-sample text dropout with per-segment dropout, "
                f"but dataset.uncond_p={dataset_uncond_p} still empties every "
                "segment of a sample before tokenization. Set dataset.uncond_p=0 "
                "and tune segment_self_forcing.segment_dropout_p instead."
            )
    return SegmentSelfForcingTransport(
        timestep_sample_method=getattr(
            segment_self_forcing,
            "timestep_sample_method",
            getattr(args, "timestep_sample_method", "uniform"),
        ),
        timestep_shift=getattr(
            segment_self_forcing,
            "timestep_shift",
            getattr(args.scheduler, "flow_shift_video", 3.0),
        ),
        context_low_t_min=getattr(segment_self_forcing, "context_low_t_min", 0.1),
        context_low_t_max=getattr(segment_self_forcing, "context_low_t_max", 0.3),
        df_random_prefix=getattr(segment_self_forcing, "df_random_prefix", False),
        prefix_flow_matching_loss=getattr(
            segment_self_forcing, "prefix_flow_matching_loss", False
        ),
        intra_segment_action_mask=_resolve_nested_config(
            args, "intra_segment_action_mask", False
        ),
        cross_segment_text_head_only=_resolve_nested_config(
            args, "cross_segment_text_head_only", True
        ),
        training_timesteps=getattr(args.scheduler, "training_timesteps", 1000),
        video_shift=getattr(args.scheduler, "flow_shift_video", 1.0),
        compute_dtype=precision_policy.compute_dtype,
        cache_dtype=precision_policy.cache_dtype,
        streaming_backward=getattr(segment_self_forcing, "streaming_backward", False),
        skip_last_segment_cache_write=getattr(
            segment_self_forcing, "skip_last_segment_cache_write", True
        ),
        end_noise=getattr(segment_self_forcing, "end_noise", True),
        segment_dropout=segment_dropout,
        segment_dropout_p=getattr(segment_self_forcing, "segment_dropout_p", 0.1),
        logger=logger,
    )
