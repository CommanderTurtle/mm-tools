"""Teacher-Forcing DMD pipeline.

Variant of ``SelfForcingTrainingPipeline`` for **teacher-forcing DMD**: instead
of letting the student roll out the entire video and use its own predictions
to fill the KV cache, we prime the cache with the **ground-truth** clean
latents of all segments before the target segment ``k``, and only let the
student denoise segment ``k``. The DMD distribution-matching loss is then
computed on the student's segment-``k`` prediction.

Segments here are **variable-length**: their boundaries come from the
dataloader (``prompt_video_boundaries``) or from a uniform fixed-chunk schedule
chosen by the transport — exactly the same boundary semantics as
``UnifiedVideoGen._compute_boundaries``. There is **no** fixed
``num_frame_per_block`` constraint at the pipeline level; the parent class's
``num_frame_per_block`` attribute is only used for cache sizing helpers.

This pipeline is a subclass of ``SelfForcingTrainingPipeline`` so we fully
reuse the cache layout (``_initialize_kv_cache``), RoPE accounting
(``_get_rope_params``), and frame-seq-length plumbing.
"""

from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.profiler import record_function

from hymm.models.self_forcing.causal_training_pipeline import (
    KVCacheWrapper,
    SelfForcingTrainingPipeline,
    modelname2config,
)
from hymm.distributed.parallel_states import nccl_info


class TeacherForcingDMDPipeline(SelfForcingTrainingPipeline):
    """Teacher-Forcing rollout for DMD distillation, variable-segment aware.

    Key differences from ``SelfForcingTrainingPipeline``:
      * Segments are variable-length (driven by ``boundaries``).
      * Blocks 0..k-1 are *teacher-forced*: their GT clean latents are written
        into the KV cache (``context_noise`` honored, default 0 == clean).
      * Returns the predicted segment-``k`` latent, ready to be fed into the
        DMD ``compute_distribution_matching_loss`` / critic paths.
      * Optionally accepts a *per-segment* ``conditional_dict`` list so each
        segment can use its own text states (multi-prompt training).
    """

    @staticmethod
    def _with_local_segment_extra(
        conditional_dict: dict,
        num_frames: int,
        frame_start: int = 0,
    ) -> dict:
        """Attach local one-segment boundaries for magi_mp_causal forwards.

        ``frame_start`` is the segment's first latent frame in SAMPLE coords.
        Dataset action windows are sample-level while this forward covers ONE
        segment and expects segment-relative frames, so windows are re-based by
        ``frame_start`` (mirrors stage6's ``_prepare_intra_segment_payload``).
        The default 0 is correct only when the segment starts at frame 0, and is
        inert unless the caller actually put ``intra_seg_vid_windows`` in
        ``extra_kwargs``.
        """
        result = dict(conditional_dict)
        extra_kwargs = dict(result.get("extra_kwargs", {}) or {})
        extra_kwargs.setdefault("chunk_size", 0)
        extra_kwargs["prompt_video_boundaries"] = [0, int(num_frames)]
        windows = extra_kwargs.get("intra_seg_vid_windows", None)
        if windows is not None:
            base = int(frame_start)
            local_windows = []
            for start, end, action_idx in windows:
                start = max(0, min(int(start) - base, int(num_frames)))
                end = max(0, min(int(end) - base, int(num_frames)))
                if end > start:
                    local_windows.append([start, end, int(action_idx)])
            # The range builder emits ONLY the supplied windows, so a
            # non-contiguous tiling would silently drop video queries. Require
            # an exact [0, num_frames) cover, matching stage6.
            if local_windows and (
                local_windows[0][0] != 0
                or local_windows[-1][1] != int(num_frames)
                or any(
                    local_windows[i][1] != local_windows[i + 1][0]
                    for i in range(len(local_windows) - 1)
                )
            ):
                local_windows = []
            extra_kwargs["intra_seg_vid_windows"] = local_windows or None
            if not local_windows:
                extra_kwargs.pop("intra_seg_txt_offsets", None)
        result["extra_kwargs"] = extra_kwargs
        return result

    @staticmethod
    def _action_mask_kwargs(conditional_dict: dict) -> dict:
        """Lift the intra-segment action-mask payload out of ``extra_kwargs``.

        Returns the four cache-path kwarg names at their inert defaults when the
        conditional dict carries no payload, so splatting the result into
        ``forward_cache_update`` never changes behaviour on its own.
        """
        extra_kwargs = (
            conditional_dict.get("extra_kwargs", {}) if conditional_dict else {}
        ) or {}
        return {
            "hide_cached_action_tails": bool(
                extra_kwargs.get("cross_segment_text_head_only", False)
            ),
            "current_action_start": int(extra_kwargs.get("current_action_start", -1)),
            "intra_seg_txt_offsets": extra_kwargs.get("intra_seg_txt_offsets", None),
            "intra_seg_vid_windows": extra_kwargs.get("intra_seg_vid_windows", None),
        }

    @staticmethod
    def _text_tokens_per_segment(conditional_dicts: List[dict]) -> int:
        max_tokens = 0
        for cd in conditional_dicts:
            text_states = cd.get("text_states", None)
            if text_states is not None and text_states.dim() >= 2:
                max_tokens = max(max_tokens, int(text_states.shape[-2]))
        return max_tokens

    def _make_context_timestep(
        self,
        *,
        batch_size: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Sample the noise timestep used when teacher-forcing context.

        Backward-compatible default is a fixed scalar ``context_noise``. For
        ablations, ``context_noise_mode`` can be:
          - fixed:   use ``context_noise`` for every context frame
          - choice:  sample from ``context_noise_values``
          - uniform: sample from [``context_noise_min``, ``context_noise_max``]

        By default one timestep is sampled per context segment and expanded to
        all frames in that segment. Set ``context_noise_per_frame`` for
        frame-wise sampling.
        """
        mode = str(getattr(self, "context_noise_mode", "fixed") or "fixed").lower()
        per_frame = bool(getattr(self, "context_noise_per_frame", False))
        sample_shape = (batch_size, num_frames) if per_frame else (batch_size, 1)

        if mode in ("fixed", "constant"):
            value = float(getattr(self, "context_noise", 0.0) or 0.0)
            timestep = torch.full(
                sample_shape, value, device=device, dtype=torch.float32
            )
        elif mode in ("choice", "discrete", "list"):
            values = getattr(self, "context_noise_values", None)
            if values is None or len(values) == 0:
                values = [float(getattr(self, "context_noise", 0.0) or 0.0)]
            values = torch.as_tensor(list(values), device=device, dtype=torch.float32)
            idx = torch.randint(0, values.numel(), sample_shape, device=device)
            timestep = values[idx]
        elif mode in ("uniform", "continuous"):
            low = float(getattr(self, "context_noise_min", 0.0) or 0.0)
            high = float(getattr(self, "context_noise_max", low) or low)
            if high < low:
                raise ValueError(
                    f"context_noise_max must be >= context_noise_min, got {high} < {low}"
                )
            timestep = torch.empty(
                sample_shape, device=device, dtype=torch.float32
            ).uniform_(low, high)
        else:
            raise ValueError(
                f"Unsupported context_noise_mode={mode!r}. "
                "Supported values: fixed, choice, uniform."
            )

        if not per_frame and num_frames != 1:
            timestep = timestep.expand(batch_size, num_frames)
        return timestep

    @staticmethod
    def _is_clean_context_timestep(timestep: torch.Tensor) -> bool:
        return bool(torch.count_nonzero(timestep).item() == 0)

    def needs_shared_context_prefix(self) -> bool:
        """Whether context prefill should reuse one sampled noisy prefix.

        Clean fixed context is deterministic, so the old path is equivalent and
        avoids holding an extra prefix copy. Any non-zero/noisy context must be
        shared across generator, fake score, and real score.
        """
        if float(getattr(self, "context_noise_sigma", 0.0) or 0.0) > 0.0:
            return True
        mode = str(getattr(self, "context_noise_mode", "fixed") or "fixed").lower()
        if mode in ("fixed", "constant"):
            return float(getattr(self, "context_noise", 0.0) or 0.0) != 0.0
        if mode in ("choice", "discrete", "list"):
            values = getattr(self, "context_noise_values", None)
            if values is None or len(values) == 0:
                values = [float(getattr(self, "context_noise", 0.0) or 0.0)]
            return any(float(v) != 0.0 for v in values)
        if mode in ("uniform", "continuous"):
            low = float(getattr(self, "context_noise_min", 0.0) or 0.0)
            high = float(getattr(self, "context_noise_max", low) or low)
            return low != 0.0 or high != 0.0
        return True

    def _sample_context_chunk(
        self,
        clean_chunk: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = clean_chunk.shape[0]
        num_frames = clean_chunk.shape[1]
        device = clean_chunk.device
        context_noise_sigma = float(getattr(self, "context_noise_sigma", 0.0) or 0.0)
        if context_noise_sigma > 0.0:
            ctx_timestep = torch.full(
                (batch_size, num_frames),
                context_noise_sigma * 1000.0,
                device=device,
                dtype=torch.float32,
            )
            eps = torch.randn_like(clean_chunk)
            ctx_input = (
                (1.0 - context_noise_sigma) * clean_chunk.float()
                + context_noise_sigma * eps.float()
            ).to(clean_chunk.dtype)
            return ctx_input.detach(), ctx_timestep.detach()

        ctx_timestep = self._make_context_timestep(
            batch_size=batch_size,
            num_frames=num_frames,
            device=device,
        )

        if self._is_clean_context_timestep(ctx_timestep):
            ctx_input = clean_chunk
        else:
            ctx_input = self.scheduler.add_noise(
                clean_chunk.flatten(0, 1),
                torch.randn_like(clean_chunk.flatten(0, 1)),
                ctx_timestep.flatten(),
            ).unflatten(0, clean_chunk.shape[:2])

        return ctx_input.detach(), ctx_timestep.detach()

    @staticmethod
    def _get_context_chunk_from_prefix(
        *,
        clean_chunk: torch.Tensor,
        context_entry: Optional[Tuple[torch.Tensor, torch.Tensor]],
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if context_entry is None:
            return None

        ctx_input, ctx_timestep = context_entry
        if tuple(ctx_input.shape) != tuple(clean_chunk.shape):
            raise ValueError(
                "Shared context latent shape mismatch: "
                f"expected {tuple(clean_chunk.shape)}, got {tuple(ctx_input.shape)}"
            )
        expected_timestep_shape = (int(clean_chunk.shape[0]), int(clean_chunk.shape[1]))
        if tuple(ctx_timestep.shape) != expected_timestep_shape:
            raise ValueError(
                "Shared context timestep shape mismatch: "
                f"expected {expected_timestep_shape}, got {tuple(ctx_timestep.shape)}"
            )

        return (
            ctx_input.to(device=clean_chunk.device, dtype=clean_chunk.dtype).detach(),
            ctx_timestep.to(device=clean_chunk.device, dtype=torch.float32).detach(),
        )

    @torch.no_grad()
    def build_shared_context_prefix(
        self,
        *,
        clean_video_latent: torch.Tensor,
        boundaries: List[int],
        target_segment_idx: int,
        conditional_dict_per_segment: List[dict],
    ) -> Optional[List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Build one shared context prefix for generator, fake, and real models."""
        if target_segment_idx <= 0 or not self.needs_shared_context_prefix():
            return None

        target_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else clean_video_latent.dtype
        )
        if clean_video_latent.dtype != target_dtype:
            clean_video_latent = clean_video_latent.to(dtype=target_dtype)

        mode = str(getattr(self, "context_cache_mode", "noisy") or "noisy").lower()
        if mode in ("noisy", "direct", "xt"):
            context_prefix = []
            for j in range(target_segment_idx):
                seg_start_j = boundaries[j]
                seg_end_j = boundaries[j + 1]
                clean_chunk_j = clean_video_latent[:, seg_start_j:seg_end_j]
                context_prefix.append(self._sample_context_chunk(clean_chunk_j))
            return context_prefix

        if mode not in ("denoised", "generated", "x0"):
            raise ValueError(
                f"Unsupported context_cache_mode={mode!r}. "
                "Supported values: noisy, denoised."
            )

        if float(getattr(self, "context_noise_sigma", 0.0) or 0.0) > 0.0:
            raise ValueError(
                "context_noise_sigma is incompatible with "
                f"context_cache_mode={mode!r}: that mode caches a re-denoised "
                "x0 labelled timestep=0. Use context_cache_mode=noisy."
            )

        batch_size, _F_lat, _C, H, W = clean_video_latent.shape
        prefix_frames = int(boundaries[target_segment_idx])
        if "nomas" not in self.model_name.lower():
            raise NotImplementedError(
                "Denoised shared context prefix is only implemented for Nomas "
                "cache models."
            )

        if len(conditional_dict_per_segment) < target_segment_idx:
            raise ValueError(
                f"conditional_dict_per_segment has {len(conditional_dict_per_segment)} "
                f"entries, but target_segment_idx={target_segment_idx}"
            )

        max_cache_tokens = self._max_cache_tokens(
            model=self.generator,
            total_frames=prefix_frames,
            latent_height=H,
            latent_width=W,
            num_segments=int(target_segment_idx),
            conditional_dicts=conditional_dict_per_segment,
        )
        kv_cache, cache_mask_list, _tokens_per_frame = self._allocate_nomas_kv_cache(
            model=self.generator,
            batch_size=batch_size,
            dtype=target_dtype,
            device=clean_video_latent.device,
            latent_height=H,
            latent_width=W,
            max_cache_tokens=max_cache_tokens,
        )

        context_prefix = []
        cache_frame_count = 0
        cache_token_count = 0
        for j in range(target_segment_idx):
            seg_start_j = boundaries[j]
            seg_end_j = boundaries[j + 1]
            clean_chunk_j = clean_video_latent[:, seg_start_j:seg_end_j]
            num_frames = clean_chunk_j.shape[1]
            cd_j = self._with_local_segment_extra(
                conditional_dict_per_segment[j],
                num_frames,
                frame_start=int(seg_start_j),
            )

            noisy_context, noisy_timestep = self._sample_context_chunk(clean_chunk_j)
            _cvfl, _gfc = self._get_rope_params(cache_frame_count)
            _, denoised_context = self.generator(
                latent=noisy_context,
                timestep=noisy_timestep,
                kv_cache_list=kv_cache,
                cache_valid_len=cache_token_count,
                cache_valid_frame_len=_cvfl,
                global_frame_counter=_gfc,
                update_cache=False,
                cache_txt=True,
                cache_mask_list=cache_mask_list,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
                **self._action_mask_kwargs(cd_j),
                **cd_j,
            )

            cache_timestep = torch.zeros_like(noisy_timestep)
            _cvfl, _gfc = self._get_rope_params(cache_frame_count)
            _, new_tokens = self.generator.forward_cache_update(
                latent=denoised_context.detach(),
                timestep=cache_timestep,
                kv_cache_list=kv_cache,
                cache_valid_len=cache_token_count,
                cache_valid_frame_len=_cvfl,
                global_frame_counter=_gfc,
                update_cache=True,
                cache_txt=True,
                cache_mask_list=cache_mask_list,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
                **self._action_mask_kwargs(cd_j),
                **cd_j,
            )
            context_prefix.append((denoised_context.detach(), cache_timestep.detach()))
            cache_frame_count += num_frames
            cache_token_count += int(new_tokens)
        return context_prefix

    def release_runtime_prefix_cache(self):
        """Drop pipeline-owned rollout cache references.

        The target forward's autograd graph keeps any tensors genuinely needed
        for backward. Clearing these owner references prevents a completed
        generator rollout cache from remaining live while DDMD builds the
        fake/real score-model prefix caches.
        """
        self.kv_cache1 = None
        self.cache_mask_list = None
        if hasattr(self, "crossattn_cache"):
            self.crossattn_cache = None

    @staticmethod
    def _allocate_nomas_kv_cache(
        *,
        model,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        latent_height: int,
        latent_width: int,
        max_cache_tokens: int,
    ):
        patch_size = model.patch_size
        heads_num = model.heads_num
        head_dim = model.hidden_size // heads_num

        if nccl_info.sp_size > 1:
            assert heads_num % nccl_info.sp_size == 0, (
                f"heads_num {heads_num} must be divisible by sp_size {nccl_info.sp_size}"
            )
            heads_num = heads_num // nccl_info.sp_size

        h_tokens = latent_height // patch_size[1]
        w_tokens = latent_width // patch_size[2]
        tokens_per_frame = h_tokens * w_tokens
        num_blocks = len(model.double_blocks)

        kv_cache = []
        cache_mask_list = []
        for _ in range(num_blocks):
            cache_dict = {
                "k": torch.zeros(
                    [batch_size, max_cache_tokens, heads_num, head_dim],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, max_cache_tokens, heads_num, head_dim],
                    dtype=dtype,
                    device=device,
                ),
            }
            kv_cache.append(KVCacheWrapper(cache_dict))
            cache_mask_list.append(
                torch.zeros(
                    [batch_size, max_cache_tokens], dtype=torch.bool, device=device
                )
            )

        return kv_cache, cache_mask_list, tokens_per_frame

    def _max_cache_tokens(
        self,
        *,
        model,
        total_frames: int,
        latent_height: int,
        latent_width: int,
        num_segments: int,
        conditional_dicts: List[dict],
    ) -> int:
        patch_size = model.patch_size
        tokens_per_frame = (latent_height // patch_size[1]) * (
            latent_width // patch_size[2]
        )
        text_tokens = self._text_tokens_per_segment(conditional_dicts)
        # Allocate for the full video image cache plus one text block per
        # segment. This covers the worst case where k is the last segment.
        return total_frames * tokens_per_frame + num_segments * text_tokens

    @torch.no_grad()
    def build_prefix_cache_kwargs(
        self,
        *,
        model,
        clean_video_latent: torch.Tensor,
        boundaries: List[int],
        target_segment_idx: int,
        conditional_dict_per_segment: List[dict],
        context_prefix: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> dict:
        """Build a clean-prefix KV cache for a score model.

        The returned kwargs are suitable for ``DMDModelAdapter`` cache-mode
        forwards on the target segment. Caches are model-specific because K/V
        tensors depend on the model weights; generator, fake_score, and
        real_score cannot share one cache.

        Two behaviours live here, selected by ``distill_prefix_mode`` (set by the
        ddmd / ddmd_adv / pcm transports; absent for ``teacher_forcing_dmd``):

        * legacy (flag absent) — allocate for the WHOLE video, prefill with the
          fixed-``context_noise`` timestep, and return ``cache_txt: False``.
        * distill (flag set) — allocate only the prefix span, honour the
          ``context_noise_mode`` sampler / shared ``context_prefix``, forward the
          intra-segment action mask, and return ``cache_txt: True`` so the target
          segment's own text participates in the cache-mode attention.

        Keeping the legacy branch verbatim is what makes this file safe to share:
        ``teacher_forcing_dmd`` is a master training mode and must not move.
        """
        distill_mode = bool(getattr(self, "distill_prefix_mode", False))

        target_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else clean_video_latent.dtype
        )
        if clean_video_latent.dtype != target_dtype:
            clean_video_latent = clean_video_latent.to(dtype=target_dtype)

        batch_size, F_lat, _C, H, W = clean_video_latent.shape
        device = clean_video_latent.device
        num_segments = len(boundaries) - 1

        if distill_mode:
            prefix_frames = int(boundaries[target_segment_idx])
            prefix_segments = int(target_segment_idx)
            if prefix_segments <= 0 or prefix_frames <= 0:
                return {}
        if "nomas" not in self.model_name.lower():
            return {}

        if context_prefix is not None and len(context_prefix) < target_segment_idx:
            raise ValueError(
                f"context_prefix has {len(context_prefix)} entries, "
                f"but target_segment_idx={target_segment_idx}"
            )

        max_cache_tokens = self._max_cache_tokens(
            model=model,
            # Distill only ever reads the prefix, so allocating the whole video
            # would waste cache memory proportional to the untouched tail.
            total_frames=prefix_frames if distill_mode else F_lat,
            latent_height=H,
            latent_width=W,
            num_segments=prefix_segments if distill_mode else num_segments,
            conditional_dicts=conditional_dict_per_segment,
        )
        kv_cache, cache_mask_list, _tokens_per_frame = self._allocate_nomas_kv_cache(
            model=model,
            batch_size=batch_size,
            dtype=target_dtype,
            device=device,
            latent_height=H,
            latent_width=W,
            max_cache_tokens=max_cache_tokens,
        )

        cache_frame_count = 0
        cache_token_count = 0
        for j in range(target_segment_idx):
            seg_start_j = boundaries[j]
            seg_end_j = boundaries[j + 1]
            clean_chunk_j = clean_video_latent[:, seg_start_j:seg_end_j]
            num_frames = clean_chunk_j.shape[1]
            if distill_mode:
                cd_j = self._with_local_segment_extra(
                    conditional_dict_per_segment[j],
                    num_frames,
                    frame_start=int(seg_start_j),
                )
                context_entry = (
                    context_prefix[j] if context_prefix is not None else None
                )
                planned_context = self._get_context_chunk_from_prefix(
                    clean_chunk=clean_chunk_j,
                    context_entry=context_entry,
                )
                if planned_context is None:
                    ctx_input, ctx_timestep = self._sample_context_chunk(clean_chunk_j)
                else:
                    ctx_input, ctx_timestep = planned_context
                action_mask_kwargs = self._action_mask_kwargs(cd_j)
            else:
                cd_j = self._with_local_segment_extra(
                    conditional_dict_per_segment[j],
                    num_frames,
                )
                ctx_timestep = (
                    torch.ones(
                        (batch_size, num_frames),
                        device=device,
                        dtype=torch.long,
                    )
                    * self.context_noise
                )

                if self.context_noise == 0:
                    ctx_input = clean_chunk_j
                else:
                    ctx_input = self.scheduler.add_noise(
                        clean_chunk_j.flatten(0, 1),
                        torch.randn_like(clean_chunk_j.flatten(0, 1)),
                        ctx_timestep.flatten(),
                    ).unflatten(0, clean_chunk_j.shape[:2])
                action_mask_kwargs = {}

            _cvfl, _gfc = self._get_rope_params(cache_frame_count)
            _, new_tokens = model.forward_cache_update(
                latent=ctx_input,
                timestep=ctx_timestep,
                kv_cache_list=kv_cache,
                cache_valid_len=cache_token_count,
                cache_valid_frame_len=_cvfl,
                global_frame_counter=_gfc,
                update_cache=True,
                cache_txt=True,
                cache_mask_list=cache_mask_list,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
                **action_mask_kwargs,
                **cd_j,
            )
            cache_frame_count += num_frames
            cache_token_count += int(new_tokens)

        _cvfl, _gfc = self._get_rope_params(cache_frame_count)
        return {
            "kv_cache_list": kv_cache,
            "cache_valid_len": cache_token_count,
            "cache_valid_frame_len": _cvfl,
            "global_frame_counter": _gfc,
            "cache_mask_list": cache_mask_list,
            "update_cache": False,
            # Distill needs the target segment's own text inside the cache-mode
            # attention (and it is what arms the intra-segment action mask,
            # which is gated on cache_txt in nomas_cpt). The legacy
            # teacher_forcing_dmd path must keep False.
            "cache_txt": bool(distill_mode),
        }

    @torch.no_grad()
    def _prefill_with_teacher_chunk(
        self,
        *,
        clean_chunk: torch.Tensor,  # [B, segment_length, C, H, W]
        cache_frame_count: int,
        cache_token_count: int,
        conditional_dict: dict,
        # Distillation: reuse a pre-sampled (ctx_input, ctx_timestep) pair so the
        # generator / fake score / real score all see the SAME noisy context.
        # None -> sample locally, i.e. master's original behaviour.
        context_entry: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[int, int]:
        """Write a single GT clean chunk into the KV cache.

        Mirrors the cache-update branch in ``inference_with_trajectory``, but
        the input is the ground-truth clean latent rather than the student's
        own ``denoised_pred``. ``context_noise`` is honored.
        """
        batch_size = clean_chunk.shape[0]
        num_frames = clean_chunk.shape[1]
        device = clean_chunk.device
        distill_mode = bool(getattr(self, "distill_prefix_mode", False))
        conditional_dict = self._with_local_segment_extra(
            conditional_dict,
            num_frames,
        )

        planned = self._get_context_chunk_from_prefix(
            clean_chunk=clean_chunk,
            context_entry=context_entry,
        )
        if planned is not None:
            ctx_input, ctx_timestep = planned
        elif distill_mode:
            # Honour context_noise_mode / values / per_frame sampling.
            ctx_input, ctx_timestep = self._sample_context_chunk(clean_chunk)
        else:
            ctx_timestep = (
                torch.ones((batch_size, num_frames), device=device, dtype=torch.long)
                * self.context_noise
            )

            if self.context_noise == 0:
                ctx_input = clean_chunk
            else:
                ctx_input = self.scheduler.add_noise(
                    clean_chunk.flatten(0, 1),
                    torch.randn_like(clean_chunk.flatten(0, 1)),
                    ctx_timestep.flatten(),
                ).unflatten(0, clean_chunk.shape[:2])

        action_mask_kwargs = (
            self._action_mask_kwargs(conditional_dict) if distill_mode else {}
        )

        if "nomas" in self.model_name.lower():
            _cvfl, _gfc = self._get_rope_params(cache_frame_count)
            _, new_tokens = self.generator.forward_cache_update(
                latent=ctx_input,
                timestep=ctx_timestep,
                kv_cache_list=self.kv_cache1,
                cache_valid_len=cache_token_count,
                cache_valid_frame_len=_cvfl,
                global_frame_counter=_gfc,
                update_cache=True,
                cache_txt=True,
                cache_mask_list=self.cache_mask_list,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
                **action_mask_kwargs,
                **conditional_dict,
            )
        else:
            self.generator(
                noisy_image_or_video=ctx_input,
                conditional_dict=conditional_dict,
                timestep=ctx_timestep,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=cache_frame_count * self.frame_seq_length,
            )
            return cache_frame_count + num_frames, cache_token_count

        return cache_frame_count + num_frames, cache_token_count + int(new_tokens)

    def _denoise_target_chunk(
        self,
        *,
        noise: torch.Tensor,  # [B, segment_length, C, H, W]
        conditional_dict: dict,
        unconditional_dict: Optional[dict],
        cache_frame_count: int,
        cache_token_count: int,
        exit_flag_idx: int,
        cfg_scale: float,
        requires_grad: bool = True,
        update_target_cache: bool = False,
    ) -> Tuple[torch.Tensor, int]:
        """Run the student through ``denoising_step_list`` for the target chunk.

        The last step (``exit_flag_idx``) is computed *with* gradient enabled.
        Earlier steps run no-grad. Optional CFG is applied with ``cfg_scale``
        when ``unconditional_dict`` is provided.

        ``cache_frame_count`` is the number of latent frames already written
        into the KV cache (used as the RoPE / global-frame anchor for the
        target chunk's queries). Length of the chunk is taken from
        ``noise.shape[1]`` so variable segment sizes work transparently.

        Returns ``(denoised_pred, new_cache_frame_count)``. Segment-wise DMD
        consumes only the target segment, so the target cache update is
        disabled by default; callers that continue autoregressively can opt in.
        """
        batch_size = noise.shape[0]
        current_num_frames = noise.shape[1]
        device = noise.device
        conditional_dict = self._with_local_segment_extra(
            conditional_dict,
            current_num_frames,
        )
        if unconditional_dict is not None:
            unconditional_dict = self._with_local_segment_extra(
                unconditional_dict,
                current_num_frames,
            )

        noisy_input = noise

        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = index == exit_flag_idx
            timestep = (
                torch.ones(
                    (batch_size, current_num_frames),
                    device=device,
                    dtype=torch.long,
                )
                * current_timestep
            )

            should_grad = requires_grad and exit_flag

            with torch.set_grad_enabled(should_grad):
                _cvfl, _gfc = self._get_rope_params(cache_frame_count)
                _, denoised_pred = self.generator(
                    latent=noisy_input,
                    timestep=timestep,
                    kv_cache_list=self.kv_cache1,
                    cache_valid_len=cache_token_count,
                    cache_valid_frame_len=_cvfl,
                    global_frame_counter=_gfc,
                    update_cache=False,
                    cache_mask_list=self.cache_mask_list,
                    max_cache_frames=self.max_cache_frames,
                    sink_frames=self.sink_frames,
                    rolling_rope=self.rolling_rope,
                    **conditional_dict,
                )
                if unconditional_dict is not None and cfg_scale != 1.0:
                    _, uncond_denoised_pred = self.generator(
                        latent=noisy_input,
                        timestep=timestep,
                        kv_cache_list=self.kv_cache1,
                        cache_valid_len=cache_token_count,
                        cache_valid_frame_len=_cvfl,
                        global_frame_counter=_gfc,
                        update_cache=False,
                        cache_mask_list=self.cache_mask_list,
                        max_cache_frames=self.max_cache_frames,
                        sink_frames=self.sink_frames,
                        rolling_rope=self.rolling_rope,
                        **unconditional_dict,
                    )
                    denoised_pred = uncond_denoised_pred + cfg_scale * (
                        denoised_pred - uncond_denoised_pred
                    )

            if not exit_flag:
                next_timestep = self.denoising_step_list[index + 1]
                noisy_input = self.scheduler.add_noise(
                    denoised_pred.detach().flatten(0, 1),
                    torch.randn_like(denoised_pred.detach().flatten(0, 1)),
                    (torch.ones_like(timestep) * next_timestep).flatten(),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                break

        new_tokens = 0
        if update_target_cache:
            ctx_timestep = torch.ones_like(timestep) * self.context_noise
            if self.context_noise == 0:
                noisy_context_input = denoised_pred.detach()
            else:
                noisy_context_input = self.scheduler.add_noise(
                    denoised_pred.detach().flatten(0, 1),
                    torch.randn_like(denoised_pred.detach().flatten(0, 1)),
                    ctx_timestep.flatten(),
                ).unflatten(0, denoised_pred.shape[:2])

            with torch.no_grad():
                _cvfl, _gfc = self._get_rope_params(cache_frame_count)
                if "nomas" in self.model_name.lower():
                    _, new_tokens = self.generator.forward_cache_update(
                        latent=noisy_context_input,
                        timestep=ctx_timestep,
                        kv_cache_list=self.kv_cache1,
                        cache_valid_len=cache_token_count,
                        cache_valid_frame_len=_cvfl,
                        global_frame_counter=_gfc,
                        update_cache=True,
                        cache_txt=False,
                        cache_mask_list=self.cache_mask_list,
                        max_cache_frames=self.max_cache_frames,
                        sink_frames=self.sink_frames,
                        rolling_rope=self.rolling_rope,
                        **conditional_dict,
                    )

        new_cache_frame_count = cache_frame_count + current_num_frames
        new_cache_token_count = cache_token_count + int(new_tokens)
        return denoised_pred, new_cache_frame_count, new_cache_token_count

    def _resolve_exit_flag(self, device: torch.device) -> int:
        """Pick a single (broadcasted) denoising-step index for this iter."""
        num_denoising_steps = len(self.denoising_step_list)
        if not dist.is_initialized():
            idx = torch.randint(0, num_denoising_steps, (1,), device=device)
            if self.last_step_only:
                idx = torch.full_like(idx, num_denoising_steps - 1)
            return int(idx.item())

        group = self.shard_group
        if group is not None:
            local_src = dist.get_process_group_ranks(group)[0]
        else:
            local_src = 0
            group = None

        # Keep RNG state aligned across SP/FSDP ranks. The leader's sampled
        # exit step is still broadcast below, but every rank consumes the same
        # random op before later torch.randn calls inside denoising.
        idx = torch.randint(0, num_denoising_steps, (1,), device=device)
        if self.last_step_only:
            idx = torch.full_like(idx, num_denoising_steps - 1)

        dist.broadcast(idx, src=local_src, group=group)
        return int(idx.item())

    @staticmethod
    def _get_segment_conditional(
        conditional_dict: Optional[dict],
        conditional_dict_per_segment: Optional[List[dict]],
        segment_idx: int,
    ) -> dict:
        """Return the conditional_dict for ``segment_idx``.

        Multi-prompt callers pass ``conditional_dict_per_segment`` (one dict
        per segment); single-prompt callers pass a single ``conditional_dict``
        that is reused across all segments.
        """
        if conditional_dict_per_segment is not None:
            return conditional_dict_per_segment[segment_idx]
        assert conditional_dict is not None, (
            "Either conditional_dict or conditional_dict_per_segment must be set."
        )
        return conditional_dict

    def rollout_with_teacher_prefix(
        self,
        *,
        clean_video_latent: torch.Tensor,  # [B, F_lat, C, H, W]  GT
        boundaries: List[
            int
        ],  # in latent frames; segment j = [boundaries[j], boundaries[j+1])
        target_segment_idx: int,
        noise: Optional[
            torch.Tensor
        ],  # [B, segment_length_k, C, H, W]; None only with real_noised_one_step
        conditional_dict: Optional[dict] = None,
        conditional_dict_per_segment: Optional[List[dict]] = None,
        unconditional_dict: Optional[dict] = None,
        cfg_scale: float = 1.0,
        requires_grad: bool = True,
        # Distillation extras (inert at their defaults, so master's
        # teacher_forcing_dmd keeps its exact behaviour):
        #   context_prefix        — reuse ONE sampled noisy/denoised prefix across
        #                           generator / fake score / real score.
        #   real_noised_one_step  — score the GT target noised to a single
        #                           timestep instead of running the multi-step
        #                           denoise loop; ``noise`` is then unused.
        context_prefix: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        real_noised_one_step: bool = False,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """Roll out a single segment with GT-clean teacher prefix.

        Args:
            clean_video_latent: GT video latents in ``[B, F_lat, C, H, W]``
                layout (already ``* scaling_factor``).
            boundaries: Latent-frame segment boundaries (length
                ``num_segments + 1``, starts at 0, ends at ``F_lat``).
            target_segment_idx: Index ``k`` of the segment the student should
                denoise. ``0`` = first segment (no prefix).
            noise: Pure-noise initialization for segment ``k``,
                ``[B, boundaries[k+1] - boundaries[k], C, H, W]``.
            conditional_dict: Single text-state dict reused across all
                segments (single-prompt training).
            conditional_dict_per_segment: One conditional_dict per segment
                (multi-prompt training). Mutually exclusive with
                ``conditional_dict``; takes precedence if both are set.
            unconditional_dict: Optional unconditional dict for CFG inside the
                rollout (DMD usually keeps this off; the CFG inside DMD's
                fake/real-score evaluation lives elsewhere). Single-prompt
                only — unconditional embeddings are typically segment-
                independent.
            cfg_scale: CFG scale used for the rollout itself.

        Returns:
            ``(denoised_pred, dt_from, dt_to)`` matching the contract of
            ``SelfForcingTrainingPipeline.inference_with_trajectory``:
              - denoised_pred: ``[B, boundaries[k+1] - boundaries[k], C, H, W]``
                — the student's prediction for segment ``k`` (with grad on
                the final denoising step).
              - dt_from, dt_to: training-time timestep bounds for the DMD
                ``ts_schedule`` clamp; same convention as in
                ``SelfForcingTrainingPipeline.generate_chunk_with_cache``.
        """
        batch_size, F_lat, C, H, W = clean_video_latent.shape
        device = clean_video_latent.device
        distill_mode = bool(getattr(self, "distill_prefix_mode", False))

        num_segments = len(boundaries) - 1
        assert num_segments >= 1, f"boundaries must have >= 2 entries, got {boundaries}"
        assert boundaries[0] == 0 and boundaries[-1] == F_lat, (
            f"boundaries must start at 0 and end at F_lat={F_lat}, got {boundaries}"
        )
        assert 0 <= target_segment_idx < num_segments, (
            f"target_segment_idx={target_segment_idx} out of range [0, {num_segments - 1}]"
        )
        seg_start = boundaries[target_segment_idx]
        seg_end = boundaries[target_segment_idx + 1]
        target_length = seg_end - seg_start
        target_dtype = (
            torch.get_autocast_gpu_dtype()
            if torch.is_autocast_enabled()
            else clean_video_latent.dtype
        )
        if noise is not None:
            assert noise.shape[1] == target_length, (
                f"noise must contain {target_length} latent frames "
                f"(boundaries[{target_segment_idx}:{target_segment_idx + 2}]"
                f"={boundaries[target_segment_idx]},{boundaries[target_segment_idx + 1]}); "
                f"got {noise.shape[1]}."
            )
            cache_dtype = noise.dtype
        else:
            if not real_noised_one_step:
                raise ValueError("noise is required unless real_noised_one_step=True")
            cache_dtype = target_dtype
        if conditional_dict_per_segment is not None:
            if distill_mode:
                # Distill slices per-segment conditionals up to the target only,
                # so the list legitimately covers 0..target_segment_idx.
                assert len(conditional_dict_per_segment) > target_segment_idx, (
                    "conditional_dict_per_segment must cover prefix + target "
                    f"segments 0..{target_segment_idx}, got "
                    f"{len(conditional_dict_per_segment)} entries."
                )
            else:
                assert len(conditional_dict_per_segment) == num_segments, (
                    f"conditional_dict_per_segment must have {num_segments} entries, "
                    f"got {len(conditional_dict_per_segment)}."
                )
        if context_prefix is not None and len(context_prefix) < target_segment_idx:
            raise ValueError(
                f"context_prefix has {len(context_prefix)} entries, "
                f"but target_segment_idx={target_segment_idx}"
            )

        with record_function("tf_dmd/init_kv_cache"):
            if "nomas" in self.model_name.lower():
                cache_conditionals = (
                    conditional_dict_per_segment
                    if conditional_dict_per_segment is not None
                    else ([conditional_dict] if conditional_dict is not None else [])
                )
                max_cache_tokens = self._max_cache_tokens(
                    model=self.generator,
                    # Distill only ever reads the prefix span, so sizing for the
                    # whole video would waste cache proportional to the tail.
                    total_frames=(
                        int(boundaries[target_segment_idx]) if distill_mode else F_lat
                    ),
                    latent_height=H,
                    latent_width=W,
                    num_segments=(
                        int(target_segment_idx) if distill_mode else num_segments
                    ),
                    conditional_dicts=cache_conditionals,
                )
                self.kv_cache1, self.cache_mask_list, self._tokens_per_frame = (
                    self._allocate_nomas_kv_cache(
                        model=self.generator,
                        batch_size=batch_size,
                        dtype=cache_dtype,
                        device=device,
                        latent_height=H,
                        latent_width=W,
                        max_cache_tokens=max_cache_tokens,
                    )
                )
            else:
                self._initialize_kv_cache(
                    batch_size=batch_size,
                    dtype=cache_dtype,
                    device=device,
                    latent_height=H,
                    latent_width=W,
                )
            if modelname2config(self.model_name)["use_cross_attn"]:
                self._initialize_crossattn_cache(
                    batch_size=batch_size,
                    dtype=cache_dtype,
                    device=device,
                )

        cache_frame_count = 0
        cache_token_count = 0
        for j in range(target_segment_idx):
            with record_function(f"tf_dmd/teacher_prefill_seg{j}"):
                seg_start_j = boundaries[j]
                seg_end_j = boundaries[j + 1]
                clean_chunk_j = clean_video_latent[:, seg_start_j:seg_end_j]
                cd_j = self._get_segment_conditional(
                    conditional_dict,
                    conditional_dict_per_segment,
                    j,
                )
                cache_frame_count, cache_token_count = self._prefill_with_teacher_chunk(
                    clean_chunk=clean_chunk_j,
                    cache_frame_count=cache_frame_count,
                    cache_token_count=cache_token_count,
                    conditional_dict=cd_j,
                    context_entry=(
                        context_prefix[j] if context_prefix is not None else None
                    ),
                )

        # Sanity: cache_frame_count should equal the start of the target segment.
        assert cache_frame_count == seg_start, (
            f"prefix bookkeeping mismatch: cache_frame_count={cache_frame_count} "
            f"!= boundaries[target_segment_idx]={seg_start}"
        )

        exit_flag_idx = self._resolve_exit_flag(device)
        self._last_exit_flag_idx = exit_flag_idx

        target_cd = self._get_segment_conditional(
            conditional_dict,
            conditional_dict_per_segment,
            target_segment_idx,
        )

        with record_function("tf_dmd/denoise_target_segment"):
            if real_noised_one_step:
                # Score the GT target noised to ONE timestep instead of running
                # the multi-step denoise loop (used by the pure-adversarial DDMD
                # path, where the discriminator only needs a single noised pair).
                clean_target = clean_video_latent[:, seg_start:seg_end].to(
                    dtype=cache_dtype
                )
                current_timestep = self.denoising_step_list[exit_flag_idx]
                timestep = (
                    torch.ones(
                        (batch_size, target_length),
                        device=device,
                        dtype=torch.long,
                    )
                    * current_timestep
                )
                noisy_target = self.scheduler.add_noise(
                    clean_target.flatten(0, 1),
                    torch.randn_like(clean_target.flatten(0, 1)),
                    timestep.flatten(),
                ).unflatten(0, clean_target.shape[:2])
                target_cd = self._with_local_segment_extra(target_cd, target_length)
                if unconditional_dict is not None:
                    unconditional_dict = self._with_local_segment_extra(
                        unconditional_dict,
                        target_length,
                    )

                with torch.set_grad_enabled(requires_grad):
                    _cvfl, _gfc = self._get_rope_params(cache_frame_count)
                    _, denoised_pred = self.generator(
                        latent=noisy_target,
                        timestep=timestep,
                        kv_cache_list=self.kv_cache1,
                        cache_valid_len=cache_token_count,
                        cache_valid_frame_len=_cvfl,
                        global_frame_counter=_gfc,
                        update_cache=False,
                        cache_txt=True,
                        cache_mask_list=self.cache_mask_list,
                        max_cache_frames=self.max_cache_frames,
                        sink_frames=self.sink_frames,
                        rolling_rope=self.rolling_rope,
                        **self._action_mask_kwargs(target_cd),
                        **target_cd,
                    )
                    if unconditional_dict is not None and cfg_scale != 1.0:
                        _, uncond_denoised_pred = self.generator(
                            latent=noisy_target,
                            timestep=timestep,
                            kv_cache_list=self.kv_cache1,
                            cache_valid_len=cache_token_count,
                            cache_valid_frame_len=_cvfl,
                            global_frame_counter=_gfc,
                            update_cache=False,
                            cache_txt=True,
                            cache_mask_list=self.cache_mask_list,
                            max_cache_frames=self.max_cache_frames,
                            sink_frames=self.sink_frames,
                            rolling_rope=self.rolling_rope,
                            **self._action_mask_kwargs(unconditional_dict),
                            **unconditional_dict,
                        )
                        denoised_pred = uncond_denoised_pred + cfg_scale * (
                            denoised_pred - uncond_denoised_pred
                        )
            else:
                denoised_pred, _, _ = self._denoise_target_chunk(
                    noise=noise,
                    conditional_dict=target_cd,
                    unconditional_dict=unconditional_dict,
                    cache_frame_count=cache_frame_count,
                    cache_token_count=cache_token_count,
                    exit_flag_idx=exit_flag_idx,
                    cfg_scale=cfg_scale,
                    requires_grad=requires_grad,
                    update_target_cache=False,
                )

        if exit_flag_idx == len(self.denoising_step_list) - 1:
            dt_to = 0
            dt_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flag_idx].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
        else:
            dt_to = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flag_idx + 1].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
            dt_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flag_idx].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )

        return denoised_pred, dt_from, dt_to
