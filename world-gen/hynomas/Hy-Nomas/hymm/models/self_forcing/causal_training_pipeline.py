from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.profiler import record_function

from hymm.diffusion.schedulers.scheduler_forcing import SchedulerInterface
from hymm.distributed.parallel_states import nccl_info


def modelname2config(model_name: str):
    r"""
    Convert model name to config
    config:
        num_transformer_blocks: int
        use_cross_attn: bool
    """
    config = {}
    model_name = model_name.lower()
    if "wan" in model_name:
        if "1.3b" in model_name:
            config["num_transformer_blocks"] = 30
            config["use_cross_attn"] = True

        elif "14b" in model_name:
            config["num_transformer_blocks"] = 30
            config["use_cross_attn"] = True

    elif "nomas" in model_name:
        if "g/2" in model_name:
            config["num_transformer_blocks"] = 54
            config["use_cross_attn"] = False
    else:
        raise ValueError(f"Model name {model_name} not supported")

    return config


class KVCacheWrapper:
    """
    A wrapper class to hide KV cache tensors from FSDP's argument inspection.
    FSDP rebuilds/copies dicts/lists containing tensors, breaking in-place updates.
    Passing a custom object bypasses this behavior (passed by reference).
    """

    def __init__(self, data_dict):
        self.data = data_dict

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __contains__(self, key):
        return key in self.data

    def get(self, key, default=None):
        return self.data.get(key, default)

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()


class SelfForcingTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: torch.nn.Module,
        num_frame_per_block=3,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        num_max_frames: int = 21,
        context_noise: int = 0,
        context_noise_sigma: float = 0.0,
        num_training_frames: int = 21,
        truncated_backprop_k: int = 0,
        max_cache_frames: int = -1,
        sink_frames: int = 0,
        rolling_rope: bool = False,
        shard_group=None,
        rep_penalty_threshold: float = 0.0,
        rep_penalty_alpha: float = 1.0,
        rep_penalty_window: int = 15,
        cache_keep_frames: int = 0,
        gradual_evict: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        self.shard_group = shard_group
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

        self.num_transformer_blocks = modelname2config(model_name)[
            "num_transformer_blocks"
        ]
        self.frame_seq_length = self._set_frame_seq_length()
        self.num_max_frames = max(num_max_frames, num_training_frames)
        self.num_training_frames = num_training_frames
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.context_noise_sigma = float(context_noise_sigma or 0.0)
        if not 0.0 <= self.context_noise_sigma <= 1.0:
            raise ValueError(
                "context_noise_sigma must be in [0, 1] (flow-matching SIGMA "
                "units, NOT a timestep in [0, 1000]), got "
                f"{self.context_noise_sigma}"
            )
        if self.context_noise_sigma > 0.0 and float(self.context_noise or 0.0) != 0.0:
            raise ValueError(
                "context_noise_sigma and context_noise are mutually exclusive "
                f"(got context_noise_sigma={self.context_noise_sigma}, "
                f"context_noise={self.context_noise}). Set exactly one."
            )
        self.truncated_backprop_k = truncated_backprop_k
        self.max_cache_frames = max_cache_frames
        self.sink_frames = sink_frames
        self.rolling_rope = rolling_rope
        self.cache_keep_frames = cache_keep_frames
        self.gradual_evict = gradual_evict
        self.i2v = False

        self.rep_penalty_threshold = rep_penalty_threshold
        self.rep_penalty_alpha = rep_penalty_alpha
        self.rep_penalty_window = rep_penalty_window

        self.kv_cache1 = None
        self.kv_cache2 = None
        if modelname2config(model_name)["use_cross_attn"]:
            self.crossattn_cache = None

        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def _get_rope_params(self, global_frame_idx: int):
        if self.rolling_rope and self.max_cache_frames > 0:
            clamped = min(global_frame_idx, self.max_cache_frames)
            return clamped, clamped
        return global_frame_idx, global_frame_idx

    def _pre_evict_caches(self):
        if self.kv_cache1 is None:
            return
        cache_capacity = self.kv_cache1[0]["k"].shape[1]
        tokens_per_frame = getattr(
            self, "_tokens_per_frame", cache_capacity // self.num_max_frames
        )
        max_tokens = self.max_cache_frames * tokens_per_frame
        sink_tokens = self.sink_frames * tokens_per_frame
        evict_tokens = self.num_frame_per_block * tokens_per_frame
        src_start = sink_tokens + evict_tokens
        remaining = max_tokens - src_start
        if remaining <= 0:
            return
        for bc in self.kv_cache1:
            bc["k"][:, sink_tokens : sink_tokens + remaining] = bc["k"][
                :, src_start:max_tokens
            ].clone()
            bc["v"][:, sink_tokens : sink_tokens + remaining] = bc["v"][
                :, src_start:max_tokens
            ].clone()

    def _gradual_evict_cache(self, cache_frame_count: int, evict_frames: int = None):
        if self.kv_cache1 is None:
            return cache_frame_count
        if evict_frames is None:
            evict_frames = self.num_frame_per_block

        tokens_per_frame = getattr(
            self,
            "_tokens_per_frame",
            self.kv_cache1[0]["k"].shape[1] // self.num_max_frames,
        )
        sink_tokens = self.sink_frames * tokens_per_frame
        evict_tokens = evict_frames * tokens_per_frame
        valid_tokens = cache_frame_count * tokens_per_frame

        src_start = sink_tokens + evict_tokens
        remaining = valid_tokens - src_start

        if remaining <= 0:
            return cache_frame_count

        for bc in self.kv_cache1:
            bc["k"][:, sink_tokens : sink_tokens + remaining] = bc["k"][
                :, src_start:valid_tokens
            ].clone()
            bc["v"][:, sink_tokens : sink_tokens + remaining] = bc["v"][
                :, src_start:valid_tokens
            ].clone()
            bc["k"][:, sink_tokens + remaining : valid_tokens] = 0
            bc["v"][:, sink_tokens + remaining : valid_tokens] = 0

        return cache_frame_count - evict_frames

    def _slide_kv_cache(self, cache_frame_count: int):
        if self.kv_cache1 is None:
            return cache_frame_count

        tokens_per_frame = getattr(
            self,
            "_tokens_per_frame",
            self.kv_cache1[0]["k"].shape[1] // self.num_max_frames,
        )
        sink_tokens = self.sink_frames * tokens_per_frame
        keep_tokens = self.cache_keep_frames * tokens_per_frame
        valid_tokens = cache_frame_count * tokens_per_frame
        src_start = valid_tokens - keep_tokens

        if src_start <= sink_tokens:
            return cache_frame_count

        dst_start = sink_tokens
        dst_end = dst_start + keep_tokens

        for bc in self.kv_cache1:
            bc["k"][:, dst_start:dst_end] = bc["k"][:, src_start:valid_tokens].clone()
            bc["v"][:, dst_start:dst_end] = bc["v"][:, src_start:valid_tokens].clone()
            bc["k"][:, dst_end:valid_tokens] = 0
            bc["v"][:, dst_end:valid_tokens] = 0

        return self.sink_frames + self.cache_keep_frames

    def _set_frame_seq_length(self) -> int:
        video_size = getattr(self.generator.args, "video_size", None)
        assert video_size is not None, "Video size must be provided"
        vae_spatial_ratio = getattr(self.generator.args, "vae_spatial_ratio", None)
        assert vae_spatial_ratio is not None, (
            "args.vae_spatial_ratio is not set. "
            "It should be populated by train.py after loading the VAE."
        )
        patch_size = getattr(self.generator, "patch_size", [1, 1, 1])
        h_latent = video_size[0] // vae_spatial_ratio
        w_latent = video_size[1] // vae_spatial_ratio
        frame_seq_length = (h_latent // patch_size[1]) * (w_latent // patch_size[2])
        return frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        # Every rank consumes the same RNG draw. The broadcast below still
        # canonicalizes the actual exit flags, but avoids advancing only the
        # source rank's RNG state before later torch.randn_like calls.
        indices = torch.randint(
            0, num_denoising_steps, size=(num_blocks,), device=device
        )
        if self.last_step_only:
            indices = torch.ones_like(indices) * (num_denoising_steps - 1)

        if not dist.is_initialized():
            return indices.tolist()

        group = self.shard_group
        if group is not None:
            group_ranks = dist.get_process_group_ranks(group)
            local_src = group_ranks[0]
        else:
            local_src = 0
            group = None

        dist.broadcast(indices, src=local_src, group=group)
        return indices.tolist()

    def generate_chunk_with_cache(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        *,
        current_start_frame: int = 0,
        requires_grad: bool = True,
        return_sim_step: bool = False,
        switch_frame_index: Optional[int] = None,
        switch_conditional_dict: Optional[dict] = None,
        switch_recache_frames: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        batch_size, chunk_frames = noise.shape[:2]

        num_blocks = (
            chunk_frames + self.num_frame_per_block - 1
        ) // self.num_frame_per_block
        all_num_frames = []
        remaining = chunk_frames
        for _ in range(num_blocks):
            take = min(remaining, self.num_frame_per_block)
            all_num_frames.append(take)
            remaining -= take

        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(all_num_frames), num_denoising_steps, device=noise.device
        )

        output = torch.zeros_like(noise)

        local_start = 0
        for block_idx, current_num in enumerate(all_num_frames):
            if switch_frame_index is not None and switch_conditional_dict is not None:
                if local_start >= switch_frame_index:
                    pass

            noisy_input = noise[:, local_start : local_start + current_num]

            for step_idx, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = step_idx == exit_flags[0]
                else:
                    exit_flag = step_idx == exit_flags[block_idx]

                timestep = (
                    torch.ones(
                        [batch_size, current_num], device=noise.device, dtype=torch.long
                    )
                    * current_timestep
                )

                should_grad = requires_grad and exit_flag

                with torch.set_grad_enabled(should_grad):
                    global_frame_idx = current_start_frame + local_start

                    if "nomas" in self.model_name.lower():
                        _cvfl, _gfc = self._get_rope_params(global_frame_idx)
                        _, denoised_pred = self.generator(
                            latent=noisy_input,
                            timestep=timestep,
                            kv_cache_list=self.kv_cache1,
                            cache_valid_frame_len=_cvfl,
                            global_frame_counter=_gfc,
                            update_cache=False,
                            max_cache_frames=self.max_cache_frames,
                            sink_frames=self.sink_frames,
                            rolling_rope=self.rolling_rope,
                            **conditional_dict,
                        )
                    else:
                        pass

                if not exit_flag:
                    next_timestep = self.denoising_step_list[step_idx + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        (torch.ones_like(timestep) * next_timestep).flatten(0, 1),
                    ).unflatten(0, denoised_pred.shape[:2])
                else:
                    break

            output[:, local_start : local_start + current_num] = denoised_pred

            ctx_timestep = torch.ones_like(timestep) * self.context_noise

            global_frame_idx = current_start_frame + local_start

            with torch.no_grad():
                if "nomas" in self.model_name.lower():
                    _cvfl, _gfc = self._get_rope_params(global_frame_idx)
                    self.generator(
                        latent=denoised_pred,
                        timestep=ctx_timestep,
                        kv_cache_list=self.kv_cache1,
                        cache_valid_frame_len=_cvfl,
                        global_frame_counter=_gfc,
                        update_cache=True,
                        max_cache_frames=self.max_cache_frames,
                        sink_frames=self.sink_frames,
                        rolling_rope=self.rolling_rope,
                        **conditional_dict,
                    )

            local_start += current_num

        if not self.same_step_across_blocks:
            dt_from, dt_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            dt_to = 0
            dt_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
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
                        - self.denoising_step_list[exit_flags[0] + 1].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
            dt_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )

        return output, dt_from, dt_to

    def _recache_after_switch(
        self,
        output_buffer: torch.Tensor,
        current_global_frame: int,
        new_conditional_dict: dict,
        overlap_len: int = 21,
    ):
        recache_start_frame = max(0, current_global_frame - overlap_len)

        if output_buffer.shape[1] >= current_global_frame:
            frames_to_recache = output_buffer[
                :, recache_start_frame:current_global_frame, :, :, :
            ]
        else:
            frames_to_recache = output_buffer[:, -overlap_len:, :, :, :]

        frames_to_recache = frames_to_recache.detach()
        num_recache_frames = frames_to_recache.shape[1]
        if num_recache_frames == 0:
            return

        print(
            f"[Recache] Switching Prompt... Rewinding cache to frame {recache_start_frame}, "
            f"re-computing {num_recache_frames} frames."
        )

        batch_size = frames_to_recache.shape[0]
        timestep = (
            torch.zeros(
                (batch_size, num_recache_frames),
                device=frames_to_recache.device,
                dtype=torch.long,
            )
            + self.context_noise
        )

        _ratio = 0.0
        frames_to_recache = (
            1 - _ratio
        ) * frames_to_recache + _ratio * torch.randn_like(frames_to_recache)
        timestep = (
            torch.ones(
                (batch_size, num_recache_frames),
                device=frames_to_recache.device,
                dtype=torch.long,
            )
            * _ratio
            * 1000
        )

        with torch.no_grad():
            _cvfl, _gfc = self._get_rope_params(recache_start_frame)
            self.generator(
                latent=frames_to_recache,
                timestep=timestep,
                **new_conditional_dict,
                kv_cache_list=self.kv_cache1,
                cache_valid_frame_len=_cvfl,
                global_frame_counter=_gfc,
                update_cache=True,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
            )

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict = None,
        unconditional_dict: dict = None,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        exit_flags: list = None,
        multi_prompt_schedule: dict = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (
            self.independent_first_frame and initial_latent is not None
        ):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0, (
                f"num_frames: {num_frames}, num_frame_per_block: {self.num_frame_per_block}"
            )
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype,
        )

        k_key, v_key = "k", "v"

        with record_function("trajectory/init_kv_cache"):
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                latent_height=height,
                latent_width=width,
            )

            if modelname2config(self.model_name)["use_cross_attn"]:
                self._initialize_crossattn_cache(
                    batch_size=batch_size, dtype=noise.dtype, device=noise.device
                )

        current_start_frame = 0
        cache_frame_count = 0
        use_jump_slide = self.cache_keep_frames > 0 and self.max_cache_frames > 0

        if initial_latent is not None:
            with record_function("trajectory/cache_initial_latent"):
                timestep = (
                    torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64)
                    * 0
                )
                output[:, :1] = initial_latent
                with torch.no_grad():
                    if "nomas" in self.model_name.lower():
                        _cvfl, _gfc = self._get_rope_params(cache_frame_count)
                        self.generator(
                            latent=initial_latent,
                            timestep=timestep * 0,
                            **conditional_dict,
                            kv_cache_list=self.kv_cache1,
                            cache_valid_frame_len=_cvfl,
                            global_frame_counter=_gfc,
                            update_cache=True,
                            max_cache_frames=self.max_cache_frames,
                            sink_frames=self.sink_frames,
                            rolling_rope=self.rolling_rope,
                        )
                    else:
                        self.generator(
                            noisy_image_or_video=initial_latent,
                            conditional_dict=conditional_dict,
                            timestep=timestep * 0,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                        )
                current_start_frame += 1
                cache_frame_count += 1

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        if exit_flags is None:
            exit_flags = self.generate_and_sync_list(
                len(all_num_frames), num_denoising_steps, device=noise.device
            )
        if self.truncated_backprop_k > 0:
            k_frames = self.truncated_backprop_k * self.num_frame_per_block
            start_gradient_frame_index = num_output_frames - k_frames
        else:
            start_gradient_frame_index = num_output_frames - self.num_training_frames

        current_cond_dict = conditional_dict
        current_uncond_dict = unconditional_dict

        for block_index, current_num_frames in enumerate(all_num_frames):
            with record_function(f"trajectory/block_{block_index}"):
                if (
                    self.max_cache_frames > 0
                    and cache_frame_count + current_num_frames > self.max_cache_frames
                ):
                    if self.gradual_evict:
                        evict_needed = (
                            cache_frame_count
                            + current_num_frames
                            - self.max_cache_frames
                        )
                        old_count = cache_frame_count
                        cache_frame_count = self._gradual_evict_cache(
                            cache_frame_count, evict_needed
                        )
                        if (
                            dist.is_initialized()
                            and dist.get_rank() == 0
                            or not dist.is_initialized()
                        ):
                            print(
                                f"[GradEvict] block {block_index}: evicted {evict_needed} frames, "
                                f"{old_count} -> {cache_frame_count}"
                            )
                    elif use_jump_slide:
                        cache_frame_count = self._slide_kv_cache(cache_frame_count)
                        if (
                            dist.is_initialized()
                            and dist.get_rank() == 0
                            or not dist.is_initialized()
                        ):
                            print(
                                f"[Slide] block {block_index}: cache full, kept last "
                                f"{self.cache_keep_frames} frames, cache_frame_count={cache_frame_count}"
                            )

                if (
                    multi_prompt_schedule is not None
                    and block_index in multi_prompt_schedule
                ):
                    print(
                        f"Switching Prompt at Block {block_index} (frame {current_start_frame})... "
                        f"[clean switch, no recache]"
                    )
                    current_cond_dict, current_uncond_dict = multi_prompt_schedule[
                        block_index
                    ]

                noisy_input = noise[
                    :,
                    current_start_frame - num_input_frames : current_start_frame
                    + current_num_frames
                    - num_input_frames,
                ]

                for index, current_timestep in enumerate(self.denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = index == exit_flags[0]
                    else:
                        exit_flag = index == exit_flags[block_index]
                    timestep = (
                        torch.ones(
                            [batch_size, current_num_frames],
                            device=noise.device,
                            dtype=torch.int64,
                        )
                        * current_timestep
                    )

                    if not exit_flag:
                        with record_function(
                            f"trajectory/blk{block_index}_denoise_step{index}_nograd"
                        ):
                            with torch.no_grad():
                                if "nomas" in self.model_name.lower():
                                    _cvfl, _gfc = self._get_rope_params(
                                        cache_frame_count
                                    )
                                    model_output, denoised_pred = self.generator(
                                        latent=noisy_input,
                                        timestep=timestep,
                                        kv_cache_list=self.kv_cache1,
                                        cache_valid_frame_len=_cvfl,
                                        global_frame_counter=_gfc,
                                        update_cache=False,
                                        max_cache_frames=self.max_cache_frames,
                                        sink_frames=self.sink_frames,
                                        rolling_rope=self.rolling_rope,
                                        **current_cond_dict,
                                    )
                                    if unconditional_dict is not None:
                                        uncond_model_output, uncond_denoised_pred = (
                                            self.generator(
                                                latent=noisy_input,
                                                timestep=timestep,
                                                kv_cache_list=self.kv_cache1,
                                                cache_valid_frame_len=_cvfl,
                                                global_frame_counter=_gfc,
                                                update_cache=False,
                                                max_cache_frames=self.max_cache_frames,
                                                sink_frames=self.sink_frames,
                                                rolling_rope=self.rolling_rope,
                                                **current_uncond_dict,
                                            )
                                        )
                                        denoised_pred = uncond_denoised_pred + 5.0 * (
                                            denoised_pred - uncond_denoised_pred
                                        )
                                        model_output = uncond_model_output + 5.0 * (
                                            model_output - uncond_model_output
                                        )
                                else:
                                    _, denoised_pred = self.generator(
                                        noisy_image_or_video=noisy_input,
                                        conditional_dict=current_cond_dict,
                                        timestep=timestep,
                                        kv_cache=self.kv_cache1,
                                        crossattn_cache=self.crossattn_cache,
                                        current_start=current_start_frame
                                        * self.frame_seq_length,
                                    )
                                next_timestep = self.denoising_step_list[index + 1]
                                noisy_input = self.scheduler.add_noise(
                                    denoised_pred.flatten(0, 1),
                                    torch.randn_like(denoised_pred.flatten(0, 1)),
                                    next_timestep
                                    * torch.ones(
                                        [batch_size * current_num_frames],
                                        device=noise.device,
                                        dtype=torch.long,
                                    ),
                                ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        is_context_block = (
                            current_start_frame < start_gradient_frame_index
                        )
                        grad_tag = "nograd" if is_context_block else "grad"
                        with record_function(
                            f"trajectory/blk{block_index}_exit_step{index}_{grad_tag}"
                        ):
                            if current_start_frame < start_gradient_frame_index:
                                with torch.no_grad():
                                    if "nomas" in self.model_name.lower():
                                        _cvfl, _gfc = self._get_rope_params(
                                            cache_frame_count
                                        )
                                        _, denoised_pred = self.generator(
                                            latent=noisy_input,
                                            timestep=timestep,
                                            kv_cache_list=self.kv_cache1,
                                            cache_valid_frame_len=_cvfl,
                                            global_frame_counter=_gfc,
                                            update_cache=False,
                                            max_cache_frames=self.max_cache_frames,
                                            sink_frames=self.sink_frames,
                                            rolling_rope=self.rolling_rope,
                                            **current_cond_dict,
                                        )

                                        if unconditional_dict is not None:
                                            _, uncond_denoised_pred = self.generator(
                                                latent=noisy_input,
                                                timestep=timestep,
                                                kv_cache_list=self.kv_cache1,
                                                cache_valid_frame_len=_cvfl,
                                                global_frame_counter=_gfc,
                                                update_cache=False,
                                                max_cache_frames=self.max_cache_frames,
                                                sink_frames=self.sink_frames,
                                                rolling_rope=self.rolling_rope,
                                                **current_uncond_dict,
                                            )
                                            denoised_pred = (
                                                uncond_denoised_pred
                                                + 5.0
                                                * (denoised_pred - uncond_denoised_pred)
                                            )
                                    else:
                                        _, denoised_pred = self.generator(
                                            noisy_image_or_video=noisy_input,
                                            conditional_dict=current_cond_dict,
                                            timestep=timestep,
                                            kv_cache=self.kv_cache1,
                                            crossattn_cache=self.crossattn_cache,
                                            current_start=current_start_frame
                                            * self.frame_seq_length,
                                        )
                            else:
                                if "nomas" in self.model_name.lower():
                                    _cvfl, _gfc = self._get_rope_params(
                                        cache_frame_count
                                    )
                                    _, denoised_pred = self.generator(
                                        latent=noisy_input,
                                        timestep=timestep,
                                        kv_cache_list=self.kv_cache1,
                                        cache_valid_frame_len=_cvfl,
                                        global_frame_counter=_gfc,
                                        update_cache=False,
                                        max_cache_frames=self.max_cache_frames,
                                        sink_frames=self.sink_frames,
                                        rolling_rope=self.rolling_rope,
                                        **current_cond_dict,
                                    )
                                else:
                                    _, denoised_pred = self.generator(
                                        noisy_image_or_video=noisy_input,
                                        conditional_dict=current_cond_dict,
                                        timestep=timestep,
                                        kv_cache=self.kv_cache1,
                                        crossattn_cache=self.crossattn_cache,
                                        current_start=current_start_frame
                                        * self.frame_seq_length,
                                    )
                        break

                if self.rep_penalty_threshold > 0 and block_index > 0:
                    window_start = max(0, current_start_frame - self.rep_penalty_window)
                    if window_start < current_start_frame:
                        with torch.no_grad():
                            recent = output[:, window_start:current_start_frame]
                            recent_mean = recent.mean(dim=1, keepdim=True)
                            sim = F.cosine_similarity(
                                denoised_pred.flatten(2),
                                recent_mean.expand_as(denoised_pred).flatten(2),
                                dim=2,
                            ).mean()
                            if sim > self.rep_penalty_threshold:
                                scale = (
                                    sim - self.rep_penalty_threshold
                                ) * self.rep_penalty_alpha
                                denoised_pred = (
                                    denoised_pred
                                    + scale * torch.randn_like(denoised_pred)
                                )

                output[
                    :, current_start_frame : current_start_frame + current_num_frames
                ] = denoised_pred

                denoised_pred_detached = denoised_pred.detach()

                with record_function(f"trajectory/blk{block_index}_update_cache"):
                    if self.context_noise_sigma > 0.0:
                        sigma = self.context_noise_sigma
                        context_timestep = torch.ones_like(timestep) * (sigma * 1000.0)
                        context_eps = torch.randn_like(denoised_pred_detached)
                        noisy_context_input = (
                            (1.0 - sigma) * denoised_pred_detached.float()
                            + sigma * context_eps.float()
                        ).to(denoised_pred_detached.dtype)
                    elif self.context_noise == 0:
                        context_timestep = (
                            torch.ones_like(timestep) * self.context_noise
                        )
                        noisy_context_input = denoised_pred_detached
                    else:
                        context_timestep = (
                            torch.ones_like(timestep) * self.context_noise
                        )
                        noisy_context_input = self.scheduler.add_noise(
                            denoised_pred_detached.flatten(0, 1),
                            torch.randn_like(denoised_pred_detached.flatten(0, 1)),
                            context_timestep.flatten(),
                        ).unflatten(0, denoised_pred_detached.shape[:2])
                    with torch.no_grad():
                        if "nomas" in self.model_name.lower():
                            _cvfl, _gfc = self._get_rope_params(cache_frame_count)
                            self.generator(
                                latent=noisy_context_input,
                                timestep=context_timestep,
                                kv_cache_list=self.kv_cache1,
                                cache_valid_frame_len=_cvfl,
                                global_frame_counter=_gfc,
                                update_cache=True,
                                max_cache_frames=self.max_cache_frames,
                                sink_frames=self.sink_frames,
                                rolling_rope=self.rolling_rope,
                                **current_cond_dict,
                            )
                        else:
                            self.generator(
                                noisy_image_or_video=noisy_context_input,
                                conditional_dict=current_cond_dict,
                                timestep=context_timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame
                                * self.frame_seq_length,
                            )

                current_start_frame += current_num_frames
                cache_frame_count += current_num_frames

        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
        else:
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0] + 1].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (
                        self.scheduler.timesteps.cuda()
                        - self.denoising_step_list[exit_flags[0]].cuda()
                    ).abs(),
                    dim=0,
                ).item()
            )

        if return_sim_step:
            return (
                output,
                denoised_timestep_from,
                denoised_timestep_to,
                exit_flags[0] + 1,
            )

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(
        self, batch_size, dtype, device, latent_height=None, latent_width=None
    ):
        if "nomas" in self.model_name.lower():
            if latent_height is not None and latent_width is not None:
                height = latent_height
                width = latent_width
            else:
                video_size = getattr(self.generator.args, "video_size", None)
                assert video_size is not None, (
                    "Video size must be provided for Nomas model"
                )
                vae_spatial_ratio = getattr(
                    self.generator.args, "vae_spatial_ratio", None
                )
                assert vae_spatial_ratio is not None, (
                    "args.vae_spatial_ratio is not set. "
                    "It should be populated by train.py after loading the VAE."
                )
                height = video_size[0] // vae_spatial_ratio
                width = video_size[1] // vae_spatial_ratio

            patch_size = self.generator.patch_size
            heads_num = self.generator.heads_num
            head_dim = self.generator.hidden_size // heads_num

            if nccl_info.sp_size > 1:
                assert heads_num % nccl_info.sp_size == 0, (
                    f"heads_num {heads_num} must be divisible by sp_size {nccl_info.sp_size}",
                    f"heads_num: {heads_num}, sp_size: {nccl_info.sp_size}",
                )
                heads_num = heads_num // nccl_info.sp_size

            h_tokens = height // patch_size[1]
            w_tokens = width // patch_size[2]
            tokens_per_frame = h_tokens * w_tokens
            self._tokens_per_frame = tokens_per_frame

            effective_max_frames = self.num_max_frames
            if self.max_cache_frames > 0:
                effective_max_frames = self.max_cache_frames
            max_tokens = effective_max_frames * tokens_per_frame

            kv_cache1 = []
            num_blocks = len(self.generator.double_blocks)

            for _ in range(num_blocks):
                cache_dict = {
                    "k": torch.zeros(
                        [batch_size, max_tokens, heads_num, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, max_tokens, heads_num, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                }
                kv_cache1.append(KVCacheWrapper(cache_dict))
            self.kv_cache1 = kv_cache1

        else:
            kv_cache1 = []

            if hasattr(self.generator, "num_heads"):
                num_heads = self.generator.num_heads
            elif hasattr(self.generator, "module") and hasattr(
                self.generator.module, "num_heads"
            ):
                num_heads = self.generator.module.num_heads
            elif hasattr(self.generator, "config") and hasattr(
                self.generator.config, "num_heads"
            ):
                num_heads = self.generator.config.num_heads
            else:
                if "1.3b" in self.model_name.lower():
                    num_heads = 12
                elif "14b" in self.model_name.lower():
                    num_heads = 40
                else:
                    raise ValueError(
                        f"Could not infer num_heads for model {self.model_name}"
                    )

            if dist.is_initialized() and nccl_info.sp_size > 1:
                assert num_heads % nccl_info.sp_size == 0
                num_heads = num_heads // nccl_info.sp_size

            head_dim = 128

            for _ in range(self.num_transformer_blocks):
                cache_dict = {
                    "k": torch.zeros(
                        [batch_size, self.kv_cache_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, self.kv_cache_size, num_heads, head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "global_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "local_end_index": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                }
                kv_cache1.append(KVCacheWrapper(cache_dict))

            self.kv_cache1 = kv_cache1

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        crossattn_cache = []

        for _ in range(self.num_transformer_blocks):
            cache_dict = {
                "k": torch.zeros(
                    [batch_size, 512, self.num_transformer_blocks, 128],
                    dtype=dtype,
                    device=device,
                ),
                "v": torch.zeros(
                    [batch_size, 512, self.num_transformer_blocks, 128],
                    dtype=dtype,
                    device=device,
                ),
                "is_init": False,
            }
            crossattn_cache.append(KVCacheWrapper(cache_dict))
        self.crossattn_cache = crossattn_cache
