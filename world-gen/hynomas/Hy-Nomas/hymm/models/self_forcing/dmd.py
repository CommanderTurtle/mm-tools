# modified from
# https://github.com/GoatWu/Self-Forcing-Plus/blob/main/model/dmd.py

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Any
import torch.distributed as dist
from einops import rearrange
from torch.profiler import record_function

from hymm.utils.loss import get_denoising_loss
from hymm.utils.loss import _convert_x0_to_flow_pred, _convert_flow_pred_to_x0
from hymm.diffusion.schedulers.scheduler_forcing import SchedulerInterface
from hymm.models.self_forcing.bidirectional_training_pipeline import (
    BidirectionalTrainingPipeline,
)
from hymm.models.self_forcing.causal_training_pipeline import (
    SelfForcingTrainingPipeline,
)


class DMD(object):
    def __init__(
        self,
        generator: torch.nn.Module,
        fake_score: torch.nn.Module,
        real_score: torch.nn.Module,
        scheduler: SchedulerInterface,
        num_frame_per_block: int = 1,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = True,
        num_training_frames: int = 21,
        num_train_timestep: int = 1000,
        is_causal: bool = False,
        min_step: int = 0,
        max_step: int = 0,
        real_guidance_scale: float = 1.0,
        fake_guidance_scale: float = 0.0,
        timestep_shift: float = 1.0,
        ts_schedule: bool = True,
        ts_schedule_max: bool = False,
        min_score_timestep: int = 0,
        gradient_checkpointing: bool = False,
        denoising_loss_type: str = "flow",
        i2v_mode: bool = False,
        backward_simulation: bool = True,
        dtype: torch.dtype = None,
        vae: torch.nn.Module = None,
        last_step_only: bool = False,
        context_noise: int = 0,
        truncated_backprop_k: int = 0,
        num_max_rollout_blocks: int = 0,
        rollout_warmup_steps: int = 0,
        max_cache_frames: int = -1,
        sink_frames: int = 0,
        rolling_rope: bool = False,
        shard_group=None,
        decoupled_dmd: bool = False,
        rep_penalty_threshold: float = 0.0,
        rep_penalty_alpha: float = 1.0,
        rep_penalty_window: int = 15,
        cache_keep_frames: int = 0,
        gradual_evict: bool = False,
    ):
        super().__init__()
        self.generator = generator
        self.fake_score = fake_score
        self.real_score = real_score
        self.scheduler = scheduler
        self.vae = vae
        self.is_causal = is_causal
        self.decoupled_dmd = decoupled_dmd
        self.num_frame_per_block = num_frame_per_block
        self.same_step_across_blocks = same_step_across_blocks
        self.num_training_frames = num_training_frames
        self.last_step_only = last_step_only
        self.context_noise = context_noise
        self.truncated_backprop_k = truncated_backprop_k
        self.num_max_rollout_blocks = num_max_rollout_blocks
        self.rollout_warmup_steps = rollout_warmup_steps
        self.max_cache_frames = max_cache_frames
        self.sink_frames = sink_frames
        self.rolling_rope = rolling_rope
        self.shard_group = shard_group
        self.rep_penalty_threshold = rep_penalty_threshold
        self.rep_penalty_alpha = rep_penalty_alpha
        self.rep_penalty_window = rep_penalty_window
        self.cache_keep_frames = cache_keep_frames
        self.gradual_evict = gradual_evict
        self._global_step = 0
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = independent_first_frame
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        self.inference_pipeline = None

        self.num_train_timestep = num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_guidance_scale = real_guidance_scale
        self.fake_guidance_scale = fake_guidance_scale
        self.timestep_shift = timestep_shift
        self.ts_schedule = ts_schedule
        self.ts_schedule_max = ts_schedule_max
        self.min_score_timestep = min_score_timestep
        self.denoising_loss_type = denoising_loss_type
        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(
                generator.device
            )
        else:
            self.scheduler.alphas_cumprod = None
        self.denoising_loss_func = get_denoising_loss(denoising_loss_type)()

        self.i2v_mode = i2v_mode
        self.backward_simulation = backward_simulation
        self.dtype = (
            dtype if dtype is not None else next(self.generator.parameters()).dtype
        )

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        fake_cond_kwargs: Optional[dict] = None,
        fake_uncond_kwargs: Optional[dict] = None,
        real_cond_kwargs: Optional[dict] = None,
        real_uncond_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        fake_cond_call_kwargs = {**kwargs, **(fake_cond_kwargs or {})}
        fake_uncond_call_kwargs = {**kwargs, **(fake_uncond_kwargs or {})}
        real_cond_call_kwargs = {**kwargs, **(real_cond_kwargs or {})}
        real_uncond_call_kwargs = {**kwargs, **(real_uncond_kwargs or {})}

        with record_function("kl_grad/fake_score_cond"):
            _, pred_fake_image_cond = self.fake_score(
                latent=noisy_image_or_video,
                timestep=timestep,
                **conditional_dict,
                **fake_cond_call_kwargs,
            )

        if self.fake_guidance_scale != 0.0:
            with record_function("kl_grad/fake_score_uncond"):
                _, pred_fake_image_uncond = self.fake_score(
                    latent=noisy_image_or_video,
                    timestep=timestep,
                    **unconditional_dict,
                    **fake_uncond_call_kwargs,
                )
            if isinstance(pred_fake_image_uncond, tuple):
                pred_fake_image_uncond = pred_fake_image_uncond[0]
            pred_fake_image = (
                pred_fake_image_cond
                + (pred_fake_image_cond - pred_fake_image_uncond)
                * self.fake_guidance_scale
            )
        else:
            pred_fake_image = pred_fake_image_cond

        with record_function("kl_grad/real_score_cond"):
            _, pred_real_image_cond = self.real_score(
                latent=noisy_image_or_video,
                timestep=timestep,
                **conditional_dict,
                **real_cond_call_kwargs,
            )

        with record_function("kl_grad/real_score_uncond"):
            _, pred_real_image_uncond = self.real_score(
                latent=noisy_image_or_video,
                timestep=timestep,
                **unconditional_dict,
                **real_uncond_call_kwargs,
            )

        with record_function("kl_grad/cfg_and_normalize"):
            pred_real_image = (
                pred_real_image_cond
                + (pred_real_image_cond - pred_real_image_uncond)
                * self.real_guidance_scale
            )

            grad = pred_fake_image - pred_real_image

            if normalization:
                p_real = estimated_clean_image_or_video - pred_real_image
                normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
                grad = grad / normalizer
            grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
        }

    def _compute_kl_grad_decoupled(
        self,
        estimated_clean_image_or_video: torch.Tensor,
        timestep_ca: torch.Tensor,
        timestep_dm: torch.Tensor,
        noisy_ca: torch.Tensor,
        noisy_dm: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
        fake_cond_kwargs: Optional[dict] = None,
        real_cond_kwargs: Optional[dict] = None,
        real_uncond_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        fake_cond_call_kwargs = {**kwargs, **(fake_cond_kwargs or {})}
        real_cond_call_kwargs = {**kwargs, **(real_cond_kwargs or {})}
        real_uncond_call_kwargs = {**kwargs, **(real_uncond_kwargs or {})}

        with record_function("kl_grad_decoupled/real_score_cond_ca"):
            _, pred_real_cond_ca = self.real_score(
                latent=noisy_ca,
                timestep=timestep_ca,
                **conditional_dict,
                **real_cond_call_kwargs,
            )

        with record_function("kl_grad_decoupled/real_score_uncond_ca"):
            _, pred_real_uncond_ca = self.real_score(
                latent=noisy_ca,
                timestep=timestep_ca,
                **unconditional_dict,
                **real_uncond_call_kwargs,
            )

        with record_function("kl_grad_decoupled/real_score_cond_dm"):
            _, pred_real_cond_dm = self.real_score(
                latent=noisy_dm,
                timestep=timestep_dm,
                **conditional_dict,
                **real_cond_call_kwargs,
            )

        with record_function("kl_grad_decoupled/fake_score_cond_dm"):
            _, pred_fake_cond_dm = self.fake_score(
                latent=noisy_dm,
                timestep=timestep_dm,
                **conditional_dict,
                **fake_cond_call_kwargs,
            )

        with record_function("kl_grad_decoupled/combine_grad"):
            delta_ca = (self.real_guidance_scale - 1) * (
                pred_real_cond_ca - pred_real_uncond_ca
            )
            delta_dm = pred_real_cond_dm - pred_fake_cond_dm
            grad = -(delta_ca + delta_dm)

            if normalization:
                p_real = estimated_clean_image_or_video - pred_real_cond_dm
                normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
                grad = grad / normalizer
            grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep_ca": timestep_ca.detach(),
            "timestep_dm": timestep_dm.detach(),
        }

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        fake_cond_kwargs: Optional[dict] = None,
        fake_uncond_kwargs: Optional[dict] = None,
        real_cond_kwargs: Optional[dict] = None,
        real_uncond_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        original_latent = image_or_video
        if image_or_video.ndim == 5:
            batch_size, num_frame, channels, height, width = image_or_video.shape
        else:
            batch_size, channels, height, width = image_or_video.shape
            num_frame = 1

        with torch.no_grad():
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )

            if self.decoupled_dmd:
                with record_function("dmd_loss/sample_decoupled_timesteps"):
                    ca_min = (
                        denoised_timestep_to
                        if denoised_timestep_to is not None
                        else min_timestep
                    )
                    ca_min = max(ca_min, self.min_step)
                    timestep_ca = self._get_timestep(
                        ca_min,
                        max_timestep,
                        batch_size,
                        num_frame,
                        self.num_frame_per_block,
                        uniform_timestep=True,
                    )
                    if self.timestep_shift > 1:
                        timestep_ca = (
                            self.timestep_shift
                            * (timestep_ca / 1000)
                            / (1 + (self.timestep_shift - 1) * (timestep_ca / 1000))
                            * 1000
                        )
                    timestep_ca = timestep_ca.clamp(self.min_step, self.max_step)

                    timestep_dm = self._get_timestep(
                        min_timestep,
                        max_timestep,
                        batch_size,
                        num_frame,
                        self.num_frame_per_block,
                        uniform_timestep=True,
                    )
                    if self.timestep_shift > 1:
                        timestep_dm = (
                            self.timestep_shift
                            * (timestep_dm / 1000)
                            / (1 + (self.timestep_shift - 1) * (timestep_dm / 1000))
                            * 1000
                        )
                    timestep_dm = timestep_dm.clamp(self.min_step, self.max_step)

                    noise_ca = torch.randn_like(image_or_video)
                    noise_dm = torch.randn_like(image_or_video)

                with record_function("dmd_loss/add_noise_decoupled"):
                    if num_frame > 1:
                        iv_flat = rearrange(image_or_video, "b f c h w -> (b f) c h w")
                        noisy_ca = self.scheduler.add_noise(
                            iv_flat,
                            rearrange(noise_ca, "b f c h w -> (b f) c h w"),
                            timestep_ca.flatten(),
                        )
                        noisy_ca = rearrange(
                            noisy_ca,
                            "(b f) c h w -> b f c h w",
                            b=batch_size,
                            f=num_frame,
                        )
                        noisy_dm = self.scheduler.add_noise(
                            iv_flat,
                            rearrange(noise_dm, "b f c h w -> (b f) c h w"),
                            timestep_dm.flatten(),
                        )
                        noisy_dm = rearrange(
                            noisy_dm,
                            "(b f) c h w -> b f c h w",
                            b=batch_size,
                            f=num_frame,
                        )
                    else:
                        noisy_ca = self.scheduler.add_noise(
                            image_or_video, noise_ca, timestep_ca.flatten()
                        )
                        noisy_dm = self.scheduler.add_noise(
                            image_or_video, noise_dm, timestep_dm.flatten()
                        )

                with record_function("dmd_loss/compute_kl_grad_decoupled"):
                    grad, dmd_log_dict = self._compute_kl_grad_decoupled(
                        estimated_clean_image_or_video=original_latent,
                        timestep_ca=timestep_ca,
                        timestep_dm=timestep_dm,
                        noisy_ca=noisy_ca,
                        noisy_dm=noisy_dm,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        fake_cond_kwargs=fake_cond_kwargs,
                        real_cond_kwargs=real_cond_kwargs,
                        real_uncond_kwargs=real_uncond_kwargs,
                        **kwargs,
                    )
            else:
                with record_function("dmd_loss/sample_timestep_and_noise"):
                    timestep = self._get_timestep(
                        min_timestep,
                        max_timestep,
                        batch_size,
                        num_frame,
                        self.num_frame_per_block,
                        uniform_timestep=True,
                    )
                    if self.timestep_shift > 1:
                        timestep = (
                            self.timestep_shift
                            * (timestep / 1000)
                            / (1 + (self.timestep_shift - 1) * (timestep / 1000))
                            * 1000
                        )
                    timestep = timestep.clamp(self.min_step, self.max_step)
                    noise = torch.randn_like(image_or_video)

                with record_function("dmd_loss/add_noise"):
                    if num_frame > 1:
                        image_or_video_flat = rearrange(
                            image_or_video, "b f c h w -> (b f) c h w"
                        )
                        noise_flat = rearrange(noise, "b f c h w -> (b f) c h w")
                        timestep_flat = timestep.flatten()
                        noisy_latent_flat = self.scheduler.add_noise(
                            image_or_video_flat, noise_flat, timestep_flat
                        )
                        noisy_latent = rearrange(
                            noisy_latent_flat,
                            "(b f) c h w -> b f c h w",
                            b=batch_size,
                            f=num_frame,
                        )
                    else:
                        noisy_latent = self.scheduler.add_noise(
                            image_or_video, noise, timestep.flatten()
                        )

                with record_function("dmd_loss/compute_kl_grad"):
                    grad, dmd_log_dict = self._compute_kl_grad(
                        noisy_image_or_video=noisy_latent,
                        estimated_clean_image_or_video=original_latent,
                        timestep=timestep,
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        fake_cond_kwargs=fake_cond_kwargs,
                        fake_uncond_kwargs=fake_uncond_kwargs,
                        real_cond_kwargs=real_cond_kwargs,
                        real_uncond_kwargs=real_uncond_kwargs,
                        **kwargs,
                    )

        with record_function("dmd_loss/mse_loss"):
            if gradient_mask is not None:
                dmd_loss = 0.5 * F.mse_loss(
                    original_latent.double()[gradient_mask],
                    (original_latent.double() - grad.double()).detach()[gradient_mask],
                    reduction="mean",
                )
            else:
                dmd_loss = 0.5 * F.mse_loss(
                    original_latent.double(),
                    (original_latent.double() - grad.double()).detach(),
                    reduction="mean",
                )
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        exit_flags: list = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = (
            self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
                exit_flags=exit_flags,
                **kwargs,
            )
        )

        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            **kwargs,
        )

        del pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
        exit_flags: list = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        with record_function("critic_loss/run_generator_no_grad"):
            with torch.no_grad():
                generated_image, _, denoised_timestep_from, denoised_timestep_to = (
                    self._run_generator(
                        image_or_video_shape=image_or_video_shape,
                        conditional_dict=conditional_dict,
                        initial_latent=initial_latent,
                        exit_flags=exit_flags,
                        **kwargs,
                    )
                )

        with record_function("critic_loss/sample_timestep_and_add_noise"):
            actual_shape = list(generated_image.shape)
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                actual_shape[0],
                actual_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            if self.timestep_shift > 1:
                critic_timestep = (
                    self.timestep_shift
                    * (critic_timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                    * 1000
                )

            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            critic_noise = torch.randn_like(generated_image)
            noisy_generated_image = self.scheduler.add_noise(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1),
            ).unflatten(0, actual_shape[:2])

        with record_function("critic_loss/fake_score_forward"):
            _, pred_fake_image = self.fake_score(
                latent=noisy_generated_image,
                timestep=critic_timestep,
                **conditional_dict,
            )

        with record_function("critic_loss/denoising_loss"):
            if self.denoising_loss_type == "flow":
                flow_pred = _convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                )
                pred_fake_noise = None
            else:
                flow_pred = None
                pred_fake_noise = self.scheduler.convert_x0_to_noise(
                    x0=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                ).unflatten(0, actual_shape[:2])

            denoising_loss = self.denoising_loss_func(
                x=generated_image.flatten(0, 1),
                x_pred=pred_fake_image.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=pred_fake_noise,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred,
            )

        critic_log_dict = {"critic_timestep": critic_timestep.detach()}

        return denoising_loss, critic_log_dict

    def _compute_critic_from_generated(
        self,
        generated_image: torch.Tensor,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        denoised_timestep_from: int = None,
        denoised_timestep_to: int = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, dict]:
        with record_function("critic_from_gen/sample_ts_add_noise"):
            actual_shape = list(generated_image.shape)
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )

            critic_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                actual_shape[0],
                actual_shape[1],
                self.num_frame_per_block,
                uniform_timestep=True,
            )

            if self.timestep_shift > 1:
                critic_timestep = (
                    self.timestep_shift
                    * (critic_timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000))
                    * 1000
                )
            critic_timestep = critic_timestep.clamp(self.min_step, self.max_step)

            critic_noise = torch.randn_like(generated_image)
            noisy_generated_image = self.scheduler.add_noise(
                generated_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1),
            ).unflatten(0, actual_shape[:2])

        with record_function("critic_from_gen/fake_score_forward"):
            _, pred_fake_image = self.fake_score(
                latent=noisy_generated_image,
                timestep=critic_timestep,
                **conditional_dict,
                **kwargs,
            )

        with record_function("critic_from_gen/denoising_loss"):
            if self.denoising_loss_type == "flow":
                flow_pred = _convert_x0_to_flow_pred(
                    scheduler=self.scheduler,
                    x0_pred=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                )
                pred_fake_noise = None
            else:
                flow_pred = None
                pred_fake_noise = self.scheduler.convert_x0_to_noise(
                    x0=pred_fake_image.flatten(0, 1),
                    xt=noisy_generated_image.flatten(0, 1),
                    timestep=critic_timestep.flatten(0, 1),
                ).unflatten(0, actual_shape[:2])

            denoising_loss = self.denoising_loss_func(
                x=generated_image.flatten(0, 1),
                x_pred=pred_fake_image.flatten(0, 1),
                noise=critic_noise.flatten(0, 1),
                noise_pred=pred_fake_noise,
                alphas_cumprod=self.scheduler.alphas_cumprod,
                timestep=critic_timestep.flatten(0, 1),
                flow_pred=flow_pred,
            )

        return denoising_loss, {"critic_timestep": critic_timestep.detach()}

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        device = (
            self.generator.device
            if hasattr(self.generator, "device")
            else torch.device("cuda")
        )
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=device,
                dtype=torch.long,
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=device,
                dtype=torch.long,
            )
            if self.independent_first_frame:
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block
                )
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1
                )
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        initial_latent: torch.Tensor = None,
        unconditional_dict: dict = None,
        exit_flags: list = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.backward_simulation, "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.i2v_mode:
            noise_shape = [
                image_or_video_shape[0],
                image_or_video_shape[1] - 1,
                *image_or_video_shape[2:],
            ]
        else:
            noise_shape = list(image_or_video_shape)

        device = (
            self.generator.device
            if hasattr(self.generator, "device")
            else torch.device("cuda")
        )

        if self.truncated_backprop_k > 0:
            k_blocks = self.truncated_backprop_k
            max_rollout_blocks = self.num_max_rollout_blocks

            if self.rollout_warmup_steps > 0 and max_rollout_blocks > k_blocks:
                progress = min(self._global_step / self.rollout_warmup_steps, 1.0)
                max_rollout_blocks = k_blocks + int(
                    (max_rollout_blocks - k_blocks) * progress
                )
                max_rollout_blocks = max(max_rollout_blocks, k_blocks)

            with record_function("run_gen/sync_num_blocks"):
                num_generated_blocks = torch.randint(
                    k_blocks, max_rollout_blocks + 1, (1,), device=device
                )
                if dist.is_initialized():
                    dist.broadcast(num_generated_blocks, src=0)
                num_generated_blocks = num_generated_blocks.item()
            num_generated_frames = num_generated_blocks * self.num_frame_per_block
            if self.independent_first_frame and initial_latent is None:
                num_generated_frames += 1

            noise_shape[1] = num_generated_frames

            with record_function("run_gen/backward_simulation"):
                pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
                    self._consistency_backward_simulation(
                        noise=torch.randn(noise_shape, device=device, dtype=self.dtype),
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        exit_flags=exit_flags,
                        **kwargs,
                    )
                )

            k_frames = k_blocks * self.num_frame_per_block
            pred_image_or_video_last_x = pred_image_or_video[:, -k_frames:, ...]

            if num_generated_blocks > k_blocks:
                gradient_mask = torch.ones_like(
                    pred_image_or_video_last_x, dtype=torch.bool
                )
                gradient_mask[:, : self.num_frame_per_block] = False
            else:
                gradient_mask = None

        else:
            target_min = (
                self.num_training_frames - 1
                if self.independent_first_frame
                else self.num_training_frames
            )
            min_num_frames = min(target_min, self.num_training_frames)
            max_num_frames = (
                self.num_training_frames - 1
                if self.independent_first_frame
                else self.num_training_frames
            )

            max_num_blocks = max_num_frames // self.num_frame_per_block
            raw_min_blocks = min_num_frames // self.num_frame_per_block
            min_num_blocks = max(1, raw_min_blocks)

            if max_num_blocks < min_num_blocks:
                max_num_blocks = min_num_blocks

            with record_function("run_gen/sync_num_blocks"):
                num_generated_blocks = torch.randint(
                    min_num_blocks, max_num_blocks + 1, (1,), device=device
                )
                if dist.is_initialized():
                    dist.broadcast(num_generated_blocks, src=0)
                num_generated_blocks = num_generated_blocks.item()
            num_generated_frames = num_generated_blocks * self.num_frame_per_block
            if self.independent_first_frame and initial_latent is None:
                num_generated_frames += 1
                min_num_frames += 1

            noise_shape[1] = num_generated_frames

            with record_function("run_gen/backward_simulation"):
                pred_image_or_video, denoised_timestep_from, denoised_timestep_to = (
                    self._consistency_backward_simulation(
                        noise=torch.randn(noise_shape, device=device, dtype=self.dtype),
                        conditional_dict=conditional_dict,
                        unconditional_dict=unconditional_dict,
                        exit_flags=exit_flags,
                        **kwargs,
                    )
                )
            if pred_image_or_video.shape[1] > self.num_training_frames:
                latent_to_decode = pred_image_or_video[
                    :, : -(self.num_training_frames - 1), ...
                ]
                if not hasattr(self, "vae") or self.vae is None:
                    pred_image_or_video_last_x = pred_image_or_video[
                        :, -self.num_training_frames :, ...
                    ]
                else:
                    with torch.no_grad():
                        pixels = self.vae.decode_to_pixel(latent_to_decode)
                        frame = pixels[:, -1:, ...].to(self.dtype)
                        frame = rearrange(frame, "b t c h w -> b c t h w")
                        image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
                    pred_image_or_video_last_x = torch.cat(
                        [
                            image_latent,
                            pred_image_or_video[:, -self.num_training_frames :, ...],
                        ],
                        dim=1,
                    )
            else:
                pred_image_or_video_last_x = pred_image_or_video

            if num_generated_frames != min_num_frames:
                gradient_mask = torch.ones_like(
                    pred_image_or_video_last_x, dtype=torch.bool
                )
                if self.independent_first_frame:
                    gradient_mask[:, :1] = False
                else:
                    gradient_mask[:, : self.num_frame_per_block] = False
            else:
                gradient_mask = None

        pred_image_or_video_last_x = pred_image_or_video_last_x.to(self.dtype)
        return (
            pred_image_or_video_last_x,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
        )

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        conditional_dict: dict = None,
        unconditional_dict: dict = None,
        exit_flags: list = None,
        multi_prompt_schedule: dict = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            exit_flags=exit_flags,
            multi_prompt_schedule=multi_prompt_schedule,
            **kwargs,
        )

    def _initialize_inference_pipeline(self):
        if self.is_causal:
            max_frames = (
                self.num_max_rollout_blocks * self.num_frame_per_block
                if self.truncated_backprop_k > 0
                else self.num_training_frames
            )
            self.inference_pipeline = SelfForcingTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.independent_first_frame,
                same_step_across_blocks=self.same_step_across_blocks,
                last_step_only=self.last_step_only,
                num_max_frames=max_frames,
                context_noise=self.context_noise,
                num_training_frames=self.num_training_frames,
                truncated_backprop_k=self.truncated_backprop_k,
                max_cache_frames=self.max_cache_frames,
                sink_frames=self.sink_frames,
                rolling_rope=self.rolling_rope,
                shard_group=self.shard_group,
                rep_penalty_threshold=self.rep_penalty_threshold,
                rep_penalty_alpha=self.rep_penalty_alpha,
                rep_penalty_window=self.rep_penalty_window,
                cache_keep_frames=self.cache_keep_frames,
                gradual_evict=self.gradual_evict,
            )
        else:
            self.inference_pipeline = BidirectionalTrainingPipeline(
                model_name=self.generator_name,
                denoising_step_list=self.denoising_step_list,
                scheduler=self.scheduler,
                generator=self.generator,
            )
