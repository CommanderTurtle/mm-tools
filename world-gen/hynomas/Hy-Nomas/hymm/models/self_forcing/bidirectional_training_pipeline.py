from typing import List
import torch
import torch.distributed as dist

from hymm.diffusion.schedulers.scheduler_forcing import SchedulerInterface
from hymm.utils.loss import _convert_flow_pred_to_x0


class BidirectionalTrainingPipeline(torch.nn.Module):
    def __init__(
        self,
        model_name: str,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: torch.nn.Module,
    ):
        super().__init__()
        self.model_name = model_name
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]

    def _ensure_required_params(self, conditional_dict, latent):
        device = latent.device
        dtype = latent.dtype
        batch_size = latent.shape[0]

        if "extra_kwargs" not in conditional_dict:
            conditional_dict["extra_kwargs"] = {}

        if "is_token_replace" not in conditional_dict["extra_kwargs"]:
            conditional_dict["extra_kwargs"]["is_token_replace"] = False

        if "mask_type" not in conditional_dict:
            conditional_dict["mask_type"] = "t2v"

        if "vision_states" not in conditional_dict:
            vision_num_tokens = 729
            vision_dim = 1152
            conditional_dict["vision_states"] = torch.zeros(
                batch_size, vision_num_tokens, vision_dim, device=device, dtype=dtype
            )

    def inference_with_trajectory(
        self,
        noise: torch.Tensor,
        conditional_dict: dict = None,
        unconditional_dict: dict = None,
        exit_flags: list = None,
        **kwargs,
    ) -> torch.Tensor:
        noisy_image_or_video = noise
        num_denoising_steps = len(self.denoising_step_list)
        assert exit_flags is not None, "exit_flags must be provided"

        if conditional_dict is not None:
            self._ensure_required_params(conditional_dict, noisy_image_or_video)
        if unconditional_dict is not None:
            self._ensure_required_params(unconditional_dict, noisy_image_or_video)

        for index, current_timestep in enumerate(self.denoising_step_list):
            exit_flag = index == exit_flags[0]
            timestep = (
                torch.ones(noise.shape[:2], device=noise.device, dtype=torch.int64)
                * current_timestep
            )

            if "wan" in self.model_name.lower():
                timestep = timestep[:, 0]

            if not exit_flag:
                with torch.no_grad():
                    model_output, denoised_pred = self.generator(
                        latent=noisy_image_or_video,
                        timestep=timestep,
                        **conditional_dict,
                    )

                    if unconditional_dict is not None:
                        model_output_unconditional, denoised_pred_unconditional = (
                            self.generator(
                                latent=noisy_image_or_video,
                                timestep=timestep,
                                **unconditional_dict,
                            )
                        )
                        denoised_pred = denoised_pred_unconditional + 5.0 * (
                            denoised_pred - denoised_pred_unconditional
                        )
                        model_output = model_output_unconditional + 5.0 * (
                            model_output - model_output_unconditional
                        )

                    next_timestep = self.denoising_step_list[index + 1]
                    next_timestep = (
                        torch.ones(
                            noise.shape[:2], device=noise.device, dtype=torch.int64
                        )
                        * next_timestep
                    )

                    noisy_image_or_video = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep.flatten(0, 1),
                    ).unflatten(0, denoised_pred.shape[:2])

            else:
                model_output, denoised_pred = self.generator(
                    latent=noisy_image_or_video, timestep=timestep, **conditional_dict
                )

                if unconditional_dict is not None:
                    model_output_unconditional, denoised_pred_unconditional = (
                        self.generator(
                            latent=noisy_image_or_video,
                            timestep=timestep,
                            **unconditional_dict,
                        )
                    )
                    model_output = model_output_unconditional + 5.0 * (
                        model_output - model_output_unconditional
                    )
                    denoised_pred = denoised_pred_unconditional + 5.0 * (
                        denoised_pred - denoised_pred_unconditional
                    )
                break

        if exit_flags[0] == len(self.denoising_step_list) - 1:
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

        return denoised_pred, denoised_timestep_from, denoised_timestep_to
