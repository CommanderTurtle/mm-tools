from abc import ABC, abstractmethod
import torch


class DenoisingLoss(ABC):
    @abstractmethod
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        pass


class X0PredLoss(DenoisingLoss):
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.mean((x.float() - x_pred.float()) ** 2)


class VPredLoss(DenoisingLoss):
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        weights = 1 / (1 - alphas_cumprod[timestep].reshape(*timestep.shape, 1, 1, 1))
        return torch.mean(weights * (x.float() - x_pred.float()) ** 2)


class NoisePredLoss(DenoisingLoss):
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.mean((noise.float() - noise_pred.float()) ** 2)


class FlowPredLoss(DenoisingLoss):
    def __call__(
        self,
        x: torch.Tensor,
        x_pred: torch.Tensor,
        noise: torch.Tensor,
        noise_pred: torch.Tensor,
        alphas_cumprod: torch.Tensor,
        timestep: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return torch.mean(
            (kwargs["flow_pred"].float() - (noise.float() - x.float())) ** 2
        )


NAME_TO_CLASS = {
    "x0": X0PredLoss,
    "v": VPredLoss,
    "noise": NoisePredLoss,
    "flow": FlowPredLoss,
}


def get_denoising_loss(loss_type: str) -> DenoisingLoss:
    return NAME_TO_CLASS[loss_type]


def _convert_flow_pred_to_x0(
    scheduler, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
) -> torch.Tensor:
    """
    Convert flow matching's prediction to x0 prediction.
    pred = noise - x0
    x_t = (1-sigma_t) * x0 + sigma_t * noise
    we have x0 = x_t - sigma_t * pred
    """
    original_dtype = flow_pred.dtype
    flow_pred, xt, sigmas, timesteps = map(
        lambda x: x.double().to(flow_pred.device),
        [flow_pred, xt, scheduler.sigmas, scheduler.timesteps],
    )
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    x0_pred = xt - sigma_t * flow_pred
    return x0_pred.to(original_dtype)


def _convert_x0_to_flow_pred(
    scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
) -> torch.Tensor:
    """
    Convert x0 prediction to flow matching's prediction.
    pred = (x_t - x_0) / sigma_t
    """
    original_dtype = x0_pred.dtype
    x0_pred, xt, sigmas, timesteps = map(
        lambda x: x.double().to(x0_pred.device),
        [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
    )
    timestep_id = torch.argmin(
        (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1
    )
    sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
    flow_pred = (xt - x0_pred) / sigma_t
    return flow_pred.to(original_dtype)
