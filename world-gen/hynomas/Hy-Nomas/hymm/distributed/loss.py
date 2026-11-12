"""Distributed loss reductions with autograd-correct numerator collectives."""

from typing import Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn


def global_token_count(
    local_valid_tokens: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """Sum the detached valid-token counts across independent data replicas."""
    if local_valid_tokens.numel() != 1:
        raise ValueError(
            "local_valid_tokens must be scalar, "
            f"but got shape {tuple(local_valid_tokens.shape)}"
        )
    global_valid_tokens = local_valid_tokens.detach().clone()
    if group is not None:
        dist.all_reduce(
            global_valid_tokens,
            op=dist.ReduceOp.SUM,
            group=group,
        )
    return global_valid_tokens


def global_token_mean_loss(
    local_loss_sum: torch.Tensor,
    local_valid_tokens: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
    global_valid_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return a differentiable global per-token mean over ``group``.

    ``group=None`` intentionally means no distributed reduction. Callers must
    pass an explicit process group when multiple independent data replicas
    contribute tokens. The numerator uses ``dist_nn.all_reduce`` so its
    backward contributes the group-size factor that FSDP's parameter-gradient
    averaging removes. The token count is detached and reduced normally.
    ``global_valid_tokens`` may be supplied when the count was reduced before
    backward, as required by serial segment training.
    """
    if local_loss_sum.numel() != 1:
        raise ValueError(
            "local_loss_sum must be scalar, "
            f"but got shape {tuple(local_loss_sum.shape)}"
        )
    if local_valid_tokens.numel() != 1:
        raise ValueError(
            "local_valid_tokens must be scalar, "
            f"but got shape {tuple(local_valid_tokens.shape)}"
        )
    if global_valid_tokens is None:
        global_valid_tokens = global_token_count(
            local_valid_tokens.to(
                device=local_loss_sum.device,
                dtype=local_loss_sum.dtype,
            ),
            group=group,
        )
    elif global_valid_tokens.numel() != 1:
        raise ValueError(
            "global_valid_tokens must be scalar, "
            f"but got shape {tuple(global_valid_tokens.shape)}"
        )
    global_valid_tokens = global_valid_tokens.detach().to(
        device=local_loss_sum.device, dtype=local_loss_sum.dtype
    )

    # An all-masked batch has a zero numerator. Keep its loss finite while
    # preserving the previous clamp(min=1) behavior.
    local_normalized_loss = local_loss_sum / global_valid_tokens.clamp(min=1.0)
    if group is None:
        return local_normalized_loss

    # Do not replace this with torch.distributed.all_reduce: that collective
    # does not install the backward kernel required by this loss formulation.
    return dist_nn.all_reduce(
        local_normalized_loss,
        op=dist.ReduceOp.SUM,
        group=group,
    )
