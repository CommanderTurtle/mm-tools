# PTM Muon Optimizer
# This code file implemented the distributed version of MUON to parallize the N-S iterations of different parameters
# Author: neowywang
# Modified: Fixed use_muon serialization issue for DCP checkpointing

import os
import math
import fnmatch
import torch
import torch.distributed as dist
import torch.distributed
from .canzona_muon import (
    CanzonaMuonExecutor,
    ensure_sharded_momentum,
    get_dtensor_shard_spec,
)

try:
    from torch.distributed.tensor import distribute_tensor
    from torch.distributed.tensor import DTensor
    from torch.distributed.tensor.placement_types import (
        Partial,
        Placement,
        Replicate,
        Shard,
    )
    from torch.distributed.tensor._api import _ToTorchTensor
except ImportError:
    # handle old pytorch versions
    DTensor = None
    Partial = None
    Placement = None
    Replicate = None
    Shard = None


# This code is modified from the GitHub repository of KellerJordan:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert (
        G.ndim >= 2
    )  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    # X = G.bfloat16()
    X = G
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = (
            b * A + c * A @ A
        )  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Dist_Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
        dist_size: the scale of the distributed muon.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=None,
        fsdp_strategy="none",
        dist_size=8,
        muon_comm_mode="legacy",
        muon_canzona_fused_comm=True,
        muon_canzona_balance="global",
        muon_canzona_balance_cost="numel",
        muon_canzona_overlap="none",
    ):
        if adamw_wd is None:
            adamw_wd = wd

        defaults = dict(
            lr=lr,
            wd=wd,
            adamw_wd=adamw_wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Track which parameters use Muon using a set of parameter IDs (not stored in self.state)
        # This avoids serialization issues with distributed checkpointing
        self._muon_param_ids = set()

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            assert p.ndim >= 2, p.ndim
            self._muon_param_ids.add(id(p))
        # adamw_params don't need to be tracked - any param not in _muon_param_ids uses AdamW

        self.dist_size = dist_size
        self.muon_comm_mode = str(muon_comm_mode or "legacy").lower()
        self.muon_canzona_fused_comm = bool(muon_canzona_fused_comm)
        self.muon_canzona_balance = str(muon_canzona_balance or "global")
        self.muon_canzona_balance_cost = str(muon_canzona_balance_cost or "numel")
        self.muon_canzona_overlap = str(muon_canzona_overlap or "none")
        if self.muon_canzona_balance_cost != "numel":
            raise NotImplementedError(
                "This Canzona Muon integration currently supports balance_cost='numel' only."
            )
        self.rank = (
            torch.distributed.get_rank()
            if dist.is_available() and dist.is_initialized()
            else 0
        )
        self.world_size = (
            torch.distributed.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        if self.muon_comm_mode == "legacy":
            assert self.world_size % self.dist_size == 0, (
                "Error: self.world_size %% self.dist_size != 0"
            )
        self.local_rank = (
            self.rank % self.dist_size if self.muon_comm_mode == "legacy" else 0
        )
        self.pg_groups = []
        if self.muon_comm_mode == "legacy":
            for i in range(self.world_size // self.dist_size):
                offset = i * self.dist_size
                ranks = [offset + local_i for local_i in range(self.dist_size)]
                self.pg_groups.append(torch.distributed.new_group(ranks))
        elif self.muon_comm_mode != "canzona":
            raise ValueError(f"Unknown Muon communication mode: {self.muon_comm_mode}")

    def _clean_use_muon_from_state(self):
        """
        Remove any 'use_muon' entries from self.state.
        Also remove any invalid keys (non-parameter keys like integers or strings).
        This is necessary because old checkpoints may contain these entries.
        """
        # First, collect all valid parameter objects from param_groups
        valid_params = set()
        for group in self.param_groups:
            for p in group["params"]:
                valid_params.add(id(p))

        # Remove 'use_muon' from all state entries
        for param_state in self.state.values():
            if isinstance(param_state, dict) and "use_muon" in param_state:
                del param_state["use_muon"]

        # Also remove any keys that aren't valid parameter objects
        # (e.g., leftover integer or string keys from checkpoint loading)
        keys_to_remove = []
        for key in self.state.keys():
            if not hasattr(key, "data"):  # Not a parameter tensor
                keys_to_remove.append(key)
            elif id(key) not in valid_params:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.state[key]

    def state_dict(self):
        """
        Override state_dict to exclude 'use_muon' from state.
        This prevents serialization issues with distributed checkpointing.
        """
        # First clean up self.state to ensure no use_muon entries exist
        self._clean_use_muon_from_state()

        # Get state dict from parent
        state_dict = super().state_dict()

        # Also filter the returned dict (belt and suspenders approach)
        if "state" in state_dict:
            # Check for duplicate keys (int vs str that would flatten to same key)
            # This can happen when loading from DCP which uses string keys
            seen_keys = set()
            keys_to_remove = []

            for param_key in list(state_dict["state"].keys()):
                param_state = state_dict["state"][param_key]

                # Remove 'use_muon' if present
                if isinstance(param_state, dict) and "use_muon" in param_state:
                    del param_state["use_muon"]

                # Check for duplicate keys (e.g., 0 and "0" both exist)
                str_key = str(param_key)
                if str_key in seen_keys:
                    # This is a duplicate, mark for removal
                    keys_to_remove.append(param_key)
                else:
                    seen_keys.add(str_key)

            # Remove duplicate keys
            for key in keys_to_remove:
                del state_dict["state"][key]

        return state_dict

    def load_state_dict(self, state_dict):
        """
        Override load_state_dict to:
        1. Filter out 'use_muon' entries from old checkpoints (compat).
        2. Restore the intended bf16 dtype of `momentum_buffer`.

        Why (2) is needed:
        ``step()`` deliberately allocates ``momentum_buffer = torch.zeros_like(g)``
        where ``g = p.grad.full_tensor().bfloat16()`` — Muon's NS computation runs
        in bf16, so we want momentum in bf16 too (also saves ~50% optimizer mem).

        But ``torch.optim.Optimizer.load_state_dict`` applies
        ``_process_value_according_to_param_policy`` to every per-param tensor
        state, which silently casts ``value.to(dtype=param.dtype, ...)``. With
        ``master_weight_type=fp32`` + FSDP2 mixed precision the live params are
        fp32, so a saved bf16 ``momentum_buffer`` comes back as fp32. That:
          * breaks save → load → save round-trips (debug_save_load_test);
          * doubles Muon optimizer memory after the first resume;
          * silently changes the numerical trajectory of subsequent steps.

        We undo the upcast for Muon-owned params only; AdamW backup params keep
        the default cast-to-param-dtype behaviour for their moments.
        """
        # Filter out 'use_muon' from the loaded state dict before passing to parent
        if "state" in state_dict:
            for param_key in list(state_dict["state"].keys()):
                param_state = state_dict["state"][param_key]
                if isinstance(param_state, dict) and "use_muon" in param_state:
                    del param_state["use_muon"]

        super().load_state_dict(state_dict)

        # Also clean up self.state after loading (in case parent added entries)
        self._clean_use_muon_from_state()

        # Restore the intended bf16 momentum_buffer dtype (Muon-owned params only).
        for p, st in self.state.items():
            if not isinstance(st, dict):
                continue
            if id(p) not in self._muon_param_ids:
                continue
            mb = st.get("momentum_buffer")
            if DTensor is not None and isinstance(mb, DTensor):
                local = mb._local_tensor
                if local.dtype != torch.bfloat16:
                    st["momentum_buffer"] = DTensor.from_local(
                        local.to(torch.bfloat16),
                        device_mesh=mb.device_mesh,
                        placements=mb.placements,
                        run_check=False,
                    )
                continue
            if isinstance(mb, torch.Tensor) and mb.dtype != torch.bfloat16:
                st["momentum_buffer"] = mb.to(torch.bfloat16)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #           Muon           #
            ############################
            params = [
                p
                for p in group["params"]
                if id(p) in self._muon_param_ids and p.grad is not None
            ]
            if len(params) == 0:
                continue
            if self.muon_comm_mode == "canzona":
                self._step_muon_canzona(group, params)
            else:
                self._step_muon_legacy(group, params)

            ############################
            #       AdamW backup       #
            ############################
            params = [p for p in group["params"] if id(p) not in self._muon_param_ids]
            lr = group["lr"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)

        return loss

    def _step_muon_legacy(self, group, params):
        dist_size = self.dist_size
        rank = self.rank
        local_rank = self.local_rank
        padding = torch.empty_like(params[-1])
        padding.grad = torch.zeros_like(params[-1].grad)
        params_pad = params + [padding] * (dist_size - len(params) % dist_size)

        lr = group["lr"]
        wd = group["wd"]
        momentum = group["momentum"]

        # generate weight updates in distributed fashion
        for base_i in range(len(params))[::dist_size]:
            g_buf = []
            device_mesh_list = []
            for i, p in enumerate(params_pad[base_i : base_i + dist_size]):
                if DTensor is not None and isinstance(p.grad, DTensor):
                    device_mesh_list.append(p.grad.device_mesh)
                    g_buf.append(p.grad.full_tensor().bfloat16())
                else:
                    device_mesh_list.append(None)
                    g_buf.append(p.grad)
            if base_i + local_rank < len(params):
                g = g_buf[local_rank]
                p = params_pad[base_i : base_i + dist_size][local_rank]
                # sanity check
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None

                # calc update
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # update
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                g_buf[local_rank] = g.view(p.shape)

            # allgather update
            group_idx = rank // dist_size
            dist.all_gather(g_buf, g_buf[local_rank], group=self.pg_groups[group_idx])

            for i in range(dist_size):
                p = params_pad[base_i + i]
                g = g_buf[i]
                device_mesh = device_mesh_list[i]
                if device_mesh != None:
                    u = DTensor.from_local(g, device_mesh)
                    # u = distribute_tensor(g, device_mesh=p.grad.device_mesh, placements=p.grad.placements)
                else:
                    u = g
                # scale update
                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)

                # apply weight decay
                p.data.mul_(1 - lr * wd)

                # apply update
                # p.data.add_(u.view(p.shape), alpha=-adjusted_lr)
                p.data.add_(u, alpha=-adjusted_lr)

    def _step_muon_canzona(self, group, params):
        specs = []
        momentum_by_param_id = {}
        for p in params:
            if DTensor is None or not isinstance(p.grad, DTensor):
                raise ValueError(
                    "muon_comm_mode='canzona' requires DTensor gradients for all Muon parameters."
                )
            spec = get_dtensor_shard_spec(p, p.grad)
            state = self.state[p]
            mb = ensure_sharded_momentum(state, p.grad, spec)
            specs.append(spec)
            momentum_by_param_id[id(p)] = mb

        momentum = group["momentum"]
        nesterov = group["nesterov"]
        ns_steps = group["ns_steps"]
        lr = group["lr"]
        wd = group["wd"]

        def step_fn(p, spec, full_grad, full_mom):
            grad_shape = full_grad.shape
            g = (
                full_grad.view(full_grad.size(0), -1)
                if full_grad.ndim > 2
                else full_grad
            )
            buf = full_mom.view(g.shape)
            buf.mul_(momentum).add_(g)
            if nesterov:
                g = g.add(buf, alpha=momentum)
            else:
                g = buf
            update = zeropower_via_newtonschulz5(g, steps=ns_steps).view(
                spec.full_shape
            )
            return update, buf.view(grad_shape)

        def update_fn(p, spec, shard_update, shard_mom):
            state = self.state[p]
            state["momentum_buffer"] = DTensor.from_local(
                shard_mom.contiguous(),
                device_mesh=spec.device_mesh,
                placements=spec.placements,
                run_check=False,
            )
            adjusted_lr = self.adjust_lr_for_muon(lr, spec.full_shape)
            p.data.mul_(1 - lr * wd)
            u = DTensor.from_local(
                shard_update.contiguous(),
                device_mesh=p.device_mesh,
                placements=p.placements,
                run_check=False,
            )
            p.data.add_(u, alpha=-adjusted_lr)

        executor = CanzonaMuonExecutor(
            balance=self.muon_canzona_balance,
            fused_comm=self.muon_canzona_fused_comm,
            overlap=self.muon_canzona_overlap,
        )
        executor.execute(params, specs, momentum_by_param_id, step_fn, update_fn)


def _matches_any_pattern(name, patterns):
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def get_dist_muon_optimizer(
    model,
    lr=1e-3,
    weight_decay=0.1,
    momentum=0.95,
    adamw_betas=(0.95, 0.95),
    adamw_eps=1e-8,
    adamw_weight_decay=None,
    muon_exclude_patterns=None,
    dist_size=8,
    muon_comm_mode="legacy",
    muon_canzona_fused_comm=True,
    muon_canzona_balance="global",
    muon_canzona_balance_cost="numel",
    muon_canzona_overlap="none",
):
    muon_exclude_patterns = list(muon_exclude_patterns or [])
    muon_params = []
    adamw_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not _matches_any_pattern(name, muon_exclude_patterns):
            muon_params.append(p)
        else:
            adamw_params.append(p)

    return Dist_Muon(
        lr=lr,
        wd=weight_decay,
        muon_params=muon_params,
        momentum=momentum,
        adamw_params=adamw_params,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        adamw_wd=adamw_weight_decay,
        dist_size=dist_size,
        muon_comm_mode=muon_comm_mode,
        muon_canzona_fused_comm=muon_canzona_fused_comm,
        muon_canzona_balance=muon_canzona_balance,
        muon_canzona_balance_cost=muon_canzona_balance_cost,
        muon_canzona_overlap=muon_canzona_overlap,
    )
