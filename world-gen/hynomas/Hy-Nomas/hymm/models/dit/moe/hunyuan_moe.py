import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN
from hymm.utils.logging_ import random_print


class MoeConfig:
    def __init__(
        self,
        num_experts=1,
        use_mixed_mlp_moe=False,
        num_shared_expert=1,
        moe_topk=1,
        capacity_factor=1.0,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        hidden_act=None,
        intermediate_size=None,
        hidden_size=None,
    ):
        self.num_experts = num_experts
        self.use_mixed_mlp_moe = use_mixed_mlp_moe
        self.num_shared_expert = num_shared_expert
        self.moe_topk = moe_topk
        self.capacity_factor = capacity_factor
        self.moe_drop_tokens = moe_drop_tokens
        self.moe_random_routing_dropped_token = moe_random_routing_dropped_token
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size

    def to_dict(self):
        return self.__dict__


class HunYuanMLP(nn.Module):
    def __init__(self, config: MoeConfig, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if is_shared_mlp:
            self.intermediate_size = config.intermediate_size * config.num_shared_expert
        else:
            self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, attn_out=None):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class HunYuanMLP2(nn.Module):
    """
    concat attention output with mlp output
    """

    def __init__(self, config: MoeConfig, is_shared_mlp=False):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        if is_shared_mlp:
            self.intermediate_size = config.intermediate_size * config.num_shared_expert
        else:
            self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(
            self.intermediate_size + self.hidden_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x, attn_out):
        mlp = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        mlp = torch.cat(
            [mlp, attn_out], dim=-1
        )  # concat attention output with mlp output
        down_proj = self.down_proj(mlp)
        return down_proj


def topkgating(logits: Tensor, topk: int):
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    expert_capacity = topk * gates.shape[0]
    num_experts = int(gates.shape[1])
    # Top-k router probability and corresponding expert indices for each token.
    # Shape: [tokens_per_group, num_selected_experts].
    expert_gate, expert_index = torch.topk(gates, topk)
    expert_mask = F.one_hot(expert_index, num_experts)
    # For a given token, determine if it was routed to a given expert.
    # Shape: [tokens_per_group, num_experts]
    expert_mask_aux = expert_mask.max(dim=-2)[0]
    tokens_per_group_and_expert = torch.mean(expert_mask_aux.float(), dim=-2)
    router_prob_per_group_and_expert = torch.mean(gates.float(), dim=-2)
    l_aux = num_experts**2 * torch.mean(
        tokens_per_group_and_expert * router_prob_per_group_and_expert
    )

    gates_s = torch.clamp(
        torch.matmul(expert_mask.float(), gates.unsqueeze(-1)).sum(dim=1),
        min=torch.finfo(gates.dtype).eps,
    )
    router_probs = gates / gates_s
    # Make num_selected_experts the leading axis to ensure that top-1 choices
    # have priority over top-2 choices, which have priority over top-3 choices,
    # etc.
    expert_index = torch.transpose(expert_index, 0, 1)
    # Shape: [num_selected_experts * tokens_per_group]
    expert_index = expert_index.reshape(-1)

    # Create mask out of indices.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    expert_mask = F.one_hot(expert_index, num_experts).to(torch.int32)
    exp_counts = torch.sum(expert_mask, dim=0).detach()

    # Experts have a fixed capacity that we cannot exceed. A token's priority
    # within the expert's buffer is given by the masked, cumulative capacity of
    # its target expert.
    # Shape: [tokens_per_group * num_selected_experts, num_experts].
    token_priority = torch.cumsum(expert_mask, dim=0) * expert_mask - 1
    # Shape: [num_selected_experts, tokens_per_group, num_experts].
    token_priority = token_priority.reshape((topk, -1, num_experts))
    # Shape: [tokens_per_group, num_selected_experts, num_experts].
    token_priority = torch.transpose(token_priority, 0, 1)
    # For each token, across all selected experts, select the only non-negative
    # (unmasked) priority. Now, for group G routing to expert E, token T has
    # non-negative priority (i.e. token_priority[G,T,E] >= 0) if and only if E
    # is its targeted expert.
    # Shape: [tokens_per_group, num_experts].
    token_priority = torch.max(token_priority, dim=1)[0]

    # Token T can only be routed to expert E if its priority is positive and
    # less than the expert capacity. One-hot matrix will ignore indices outside
    # the range [0, expert_capacity).
    # Shape: [tokens_per_group, num_experts, expert_capacity].
    valid_mask = torch.logical_and(
        token_priority >= 0, token_priority < expert_capacity
    )
    token_priority = torch.masked_fill(token_priority, ~valid_mask, 0)
    dispatch_mask = F.one_hot(token_priority, expert_capacity).to(torch.bool)
    valid_mask = valid_mask.unsqueeze(-1).expand(-1, -1, expert_capacity)
    dispatch_mask = torch.masked_fill(dispatch_mask, ~valid_mask, 0)

    # The combine array will be used for combining expert outputs, scaled by the
    # router probabilities. Shape: [num_groups, tokens_per_group, num_experts,
    # expert_capacity].
    combine_weights = torch.einsum("...te,...tec->...tec", router_probs, dispatch_mask)
    exp_counts_capacity = torch.sum(dispatch_mask)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0] * topk)

    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


def top1gating(logits: Tensor, random_routing_dropped_token: bool = False):
    """Implements Top1Gating on logits."""
    # everything is in fp32 in this function
    logits = logits.float()
    gates = F.softmax(logits, dim=1)
    capacity = gates.shape[0]

    # Create a mask for 1st's expert per token
    # noisy gating
    indices1_s = torch.argmax(gates, dim=1)
    num_experts = int(gates.shape[1])
    mask1 = F.one_hot(indices1_s, num_classes=num_experts)

    # gating decisions
    # exp_counts = torch.sum(mask1, dim=0).detach().to('cpu')
    exp_counts = torch.sum(mask1, dim=0).detach()

    # Compute l_aux
    me = torch.mean(gates, dim=0)
    ce = torch.mean(mask1.float(), dim=0)
    l_aux = torch.sum(me * ce) * num_experts
    mask1_rand = mask1

    top_idx = torch.topk(mask1_rand, k=capacity, dim=0)[1]

    new_mask1 = mask1 * torch.zeros_like(mask1).scatter_(0, top_idx, 1)
    mask1 = new_mask1
    mask1_bk = mask1
    if random_routing_dropped_token:
        not_full = capacity - new_mask1.sum(dim=0)
        sorted_notfull, indices_notfull = torch.sort(not_full, descending=True)
        sorted_notfull = sorted_notfull.to(torch.int64)
        not_full_experts_ids = torch.repeat_interleave(indices_notfull, sorted_notfull)
        shuffle_not_full_ids = torch.randperm(not_full_experts_ids.shape[0])
        not_full_experts_ids = not_full_experts_ids[shuffle_not_full_ids]
        indices1_s_after_drop = torch.argmax(new_mask1, dim=1)
        # get drop idx
        drop_mask = 1 - new_mask1.sum(dim=1)
        drop_mask = drop_mask.bool()
        drop_idx = drop_mask.nonzero().view(-1)
        drop_num = drop_mask.sum().to(torch.int64)
        indices1_s_after_drop.scatter_(0, drop_idx, not_full_experts_ids[:drop_num])
        nodrop_mask1 = F.one_hot(indices1_s_after_drop, num_classes=num_experts)
        mask1 = nodrop_mask1

    # Compute locations in capacity buffer
    locations1 = torch.cumsum(mask1, dim=0) - 1

    # Store the capacity location for each token
    locations1_s = torch.sum(locations1 * mask1, dim=1)

    # Normalize gate probabilities
    mask1_float = mask1.float()
    gates = gates * mask1_float

    locations1_sc = F.one_hot(
        locations1_s, num_classes=capacity
    ).float()  # one hot to float
    combine_weights = torch.einsum("se,sc->sec", gates, locations1_sc)

    dispatch_mask = combine_weights.bool()

    exp_counts_capacity = torch.sum(mask1_bk)
    exp_capacity_rate = exp_counts_capacity / (logits.shape[0])
    return [l_aux, exp_capacity_rate], combine_weights, dispatch_mask, exp_counts


class HunYuanTopKGate(nn.Module):
    def __init__(self, config: MoeConfig):
        super().__init__()
        self.config = config
        self.moe_topk = config.moe_topk
        self.drop_tokens = config.moe_drop_tokens
        self.min_capacity = 8
        self.random_routing_dropped_token = config.moe_random_routing_dropped_token
        self.wg = nn.Linear(
            config.hidden_size, config.num_experts, bias=False, dtype=torch.float32
        )

    def forward(self, hidden_states):
        bsz, seq_len, hidden_size = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_size)
        if self.wg.weight.dtype == torch.float32:
            hidden_states = hidden_states.float()
        logits = self.wg(hidden_states)
        if self.moe_topk == 1:
            gate_output = top1gating(
                logits, random_routing_dropped_token=self.random_routing_dropped_token
            )
        else:
            gate_output = topkgating(logits, self.moe_topk)

        return gate_output


class HunYuanMoE(nn.Module):
    def __init__(self, config: MoeConfig, mlp_class="HunYuanMLP"):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.mlp_class = mlp_class

        if mlp_class == "HunYuanMLP":
            if config.use_mixed_mlp_moe:
                self.shared_mlp = HunYuanMLP(config, is_shared_mlp=True)
            self.gate = HunYuanTopKGate(config)
            self.experts = nn.ModuleList(
                [
                    HunYuanMLP(config, is_shared_mlp=False)
                    for _ in range(config.num_experts)
                ]
            )
        elif mlp_class == "HunYuanMLP2":
            if config.use_mixed_mlp_moe:
                self.shared_mlp = HunYuanMLP2(config, is_shared_mlp=True)
            self.gate = HunYuanTopKGate(config)
            self.experts = nn.ModuleList(
                [
                    HunYuanMLP2(config, is_shared_mlp=False)
                    for _ in range(config.num_experts)
                ]
            )
        else:
            raise ValueError("mlp_class must be HunYuanMLP or HunYuanMLP2")

        self.l_moe = None

    def get_moe_loss(self):
        return self.l_moe

    def forward(self, hidden_states, attn_out=None):
        bsz, seq_len, hidden_size = hidden_states.shape

        if self.config.use_mixed_mlp_moe:
            hidden_states_mlp = self.shared_mlp(hidden_states, attn_out)

        # TODO(jef, gate 是否有必要传入attn_out当使用HunYuanMLP2时)
        l_moe, combine_weights, dispatch_mask, exp_counts = self.gate(hidden_states)

        reshaped_input = hidden_states.reshape(-1, hidden_size)
        dispatched_input = torch.einsum(
            "sec,sm->ecm", dispatch_mask.type_as(hidden_states), reshaped_input
        )
        chunks = dispatched_input.chunk(self.num_experts, dim=0)
        expert_outputs = []
        if self.mlp_class == "HunYuanMLP2":
            attn_out = attn_out.reshape(-1, hidden_size)
            attn_out = torch.einsum(
                "sec,sm->ecm", dispatch_mask.type_as(hidden_states), attn_out
            )
            attn_out_cunks = attn_out.chunk(self.num_experts, dim=0)
            for chunk, attn_out_chunk, expert in zip(
                chunks, attn_out_cunks, self.experts
            ):
                expert_outputs.append(expert(chunk, attn_out_chunk))
        else:
            for chunk, expert in zip(chunks, self.experts):
                expert_outputs.append(expert(chunk, attn_out))

        expert_output = torch.cat(expert_outputs, dim=0)
        combined_output = torch.einsum(
            "sec,ecm->sm", combine_weights.type_as(hidden_states), expert_output
        )
        combined_output = combined_output.reshape(bsz, seq_len, hidden_size)

        if self.config.use_mixed_mlp_moe:
            output = hidden_states_mlp + combined_output
        else:
            output = combined_output
        self.l_moe = l_moe[0]
        random_print(
            "moe aux loss: {}, moe capacity rate: {}".format(l_moe[0], l_moe[1]),
            random_prob=0.01,
        )
        return output


if __name__ == "__main__":
    config = MoeConfig(
        num_experts=16,
        use_mixed_mlp_moe=True,
        num_shared_expert=1,
        moe_topk=4,
        capacity_factor=1.0,
        moe_drop_tokens=False,
        moe_random_routing_dropped_token=False,
        hidden_act="gelu",
        intermediate_size=10,
        hidden_size=1024,
    )
    moe = HunYuanMoE(config, mlp_class="HunYuanMLP")
    input = torch.randn(2, 32, 1024)  # batch_size, seq_len, hidden_size
    output, l_moe = moe(input)
    print(output.shape)
    print(l_moe)

    moe = HunYuanMoE(config, mlp_class="HunYuanMLP2")
    attn_out = torch.randn(2, 32, 1024)  # batch_size, seq_len, hidden_size
    input = torch.randn(2, 32, 1024)  # batch_size, seq_len, hidden_size
    output, l_moe = moe(input, attn_out)
    print(output.shape)
    print(l_moe)
