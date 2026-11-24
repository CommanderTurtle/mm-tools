"""What the two trainers (scripts/train.py, scripts/train_turbo.py) share: argument parsing with YAML recipes,
optimizer groups, the learning-rate schedule and the process set-up."""
import math
import os

import torch
import torch.distributed as dist

# never weight-decayed: norms, embeddings, learned positions, slots and the shift tables
NO_DECAY = ("pos_pe", "shift_table", "bp_scale_emb", ".slots", ".pos", "freqs_", ".gamma", "norm")


def param_groups(model, backbone_lr, adapter_mult, weight_decay):
    """Four AdamW groups: backbone / added modules x decay / no-decay.  The added modules (``bp_*``) train
    ``adapter_mult`` times faster than the released weights."""
    groups = {k: {"params": [], "lr": backbone_lr * (adapter_mult if "adapter" in k else 1.0),
                  "weight_decay": weight_decay if "decay" in k and "no" not in k else 0.0}
              for k in ("backbone_decay", "backbone_nodecay", "adapter_decay", "adapter_nodecay")}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        which = "adapter" if name.startswith("bp_") else "backbone"
        decay = "decay" if (p.ndim >= 2 and not any(k in name for k in NO_DECAY)) else "nodecay"
        groups[f"{which}_{decay}"]["params"].append(p)
    return [g for g in groups.values() if g["params"]]


def lr_scale(step, warmup, total, min_ratio):
    """Linear warm-up over ``warmup`` steps, then a cosine from 1 down to ``min_ratio`` at ``total``."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))


def add_common_args(ap, *, out, max_steps, rollout_steps, rollout_cfg, grid_jitter, k0_stay_p, cfg_dropout):
    """The arguments both trainers take; the keyword values are the defaults that differ between them."""
    ap.add_argument("--data", required=True, help="Dome-Objaverse subset (scripts/fetch_dome.py)")
    ap.add_argument("--weights", default=None, help="Idol weights dir (downloaded if omitted)")
    ap.add_argument("--out", default=out)
    ap.add_argument("--max_steps", type=int, default=max_steps)
    ap.add_argument("--batch_per_gpu", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--backbone_lr", type=float, default=5e-5)
    ap.add_argument("--adapter_mult", type=float, default=6.0)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--min_lr_ratio", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--grad_clip", type=float, default=0.5)
    ap.add_argument("--ema_decay", type=float, default=0.9995)
    ap.add_argument("--rollout_steps", type=int, default=rollout_steps)
    ap.add_argument("--rollout_cfg", type=float, default=rollout_cfg)
    ap.add_argument("--rescale_t", type=float, default=3.0)
    ap.add_argument("--grid_jitter", type=float, default=grid_jitter)
    ap.add_argument("--k0_stay_p", type=float, default=k0_stay_p)
    ap.add_argument("--cfg_dropout", type=float, default=cfg_dropout)
    ap.add_argument("--coarse_boost", type=float, default=0.5)
    ap.add_argument("--cam_weight", type=float, default=0.2)
    ap.add_argument("--feat_weight", type=float, default=0.2)
    ap.add_argument("--feat_cos_weight", type=float, default=0.15)
    ap.add_argument("--anchor_p", type=float, default=0.12)
    ap.add_argument("--uniform_coarse_p", type=float, default=0.10)
    ap.add_argument("--budget_p", type=float, default=0.65)
    ap.add_argument("--random_placement_p", type=float, default=0.15)
    ap.add_argument("--no_grad_checkpointing", action="store_true",
                    help="not advisable: without gradient checkpointing the student runs out of memory above ~2.5k tokens")
    ap.add_argument("--compile_teacher", action="store_true",
                    help="torch.compile the frozen teacher's blocks (static dense shapes); the teacher is ~40%% of a step")
    ap.add_argument("--tf32", action="store_true", help="TF32 for the fp32 matmuls outside the blocks")
    ap.add_argument("--freeze_backbone", action="store_true",
                    help="train only the 7.56M added parameters, leaving the released weights fixed")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--views_per_object", type=int, default=6)
    ap.add_argument("--log_every", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--snapshot_every", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>/checkpoint-last.pth if it exists")
    ap.add_argument("--skip_parity", action="store_true", help="skip the step-0 teacher parity gate")
    ap.add_argument("--print_args", action="store_true", help="print the resolved settings and exit")
    ap.add_argument("--config", default=None, help="a YAML file of these arguments (configs/*.yaml); flags given on the "
                                                    "command line override it")
    return ap


def parse_args(ap, argv=None):
    """Parse with an optional ``--config`` YAML: its entries become the defaults, command-line flags override them,
    and an entry the parser does not know is an error."""
    pre, _ = ap.parse_known_args(argv)
    if pre.config:
        import yaml
        with open(pre.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = sorted(set(cfg) - {x.dest for x in ap._actions})
        if unknown:
            raise SystemExit(f"{pre.config}: unknown settings {unknown}")
        ap.set_defaults(**cfg)
    return ap.parse_args(argv)


def init_process(seed):
    """Distributed set-up from torchrun's environment -> (world size, rank, local rank, device, is main process)."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + rank)
    return world, rank, local, device, rank == 0
