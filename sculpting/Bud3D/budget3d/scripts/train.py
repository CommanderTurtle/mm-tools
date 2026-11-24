"""Train Budget3D-Idol: distil Idol's released denoiser into the adaptive-token model by online teacher rollout.

    torchrun --nproc_per_node=4 scripts/train.py --config configs/idol.yaml --data data/dome_objaverse --out runs/budget3d_idol --resume

Every weight is trained: the released ones at ``--backbone_lr``, the added zero-initialised modules 6x faster.
The run is preemption-safe -- model, EMA and optimizer state are checkpointed every ``--save_every`` steps and
``--resume`` picks up the latest.  Before step 0 the student at the dense layout is checked to equal the released
Idol exactly.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))    # the repository root
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from budget3d.models.encoders import build_encoder
from budget3d.models.load import check_dense_parity, idol_weights_dir, load_student, load_teacher
from budget3d.octree import point_scales
from budget3d.training.curriculum import LayoutSampler
from budget3d.training.data import image_stream
from budget3d.training.losses import distill_loss
from budget3d.training.rollout import RolloutStream
from budget3d.training.trainer import add_common_args, init_process, lr_scale, param_groups, parse_args


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]), out="runs/budget3d_idol",
                         max_steps=20000, rollout_steps=25, rollout_cfg=3.0, grid_jitter=0.4, k0_stay_p=0.70,
                         cfg_dropout=0.12)
    a = parse_args(ap)
    if a.print_args:
        print(vars(a))
        return
    world, rank, local, device, main_proc = init_process(a.seed)
    os.makedirs(a.out, exist_ok=True)

    weights = a.weights or idol_weights_dir()
    student = load_student(weights, device=device)
    student.grad_checkpointing = not a.no_grad_checkpointing
    if a.freeze_backbone:
        for p_ in student.backbone_parameters:
            p_.requires_grad_(False)
    teacher = load_student(weights, device=device).eval().requires_grad_(False)

    # the contract: with the adapters zero, the student at the dense layout *is* the released
    # model.  Checked in fp32 against Idol's own class, so it is a real identity, not a tolerance.
    if main_proc and not a.skip_parity:
        ref = load_teacher(weights, device=device)
        rep = check_dense_parity(student, ref, batch=1, device=device, tol=0.0)
        print(f"[parity] student == released Idol at the dense layout: {rep}", flush=True)
        del ref
        torch.cuda.empty_cache()
    if world > 1:
        dist.barrier()
    if a.tf32:
        # the blocks already run bf16; this only touches the fp32 matmuls (input/out layers,
        # adaLN, the mixers). Enabled after the parity gate so the gate remains exact.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if a.compile_teacher:
        teacher.compile_blocks()
        if main_proc:
            print("[train] teacher blocks compiled (first step pays the compile)", flush=True)

    n_traj = a.batch_per_gpu * a.grad_accum
    gbs = a.batch_per_gpu * a.grad_accum * world
    encoder = build_encoder(device=device)
    images = image_stream(a.data, a.batch_per_gpu, views_per_object=a.views_per_object,
                          rank=rank, world_size=world, seed=a.seed, num_workers=a.num_workers)
    stream = RolloutStream(teacher, encoder, images, n_traj, steps=a.rollout_steps, cfg=a.rollout_cfg,
                           rescale_t=a.rescale_t, grid_jitter=a.grid_jitter, k0_stay_p=a.k0_stay_p,
                           device=device, seed=a.seed + 1000 * rank)
    layouts = LayoutSampler(a.anchor_p, a.uniform_coarse_p, a.budget_p, a.random_placement_p,
                            seed=a.seed + rank, device=device)

    opt = torch.optim.AdamW(param_groups(student, a.backbone_lr, a.adapter_mult, a.weight_decay),
                            betas=(0.9, 0.99), eps=1e-8, fused=device.type == "cuda")
    base_lrs = [g["lr"] for g in opt.param_groups]
    trainable = [(n, p) for n, p in student.named_parameters() if p.requires_grad]
    ema = {n: p.detach().clone().float() for n, p in trainable}
    bp_params = student.bp_parameters

    step = 0
    last = os.path.join(a.out, "checkpoint-last.pth")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["model"])
        ema = {n: v.to(device).float() for n, v in ck["ema"].items()}
        opt.load_state_dict(ck["optimizer"])
        step = int(ck["step"]) + 1
        if main_proc:
            print(f"[train] resumed from {last} at step {step}", flush=True)
        del ck
        torch.cuda.empty_cache()

    # fixed tensor lists for the fused EMA update (same order every step)
    ema_list = [ema[n] for n, _ in trainable]
    param_list = [p for _, p in trainable]
    net = DDP(student, device_ids=[local], find_unused_parameters=False) if world > 1 else student
    student.train()
    if main_proc:
        n_bp = sum(p.numel() for p in bp_params)
        n_all = sum(p.numel() for p in student.parameters())
        print(f"[train] global batch {gbs} ({a.batch_per_gpu} x {a.grad_accum} x {world}), "
              f"{n_traj} live trajectories/GPU, {a.max_steps} steps\n"
              f"[train] {n_all / 1e6:.2f}M parameters, of which {n_bp / 1e6:.2f}M added adapters; "
              f"backbone lr {a.backbone_lr:.1e}, adapters x{a.adapter_mult:g}", flush=True)

    def save(path, with_optimizer=True):
        state = {"model": student.state_dict(), "ema": {k: v.cpu() for k, v in ema.items()},
                 "step": step, "args": vars(a)}
        if with_optimizer:
            state["optimizer"] = opt.state_dict()
        torch.save(state, path + f".tmp{rank}")
        os.replace(path + f".tmp{rank}", path)

    t0, step0, stats, seen = time.time(), step, {}, []
    while step < a.max_steps:
        scale = lr_scale(step, a.warmup_steps, a.max_steps, a.min_lr_ratio)
        for g, base in zip(opt.param_groups, base_lrs):
            g["lr"] = base * scale
        opt.zero_grad(set_to_none=True)
        for micro in range(a.grad_accum):
            idx = torch.arange(micro * a.batch_per_gpu, (micro + 1) * a.batch_per_gpu, device=device)
            tg = stream.targets(idx)
            z_t, t, t01 = tg["x"]["latent"], tg["t01"] * 1000.0, tg["t01"]
            z_ref = layouts.reference_latent(z_t, stream.z0[idx], stream.k[idx])
            layout, mode = layouts(z_ref[:, student.perm_to_octree].reshape(len(idx), 16, 16, 16, 2, -1),
                                   stream.k[idx])
            # the dropout is shared: an example whose conditioning is zeroed regresses the
            # teacher's *unconditional* outputs, which is what keeps the CFG direction trained.
            # One code path for any number of dropped rows, so there is no rare branch to get wrong.
            drop = torch.rand(len(idx), device=device) < a.cfg_dropout
            if bool(drop.any()):
                for key in ("v", "cam", "h"):
                    m = drop.reshape(-1, *([1] * (tg[f"{key}_pos"].ndim - 1)))
                    tg[f"{key}_pos"] = torch.where(m, tg[f"{key}_neg"], tg[f"{key}_pos"])
            sync = net.no_sync() if world > 1 and micro < a.grad_accum - 1 else nullcontext()
            # precision is handled inside the model (see budget3d.models.denoiser.bf16_matmuls): bf16 only in the
            # transformer blocks, fp32 residual stream, norms and output heads
            with sync:
                out = net(tg["x"], t, stream.cond(idx, drop), layout=layout, return_features=True)
                ps = point_scales(layout, student.inv_perm)
                loss, parts = distill_loss(out, tg, ps, a.coarse_boost, a.cam_weight, a.feat_weight,
                                           a.feat_cos_weight, branch="pos")
                # a layout without e.g. 16-point tokens leaves those modules out of the graph; one scalar per
                # added tensor keeps it attached, so DDP sees the same set of parameters on every rank
                loss = loss / a.grad_accum + 0.0 * sum(p.view(-1)[0] for p in bp_params)
            loss.backward()
            for k, v in parts.items():
                stats[k] = stats.get(k, 0.0) + float(v) / a.grad_accum
            seen.append(layout.n_tokens)
            stream.advance(idx, tg)
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], a.grad_clip)
        if torch.isfinite(gnorm):
            opt.step()
        decay = min(a.ema_decay, (1 + step) / (10 + step))
        with torch.no_grad():
            torch._foreach_mul_(ema_list, decay)
            torch._foreach_add_(ema_list, param_list, alpha=1 - decay)

        if main_proc and (step % a.log_every == 0 or step == a.max_steps - 1):
            print(f"step {step:6d}  loss {stats.get('total', 0):.4f}  "
                  f"vel {stats.get('vel', 0):.4f}  cam {stats.get('cam', 0):.4f}  "
                  f"feat {stats.get('feat_nmse', 0):.4f}/{stats.get('feat_cos', 0):.4f}  "
                  f"tokens {sum(seen) / len(seen):.0f}  gnorm {float(gnorm):.2f}  lr {base_lrs[0] * scale:.2e}  "
                  f"{(step - step0 + 1) * gbs / (time.time() - t0):.2f} traj/s  imgs {stream.restarts}",
                  flush=True)
            stats, seen = {}, []
        else:
            stats, seen = {}, seen[-64:]
        if main_proc and (step + 1) % a.save_every == 0:
            save(last)
        if main_proc and (step + 1) % a.snapshot_every == 0:
            save(os.path.join(a.out, f"checkpoint-s{step + 1}.pth"), with_optimizer=False)
        step += 1

    if main_proc:
        save(os.path.join(a.out, "checkpoint-final.pth"), with_optimizer=False)
        print("[train] done:", os.path.join(a.out, "checkpoint-final.pth"), flush=True)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
