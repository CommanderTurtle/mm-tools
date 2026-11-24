"""Train Budget3D-Idol-Turbo: the same distillation on speridlabs/idol-turbo (Idol + phase-LoRA experts, 5 steps).

    torchrun --nproc_per_node=4 scripts/train_turbo.py --config configs/idol_turbo.yaml --data data/dome_objaverse --out runs/budget3d_idol_turbo --resume

Same trainer as scripts/train.py, with the turbo teacher (TurboRolloutStream: one guidance-free pass per step, the
image condition cached per phase) and a student that starts as Idol-turbo exactly (same LoRAs, zero added modules).
The LoRA experts are trained along with the backbone at the backbone learning rate.

Two measured speed settings are on by default (36 % more trajectories per second together on 4x RTX PRO 6000):
``--ckpt_policy auto`` checkpoints gradients only on the passes whose activations would not fit the GPU, and
``--lora_fused 1`` uses the one-pass form of the phase-LoRA.  The step-0 parity gate always compares the classic
LoRA form against Idol-turbo's own code, where the difference is exactly zero.
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
from budget3d.models.load import idol_weights_dir, load_student
from budget3d.models.turbo import lora_path, make_turbo, set_lora_fused
from budget3d.octree import point_scales
from budget3d.training.curriculum import LayoutSampler
from budget3d.training.data import image_stream
from budget3d.training.losses import distill_loss
from budget3d.training.rollout_turbo import TurboRolloutStream
from budget3d.training.trainer import add_common_args, init_process, lr_scale, param_groups, parse_args

CTX_TOKENS = 4102                  # image-condition tokens every sequence carries next to the latent tokens
ACT_BYTES = 2.4 * 2 ** 20          # activations kept per token and row without gradient checkpointing: 2.05 MiB
                                   # measured on the student (bf16 blocks), plus a margin


def main():
    ap = add_common_args(argparse.ArgumentParser(description=__doc__.split("\n")[0]), out="runs/budget3d_idol_turbo",
                         max_steps=20000, rollout_steps=5, rollout_cfg=0.0, grid_jitter=0.15, k0_stay_p=0.25,
                         cfg_dropout=0.0)
    ap.add_argument("--lora", default=None, help="idol-turbo LoRA file (downloaded from speridlabs/idol-turbo if omitted)")
    ap.add_argument("--noise_layout_early_k", type=int, default=2)
    ap.add_argument("--ckpt_policy", default="auto", choices=["auto", "always"],
                    help="gradient checkpointing of the student: auto = only on the passes whose activations would not "
                         "fit this GPU (same gradients, one forward less wherever memory allows); always = every pass")
    ap.add_argument("--lora_fused", type=int, default=1, choices=[0, 1],
                    help="1: the one-pass phase-LoRA (same function up to float rounding, faster); 0: the classic form")
    ap.add_argument("--rewarm_steps", type=int, default=0,
                    help="when resuming into a longer schedule, ramp the learning rate in linearly over this many steps")
    a = parse_args(ap)
    if a.print_args:
        print(vars(a))
        return
    assert a.cfg_dropout == 0 and a.rollout_cfg == 0, "the turbo teacher is guidance-free: no unconditional branch"
    world, rank, local, device, main_proc = init_process(a.seed)
    set_lora_fused(a.lora_fused)
    os.makedirs(a.out, exist_ok=True)

    weights = a.weights or idol_weights_dir()
    lora = a.lora or lora_path()
    student = load_student(weights, device=device)
    # training: both experts every step (rows mix phases; keeps every LoRA parameter in the DDP graph)
    _, student_phase, n_lora = make_turbo(student, lora, allow_uniform_fastpath=False)
    student.grad_checkpointing = not a.no_grad_checkpointing
    if a.freeze_backbone:
        for p_ in student.backbone_parameters:
            p_.requires_grad_(False)
    teacher = load_student(weights, device=device).eval().requires_grad_(False)
    make_turbo(teacher, lora)

    # the contract: with the adapters zero, the student at the dense layout *is* the released
    # model.  Checked in fp32 against Idol's own class, so it is a real identity, not a tolerance.
    if main_proc and not a.skip_parity:
        from budget3d.models.turbo_parity import check_turbo_parity
        rep = check_turbo_parity(student, weights, lora, device=device)       # in the classic LoRA form: exact
        print(f"[parity] student == idol-turbo (official LoRA code) at the dense layout: {rep}", flush=True)
        torch.cuda.empty_cache()
    if main_proc:
        print(f"[train] idol-turbo LoRAs: {n_lora} tensors from {lora}; "
              f"{'one-pass' if a.lora_fused else 'classic'} LoRA, gradient checkpointing: {a.ckpt_policy}", flush=True)
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
    stream = TurboRolloutStream(teacher, encoder, images, n_traj, steps=a.rollout_steps,
                                rescale_t=a.rescale_t, grid_jitter=a.grid_jitter, k0_stay_p=a.k0_stay_p,
                                device=device, seed=a.seed + 1000 * rank)
    layouts = LayoutSampler(a.anchor_p, a.uniform_coarse_p, a.budget_p, a.random_placement_p,
                            noise_layout_early_k=a.noise_layout_early_k, seed=a.seed + rank, device=device)

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

    auto = a.ckpt_policy == "auto" and not a.no_grad_checkpointing and device.type == "cuda"
    budget = 0.9 * torch.cuda.get_device_properties(device).total_memory if auto else 0
    forced, plain, passes, t_log, s_log = False, 0, 0, time.time(), step      # forced: back to checkpointing for good

    t0, step0, stats, seen = time.time(), step, {}, []
    while step < a.max_steps:
        skip = False
        scale = lr_scale(step, a.warmup_steps, a.max_steps, a.min_lr_ratio)
        if a.rewarm_steps and step0 > 0:
            # continuing past the previous schedule's end: the cosine for the longer run sits higher than the
            # LR the run ended on, so ramp up to it instead of jumping
            scale *= min(1.0, (step - step0 + 1) / a.rewarm_steps)
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
            last_pass = micro == a.grad_accum - 1
            sync = net.no_sync() if world > 1 and not last_pass else nullcontext()
            if auto:
                # without checkpointing a pass keeps every block's activations: rows x tokens x ACT_BYTES on top of
                # what is resident.  The last pass of a step, whose backward synchronises the ranks, always
                # checkpoints, so a pass that does run out of memory can simply be dropped (see below).
                need = torch.cuda.memory_allocated(device) + len(idx) * (layout.n_tokens + CTX_TOKENS) * ACT_BYTES + 4 * 2 ** 30
                student.grad_checkpointing = forced or last_pass or need > budget
                plain += not student.grad_checkpointing
                passes += 1
            # precision is handled inside the model (see budget3d.models.denoiser.bf16_matmuls): bf16 only in the
            # transformer blocks, fp32 residual stream, norms and output heads
            parts = None
            try:
                with sync:
                    out = net(tg["x"], t, stream.cond(idx), layout=layout, return_features=True)
                    ps = point_scales(layout, student.inv_perm)
                    loss, parts = distill_loss(out, tg, ps, a.coarse_boost, a.cam_weight, a.feat_weight,
                                               a.feat_cos_weight, branch="pos")
                    # a layout without e.g. 16-point tokens leaves those modules out of the graph; one scalar per
                    # added tensor keeps it attached, so DDP sees the same set of parameters on every rank
                    loss = loss / a.grad_accum + 0.0 * sum(p.view(-1)[0] for p in bp_params)
                loss.backward()
            except torch.OutOfMemoryError:
                if not auto or student.grad_checkpointing:
                    raise
                parts = None                                         # handled below, outside the handler
            if parts is None:
                # a pass without checkpointing did not fit after all: drop this optimizer step on every rank
                # (the gradient of this pass is incomplete) and checkpoint every pass from here on
                out = loss = ps = None
                torch.cuda.empty_cache()
                forced, skip = True, True
                print(f"[train] rank {rank}: out of memory without checkpointing at {layout.n_tokens} tokens x {len(idx)} "
                      f"(step {step}); this step is skipped and checkpointing stays on from now", flush=True)
            else:
                for k, v in parts.items():
                    stats[k] = stats.get(k, 0.0) + float(v) / a.grad_accum
            seen.append(layout.n_tokens)
            stream.advance(idx, tg)
        if auto and world > 1:                                       # one decision for all ranks
            flag = torch.tensor([float(skip)], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            skip = bool(flag.item() > 0)
        gnorm = torch.nn.utils.clip_grad_norm_([p for _, p in trainable], a.grad_clip)
        if torch.isfinite(gnorm) and not skip:
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
                  f"{(step - step0 + 1) * gbs / (time.time() - t0):.2f} traj/s  imgs {stream.restarts}"
                  + (f"  last {(step - s_log) * gbs / (time.time() - t_log):.2f}" if step > s_log else "")
                  + (f"  mem {torch.cuda.max_memory_allocated(device) / 2 ** 30:.0f}G  plain {plain}/{passes}" if auto else ""),
                  flush=True)
            stats, seen, t_log, s_log, plain, passes = {}, [], time.time(), step, 0, 0
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
