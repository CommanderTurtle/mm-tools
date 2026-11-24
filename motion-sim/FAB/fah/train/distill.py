"""Stage-1 distillation: a small student encoder mimics the SAM 3D Body teacher backbone.

Teacher: DINOv3-H+ on the hand crop at --teacher_size (384 default), features area-pooled to 16x16
(frozen decoder on FreiHAND: 512 native 5.16 mm, 512->16 pooled 5.07, 384->16 pooled 5.07). Student: fah.student
encoder on the same region at 224/256. Label-free (any hand crops); the eval plugs the student into
the frozen teacher decoder and reports FreiHAND PA-MPJPE (MANO-convention joints, GT boxes).

    torchrun --nproc_per_node "${TRAIN_PROCESSES:?set TRAIN_PROCESSES}" -m train.distill --batch_size "${TRAIN_BATCH_SIZE:?set TRAIN_BATCH_SIZE}" --crops $DATA_ROOT/crops --sources coco_wb,cmu \
        --freihand $DATA_ROOT/freihand --out $CKPT_ROOT/distill/<run> --run_name <run>
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fah.student import STUDENTS, StudentEncoder, teacher_target

CROP = 640  # train.build_crops.SIZE
CONTEXT = 1.25  # train.build_crops.CONTEXT -> teacher region = CROP / CONTEXT px, centred


class CropPool(Dataset):
    """Right-hand crops from train.build_crops; returns a size x size RGB uint8 view of the teacher region
    under a random similarity transform + photometric jitter (no flips: right-hand convention)."""

    def __init__(self, root, sources, size=512, aug=True):
        self.items, self.src = [], []
        for k, s in enumerate(sources):
            for line in open(os.path.join(root, s, "index.jsonl")):
                self.items.append(os.path.join(root, s, json.loads(line)["file"]))
                self.src.append(k)
        self.src = np.asarray(self.src)
        self.size, self.aug = size, aug

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img = cv2.imread(self.items[i])
        if img is None:
            img = np.zeros((CROP, CROP, 3), np.uint8)
        img = img[:, :, ::-1]
        side = CROP / CONTEXT
        c = np.array([CROP / 2, CROP / 2])
        rot = 0.0
        if self.aug:
            side *= np.exp(np.random.uniform(np.log(0.85), np.log(1.15)))
            c = c + np.random.uniform(-0.08, 0.08, 2) * side
            rot = np.random.uniform(-25, 25) if np.random.rand() < 0.7 else 0.0
        M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), rot, self.size / side)
        M[:, 2] += self.size / 2 - c
        out = cv2.warpAffine(np.ascontiguousarray(img), M, (self.size, self.size), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT)
        if self.aug:
            out = photometric(out)
        return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1)


def seed_worker(worker_id):
    # DataLoader workers inherit the parent's numpy RNG; reseed so augmentations differ per worker / epoch
    np.random.seed(torch.initial_seed() % 2**32)


def photometric(img):
    x = img.astype(np.float32)
    if np.random.rand() < 0.8:
        x = x * np.random.uniform(0.6, 1.4) + np.random.uniform(-30, 30)  # contrast / brightness
        g = x.mean(2, keepdims=True)
        x = g + (x - g) * np.random.uniform(0.5, 1.5)  # saturation
        x = x * np.random.uniform(0.9, 1.1, 3)  # white balance
    if np.random.rand() < 0.05:
        x = np.repeat(x.mean(2, keepdims=True), 3, 2)
    x = np.clip(x, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.15:
        k = int(np.random.choice([3, 5]))
        x = cv2.GaussianBlur(x, (k, k), 0)
    if np.random.rand() < 0.15:  # motion / compression artefacts typical of ego video
        q = int(np.random.uniform(30, 80))
        x = cv2.imdecode(cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
    return x


# ---------------------------------------------------------------------------- eval
@torch.no_grad()
def eval_freihand(model, cfg, student, root, num, mano_dir=None):
    from bench.common import procrustes
    from bench.datasets.freihand import FreiHANDEval
    from fah import HandEstimator

    ds = FreiHANDEval(root)
    ds = ds.subset(np.linspace(0, len(ds) - 1, num).astype(int))
    student.eval()
    est = HandEstimator(model, cfg, student=student)
    errs, errs_rig = [], []
    for i in range(len(ds)):
        fr = ds.frame(i)
        g = fr["hands"][0]
        hands = est.process_image(fr["image"], boxes=g["bbox"][None], is_right=[True], cam_int=fr["K"])
        h = hands[0]
        for pred, out in ((h.get("keypoints_3d_mano"), errs), (h["keypoints_3d"], errs_rig)):
            if pred is not None:
                out.append(np.linalg.norm(procrustes(pred, g["joints3d"]) - g["joints3d"], axis=1).mean() * 1000)
    student.train()
    return dict(freihand_pa_mpjpe=float(np.mean(errs)), freihand_pa_mpjpe_rig=float(np.mean(errs_rig)))


# ---------------------------------------------------------------------------- train
def main(args):
    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    else:
        rank, world = 0, 1
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from fah import load_hand_model

    teacher, cfg = load_hand_model(args.checkpoint, device=dev, backbone_dtype=torch.bfloat16, verbose=False)
    if args.teacher_ckpt:
        from train.release_guards import finetuned_teacher_state

        fine_tune = torch.load(args.teacher_ckpt, map_location="cpu", weights_only=False)
        state = finetuned_teacher_state(fine_tune, teacher.state_dict())
        teacher.load_state_dict(state, strict=False)  # unchanged tensors retain the base SAM weights
        if rank == 0:
            print(f"distillation teacher: {args.teacher_ckpt} ({len(state)} updated tensors)", flush=True)
    elif rank == 0:
        print("distillation teacher: base SAM checkpoint (no --teacher_ckpt supplied)", flush=True)
    teacher.eval().requires_grad_(False)
    student = StudentEncoder(args.student, pretrained=not args.init_teacher).to(dev)
    if args.init_teacher:  # student ViT = the teacher's own backbone (same architecture; official DINOv3 keys -> timm)
        from timm.models.eva import checkpoint_filter_fn

        sd = checkpoint_filter_fn(teacher.backbone.encoder.state_dict(), student.vit)
        missing, unexpected = student.vit.load_state_dict(sd, strict=False)
        if rank == 0:
            print("init from teacher backbone: missing", missing, "unexpected", unexpected, flush=True)
    if args.freeze_vit:  # only the adapter learns (e.g. after --init_teacher)
        student.vit.requires_grad_(False)
    s_in = student.input_size

    start_step, wandb_id = 0, None
    ckpt_path = os.path.join(args.out, "last.pt")
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ck["student"])
        start_step, wandb_id = ck["step"], ck.get("wandb_id")

    model = torch.nn.parallel.DistributedDataParallel(student, device_ids=[torch.cuda.current_device()]) if ddp else student
    vit_params = [p for n, p in student.named_parameters() if n.startswith("vit.")]
    head_params = [p for n, p in student.named_parameters() if not n.startswith("vit.")]
    opt = torch.optim.AdamW([dict(params=vit_params, lr=args.lr), dict(params=head_params, lr=args.lr * args.head_lr_mult)],
                            weight_decay=args.wd, betas=(0.9, 0.999))
    base_lrs = [g["lr"] for g in opt.param_groups]
    if start_step and "opt" in ck:
        opt.load_state_dict(ck["opt"])

    sources = args.sources.split(",")
    ds = CropPool(args.crops, sources, size=args.teacher_size)
    weights = [float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(sources)
    counts = np.bincount(ds.src, minlength=len(sources))
    w = np.asarray(weights)[ds.src] / counts[ds.src]  # source mix independent of pool sizes
    n_per_epoch = args.batch_size * world * 1000

    def make_loader(epoch):
        g = torch.Generator().manual_seed(args.seed * 1000 + epoch)
        idx = torch.multinomial(torch.as_tensor(w), n_per_epoch, replacement=True, generator=g)
        idx = idx[rank::world].tolist()
        return DataLoader(torch.utils.data.Subset(ds, idx), batch_size=args.batch_size, num_workers=args.workers,
                          pin_memory=True, drop_last=True, prefetch_factor=4 if args.workers else None,
                          worker_init_fn=seed_worker)

    run = None
    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        print(f"pool: {len(ds)} crops, per source {dict(zip(sources, counts.tolist()))}, mix {weights}", flush=True)
        if args.wandb:
            import wandb

            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "FAH"),
                             entity=os.environ.get("WANDB_ENTITY"), name=args.run_name,
                             id=wandb_id, resume="allow", config=vars(args), dir=args.out)
            wandb_id = run.id

    def save(step):
        if rank != 0:
            return
        ck = dict(student=student.state_dict(), opt=opt.state_dict(), step=step, wandb_id=wandb_id,
                  args=vars(args), student_name=args.student)
        torch.save(ck, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)

    def evaluate(step):
        if rank == 0 and args.freihand:
            t0 = time.time()
            m = eval_freihand(teacher, cfg, student, args.freihand, args.eval_num)
            m["eval_s"] = time.time() - t0
            print(f"[eval {step}] {m}", flush=True)
            if run:
                run.log({f"eval/{k}": v for k, v in m.items()}, step=step)
            best = os.path.join(args.out, "best.json")
            prev = json.load(open(best))["freihand_pa_mpjpe"] if os.path.exists(best) else 1e9
            if m["freihand_pa_mpjpe"] < prev:
                torch.save(dict(student=student.state_dict(), step=step, student_name=args.student, metrics=m),
                           os.path.join(args.out, "best.pt"))
                json.dump(dict(step=step, **m), open(best, "w"))
        if ddp:
            dist.barrier()

    if start_step == 0:
        evaluate(0)
    step, epoch = start_step, start_step // 1000
    t_last, seen = time.time(), 0
    while step < args.steps:
        for img in make_loader(epoch):
            if step >= args.steps:
                break
            lr_scale = min(1.0, (step + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(step / args.steps, 1.0)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * lr_scale
            x_t = img.to(dev, non_blocking=True).float() / 255.0
            with torch.no_grad():
                tgt = teacher_target(teacher, x_t)  # (B, 1280, 16, 16) fp32
            xs = F.interpolate(x_t, size=(s_in, s_in), mode="bilinear", align_corners=False, antialias=True)
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                pred = model(xs)
            pred = pred.float()
            mse = F.mse_loss(pred, tgt)
            cos = 1 - F.cosine_similarity(pred, tgt, dim=1).mean()
            loss = mse / tgt.var().detach() + args.cos_w * cos  # scale-free feature MSE
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            step += 1
            seen += img.shape[0] * world
            if rank == 0 and step % 50 == 0:
                dt = time.time() - t_last
                log = dict(loss=loss.item(), mse=mse.item(), cos=cos.item(), grad_norm=gn.item(),
                           lr=opt.param_groups[0]["lr"], img_per_s=seen / dt, tgt_std=tgt.std().item())
                print(f"step {step} " + " ".join(f"{k}={v:.4g}" for k, v in log.items()), flush=True)
                if run:
                    run.log({f"train/{k}": v for k, v in log.items()}, step=step)
                t_last, seen = time.time(), 0
            if step % args.save_every == 0:
                save(step)
            if step % args.eval_every == 0:
                evaluate(step)
        epoch += 1
    save(step)
    evaluate(step)
    if run:
        run.finish()
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", required=True)
    p.add_argument("--sources", default="coco_wb,cmu")
    p.add_argument("--weights", default="", help="per-source sampling weights (default equal)")
    p.add_argument("--student", default="dinov3_vits16", choices=list(STUDENTS))
    p.add_argument("--checkpoint", default="checkpoints/sam-3d-body-dinov3/model.ckpt")
    p.add_argument("--teacher_ckpt", default="", help="train.finetune teacher checkpoint; applied before feature targets and --init_teacher")
    p.add_argument("--teacher_size", type=int, default=384)
    p.add_argument("--freihand", default="", help="FreiHAND root (evaluation set) for the decoder-in-the-loop eval")
    p.add_argument("--eval_num", type=int, default=500)
    p.add_argument("--out", required=True)
    p.add_argument("--run_name", default=None)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--batch_size", type=int, required=True, help="training samples per process; choose for your setup")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--head_lr_mult", type=float, default=5.0)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--cos_w", type=float, default=1.0)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init_teacher", action="store_true", help="student ViT from the teacher's backbone weights (dinov3_vith16plus)")
    p.add_argument("--freeze_vit", action="store_true", help="train the adapter only")
    p.add_argument("--wandb", action="store_true", help="log to Weights & Biases (opt-in; WANDB_PROJECT / WANDB_ENTITY from the environment)")
    main(p.parse_args())
