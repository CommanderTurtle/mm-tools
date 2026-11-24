"""Train one FAH component using configs/local.json. Use --dry-run to inspect the recipe."""
import argparse
import sys
from pathlib import Path
import subprocess

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._recipes import MODELS, ROOT, POSE_POOLS, CONTACT_POOLS, args, config, distributed, execute, module, path, pools, positive, required

STAGES = ("teacher", "distill", "heads", "pose", "parts", "contact", "contact-folds", "contact-video",
          "detector-labels", "detector", "mesh", "mhr", "video", "camera")


def commands(c, stage, model=None):
    model = model or c.get("model", "fah")
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    data, ckpt = path(c, "data_root"), path(c, "ckpt_root")
    crops, teacher = data / "crops", ckpt / "finetune/teacher/final.pt"
    batch = positive(c, "train_batch_size") if stage != "detector-labels" else None
    weights = path(c, "weights")
    checkpoint = str(path(c, "sam_checkpoint")) if c.get("sam_checkpoint") else None
    upstream = args(checkpoint=checkpoint)
    distill = ckpt / "distill" / model / "best.pt"
    pose_dir = ckpt / "finetune" / model
    init = pose_dir / "heads/final.pt"
    released = path(c, "model_checkpoint", str(weights / model / "model.pt"))
    size = 448 if model == "fah-h" else 320
    def fine(*extra):
        return distributed(c, "train.finetune", *upstream, *extra)
    common = args(crops=crops) + pools(c, "pose_pools", POSE_POOLS) + [
        "--kp_head", "--kp_only", "--kp_dense", "--vis_head", "--vis_gate", "--verify_head"] + args(
        kp_dim=64, p_wrong=.15, wroot=1., wrep=1., wrep_surf=.5, wsmooth=1., smooth_k=200,
        wsurf_prior=1., dense_n=400, dense_sel="fps", p_dark=.3, p_ego=.25, p_hocc=.2, box_jitter=1.6,
        min_crop=1.1, size=448 if model == "fah-h" else 384, student_size=size, enc_lr_mult=.5,
        warmup=300, skip_gn=1000, w3d_sources="egodex_test_mano:0.5,egodex_part1_mano:0.5",
        kd_sources="egodex_test_mano:0.5,egodex_part1_mano:0.5",
        teacher_regressor=ROOT / "train/assets/mhr_to_mano21_kin_regressor.npz", teacher_ckpt=teacher,
        freihand=data / "freihand", batch_size=batch) + ["--kp_depth"]
    if model == "fah-h":
        common += ["--grad_ckpt"]
    if stage == "teacher":
        return [fine(*args(crops=crops, sources="hot3d_lab,coco_wb,cmu_lab,hint_lab,freihand_lab,egodex_test,halpe_lab",
                          weights="1,1,0.3,0.6,0.5,0.6,0.8", w3d_sources="egodex_test:0.3", train_blocks=8,
                          enc_lr_mult=.25, size=384, steps=10000, batch_size=batch, lr=2e-5, warmup=500,
                          freihand=data / "freihand", out=teacher.parent))]
    if stage == "distill":
        cmd = distributed(c, "train.distill", *upstream, *args(crops=crops, sources="hot3d_lab_mesh,coco_wb,cmu_lab",
                          weights="1,1,0.15", student=MODELS[model][1], steps=2000 if model == "fah-h" else 20000,
                          batch_size=batch, lr=2e-4, teacher_size=384, teacher_ckpt=teacher,
                          freihand=data / "freihand", out=distill.parent))
        if model == "fah-h":
            cmd += ["--init_teacher", "--freeze_vit", "--warmup", "200"]
        return [cmd]
    if stage == "heads":
        return [fine(*common, *args(student=distill, steps=80000, lr=1e-4,
                                   enc_lr_mult=.1, out=init.parent))]
    if stage == "pose":
        return [fine(*common, *args(student=init, init_ckpt=init, steps=30000, lr=1e-4,
                                   out=pose_dir / "pose"))]
    if stage == "parts":
        base = pose_dir / "pose/final.pt"
        return [fine(*common, *args(student=base, init_ckpt=base, steps=16000, lr=1e-4,
                     w_partseg=1., w_partmask=1., part_depth=6, partseg_finger_w=3, part_routes="mesh+mask,model+mask",
                     part_sources="hot3d_lab_mesh,dexycb_lab,oakink2_ego_lab,hanco_lab,freihand_lab_mesh_kin,coco_wb_kin,cmu_lab_kin,hint_lab_kin",
                     part_dirs="coco_wb_kin:masks_parts_model,hint_lab_kin:masks_parts_model,cmu_lab_kin:masks_parts_model",
                     out=pose_dir / "parts"), "--train_only_parts", "--part_hr")]
    if stage in ("contact", "contact-folds"):
        base = module(c, "train.contact_head", *upstream, *args(crops=crops, contact=data / "contact/root",
                      init_ckpt=released, detector_weights=weights / "shared/detector.pt"),
                      *pools(c, "contact_pools", CONTACT_POOLS), *args(zero_sources="interhand_lab", layers=4,
                      contact_dim=256, branch=4, branch_lr=1e-4, dists="5,10,20,35", steps=24000,
                      batch_size=batch, lr=3e-4, eval_every=2000, size=448 if model == "fah-h" else 384))
        if stage == "contact":
            return [base + args(out=ckpt / "contact" / model)]
        return [base + args(steps=8000, fold_out=k, fold_pools="hot3d_lab_mesh,dexycb_lab,arctic_ego,hoi4d",
                           fold_key="sequence", out=ckpt / "contact" / f"{model}-f{k}") for k in (0, 1)]
    if stage == "contact-video":
        return [module(c, "train.contact_video", "train", *args(train=required(c, "contact_tracks"), vidx=released,
                       noise=.05, steps=20000, batch=batch, lr=5e-4, out=ckpt / "contact_video" / model), "--use_tok")]
    if stage == "detector-labels":
        boxes, sam, seg = data / "det_boxes", data / "det_sam3", data / "det_seg"
        cmd = module(c, "train.build_det_seg_sam3", "sam3", *args(src=boxes, sam_dir=sam))
        cmd[0] = str(c.get("sam_python") or cmd[0])
        if c.get("sam3_checkpoint"):
            cmd += args(ckpt=path(c, "sam3_checkpoint"))
        return [cmd, module(c, "train.build_det_seg_sam3", "write", *args(src=boxes, sam_dir=sam, dst=seg,
                    recover=0, kp_join=.04, min_kp_in=.6, split_pairs=.5)),
                module(c, "train.build_det_dark", *args(src=seg, dst=data / "det_seg_dark"))]
    if stage == "detector":
        return [module(c, "train.train_detector", *args(data=data / "det_seg_dark/hands.yaml",
                       init=weights / "shared/detector.pt", imgsz=800, epochs=6, batch=batch,
                       device=required(c, "train_devices"), out=ckpt / "detector", name="detector"))]
    if stage in ("mesh", "mhr"):
        cmd = module(c, "train.mesh_fitter" if stage == "mesh" else "train.mhr_fitter", "train", *args(
                     data=path(c, "mesh_data" if stage == "mesh" else "mhr_data"), out=ckpt / stage,
                     noise=.003, epochs=20, batch=batch, lr=1e-3, hidden=1024))
        if stage == "mhr":
            cmd += upstream + args(vreg=ROOT / "fah/assets/mhr_to_mano778.npz")
        return [cmd]
    if stage == "video":
        return [module(c, "train.temporal", "train", *args(train=required(c, "pose_tracks"), out=ckpt / "video",
                       T=32, dim=256, layers=4, steps=20000, batch=batch, lr=5e-4, w_root=.1, p_drop=.1,
                       noise=.01, w_pa=1., w_acc=3., geo_shift=.3, geo_depth=.4, geo_rot=3.1416, strides="1,2,3"),
                       "--causal", "--geo_aug")]
    if stage == "camera":
        # The loader also enforces the reserved EgoDex holdout and HOI4D subject split.
        exclusion = ckpt / "camera_exclude.json"
        prepare = [str(c.get("python") or sys.executable), "-c",
                   "import json,pathlib,sys; p=pathlib.Path(sys.argv[2]); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(json.load(open(sys.argv[1]))['hawor_val_sequences']))",
                   str(ROOT / "bench/splits/hot3d_bench.json"), str(exclusion)]
        return [prepare, distributed(c, "train.camera_motion", *args(data=data / "camframes", out=ckpt / "camera_motion",
                       sources="hot3d:0.25,egodex:0.3,arctic:0.2,hoi4d:0.25", exclude=exclusion,
                       val_frac=.08, window=9, fast=.3, crop_min=.5, batch_size=batch, long_pairs=4, w_long=2,
                       strides="1,2,3,4,6,8,10", steps=12000, lr=1e-4, warmup=300), "--raft", "--no_corr")]
    raise ValueError(f"Unknown stage: {stage}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--config", default="configs/local.json")
    p.add_argument("--model", choices=list(MODELS))
    p.add_argument("--dry-run", action="store_true", help="print commands without starting training")
    a = p.parse_args()
    try:
        c = config(a.config)
        execute(commands(c, a.stage, a.model), c, a.dry_run)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        p.exit(1, f"{e}\n")


if __name__ == "__main__":
    main()
