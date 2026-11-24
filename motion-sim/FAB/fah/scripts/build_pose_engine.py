"""TensorRT FP16 engine of the pose model's student encoder (fah.student.StudentEncoder: ViT + adapter) for
HandEstimator(student_trt=...): crops (B, 3, S, S) RGB [0, 1] fp32 -> features (B, out_dim, grid, grid) fp32, dynamic batch
1..max_batch (HandEstimator's max_batch_size 8; optimum 2 = two hands). Two stages (modelopt, needed for the fp16 conversion of TensorRT 11's strongly
typed networks, lives in its own venv):

    # 1) main env (needs onnx): ONNX export of the fine-tuned encoder + calibration crops
    python scripts/build_pose_engine.py export --finetune_ckpt $FAH_WEIGHTS/fah/model.pt --work /tmp/trt_fah --calib_frames <dir of images>
    # 2) TensorRT venv (tensorrt + nvidia-modelopt[onnx]): fp16 conversion and engine build on the target GPU
    python scripts/build_pose_engine.py build --work /tmp/trt_fah --out $FAH_WEIGHTS/trt/fah-pose-fp16.engine
    # fp8 / int8: calibration crops from training pools first, then build --precision fp8 and compare against fp16
    python scripts/build_pose_engine.py calib --work /tmp/trt_fah --pools $DATA_ROOT/crop_pools/<pool>.tar,...
    python scripts/build_pose_engine.py build --precision fp8 --work /tmp/trt_fah --out $FAH_WEIGHTS/trt/fah-pose-fp8.engine
    python scripts/build_pose_engine.py compare --work /tmp/trt_fah --engines <fp16 engine>,<fp8 engine>
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fah.api import DEFAULT_SAM3D, DEFAULT_WEIGHTS  # noqa: E402


def export(args):
    import torch

    from bench.methods.fah import FAHMethod

    os.makedirs(args.work, exist_ok=True)
    ck = os.path.join(args.sam3d, "model.ckpt")
    m = FAHMethod(checkpoint=ck, base_weights=os.path.exists(ck), dtype="bfloat16", cuda_graph=False, fast_det=True,
                      finetune_ckpt=args.finetune_ckpt, detector_weights=args.detector, verify=True, min_crop=1.1, gpu_frame=True,
                      contact_ckpt=args.contact_ckpt)
    hb = args.hidden_block if args.hidden_block >= 0 else getattr(m.est.model.keypoint_head, "contact_branch_block", None)
    st = m.est.student
    caps = []
    fwd = m.est._forward

    def cap(batch):
        caps.append(batch["img"][:, 0].detach().clone())
        return fwd(batch)

    m.est._forward = cap
    frames = sorted(glob.glob(os.path.join(args.calib_frames, "*[0-9].npy")))[:2]
    for f in frames:  # crops of real frames (benchmark_latency cache) for the fp16 value-range check
        fr = np.load(f, mmap_mode="r")
        K = np.load(f[:-4] + "_K.npy").astype(np.float32)
        for i in range(0, 40, 8):
            m._verified(np.ascontiguousarray(fr[i]), K if K.ndim == 2 else K[i])
    if not frames:  # a dir of images with hands (default intrinsics)
        import cv2

        from fah import default_cam_int

        for f in sorted(glob.glob(os.path.join(args.calib_frames, "*.jp*g")) + glob.glob(os.path.join(args.calib_frames, "*.png")))[:10]:
            img = cv2.imread(f)
            m._verified(img, default_cam_int(*img.shape[:2]))
    assert caps, f"no hands found in --calib_frames {args.calib_frames}"
    x = torch.cat(caps)[:2]
    x = torch.nn.functional.interpolate(x, size=(st.input_size,) * 2, mode="bilinear", align_corners=False, antialias=True)
    np.savez(os.path.join(args.work, "calib.npz"), x=x.float().cpu().numpy())

    class Wrap(torch.nn.Module):  # StudentEncoder.forward with the grid fixed (its int(round(N ** 0.5)) does not trace)
        def __init__(self, s):
            super().__init__()
            self.s = s

        def forward(self, x):
            s = self.s
            t = s.vit.forward_features((x - s.mean) / s.std)[:, s.num_prefix:]
            return s.adapter(t).transpose(1, 2).reshape(x.shape[0], s.out_dim, s.grid, s.grid)

    class WrapHidden(Wrap):  # + the hidden state entering block n (every token): a contact branch's input, second output
        def __init__(self, s, n):
            super().__init__(s)
            self.n = n

        def forward(self, x):
            got = []
            hook = self.s.vit.blocks[self.n].register_forward_pre_hook(lambda mod, a: got.append(a[0]))
            try:
                feat = super().forward(x)
            finally:
                hook.remove()
            return feat, got[0]

    net, outs = (Wrap(st.float().eval()), ["feat"]) if hb is None else (WrapHidden(st.float().eval(), hb), ["feat", f"hidden{hb}"])
    with torch.no_grad():
        torch.onnx.export(net, x.float(), os.path.join(args.work, "student.onnx"), input_names=["x"], output_names=outs,
                          dynamic_axes={"x": {0: "b"}, **{o: {0: "b"} for o in outs}}, opset_version=19, dynamo=False)
    print("exported ->", args.work, "input", tuple(x.shape), "outputs", outs)


def calib(args):
    """Training-pool crops for fp8 / int8 calibration (no benchmark frames): every k-th image of each pool tar
    (train.build_crops 640 crops), the centred teacher region (CROP / CONTEXT) resized to the student input."""
    import tarfile

    import cv2

    from train.distill import CONTEXT, CROP

    pools, xs = args.pools.split(","), []
    per = args.n // len(pools)
    for pool in pools:
        got, seen = 0, 0
        with tarfile.open(pool, "r|") as tf:
            for m in tf:
                if not m.isfile() or not m.name.endswith((".jpg", ".png")) or "/masks/" in m.name:
                    continue
                seen += 1
                if seen % args.every:
                    continue
                img = cv2.imdecode(np.frombuffer(tf.extractfile(m).read(), np.uint8), cv2.IMREAD_COLOR)
                if img is None or img.shape[0] != CROP:
                    continue
                o = int(round(CROP * (1 - 1 / CONTEXT) / 2))
                v = cv2.resize(img[o:CROP - o, o:CROP - o, ::-1], (args.size, args.size), interpolation=cv2.INTER_AREA)
                xs.append(v.transpose(2, 0, 1).astype(np.float32) / 255.0)
                got += 1
                if got >= per:
                    break
        print(pool, got)
    x = np.stack(xs)
    os.makedirs(args.work, exist_ok=True)
    np.savez(os.path.join(args.work, "calib_train.npz"), x=x)
    print("calib ->", os.path.join(args.work, "calib_train.npz"), x.shape)


def compare(args):
    """Engines (--engines a,b,...; the first = reference) on the calibration crops: feature error vs the reference and
    the median latency per batch size (TRTEncoder on one stream, 200 calls after warm-up)."""
    import torch

    from fah.student import TRTEncoder

    x = torch.as_tensor(np.load(os.path.join(args.work, "calib_train.npz"))["x"][-64:]).cuda()
    engines = args.engines.split(",")
    ref = None
    for e in engines:
        enc = TRTEncoder(e)
        f = torch.cat([enc(x[i:i + 2]).clone() for i in range(0, len(x), 2)])
        if ref is None:
            ref = f
        rel = ((f - ref).norm() / ref.norm()).item()
        ms = {}
        for B in (1, 2, 4):
            xb = x[:B].contiguous()
            for _ in range(20):
                enc(xb)
            t = []
            for _ in range(200):
                a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                a.record()
                enc(xb)
                b.record()
                b.synchronize()
                t.append(a.elapsed_time(b))
            ms[B] = round(float(np.median(t)), 3)
        print(os.path.basename(e), "rel err %.4f" % rel, "max abs %.4f" % (f - ref).abs().max().item(), "ms", ms)


def build(args):
    import onnx
    import tensorrt as trt
    from modelopt.onnx.autocast import convert_to_mixed_precision

    src, f16 = os.path.join(args.work, "student.onnx"), os.path.join(args.work, f"student_{args.precision}.onnx")
    x = np.load(os.path.join(args.work, "calib.npz"))["x"]
    if args.precision == "fp16":
        cal = os.path.join(args.work, "calib.npz")
        if args.fp16_calib_n:  # value ranges from training-pool crops (calib stage) instead of benchmark-frame crops
            xt = np.load(os.path.join(args.work, "calib_train.npz"))["x"]
            cal = os.path.join(args.work, "calib_fp16_train.npz")
            np.savez(cal, x=xt[np.linspace(0, len(xt) - 1, args.fp16_calib_n).astype(int)])
        onnx.save(convert_to_mixed_precision(src, low_precision_type="fp16", keep_io_types=True, calibration_data=cal),
                  f16, save_as_external_data=True)
    else:  # fp8 / int8 Q/DQ (modelopt PTQ on training crops), the rest fp16
        from modelopt.onnx.quantization import quantize

        quantize(src, quantize_mode=args.precision, calibration_data={"x": np.load(os.path.join(args.work, "calib_train.npz"))["x"]},
                 calibration_method=args.calib_method, calibration_eps=args.calib_eps.split(","), output_path=f16, high_precision_dtype="fp16",
                 use_external_data_format=True, disable_mha_qdq=args.no_mha_qdq,
                 op_types_to_quantize=args.quant_ops.split(",") if args.quant_ops else None,
                 nodes_to_exclude=args.exclude_nodes.split(",") if args.exclude_nodes else None)
    lg = trt.Logger(trt.Logger.WARNING)
    b = trt.Builder(lg)
    net = b.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    p = trt.OnnxParser(net, lg)
    assert p.parse_from_file(f16), [p.get_error(i) for i in range(p.num_errors)]
    cfg = b.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
    pr = b.create_optimization_profile()
    S = x.shape[-1]
    pr.set_shape("x", (1, 3, S, S), (2, 3, S, S), (args.max_batch, 3, S, S))
    cfg.add_optimization_profile(pr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    open(args.out, "wb").write(b.build_serialized_network(net, cfg))
    print("engine ->", args.out, os.path.getsize(args.out) / 1e6, "MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["export", "calib", "build", "compare"])
    p.add_argument("--work", required=True, help="scratch directory (ONNX, calibration crops)")
    p.add_argument("--finetune_ckpt", default="")
    p.add_argument("--contact_ckpt", default="", help="export: a contact.pt; with a contact branch the engine also gives the "
                   "branch's input (the encoder's hidden state at its first block) as output hidden<N>")
    p.add_argument("--hidden_block", type=int, default=-1, help="export: output hidden<N> at this block (default: the contact "
                   "branch's first block, none without a branch)")
    p.add_argument("--detector", default=os.path.join(DEFAULT_WEIGHTS, "shared", "detector.pt"))
    p.add_argument("--sam3d", default=DEFAULT_SAM3D, help="SAM 3D Body config dir (fah.download_sam3d)")
    p.add_argument("--calib_frames", required=False, default="", help="dir of images, or a scripts/benchmark_latency.py frame cache (*.npy)")
    p.add_argument("--out", default="")
    p.add_argument("--max_batch", type=int, default=8, help="HandEstimator max_batch_size")
    p.add_argument("--precision", default="fp16", choices=["fp16", "fp8", "int8"])
    p.add_argument("--fp16_calib_n", type=int, default=0, help="fp16: value ranges from this many training-pool crops (calib stage)")
    p.add_argument("--calib_method", default="max", help="fp8 / int8 calibration (modelopt: max / entropy)")
    p.add_argument("--calib_eps", default="cpu", help="onnxruntime providers for the calibration pass (the TensorRT venv's cuDNN breaks the CUDA one)")
    p.add_argument("--no_mha_qdq", action="store_true", help="fp8 / int8: attention matmuls stay fp16")
    p.add_argument("--quant_ops", default="", help="fp8 / int8: only these op types (e.g. MatMul,Gemm)")
    p.add_argument("--exclude_nodes", default="", help="fp8 / int8: node-name regexes kept in fp16")
    p.add_argument("--pools", default="", help="calib: comma-separated training crop-pool tars")
    p.add_argument("--n", type=int, default=512, help="calib: number of crops")
    p.add_argument("--every", type=int, default=20, help="calib: every k-th image of a pool")
    p.add_argument("--size", type=int, default=320, help="calib: student input size")
    p.add_argument("--engines", default="", help="compare: comma-separated engines, the first = reference")
    a = p.parse_args()
    {"export": export, "calib": calib, "build": build, "compare": compare}[a.stage](a)
