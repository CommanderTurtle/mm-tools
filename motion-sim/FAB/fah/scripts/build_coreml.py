"""Core ML (mlprogram) exports of the FAH networks for Apple silicon, the counterparts of the TensorRT engines
(scripts/build_pose_engine.py, scripts/build_det_engine.py), loaded by fah.coreml; and the end-to-end parity of a
Core ML configuration against the PyTorch path. `fah-build-coreml` runs all three for a released model (macOS).

pose: the student encoder of a finetune checkpoint (crops (B, 3, S, S) RGB [0, 1] -> features, HandEstimator student_coreml),
or with --heads the whole decoder-free pose network (encoder, mask prompt, ray embedding, keypoint / depth / verification
heads: crops + per-crop ray affine + CLIFF condition -> the kp_only outputs of SAM3DHand.forward; --no_parts: without the
part head, the other outputs unchanged). Fixed shapes, one function
per batch size (b1, b2: shared weights; larger batches run in chunks), float32 by default (optional float16); compute units are chosen
at load time (GPU: the Neural Engine is no faster on the ViT and much less accurate).
det: the YOLO(-seg) network (conv + BN fused, as the fast detector path) at the letterbox shape of each --frames size
(one function per shape), fp16, raw head outputs (box distributions, class logits, mask coefficients, protos: the box
decode, NMS and masks stay in fah.detector, on the host in fp32).
parity: bench.methods.fah on local frames with --base_args and with --base_args + each of --args; matched hands (same
frame and handedness, box IoU > 0.5): 3D / 2D joint and mesh vertex differences (median / p90 / max) and unmatched hands.

    python scripts/build_coreml.py pose --finetune_ckpt $FAH_WEIGHTS/fah-s/model.pt --heads \
        --detector_weights $FAH_WEIGHTS/shared/detector.pt --out $FAH_WEIGHTS/coreml/fah-s-pose.mlpackage
    python scripts/build_coreml.py det --weights $FAH_WEIGHTS/shared/detector.pt --imgsz 640 --frames 1920x1080,1440x1080,1080x1080 \
        --out $FAH_WEIGHTS/coreml/detector-640.mlpackage
    python scripts/build_coreml.py parity --frames_dir frames/ --base_args "$(cat fahS_fp32.json)" \
        --args '[{"dtype": "float16", "det_half": true}, {"pose_coreml": "...", "det_coreml": "...", "mesh_fit_device": "cpu"}]' \
        --names mps_fp16,coreml --cache parity_cache_S
"""
import argparse
import glob
import json
import os
import sys
import tempfile
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fah.api import DEFAULT_SAM3D  # noqa: E402


def rescale_residual(enc, x, limit=8192.0):
    """fp16-safe copy of a student encoder: its ViT's residual stream divided by a power of two s (patch embedding and prefix
    tokens / s, every attention / MLP output projection / s, eps / s^2 in the LayerNorms that read the stream: the same
    function). DINOv3 ViT-L carries ~9e4 activations in a few channels for any input (fp16 overflows above 65504; the fp16
    MPS path returns no hands), ViT-S ~1e3 (s = 1). s from the largest residual on x, to at most `limit`."""
    import copy

    import torch

    v, mx = enc.vit, [0.0]
    hooks = [b.register_forward_hook(lambda m, i, o: mx.__setitem__(0, max(mx[0], o.abs().max().item()))) for b in v.blocks]
    with torch.no_grad():
        enc(x)
    for h in hooks:
        h.remove()
    s = float(2 ** max(int(np.ceil(np.log2(mx[0] / limit))), 0))
    if s == 1:
        return enc, s
    enc = copy.deepcopy(enc)
    v = enc.vit
    with torch.no_grad():
        for p in (v.patch_embed.proj.weight, v.patch_embed.proj.bias, v.cls_token, getattr(v, "reg_token", None),
                  getattr(v, "pos_embed", None)):
            if p is not None:
                p /= s
        for b in v.blocks:
            for lin in (b.attn.proj, b.mlp.fc2):
                lin.weight /= s
                lin.bias /= s
            b.norm1.eps, b.norm2.eps = b.norm1.eps / s ** 2, b.norm2.eps / s ** 2
        for n in (v.norm_pre, v.norm):
            if isinstance(n, torch.nn.LayerNorm):
                n.eps = n.eps / s ** 2
    return enc, s


def pose_net(est, heads, fold=True, parts=True):
    import torch
    import torch.nn.functional as F

    class PoseNet(torch.nn.Module):
        """est's student (+ the kp_only branch of SAM3DHand.forward) with static shapes. ray (B, 4) = [sx, ox, sy, oy]: the
        crop's ray grid is x = sx u + ox, y = sy v + oy (u, v crop pixels); the CameraEncoder's antialiased 1 / patch
        downsampling is linear, so it is applied once to the pixel grid (buffer) instead of per crop. fold: the adapter's last
        linear layer (+ its bias and the no-mask embedding) folded into the image half of the ray embedding's 1x1 conv (no
        nonlinearity between them: one 1280 x 1280 matmul per token less), token-major throughout (its LayerNorm2d = a
        LayerNorm over the channels)."""

        def __init__(self):
            super().__init__()
            m = est.model  # only the used submodules (the MHR TorchScript model does not export)
            self.enc, self.residual_scale = rescale_residual(est.student, torch.rand(2, 3, est.student.input_size, est.student.input_size))
            self.ce, self.kh, self.vh = m.ray_cond_emb_hand, m.keypoint_head, getattr(m, "verify_head", None)
            if not parts and getattr(self.kh, "part_emb", None) is not None:  # part-free program: the part branch is not traced
                import copy

                self.kh = copy.deepcopy(self.kh)
                self.kh.part_emb = None
            self.register_buffer("image_mean", m.image_mean.clone(), False)
            self.register_buffer("image_std", m.image_std.clone(), False)
            nm = m.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None  # no mask prompt: + the no-mask embedding
            self.register_buffer("no_mask", m.prompt_encoder.no_mask_embed.weight.detach().reshape(1, -1, 1, 1).clone() if nm else
                                 torch.zeros(1, 1, 1, 1), False)
            S = est.input_size[0]
            assert self.enc.input_size == S, "student input size != crop size"
            v, u = torch.meshgrid(torch.arange(S, dtype=torch.float32), torch.arange(S, dtype=torch.float32), indexing="ij")
            ps = self.ce.patch_size
            self.register_buffer("uv_ds", F.interpolate(torch.stack([u, v])[None], scale_factor=(1 / ps, 1 / ps), mode="bilinear",
                                                        align_corners=False, antialias=True)[0], False)
            if heads and fold:
                with torch.no_grad():
                    W = self.ce.conv.weight[:, :, 0, 0].double()
                    Wi, Wr, l2 = W[:, :self.enc.out_dim], W[:, self.enc.out_dim:], self.enc.adapter[3]
                    self.register_buffer("w_fold", (Wi @ l2.weight.double()).float(), False)
                    self.register_buffer("b_fold", (Wi @ (l2.bias.double() + self.no_mask.reshape(-1).double())).float(), False)
                    self.register_buffer("w_ray", Wr.float().clone(), False)
            self.outs = []

        def forward(self, img, ray=None, cond=None):
            e = self.enc
            t = e.vit.forward_features((img - e.mean) / e.std)[:, e.num_prefix:]
            B, g = img.shape[0], e.grid
            if not heads:
                return e.adapter(t).transpose(1, 2).reshape(B, e.out_dim, g, g)
            x = (img - self.image_mean) / self.image_std
            ce = self.ce
            r = self.uv_ds[None] * ray[:, [0, 2], None, None] + ray[:, [1, 3], None, None]  # (B, 2, g, g)
            r = r.permute(0, 2, 3, 1)
            r = torch.cat([r, torch.ones_like(r[..., :1])], -1).reshape(B, g * g, 3)
            re = ce.camera(pos=r)  # (B, g g, 99)
            if fold:
                h = e.adapter[2](e.adapter[1](e.adapter[0](t)))
                tok = F.layer_norm(F.linear(h, self.w_fold, self.b_fold) + F.linear(re, self.w_ray), (self.w_fold.shape[0],),
                                   ce.norm.weight, ce.norm.bias, ce.norm.eps)
            else:
                emb = e.adapter(t).transpose(1, 2).reshape(B, e.out_dim, g, g) + self.no_mask
                emb = ce.norm(ce.conv(torch.cat([emb, re.transpose(1, 2).reshape(B, -1, g, g)], 1)))
                tok = emb.flatten(2).transpose(1, 2)
            kh = self.kh
            pose = kh.pool(torch.cat([tok.float().mean(1), cond.float()], -1))
            uv, xyz, _ = kh(tok, (g, g), pose, x, cond=cond, return_conf=True)
            res = dict(pt_uv=uv, pt_xyz=xyz, pt_conf=kh.last_conf)
            if getattr(kh, "vis", None) is not None:
                res["pt_vis"] = kh.last_vis
            if getattr(kh, "part_emb", None) is not None:
                res["part_logits"] = kh.last_part_logits
            if getattr(kh, "depth", None) is not None:
                res["pt_depth"] = kh.depth(pose)[:, 0]
                res["pt_token"] = pose
            if self.vh is not None:
                res["verify_logits"] = self.vh(pose)
            self.outs = list(res)
            return tuple(res.values())

    return PoseNet().eval()


def convert(net, inputs, names, path, precision="float16"):
    import coremltools as ct
    import torch

    with torch.no_grad():
        ep = torch.export.export(net, tuple(inputs)).run_decompositions({})
    ml = ct.convert(ep, inputs=[ct.TensorType(name=n, shape=tuple(x.shape)) for n, x in zip(names[0], inputs)],
                    outputs=[ct.TensorType(name=n) for n in names[1]], convert_to="mlprogram",
                    compute_precision=getattr(ct.precision, precision.upper()), minimum_deployment_target=ct.target.macOS15)
    ml.save(path)
    return ml


def save_functions(parts, out, default):
    """parts: [(function name, single-function mlpackage)] -> one multifunction mlpackage (identical weights shared)."""
    import shutil

    import coremltools as ct

    desc = ct.utils.MultiFunctionDescriptor()
    for name, p in parts:
        desc.add_function(p, src_function_name="main", target_function_name=name)
    desc.default_function_name = default
    for o in (out, os.path.splitext(out)[0] + ".mlmodelc"):  # the package and its compiled model (fah.coreml.compiled_path)
        if os.path.exists(o):
            shutil.rmtree(o)
    ct.utils.save_multifunction(desc, out)
    ct.utils.compile_model(out, os.path.splitext(out)[0] + ".mlmodelc")


def build_pose(args):
    import torch

    from bench.methods.fah import FAHMethod

    ck = os.path.join(args.sam3d, "model.ckpt")
    m = FAHMethod(checkpoint=ck, base_weights=os.path.exists(ck), device="cpu", fast_det=True, finetune_ckpt=args.finetune_ckpt,
                  detector_weights=args.detector_weights)
    est = m.est
    assert est.student is not None, "needs a student encoder"
    assert not args.heads or getattr(est.model, "kp_only", False), "--heads needs a decoder-free (kp_only) model"
    assert not args.heads or getattr(est.model, "mano_head", None) is None, "MANO parameter head: not exported"
    net = pose_net(est, args.heads, not args.no_fold, not args.no_parts)
    S = est.input_size[0]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(args.out))) as tmp:
        parts = []
        for B in args.batch:
            torch.manual_seed(0)
            inputs = [torch.rand(B, 3, S, S)]
            if args.heads:
                inputs += [torch.tensor([[0.002, -0.3, 0.002, -0.3]]).repeat(B, 1), torch.zeros(B, 3)]
            with torch.no_grad():
                net(*inputs)
            names = (["img", "ray", "cond"][:len(inputs)], net.outs if args.heads else ["feat"])
            p = os.path.join(tmp, f"b{B}.mlpackage")
            convert(net, inputs, names, p, getattr(args, "precision", "float32"))
            parts.append((f"b{B}", p))
            print(f"batch {B}: outputs {names[1]}, residual scale {net.residual_scale:g}")
        save_functions(parts, args.out, f"b{max(args.batch)}")
    print("saved", args.out)


def letterbox_shape(h, w, imgsz, stride=32):
    """(H, W) of fah.detector's letterbox (ultralytics auto=True: min padding to a stride multiple)."""
    r = min(imgsz / h, imgsz / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    return nh + (imgsz - nh) % stride, nw + (imgsz - nw) % stride


def build_det(args):
    import torch

    from fah.yolo import load_yolo

    net = load_yolo(args.weights)  # batch norms folded, float32 (no ultralytics package)
    seg = net.seg

    class DetNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = net

        def forward(self, x):  # raw head outputs: the box decode (DFL, anchors) runs in fp32 on the host
            d = self.net(x, raw=True)
            return (d["boxes"], d["scores"], d["mask_coefficient"], d["proto"]) if seg else (d["boxes"], d["scores"])

    shapes = sorted({letterbox_shape(*map(int, f.split("x")[::-1]), args.imgsz) for f in args.frames.split(",")})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(args.out))) as tmp:
        parts = []
        for H, W in shapes:
            p = os.path.join(tmp, f"s{H}x{W}.mlpackage")
            convert(DetNet().eval(), [torch.rand(1, 3, H, W)], (["x"], ["boxes", "scores", "mc", "proto"][:4 if seg else 2]), p)
            parts.append((f"s{H}x{W}", p))
            print(f"letterbox {H}x{W}")
        save_functions(parts, args.out, parts[0][0])
    print("saved", args.out)


def compare(base, test):
    """Per frame: base hands matched to test hands of the same side (box IoU > 0.5, greedy); per matched hand the worst
    joint / vertex difference, then median / p90 / max over hands."""
    from bench.common import box_iou

    d, where = defaultdict(list), []
    n_base = n_test = 0
    for f in base:
        a, b = base[f], test[f]
        n_base, n_test = n_base + len(a), n_test + len(b)
        used = set()
        for h in a:
            c = [(box_iou(h["bbox"], g["bbox"]), k) for k, g in enumerate(b) if k not in used and bool(g["is_right"]) == bool(h["is_right"])]
            if not c or max(c)[0] < 0.5:
                continue
            k = max(c)[1]
            used.add(k)
            g = b[k]
            d["joints3d_mm"].append(1000 * np.linalg.norm(h["joints3d"] - g["joints3d"], axis=-1).max())
            where.append(os.path.basename(f))
            d["joints3d_rel_mm"].append(1000 * np.linalg.norm((h["joints3d"] - h["joints3d"][:1]) - (g["joints3d"] - g["joints3d"][:1]), axis=-1).max())
            d["joints2d_px"].append(np.linalg.norm(h["joints2d"] - g["joints2d"], axis=-1).max())
            if "verts3d" in h and "verts3d" in g:
                d["verts3d_mm"].append(1000 * np.linalg.norm(h["verts3d"] - g["verts3d"], axis=-1).max())
            if "surface2d_head" in h and "surface2d_head" in g:  # dense points' 2D (the head's own heatmap 2D)
                e = np.linalg.norm(h["surface2d_head"] - g["surface2d_head"], axis=-1)
                d["dense2d_px"].append(e.max())
                d["dense2d_px_mean"].append(e.mean())  # the max is dominated by hidden points whose visibility flips (2D <- 3D)
            d["bbox_px"].append(np.abs(h["bbox"] - g["bbox"]).max())
            d["score"].append(abs(float(h["score"]) - float(g["score"])))
    tips = lambda r: float(np.mean([(np.asarray(h["tip_ext"]) > 0).any() for v in r.values() for h in v if "tip_ext" in h] or [0]))
    rep = dict(frames=len(base), hands_base=n_base, hands_test=n_test, matched=len(d["joints3d_mm"]),
               tips_extended_base=tips(base), tips_extended_test=tips(test))  # fraction of hands with an extended fingertip
    for k, v in d.items():
        v = np.asarray(v)
        rep[k] = dict(median=float(np.median(v)), p90=float(np.percentile(v, 90)), max=float(v.max()))
    rep["worst_frames"] = [where[i] for i in np.argsort(d["joints3d_mm"])[::-1][:5]] if where else []
    return rep


def parity(args):
    import pickle

    import cv2

    from bench.run import load_method

    base = json.loads(os.path.expandvars(args.base_args))
    tests = json.loads(os.path.expandvars(args.args))
    tests = tests if isinstance(tests, list) else [tests]
    names = args.names.split(",") if args.names else [f"test{i}" for i in range(len(tests))]
    files = sorted(glob.glob(os.path.join(os.path.expanduser(args.frames_dir), "*.jpg")))[::args.every]
    runs = {}
    for tag, extra in [("base", {})] + list(zip(names, tests)):
        cache = os.path.join(args.cache, f"{tag}.pkl") if args.cache else ""
        if cache and os.path.exists(cache):  # predictions of an earlier run with this tag
            runs[tag] = pickle.load(open(cache, "rb"))
            continue
        m = load_method(args.method, {**base, **extra})
        last, res = None, {}
        for f in files:
            stream = os.path.basename(f).rsplit("_", 1)[0]
            img = cv2.imread(f)
            if stream != last:
                m.reset()
                last = stream
                h, w = img.shape[:2]
                K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], np.float32)
            res[f] = [{k: np.asarray(h[k]) for k in ("is_right", "bbox", "joints3d", "joints2d", "verts3d", "surface2d_head", "tip_ext", "score")
                       if h.get(k) is not None}
                      for h in m(img, K)]
        runs[tag] = res
        if cache:
            os.makedirs(args.cache, exist_ok=True)
            pickle.dump(res, open(cache, "wb"))
        del m
    reps = {n: compare(runs["base"], runs[n]) for n in names}
    print(json.dumps(reps, indent=1))
    if args.out:
        json.dump(dict(reps=reps, base_args=base, args=dict(zip(names, tests))), open(args.out, "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("pose")
    a.add_argument("--finetune_ckpt", required=True)
    a.add_argument("--detector_weights", required=True, help="any YOLO weights (the method loads a detector)")
    a.add_argument("--sam3d", default=DEFAULT_SAM3D, help="SAM 3D Body config dir (fah.download_sam3d)")
    a.add_argument("--heads", action="store_true", help="whole decoder-free pose network, not only the encoder")
    a.add_argument("--no_fold", action="store_true", help="keep the adapter / ray-embedding linear layers separate")
    a.add_argument("--no_parts", action="store_true", help="without the part head (no part_logits): the part-maps-off runtime")
    a.add_argument("--precision", choices=("float32", "float16"), default="float32", help="pose compute precision; float32 avoids nonfinite fp16 outputs on recent toolchains")
    a.add_argument("--batch", type=int, nargs="+", default=[1, 2])
    a.add_argument("--out", required=True)
    a = sub.add_parser("det")
    a.add_argument("--weights", required=True)
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--frames", default="1408x1408", help="frame sizes WxH, comma separated (one letterbox shape each)")
    a.add_argument("--out", required=True)
    a = sub.add_parser("parity")
    a.add_argument("--frames_dir", required=True)
    a.add_argument("--method", default="fah_pipeline")
    a.add_argument("--base_args", required=True)
    a.add_argument("--args", required=True, help="json overrides of the tested configuration (or a list of them)")
    a.add_argument("--names", default="", help="comma-separated names of the tested configurations")
    a.add_argument("--cache", default="", help="dir for the predictions per name (reused when present: name base = --base_args)")
    a.add_argument("--focal", type=float, default=610.0)
    a.add_argument("--every", type=int, default=1)
    a.add_argument("--out", default="")
    args = p.parse_args()
    {"pose": build_pose, "det": build_det, "parity": parity}[args.cmd](args)
