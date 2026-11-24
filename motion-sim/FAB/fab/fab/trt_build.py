"""TensorRT engines of a FAB model for THIS GPU (the release recipe; run by fab-build-trt / scripts/build_engines.sh).

The model's whole network (backbone, decoder + MHR rig, dense head with its 2.5D lift, visibility / contact / G1 heads,
and the part head when the model has one) as one fixed-shape engine per batch size: the backbone in BF16 and every
other matrix operator in BF16 ("fullbb_all"), the rest FP32 (norms, softmax, RoPE, residuals) in a strongly typed
network. Engines are specific to the GPU type, the TensorRT version and the weights file; they are never distributed.

  info      the engine folder, the weights file, the export flags and the 2.5D mix for a model (for the shell script)
  export    the model in FP32 -> <work>/body.onnx + agreement.json (ONNX Runtime CPU vs PyTorch FP32 on N real crops of
            your images: the released detector's largest person, else a centred box)
  build     <work>/body.onnx -> the engine + <engine>.json (the metadata FABPredictor's TensorRTBody verifies)
  detector  the person detector -> <engines dir>/detector.engine (FP16), kept only if its boxes agree with the PyTorch
            detector on up to 20 of your images (same count, every box matched at IoU >= 0.98)

    python -m fab.trt_build info --model fab-l
    python -m fab.trt_build export --model fab-l --images photos/ --batch 1 --out <work>/body.onnx [--seg]
    python -m fab.trt_build build --onnx <work>/body.onnx --out <engines dir>/fab-l/body-m1.engine
    python -m fab.trt_build detector --images photos/
"""

import argparse
import gc
import json
import os
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fab import paths  # noqa: E402

OUTPUTS = ("pred_keypoints_3d", "pred_keypoints_2d", "pred_cam_t", "pred_vertices", "dense_uv", "dense_xyz",
           "vis_logits", "contact_logits", "g1_q", "g1_rot6d")
NAMES_IN = ("img", "bbox_center", "bbox_scale", "ori_size", "cam_int", "affine")
IMAGES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MODELS = ("fab-l", "fab-s")  # the bundles with a dense head (FAB-H runs on the PyTorch path)


def model_path(name):
    return Path(paths.model_file(name)).resolve()


def mix_value(mix):
    return mix if mix == "auto" else float(mix)


def images(folder, n=None):
    files = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGES) if folder else []
    if not files:
        raise SystemExit(f"no images in {folder!r}: pass --images DIR with a few photos of people")
    return files if n is None else [files[i % len(files)] for i in range(n)]


def person_box(img_bgr, detector):
    """The released detector's largest person (conf 0.25, as FABPredictor), else a centred box of 80 % of the image."""
    H, W = img_bgr.shape[:2]
    if detector is not None:
        b = detector.predict(np.ascontiguousarray(img_bgr), classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
        xyxy = b.boxes.xyxy.float().cpu().numpy()
        if len(xyxy):
            return xyxy[np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1]))].astype(np.float32)
    return np.array([0.1 * W, 0.1 * H, 0.9 * W, 0.9 * H], np.float32)


def crops(fab, files):
    """One real crop per image: the predictor's own inputs (CPU tensors), with the default intrinsics (the image
    diagonal as the focal length, FABPredictor's default)."""
    from fab.data.utils.prepare_batch import prepare_batch

    os.environ.setdefault("YOLO_AUTOINSTALL", "False")
    try:
        from ultralytics import YOLO

        det = YOLO(str(paths.shared("detector.pt"))) if paths.shared("detector.pt").exists() else None
    except ImportError:
        det = None
    out = []
    for f in files:
        bgr = cv2.imread(str(f))
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        fl = float(np.hypot(H, W))
        K = np.array([[fl, 0, W / 2], [0, fl, H / 2], [0, 0, 1]], np.float32)
        box = person_box(bgr, det)
        b = prepare_batch(img, fab.transform, box[None], None, None, cam_int=torch.from_numpy(K).float()[None],
                          include_masks=False)
        out.append(tuple(t.float().cpu() for t in (b["img"][0], b["bbox_center"][0], b["bbox_scale"][0],
                                                   b["ori_img_size"][0], b["cam_int"], b["affine_trans"][0])))
    return out


class Wrapper(torch.nn.Module):
    def __init__(self, model, names):
        super().__init__()
        self.model, self.names = model, names

    def forward(self, img, bbox_center, bbox_scale, ori_size, cam_int, affine):
        out = self.model(img, bbox_center, bbox_scale, ori_size, cam_int, affine=affine)
        return tuple(out[k].float() for k in self.names)


def _hypot_translation():
    """prims.hypot (the rig's euler angles) has no ONNX op / torchlib entry: sqrt(a * a + b * b)."""
    import onnxscript
    from onnxscript import FLOAT
    from onnxscript import opset18 as op

    globals().update(FLOAT=FLOAT, op=op)  # onnxscript resolves the annotations and ops in the module globals

    @onnxscript.script()
    def hypot(a: FLOAT, b: FLOAT) -> FLOAT:
        return op.Sqrt(op.Add(op.Mul(a, a), op.Mul(b, b)))

    return {torch.ops.prims.hypot.default: hypot}


def has_seg_head(model):
    dh = model.dense_head
    return dh is not None and (getattr(dh, "part_emb", None) is not None or getattr(dh, "mask_dconv", None) is not None)


def info(a):
    from fab.predictor import FABPredictor

    ck = model_path(a.model)
    fab = FABPredictor(str(ck), dense_25d_mix=mix_value(a.mix), smpl_fitter="", detector="", device="cpu", fast=False)
    dh = fab.model.dense_head
    mix = float(getattr(dh, "p25d_mix", 0.0)) if dh is not None else 0.0
    print(json.dumps({"engines_dir": str(paths.engines_dir(0) / ck.parent.name), "weights": str(ck),
                      "seg": "--seg" if has_seg_head(fab.model) else "", "mix": f"{mix:g}"}))


def export(a):
    from fab.predictor import FABPredictor

    ck = model_path(a.model)
    fab = FABPredictor(str(ck), dense_25d_mix=mix_value(a.mix), smpl_fitter="", detector="", device="cpu", fast=False,
                       parts=a.seg, mask=a.seg)
    if a.seg and fab.mask_extent:
        raise ValueError("a mask head that reads the prompt box is not in the fixed engine interface")
    model = fab.model.eval().float()
    model.body.inputs_in_01 = True
    model.map_outputs = False
    model.body.head_pose.mhr = None  # the TorchScript rig is not exportable; the eager copy is used
    if hasattr(model.body, "head_pose_hand"):
        model.body.head_pose_hand.mhr = None
    files = images(a.images, a.batch)
    xs = crops(fab, files)  # --batch N: a fixed N-person engine from N real crops
    x = tuple(torch.cat([c[i] for c in xs]) for i in range(len(NAMES_IN)))
    if a.ln_nhwc:  # the dense head's channel LayerNorm2d as NHWC LayerNormalization (the same normalization)
        from fab.dense_head import LayerNorm2d

        def _ln_nhwc(self, x):
            return torch.nn.functional.layer_norm(x.permute(0, 2, 3, 1), (x.shape[1],), self.weight, self.bias,
                                                  self.eps).permute(0, 3, 1, 2)

        LayerNorm2d.forward = _ln_nhwc
    with torch.no_grad():
        probe = model(*x[:5], affine=x[5])
    names = [k for k in OUTPUTS + (("part_logits", "mask_logits") if a.seg else ()) if k in probe and probe[k] is not None]
    print("outputs", {k: tuple(probe[k].shape) for k in names}, flush=True)
    wrapped = Wrapper(model, names).eval()
    with torch.no_grad():
        ref = wrapped(*x)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    torch.onnx.export(wrapped, x, a.out, input_names=list(NAMES_IN), output_names=names, opset_version=18,
                      dynamo=True, external_data=True, custom_translation_table=_hypot_translation())
    print(f"exported {a.out} in {time.time() - t0:.0f} s", flush=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    outs = sess.run(None, {k: v.numpy() for k, v in zip(NAMES_IN, x)})
    diffs, ok = {}, True
    for name, o, r in zip(names, outs, ref):
        diffs[name] = float(np.abs(o - r.numpy()).max())
        tol = 0.05 if name in ("pred_keypoints_2d", "dense_uv") else (1e-2 if name.endswith("_logits") else 1e-3)
        ok &= bool(np.isfinite(o).all()) and diffs[name] <= (tol if name != "dense_uv" else 1e-3)
        print(f"  {name} {tuple(o.shape)} max |onnx - torch| {diffs[name]:.2e}", flush=True)
    st = ck.stat()
    rep = {"checkpoint": str(ck), "passed": ok, "outputs": names, "max_abs_errors": diffs, "output_mode": "deploy",
           "batch": a.batch, "ln_nhwc": bool(a.ln_nhwc), "dense_25d_mix": float(getattr(model.dense_head, "p25d_mix", 0.0)),
           "checkpoint_identity": {"path": str(ck), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns},
           "scope": f"{len(files)} real crops of {a.images}; ONNX Runtime CPU vs PyTorch FP32, full body graph"}
    Path(a.out).with_name("agreement.json").write_text(json.dumps(rep, indent=2) + "\n")
    print("EXPORT", "OK" if ok else "FAILED", flush=True)
    if not ok:
        raise SystemExit(1)


KEEP32 = {"LayerNormalization", "Resize", "Cast", "CastLike", "Constant", "ConstantOfShape", "Range", "Sin", "Cos", "Shape",
          "Size", "RandomNormal", "RandomUniform", "EyeLike"}


def to_half(graph, dtype, namespace, sub=None):
    """The whole subgraph in namespace computed in dtype (not only its matrix operators: the attention's
    MatMul -> Softmax -> MatMul then stays in one precision, which TensorRT can fuse), except KEEP32 (LayerNorm,
    RoPE trig, Resize, casts / constants / shapes). Every float output of a converted node gets a dtype copy
    (<name>__h) and keeps its FP32 name through a Cast, so FP32 consumers are unchanged; converted consumers read
    the copy, or a Cast of an FP32 input (initializers included; folded by TensorRT). sub: only nodes whose scope also
    contains one of these."""
    import onnx

    to = {"fp16": onnx.TensorProto.FLOAT16, "bf16": onnx.TensorProto.BFLOAT16}[dtype]
    inferred = onnx.shape_inference.infer_shapes(graph)
    elem = {v.name: v.type.tensor_type.elem_type for v in [*inferred.graph.value_info, *inferred.graph.input,
                                                            *inferred.graph.output]}
    elem.update({t.name: t.data_type for t in graph.graph.initializer})
    FLOAT = onnx.TensorProto.FLOAT
    half, nodes, casts, n = {}, [], {}, 0
    for node in graph.graph.node:
        scopes = next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.name_scopes"), "")
        outs_float = [o for o in node.output if o and elem.get(o) == FLOAT]
        if (repr(namespace) not in scopes or node.op_type in KEEP32 or node.domain not in ("", "ai.onnx")
                or not outs_float or (sub is not None and not any(x in scopes for x in sub))):
            nodes.append(node)
            continue
        for i, name in enumerate(node.input):
            if not name or elem.get(name) != FLOAT:
                continue
            if name in half:
                node.input[i] = half[name]
                continue
            if name not in casts:
                casts[name] = f"{name}__in{dtype}"
                nodes.append(onnx.helper.make_node("Cast", [name], [casts[name]], to=to))
            node.input[i] = casts[name]
        back = []
        for j, o in enumerate(node.output):
            if o in outs_float:
                h = f"{o}__h"
                node.output[j] = h
                half[o] = h
                back.append(onnx.helper.make_node("Cast", [h], [o], to=FLOAT))
        nodes.append(node)
        nodes.extend(back)
        n += 1
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return n  # value_info stays valid: every original name keeps its FP32 type (the __h copies have none)


def mixed_outside(graph, dtype, namespace, extra=()):
    """The matrix operators OUTSIDE namespace (the decoder / heads next to a to_half backbone) computed in dtype from
    explicit casts, outputs cast back to FP32; extra: more op types treated as matrix operators."""
    import onnx

    to = {"fp16": onnx.TensorProto.FLOAT16, "bf16": onnx.TensorProto.BFLOAT16}[dtype]
    nodes, changed = [], 0
    for node in graph.graph.node:
        scopes = next((p.value for p in node.metadata_props if p.key == "pkg.torch.onnx.name_scopes"), "")
        if node.op_type not in {"Conv", "Gemm", "MatMul", *extra} or node.domain not in ("", "ai.onnx") or repr(namespace) in scopes:
            nodes.append(node)
            continue
        for i, name in enumerate(node.input):
            if not name:
                continue
            cast = f"{name}__o{dtype}_{changed}_{i}"
            nodes.append(onnx.helper.make_node("Cast", [name], [cast], to=to))
            node.input[i] = cast
        output = node.output[0]
        node.output[0] = f"{output}__o{dtype}_{changed}"
        nodes.append(node)
        nodes.append(onnx.helper.make_node("Cast", [node.output[0]], [output], to=onnx.TensorProto.FLOAT))
        changed += 1
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return changed


def lower_antialiased_resize(graph):
    """Exact fixed-shape bilinear antialiasing via separable FP32 matrices.

    TensorRT does not implement ONNX Resize(antialias=1). Derive the linear filter from PyTorch
    using impulse columns, and verify it before replacing the operator. Unsupported modes fail closed.
    """
    import onnx

    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]}
    nodes, reports = [], []
    rng = torch.Generator().manual_seed(17)
    for node in graph.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type != "Resize" or not attrs.get("antialias", 0):
            nodes.append(node)
            continue
        source, target = shapes.get(node.input[0], ()), shapes.get(node.output[0], ())
        if (len(source) != 4 or len(target) != 4 or min(source + target) <= 0
                or source[:2] != target[:2] or attrs.get("mode") != b"linear"
                or attrs.get("coordinate_transformation_mode") != b"half_pixel"
                or max(source[-2:]) > 2048):
            raise ValueError(f"Unsupported antialiased resize: {node.name}: {source} -> {target}")
        h, w = source[-2:]
        oh, ow = target[-2:]
        interpolate = torch.nn.functional.interpolate
        wy = interpolate(torch.eye(h)[None, None], size=(oh, h), mode="bilinear",
                         align_corners=False, antialias=True)[0, 0]
        wx = interpolate(torch.eye(w)[None, None], size=(w, ow), mode="bilinear",
                         align_corners=False, antialias=True)[0, 0]
        sample = torch.randn(source, generator=rng)
        expected = interpolate(sample, size=(oh, ow), mode="bilinear", align_corners=False, antialias=True)
        error = float((wy @ sample @ wx - expected).abs().max())
        if error > 2e-6:
            raise ValueError(f"Resize linear-filter regression failed: {node.name}: {error}")
        prefix = node.name + "_linear_filter"
        graph.graph.initializer.extend([
            onnx.numpy_helper.from_array(wy.numpy(), prefix + "_y"),
            onnx.numpy_helper.from_array(wx.numpy(), prefix + "_x"),
        ])
        nodes.extend([
            onnx.helper.make_node("MatMul", [prefix + "_y", node.input[0]], [prefix + "_vertical"]),
            onnx.helper.make_node("MatMul", [prefix + "_vertical", prefix + "_x"], [node.output[0]]),
        ])
        reports.append({"node": node.name, "input": source, "output": target, "max_abs_error": error})
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return reports


def lower_static_scatter_add(graph, base_dir):
    """Lower fixed skinning scatter-add to padded Gather + ReduceSum, retaining duplicate contributions."""
    import onnx

    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim)
              for v in [*graph.graph.input, *graph.graph.output, *graph.graph.value_info]}
    initializers = {v.name: v for v in graph.graph.initializer}
    producers = {name: n for n in graph.graph.node for name in n.output}
    nodes, reports = [], []
    rng = np.random.default_rng(17)
    for node in graph.graph.node:
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if node.op_type != "ScatterND" or attrs.get("reduction", b"none") == b"none":
            nodes.append(node)
            continue
        data, index_name, updates = node.input
        shape, update_shape = shapes.get(data, ()), shapes.get(updates, ())
        if (attrs.get("reduction") != b"add"
                or not shape or min(shape + update_shape) <= 0 or shape[1:] != update_shape[1:]):
            raise ValueError(f"Unsupported scatter reduction: {node.name}")

        def constant(name):
            return onnx.numpy_helper.to_array(initializers[name], base_dir=str(base_dir))

        if index_name in initializers:
            indices = constant(index_name)
        else:
            producer = producers.get(index_name)
            if (producer is None or producer.op_type != "Unsqueeze" or len(producer.input) != 2
                    or not all(x in initializers for x in producer.input)):
                raise ValueError(f"Scatter indices are not a supported constant: {node.name}")
            indices = np.expand_dims(constant(producer.input[0]), tuple(constant(producer.input[1]).tolist()))
        if indices.shape != (update_shape[0], 1):
            raise ValueError(f"Only static first-axis scatter indices are supported: {node.name}")
        indices = indices[:, 0].copy()
        indices[indices < 0] += shape[0]
        if ((indices < 0) | (indices >= shape[0])).any():
            raise ValueError(f"Invalid scatter indices: {node.name}")
        counts = np.bincount(indices, minlength=shape[0])
        fanin = int(counts.max())
        if fanin > 64 or shape[0] * fanin > 2_000_000:
            raise ValueError(f"Scatter lowering would be too large: {node.name}")
        table = np.full((shape[0], fanin), len(indices), np.int64)
        cursor = np.zeros(shape[0], np.int64)
        for i, dest in enumerate(indices):
            table[dest, cursor[dest]] = i
            cursor[dest] += 1
        base = rng.standard_normal(shape).astype(np.float32)
        values = rng.standard_normal(update_shape).astype(np.float32)
        expected = base.copy()
        np.add.at(expected, indices, values)
        zeros = np.zeros((1,) + update_shape[1:], np.float32)
        actual = base + np.concatenate([values, zeros])[table].sum(axis=1)
        error = float(np.abs(actual - expected).max())
        if error > 1e-5:
            raise ValueError(f"Scatter-add regression failed: {node.name}: {error}")
        prefix = node.name + "_gather_sum"
        graph.graph.initializer.extend([
            onnx.numpy_helper.from_array(table, prefix + "_indices"),
            onnx.numpy_helper.from_array(zeros, prefix + "_zero"),
            onnx.numpy_helper.from_array(np.array([1], np.int64), prefix + "_axis"),
        ])
        nodes.extend([
            onnx.helper.make_node("Concat", [updates, prefix + "_zero"], [prefix + "_padded"], axis=0),
            onnx.helper.make_node("Gather", [prefix + "_padded", prefix + "_indices"],
                                  [prefix + "_gathered"], axis=0),
            onnx.helper.make_node("ReduceSum", [prefix + "_gathered", prefix + "_axis"],
                                  [prefix + "_sum"], keepdims=0),
            onnx.helper.make_node("Add", [data, prefix + "_sum"], [node.output[0]]),
        ])
        reports.append({"node": node.name, "rows": shape[0], "updates": len(indices),
                        "max_fanin": fanin, "max_abs_error": error})
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    return reports


def build(a):
    import onnx
    import tensorrt as trt

    src = Path(a.onnx).resolve()
    rep = json.loads(src.with_name("agreement.json").read_text())
    if not rep["passed"]:
        raise ValueError("the FP32 export did not pass its agreement check")
    tag = f"{a.prec}-{a.scope}" + ("-ct" if a.convT else "") + (f"-o{a.opt_level}" if a.opt_level >= 0 else "")
    graph = onnx.load(str(src), load_external_data=False)
    sub = None  # fullbb / fullbb_all: the whole backbone in a.prec (+ the matrix operators elsewhere for _all)
    changed = to_half(graph, a.prec, "model.body.backbone", sub)
    if a.scope.endswith("_all"):
        changed += mixed_outside(graph, a.prec, "model.body.backbone", ("ConvTranspose",) if a.convT else ())
    resize = lower_antialiased_resize(graph)
    scatter = lower_static_scatter_add(graph, src.parent)
    mp = src.with_name(f"body-{tag}.onnx")
    onnx.save(graph, str(mp))
    print(f"{tag}: {changed} operators in {a.prec}; resizes {len(resize)}, scatters {len(scatter)}", flush=True)
    del graph
    gc.collect()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(mp)):
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if a.opt_level >= 0:  # TensorRT builder optimization level (default 3)
        config.builder_optimization_level = a.opt_level
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("TensorRT build failed")
    print(f"{tag}: built in {time.time() - t0:.0f} s, TensorRT {trt.__version__}", flush=True)
    dst = Path(a.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(bytes(ser))
    meta = {"passed": True, "checkpoint_identity": rep["checkpoint_identity"], "source_onnx": str(src),
            "dense_25d_mix": rep.get("dense_25d_mix"), "precision": a.prec, "scope": a.scope, "opt_level": a.opt_level,
            "convT": bool(a.convT), "matrix_operators": changed, "tensorrt": trt.__version__, "batch": rep.get("batch", 1),
            "gpu": torch.cuda.get_device_name(), "outputs": rep["outputs"],
            "note": "passed = built from a passing FP32 export"}
    Path(str(dst) + ".json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"ENGINE {dst}", flush=True)


def detector(a):
    """The person detector as an FP16 TensorRT engine (Ultralytics export), kept only if it agrees with the PyTorch
    detector on up to 20 of your images."""
    os.environ["YOLO_AUTOINSTALL"] = "False"  # the export must not install packages on its own
    from ultralytics import YOLO
    from ultralytics.cfg import DEFAULT_CFG_DICT

    from bench.detected_boxes import associate

    dest = paths.engines_dir(0) / "detector.engine"
    work = dest.parent / "detector-work"
    work.mkdir(parents=True, exist_ok=True)
    ck = work / "detector.pt"
    shutil.copyfile(paths.require(paths.shared("detector.pt"), "the person detector"), ck)
    precision = {"quantize": 16} if "quantize" in DEFAULT_CFG_DICT else {"half": True}
    engine = YOLO(str(ck)).export(format="engine", imgsz=640, batch=1, dynamic=True, nms=False, device=0, workspace=4,
                                  **precision)
    ref, cand = YOLO(str(ck)), YOLO(str(engine), task="detect")
    files = images(a.images)
    files = [files[i] for i in np.linspace(0, len(files) - 1, min(20, len(files)), dtype=int)]
    ok = True
    for f in files:
        im = cv2.imread(str(f))
        x = ref.predict(im, classes=[0], imgsz=640, verbose=False, **precision)[0].boxes.xyxy.cpu().numpy()
        y = cand.predict(im, classes=[0], imgsz=640, verbose=False, **precision)[0].boxes.xyxy.cpu().numpy()
        m, _ = associate(x, y, min_iou=.98)
        ok &= len(x) == len(y) == int((m >= 0).sum())
    print(f"detector agreement on {len(files)} images: {ok}", flush=True)
    if ok:
        import tensorrt as trt

        shutil.copyfile(engine, dest)
        # the build record FABPredictor checks before using the engine (fab/predictor.detector_engine_unusable)
        Path(str(dest) + ".json").write_text(json.dumps({"tensorrt": trt.__version__, "imgsz": 640, "dynamic": True, **precision}))
        print(f"ENGINE {dest}", flush=True)
    else:
        raise SystemExit("detector engine disagrees with the PyTorch detector; not installed (the PyTorch detector stays)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("info", "export"):
        s = sub.add_parser(name)
        s.add_argument("--model", default="fab-l", choices=MODELS)
        s.add_argument("--mix", default="1", help="the 2.5D mix baked into the engine (the released setting: 1)")
    e = sub.choices["export"]
    e.add_argument("--images", required=True, help="a folder of images with people (real crops for the agreement check)")
    e.add_argument("--out", required=True, help="<work dir>/body.onnx")
    e.add_argument("--batch", type=int, default=1, help="people per engine call (a fixed-shape batch-N engine)")
    e.add_argument("--seg", action="store_true", help="+ the model's part / mask heads (part_logits, mask_logits)")
    e.add_argument("--ln_nhwc", action="store_true", help="export the dense head's LayerNorm2d as NHWC LayerNormalization")
    b = sub.add_parser("build")
    b.add_argument("--onnx", required=True)
    b.add_argument("--out", required=True, help="the engine file (its metadata goes to <out>.json)")
    b.add_argument("--prec", choices=("fp16", "bf16"), default="bf16")
    b.add_argument("--scope", choices=("fullbb", "fullbb_all"), default="fullbb_all")
    b.add_argument("--opt_level", type=int, default=-1, help="TensorRT builder optimization level (-1: TensorRT's default)")
    b.add_argument("--convT", action="store_true", help="_all scope: ConvTranspose in --prec too")
    d = sub.add_parser("detector")
    d.add_argument("--images", required=True, help="a folder of images with people (up to 20 are compared)")
    a = p.parse_args()
    {"info": info, "export": export, "build": build, "detector": detector}[a.cmd](a)


if __name__ == "__main__":
    main()
