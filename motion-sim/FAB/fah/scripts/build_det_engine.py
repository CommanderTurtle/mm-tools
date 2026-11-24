"""TensorRT FP16 engine of a YOLO(-seg) hand detector for fah.detector (det_trt): ONNX export of the fah.yolo network
(no ultralytics package; outputs: the decoded detections and, for segmentation, the mask prototypes) with dynamic input
shapes (the fast path letterboxes to a stride multiple, so the input is (1, 3, <= imgsz, <= imgsz)), FP16 conversion with
fp32 inputs / outputs (modelopt autocast: TensorRT 11 builds strongly typed networks), one optimisation profile up to
imgsz x imgsz (optimum at the full square). Engines are specific to the GPU model and the TensorRT version: build on the
target GPU. Needs tensorrt + nvidia-modelopt[onnx] (a separate venv: modelopt pulls its own torch; the ONNX export runs
on the CPU).

    python scripts/build_det_engine.py --weights $FAH_WEIGHTS/shared/detector.pt --imgsz 640 \
        --out $FAH_WEIGHTS/trt/detector-640-fp16.engine
"""
import argparse
import os
import shutil
import sys
import tempfile

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(args):
    import onnx
    import tensorrt as trt
    import torch
    from modelopt.onnx.autocast import convert_to_mixed_precision

    from fah.yolo import load_yolo

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="det_engine_")
    net = load_yolo(args.weights)
    net.head.export = True  # anchors from the input shape inside the graph (dynamic shapes)
    onnx_path = os.path.join(tmp, "model.onnx")
    outs = ["output0", "output1"] if net.seg else ["output0"]
    with torch.no_grad():
        torch.onnx.export(net, torch.zeros(1, 3, args.imgsz, args.imgsz), onnx_path, input_names=["images"], output_names=outs,
                          dynamic_axes={"images": {2: "height", 3: "width"}, "output0": {2: "anchors"},
                                        **({"output1": {2: "ph", 3: "pw"}} if net.seg else {})}, opset_version=17, dynamo=False)
    x = np.random.default_rng(0).random((1, 3, args.imgsz, args.imgsz), np.float32)
    if args.calib_image:  # value ranges for the fp16 / fp32 node split from a real frame (letterboxed square, RGB, [0, 1])
        img = cv2.imread(args.calib_image)
        h, w = img.shape[:2]
        r = args.imgsz / max(h, w)
        img = cv2.resize(img, (round(w * r), round(h * r)))
        x = np.full((args.imgsz, args.imgsz, 3), 114, np.uint8)
        x[:img.shape[0], :img.shape[1]] = img
        x = (x[:, :, ::-1].transpose(2, 0, 1)[None] / 255.0).astype(np.float32)
    calib = os.path.join(tmp, "calib.npz")
    np.savez(calib, **{onnx.load(onnx_path).graph.input[0].name: x})
    m16 = convert_to_mixed_precision(onnx_path, low_precision_type="fp16", keep_io_types=True, calibration_data=calib)
    onnx16 = os.path.join(tmp, "model_fp16.onnx")
    onnx.save(m16, onnx16)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, logger)
    assert parser.parse(open(onnx16, "rb").read()), [parser.get_error(i) for i in range(parser.num_errors)]
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    prof = builder.create_optimization_profile()
    name = net.get_input(0).name
    s = args.imgsz
    prof.set_shape(name, (1, 3, 32, 32), (1, 3, s, s), (1, 3, s, s))
    cfg.add_optimization_profile(prof)
    open(args.out, "wb").write(builder.build_serialized_network(net, cfg))
    shutil.rmtree(tmp, ignore_errors=True)
    print("engine ->", args.out, os.path.getsize(args.out) / 1e6, "MB")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", required=True)
    p.add_argument("--calib_image", default="", help="a typical frame for the fp16 value-range check (default: noise)")
    main(p.parse_args())
