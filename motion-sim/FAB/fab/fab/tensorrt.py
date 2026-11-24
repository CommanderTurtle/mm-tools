"""Optional, fixed-shape TensorRT backbone for inference-only validation and deployment."""

import json
import time
from pathlib import Path

import torch


class TensorRTBackbone(torch.nn.Module):
    def __init__(self, path, checkpoint):
        super().__init__()
        import tensorrt as trt

        path, checkpoint = Path(path), Path(checkpoint).resolve()
        metadata = json.loads(Path(str(path) + ".json").read_text())
        identity = metadata["checkpoint_identity"]
        stat = checkpoint.stat()
        if not metadata["passed"] or identity != {
            "path": str(checkpoint), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns
        }:
            raise ValueError("TensorRT engine was not verified for this checkpoint; rebuild it")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine deserialization failed")
        for name in ("x", "emb"):
            if (self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR):
                raise ValueError("This adapter requires contiguous FP32 input/output bindings")
        self.shape = tuple(self.engine.get_tensor_shape("x"))
        out_shape = tuple(self.engine.get_tensor_shape("emb"))
        if min(self.shape + out_shape) <= 0:
            raise ValueError("Only fixed-shape engines are supported")
        self.context = self.engine.create_execution_context()
        self.register_buffer("output", torch.empty(out_shape, device="cuda"), persistent=False)
        self.eval()

    def forward(self, x, extra_embed=None):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("TensorRTBackbone requires eval and inference/no-grad mode")
        if tuple(x.shape) != self.shape or x.dtype != torch.float32 or x.device != self.output.device:
            raise ValueError(f"Expected CUDA FP32 input {self.shape} on {self.output.device}; got "
                             f"{tuple(x.shape)} {x.dtype} on {x.device}")
        x = x.contiguous()
        self.context.set_tensor_address("x", x.data_ptr())
        self.context.set_tensor_address("emb", self.output.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(x.device).cuda_stream):
            raise RuntimeError("TensorRT backbone execution failed")
        return self.output


class TensorRTBody(torch.nn.Module):
    """Fixed one-person body graph. Returned buffers are reused by the next call; copy retained outputs."""

    input_names = ("img", "bbox_center", "bbox_scale", "ori_size", "cam_int", "affine")

    def __init__(self, path, checkpoint):
        super().__init__()
        import tensorrt as trt

        path, checkpoint = Path(path), Path(checkpoint).resolve()
        metadata = json.loads(Path(str(path) + ".json").read_text())
        stat = checkpoint.stat()
        if not metadata["passed"] or metadata["checkpoint_identity"] != {
            "path": str(checkpoint), "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns
        }:
            raise ValueError("Full-body engine was not verified for this checkpoint")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Full-body engine deserialization failed")
        self.context = self.engine.create_execution_context()
        self.shapes, self.output_names = {}, []
        actual_inputs = set()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if (self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT
                    or self.engine.get_tensor_format(name) != trt.TensorFormat.LINEAR or min(shape) <= 0):
                raise ValueError(f"Expected fixed contiguous FP32 binding: {name}")
            self.shapes[name] = shape
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                actual_inputs.add(name)
            else:
                self.output_names.append(name)
                self.register_buffer(name, torch.empty(shape, device="cuda"), persistent=False)
        if actual_inputs != set(self.input_names) or "pred_vertices" not in self.output_names:
            raise ValueError("Unexpected full-body engine interface")
        self.eval()

    def forward(self, img, bbox_center, bbox_scale, ori_size, cam_int, *, affine, return_vertices=True):
        if self.training or torch.is_grad_enabled():
            raise RuntimeError("TensorRTBody requires eval and inference/no-grad mode")
        # Retain contiguous inputs until the asynchronous enqueue has consumed their addresses.
        inputs = [x.contiguous() for x in (img, bbox_center, bbox_scale, ori_size, cam_int, affine)]
        device = self.pred_vertices.device
        for name, value in zip(self.input_names, inputs):
            if tuple(value.shape) != self.shapes[name] or value.dtype != torch.float32 or value.device != device:
                raise ValueError(f"Invalid {name} binding: expected CUDA FP32 {self.shapes[name]} on {device}")
            self.context.set_tensor_address(name, value.data_ptr())
        outputs = {name: getattr(self, name) for name in self.output_names}
        for name, value in outputs.items():
            self.context.set_tensor_address(name, value.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(device).cuda_stream):
            raise RuntimeError("Full-body TensorRT execution failed")
        return outputs


class TensorRTBundle:
    """A deployable checkpoint's whole one-person body forward as one TensorRT engine (fab-build-trt): inputs img
    (1, 3, 512, 512) in [0, 1], bbox_center, bbox_scale, ori_size, cam_int, affine (FP32, CUDA); outputs = the forward's dict keys of the export (mesh, keypoints, camera, dense points,
    visibility / contact logits, G1). The engine's <engine>.json must have passed and name this checkpoint (path, bytes,
    mtime). Called with n people, the engine runs once per person; returned tensors are reused by the next call."""

    inputs = ("img", "bbox_center", "bbox_scale", "ori_size", "cam_int", "affine")

    def __init__(self, path, checkpoint, allow_unverified=False):
        import tensorrt as trt

        path, checkpoint = Path(path), Path(checkpoint).resolve()
        self.meta = json.loads(Path(str(path) + ".json").read_text())
        ident, st = self.meta["checkpoint_identity"], checkpoint.stat()  # path / bytes / mtime_ns (as TensorRTBody)
        same = (ident["path"] == str(checkpoint) and ident["bytes"] == st.st_size and ident["mtime_ns"] == st.st_mtime_ns)
        if not allow_unverified and not (self.meta.get("passed") and same):
            raise ValueError(f"{path}: not a verified engine of {checkpoint}; rebuild it with fab-build-trt")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"{path}: deserialization failed (built for another GPU / TensorRT version?)")
        self.context = self.engine.create_execution_context()
        self.shapes, self.outputs, self.buf = {}, [], {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_dtype(name) != trt.DataType.FLOAT or min(shape) <= 0:
                raise ValueError(f"expected a fixed-shape FP32 binding: {name}")
            self.shapes[name] = shape
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                self.outputs.append(name)
                self.buf[name] = torch.empty(shape, device="cuda")
                self.context.set_tensor_address(name, self.buf[name].data_ptr())
        if not set(self.inputs) <= set(self.shapes) or "pred_vertices" not in self.outputs:
            raise ValueError(f"{path}: unexpected engine interface {sorted(self.shapes)}")
        self.static = {k: torch.zeros(self.shapes[k], device="cuda") for k in self.inputs}
        for k, t in self.static.items():
            self.context.set_tensor_address(k, t.data_ptr())

    def _one(self, x):
        for k, t in zip(self.inputs, x):
            self.static[k].copy_(t.reshape(self.shapes[k]))
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT body execution failed")
        return self.buf

    @torch.no_grad()
    def __call__(self, img, bbox_center, bbox_scale, ori_size, cam_int, affine):
        x = (img, bbox_center, bbox_scale, ori_size, cam_int, affine)
        n = len(img)
        if n == 1:
            return self._one(x)
        outs = [{k: v.clone() for k, v in self._one(tuple(t[i : i + 1] for t in x)).items()} for i in range(n)]
        return {k: torch.cat([o[k] for o in outs]) for k in self.outputs}

    def time(self, x, iters=10):
        """median ms of the engine alone (inputs already on the GPU)."""
        x = tuple(t.cuda() for t in x)
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            self._one(x)
            torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000)
        return float(sorted(ts)[len(ts) // 2])
