"""Core ML runtimes (Apple silicon) of the networks exported by scripts/build_coreml.py, the counterparts of the TensorRT
engines (fah.student.TRTEncoder, fah.detector._TRTNet): host numpy arrays in and out (Core ML moves them to the GPU /
Neural Engine itself), so the pre- / post-processing around them stays on the CPU."""
import os

import numpy as np


def compiled_path(path):
    """The package's compiled model, kept next to it (<name>.mlmodelc) and reused across launches. Loading the .mlpackage
    itself recompiles it into a new temporary directory each time, and Core ML's GPU graph cache
    (~/Library/Caches/<app>/com.apple.e5rt.e5bundlecache) keys on that path: every launch added new entries (~2.5 GB for
    FAH-S b1-b4) and recompiled. Written by scripts/build_coreml.py, else here on first use (again after the package is
    rebuilt); None (load the package) if the directory is read-only."""
    import shutil

    import coremltools as ct

    out = os.path.splitext(os.path.abspath(path))[0] + ".mlmodelc"
    src = max((os.path.getmtime(os.path.join(d, f)) for d, _, fs in os.walk(os.path.join(path, "Data")) for f in fs), default=0)
    if os.path.exists(out) and os.path.getmtime(out) < src:  # the package was rebuilt
        shutil.rmtree(out)
    if not os.path.exists(out):
        try:
            ct.utils.compile_model(path, out)
        except OSError:  # read-only weights dir: load the package (temporary compile per launch)
            return None
    return out


class CoreMLNet:
    """A multifunction mlprogram (functions = batch sizes "b<B>" or letterbox shapes "s<H>x<W>"), each function loaded on
    first use. units: coremltools ComputeUnit name (ALL, CPU_AND_GPU, CPU_AND_NE, CPU_ONLY)."""

    def __init__(self, path, units="ALL"):
        import coremltools as ct

        self.path, self.units, self.fns = path, getattr(ct.ComputeUnit, units), {}
        spec = ct.models.MLModel(path, skip_model_load=True).get_spec()
        self.functions = [f.name for f in spec.description.functions] or ["main"]
        fs = list(spec.description.functions) or [spec.description]
        d = fs[0]
        self.inputs, self.outputs = [i.name for i in d.input], [o.name for o in d.output]
        self.shapes = {n: {i.name: tuple(i.type.multiArrayType.shape) for i in f.input} for n, f in zip(self.functions, fs)}
        self.compiled = compiled_path(path)

    def fn(self, name):
        if name not in self.fns:
            import coremltools as ct

            kw = {"function_name": name} if name != "main" else {}
            self.fns[name] = ct.models.CompiledMLModel(self.compiled, compute_units=self.units, **kw) if self.compiled else \
                ct.models.MLModel(self.path, compute_units=self.units, **kw)
        return self.fns[name]

    def run(self, name, **inputs):
        out = self.fn(name).predict({k: np.ascontiguousarray(v, np.float32) for k, v in inputs.items()})
        return {k: np.asarray(v, np.float32) for k, v in out.items()}


class CoreMLPose(CoreMLNet):
    """Student encoder (output "feat") or whole decoder-free pose network (outputs pt_uv, ...) at fixed batch sizes (all
    loaded and warmed up at construction): a batch runs in chunks of the largest exported size that fits."""

    def __init__(self, path, units="CPU_AND_GPU"):
        super().__init__(path, units)
        self.sizes = sorted(int(f[1:]) for f in self.functions)
        self.heads = "feat" not in self.outputs
        for b in self.sizes:  # load (compile) every batch size and run it once now: the first predict of a function costs
            self.run(f"b{b}", **{k: np.zeros(v, np.float32) for k, v in self.shapes[f"b{b}"].items()})  # +50-250 ms

    def __call__(self, **inputs):
        B = len(next(iter(inputs.values())))
        outs, i = [], 0
        while i < B:
            b = max([s for s in self.sizes if s <= B - i] or [self.sizes[0]])
            chunk = {k: v[i:i + b] for k, v in inputs.items()}
            n = len(next(iter(chunk.values())))
            if n < b:  # smaller than every exported size: zero-padded
                chunk = {k: np.concatenate([v, np.zeros((b - n, *v.shape[1:]), v.dtype)]) for k, v in chunk.items()}
            o = self.run(f"b{b}", **chunk)
            outs.append({k: v[:n] for k, v in o.items()})
            i += n
        combined = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        for key in ("pt_uv", "pt_xyz", "feat"):
            if key in combined and not np.isfinite(combined[key]).all():
                raise FloatingPointError(f"Core ML returned nonfinite {key}; rebuild with fah-build-coreml --force --pose_precision float32")
        return combined


def fit_letterbox(img, px, py, functions):
    """img (h, w, 3) letterboxed frame (pad px, py) -> (img, px, py) in a built letterbox shape: unchanged when "s<h>x<w>"
    is built, else padded (gray 114, centred) up to the smallest built shape that holds it, as ultralytics' non-auto
    letterbox pads; the network then sees the extra gray rows, so the detections can differ slightly from an export of the
    frame's own shape (about 6 % of the frames changed their hand count on a 1134 x 612 video); None if no built shape holds it."""
    h, w = img.shape[:2]
    if f"s{h}x{w}" in functions:
        return img, px, py
    fit = [(H * W, H, W) for H, W in (map(int, n[1:].split("x")) for n in functions if n.startswith("s") and "x" in n)
           if H >= h and W >= w]
    if not fit:
        return None
    import cv2

    _, H, W = min(fit)
    top, left = (H - h) // 2, (W - w) // 2
    img = cv2.copyMakeBorder(img, top, H - h - top, left, W - w - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, px + left, py + top


class CoreMLDet(CoreMLNet):
    """YOLO(-seg) network, one function per letterbox shape (build the frame size's shape with fah-build-coreml
    --frames): x (1, 3, H, W) RGB [0, 1] -> {"out": (1, 4 + nc (+ 32), N), "proto": (1, 32, H / 4, W / 4)} host torch tensors,
    as the torch network. The raw box distributions are decoded in fp32 with a host copy of the network's own head (head:
    fah.yolo.YOLOHead), with conf only for the anchors whose best class score exceeds it (out then has
    those columns only, in anchor order: the same detections after the caller's threshold)."""

    def __init__(self, path, head, units="ALL"):
        import copy

        super().__init__(path, units)
        self.head = copy.deepcopy(head).float().cpu().eval()
        self.head.stride = head.stride.detach().float().cpu()
        self.shape, self.hinted = None, set()

    def fit(self, img, px, py, frame="<W>x<H>"):
        """The letterboxed frame in a built shape (fit_letterbox), with a one-time rebuild hint per missing shape (frame:
        the source frame size WxH, for the hint)."""
        h, w = img.shape[:2]
        r = fit_letterbox(img, px, py, self.functions)
        if r is None:
            raise KeyError(f"{self.path} has no letterbox shape {h}x{w} or larger (has {self.functions}): "
                           "rebuild with fah-build-coreml --frames <W>x<H>,... (or scripts/build_coreml.py det --frames)")
        if r[0].shape[:2] != (h, w) and (h, w) not in self.hinted:
            import sys

            self.hinted.add((h, w))
            print(f"Core ML detector: no letterbox shape {h}x{w}, padded to {r[0].shape[0]}x{r[0].shape[1]} (detections can "
                  f"differ slightly from an exact export); fah-build-coreml --frames {frame} builds it", file=sys.stderr)
        return r

    def __call__(self, x, conf=None):
        import torch

        from .yolo import make_anchors

        name = f"s{x.shape[2]}x{x.shape[3]}"
        if name not in self.functions:
            raise KeyError(f"{self.path} has no letterbox shape {x.shape[2]}x{x.shape[3]} (has {self.functions}): "
                           "rebuild with fah-build-coreml --frames <W>x<H>,... (or scripts/build_coreml.py det --frames)")
        o = {k: torch.from_numpy(v) for k, v in self.run(name, x=x).items()}
        h = self.head
        if self.shape != x.shape:  # the head's anchors for this input shape
            feats = [torch.zeros(1, 1, x.shape[2] // int(s), x.shape[3] // int(s)) for s in h.stride]
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(feats, h.stride, 0.5))
            self.shape = x.shape
        with torch.no_grad():
            sc = o["scores"].sigmoid()
            sel = torch.nonzero(sc[0].amax(0) > conf, as_tuple=True)[0] if conf is not None else slice(None)
            b = o["boxes"][:, :, sel].contiguous()
            box = h.decode(b, self.anchors[:, sel], self.strides[:, sel]) if b.shape[-1] else b[:, :4]
            out = torch.cat([box, sc[:, :, sel]] + ([o["mc"][:, :, sel]] if "mc" in o else []), 1)
        return {"out": out, "proto": o["proto"]} if "proto" in o else {"out": out}
