"""YOLOv8 hand detection / segmentation network for fah.detector, without the ultralytics package.

load_yolo(path) rebuilds the network from an ultralytics training checkpoint: the checkpoint is read with an unpickler
that never imports ultralytics (its objects come back as plain attribute records), the layer graph (each layer's inputs
`f`, index `i`, the saved outputs) and every tensor are taken from those records, batch norms are folded into their
convolutions (torch.nn.utils.fusion.fuse_conv_bn_eval) and the layers run with this module's own forward passes: the
backbone / neck graph, the box head (distance distributions decoded to boxes on the anchor grid), the class scores, the
mask coefficients and the mask prototypes. Supported layers: Conv, C2f, Bottleneck, SPPF, Concat, Upsample and a
Detect / Segment head (the YOLOv8 detection and segmentation models; other checkpoints raise ValueError).

YOLONet(x) -> out (B, 4 + nc (+ nm), A): boxes (cx, cy, w, h in input pixels), class probabilities (+ mask
coefficients) per anchor, and for segmentation the prototypes (B, nm, H / 4, W / 4) as a second output; raw=True: the
undecoded head outputs (boxes as distance distributions, class logits, mask coefficients, prototypes), as the Core ML
export runs them (the decode then runs on the host: YOLOHead.decode).
"""
import pickle
import types

import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


class _Record:
    """An ultralytics object in a checkpoint: only its pickled attributes."""

    def __setstate__(self, state):
        self.__dict__.update(state if isinstance(state, dict) else {"_state": state})


_RECORDS = {}


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "ultralytics" or module.startswith("ultralytics."):
            return _RECORDS.setdefault((module, name), type(name, (_Record,), {"_source": f"{module}.{name}"}))
        return super().find_class(module, name)


_PICKLE = types.ModuleType("fah_yolo_pickle")
_PICKLE.Unpickler, _PICKLE.load, _PICKLE.Pickler, _PICKLE.dump = _Unpickler, pickle.load, pickle.Pickler, pickle.dump


def read_checkpoint(path):
    """The checkpoint dict with ultralytics objects as attribute records (no ultralytics import)."""
    return torch.load(path, map_location="cpu", weights_only=False, pickle_module=_PICKLE)


def make_anchors(feats, strides, offset=0.5):
    """Anchor centres (A, 2) in grid units and their strides (A, 1) for the feature maps feats (one per level)."""
    centres, steps = [], []
    kw = dict(dtype=feats[0].dtype, device=feats[0].device)
    for f, s in zip(feats, strides):
        h, w = f.shape[-2:]
        col, row = torch.arange(w, **kw) + offset, torch.arange(h, **kw) + offset
        centres.append(torch.stack([col.repeat(h), row.repeat_interleave(w)], 1))  # row-major over the level's cells
        steps.append(torch.full((h * w, 1), float(s), **kw))
    return torch.cat(centres), torch.cat(steps)


def dist2box(dist, anchors):
    """(left, top, right, bottom) distances (B, 4, A) from the anchors (1, 2, A) -> (cx, cy, w, h) (B, 4, A)."""
    near, far = anchors - dist[:, :2], anchors + dist[:, 2:]
    return torch.cat([(near + far) / 2, far - near], 1)


class ConvAct(nn.Module):
    """Convolution with its batch norm folded in, then the activation."""

    def __init__(self, conv, act):
        super().__init__()
        self.conv, self.act = conv, act

    def forward(self, x):
        return self.act(self.conv(x))


class Residual(nn.Module):
    """Two convolutions, plus the input when the block has a shortcut."""

    def __init__(self, first, second, skip):
        super().__init__()
        self.first, self.second, self.skip = first, second, skip

    def forward(self, x):
        out = self.second(self.first(x))
        return out + x if self.skip else out


class SplitChain(nn.Module):
    """Entry conv, its channels in two halves, a chain of residual blocks fed from the second half, then every
    intermediate map concatenated into the exit conv (the checkpoints' C2f)."""

    def __init__(self, entry, exit, blocks):
        super().__init__()
        self.entry, self.exit, self.blocks = entry, exit, blocks

    def forward(self, x):
        t = self.entry(x)
        half = t.shape[1] // 2
        maps = [t[:, :half], t[:, half:]]
        for block in self.blocks:
            maps.append(block(maps[-1]))
        return self.exit(torch.cat(maps, 1))


class PoolPyramid(nn.Module):
    """Entry conv, the same max pool applied `repeats` times in a row, all maps concatenated into the exit conv (the
    checkpoints' SPPF)."""

    def __init__(self, entry, exit, pool, repeats=3, skip=False):
        super().__init__()
        self.entry, self.exit, self.pool, self.repeats, self.skip = entry, exit, pool, repeats, skip

    def forward(self, x):
        maps = [self.entry(x)]
        while len(maps) <= self.repeats:
            maps.append(self.pool(maps[-1]))
        out = self.exit(torch.cat(maps, 1))
        return out + x if self.skip else out


class Join(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)


class MaskBasis(nn.Module):
    """Mask prototypes from the finest feature map: conv, 2x transposed conv, conv, projection to the basis."""

    def __init__(self, reduce, up, refine, project):
        super().__init__()
        self.reduce, self.up, self.refine, self.project = reduce, up, refine, project

    def forward(self, x):
        x = self.up(self.reduce(x))
        return self.project(self.refine(x))


class YOLOHead(nn.Module):
    """Per level: box branch (4 x bins distance logits), class branch, mask-coefficient branch (segmentation); the
    expected distance of each bin distribution (an integral over the bins: a fixed 1x1 conv), boxes on the anchor grid."""

    def __init__(self, box_branch, cls_branch, coef_branch, integral, basis, classes, bins, coefs, stride):
        super().__init__()
        self.box_branch, self.cls_branch, self.coef_branch = box_branch, cls_branch, coef_branch
        self.integral, self.basis = integral, basis
        self.classes, self.bins, self.coefs, self.levels = classes, bins, coefs, len(box_branch)
        self.register_buffer("stride", stride.float(), persistent=False)
        self.export = False  # anchors from the input shape inside the graph (ONNX / TensorRT export)
        self._grid = {}

    def raw(self, feats):
        n = feats[0].shape[0]

        def gather(branch, width):
            return torch.cat([branch[k](feats[k]).reshape(n, width, -1) for k in range(self.levels)], dim=2)

        out = {"boxes": gather(self.box_branch, 4 * self.bins), "scores": gather(self.cls_branch, self.classes)}
        if self.coef_branch is not None:
            out["mask_coefficient"], out["proto"] = gather(self.coef_branch, self.coefs), self.basis(feats[0])
        return out

    def expect(self, boxes):
        """(B, 4 x bins, A) bin logits -> (B, 4, A) expected distances."""
        n, _, a = boxes.shape
        p = boxes.reshape(n, 4, self.bins, a).transpose(1, 2).softmax(dim=1)
        return self.integral(p)[:, 0]

    def decode(self, boxes, anchors, strides):
        """Bin logits (B, 4 x bins, A), anchors (2, A), strides (1, A) -> (cx, cy, w, h) in input pixels (B, 4, A)."""
        return dist2box(self.expect(boxes), anchors[None]) * strides

    def anchors(self, feats):
        if self.export:
            return tuple(t.T for t in make_anchors(feats, self.stride))
        key = tuple(tuple(f.shape[-2:]) for f in feats) + (feats[0].dtype, feats[0].device)
        if key not in self._grid:
            self._grid[key] = tuple(t.T for t in make_anchors(feats, self.stride))
        return self._grid[key]

    def forward(self, feats):
        r = self.raw(feats)
        grid, step = self.anchors(feats)
        parts = [self.decode(r["boxes"], grid, step), r["scores"].sigmoid()]
        if "proto" in r:
            return torch.cat(parts + [r["mask_coefficient"]], 1), r["proto"]
        return torch.cat(parts, 1)


class YOLONet(nn.Module):
    """The layers in order; a layer reads the previous output (f = -1), an earlier layer's output (f = index) or a list
    of them, and the outputs that later layers read (save) are kept."""

    def __init__(self, layers, save, names):
        super().__init__()
        self.layers, self.save, self.names = nn.ModuleList(layers), set(save), dict(names)
        self.seg = self.head.coef_branch is not None

    @property
    def head(self):
        return self.layers[-1]

    @property
    def stride(self):
        return self.head.stride

    def forward(self, x, raw=False):
        kept, last = {}, len(self.layers) - 1
        for k, m in enumerate(self.layers):  # the head by position (module identity does not survive torch.export)
            src = [m.f] if isinstance(m.f, int) else m.f
            ins = [x if j == -1 else kept[j if j >= 0 else m.i + j] for j in src]
            arg = ins[0] if isinstance(m.f, int) else ins
            x = m.raw(arg) if raw and k == last else m(arg)
            if m.i in self.save:
                kept[m.i] = x
        return x


def _float(m):
    return m.float() if isinstance(m, nn.Module) else m


def _conv(rec):
    c = rec._modules
    conv = _float(c["conv"]).eval()
    if "bn" in c:  # fold the batch norm (as at inference)
        conv = fuse_conv_bn_eval(conv, _float(c["bn"]).eval())
    return ConvAct(conv.requires_grad_(False), c["act"])


def _branches(container):
    """A ModuleList of per-level conv stacks from the checkpoint."""
    return nn.ModuleList(nn.Sequential(*[_conv(s) if type(s).__name__ == "Conv" else _float(s) for s in stack._modules.values()])
                         for stack in container._modules.values())


def _layer(rec):
    kind, c, d = type(rec).__name__, getattr(rec, "_modules", {}), rec.__dict__
    if isinstance(rec, nn.Module):
        m = _float(rec)  # a torch layer (nn.Upsample, ...)
    elif kind == "Conv":
        m = _conv(rec)
    elif kind == "C2f":
        m = SplitChain(_conv(c["cv1"]), _conv(c["cv2"]), nn.ModuleList(_layer(b) for b in c["m"]._modules.values()))
    elif kind == "Bottleneck":
        m = Residual(_conv(c["cv1"]), _conv(c["cv2"]), bool(d.get("add", False)))
    elif kind == "SPPF":
        m = PoolPyramid(_conv(c["cv1"]), _conv(c["cv2"]), c["m"], int(d.get("n", 3)), bool(d.get("add", False)))
    elif kind == "Concat":
        m = Join(d.get("d", 1))
    elif kind in ("Detect", "Segment"):
        if d.get("_end2end") or d.get("end2end") is True or c.get("one2one_cv2") is not None:
            raise ValueError("end-to-end (one-to-one) YOLO heads are not supported")
        p = c.get("proto")
        basis = MaskBasis(*(_conv(p._modules[k]) if k != "upsample" else _float(p._modules[k]) for k in ("cv1", "upsample", "cv2", "cv3"))) \
            if p is not None else None
        m = YOLOHead(_branches(c["cv2"]), _branches(c["cv3"]), _branches(c["cv4"]) if kind == "Segment" else None,
                     _float(c["dfl"]._modules["conv"]).requires_grad_(False), basis, int(d["nc"]), int(d["reg_max"]),
                     int(d.get("nm", 0)), torch.as_tensor(d["stride"]))
    else:
        raise ValueError(f"unsupported YOLO layer {kind} ({getattr(rec, '_source', '')})")
    m.f, m.i = d.get("f", -1), d.get("i", 0)
    return m


def load_yolo(path):
    """YOLONet (float32, eval, batch norms folded) of an ultralytics YOLOv8 detection / segmentation checkpoint."""
    ck = read_checkpoint(path)
    rec = ck.get("ema") or ck["model"]
    layers = [_layer(r) for r in rec._modules["model"]._modules.values()]
    if not isinstance(layers[-1], YOLOHead):
        raise ValueError(f"{path}: not a YOLO detection / segmentation model")
    return YOLONet(layers, rec.save, rec.names).float().eval().requires_grad_(False)
