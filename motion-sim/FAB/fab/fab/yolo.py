"""The person detector (an Ultralytics YOLO11 detection checkpoint) without the ultralytics package.

load_yolo(path, device) rebuilds the network from the training checkpoint (the same design as the YOLOv8 reader of
FAH): the checkpoint is read with an unpickler that never imports ultralytics (its objects come back as plain attribute
records), the layer graph (each layer's inputs `f`, index `i`, the saved outputs) and every tensor are taken from those
records, the batch norms are folded into their convolutions in float32 on the CPU (torch.nn.utils.fusion.fuse_conv_bn_eval;
the fold, device and weight memory format of Ultralytics' own predictor, which folds the model where YOLO() loaded it and
then moves it: channels-last weights on x86 Linux / Windows CPUs with oneDNN, contiguous otherwise) and the layers run
with this module's forward passes. Supported layers: Conv / DWConv, Bottleneck, C2f, C3k2 (Bottleneck or
C3k blocks), C3 / C3k, SPPF, C2PSA (PSABlock: multi-head attention on the feature map + a 1x1 feed-forward), Concat,
torch layers (Upsample, ...) and the Detect head (box distance distributions decoded on the anchor grid + class
probabilities); other checkpoints raise ValueError.

YOLONet(x) -> (B, 4 + nc, A): boxes (cx, cy, w, h in input pixels) and class probabilities per anchor, the layout of
the Ultralytics DetectionModel's inference output.

YOLODetector(path, device).detect(image) runs it as FABPredictor's detector: letterbox() (the image resized to fit
640 px, cv2.INTER_LINEAR on the host, centred on a stride-padded 114 canvas, / 255), the network, nms_one_class() (the
confidence candidates, xywh -> xyxy, torchvision's NMS, max_det) and scale_boxes() (back to image pixels, clipped)
-> (N, 4) xyxy float32 boxes, the boxes of the ultralytics predictor on the same checkpoint (scripts/check_detector.py).
IoU PREDICT_IOU = 0.5: the pinned ultralytics predictor's default for detection (earlier releases used 0.7).
"""
import pickle
import platform
import types

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

PREDICT_IOU = 0.5  # NMS IoU of the ultralytics (8.4.211) predictor for detection, which FABPredictor ran with


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


_PICKLE = types.ModuleType("fab_yolo_pickle")
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
    """Two convolutions, plus the input when the block has a shortcut (Bottleneck)."""

    def __init__(self, first, second, skip):
        super().__init__()
        self.first, self.second, self.skip = first, second, skip

    def forward(self, x):
        out = self.second(self.first(x))
        return x + out if self.skip else out


class SplitChain(nn.Module):
    """Entry conv, its channels in two halves, a chain of blocks fed from the second half, then every intermediate map
    concatenated into the exit conv (C2f; C3k2, whose blocks are Bottlenecks or C3k)."""

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


class CrossStage(nn.Module):
    """Two 1x1 convs side by side, the first through a stack of blocks, both concatenated into the exit conv (C3, C3k)."""

    def __init__(self, deep, side, exit, blocks):
        super().__init__()
        self.deep, self.side, self.exit, self.blocks = deep, side, exit, blocks

    def forward(self, x):
        return self.exit(torch.cat([self.blocks(self.deep(x)), self.side(x)], 1))


class PoolPyramid(nn.Module):
    """Entry conv, the same max pool applied `repeats` times in a row, all maps concatenated into the exit conv (SPPF)."""

    def __init__(self, entry, exit, pool, repeats=3, skip=False):
        super().__init__()
        self.entry, self.exit, self.pool, self.repeats, self.skip = entry, exit, pool, repeats, skip

    def forward(self, x):
        maps = [self.entry(x)]
        while len(maps) <= self.repeats:
            maps.append(self.pool(maps[-1]))
        out = self.exit(torch.cat(maps, 1))
        return out + x if self.skip else out


class MapAttention(nn.Module):
    """Multi-head self-attention over the cells of a feature map: one 1x1 conv gives every head's query, key and value
    channels; the attention output plus a depthwise 3x3 conv of the values (a positional term) goes through the 1x1
    output conv (the Attention of PSABlock)."""

    def __init__(self, qkv, out, pos, heads, key_dim, head_dim, scale):
        super().__init__()
        self.qkv, self.out, self.pos = qkv, out, pos
        self.heads, self.key_dim, self.head_dim, self.scale = heads, key_dim, head_dim, scale

    def forward(self, x):
        n, c, h, w = x.shape
        t = self.qkv(x).view(n, self.heads, 2 * self.key_dim + self.head_dim, h * w)
        q, k, v = t.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        weights = torch.matmul((q * self.scale).transpose(-2, -1), k).softmax(dim=-1)  # (n, heads, cells, cells)
        mixed = torch.matmul(v, weights.transpose(-2, -1)).view(n, c, h, w)
        return self.out(mixed + self.pos(v.reshape(n, c, h, w)))


class AttentionBlock(nn.Module):
    """Attention, then a two-conv feed-forward, each added to its input when the block has a shortcut (PSABlock)."""

    def __init__(self, attn, ffn, skip):
        super().__init__()
        self.attn, self.ffn, self.skip = attn, ffn, skip

    def forward(self, x):
        x = x + self.attn(x) if self.skip else self.attn(x)
        return x + self.ffn(x) if self.skip else self.ffn(x)


class SplitAttention(nn.Module):
    """Entry conv, its channels in two halves, the second through the attention blocks, both into the exit conv
    (C2PSA)."""

    def __init__(self, entry, exit, blocks):
        super().__init__()
        self.entry, self.exit, self.blocks = entry, exit, blocks

    def forward(self, x):
        t = self.entry(x)
        half = t.shape[1] // 2
        return self.exit(torch.cat([t[:, :half], self.blocks(t[:, half:])], 1))


class Join(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, dim=self.dim)


class YOLOHead(nn.Module):
    """Per level: box branch (4 x bins distance logits) and class branch; the expected distance of each bin
    distribution (an integral over the bins: the checkpoint's fixed 1x1 conv), boxes on the anchor grid, class
    probabilities."""

    def __init__(self, box_branch, cls_branch, integral, classes, bins, stride):
        super().__init__()
        self.box_branch, self.cls_branch, self.integral = box_branch, cls_branch, integral
        self.classes, self.bins, self.levels = classes, bins, len(box_branch)
        self.register_buffer("stride", stride.float(), persistent=False)
        self._grid = {}

    def expect(self, boxes):
        """(B, 4 x bins, A) bin logits -> (B, 4, A) expected distances."""
        if self.integral is None:  # one bin: the distances themselves
            return boxes
        n, _, a = boxes.shape
        p = boxes.reshape(n, 4, self.bins, a).transpose(1, 2).softmax(dim=1)
        return self.integral(p)[:, 0]

    def anchors(self, feats):
        key = tuple(tuple(f.shape[-2:]) for f in feats) + (feats[0].dtype, feats[0].device)
        if key not in self._grid:
            self._grid[key] = tuple(t.T for t in make_anchors(feats, self.stride.tolist()))
        return self._grid[key]

    def forward(self, feats):
        n = feats[0].shape[0]
        boxes = torch.cat([self.box_branch[k](feats[k]).reshape(n, 4 * self.bins, -1) for k in range(self.levels)], 2)
        scores = torch.cat([self.cls_branch[k](feats[k]).reshape(n, self.classes, -1) for k in range(self.levels)], 2)
        grid, step = self.anchors(feats)
        return torch.cat([dist2box(self.expect(boxes), grid[None]) * step, scores.sigmoid()], 1)


class YOLONet(nn.Module):
    """The layers in order; a layer reads the previous output (f = -1), an earlier layer's output (f = index) or a list
    of them, and the outputs that later layers read (save) are kept."""

    def __init__(self, layers, save, names):
        super().__init__()
        self.layers, self.save, self.names = nn.ModuleList(layers), set(save), dict(names)

    @property
    def head(self):
        return self.layers[-1]

    @property
    def stride(self):
        return self.head.stride

    def forward(self, x):
        kept = {}
        for m in self.layers:
            src = [m.f] if isinstance(m.f, int) else m.f
            ins = [x if j == -1 else kept[j if j >= 0 else m.i + j] for j in src]
            x = m(ins[0] if isinstance(m.f, int) else ins)
            if m.i in self.save:
                kept[m.i] = x
        return x


def _torch(m):
    m = m.float().eval()
    if isinstance(m, nn.Upsample) and not hasattr(m, "recompute_scale_factor"):  # pickled by an older torch
        m.recompute_scale_factor = None
    return m


def _conv(rec):
    c = rec._modules
    conv = _torch(c["conv"])
    if c.get("bn") is not None:  # fold the batch norm (as at inference; float32, on the CPU)
        conv = fuse_conv_bn_eval(conv, _torch(c["bn"]))
    return ConvAct(conv.requires_grad_(False), _torch(c["act"]))


def _build(rec):
    """A checkpoint module (an ultralytics record, a torch container of them or a torch layer) -> this module's."""
    if isinstance(rec, nn.Sequential):
        return nn.Sequential(*[_build(r) for r in rec._modules.values()])
    if isinstance(rec, nn.ModuleList):
        return nn.ModuleList(_build(r) for r in rec._modules.values())
    if isinstance(rec, nn.Module):
        return _torch(rec)  # a torch layer (nn.Upsample, nn.MaxPool2d, nn.Conv2d, ...)
    kind, c, d = type(rec).__name__, getattr(rec, "_modules", {}), rec.__dict__
    sub = lambda k: _build(c[k])  # noqa: E731
    if kind in ("Conv", "DWConv"):
        return _conv(rec)
    if kind == "Bottleneck":
        return Residual(sub("cv1"), sub("cv2"), bool(d.get("add", False)))
    if kind in ("C2f", "C3k2"):
        return SplitChain(sub("cv1"), sub("cv2"), sub("m"))
    if kind in ("C3", "C3k"):
        return CrossStage(sub("cv1"), sub("cv2"), sub("cv3"), sub("m"))
    if kind == "SPPF":
        return PoolPyramid(sub("cv1"), sub("cv2"), sub("m"), int(d.get("n", 3)), bool(d.get("add", False)))
    if kind == "Attention":
        if d.get("format") is not None:
            raise ValueError(f"unsupported attention format {d['format']}")
        return MapAttention(sub("qkv"), sub("proj"), sub("pe"), int(d["num_heads"]), int(d["key_dim"]),
                            int(d["head_dim"]), float(d["scale"]))
    if kind == "PSABlock":
        return AttentionBlock(sub("attn"), sub("ffn"), bool(d.get("add", True)))
    if kind == "C2PSA":
        return SplitAttention(sub("cv1"), sub("cv2"), sub("m"))
    if kind == "Concat":
        return Join(d.get("d", 1))
    if kind == "Detect":
        if d.get("_end2end") or c.get("one2one_cv2") is not None or d.get("xyxy"):
            raise ValueError("end-to-end (one-to-one) / xyxy YOLO heads are not supported")
        dfl = c["dfl"]
        integral = None if isinstance(dfl, nn.Identity) else _torch(dfl._modules["conv"]).requires_grad_(False)
        return YOLOHead(sub("cv2"), sub("cv3"), integral, int(d["nc"]), int(d["reg_max"]),
                        torch.as_tensor(d["stride"]).float())
    raise ValueError(f"unsupported YOLO layer {kind} ({getattr(rec, '_source', type(rec).__module__)})")


def channels_last(device):
    """The weight memory format Ultralytics' predictor picks for a PyTorch model on `device`: channels-last on x86
    CPUs with oneDNN under Linux / Windows (its auto default), contiguous on CUDA and elsewhere."""
    return (torch.device(device).type == "cpu" and platform.machine() in ("x86_64", "AMD64")
            and platform.system() in ("Linux", "Windows") and torch.backends.mkldnn.is_available()
            and torch.backends.mkldnn.enabled)


def load_yolo(path, device="cpu"):
    """YOLONet (float32, eval, batch norms folded on the CPU, then moved to `device`) of an Ultralytics YOLO11 / YOLOv8
    detection checkpoint."""
    ck = read_checkpoint(path)
    if ck.get("modelopt"):
        raise ValueError(f"{path}: quantization-aware (QAT) checkpoints are not supported")
    rec = ck.get("ema") or ck["model"]
    layers = []
    for r in rec._modules["model"]._modules.values():
        m = _build(r)
        m.f, m.i = r.__dict__.get("f", -1), r.__dict__.get("i", len(layers))
        layers.append(m)
    if not isinstance(layers[-1], YOLOHead):
        raise ValueError(f"{path}: not a YOLO detection model")
    net = YOLONet(layers, rec.save, rec.names).to(device).eval().requires_grad_(False)
    return net.to(memory_format=torch.channels_last if channels_last(device) else torch.contiguous_format)


def letterbox_shape(H, W, size=640, stride=32):
    """-> (canvas h, canvas w, resized h, resized w): the image scaled to fit size x size (aspect kept, each side
    rounded, at least 1 px), the canvas the smallest stride multiple padding of it (Ultralytics' rect letterbox)."""
    r = min(size / H, size / W)
    nh, nw = max(round(H * r), 1), max(round(W * r), 1)
    return nh + (size - nh) % stride, nw + (size - nw) % stride, nh, nw


def letterbox(image, size=640, stride=32, device="cpu", h2d=None):
    """(H, W, 3) uint8 image -> (1, 3, h, w) float32 network input on `device`: cv2.INTER_LINEAR resize on the host,
    centred on a 114 canvas (the odd padding pixel at the bottom / right), / 255. h2d(key, array): the host -> device
    copy to use (FABPredictor's page-locked one), else torch's."""
    import cv2

    H, W = image.shape[:2]
    h2, w2, nh, nw = letterbox_shape(H, W, size, stride)
    small = image if (nh, nw) == (H, W) else cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    small = h2d("small", small) if h2d is not None else torch.from_numpy(np.ascontiguousarray(small)).to(device)
    top, left = round((h2 - nh) / 2 - 0.1), round((w2 - nw) / 2 - 0.1)
    im = torch.full((1, 3, h2, w2), 114.0, device=device)
    im[:, :, top : top + nh, left : left + nw] = small.permute(2, 0, 1)[None].float()
    return im / 255.0


def xywh2xyxy(b):
    """(n, 4) centre / size -> corners."""
    half = b[:, 2:] / 2
    return torch.cat([b[:, :2] - half, b[:, :2] + half], 1)


def nms_one_class(pred, conf=0.25, iou=PREDICT_IOU, max_det=300):
    """One image's output (5, A) of a one-class model -> (n, 5) xyxy (input px) + score: the anchors scoring above conf,
    torchvision's NMS at iou, the max_det best."""
    import torchvision

    if pred.shape[0] != 5:
        raise ValueError(f"nms_one_class: a one-class output (5, A) expected, got {tuple(pred.shape)}")
    x = pred.transpose(0, 1)[pred[4] > conf]
    if not len(x):
        return x
    x[:, :4] = xywh2xyxy(x[:, :4])
    return x[torchvision.ops.nms(x[:, :4], x[:, 4], iou)[:max_det]]


def scale_boxes(boxes, input_hw, image_hw):
    """(n, 4) xyxy boxes on the letterboxed input (h, w) -> image pixels (in place), clipped to the image (H, W): the
    padding removed, each axis divided by its own resize factor (the rounded resized side over the image side)."""
    H, W = image_hw[:2]
    r = min(input_hw[0] / H, input_hw[1] / W)
    nh, nw = max(round(H * r), 1), max(round(W * r), 1)
    gy, gx = nh / H, nw / W
    py, px = round((input_hw[0] - nh) / 2 - 0.1), round((input_hw[1] - nw) / 2 - 0.1)
    for k, (pad, gain, hi) in enumerate(((px, gx, W), (py, gy, H), (px, gx, W), (py, gy, H))):
        boxes[:, k] -= pad
        boxes[:, k] /= gain
        boxes[:, k].clamp_(0, hi)
    return boxes


class YOLODetector:
    """The person detector: detect(image) -> (N, 4) xyxy float32 boxes (image px) of class 0."""

    def __init__(self, path, device="cpu", size=640):
        self.device, self.size = torch.device(device), int(size)
        self.net = load_yolo(path, self.device)
        if self.net.head.classes != 1:
            raise ValueError(f"{path}: a one-class detector expected ({self.net.head.classes} classes)")
        self.stride = max(int(self.net.stride.max()), 32)
        self.names = self.net.names

    @torch.no_grad()
    def detect(self, image, conf=0.25, iou=PREDICT_IOU, max_det=300, h2d=None):
        """image (H, W, 3) uint8 RGB -> (N, 4) xyxy float32 boxes, best first."""
        im = letterbox(image, self.size, self.stride, self.device, h2d)
        x = nms_one_class(self.net(im)[0], conf, iou, max_det)
        if not len(x):
            return np.zeros((0, 4), np.float32)
        return np.ascontiguousarray(scale_boxes(x[:, :4], im.shape[2:], image.shape[:2]).float().cpu().numpy())
