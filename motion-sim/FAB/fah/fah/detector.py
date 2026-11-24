import numpy as np


def bilinear_weights(c, n):
    """(len(c), k) weights over the rows / columns [lo, lo + k) of a size-n axis of torchvision roi_align's bilinear
    sample at the coordinates c (zero outside [-1, n], clamped at the borders), and lo."""
    c = c.astype(np.float32)
    out = (c < -1) | (c > n)
    c = np.maximum(c, 0)
    low = c.astype(np.int64)
    top = low >= n - 1
    low = np.where(top, n - 1, low)
    c = np.where(top, low.astype(np.float32), c)
    high = np.where(top, low, low + 1)
    lw = c - low
    lo, k = int(low.min()), int(high.max() - low.min()) + 1
    W = np.zeros((len(c), k), np.float32)
    rows = np.arange(len(c))
    np.add.at(W, (rows, low - lo), np.where(out, 0, 1 - lw))
    np.add.at(W, (rows, high - lo), np.where(out, 0, lw))
    return W, lo


def box_logits(mc, P, x0, y0, x1, y1, r, s, px, py):
    """roi_align(sum_c mc[c] P[c], box, (y1 - y0, x1 - x0), aligned=True, 1 sample) at the frame pixels of the box
    [x0, x1) x [y0, y1), as two small matmuls (bilinear sampling is separable): rows (y1 - y0, kh) @ the combined protos
    in the window the box samples (kh, kw) @ columns (kw, x1 - x0). Same logits up to float rounding, ~5x faster on the
    CPU than roi_align at frame resolution. P: (32, ph, pw) protos, r / (px, py): letterbox scale / pad, s: proto scale."""
    ph, pw = P.shape[-2:]
    Wy, ly = bilinear_weights((np.arange(y0, y1) + 0.5) * r * s + py * s - 0.5, ph)
    Wx, lx = bilinear_weights((np.arange(x0, x1) + 0.5) * r * s + px * s - 0.5, pw)
    win = P[:, ly:ly + Wy.shape[1], lx:lx + Wx.shape[1]]
    comb = (mc.reshape(-1) @ win.reshape(win.shape[0], -1)).reshape(win.shape[1:])
    return Wy @ comb @ Wx.T


class _TRTNet:
    """TensorRT engine of the detector network (scripts/build_det_engine.py: dynamic input shape, fp32 inputs /
    outputs): x (1, 3, H, W) -> {"out": (1, 4 + nc + 32, N), "proto": (1, 32, H / 4, W / 4)} like the torch path; output
    buffers per input shape, reused (consume them before the next call), run on the current torch stream."""

    def __init__(self, path):
        import tensorrt as trt

        self.trt = trt
        self.engine = trt.Runtime(trt.Logger(trt.Logger.ERROR)).deserialize_cuda_engine(open(path, "rb").read())
        self.ctx = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inp = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
        self.outs = [n for n in names if n != self.inp]
        self.bufs = {}

    def __call__(self, x):
        import torch

        x = x.float().contiguous()
        shape = tuple(x.shape)
        if shape not in self.bufs:
            self.ctx.set_input_shape(self.inp, shape)
            self.bufs[shape] = [torch.empty(tuple(self.ctx.get_tensor_shape(n)), dtype=torch.float32, device=x.device) for n in self.outs]
        self.ctx.set_input_shape(self.inp, shape)
        self.ctx.set_tensor_address(self.inp, x.data_ptr())
        out = self.bufs[shape]
        for n, b in zip(self.outs, out):
            self.ctx.set_tensor_address(n, b.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        self._x = x  # keep the input alive until the stream has consumed it
        o = {"out": out[0]}
        if len(out) > 1:
            o["proto"] = out[1]
        return o


class WiLoRHandDetector:
    """YOLO hand detector (the WiLoR detector format, https://github.com/rolpotamias/WiLoR, and FAH's YOLO-seg detectors).
    Returns xyxy boxes, scores and handedness (class 1 = right hand). With YOLO-seg weights (train/build_det_seg.py)
    the instance masks of the last call's detections are available from masks(). The fast path (fast=True with a
    weights file) runs the network with fah.yolo, without the ultralytics package; the ultralytics call (fast=False,
    the WiLoR download when weights is empty, other model types) needs ultralytics (the train / eval extras)."""

    def __init__(
        self,
        weights="",
        device="cpu",
        conf=0.3,
        iou=0.5,
        hf_repo="rolpotamias/WiLoR",
        hf_file="detector.pt",
        fast=False,
        half=False,
        imgsz=512,
        trt="",
        coreml="",
        coreml_units="ALL",
    ):
        self.model, net = None, None  # ultralytics YOLO (the ultralytics call) / fah.yolo.YOLONet (the fast path)
        if fast and weights:
            from .yolo import load_yolo

            try:
                net = load_yolo(weights)
            except ValueError:  # another model type (e.g. a pose model): the ultralytics network
                net = None
        if net is None:
            from ultralytics import YOLO

            if not weights:
                from huggingface_hub import hf_hub_download

                weights = hf_hub_download(hf_repo, hf_file)
            self.model = YOLO(weights)
        self.names = net.names if net is not None else self.model.names
        self.seg = net.seg if net is not None else self.model.task == "segment"
        self.last = None  # last call's detections (seg weights): boxes + mask coefficients / masks
        self.device = str(device)
        self.conf = conf
        self.iou = iou
        # own letterbox/decode/NMS around the network forward, CUDA-graphed on GPU (the eager
        # ultralytics call is kernel-launch bound); on CPU it skips the
        # ultralytics pre/post-processing and honours imgsz
        self.fast = fast
        self.use_graph = fast and self.device.startswith("cuda")
        if self.fast:
            import torch

            from .cuda_graph import GraphedForward

            net = net if net is not None else self.model.model.fuse(verbose=False)  # conv + BN folded, as at inference
            self.net = net.to(self.device).eval()
            if half:
                self.net = self.net.half()
            self.dtype = torch.float16 if half else torch.float32
            self.stride = int(self.net.stride.max())
            self.imgsz = imgsz
            self._GraphedForward = GraphedForward
            self.graphs = {}  # letterbox shape -> graph over its own copy of the network (the detect
            # head caches anchors per input shape, which must not change under a captured graph)
            self.trt = _TRTNet(trt) if trt and self.device.startswith("cuda") else None  # scripts/build_det_engine.py
            # Core ML network (Apple silicon, scripts/build_coreml.py det): letterbox, decode, NMS and masks on the CPU
            self.coreml = None
            if coreml:
                from .coreml import CoreMLDet

                self.coreml = CoreMLDet(coreml, self.net.head, coreml_units)

    def _net_out(self, y):
        if self.model is not None:  # ultralytics: (outputs, raw predictions)
            y = y[0] if isinstance(y, (tuple, list)) else y
        return {"out": y[0], "proto": y[1]} if isinstance(y, (tuple, list)) else {"out": y}  # seg: (preds, protos)

    def _letterbox(self, img):
        # the letterbox the YOLO weights expect (as ultralytics' auto letterbox): the longer side scaled to imgsz, the
        # other padded to the next stride multiple, the padding split evenly (odd pixel right / bottom), gray 114
        import cv2

        h, w = img.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        pw, ph = (self.imgsz - nw) % self.stride, (self.imgsz - nh) % self.stride
        if (w, h) != (nw, nh):
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        left, top = pw // 2, ph // 2
        img = cv2.copyMakeBorder(img, top, ph - top, left, pw - left, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        return img, r, left, top

    def _fast_call(self, img_bgr):
        import torch
        import torchvision

        h, w = img_bgr.shape[:2]
        coreml = getattr(self, "coreml", None)
        if getattr(self, "gpu_letterbox", False) and self.device.startswith("cuda"):  # frame uploaded once (shared with the crops)
            from .frame_cache import frame_tensor

            r = min(self.imgsz / h, self.imgsz / w)
            nw, nh = int(round(w * r)), int(round(h * r))
            pw, ph = (self.imgsz - nw) % self.stride, (self.imgsz - nh) % self.stride  # as _letterbox
            left, top = pw // 2, ph // 2
            right, bottom = pw - left, ph - top
            px, py = left, top
            f = frame_tensor(img_bgr, self.device)
            if (w, h) != (nw, nh):
                f = torch.nn.functional.interpolate(f, size=(nh, nw), mode="bilinear", align_corners=False)
            f = torch.nn.functional.pad(f, (left, right, top, bottom), value=114.0)
            x = (f.flip(1).round() / 255.0).to(self.dtype)  # BGR -> RGB, the uint8 rounding of the CPU path
        else:
            lb, r, px, py = self._letterbox(img_bgr)
            if coreml is not None:  # RGB [0, 1] NCHW float32 in one pass (= x / 255 after the network's fp16 cast)
                import cv2

                lb, px, py = coreml.fit(lb, px, py, f"{w}x{h}")  # a shape not built: padded up to the nearest built one
                x = torch.from_numpy(cv2.dnn.blobFromImage(lb, 1 / 255.0, swapRB=True))
            else:
                x = torch.from_numpy(np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1))).to(self.device)
                x = (x[None].to(self.dtype) / 255.0)
        with torch.no_grad():
            if coreml is not None:
                o = coreml(x.numpy(), conf=self.conf)
            elif self.trt is not None:
                o = self.trt(x)
            elif self.use_graph:
                g = self.graphs.get(x.shape)
                if g is None:
                    import copy

                    net = copy.deepcopy(self.net)
                    g = self.graphs[x.shape] = self._GraphedForward(lambda b, net=net: self._net_out(net(b["x"])))
                o = g({"x": x})
            else:
                o = self._net_out(self.net(x))
            p = o["out"][0].float().T  # (N, 4 + nc + kpts / mask coefficients)
            nc = len(self.names if getattr(self, "names", None) else self.model.names)
            score, cls = p[:, 4 : 4 + nc].max(1)
            # Boolean indexing repeats nonzero (and a CUDA host sync) for every
            # tensor. Resolve the candidate indices once, then gather by index.
            keep = torch.nonzero(score > self.conf, as_tuple=True)[0]
            p, score, cls = p[keep], score[keep], cls[keep]
            cxcy, wh = p[:, :2], p[:, 2:4]
            boxes = torch.cat([cxcy - wh / 2, cxcy + wh / 2], 1)
            idx = torchvision.ops.batched_nms(boxes, score, cls, self.iou)[:300]
            boxes, score, cls = boxes[idx], score[idx], cls[idx]
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - px) / r).clamp(0, w)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - py) / r).clamp(0, h)
        # One device-to-host copy for all detections; class IDs are small integers
        # and are represented exactly in the float32 transfer buffer.
        packed = torch.cat([boxes, score[:, None], cls[:, None].to(boxes.dtype)], 1).cpu().numpy()
        res = (packed[:, :4].astype(np.float32), packed[:, 4].astype(np.float32),
               packed[:, 5].astype(bool))
        if self.seg:  # protos stay a graph output buffer: masks() must run before the next call
            self.last = dict(boxes=res[0], hw=(h, w), mc=p[idx, 4 + nc:], proto=o["proto"][0],
                             scale=o["proto"].shape[-1] / x.shape[-1], r=r, pad=(px, py))
        return res

    def masks(self, keep=None):
        """Full-frame instance masks (n, H, W) bool of the last call's detections (YOLO-seg weights), optionally only
        of the detections `keep`: masks_crop() pasted into the frame. Call before the next detection."""
        L = self.last
        idx = np.arange(len(L["boxes"])) if keep is None else np.asarray(keep, dtype=int).reshape(-1)
        h, w = L["hw"]
        if "masks" in L:  # ultralytics call: already full-frame
            return L["masks"][idx]
        out = np.zeros((len(idx), h, w), bool)
        for k, c in enumerate(self.masks_crop(idx)):
            if c is not None:
                x0, y0, m = c
                out[k, y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        return out

    def masks_crop(self, keep=None):
        """masks() inside each detection box only, without the full-frame fill: per kept detection (x0, y0, crop
        (y1 - y0, x1 - x0) bool) or None (empty box); the full-frame mask is False outside the box (bench.seg.rle
        paste_crop / encode_crop). Fast path: mask logits sampled from the protos at the frame pixels inside each box
        (ultralytics' retina masks, without the letterbox-size upsampling); a pixel claimed by two hands goes to the
        higher logit. Host protos (Core ML) are sampled with two small matmuls per box (box_logits), device protos
        with roi_align. Call before the next detection."""
        L = self.last
        idx = np.arange(len(L["boxes"])) if keep is None else np.asarray(keep, dtype=int).reshape(-1)
        h, w = L["hw"]
        if "masks" in L:  # ultralytics call: full-frame masks, cropped to the box
            out = []
            for i in idx:
                x0, y0, x1, y1 = np.ceil(L["boxes"][i]).astype(int).clip(0, [w, h, w, h])
                out.append((int(x0), int(y0), L["masks"][i][y0:y1, x0:x1]) if x1 > x0 and y1 > y0 else None)
            return out
        if not len(idx):
            return []
        import torch
        import torchvision

        proto, s, r, (px, py) = L["proto"].float(), L["scale"], L["r"], L["pad"]
        host = proto.device.type == "cpu"
        rois, logits = [], []
        with torch.no_grad():
            mc = L["mc"][torch.from_numpy(idx).to(proto.device)].float()
            if host:
                mc, P = mc.numpy(), proto.numpy()
            else:  # combine the protos per detection first (sampling is linear): 1 channel per box is resampled, not 32
                comb = torch.einsum("nc,chw->nhw", mc, proto)[:, None]
            for a, i in enumerate(idx):  # ultralytics crop_mask keeps the pixels x1 <= x < x2
                x0, y0, x1, y1 = np.ceil(L["boxes"][i]).astype(int).clip(0, [w, h, w, h])
                if x1 <= x0 or y1 <= y0:
                    rois.append(None)
                    continue
                rois.append((x0, y0, x1, y1))
                if host:
                    logits.append(torch.from_numpy(box_logits(mc[a], P, x0, y0, x1, y1, r, s, px, py)))
                else:
                    b = torch.tensor([[a, (x0 * r + px) * s, (y0 * r + py) * s, (x1 * r + px) * s, (y1 * r + py) * s]],
                                     device=proto.device, dtype=proto.dtype)
                    logits.append(torchvision.ops.roi_align(comb, b, (y1 - y0, x1 - x0), 1.0, 1, aligned=True)[0, 0])
            keep_m = [lg > 0 for lg in logits]
            live = [k for k, b in enumerate(rois) if b is not None]
            for a, k in enumerate(live):  # overlap: higher logit wins (lower index on ties)
                xa0, ya0, xa1, ya1 = rois[k]
                for c, j in enumerate(live):
                    if j == k:
                        continue
                    xb0, yb0, xb1, yb1 = rois[j]
                    ix0, iy0, ix1, iy1 = max(xa0, xb0), max(ya0, yb0), min(xa1, xb1), min(ya1, yb1)
                    if ix1 <= ix0 or iy1 <= iy0:
                        continue
                    la = logits[a][iy0 - ya0:iy1 - ya0, ix0 - xa0:ix1 - xa0]
                    lb = logits[c][iy0 - yb0:iy1 - yb0, ix0 - xb0:ix1 - xb0]
                    keep_m[a][iy0 - ya0:iy1 - ya0, ix0 - xa0:ix1 - xa0] &= (la > lb) if j < k else (la >= lb)
            flat = torch.cat([m.reshape(-1) for m in keep_m]).cpu().numpy() if keep_m else None  # one transfer
        out, o = [None] * len(idx), 0
        for a, k in enumerate(live):
            x0, y0, x1, y1 = rois[k]
            n = (y1 - y0) * (x1 - x0)
            out[k] = (int(x0), int(y0), flat[o:o + n].reshape(y1 - y0, x1 - x0))
            o += n
        return out

    def mask_at(self, keep, pts):
        """masks(keep)[a][ys, xs] for pts[a] = (xs, ys) int frame pixels (False outside the frame) without building the
        full-frame masks: every kept detection's mask logits sampled from the protos at those pixels only (the bilinear
        sample roi_align takes for that pixel in masks()), then the same box crop and overlap rule. Call before the next
        detection."""
        L = self.last
        idx = np.asarray(keep, dtype=int).reshape(-1)
        h, w = L["hw"]
        n = np.array([len(p[0]) for p in pts], dtype=int)
        if "masks" in L:  # ultralytics call: full-frame masks already there
            out = []
            for i, (xs, ys) in zip(idx, pts):
                ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
                v = np.zeros(len(xs), bool)
                v[ok] = L["masks"][i][ys[ok], xs[ok]]
                out.append(v)
            return out
        if not len(idx) or not n.sum():
            return [np.zeros(k, bool) for k in n]
        import torch

        X = np.concatenate([p[0] for p in pts]).astype(np.float64)
        Y = np.concatenate([p[1] for p in pts]).astype(np.float64)
        proto, s, r, (px, py) = L["proto"].float()[None], L["scale"], L["r"], L["pad"]
        ph, pw = proto.shape[-2:]
        u, v = ((X + 0.5) * r + px) * s - 0.5, ((Y + 0.5) * r + py) * s - 0.5  # proto px (roi_align aligned=True, 1 sample)
        g = torch.from_numpy(np.stack([(2 * u + 1) / pw - 1, (2 * v + 1) / ph - 1], -1)).to(proto.device, proto.dtype)
        with torch.no_grad():  # protos combined per detection before sampling (linear): n_keep channels, not 32
            comb = torch.einsum("nc,chw->nhw", L["mc"][torch.from_numpy(idx).to(proto.device)].float(), proto[0])[None]
            lg = torch.nn.functional.grid_sample(comb, g[None, None], mode="bilinear", padding_mode="border",
                                                 align_corners=False)[0, :, 0].cpu().numpy()  # (n_keep, P), one transfer
        b = np.ceil(L["boxes"][idx]).astype(int).clip(0, [w, h, w, h])
        inb = (X >= b[:, :1]) & (X < b[:, 2:3]) & (Y >= b[:, 1:2]) & (Y < b[:, 3:4])
        inb &= ((b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1]))[:, None]
        a, P = np.repeat(np.arange(len(idx)), n), np.arange(len(X))  # each point's own detection (position in keep)
        own = lg[a, P]
        val = inb[a, P] & (own > 0)
        for c in range(len(idx)):  # overlap: inside both boxes the higher logit wins (lower position on ties)
            val &= ~((c != a) & inb[c] & np.where(c < a, lg[c] >= own, lg[c] > own))
        return np.split(val, np.cumsum(n)[:-1])

    def mask_roi(self, keep, rois):
        """= mask_at(keep, pts) with pts[a] every pixel of rois[a] = (x0, y0, w, h) (row-major), as (h, w) bool arrays:
        the pixel grid, the proto sampling, the box crop and the overlap rule on the device, one bool transfer."""
        L = self.last
        idx = np.asarray(keep, dtype=int).reshape(-1)
        if "masks" in L or not len(idx):
            pts = [(np.tile(np.arange(x0, x0 + w), h), np.repeat(np.arange(y0, y0 + h), w)) for x0, y0, w, h in rois]
            return [m.reshape(r[3], r[2]) for m, r in zip(self.mask_at(keep, pts), rois)]
        import torch

        H, W = L["hw"]
        proto, s, r, (px, py) = L["proto"].float()[None], L["scale"], L["r"], L["pad"]
        ph, pw = proto.shape[-2:]
        dev = proto.device
        mc = L["mc"][torch.from_numpy(idx).to(dev)].float()
        b = torch.from_numpy(np.ceil(L["boxes"][idx]).astype(np.int64).clip(0, [W, H, W, H])).to(dev)
        bok = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
        vals = []
        with torch.no_grad():
            comb = torch.einsum("nc,chw->nhw", mc, proto[0])[None]  # protos combined before sampling (linear): n_keep channels, not 32
            for a, (x0, y0, w, h) in enumerate(rois):
                X = torch.arange(x0, x0 + w, device=dev)  # integer pixels for the box tests (MPS has no float64)
                Y = torch.arange(y0, y0 + h, device=dev)
                xs, ys = np.arange(x0, x0 + w, dtype=np.float64), np.arange(y0, y0 + h, dtype=np.float64)  # sample grid in float64 on the host, as mask_at
                gx = torch.from_numpy(((2 * (((xs + 0.5) * r + px) * s - 0.5) + 1) / pw - 1).astype(np.float32)).to(dev, proto.dtype)
                gy = torch.from_numpy(((2 * (((ys + 0.5) * r + py) * s - 0.5) + 1) / ph - 1).astype(np.float32)).to(dev, proto.dtype)
                g = torch.stack([gx[None, :].expand(h, w), gy[:, None].expand(h, w)], -1)
                lg = torch.nn.functional.grid_sample(comb, g[None], mode="bilinear", padding_mode="border",
                                                     align_corners=False)[0]  # (n_keep, h, w)
                inb = (X[None, None] >= b[:, 0, None, None]) & (X[None, None] < b[:, 2, None, None]) & \
                      (Y[None, :, None] >= b[:, 1, None, None]) & (Y[None, :, None] < b[:, 3, None, None]) & bok[:, None, None]
                own = lg[a]
                val = inb[a] & (own > 0)
                for c in range(len(idx)):  # overlap: inside both boxes the higher logit wins (lower position on ties)
                    if c != a:
                        val &= ~(inb[c] & ((lg[c] >= own) if c < a else (lg[c] > own)))
                vals.append(val.reshape(-1))
            flat = torch.cat(vals).cpu().numpy()
        n = np.array([w * h for _, _, w, h in rois])
        return [v.reshape(rr[3], rr[2]) for v, rr in zip(np.split(flat, np.cumsum(n)[:-1]), rois)]

    def __call__(self, img_bgr):
        if self.fast:
            return self._fast_call(img_bgr)
        det = self.model(
            img_bgr, conf=self.conf, iou=self.iou, device=self.device, verbose=False,
            **({"retina_masks": True} if self.seg else {}),
        )[0]
        boxes = det.boxes.xyxy.cpu().numpy().astype(np.float32)
        scores = det.boxes.conf.cpu().numpy().astype(np.float32)
        is_right = det.boxes.cls.cpu().numpy().astype(bool)
        if self.seg:
            m = det.masks.data.cpu().numpy() > 0.5 if det.masks is not None else np.zeros((0, *img_bgr.shape[:2]), bool)
            self.last = dict(boxes=boxes, hw=img_bgr.shape[:2], masks=m)
        return boxes, scores, is_right


def build_hand_detector(name="wilor", device="cpu", **kwargs):
    if name == "wilor":
        return WiLoRHandDetector(device=device, **kwargs)
    raise ValueError(f"Unknown hand detector: {name}")
