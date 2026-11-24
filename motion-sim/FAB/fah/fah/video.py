import numpy as np

from .vis import project


def box_iou(a, b):
    x1, y1 = np.maximum(a[:2], b[:2])
    x2, y2 = np.minimum(a[2:], b[2:])
    inter = max(x2 - x1, 0) * max(y2 - y1, 0)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


BONES = np.array([(0 if j % 4 == 1 else j - 1, j) for j in range(1, 21)])  # (parent, child), 21-joint MANO order


def follow_joints(verts, joints, delta, sigma=0.004):
    """Move a hand mesh with a change of its 21 joints (e.g. a temporal refiner's articulation update): each vertex
    gets the displacement interpolated along its bones ((1 - t) x parent + t x child at its projection t on the
    bone), soft-weighted over bones by distance (sigma, m). verts (V, 3), joints / delta (21, 3)."""
    verts, joints, delta = (np.asarray(x, np.float32) for x in (verts, joints, delta))
    p, c = joints[BONES[:, 0]], joints[BONES[:, 1]]
    seg = c - p
    vp = verts[:, None] - p[None]
    t = np.clip(np.einsum("vbc,bc->vb", vp, seg) / np.maximum((seg ** 2).sum(-1), 1e-12), 0, 1)
    dist = np.sqrt(((vp - t[..., None] * seg) ** 2).sum(-1))
    w = np.exp((dist.min(1, keepdims=True) - dist) / sigma)
    w /= w.sum(1, keepdims=True)
    return verts + (w * (1 - t)) @ delta[BONES[:, 0]] + (w * t) @ delta[BONES[:, 1]]


_FOLLOW = {}


def follow_matrix(n_verts=778, sigma=0.004):
    """follow_joints as a fixed linear map (n_verts, 21) from joint changes to vertex changes, from the MANO template
    (rest pose; the bone weights barely change with the pose for small updates). Cached; None if no MANO model."""
    if n_verts not in _FOLLOW:
        import os
        import pickle

        _FOLLOW[n_verts] = None
        path = os.path.join(os.environ.get("MANO_DIR", "data/mano"), "MANO_RIGHT.pkl")
        if n_verts == 778 and os.path.exists(path):
            m = pickle.load(open(path, "rb"), encoding="latin1")
            dense = lambda x: np.asarray(x.todense() if hasattr(x, "todense") else x, np.float64)
            v = dense(m["v_template"])
            j16 = dense(m["J_regressor"]) @ v
            j21 = np.concatenate([j16, v[[744, 320, 443, 554, 671]]])[[0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]]
            A = np.stack([follow_joints(v, j21, np.eye(21)[:, k:k + 1].repeat(3, 1)) - v for k in range(21)], 1)[..., 0]
            _FOLLOW[n_verts] = A.astype(np.float32)  # (778, 21): vertex change per unit change of joint k
    return _FOLLOW[n_verts]


class OneEuroFilter:
    """One Euro filter (Casiez et al. 2012) on arrays of any shape."""

    def __init__(self, min_cutoff=1.0, beta=0.5, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x_prev, self.dx_prev, self.t_prev = None, None, None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.array(x, dtype=np.float64, copy=True)  # never alias the caller's array
        if self.x_prev is None:
            self.x_prev, self.dx_prev, self.t_prev = x, np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t_prev, 1e-6)
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat.copy()


class HandTracker:
    """Greedy per-frame association (same handedness, box IoU) and One Euro smoothing.
    The camera translation (m) and the root-relative pose (m) are filtered separately, with betas
    in meter units (beta multiplies |speed| in m/s), so fast motion is not lagged; every exported
    3D/2D field is rebuilt from the smoothed translation + relative pose."""

    REL_KEYS = ("vertices", "vertices_mano", "vertices_dense", "keypoints_3d", "keypoints_3d_mano")

    def __init__(self, iou_thresh=0.2, max_missing=10, min_cutoff=1.0, beta_trans=20.0,
                 min_cutoff_rel=1.5, beta_rel=40.0, smooth=True):
        self.iou_thresh = iou_thresh
        self.max_missing = max_missing
        self.trans_args = dict(min_cutoff=min_cutoff, beta=beta_trans)
        self.rel_args = dict(min_cutoff=min_cutoff_rel, beta=beta_rel)
        self.smooth = smooth
        self.tracks = []
        self.next_id = 0

    def update(self, hands, t):
        for track in self.tracks:
            track["missing"] += 1
        order = np.argsort([-h["score"] for h in hands])
        used = set()
        for i in order:
            hand = hands[i]
            best, best_iou = None, self.iou_thresh
            for k, track in enumerate(self.tracks):
                if k in used or track["is_right"] != hand["is_right"]:
                    continue
                iou = box_iou(track["bbox"], hand["bbox"])
                if iou > best_iou:
                    best, best_iou = k, iou
            if best is None:
                track = dict(
                    id=self.next_id,
                    is_right=hand["is_right"],
                    filters={"cam_t": OneEuroFilter(**self.trans_args),
                             **{k: OneEuroFilter(**self.rel_args) for k in self.REL_KEYS}},
                )
                self.next_id += 1
                self.tracks.append(track)
                best = len(self.tracks) - 1
            used.add(best)
            track = self.tracks[best]
            track["bbox"], track["missing"] = hand["bbox"], 0
            hand["track_id"] = track["id"]
            if self.smooth:
                cam_t = track["filters"]["cam_t"](hand["cam_t"], t)
                for key in self.REL_KEYS:
                    if hand.get(key) is None:
                        continue
                    rel = track["filters"][key](np.asarray(hand[key]) - hand["cam_t"], t)
                    hand[key] = rel + cam_t
                hand["keypoints_3d_rel"] = hand["keypoints_3d"] - cam_t
                hand["cam_t"] = cam_t
                hand["keypoints_2d"] = project(hand["keypoints_3d"], hand["cam_int"])
                if hand.get("keypoints_3d_mano") is not None:
                    hand["keypoints_2d_mano"] = project(hand["keypoints_3d_mano"], hand["cam_int"])
        self.tracks = [tr for tr in self.tracks if tr["missing"] <= self.max_missing]
        return hands


class DetectionSchedule:
    """Run the hand detector only every `every` frames (or when no hand is tracked); in between, reuse
    each hand's box as predicted by the decoder on the previous frame (`pred_box`, same convention
    as the detector boxes) and its handedness. Saves the detector cost on most frames."""

    def __init__(self, detector, every=1):
        self.detector, self.every, self.t = detector, max(int(every), 1), 0

    def reset(self):
        self.t = 0

    def __call__(self, img_bgr, prev_hands):
        run = self.t % self.every == 0 or not prev_hands
        self.t += 1
        if run:
            return self.detector(img_bgr)
        boxes = np.stack([h["pred_box"] for h in prev_hands]).astype(np.float32)
        h, w = img_bgr.shape[:2]
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
        keep = (boxes[:, 2] - boxes[:, 0] > 4) & (boxes[:, 3] - boxes[:, 1] > 4)
        scores = np.array([h["score"] for h in prev_hands], np.float32)
        is_right = np.array([h["is_right"] for h in prev_hands], bool)
        return boxes[keep], scores[keep], is_right[keep]


def wrist_dist(a, b):
    """Distance of two hands' wrists (2D joints (21, 2), MANO order) in mean palm lengths (wrist - middle MCP): the
    identity rule of FAHMethod pair_wrist (two crossing hands have overlapping boxes but different wrists)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a[0] - b[0]) / max(0.5 * (np.linalg.norm(a[0] - a[9]) + np.linalg.norm(b[0] - b[9])), 1e-6))


def point_box(h):
    """Tight xyxy box of a hand's predicted 2D points (joints + dense surface points when present)."""
    uv = np.asarray(h["keypoints_2d"], np.float32)
    if h.get("surface_2d_head") is not None and len(h["surface_2d_head"]):
        uv = np.concatenate([uv, np.asarray(h["surface_2d_head"], np.float32)])
    return np.concatenate([uv.min(0), uv.max(0)])


class TrackProposals:
    """Causal video detector memory: hand tracks from the kept hands of past frames, used to (a) propose a box for each
    confirmed track that the detector has no confident box for in the current frame (the track's last point box moved
    by its constant-velocity prediction, enlarged; verified by the pose network's presence like a detector box) and
    (b) gate the refiner's gap fill of tracks the proposal found empty. Online: only past frames are used."""

    def __init__(self, max_missing=3, min_hits=2, enlarge=1.1, cover_iou=0.5, cover_score=0.25, match_iou=0.3, vel_ema=0.5,
                 min_inside=0.6, cover_ioa=0.7, match_wrist=0.0):
        self.max_missing, self.min_hits, self.enlarge, self.cover_ioa = max_missing, min_hits, enlarge, cover_ioa
        self.match_wrist = match_wrist  # > 0: hands join tracks by wrist distance (palm lengths) to the predicted wrist; 0: box IoU
        self.cover_iou, self.cover_score, self.match_iou, self.vel_ema = cover_iou, cover_score, match_iou, vel_ema
        self.min_inside = min_inside  # a predicted box must keep this fraction of its area inside the image
        self.reset()

    def reset(self):
        self.tracks, self.next_id = [], 0
        self.leaving = set()  # ids of the tracks whose predicted box left the image (last proposals call)

    def predicted(self, hw=None):
        """[(track, box)] of the live tracks: box = last point box moved by the velocity over the frames since."""
        out = []
        if hw is not None:
            self.leaving = set()
        for tr in self.tracks:
            if tr["since_det"] > self.max_missing:
                continue
            b = tr["box"].copy()
            d = tr["vel"] * (tr["missing"] + 1)
            b[[0, 2]] += d[0]
            b[[1, 3]] += d[1]
            if hw is not None:
                area = (b[2] - b[0]) * (b[3] - b[1])
                h, w = hw
                c = np.array([max(b[0], 0), max(b[1], 0), min(b[2], w - 1), min(b[3], h - 1)], np.float32)
                if area <= 0 or max(c[2] - c[0], 0) * max(c[3] - c[1], 0) < self.min_inside * area:
                    self.leaving.add(tr["id"])
                    continue  # the hand is leaving the image
            out.append((tr, b))
        return out

    def proposals(self, boxes, scores, hw):
        """Boxes (n, 4), sides (n,) and track ids for the confirmed tracks without a confident detector box (score >=
        cover_score, _covers) at their predicted position; the boxes enlarged and clipped to the image."""
        props, sides, ids = [], [], []
        for tr, b in self.predicted(hw):
            if tr["hits"] < self.min_hits:
                continue
            if any(s >= self.cover_score and self._covers(d, b) for d, s in zip(boxes, scores)):
                continue
            c, half = (b[:2] + b[2:]) / 2, (b[2:] - b[:2]) / 2 * self.enlarge
            h, w = hw
            props.append(np.clip(np.concatenate([c - half, c + half]), 0, [w - 1, h - 1, w - 1, h - 1]))
            sides.append(tr["is_right"])
            ids.append(tr["id"])
        return np.array(props, np.float32).reshape(-1, 4), np.array(sides, bool), ids

    def _covers(self, d, b):
        """Detector box d holds the predicted hand box b: IoU >= cover_iou, or b mostly inside d (>= cover_ioa of its area)
        with d at most 3x b's area (detector boxes and point boxes differ in convention: wrist / forearm, padding)."""
        if box_iou(b, d) >= self.cover_iou:
            return True
        x1, y1 = np.maximum(b[:2], d[:2])
        x2, y2 = np.minimum(b[2:], d[2:])
        ab, ad = (b[2] - b[0]) * (b[3] - b[1]), (d[2] - d[0]) * (d[3] - d[1])
        return max(x2 - x1, 0) * max(y2 - y1, 0) >= self.cover_ioa * ab and ad <= 3 * ab

    def expected(self, track_id):
        """The track's last 2D joints moved by its velocity to the current frame (21, 2), or None."""
        for tr in self.tracks:
            if tr["id"] == track_id and tr.get("kp") is not None:
                return tr["kp"] + tr["vel"] * (tr["missing"] + 1)
        return None

    def update(self, hands):
        """Associate this frame's kept hands to the tracks: by identity (wrist within match_wrist palm lengths of the
        track's predicted wrist, nearest pairs first), or (match_wrist 0) greedy by score on point-box IoU vs the
        predicted box. Hands from proposals (h["vd_prop"]) move their track but do not keep it alive: a track lives
        max_missing frames after its last detector hand, and only detector hands count as hits."""
        pred = self.predicted()
        exp = {tr["id"]: tr["kp"] + tr["vel"] * (tr["missing"] + 1) for tr, _ in pred if tr.get("kp") is not None}
        for tr in self.tracks:
            tr["missing"] += 1
            tr["since_det"] += 1
        order = sorted(range(len(hands)), key=lambda i: -hands[i].get("score", 0.0))
        assign, used = {}, set()
        if self.match_wrist:
            pairs = sorted(((wrist_dist(hands[i]["keypoints_2d"], exp[tr["id"]]), k, tr["id"]), i, tr) for k, i in enumerate(order)
                           for tr, _ in pred if tr["id"] in exp)
            for (d, _, _), i, tr in pairs:
                if d < self.match_wrist and i not in assign and tr["id"] not in used:
                    assign[i] = tr
                    used.add(tr["id"])
        else:
            for i in order:
                pb, best, best_iou = point_box(hands[i]), None, self.match_iou
                for tr, b in pred:
                    if tr["id"] not in used and box_iou(pb, b) > best_iou:
                        best, best_iou = tr, box_iou(pb, b)
                if best is not None:
                    assign[i] = best
                    used.add(best["id"])
        for i in order:
            h = hands[i]
            pb, best = point_box(h), assign.get(i)
            prop = bool(h.get("vd_prop"))
            if best is None:
                if prop:  # a proposal always continues a track; no new track from it
                    continue
                best = dict(id=self.next_id, box=pb, vel=np.zeros(2, np.float32), hits=0, missing=1, since_det=1)
                self.next_id += 1
                self.tracks.append(best)
            else:
                step = (pb[:2] + pb[2:]) / 2 - (best["box"][:2] + best["box"][2:]) / 2
                best["vel"] = self.vel_ema * step / best["missing"] + (1 - self.vel_ema) * best["vel"]
            used.add(best["id"])
            best.update(box=pb, is_right=bool(h["is_right"]), missing=0, hits=best["hits"] + (not prop),
                        kp=np.asarray(h["keypoints_2d"], np.float64).copy())
            if not prop:
                best["since_det"] = 0
            h["vd_track"] = best["id"]
        self.tracks = [tr for tr in self.tracks if tr["since_det"] <= self.max_missing]


class LearnedSmoother:
    """Online (causal) learned video refiner (train/temporal.py, trained with --causal) as a drop-in for
    HandTracker.update: per hand side, the last T frames of per-frame outputs go through the transformer; the
    hand's root is replaced by the refined root and its articulation by `blend` x refined + (1 - blend) x
    per-frame (the setting the refiner was evaluated with); mesh / 2D follow. Needs the per-frame model the
    refiner was trained on. cuda_graph: replay the full-window (T frames) forward from a captured CUDA graph (same
    kernels and outputs)."""

    def __init__(self, ckpt, blend=0.75, device="cpu", gap=2, max_shift=0.05, cuda_graph=False, mesh_follow=False, euro=None,
                 anchor_2d=False, keep_2d=False, euro_2d=None):
        import torch
        from train.temporal import TemporalRefiner

        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck["args"]
        assert a["causal"], "LearnedSmoother needs an online (causal) refiner"
        self.model = TemporalRefiner(dim=a["dim"], layers=a["layers"], max_len=a["T"], causal=True).eval()
        self.model.load_state_dict(ck["model"], strict=False)
        self.model.to(device)
        self.T, self.blend, self.device, self.gap, self.max_shift = a["T"], blend, device, gap, max_shift
        self.mesh_follow = mesh_follow  # move the mesh with the refined articulation too (fah.video.follow_joints)
        self.euro = euro  # (min_cutoff, beta): One Euro filter on the refined root-relative joints; the mesh follows it
        self.anchor_2d = anchor_2d  # 2D = the per-frame model's own 2D + the change of the projection (stays pixel-aligned)
        self.keep_2d = keep_2d  # 2D = the per-frame model's own 2D (the refiner changes 3D / mesh only)
        self.euro_2d = euro_2d  # (min_cutoff, beta per px/s): One Euro filter on the exported 2D joints (visual coherence)
        self.cuda_graph, self.graph = cuda_graph and str(device).startswith("cuda"), None
        self.no_fill = set()  # track ids (hand["vd_track"]) not to gap-fill this frame (set by the caller: the video detector)
        self.reset()

    def _forward(self, x):
        import torch

        if not self.cuda_graph or x.shape[1] != self.T:
            return self.model(x)
        if self.graph is None:
            self.static_x = x.clone()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self.model(self.static_x)
            torch.cuda.current_stream().wait_stream(s)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_out = self.model(self.static_x)
        self.static_x.copy_(x)
        self.graph.replay()
        return self.static_out

    def reset(self):
        self.buffers = {True: [], False: []}
        self.last = {True: None, False: None}  # last real hand per side and frames since (gap filling)
        self.filters, self.n = {True: None, False: None}, 0
        self.filters_2d = {True: None, False: None}

    def update(self, hands, t=None):
        import torch
        from train.temporal import frame_feats, root_to_xyz

        for side in (True, False):
            cand = [h for h in hands if bool(h["is_right"]) == side]
            h = max(cand, key=lambda x: x.get("score", 0)) if cand else None
            if h is None:
                self.buffers[side].append(np.zeros_like(self.buffers[side][-1]) if self.buffers[side] else None)
            else:
                j3 = h.get("keypoints_3d_mano", h["keypoints_3d"])
                j2 = h.get("keypoints_2d_mano", h["keypoints_2d"])
                self.buffers[side].append(frame_feats(dict(joints3d=j3, joints2d=j2, score=h.get("score", 1.0)), h["cam_int"]))
            self.buffers[side] = [x for x in self.buffers[side] if x is not None][-self.T:]
            fill = False
            if h is None:  # missed: predict the hand from its track if it was seen within `gap` frames
                if self.last[side] is None:
                    continue
                prev, n = self.last[side]
                if n + 1 > self.gap or not self.buffers[side] or prev.get("vd_track", -1) in getattr(self, "no_fill", ()):
                    self.last[side] = None
                    continue
                self.last[side] = (prev, n + 1)
                if any(box_iou(prev["bbox"], o["bbox"]) > 0.3 for o in hands):  # the hand is there, side flipped
                    continue
                # Gap filling predicts geometry only; last frame's image-space labels and contact
                # are not observations of the current frame.
                h = {k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in prev.items()
                     if k not in ("parts_roi", "parts_fg_roi", "mask", "mask_arm") and not k.startswith("contact")}
                h["score"], fill = 0.0, True
                hands.append(h)
            else:
                self.last[side] = (h, 0)
            x = torch.as_tensor(np.stack(self.buffers[side]), dtype=torch.float32, device=self.device)[None]
            with torch.no_grad():
                rel, root = self._forward(x)
                rel, root = rel[0, -1].cpu(), root[0, -1].cpu()
            rel = rel.numpy().astype(np.float64) / 10.0
            root = root_to_xyz(root.numpy().astype(np.float64))
            j3 = np.asarray(h.get("keypoints_3d_mano", h["keypoints_3d"]), np.float64)
            old = {k: np.asarray(h[k], np.float64) for k in ("keypoints_2d", "keypoints_2d_mano", "keypoints_3d", "keypoints_3d_mano")
                   if h.get(k) is not None}
            if not fill and np.linalg.norm(root - j3[0]) > self.max_shift:  # out-of-distribution guard
                root = j3[0]
            same = "keypoints_3d_mano" not in h or np.allclose(h["keypoints_3d"], h["keypoints_3d_mano"])  # decoder-free
            new = (rel if fill else self.blend * rel + (1 - self.blend) * (j3 - j3[:1])) + root
            shift = root - j3[0]
            for key in ("vertices", "vertices_mano", "vertices_dense", "keypoints_3d"):
                if h.get(key) is not None:
                    h[key] = np.asarray(h[key]) + shift
            if self.mesh_follow and not fill:
                for key in ("vertices", "vertices_mano", "vertices_dense"):
                    if h.get(key) is not None:
                        h[key] = follow_joints(np.asarray(h[key], np.float64), j3 + shift, new - (j3 + shift)).astype(np.float32)
            if self.euro is not None:
                if self.filters[side] is None:
                    self.filters[side] = OneEuroFilter(*self.euro)
                sm = self.filters[side](new - new[:1], self.n / 30.0 if t is None else t) + new[:1]
                for key in ("vertices", "vertices_mano", "vertices_dense"):
                    if h.get(key) is not None:
                        A = follow_matrix(len(h[key]))
                        h[key] = (np.asarray(h[key], np.float32) + A @ (sm - new).astype(np.float32)) if A is not None else \
                            follow_joints(np.asarray(h[key], np.float64), new, sm - new).astype(np.float32)
                new = sm
            h["keypoints_3d_mano" if "keypoints_3d_mano" in h else "keypoints_3d"] = new.astype(np.float32)
            if same:
                h["keypoints_3d"] = new.astype(np.float32)
            h["cam_t"] = np.asarray(h["cam_t"]) + shift
            h["keypoints_3d_rel"] = h["keypoints_3d"] - h["cam_t"]
            for k2, k3 in (("keypoints_2d", "keypoints_3d"), ("keypoints_2d_mano", "keypoints_3d_mano")):
                if h.get(k3) is None:
                    continue
                uv = project(h[k3], h["cam_int"])
                base = old.get(k2, old.get("keypoints_2d"))  # the per-frame 2D (dense models have no keypoints_2d_mano)
                if self.keep_2d and not fill and base is not None:
                    uv = base
                elif self.anchor_2d and not fill and base is not None and k3 in old:
                    uv = base + uv - project(old[k3], h["cam_int"])
                h[k2] = np.asarray(uv, np.float32)
            if self.euro_2d is not None:
                if self.filters_2d[side] is None:
                    self.filters_2d[side] = OneEuroFilter(*self.euro_2d)
                k2 = "keypoints_2d_mano" if h.get("keypoints_2d_mano") is not None and not same else "keypoints_2d"
                uv = self.filters_2d[side](h[k2], self.n / 30.0 if t is None else t).astype(np.float32)
                h[k2] = uv
                if same:  # decoder-free model: both 2D fields are the same joints (the exported one is keypoints_2d_mano)
                    for key in ("keypoints_2d", "keypoints_2d_mano"):
                        if h.get(key) is not None:
                            h[key] = uv
            if fill:  # box of the predicted hand (detector convention, padded joint box)
                uv = h.get("keypoints_2d_mano", h["keypoints_2d"])
                pad = 0.15 * (uv.max(0) - uv.min(0)).max()
                h["bbox"] = np.concatenate([uv.min(0) - pad, uv.max(0) + pad]).astype(np.float32)
        for side in (True, False):  # a side without a hand this frame restarts its filter
            if not any(bool(x["is_right"]) == side for x in hands):
                self.filters[side] = self.filters_2d[side] = None
        self.n += 1
        return hands
