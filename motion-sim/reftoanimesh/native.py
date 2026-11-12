"""Adapt native output for a bounded-memory browser viewer; preserve motion data."""

import time

import cv2
import numpy as np
import torch

from state import atomic, read


def select_person(path, candidates, default, cancel, progress):
    previews = path / "persons"
    previews.mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(path / "native/input/input_30fps.mp4"))
    try:
        for candidate in candidates:
            cancel()
            boxes = np.asarray(candidate["boxes"])
            largest = int(
                np.argmax((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            )
            candidate["preview_frame"] = candidate["frames"][largest]
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate["frames"][largest])
            ok, image = cap.read()
            if not ok:
                continue
            h, w = image.shape[:2]
            x1, y1, x2, y2 = boxes[largest]
            pad_x, pad_y = (x2 - x1) * 0.15, (y2 - y1) * 0.08
            crop = image[
                max(0, int(y1 - pad_y)) : min(h, int(y2 + pad_y)),
                max(0, int(x1 - pad_x)) : min(w, int(x2 + pad_x)),
            ]
            if crop.size:
                crop = cv2.resize(
                    crop, (max(1, round(crop.shape[1] * 192 / crop.shape[0])), 192)
                )
                cv2.imwrite(str(previews / f"{candidate['index']}.jpg"), crop)
                candidate["thumbnail"] = f"persons/{candidate['index']}.jpg"
    finally:
        cap.release()
    atomic(
        path / "selection.json",
        dict(candidates=candidates, default=default, deadline=time.time() + 600),
    )
    progress(
        "selection", 0, "Choose a person; the suggested track is used after 10 minutes"
    )
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        cancel()
        choice = read(path / "choice.json")
        if (
            choice
            and type(choice.get("index")) is int
            and 0 <= choice["index"] < len(candidates)
        ):
            return choice["index"]
        time.sleep(0.2)
    atomic(path / "choice.json", {"index": default, "automatic": True})
    return default


def export_viewer(path, motion, engine, cancel, progress):
    out = path / "viewer"
    out.mkdir(exist_ok=True)
    with np.load(motion, allow_pickle=False) as data:
        poses = np.asarray(data["poses"], dtype=np.float32).reshape(-1, 52, 3)
        trans = np.asarray(data["trans"], dtype=np.float32)
        betas = np.asarray(data["betas"], dtype=np.float32).reshape(1, -1)
    with np.load(path / "native/input/input_bbox.npz", allow_pickle=False) as data:
        start, end = map(int, data["start_end"])
        boxes = data["bbox"].astype(float).round(1).tolist()
    model = engine.smpl_mesh
    dev = model.v_template.device
    betas = torch.from_numpy(betas[:, : model.shapedirs.shape[-1]]).to(dev)
    chunks, skeleton = [], []
    low, high = np.full(3, np.inf), np.full(3, -np.inf)
    with torch.no_grad():
        for begin in range(0, len(poses), 64):
            cancel()
            finish = min(begin + 64, len(poses))
            params = dict(
                poses=torch.from_numpy(poses[begin:finish]).to(dev),
                shapes=betas,
                trans=torch.from_numpy(trans[begin:finish]).to(dev),
            )
            vertices = model(params)["vertices"].float().cpu().numpy()
            if not np.isfinite(vertices).all():
                raise ValueError("Native mesh contains non-finite coordinates")
            joints = engine.body_model(params)["keypoints3d"].float().cpu().numpy()
            skeleton.extend(joints.tolist())
            name = f"vertices-{begin:06}.bin"
            vertices.astype("<f4").tofile(out / name)
            chunks.append(dict(start=begin, frames=finish - begin, file=name))
            low = np.minimum(low, vertices.reshape(-1, 3).min(0))
            high = np.maximum(high, vertices.reshape(-1, 3).max(0))
            progress("viewer", finish * 100 // len(poses), "Preparing mesh frames")
    np.asarray(model.faces, dtype="<u4").tofile(out / "faces.bin")
    atomic(
        out / "skeleton.json",
        dict(positions=skeleton, parents=engine.body_model.parents.cpu().tolist()),
    )
    atomic(out / "track.json", dict(start=start, end=end, boxes=boxes, fps=30))
    # The native export has already rotated and floor-aligned the result. Apply
    # neither operation again: the browser shares its exact Y-up world frame.
    atomic(
        out / "scene.json",
        dict(
            frames=len(poses),
            vertices=int(vertices.shape[1]),
            fps=30,
            video_start=start,
            video="native/input/input_30fps.mp4",
            motion=motion.relative_to(path).as_posix(),
            chunks=chunks,
            bounds=[low.tolist(), high.tolist()],
            trajectory=trans.tolist(),
            dtype="float32",
            up_axis="y",
        ),
    )
