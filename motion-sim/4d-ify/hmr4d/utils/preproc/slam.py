import time
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import torch

from dpvo.config import cfg as default_cfg
from dpvo.dpvo import DPVO
from dpvo.utils import Timer
from hmr4d import PROJ_ROOT
from hmr4d.utils.geo.hmr_cam import estimate_focal_length


class SLAMModel:
    def __init__(self, video_path, width, height, intrinsics=None, stride=1, skip=0, buffer=2048, resize=0.5):
        if intrinsics is None:
            focal_length = estimate_focal_length(width, height)
            intrinsics = torch.tensor([focal_length, focal_length, width / 2.0, height / 2.0])
        else:
            intrinsics = intrinsics.clone()

        self.dpvo_cfg = Path(PROJ_ROOT) / "dpvo/config/default.yaml"
        self.dpvo_ckpt = Path(PROJ_ROOT) / "inputs/checkpoints/dpvo/dpvo.pth"
        self.buffer = buffer
        self.times = []
        self.slam = None
        self.queue = Queue(maxsize=8)
        self.reader = Process(
            target=video_stream,
            args=(self.queue, str(video_path), intrinsics, stride, skip, resize),
        )
        self.reader.start()

    def track(self):
        frame_index, image, intrinsics = self.queue.get()
        if frame_index < 0:
            return False

        image = torch.from_numpy(image).permute(2, 0, 1).cuda()
        intrinsics = intrinsics.cuda()
        if self.slam is None:
            config = default_cfg.clone()
            config.merge_from_file(str(self.dpvo_cfg))
            config.BUFFER_SIZE = self.buffer
            self.slam = DPVO(
                config,
                str(self.dpvo_ckpt),
                ht=image.shape[1],
                wd=image.shape[2],
                viz=False,
            )

        with Timer("SLAM", enabled=False):
            started = time.time()
            self.slam(time.time(), image, intrinsics)
            self.times.append(time.time() - started)
        return True

    def process(self):
        if self.slam is None:
            raise RuntimeError("DPVO did not receive any video frames.")
        for _ in range(12):
            self.slam.update()
        self.reader.join()
        return self.slam.terminate()[0]


def video_stream(queue, video_path, intrinsics, stride, skip=0, resize=0.5):
    if len(intrinsics) != 4:
        raise ValueError("intrinsics must be [fx, fy, cx, cy]")

    capture = cv2.VideoCapture(video_path)
    for _ in range(skip):
        if not capture.read()[0]:
            break

    frame_index = 0
    while True:
        image = None
        for _ in range(stride):
            ok, image = capture.read()
            if not ok:
                image = None
                break
        if image is None:
            break

        image = cv2.resize(image, None, fx=resize, fy=resize, interpolation=cv2.INTER_AREA)
        height, width, _ = image.shape
        image = image[: height - height % 16, : width - width % 16]
        queue.put((frame_index, image, intrinsics.clone() * resize))
        frame_index += 1

    queue.put((-1, None, None))
    capture.release()
