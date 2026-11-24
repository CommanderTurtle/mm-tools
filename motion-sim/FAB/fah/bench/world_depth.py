"""Metric depth models other than Metric3D for the camera's scale calls (bench/world_dpvo.py --depth_model,
scripts/benchmark_world_e2e.py --depth_model): load(name) -> metric(image RGB or path, [fx, fy, cx, cy]) -> depth (H, W) m,
invalid pixels 0 (RGB arrays, as the Metric3D wrapper it replaces). MoGe-2 (ViT-L, `Ruicheng/moge-2-vitl-normal`, horizontal FOV from K) needs numpy >= 2, so it lives in
its own env (--depth_py); the DPVO env reaches it through Client, a `python -m bench.world_depth serve <name>`
subprocess answering one request per line on stdin / stdout.
"""
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MOGE = os.environ.get("MOGE_CKPT", "Ruicheng/moge-2-vitl-normal")  # hub id or a local model.pt


def load(name):
    import cv2
    import torch

    if name != "moge2":
        raise ValueError(name)
    from moge.model.v2 import MoGeModel

    net = MoGeModel.from_pretrained(MOGE).cuda().eval()

    @torch.no_grad()
    def metric(img, k4):
        img = cv2.imread(img)[..., ::-1] if isinstance(img, str) else img  # RGB
        x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1).float() / 255
        fov = np.degrees(2 * np.arctan(img.shape[1] / (2 * float(k4[0]))))
        d = net.infer(x, fov_x=float(fov), use_fp16=True)["depth"].float().cpu().numpy()
        return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    return metric


class Client:
    """metric(path, k4) served by a `serve` subprocess in another python env (same GPU visibility)."""

    def __init__(self, name, python):
        self.p = subprocess.Popen([python, "-m", "bench.world_depth", "serve", name], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, cwd=REPO, env=dict(os.environ, PYTHONPATH=REPO), text=True)
        assert self.p.stdout.readline().strip() == "ready", "depth server failed to start"

    def __call__(self, img, k4):
        assert isinstance(img, str), "Client takes image paths"
        self.p.stdin.write(" ".join([img, *[str(float(v)) for v in k4]]) + "\n")
        self.p.stdin.flush()
        return np.load(self.p.stdout.readline().strip())


def serve(name):
    proto = os.fdopen(os.dup(1), "w")  # the protocol keeps the real stdout; library prints go to stderr
    os.dup2(2, 1)
    metric = load(name)
    out = f"/dev/shm/world_depth_{os.getpid()}.npy"
    print("ready", file=proto, flush=True)
    for line in sys.stdin:
        path, *k4 = line.split()
        np.save(out, metric(path, np.array(k4, float)).astype(np.float32))
        print(out, file=proto, flush=True)


if __name__ == "__main__":
    serve(sys.argv[2]) if sys.argv[1] == "serve" else None
