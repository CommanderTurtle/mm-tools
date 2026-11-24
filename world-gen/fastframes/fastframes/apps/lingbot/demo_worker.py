#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Resident, single-node LingBot workers. Idle ranks wait on CPU queues, not NCCL."""
from __future__ import annotations

import hashlib
import io
import json
import multiprocessing as mp
import queue
import tempfile
import time
import traceback
from pathlib import Path

from PIL import Image

ACTIONS = {"idle", "forward", "backward", "left", "right", "yaw_left", "yaw_right"}
LATENTS = 240


def _worker(rank, config, commands, replies, rendezvous):
    # Import CUDA only in spawned workers; each rank sees the same device namespace.
    import torch
    import torch.distributed as dist

    from apps.lingbot.conditioning import KeyboardCamera, LiveCamera
    from apps.lingbot.pipeline import FastFramesLingBot, configure_backends

    pipe = None
    world = config["gpus"]
    fingerprint = None
    try:
        torch.cuda.set_device(rank)
        configure_backends()
        if world > 1:
            from datetime import timedelta

            dist.init_process_group("nccl", init_method=rendezvous, world_size=world, rank=rank,
                                    timeout=timedelta(minutes=10), device_id=torch.device("cuda", rank))
        torch.cuda.reset_peak_memory_stats()
        keys = camera = None
        while True:
            request = commands.get()  # No GPU collective is pending during user idle.
            op = request["op"]
            result = {}
            if op == "shutdown":
                break
            if op == "start":
                if pipe is None:
                    root = Path(config["checkpoint_dir"])
                    pipe = FastFramesLingBot.from_pretrained(
                        "lingbot-1.3b", lingbot_root=config["lingbot_root"],
                        lingbot_checkpoint=config.get("lingbot_checkpoint") or
                        root / "lingbot-world-v2-1.3b-causal-fast",
                        assets_dir=config.get("assets_dir") or root / "lingbot-world-v2-14b-causal-fast",
                        checkpoint_dir=root / "lingbot-1.3b", world=world, rank=rank)
                    keys, camera = KeyboardCamera(), LiveCamera()
                next_fingerprint = (hashlib.sha256(request["image"]).hexdigest(), request["prompt"])
                if fingerprint == next_fingerprint:
                    pipe.restart(seed=request["seed"])
                else:
                    with Image.open(io.BytesIO(request["image"])) as image:
                        pipe.start(image.convert("RGB"), request["prompt"], seed=request["seed"],
                                   latents=LATENTS, noise="chunk")
                if fingerprint is None:
                    # Visit every cache length and both cold/steady decoder shapes twice.
                    for _ in range(2):
                        keys.reset()
                        camera.reset()
                        for _ in range(8):
                            pipe.step(camera.rays(keys.chunk("idle")))
                        pipe.restart(seed=request["seed"])
                fingerprint = next_fingerprint
                keys.reset()
                camera.reset()
            elif op == "step":
                action = request["action"]
                if action not in ACTIONS:
                    raise ValueError(f"Unknown camera action: {action}")
                torch.cuda.synchronize()
                start = time.perf_counter()
                frames = pipe.step(camera.rays(keys.chunk(action)))
                torch.cuda.synchronize()
                model_ms = (time.perf_counter() - start) * 1000
                if world > 1:
                    timing = torch.tensor(model_ms, device="cuda", dtype=torch.float64)
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                    model_ms = timing.item()
                if rank == 0:
                    encoded = []
                    for frame in frames.cpu().numpy():
                        buffer = io.BytesIO()
                        Image.fromarray(frame).save(buffer, format="JPEG", quality=90)
                        encoded.append(buffer.getvalue())
                    result = {"frames": encoded, "chunk_index": pipe.chunk - 1, "model_ms": model_ms,
                              "terminal": pipe.chunk >= pipe.num_chunks, "width": 832, "height": 464}
            elif op == "stop":
                # No generation is in flight: the parent serializes all requests.
                # Keep weights, compiled graphs and the last conditioning for restart.
                pass
            else:
                raise ValueError(f"Unknown worker operation: {op}")
            torch.cuda.synchronize()
            memory = {"rank": rank, "gpu": torch.cuda.get_device_name(),
                      "allocated_bytes": torch.cuda.memory_allocated(),
                      "reserved_bytes": torch.cuda.memory_reserved(),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                      "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
            replies.put({"rank": rank, "result": result, "memory": memory})
            torch.cuda.reset_peak_memory_stats()
    except BaseException:
        replies.put({"rank": rank, "error": traceback.format_exc()})
    finally:
        if pipe is not None:
            pipe.close()
        if dist.is_initialized():
            dist.destroy_process_group()


class WorkerPool:
    """Call from one executor thread. All commands are broadcast through CPU queues."""

    def __init__(self, config):
        self.config = config
        self.processes = []
        self.commands = []
        self.replies = None
        self.directory = None
        self.ready = False
        self.records = []

    def _launch(self):
        ctx = mp.get_context("spawn")
        self.directory = tempfile.TemporaryDirectory(prefix="lingbot-workers-")
        rendezvous = Path(self.directory.name, "rendezvous").as_uri()
        self.replies = ctx.Queue()
        for rank in range(self.config["gpus"]):
            commands = ctx.Queue()
            process = ctx.Process(target=_worker, args=(rank, self.config, commands, self.replies, rendezvous))
            process.start()
            self.commands.append(commands)
            self.processes.append(process)

    def call(self, op, **payload):
        if not self.processes:
            if op == "stop":
                return {}
            self._launch()
        started = time.perf_counter()
        for commands in self.commands:
            commands.put({"op": op, **payload})
        responses = {}
        deadline = time.monotonic() + (1800 if op == "start" else 180)
        try:
            while len(responses) < len(self.processes):
                try:
                    response = self.replies.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"LingBot {op} timed out")
                    if any(process.exitcode is not None for process in self.processes):
                        raise RuntimeError("A LingBot worker exited; check the server log")
                    continue
                if "error" in response:
                    raise RuntimeError(response["error"])
                responses[response["rank"]] = response
        except BaseException:
            self.close(force=True)
            raise
        elapsed = (time.perf_counter() - started) * 1000
        if self.config.get("profile_vram"):
            self.records.append({"operation": op, "wall_ms": elapsed,
                                 "gpus": [responses[rank]["memory"] for rank in sorted(responses)]})
        result = responses[0]["result"]
        result["worker_ms"] = elapsed
        self.ready = True
        return result

    def save_memory_report(self, path):
        report = {"model": "lingbot-1.3b", "gpus": self.config["gpus"], "latents": LATENTS,
                  "records": self.records}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(report, indent=2) + "\n")

    def close(self, force=False):
        if not force:
            for commands in self.commands:
                commands.put({"op": "shutdown"})
            for process in self.processes:
                process.join(timeout=5)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        for commands in self.commands:
            commands.cancel_join_thread()
            commands.close()
        if self.replies is not None:
            self.replies.close()
        self.processes.clear()
        self.commands.clear()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None
        self.ready = False
