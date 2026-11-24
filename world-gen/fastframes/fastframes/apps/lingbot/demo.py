#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Interactive LingBot 1.3B on one server: python apps/lingbot/demo.py --gpus 1."""
from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import logging
import os
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

sys.path[0] = str(Path(__file__).resolve().parents[2])

from PIL import Image, ImageOps, UnidentifiedImageError  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402
from websockets.datastructures import Headers  # noqa: E402
from websockets.exceptions import ConnectionClosed  # noqa: E402
from websockets.http11 import Response  # noqa: E402

from apps.lingbot.demo_waiting_room import WaitingRoom  # noqa: E402
from apps.lingbot.demo_worker import ACTIONS, WorkerPool  # noqa: E402

LOGGER = logging.getLogger("lingbot.demo")
STATIC = Path(__file__).with_name("static")
MAX_IMAGE_BYTES = 8 * 1024 * 1024
IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")  # decoders allowed for uploaded images


def examples_from(root):
    """Read upstream example images/prompts; no bundled/private demo assets."""
    examples = {}
    for directory in sorted(Path(root, "examples").glob("*")):
        image = directory / "image.jpg"
        prompt = directory / "prompt.txt"
        if not image.is_file() or not prompt.is_file():
            continue
        if not directory.name.replace("_", "").replace("-", "").isalnum():
            continue
        text = prompt.read_text().strip()
        if not text or len(text) > 1600:
            continue
        examples[directory.name] = {"id": directory.name, "label": f"Scene {directory.name}",
                                    "prompt": text, "image_url": f"/examples/{directory.name}.jpg",
                                    "path": image}
    return examples


def prepare_image(raw):
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Choose an image smaller than 8 MiB")
    try:
        with Image.open(io.BytesIO(raw), formats=IMAGE_FORMATS) as image:
            if image.width * image.height > 16_000_000:
                raise ValueError("Choose an image smaller than 16 megapixels")
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = ImageOps.fit(image, (832, 464), method=Image.Resampling.BICUBIC)
            output = io.BytesIO()
            image.save(output, format="PNG")
            return output.getvalue()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("Could not read this image; use JPEG, PNG or WebP") from exc


def start_arguments(payload, examples):
    prompt = payload.get("prompt", "")
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 1600:
        raise ValueError("Enter a prompt of 1–1600 characters")
    seed = payload.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an integer between 0 and 4294967295")
    image = payload.get("image")
    if image is not None:
        if not isinstance(image, str):
            raise ValueError("Upload a JPEG, PNG or WebP image")
        try:
            header, encoded = image.split(",", 1)
            if header not in {"data:image/jpeg;base64", "data:image/png;base64", "data:image/webp;base64"}:
                raise ValueError("Unsupported image format")
            if len(encoded) > MAX_IMAGE_BYTES * 4 // 3 + 8:
                raise ValueError("Image exceeds 8 MiB")
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Upload a valid JPEG, PNG or WebP image smaller than 8 MiB") from exc
    else:
        example = payload.get("example")
        if not isinstance(example, str) or example not in examples:
            raise ValueError("Choose an example or upload a start image")
        raw = examples[example]["path"].read_bytes()
    return {"prompt": prompt.strip(), "seed": seed, "image": prepare_image(raw)}


@dataclass
class Session:
    owner_id: str
    stopping: bool = False
    prepared: bool = False
    frame_sequence: int = 0
    chunk: int = -1
    awaiting_display: bool = False
    terminal: bool = False
    actions: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=1))
    displayed: asyncio.Event = field(default_factory=asyncio.Event)
    generation_task: asyncio.Task | None = None


class DemoApplication:
    def __init__(self, pool, examples, *, idle_seconds=120):
        self.pool, self.examples = pool, examples
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingbot")
        self.active = None
        self.closed = False
        self.waiting_room = WaitingRoom(self, idle_seconds=idle_seconds)

    @staticmethod
    async def _send(websocket, payload):
        await websocket.send(json.dumps(payload, separators=(",", ":")))

    async def _error(self, websocket, message, *, code="invalid_request"):
        await self._send(websocket, {"type": "error", "code": code, "message": message})

    async def _call(self, op, **payload):
        # A CUDA call is not cancellable. Always drain it before resetting state
        # or admitting the next player, including disconnected cold starts.
        future = asyncio.get_running_loop().run_in_executor(self.executor, lambda: self.pool.call(op, **payload))
        try:
            return await asyncio.shield(future)
        except asyncio.CancelledError:
            await future
            raise

    async def websocket(self, websocket):
        connection_id = uuid4().hex
        room = self.waiting_room
        room.connect(connection_id, websocket)
        try:
            await self._send(websocket, {"type": "hello", "gpus": self.pool.config["gpus"],
                                         "ready": self.pool.ready})
            await self._send(websocket, room.state(connection_id))
            async for raw in websocket:
                try:
                    if not isinstance(raw, str):
                        raise ValueError("Controls must be JSON text")
                    payload = json.loads(raw)
                    if not isinstance(payload, dict):
                        raise ValueError("Controls must be JSON objects")
                    await self.handle(websocket, connection_id, payload)
                except (ValueError, TypeError) as exc:
                    await self._error(websocket, str(exc))
        except ConnectionClosed:
            pass
        finally:
            await room.disconnect(connection_id)

    async def handle(self, websocket, connection_id, payload):
        kind = payload.get("type")
        room = self.waiting_room
        if kind == "start":
            # Validate before reserving the GPU; private inputs aren't broadcast.
            args = start_arguments(payload, self.examples)
            await room.start(websocket, connection_id, args)
        elif kind == "join_queue":
            await room.join(connection_id)
        elif kind in {"stop", "leave_queue"}:
            await room.leave(connection_id)
        elif kind == "activity":
            if room.owner == connection_id:
                await room.activity(connection_id)
        elif kind in {"action", "displayed"}:
            active = self.active
            if (room.owner != connection_id or active is None or active.owner_id != connection_id
                    or active.stopping or not active.prepared):
                raise ValueError("Start your world before moving the camera")
            if kind == "displayed":
                if active.awaiting_display and payload.get("chunk_index") == active.chunk:
                    active.awaiting_display = False
                    active.displayed.set()
                return
            action = payload.get("action")
            request_id = payload.get("request_id")
            if not isinstance(action, str) or action not in ACTIONS:
                raise ValueError("Use WASD or left/right arrows to move the camera")
            if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                raise ValueError("Camera actions need a request_id of 1–128 characters")
            if active.terminal:
                return
            if active.actions.full():
                active.actions.get_nowait()
            active.actions.put_nowait((action, request_id, time.monotonic()))
            await room.activity(connection_id)
        else:
            raise ValueError("Unknown message type")

    async def _start_session(self, websocket, connection_id, args):
        await self._stop_session(connection_id)
        active = Session(connection_id)
        self.active = active
        try:
            await self._send(websocket, {"type": "status", "state": "preparing",
                                         "message": "Preparing your world… First start includes model warmup."})
            await self._call("start", **args)
            active.prepared = True
            await self._send(websocket, {"type": "status", "state": "ready", "message": "World ready"})
            active.generation_task = asyncio.create_task(self._generate(websocket, active))
        except BaseException:
            await self._stop_session(connection_id)
            if not isinstance(sys.exc_info()[1], (asyncio.CancelledError, ConnectionClosed)):
                LOGGER.exception("Model preparation failed")
                await self._error(websocket, "Model preparation failed. Check the server log.", code="model_error")
            raise

    async def _generate(self, websocket, active):
        command = ("idle", None, time.monotonic())
        try:
            while not active.stopping:
                action, request_id, queued_at = command
                await self._send(websocket, {"type": "status", "state": "generating", "message": "Generating…"})
                queue_ms = (time.monotonic() - queued_at) * 1000
                batch = await self._call("step", action=action)
                frames = batch.pop("frames")
                active.chunk = batch["chunk_index"]
                active.terminal = batch["terminal"]
                active.frame_sequence += len(frames)
                active.displayed.clear()
                active.awaiting_display = True
                await self._send(websocket, {"type": "chunk", "request_id": request_id,
                                             "rgb_frames": len(frames), "frame_sizes": list(map(len, frames)),
                                             "queue_ms": queue_ms, **batch})
                await websocket.send(b"".join(frames))
                # Backpressure includes playback. Never precompute a long stale
                # action stream while the browser is still showing a block.
                await active.displayed.wait()
                if active.terminal:
                    await self._send(websocket, {"type": "status", "state": "complete",
                                                 "message": "World complete. Start again to explore a new world."})
                    break
                await self._send(websocket, {"type": "status", "state": "ready", "message": "Ready to explore"})
                command = await active.actions.get()
        except asyncio.CancelledError:
            raise
        except ConnectionClosed:
            pass
        except Exception:
            LOGGER.exception("Generation failed")
            try:
                await self._error(websocket, "Generation failed. Check the server log and start again.",
                                  code="model_error")
            except ConnectionClosed:
                pass
        finally:
            # Schedule cleanup outside this task; cleanup may cancel and await it.
            if not active.stopping:
                asyncio.create_task(self.waiting_room.session_released(active.owner_id))

    async def _stop_session(self, connection_id, notify=None):
        active = self.active
        if active is None or active.owner_id != connection_id:
            return
        active.stopping = True
        task = active.generation_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await self._call("stop")
        self.active = None
        if notify is not None:
            await self._send(notify, {"type": "status", "state": "stopped", "message": "World stopped"})

    async def shutdown(self):
        if self.closed:
            return
        await self.waiting_room.shutdown()
        await asyncio.get_running_loop().run_in_executor(self.executor, self.pool.close)
        self.executor.shutdown(wait=True)
        self.closed = True

    async def http(self, connection, request):
        path = urlsplit(request.path).path
        if path == "/ws":
            origin = request.headers.get("Origin")
            host = request.headers.get("Host")
            if origin and (urlsplit(origin).scheme not in {"http", "https"} or urlsplit(origin).netloc != host):
                return response(403, b"This demo accepts same-origin connections only", "text/plain")
            return None
        if request.headers.get("Upgrade"):
            return response(404, b"Not found", "text/plain")
        if path == "/api/health":
            return response(200, json.dumps({"ready": self.pool.ready, "gpus": self.pool.config["gpus"],
                                             "busy": self.waiting_room.owner is not None}).encode())
        if path == "/api/examples":
            examples = [{k: v for k, v in item.items() if k != "path"} for item in self.examples.values()]
            return response(200, json.dumps(examples).encode())
        for example in self.examples.values():
            if path == example["image_url"]:
                return response(200, example["path"].read_bytes(), "image/jpeg")
        files = {"/": ("index.html", "text/html"), "/index.html": ("index.html", "text/html"),
                 "/styles.css": ("styles.css", "text/css"), "/app.js": ("app.js", "text/javascript")}
        if path in files:
            name, mime = files[path]
            return response(200, (STATIC / name).read_bytes(), mime)
        if path == "/favicon.ico":
            return response(204, b"", "image/x-icon")
        return response(404, b"Not found", "text/plain")


def response(status, body, mime="application/json"):
    return Response(status, HTTPStatus(status).phrase,
                    Headers({"Content-Type": mime, "Content-Length": str(len(body)),
                             "Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"}), body)


async def run(args):
    app = DemoApplication(WorkerPool(vars(args)), examples_from(args.lingbot_root), idle_seconds=args.idle_timeout)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, stop.set)
    try:
        async with serve(app.websocket, args.host, args.port, process_request=app.http, compression=None,
                         max_size=12 * 1024 * 1024, max_queue=16, ping_interval=20, ping_timeout=60):
            print(f"LingBot 1.3B · {args.gpus} GPU(s) · http://{args.host}:{args.port}", flush=True)
            await stop.wait()
            await app.shutdown()
    finally:
        await app.shutdown()


def profile(args):
    """Profile the same resident backend used by the browser; no HTTP server."""
    examples = examples_from(args.lingbot_root)
    if not examples:
        raise ValueError("VRAM profiling needs an upstream examples/<scene>/image.jpg and prompt.txt")
    example = next(iter(examples.values()))
    image = prepare_image(example["path"].read_bytes())
    start = {"image": image, "prompt": example["prompt"], "seed": 0}
    pool = WorkerPool(vars(args))
    try:
        print("Cold start: weights, conditioning, compilation and two warmup passes", flush=True)
        pool.call("start", **start)
        pool.save_memory_report(args.profile_vram)
        for action in ("idle", "forward", "yaw_left", "yaw_right", "left", "right", "backward", "forward"):
            result = pool.call("step", action=action)
            print(f"Chunk {result['chunk_index']}: model {result['model_ms']:.1f} ms", flush=True)
            pool.save_memory_report(args.profile_vram)
        print("Warm restart, then changed prompt and image", flush=True)
        pool.call("stop")
        pool.call("start", **start)
        pool.call("step", action="idle")
        with Image.open(io.BytesIO(image)) as source:
            changed = io.BytesIO()
            ImageOps.mirror(source).save(changed, format="PNG")
        pool.call("start", image=changed.getvalue(), prompt=example["prompt"] + " Soft morning light.", seed=1)
        pool.call("step", action="idle")
        pool.save_memory_report(args.profile_vram)
        print(f"VRAM report: {args.profile_vram}", flush=True)
    finally:
        pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(1, 8), default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--lingbot-root", default=os.environ.get("LINGBOT_WORLD_ROOT"))
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--lingbot-checkpoint", help="official 1.3B DiT directory")
    parser.add_argument("--assets-dir", help="shared T5/tokenizer/VAE directory")
    parser.add_argument("--idle-timeout", type=float, default=120,
                        help="idle seconds before giving a waiting visitor a turn")
    parser.add_argument("--profile-vram", metavar="JSON", help="profile startup, generation and restarts, then exit")
    args = parser.parse_args()
    if not args.lingbot_root or not Path(args.lingbot_root, "wan", "modules", "model_fast.py").is_file():
        parser.error("set LINGBOT_WORLD_ROOT or --lingbot-root to the pinned upstream clone "
                     "(see docs/lingbot.md)")
    if args.gpus == 8:
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
            if "expandable_segments:true" in os.environ.get(name, "").replace(" ", "").lower():
                parser.error(f"unset {name}: expandable segments are unsupported on the eight-GPU path")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.profile_vram:
        profile(args)
    else:
        asyncio.run(run(args))


if __name__ == "__main__":
    main()
