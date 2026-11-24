# SPDX-License-Identifier: AGPL-3.0-only
import asyncio
import json
import os
import signal
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

from apps.lingbot.demo import DemoApplication, WorkerPool, examples_from, response
from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed
from state import LANE_STAMP

ROOT = Path(__file__).resolve().parent
MODELS = ROOT.parents[1] / "models/fastframes"
LING = ROOT / "lingbot"


async def serve_studio():
    examples = examples_from(LING)
    gpus = int(os.getenv("LINGBOT_GPUS","1"))
    if gpus not in (1,8): raise SystemExit("LINGBOT_GPUS must be 1 or 8.")
    def new_world():
        return DemoApplication(WorkerPool(dict(checkpoint_dir=str(MODELS), lingbot_root=str(LING),
                                              gpus=gpus)), examples)
    world, live, generating = new_world(), set(), False
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)

    async def http(conn, req):
        path = urlsplit(req.path).path
        if path == "/generate":
            origin = urlsplit(req.headers.get("Origin", ""))
            if origin.netloc != req.headers.get("Host") or origin.scheme not in ("http","https"):
                return response(403,b"Same origin required","text/plain")
            if not req.headers.get("Upgrade"): return response(400,b"WebSocket required","text/plain")
            return None
        if path == "/":
            return response(200,(ROOT/"studio.html").read_bytes(),"text/html; charset=utf-8")
        return await world.http(conn, req)

    async def interact(ws):
        nonlocal world, live, generating
        try:
            if ws.request.path == "/ws":
                if generating: raise ValueError("FastFrames-L is rendering. Retry after it finishes.")
                if not (MODELS/"lingbot-1.3b").is_dir(): raise ValueError("Select fastframes-live in the picker.")
                live.add(ws)
                try: await world.websocket(ws)
                finally: live.discard(ws)
                return
            if ws.request.path != "/generate": return
            if generating or world.waiting_room.owner is not None or world.waiting_room.waiters: raise ValueError("Finish the active studio session first.")
            if not (MODELS/"fastframes-l/dit.safetensors").is_file(): raise ValueError("Select fastframes-l in the picker.")
            generating = True
            try:
                spec = json.loads(await asyncio.wait_for(ws.recv(),30))
                prompt, seconds, seed = spec["prompt"],spec["seconds"],spec["seed"]
                if not isinstance(prompt,str) or not 0<len(prompt.strip())<=2000 or seconds not in (2,3,4,5) or type(seed)!=int or not 0<=seed<2**32:
                    raise ValueError("Check the prompt, seed and duration.")
                if world.waiting_room.owner is not None or world.waiting_room.waiters:
                    raise ValueError("A world started; retry after its turn.")
                await asyncio.gather(*(s.close(1001,"Switching model") for s in list(live)))
                await world.shutdown()
                world = new_world()
                with tempfile.TemporaryDirectory(prefix="fastframes-") as tmp:
                    job = Path(tmp)
                    spec["source"] = LANE_STAMP
                    (job/"spec.json").write_text(json.dumps(spec))
                    proc = await asyncio.create_subprocess_exec(sys.executable,str(ROOT/"render.py"),tmp)
                    try:
                        while True:
                            if ws.close_code is not None or stop.is_set(): return
                            for preview in sorted(job.glob("part-*.bin")):
                                await ws.send(json.dumps({"type":"preview"}))
                                await ws.send(preview.read_bytes())
                                preview.unlink()
                            if proc.returncode is not None: break
                            try: await asyncio.wait_for(proc.wait(),.15)
                            except asyncio.TimeoutError: pass
                        if proc.returncode: raise ValueError("Render failed; see the studio console.")
                        await ws.send(json.dumps({"type":"movie","seed":seed}))
                        await ws.send((job/"movie.mp4").read_bytes())
                    finally:
                        if proc.returncode is None:
                            proc.terminate()
                            try: await asyncio.wait_for(proc.wait(),5)
                            except asyncio.TimeoutError:
                                proc.kill()
                                await proc.wait()
            finally: generating = False
        except ConnectionClosed: pass
        except (ValueError,KeyError,TypeError,asyncio.TimeoutError) as exc:
            await ws.send(json.dumps({"type":"error","message":str(exc)}))

    port=int(os.getenv("GRADIO_SERVER_PORT","8296"))
    print(f"FastFrames studio: http://127.0.0.1:{port}",flush=True)
    try:
        async with serve(interact,"127.0.0.1",port,process_request=http,compression=None,max_size=12*1024*1024,ping_timeout=60):
            await stop.wait()
    finally: await world.shutdown()


if __name__ == "__main__":
    asyncio.run(serve_studio())
