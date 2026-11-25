import argparse
import asyncio
import fcntl
import json
import subprocess
import sys
import threading
import uuid
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

from paths import (
    ADAPTERS,
    DB,
    MODELS,
    NATIVE,
    OUTPUT,
    PIN,
    ROOT,
    SNAPSHOTS,
    WORK,
    WORKER_ID,
    model_identity,
    require_models,
)
from paths import LANE_STAMP

sys.path.insert(0, str(NATIVE))


def main():
    parser = argparse.ArgumentParser(description="Hy-Nomas studio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8286)
    parser.add_argument(
        "--pe-url", help="Optional local Nomas-PE /v1/chat/completions endpoint"
    )
    args = parser.parse_args()
    require_models()
    WORK.mkdir(parents=True, exist_ok=True)
    with (WORK / "studio.lock").open("a") as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("This studio is already running") from None
        import uvicorn
        from apps import seg_ar_gateway
        from apps.kimi_k3_prompt_enhancer import KimiK3Client, OnlinePromptEnhancer
        from enhancer import DisabledCloud, ImageCaptioner, LocalEnhancer
        from fastapi import HTTPException, Request
        from fastapi.responses import FileResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from processes import Processes
        from queue_store import StudioStore

        store = StudioStore(str(DB))
        processes = Processes(store)

        def export_overlay(source, destination, overlay):
            epoch = processes.epoch
            child = processes.spawn(
                [str(ROOT / ".venv/bin/python"), str(ROOT / "export_worker.py")],
                epoch=epoch,
                stdin=subprocess.PIPE,
                stdout=processes.log,
                text=True,
                encoding="utf-8",
            )
            try:
                child.communicate(
                    json.dumps(
                        {
                            "source": str(source),
                            "destination": str(destination),
                            "overlay": overlay,
                        }
                    ),
                    timeout=3600,
                )
                if child.returncode or epoch != processes.epoch:
                    raise RuntimeError("Export interrupted; see runtime log")
            except subprocess.TimeoutExpired as exc:
                processes.terminate(child)
                raise RuntimeError("Export exceeded one hour") from exc
            finally:
                processes.reap(child)

        # Keep native overlay math, but give its ffmpeg children the same lifetime
        # boundary as inference. The gateway itself never retains an encoder.
        seg_ar_gateway.burn_shared_overlay_video = export_overlay
        store.reset_worker(WORKER_ID)
        store.clear_worker(WORKER_ID)
        for job in store.list_jobs(limit=100000):
            if job["status"] in {"enhancing", "queued", "running"}:
                store.patch(
                    job["id"],
                    status="failed",
                    error="Studio restarted; create or restore a world",
                )
        client = LocalEnhancer(processes)
        if args.pe_url:
            parsed = urlsplit(args.pe_url)
            if (
                parsed.scheme != "http"
                or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                or parsed.username
            ):
                raise SystemExit("--pe-url must be a loopback HTTP endpoint")
            local_api = KimiK3Client(
                api_url=args.pe_url, model="Nomas-PE", timeout=3600, retries=1
            )

            class ExternalEnhancer:
                configured = True
                model = "Nomas-PE"

                def complete(self, messages, **kwargs):
                    epoch = processes.epoch
                    result = local_api.complete(messages, **kwargs)
                    if epoch != processes.epoch:
                        raise RuntimeError("Studio stopped this request")
                    return result

            client = ExternalEnhancer()
        enhancer = OnlinePromptEnhancer(
            output_dir=OUTPUT, client=client, k3_client=DisabledCloud()
        )
        app = seg_ar_gateway.create_gateway_app(
            store,
            frontend_dir=None,
            output_dir=OUTPUT,
            gallery_dir=WORK / "gallery",
            prompt_enhancer=enhancer,
            image_captioner=ImageCaptioner(client),
            inference_config={
                "rollout_type": "seg_ar",
                "height": 704,
                "width": 1280,
                "fps": 24,
                "num_inference_steps": 8,
                "guidance_scale": 1,
                "flow_shift": 5,
                "sp_size": 1,
                "tp_size": 1,
            },
        )
        native_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(instance):
            async with native_lifespan(instance):
                try:
                    yield
                finally:
                    await asyncio.to_thread(processes.close)

        app.router.lifespan_context = lifespan
        maintenance = {"status": "idle", "error": None}
        maintenance_lock = threading.Lock()

        @app.middleware("http")
        async def local_request(request: Request, call_next):
            origin = request.headers.get("origin")
            if origin and urlsplit(origin).netloc != request.headers.get("host"):
                return JSONResponse(
                    {"detail": "Use the studio's own page"}, status_code=403
                )
            host = request.url.hostname
            if host not in {"localhost", "127.0.0.1", "::1", args.host}:
                return JSONResponse({"detail": "Unrecognized host"}, status_code=403)
            if (
                request.method == "POST"
                and request.url.path in {"/v1/generate", "/v1/reference-images"}
                and maintenance["status"] == "running"
            ):
                return JSONResponse(
                    {"detail": "Wait for adapter preparation"}, status_code=409
                )
            if (
                request.method == "POST"
                and request.url.path == "/v1/generate"
                and not store.worker().get("online")
            ):
                return JSONResponse({"detail": "Load the model first"}, status_code=409)
            return await call_next(request)

        @app.get("/studio/status")
        def status():
            alive = processes.worker is not None and processes.worker.poll() is None
            return {
                "worker": store.worker()
                if alive
                else {"online": False, "status": "stopped"},
                "starting": alive and not store.worker().get("online"),
                "adapter": processes.adapter,
                "preparation": dict(maintenance),
                "source": LANE_STAMP,
                "enhancer": "local endpoint"
                if args.pe_url
                else "ephemeral CPU · BF16 · ~56 GB weights",
                "adapters": [
                    {
                        "id": name,
                        "ready": name == "base"
                        or (WORK / "adapters" / name / "ready.json").is_file(),
                    }
                    for name in ADAPTERS
                ],
            }

        @app.post("/studio/load")
        def load(payload: dict):
            with maintenance_lock:
                if maintenance["status"] == "running":
                    raise HTTPException(409, "Adapter preparation is active")
                try:
                    processes.start(payload.get("adapter", "base"))
                except ValueError as exc:
                    raise HTTPException(400, str(exc)) from exc
            return status()

        @app.post("/studio/stop")
        def stop():
            processes.stop()
            return status()

        @app.get("/studio/log", response_class=FileResponse)
        def log():
            return FileResponse(WORK / "runtime.log", media_type="text/plain")

        @app.post("/studio/prepare")
        def prepare(payload: dict):
            name = payload.get("adapter")
            if name not in ADAPTERS[1:]:
                raise HTTPException(400, "Choose cinematic or dual_arm_robotics")
            if not (
                MODELS / "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors"
            ).is_file():
                raise HTTPException(
                    400, "Select hynomas:1+2 in the model downloader first"
                )
            with maintenance_lock:
                if (
                    maintenance["status"] == "running"
                    or store.active_job_count()
                    or (processes.worker and processes.worker.poll() is None)
                ):
                    raise HTTPException(
                        409, "Stop the model before preparing an adapter"
                    )
                maintenance.update(status="running", error=None)
                epoch = processes.epoch

                def run():
                    child = None
                    try:
                        child = processes.spawn(
                            [
                                str(ROOT / ".venv/bin/python"),
                                str(ROOT / "prepare_adapter.py"),
                                name,
                            ],
                            stdout=processes.log,
                            epoch=epoch,
                        )
                        if child.wait() != 0 or epoch != processes.epoch:
                            raise RuntimeError(
                                "Adapter preparation interrupted; see runtime log"
                            )
                        maintenance.update(status="complete")
                    except Exception as exc:  # noqa: BLE001 — subprocess boundary reports failure to the UI
                        maintenance.update(status="failed", error=str(exc))
                    finally:
                        if child:
                            processes.reap(child)

                threading.Thread(target=run, daemon=True).start()
            return dict(maintenance)

        @app.get("/studio/jobs/{job_id}")
        def job_status(job_id: str):
            result = store.get(job_id)
            if result is None:
                raise HTTPException(404, "Job not found")
            return result

        @app.get("/studio/session/{session_id}")
        def session(session_id: str):
            result = store.get_session(session_id)
            if result is None:
                raise HTTPException(
                    404, "World expired; restore a saved snapshot or create a world"
                )
            return result

        @app.get("/studio/snapshots")
        def snapshots():
            rows = []
            for path in sorted(
                SNAPSHOTS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
            ):
                record = json.loads(path.read_text(encoding="utf-8"))
                rows.append(
                    {
                        key: record[key]
                        for key in ("id", "session_id", "title", "identity")
                    }
                )
            return rows

        def require_idle():
            if not store.worker().get("online") or store.active_job_count():
                raise HTTPException(409, "Wait for the loaded model to be idle")

        @app.post("/studio/snapshot")
        def snapshot(payload: dict):
            require_idle()
            session_id = str(payload.get("session_id", ""))
            if store.worker().get("active_session_id") != session_id:
                raise HTTPException(409, "This world is not resident")
            return store.submit(
                "snapshot",
                {
                    "session_id": session_id,
                    "snapshot_id": uuid.uuid4().hex,
                    "studio": payload.get("studio", {}),
                    "title": str(payload.get("title", "World"))[:160],
                },
            )

        @app.post("/studio/restore/{snapshot_id}")
        def restore(snapshot_id: str):
            require_idle()
            if len(snapshot_id) != 32 or any(
                c not in "0123456789abcdef" for c in snapshot_id
            ):
                raise HTTPException(400, "Invalid snapshot")
            if store.worker().get("active_session_id"):
                raise HTTPException(409, "Release the current world before restoring")
            path = SNAPSHOTS / f"{snapshot_id}.json"
            if not path.is_file():
                raise HTTPException(404, "Snapshot not found")
            record = json.loads(path.read_text(encoding="utf-8"))
            if record["identity"] != model_identity(processes.adapter):
                raise HTTPException(409, "Load the snapshot's original model/adapter")
            return store.submit(
                "restore",
                {"session_id": record["session_id"], "snapshot_id": snapshot_id},
            )

        app.mount(
            "/vendor", StaticFiles(directory=ROOT / "node_modules"), name="vendor"
        )
        app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="studio")

        class StudioServer(uvicorn.Server):
            def handle_exit(self, sig, frame):
                super().handle_exit(sig, frame)
                # Background PE responses must not delay release until lifespan exit.
                threading.Thread(target=processes.stop, daemon=True).start()

        server = StudioServer(
            uvicorn.Config(
                app,
                host=args.host,
                port=args.port,
                workers=1,
                timeout_graceful_shutdown=5,
            )
        )
        server.run()


if __name__ == "__main__":
    main()
