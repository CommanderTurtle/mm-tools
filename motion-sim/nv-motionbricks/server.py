"""Local browser bridge. Native GPU state belongs to one disposable child process."""

import asyncio
from contextlib import asynccontextmanager
import json
import multiprocessing as mp
import os
from pathlib import PurePosixPath
import tempfile
import time
from urllib.parse import urlparse
import uuid
import zipfile

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from storage import (
    ROOT,
    WORK,
    PIN,
    catalog,
    read_document,
    refs,
    resolve,
    virtual,
    write_document,
)
from storage import LANE_STAMP


class Session:
    def __init__(self):
        self.process = None
        self.pipe = None
        self.lock = asyncio.Lock()
        self.owner = None

    async def close(self):
        process, pipe = self.process, self.pipe
        self.process = self.pipe = None
        if process is not None:
            if process.is_alive():
                process.terminate()
            await asyncio.to_thread(process.join, 2)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 2)
            process.close()
        if pipe is not None:
            pipe.close()

    async def execute(self, message):
        async with self.lock:
            if message.get("op") == "load":
                await self.close()
                from worker import serve

                context = mp.get_context("spawn")
                self.pipe, child = context.Pipe()
                self.process = context.Process(target=serve, args=(child,), daemon=True)
                self.process.start()
                child.close()
            if self.pipe is None:
                raise ValueError("Load a rig first.")
            pipe, process = self.pipe, self.process
            await asyncio.to_thread(pipe.send, message)
            started = time.monotonic()
            while not pipe.poll():
                if self.pipe is not pipe or not process.is_alive():
                    raise ValueError("Native session stopped.")
                if time.monotonic() - started > 1800:
                    await self.close()
                    raise ValueError("Native operation timed out; session released.")
                await asyncio.sleep(0.005)
            return await asyncio.to_thread(pipe.recv)


session = Session()


@asynccontextmanager
async def lifespan(app):
    WORK.mkdir(parents=True, exist_ok=True)
    yield
    await session.close()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


def same_origin(headers):
    origin = headers.get("origin")
    return not origin or urlparse(origin).netloc == headers.get("host")


@app.middleware("http")
async def guard(request: Request, call_next):
    if not same_origin(request.headers):
        return Response("Cross-origin request refused.", status_code=403)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response


@app.exception_handler(ValueError)
async def invalid(request, error):
    return Response(str(error), status_code=400, media_type="text/plain")


@app.get("/api/catalog")
def list_assets():
    return {"files": catalog(), "source": LANE_STAMP}


@app.get("/api/document")
def document(path: str):
    return read_document(path)


@app.post("/api/save")
async def save(request: Request):
    data = await request.json()
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("document"), dict)
        or "path" not in data
    ):
        raise ValueError("Provide a path and document object.")
    path = write_document(data["path"], data["document"])
    return {"path": virtual(path)}


@app.post("/api/upload")
async def upload(files: list[UploadFile]):
    folder = WORK / "imports" / uuid.uuid4().hex[:12]
    result = []
    for file in files:
        name = PurePosixPath((file.filename or "upload").replace("\\", "/"))
        if (
            name.is_absolute()
            or ".." in name.parts
            or name.suffix.lower() not in (".json", ".bvh", ".csv")
        ):
            raise ValueError("Import JSON, BVH or CSV files with relative paths.")
        path = folder.joinpath(*name.parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".partial")
        try:
            with partial.open("wb") as stream:
                size = 0
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > 512 * 1024 * 1024:
                        raise ValueError("A clip must fit within 512 MiB.")
                    stream.write(chunk)
            partial.replace(path)
            result.append(virtual(path))
        finally:
            partial.unlink(missing_ok=True)
            await file.close()
    return {"paths": result}


@app.get("/api/download")
def download(path: str):
    target = resolve(path)
    if not target.is_file():
        raise HTTPException(404)
    return FileResponse(
        target, filename=target.name, media_type="application/octet-stream"
    )


@app.get("/api/package")
def package(path: str):
    entry = resolve(path)
    exports = WORK / ".exports"
    exports.mkdir(parents=True, exist_ok=True)
    archive = tempfile.NamedTemporaryFile(dir=exports, suffix=".zip", delete=False)
    archive.close()
    pending, visited = [entry], set()
    try:
        with zipfile.ZipFile(archive.name, "w", zipfile.ZIP_DEFLATED) as output:
            while pending:
                item = pending.pop()
                if item in visited:
                    continue
                visited.add(item)
                name = virtual(item)
                if item.suffix == ".json":
                    data = read_document(name)

                    def convert(ref):
                        target = resolve(ref)
                        pending.append(target)
                        return os.path.relpath(
                            virtual(target), str(PurePosixPath(name).parent)
                        ).replace("\\", "/")

                    output.writestr(
                        name, json.dumps(refs(data, convert), indent=2, allow_nan=False)
                    )
                else:
                    output.write(item, name)
    except BaseException:
        os.unlink(archive.name)
        raise
    return FileResponse(
        archive.name,
        media_type="application/zip",
        filename=entry.stem + ".zip",
        background=BackgroundTask(os.unlink, archive.name),
    )


@app.websocket("/api/session")
async def socket(ws: WebSocket):
    if not same_origin(ws.headers) or session.owner is not None:
        await ws.close(
            code=1008, reason="A native session is already open, or origin differs."
        )
        return
    session.owner = ws
    task = None

    async def run(message):
        try:
            result = await session.execute(message)
            await ws.send_json({"id": message.get("id"), **result})
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await ws.send_json({"id": message.get("id"), "error": str(error)})

    try:
        await ws.accept()
        while True:
            message = await ws.receive_json()
            if not isinstance(message, dict):
                await ws.send_json({"error": "Expected an operation object."})
                continue
            if message.get("op") == "unload":
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
                await session.close()
                await ws.send_json({"id": message.get("id"), "unloaded": True})
            elif task and not task.done():
                await ws.send_json(
                    {"id": message.get("id"), "error": "Another operation is active."}
                )
            else:
                task = asyncio.create_task(run(message))
    except WebSocketDisconnect:
        pass
    finally:
        if task:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await session.close()
        session.owner = None


app.mount(
    "/vendor/three",
    StaticFiles(directory=ROOT / "node_modules/three", check_dir=False),
    name="three",
)
app.mount("/", StaticFiles(directory=ROOT / "web", html=True), name="studio")
