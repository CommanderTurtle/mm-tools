import atexit
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
import warnings

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from waitress import create_server
from werkzeug.exceptions import HTTPException

from state import (
    ROOT,
    WORK,
    PIN,
    DEFAULTS,
    MESH_TYPES,
    atomic,
    read,
    job_path,
    settings,
    require_models,
    coarse_ready,
)
from state import LANE_STAMP

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024 * 1024
lock = threading.RLock()
closing = threading.Event()
active = child = runner = None
PUBLIC = {
    "original",
    "input.png",
    "condition.png",
    "prepared.png",
    "run.json",
    "result.json",
}
PUBLIC |= {
    f"{stem}.{ext}" for stem in ("coarse", "refined") for ext in ("glb", "ply", "obj")
}
PUBLIC |= {f"input{ext}" for ext in MESH_TYPES}
VENDOR = {
    "three": "three/build",
    "addons": "three/examples/jsm",
    "inter": "@fontsource/inter/files",
    "mono": "@fontsource/ibm-plex-mono/files",
}


@app.before_request
def local_only():
    if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
        abort(403)
    if request.method != "GET" and request.headers.get("Origin") not in (
        None,
        request.host_url.rstrip("/"),
    ):
        abort(403)


@app.after_request
def headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; "
        "connect-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
    )
    if request.path.startswith("/api/") or request.path.startswith("/files/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(ValueError)
def invalid(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(HTTPException)
def http_error(error):
    return jsonify(error=error.description), error.code


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.get("/web/<name>")
def web(name):
    return send_from_directory(ROOT / "web", name)


@app.get("/vendor/<kind>/<path:name>")
def vendor(kind, name):
    if kind not in VENDOR:
        abort(404)
    return send_from_directory(ROOT / "node_modules" / VENDOR[kind], name)


def snapshot(job):
    meta = read(job / "run.json")
    if not meta:
        abort(404)
    coarse = read(job / "coarse.json")
    return {
        **meta,
        "progress": read(job / "progress.json", {}),
        "coarse": coarse,
        "prepared": bool(read(job / "prepare.json")),
        "result": read(job / "result.json"),
        "files": sorted(
            name
            for name in PUBLIC
            if (job / name).is_file() and not (job / name).is_symlink()
        ),
    }


@app.get("/api/config")
def config():
    with lock:
        return jsonify(
            defaults=DEFAULTS, active=active, source=LANE_STAMP, coarse=coarse_ready()
        )


@app.get("/api/runs")
def runs():
    with lock:
        rows = [
            read(p / "run.json")
            for p in WORK.iterdir()
            if p.is_dir() and not p.is_symlink()
        ]
        return jsonify(
            sorted((r for r in rows if r), key=lambda r: r["created"], reverse=True)
        )


@app.get("/api/runs/<key>")
def get_run(key):
    with lock:
        return jsonify(snapshot(job_path(key)))


def update(job, **values):
    atomic(job / "run.json", {**read(job / "run.json"), **values})


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def execute(job):
    global active, child
    process = None
    try:
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            DO_NOT_TRACK="1",
            OTEL_SDK_DISABLED="true",
            WANDB_DISABLED="true",
        )
        # Stage receipts are written only after artifacts are durable. A fresh
        # worker for each stage avoids retaining the first generator's weights.
        stages = []
        if not read(job / "prepare.json"):
            stages.append("prepare")
        if read(job / "run.json")["settings"]["route"] == "image" and not read(
            job / "coarse.json"
        ):
            stages.append("coarse")
        stages.append("refine")
        complete = False
        with (job / "worker.log").open("ab", buffering=0) as log:
            for stage in stages:
                with lock:
                    if closing.is_set() or (job / "cancel").exists():
                        break
                    update(job, stage=stage)
                    process = child = subprocess.Popen(
                        [
                            sys.executable,
                            str(ROOT / "worker.py"),
                            job.name,
                            stage,
                            str(os.getpid()),
                        ],
                        cwd=ROOT,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                while process.poll() is None:
                    if closing.wait(0.2) or (job / "cancel").exists():
                        stop_group(process)
                        break
                success = process.wait() == 0
                stop_group(process)
                process = None
                with lock:
                    child = None
                if not success or closing.is_set() or (job / "cancel").exists():
                    break
                complete = stage == "refine" and (job / "result.json").is_file()
        cancelled = closing.is_set() or (job / "cancel").exists()
        with lock:
            update(
                job,
                status="cancelled"
                if cancelled
                else "complete"
                if complete
                else "failed",
                error=None
                if cancelled or complete
                else read(job / "error.json", {}).get(
                    "error", "Stage stopped; open the run log"
                ),
            )
    except Exception as error:
        with lock:
            update(job, status="failed", error=str(error))
    finally:
        stop_group(process)
        with lock:
            active = child = None


def launch(job):
    global active, runner
    active = job.name
    runner = threading.Thread(target=execute, args=(job,), daemon=True)
    runner.start()


@app.post("/api/runs")
def generate():
    config = settings(json.loads(request.form.get("settings", "{}")))
    require_models(config)
    with lock:
        if active or closing.is_set():
            return jsonify(error="A run is active; wait or cancel it"), 409
        image = request.files.get("image")
        mesh = request.files.get("mesh")
        if not image:
            raise ValueError("Choose a reference image")
        suffix = Path(mesh.filename or "").suffix.lower() if mesh else ""
        if config["route"] == "mesh" and (not mesh or suffix not in MESH_TYPES):
            raise ValueError("Choose a standalone GLB, PLY, OBJ or STL mesh")
        job = job_path(uuid.uuid4().hex)
        job.mkdir()
        try:
            image.save(job / "original")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(job / "original") as source:
                    if source.width * source.height > 40_000_000:
                        raise ValueError("Image must be at most 40 megapixels")
                    ImageOps.exif_transpose(source).convert("RGBA").save(
                        job / "input.png"
                    )
            mesh_name = "input" + suffix if config["route"] == "mesh" else None
            if mesh_name:
                mesh.save(job / mesh_name)
            atomic(
                job / "run.json",
                dict(
                    id=job.name,
                    title=Path(image.filename or "Reference").name[:120],
                    created=time.time(),
                    status="running",
                    settings=config,
                    mesh_input=mesh_name,
                    source=LANE_STAMP,
                    stage="prepare",
                    error=None,
                ),
            )
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            OSError,
        ) as error:
            shutil.rmtree(job)
            raise ValueError("Could not read upload: " + str(error)) from error
        except Exception:
            shutil.rmtree(job)
            raise
        launch(job)
        return jsonify(snapshot(job)), 202


@app.post("/api/runs/<key>/resume")
def resume(key):
    with lock:
        job = job_path(key)
        data = snapshot(job)
        if (
            active
            or closing.is_set()
            or data["status"] not in ("failed", "cancelled", "interrupted")
        ):
            return jsonify(error="Resume a stopped run when the studio is idle"), 409
        require_models(data["settings"])
        for name in (
            "error.json",
            "cancel",
            "result.json",
            "refined.json",
            "progress.json",
        ):
            (job / name).unlink(missing_ok=True)
        update(job, status="running", error=None)
        launch(job)
        return jsonify(snapshot(job)), 202


@app.post("/api/runs/<key>/cancel")
def cancel(key):
    with lock:
        job = job_path(key)
        if active == key:
            (job / "cancel").touch()
    return jsonify(ok=True)


@app.get("/api/runs/<key>/log")
def log(key):
    path = job_path(key) / "worker.log"
    if not path.is_file():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64000))
        value = handle.read().decode("utf-8", errors="replace")
    return value, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<name>")
def files(key, name):
    path = job_path(key)
    if name not in PUBLIC or (path / name).is_symlink():
        abort(404)
    return send_from_directory(
        path,
        name,
        conditional=True,
        as_attachment=name == "original" or request.args.get("download") == "1",
    )


def close():
    closing.set()
    if runner:
        runner.join(timeout=6)
    stop_group(child)


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    instance = (WORK / ".server.lock").open("a")
    try:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This studio workspace is already serving")
    for path in WORK.iterdir():
        if path.is_dir() and not path.is_symlink():
            meta = read(path / "run.json", {})
            if meta.get("status") == "running":
                update(
                    path,
                    status="interrupted",
                    error="Studio stopped; saved stages can be resumed",
                )
    atexit.register(close)

    def interrupted(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    port = int(os.environ.get("HY3D_PORT", "8284"))
    http = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=6,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"Hy3D studio: http://127.0.0.1:{port}", flush=True)
    try:
        http.run()
    finally:
        http.close()
        close()
