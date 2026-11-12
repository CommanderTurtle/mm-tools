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
    MODELS,
    WORK,
    LANE_STAMP,
    DEFAULTS,
    atomic,
    read,
    job_path,
    presets,
    require_models,
    settings,
)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
lock = threading.RLock()
closing = threading.Event()
active = None
child = None
runner = None
PUBLIC = {
    "input.png",
    "original",
    "prepared.png",
    "object.ply",
    "camera.json",
    "run.json",
    "result.json",
}
VENDOR = {
    "three": "three/build",
    "splats": "@mkkellogg/gaussian-splats-3d/build",
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
    if request.path.startswith("/api/"):
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


@app.get("/web/<path:name>")
def web(name):
    return send_from_directory(ROOT / "web", name)


@app.get("/vendor/<kind>/<name>")
def vendor(kind, name):
    if kind not in VENDOR:
        abort(404)
    return send_from_directory(ROOT / "node_modules" / VENDOR[kind], name)


@app.get("/examples/<name>")
def examples(name):
    if Path(name).suffix not in (".webp", ".ply"):
        abort(404)
    return send_from_directory(MODELS / "demo/examples", name, conditional=True)


def snapshot(job):
    meta = read(job / "run.json")
    if not meta:
        abort(404)
    return {
        **meta,
        "progress": read(job / "progress.json", {}),
        "result": read(job / "result.json"),
    }


@app.get("/api/config")
def config():
    with lock:
        return jsonify(defaults=DEFAULTS, presets=presets(), active=active, source=LANE_STAMP)


@app.get("/api/runs")
def runs():
    with lock:
        rows = [
            read(path / "run.json")
            for path in WORK.iterdir()
            if path.is_dir() and not path.is_symlink()
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
        with (job / "worker.log").open("ab", buffering=0) as log:
            with lock:
                if closing.is_set() or (job / "cancel").exists():
                    update(job, status="cancelled")
                    return
                process = child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "worker.py"),
                        job.name,
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
            cancelled = closing.is_set() or (job / "cancel").exists()
            complete = process.wait() == 0 and (job / "result.json").is_file()
            with lock:
                update(
                    job,
                    status="cancelled"
                    if cancelled
                    else "complete"
                    if complete
                    else "failed",
                    error=None
                    if complete or cancelled
                    else read(job / "error.json", {}).get(
                        "error", "Generation stopped; open the run log"
                    ),
                )
    except Exception as error:
        with lock:
            update(job, status="failed", error=str(error))
    finally:
        stop_group(process)
        with lock:
            child = None
            active = None


@app.post("/api/runs")
def generate():
    global active, runner
    config = settings(json.loads(request.form.get("settings", "{}")))
    require_models()
    with lock:
        if active or closing.is_set():
            return jsonify(
                error="A generation is already running; wait or cancel it"
            ), 409
        upload = request.files.get("image")
        preset = request.form.get("preset")
        selected = next((p for p in presets() if p["image"] == preset), None)
        if not upload and not selected:
            raise ValueError("Upload an image or choose an example")
        job = job_path(uuid.uuid4().hex)
        job.mkdir()
        try:
            if upload:
                upload.save(job / "original")
                title = Path(upload.filename or "Image").name[:120]
            else:
                source = (MODELS / "demo" / selected["image"]).resolve()
                if not source.is_relative_to((MODELS / "demo/examples").resolve()):
                    raise ValueError("Example must belong to the pinned collection")
                shutil.copyfile(source, job / "original")
                title = selected["title"]
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(job / "original") as image:
                    if image.width * image.height > 40_000_000:
                        raise ValueError("Input must be at most 40 megapixels")
                    ImageOps.exif_transpose(image).convert("RGBA").save(
                        job / "input.png"
                    )
            atomic(
                job / "run.json",
                dict(
                    id=job.name,
                    title=title,
                    created=time.time(),
                    status="running",
                    settings=config,
                    source=LANE_STAMP,
                ),
            )
        except (
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
            OSError,
        ) as error:
            shutil.rmtree(job)
            raise ValueError("Could not read image: " + str(error)) from error
        except Exception:
            shutil.rmtree(job)
            raise
        active = job.name
        runner = threading.Thread(target=execute, args=(job,), daemon=True)
        runner.start()
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
    if name not in PUBLIC:
        abort(404)
    return send_from_directory(
        job_path(key),
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
                    error="Studio stopped; input and settings were retained",
                )
    atexit.register(close)

    def interrupted(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    port = int(os.environ.get("IDOL_PORT", "8283"))
    http = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=6,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"Idol studio: http://127.0.0.1:{port}", flush=True)
    try:
        http.run()
    finally:
        http.close()
        close()
