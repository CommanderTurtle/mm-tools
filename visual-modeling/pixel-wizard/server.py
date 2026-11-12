import atexit
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory
from waitress import create_server
from werkzeug.exceptions import HTTPException

from state import (ROOT, WORK, RUNS, MODELS, PIN, DEFAULTS, TASKS, EXAMPLES,
                   atomic, read, run_path, options, require_models, offline_env)
from state import LANE_STAMP

app = Flask(__name__, static_folder=None)
app.json.sort_keys = False
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024**2
lock = threading.RLock()
closing = threading.Event()
active = child = runner = None


@app.before_request
def local_only():
    if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
        abort(403)
    if request.method != "GET" and request.headers.get("Origin") not in (None, request.host_url.rstrip("/")):
        abort(403)


@app.after_request
def headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'")
    if request.path.startswith(("/api/", "/files/")):
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


@app.get("/examples/<name>")
def example(name):
    if name not in {row["image"] for row in EXAMPLES}:
        abort(404)
    return send_from_directory(MODELS / "examples", name)


def snapshot(job):
    meta = read(job / "run.json")
    if not meta:
        abort(404)
    return dict(meta, image=read(job / "image.json"),
                progress=read(job / "progress.json", {}), result=read(job / "result.json"))


def update(job, **fields):
    atomic(job / "run.json", read(job / "run.json") | fields)


def terminate(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=6)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def execute(job):
    global active, child
    process = None
    try:
        with (job / "worker.log").open("ab", buffering=0) as log:
            with lock:
                if closing.is_set() or (job / "cancel").exists():
                    update(job, status="cancelled")
                    return
                process = child = subprocess.Popen(
                    [sys.executable, str(ROOT / "supervisor.py"), sys.executable,
                     str(ROOT / "worker.py"), job.name], cwd=ROOT, env=offline_env(),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                if closing.wait(0.2) or (job / "cancel").exists():
                    terminate(process)
                    break
            success = process.wait() == 0
            cancelled = closing.is_set() or (job / "cancel").exists()
            with lock:
                update(job, status="cancelled" if cancelled else "complete" if success else "failed",
                       error=None if success or cancelled else read(job / "error.json", {}).get(
                           "error", "Worker stopped; open the log"))
    except Exception as error:
        with lock:
            update(job, status="failed", error=str(error))
    finally:
        terminate(process)
        with lock:
            active = child = None


@app.get("/api/config")
def config():
    with lock:
        return jsonify(defaults=DEFAULTS, tasks=TASKS, examples=EXAMPLES, active=active, source=LANE_STAMP)


@app.get("/api/runs")
def runs():
    with lock:
        rows = [read(p / "run.json") for p in RUNS.iterdir() if p.is_dir() and not p.is_symlink()]
        return jsonify(sorted((r for r in rows if r), key=lambda r: r["created"], reverse=True))


@app.get("/api/runs/<key>")
def get_run(key):
    with lock:
        return jsonify(snapshot(run_path(key)))


@app.post("/api/runs")
def upload():
    global active, runner
    cfg = options(json.loads(request.form.get("settings", "{}")))
    require_models()
    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active operation or stop it"), 409
        file = request.files.get("image")
        reuse = request.form.get("reuse")
        example_name = request.form.get("example")
        source = None
        title = "Image"
        if file:
            title = Path(file.filename or title).name[:160]
        elif reuse:
            original = run_path(reuse)
            title = snapshot(original)["title"]
            source = original / "upload"
        elif example_name in {r["image"] for r in EXAMPLES}:
            source = MODELS / "examples" / example_name
            title = example_name
        else:
            raise ValueError("Choose an image")
        job = run_path(uuid.uuid4().hex)
        job.mkdir(parents=True)
        try:
            if file:
                file.save(job / "upload")
            else:
                shutil.copyfile(source, job / "upload")
            atomic(job / "run.json", dict(id=job.name, title=title, created=time.time(),
                                         settings=cfg, status="running", source=LANE_STAMP))
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
        job = run_path(key)
        if active == key:
            (job / "cancel").touch()
    return jsonify(ok=True)


@app.delete("/api/runs/<key>")
def delete(key):
    with lock:
        job = run_path(key)
        snapshot(job)
        if active == key:
            return jsonify(error="Stop the run first"), 409
        shutil.rmtree(job)
    return jsonify(ok=True)


@app.get("/api/runs/<key>/log")
def log(key):
    job = run_path(key)
    snapshot(job)
    path = job / "worker.log"
    text = ""
    if path.exists():
        with path.open("rb") as source:
            source.seek(max(0, path.stat().st_size - 64000))
            text = source.read().decode("utf-8", errors="replace")
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<path:name>")
def files(key, name):
    job = run_path(key)
    snapshot(job)
    allowed = name in {"input.png", "result.zip", "response.txt"}
    allowed |= bool(re.fullmatch(r"output/(?:response\.txt|result\.json|annotations\.json|settings\.json|overlay\.png|alpha\.png|cutout\.png|masks/mask_\d+\.png)", name))
    path = job / name
    if not allowed or path.is_symlink() or not path.resolve().is_relative_to(job.resolve()):
        abort(404)
    return send_from_directory(job, name, as_attachment=request.args.get("download") == "1")


def close():
    closing.set()
    if runner:
        runner.join(timeout=8)
    terminate(child)


if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    instance = (WORK / ".server.lock").open("a")
    try:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This studio workspace is already serving") from None
    for job in RUNS.iterdir():
        if job.is_dir() and not job.is_symlink() and read(job / "run.json", {}).get("status") == "running":
            update(job, status="interrupted", error="Studio stopped; input and partial output remain saved")
    atexit.register(close)

    def interrupted(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    port = int(os.environ.get("PIXEL_PORT", "8288"))
    server = create_server(app, host="127.0.0.1", port=port, threads=6,
                           max_request_body_size=app.config["MAX_CONTENT_LENGTH"])
    print(f"Pixel Wizard: http://127.0.0.1:{port}", flush=True)
    try:
        server.run()
    finally:
        server.close()
        close()
