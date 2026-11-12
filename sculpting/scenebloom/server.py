import atexit
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory
from waitress import create_server
from werkzeug.exceptions import HTTPException
from state import (
    ROOT,
    WORK,
    LANE_STAMP,
    STAGES,
    DEFAULTS,
    atomic,
    read,
    job_path,
    settings,
    require_models,
)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 256 * 1024
lock = threading.RLock()
closing = threading.Event()
active = child = runner = None
PUBLIC = {
    "spec.json",
    "references.json",
    "plan.json",
    "lifted.json",
    "compiled.json",
    "scene_layout.json",
    "material_assignment.json",
    "scene.glb",
    "run.json",
}
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
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data:; connect-src 'self' blob:; object-src 'none'; frame-ancestors 'none'"
    )
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


@app.get("/vendor/<kind>/<path:name>")
def vendor(kind, name):
    if kind not in VENDOR:
        abort(404)
    return send_from_directory(ROOT / "node_modules" / VENDOR[kind], name)


def allowed(job, name):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        return False
    path = job / name
    if not path.is_file() or not path.resolve().is_relative_to(job.resolve()):
        return False
    part = path
    while part != job:
        if part.is_symlink():
            return False
        part = part.parent
    stage = {
        "spec.json": "retrieve",
        "references.json": "retrieve",
        "plan.json": "zone",
        "lifted.json": "proposals",
        "compiled.json": "compile",
        "scene_layout.json": "complete",
        "material_assignment.json": "materials",
        "scene.glb": "export",
    }.get(name)
    if name.startswith("proposals/"):
        stage = "proposals"
    elif name.startswith("renders/"):
        stage = "render"
    if stage and not (job / f"{stage}.done.json").is_file():
        return False
    return name in PUBLIC or (
        name.startswith(("proposals/", "renders/")) and path.suffix == ".png"
    )


def snapshot(job):
    data = read(job / "run.json")
    if not data:
        abort(404)
    files = [name for name in PUBLIC if allowed(job, name)]
    for folder in ("renders", "proposals"):
        files += [
            p.relative_to(job).as_posix()
            for p in (job / folder).rglob("*.png")
            if allowed(job, p.relative_to(job).as_posix())
        ]
    return {
        **data,
        "done": [s for s in STAGES if (job / f"{s}.done.json").is_file()],
        "files": sorted(files),
    }


@app.get("/api/config")
def config():
    with lock:
        return jsonify(defaults=DEFAULTS, stages=STAGES, active=active, source=LANE_STAMP)


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
    for sig, delay in ((signal.SIGTERM, 4), (signal.SIGKILL, 0)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + delay
        while time.monotonic() < deadline:
            process.poll()
            try:
                os.killpg(process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
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
            for stage in STAGES:
                if (job / f"{stage}.done.json").is_file():
                    continue
                with lock:
                    if closing.is_set() or (job / "cancel").exists():
                        break
                    update(job, stage=stage)
                    child = process = subprocess.Popen(
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
                if not success:
                    break
        stopped = closing.is_set() or (job / "cancel").exists()
        complete = all((job / f"{s}.done.json").is_file() for s in STAGES)
        with lock:
            update(
                job,
                status="cancelled" if stopped else "complete" if complete else "failed",
                error=None
                if stopped or complete
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
    data = request.get_json()
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("prompt"), str)
        or not 1 <= len(data["prompt"].strip()) <= 12000
    ):
        raise ValueError("Enter a room prompt (up to 12,000 characters)")
    options = settings(data.get("settings", {}))
    require_models()
    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active run or cancel it"), 409
        job = job_path(uuid.uuid4().hex)
        job.mkdir()
        atomic(
            job / "run.json",
            dict(
                id=job.name,
                prompt=data["prompt"].strip(),
                settings=options,
                title=data["prompt"].strip()[:100],
                created=time.time(),
                status="running",
                stage=STAGES[0],
                source=LANE_STAMP,
                error=None,
            ),
        )
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
            return jsonify(error="Resume a stopped run while the studio is idle"), 409
        require_models()
        for name in ("cancel", "error.json"):
            (job / name).unlink(missing_ok=True)
        update(job, status="running", error=None)
        launch(job)
        return jsonify(snapshot(job)), 202


@app.post("/api/runs/<key>/cancel")
def cancel(key):
    with lock:
        if active == key:
            (job_path(key) / "cancel").touch()
    return jsonify(ok=True)


@app.get("/api/runs/<key>/log")
def log(key):
    path = job_path(key) / "worker.log"
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 64000))
        value = stream.read().decode("utf-8", errors="replace")
    return value, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<path:name>")
def files(key, name):
    job = job_path(key)
    if not allowed(job, name):
        abort(404)
    return send_from_directory(
        job, name, conditional=True, as_attachment=request.args.get("download") == "1"
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
        if (
            path.is_dir()
            and not path.is_symlink()
            and read(path / "run.json", {}).get("status") == "running"
        ):
            update(path, status="interrupted", error="Saved stages can be resumed")
    atexit.register(close)

    def interrupted(signum, frame):
        raise SystemExit(0)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)
    port = int(os.environ.get("SCENEBLOOM_PORT", "8285"))
    http = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=6,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"SceneBloom studio: http://127.0.0.1:{port}", flush=True)
    try:
        http.run()
    finally:
        http.close()
        close()
