import atexit
import fcntl
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory
from state import (
    DEFAULTS,
    LANGUAGES,
    MODELS,
    PIN,
    PRESETS,
    ROOT,
    RUNS,
    TASKS,
    WORK,
    atomic,
    offline_env,
    options,
    read,
    require_models,
    run_path,
)
from waitress import create_server
from werkzeug.exceptions import HTTPException
from state import LANE_STAMP

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 128 * 1024**2
lock = threading.RLock()
closing = threading.Event()
active = child = runner = None


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
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; media-src 'self' blob:; connect-src 'self'; "
        "object-src 'none'; frame-ancestors 'none'"
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


def snapshot(job):
    meta = read(job / "run.json")
    if not meta:
        abort(404)
    return dict(
        meta,
        progress=read(job / "progress.json", {}),
        result=read(job / "result.json"),
        partial=read(job / "partial.json", []),
    )


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
                child = process = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "supervisor.py"),
                        str(
                            ROOT
                            / f".venv-{read(job / 'run.json')['settings']['model']}/bin/python"
                        ),
                        str(ROOT / "worker.py"),
                        job.name,
                    ],
                    cwd=ROOT,
                    env=offline_env(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            while process.poll() is None:
                if closing.wait(0.2) or (job / "cancel").exists():
                    terminate(process)
                    break
            success = process.wait() == 0
            cancelled = closing.is_set() or (job / "cancel").exists()
            with lock:
                update(
                    job,
                    status="cancelled"
                    if cancelled
                    else "complete"
                    if success
                    else "failed",
                    error=None
                    if success or cancelled
                    else read(job / "error.json", {}).get(
                        "error", "Worker stopped; open the log"
                    ),
                )
    except Exception as error:
        with lock:
            update(job, status="failed", error=str(error))
    finally:
        terminate(process)
        with lock:
            child = active = None


@app.get("/api/config")
def config():
    with lock:
        return jsonify(
            defaults=DEFAULTS,
            presets=PRESETS,
            voices=read(MODELS / "references/voices.json", {}),
            tasks=TASKS,
            installed={
                key: (ROOT / f".venv-{key}/bin/python").is_file()
                and (MODELS / key / "config.json").is_file()
                for key in PIN["models"]
            },
            languages=LANGUAGES,
            source=LANE_STAMP,
            active=active,
        )


@app.get("/voices/<key>")
def voice(key):
    if key not in PRESETS:
        abort(404)
    return send_from_directory(MODELS / "references", PRESETS[key]["file"])


@app.get("/api/runs")
def runs():
    with lock:
        rows = [
            read(p / "run.json")
            for p in RUNS.iterdir()
            if p.is_dir() and not p.is_symlink()
        ]
        return jsonify(
            sorted((r for r in rows if r), key=lambda r: r["created"], reverse=True)
        )


@app.get("/api/runs/<key>")
def get_run(key):
    with lock:
        return jsonify(snapshot(run_path(key)))


@app.post("/api/runs")
def generate():
    global active, runner
    cfg = options(json.loads(request.form.get("settings", "{}")))
    require_models(cfg["model"])
    if not (ROOT / f".venv-{cfg['model']}/bin/python").is_file():
        raise ValueError("Run setupwithuv.sh after downloading this model")
    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active take or cancel it"), 409
        job = run_path(uuid.uuid4().hex)
        job.mkdir(parents=True)
        try:
            for number, speaker in enumerate(cfg["speakers"], 1):
                target = job / f"upload-{number}"
                if speaker["voice"] in PRESETS:
                    shutil.copyfile(
                        MODELS / "references" / PRESETS[speaker["voice"]]["file"],
                        target,
                    )
                elif file := request.files.get(f"reference-{number}"):
                    file.save(target)
                else:
                    reuse = request.form.get(f"reuse-{number}", "")
                    if not reuse:
                        raise ValueError(f"Upload or record speaker {number}")
                    old_key, _, old_number = reuse.partition(":")
                    if not re.fullmatch(r"[1-9]|1[0-6]", old_number):
                        raise ValueError("Invalid saved reference")
                    original = run_path(old_key)
                    snapshot(original)
                    source = original / f"upload-{old_number}"
                    if not source.is_file() or source.is_symlink():
                        raise ValueError(f"Upload or record speaker {number}")
                    shutil.copyfile(source, target)
            atomic(
                job / "run.json",
                dict(
                    id=job.name,
                    title=cfg["text"].strip()[:80],
                    created=time.time(),
                    settings=cfg,
                    status="running",
                    source=LANE_STAMP,
                ),
            )
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
            return jsonify(error="Stop the take first"), 409
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


@app.get("/files/<key>/<name>")
def files(key, name):
    job = run_path(key)
    snapshot(job)
    allowed = name in {
        "speech.wav",
        "speech.flac",
        "result.zip",
        "manifest.json",
        "run.json",
    } or re.fullmatch(r"reference-([1-9]|1[0-6])\.wav", name)
    path = job / name
    if (
        not allowed
        or path.is_symlink()
        or not path.resolve().is_relative_to(job.resolve())
    ):
        abort(404)
    return send_from_directory(
        job, name, as_attachment=request.args.get("download") == "1"
    )


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
        if (
            job.is_dir()
            and not job.is_symlink()
            and read(job / "run.json", {}).get("status") == "running"
        ):
            update(
                job,
                status="interrupted",
                error="Studio stopped; inputs and partial takes remain saved",
            )
    atexit.register(close)

    def interrupted(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    port = int(os.environ.get("MING3_PORT", "8291"))
    server = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=6,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"Ming3: http://127.0.0.1:{port}", flush=True)
    try:
        server.run()
    finally:
        server.close()
        close()
