"""Loopback studio with explicit stage admission and one owned worker at a time."""

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
from urllib.request import urlopen
import uuid

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps
from waitress import create_server

from state import (
    ROOT,
    WORK,
    SOURCE,
    PIN,
    DEFAULTS,
    atomic,
    read,
    settings,
    case_path,
    attempt_path,
    inside,
    attempts,
    latest,
    update,
    native_python,
)
from state import LANE_STAMP

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024
lock = threading.RLock()
closing = threading.Event()
active = None
child = None
runner = None
discovery = None
discovering = False
catalog = {}
PUBLIC = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".glb",
    ".blend",
    ".xml",
    ".json",
    ".npz",
    ".mp4",
    ".webm",
    ".zip",
    ".obj",
    ".mtl",
    ".stl",
    ".ply",
    ".bin",
    ".txt",
    ".py",
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


@app.errorhandler(ValueError)
def invalid(error):
    return jsonify(error=str(error)), 400


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    until = time.monotonic() + 4
    while time.monotonic() < until:
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


def snapshot(case):
    data = read(case / "case.json")
    if not data:
        abort(404)
    rows = attempts(case)
    for row in rows:
        path = attempt_path(case, row["id"])
        row["progress"] = read(path / "progress.json", {})
        row["delivery"] = read(path / "work/validated.json")
        row["zip"] = (path / "asset.zip").is_file()
        row["native_model"] = (
            row["status"] == "complete"
            and (path / "output/candidate_yup.glb").is_file()
        )
    return {**data, "attempts": rows}


@app.get("/")
def index():
    return send_from_directory(ROOT / "web", "index.html")


@app.get("/web/<path:name>")
def web(name):
    return send_from_directory(ROOT / "web", name)


@app.get("/vendor/three/<path:name>")
def vendor(name):
    return send_from_directory(ROOT / "node_modules/three", name)


@app.get("/api/config")
def config():
    return jsonify(
        defaults=DEFAULTS, source=LANE_STAMP, active=active, native_python=str(native_python())
    )


@app.post("/api/models")
def models():
    global discovery, discovering
    body = request.get_json(silent=True) or {}
    mode = body.get("mode", "local")
    if mode not in ("local", "nvidia"):
        raise ValueError("Choose local or NVIDIA mode")
    if mode == "nvidia" and body.get("ready") is not True:
        raise ValueError("Load the NVIDIA model on port 7999 and confirm it is ready")
    with lock:
        if active or discovering or closing.is_set():
            return jsonify(error="Finish the current stage before model discovery"), 409
        discovering = True
    process = None
    try:
        with lock:
            if closing.is_set():
                raise ValueError("Studio is closing")
            process = discovery = subprocess.Popen(
                [
                    os.environ.get("PHYS_OMP", "omp"),
                    "models",
                    "--json",
                    "--no-extensions",
                    "--config",
                    str(ROOT / "omp.yml"),
                ],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
                env=dict(os.environ, OTEL_SDK_DISABLED="true", DO_NOT_TRACK="1"),
            )
        stdout, stderr = process.communicate(timeout=30)
        if process.returncode:
            raise ValueError("OMP discovery failed: " + stderr[-2000:])
        rows = json.loads(stdout).get("models", [])
        rows = [
            m
            for m in rows
            if "image" in m.get("input", []) and isinstance(m.get("selector"), str)
        ]
        if mode == "nvidia":
            with urlopen("http://127.0.0.1:7999/v1/models", timeout=5) as response:
                ids = {r["id"] for r in json.load(response).get("data", [])}
            rows = [m for m in rows if m["id"] in ids]
        with lock:
            catalog[mode] = {m["selector"] for m in rows}
        return jsonify(
            models=rows,
            message=""
            if rows
            else "Configure this image-capable model in OMP, then refresh. NVIDIA IDs must match port 7999 exactly.",
        )
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        raise ValueError("Model discovery failed: " + str(error)) from error
    finally:
        stop_group(process)
        with lock:
            discovery = None
            discovering = False


@app.get("/api/cases")
def cases():
    with lock:
        rows = [
            snapshot(p)
            for p in WORK.iterdir()
            if p.is_dir() and not p.is_symlink() and (p / "case.json").is_file()
        ]
    return jsonify(sorted(rows, key=lambda r: r["created"], reverse=True))


@app.post("/api/cases")
def upload():
    file = request.files.get("image")
    if not file or not file.filename:
        raise ValueError("Choose an object image")
    key = uuid.uuid4().hex
    case = case_path(key)
    (case / "input").mkdir(parents=True)
    (case / "runs").mkdir()
    try:
        original = case / "input" / ("original" + Path(file.filename).suffix.lower())
        file.save(original)
        with Image.open(original) as image:
            if image.width * image.height > 40_000_000:
                raise ValueError("Use an image up to 40 megapixels")
            ImageOps.exif_transpose(image).convert("RGBA").save(
                case / "input/source.png"
            )
        atomic(
            case / "case.json",
            dict(
                id=key,
                title=Path(file.filename).stem[:160],
                created=time.time(),
                source=LANE_STAMP,
                original=original.name,
            ),
        )
        return jsonify(snapshot(case)), 201
    except Exception:
        shutil.rmtree(case)
        raise


@app.get("/api/cases/<key>")
def get_case(key):
    with lock:
        return jsonify(snapshot(case_path(key)))


def execute(case, attempt):
    global active, child
    try:
        meta = read(attempt / "run.json")
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
            HF_HUB_DISABLE_TELEMETRY="1",
            WANDB_DISABLED="true",
            DO_NOT_TRACK="1",
            OTEL_SDK_DISABLED="true",
        )
        with (attempt / "worker.log").open("ab", buffering=0) as log:
            with lock:
                if closing.is_set() or (attempt / "cancel").exists():
                    update(attempt, status="cancelled")
                    return
                child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "worker.py"),
                        str(case),
                        str(attempt),
                        str(os.getpid()),
                    ],
                    cwd=ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                process = child
            deadline = time.monotonic() + meta["settings"]["minutes"] * 60 + 60
            timed_out = False
            while process.poll() is None:
                timed_out = time.monotonic() >= deadline
                if closing.wait(0.2) or (attempt / "cancel").exists() or timed_out:
                    stop_group(process)
                    break
            cancelled = closing.is_set() or (attempt / "cancel").exists()
            result = read(attempt / "result.json", {})
            status = (
                "cancelled"
                if cancelled
                else result.get("status", "failed")
                if process.wait() == 0
                else "failed"
            )
            error = (
                "Stage time limit reached; files retained"
                if timed_out
                else read(attempt / "error.json", {}).get("error")
            )
            if status == "failed" and not error:
                error = "Stage stopped before completion; open its log and resume"
            with lock:
                update(attempt, status=status, error=error)
    except Exception as error:
        with lock:
            update(attempt, status="failed", error=str(error))
    finally:
        stop_group(child)
        with lock:
            child = None
            active = None


@app.post("/api/cases/<key>/run")
def run(key):
    global active, runner
    body = request.get_json(silent=True) or {}
    phase = body.get("phase")
    if phase not in ("plan", "native", "refine", "package"):
        raise ValueError("Choose a stage")
    config = settings(body.get("settings", {}))
    if phase != "package" and body.get("ready") is not True:
        raise ValueError("Confirm the model handoff before starting this stage")
    case = case_path(key)
    with lock:
        if active or discovering or closing.is_set():
            return jsonify(error="One stage runs at a time; cancel or wait"), 409
        if not (case / "case.json").is_file():
            abort(404)
        rows = attempts(case)
        selected = next((r for r in rows if r["id"] == body.get("attempt")), None)
        plan, native = latest(case, "plan"), latest(case, "native")
        if phase == "refine":
            if (
                selected
                and selected["phase"] == "native"
                and selected["status"] == "complete"
            ):
                native = selected
            elif selected and selected["phase"] in ("refine", "package"):
                native = next(
                    (r for r in rows if r["id"] == selected.get("native")), None
                )
            plan = next(
                (r for r in rows if native and r["id"] == native.get("plan")), None
            )
        if phase in ("native", "refine") and not plan:
            raise ValueError("Complete the plan and scaffold first")
        if phase == "refine" and not native:
            raise ValueError("Generate a native candidate first")
        if phase in ("plan", "refine") and config["model"] not in catalog.get(
            config["mode"], set()
        ):
            raise ValueError(
                "Refresh models and select an exact image-capable OMP model"
            )
        blender = shutil.which(config["blender"])
        if phase in ("plan", "refine") and not blender:
            raise ValueError("Blender executable was not found; set its path")
        if phase == "native" and (not native_python().is_file() or not SOURCE.is_dir()):
            raise ValueError("Run setupwithuv.sh first")
        previous = latest(
            case, "refine" if phase == "package" else phase, successful=False
        )
        if phase == "refine":
            previous = next(
                (
                    r
                    for r in reversed(rows)
                    if r["phase"] == "refine"
                    and r.get("native") == native["id"]
                    and r.get("plan") == plan["id"]
                ),
                None,
            )
            if selected and selected["phase"] == "refine":
                previous = selected
        if phase == "package":
            previous = selected
            if (
                not previous
                or previous["phase"] not in ("refine", "package")
                or not (attempt_path(case, previous["id"]) / "work").is_dir()
            ):
                raise ValueError("Select a saved refinement or package first")
            plan = next(
                (r for r in attempts(case) if r["id"] == previous.get("plan")), None
            )
            native = next(
                (r for r in attempts(case) if r["id"] == previous.get("native")), None
            )
        runid = uuid.uuid4().hex
        attempt = attempt_path(case, runid)
        attempt.mkdir()
        atomic(
            attempt / "run.json",
            dict(
                id=runid,
                phase=phase,
                status="running",
                created=time.time(),
                settings=config,
                blender=blender,
                plan=plan["id"] if plan else None,
                native=native["id"] if native else None,
                resume_from=previous["id"]
                if previous
                and (config["resume"] or phase == "package")
                and phase != "native"
                and (attempt_path(case, previous["id"]) / "work").is_dir()
                else None,
            ),
        )
        active = {"case": key, "attempt": runid}
        runner = threading.Thread(target=execute, args=(case, attempt), daemon=True)
        runner.start()
        return jsonify(snapshot(case)), 202


@app.post("/api/cancel")
def cancel():
    with lock:
        if active:
            (
                attempt_path(case_path(active["case"]), active["attempt"]) / "cancel"
            ).touch()
    return jsonify(ok=True)


@app.get("/api/cases/<key>/runs/<runid>/log")
def log(key, runid):
    path = attempt_path(case_path(key), runid) / "worker.log"
    if not path.is_file():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64000))
        text = handle.read().decode("utf-8", errors="replace")
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<path:name>")
def files(key, name):
    case = case_path(key)
    file = inside(case, name)
    if file.suffix.lower() not in PUBLIC or not file.is_file():
        abort(404)
    response = send_from_directory(
        case,
        name,
        conditional=True,
        as_attachment=request.args.get("download") == "1"
        or file.suffix in (".py", ".xml"),
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def close():
    closing.set()
    with lock:
        process = discovery
    stop_group(process)
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
    for case in WORK.iterdir():
        if case.is_dir() and not case.is_symlink():
            for row in attempts(case):
                if row["status"] == "running":
                    update(
                        attempt_path(case, row["id"]),
                        status="interrupted",
                        error="Studio stopped; files and session are available to resume",
                    )
    atexit.register(close)

    def interrupted(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    port = int(os.environ.get("PHYSAG3D_PORT", "8282"))
    http = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=8,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"PhysAg3D studio: http://127.0.0.1:{port}", flush=True)
    try:
        http.run()
    finally:
        http.close()
        close()
