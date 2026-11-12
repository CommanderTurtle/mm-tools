"""Local studio HTTP layer. Native model imports live only in child processes."""

import atexit
import collections
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

from flask import Flask, abort, jsonify, request, send_from_directory
from waitress import create_server

from state import (
    DEFAULTS,
    PIN,
    ROOT,
    SOURCE,
    WORK,
    atomic,
    job_path,
    physics_python,
    physics_ready,
    read,
    settings,
)

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = (
    int(os.environ.get("FLOWHMR_UPLOAD_MB", "4096")) * 1024 * 1024
)
lock = threading.RLock()
wake = threading.Condition(lock)
pending = collections.deque()
preparing = set()
readers = collections.Counter()
stopping = threading.Event()
active = None
process = None
thread = None
VIDEO = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
PUBLIC = {
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".avi",
    ".m4v",
    ".bin",
    ".json",
    ".npz",
    ".fbx",
    ".jpg",
}
TERMINAL = {"complete", "partial", "failed", "cancelled", "interrupted"}


@app.before_request
def local_requests():
    if request.host.split(":")[0] not in {"localhost", "127.0.0.1"}:
        abort(403)
    if request.method != "GET" and request.headers.get("Origin") not in (
        None,
        request.host_url.rstrip("/"),
    ):
        abort(403)


@app.errorhandler(ValueError)
def bad_input(error):
    return jsonify(error=str(error)), 400


@app.errorhandler(413)
def too_large(error):
    return jsonify(error="Upload exceeds FLOWHMR_UPLOAD_MB"), 413


def snapshot(path):
    meta = read(path / "job.json")
    if not meta:
        abort(404)
    result = {**meta, "progress": read(path / "progress.json", {})}
    if (
        (path / "selection.json").is_file()
        and meta["status"] == "running"
        and not (path / "choice.json").exists()
    ):
        result["selection"] = read(path / "selection.json")
    result["viewer"] = (path / "viewer/scene.json").is_file()
    result["physics_viewer"] = (path / "viewer/physics.json").is_file()
    result["downloads"] = [
        f.relative_to(path).as_posix()
        for f in sorted(path.glob("native/input/*.npz"))
        if "_seed" in f.name
    ]
    result["downloads"] += [
        f.relative_to(path).as_posix() for f in sorted(path.glob("native/input/*.fbx"))
    ]
    if (path / "viewer/physics.npz").is_file():
        result["downloads"] += ["viewer/physics.npz", "viewer/physics.json"]
    return result


def update(path, **fields):
    meta = read(path / "job.json", {})
    atomic(path / "job.json", {**meta, **fields, "updated": time.time()})


def stop_group(child):
    if child is None:
        return
    # The session belongs exclusively to this job, including ffmpeg/native children.
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.killpg(child.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def execute(path, phase):
    global process
    interpreter = physics_python() if phase == "physics" else Path(sys.executable)
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        HF_HUB_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TRANSFORMERS_OFFLINE="1",
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
    )
    env["PATH"] = str(interpreter.parent) + os.pathsep + env.get("PATH", "")
    with (path / "worker.log").open("ab", buffering=0) as log:
        with lock:
            if stopping.is_set() or (path / "cancel").exists():
                return -1
            process = subprocess.Popen(
                [
                    str(interpreter),
                    str(ROOT / "worker.py"),
                    str(path),
                    phase,
                    str(os.getpid()),
                ],
                cwd=SOURCE,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            child = process
        try:
            while child.poll() is None:
                if stopping.wait(0.2) or (path / "cancel").exists():
                    stop_group(child)
                    break
            return child.wait()
        finally:
            stop_group(child)
            with lock:
                process = None


def run_queue():
    global active
    while not stopping.is_set():
        with wake:
            wake.wait_for(lambda: pending or stopping.is_set())
            if stopping.is_set():
                return
            key = pending.popleft()
            path = job_path(key)
            active = key
            update(path, status="running", error=None)
        try:
            code = execute(path, "reconstruct")
            cancelled = (path / "cancel").exists() or stopping.is_set()
            if cancelled:
                status, error = "cancelled", None
            elif code != 0 or not (path / "viewer/scene.json").exists():
                status, error = (
                    "failed",
                    read(path / "error.json", {}).get(
                        "error", "Reconstruction stopped; open the job log"
                    ),
                )
            elif read(path / "job.json")["settings"]["physics"]:
                atomic(
                    path / "progress.json",
                    dict(
                        stage="physics",
                        percent=0,
                        message="Reconstruction released; starting PHC",
                    ),
                )
                code = execute(path, "physics")
                if (path / "cancel").exists() or stopping.is_set():
                    status, error = "cancelled", None
                elif code or not (path / "viewer/physics.json").is_file():
                    status, error = (
                        "partial",
                        read(path / "error.json", {}).get(
                            "error", "Physics stopped; reconstruction is saved"
                        ),
                    )
                else:
                    status, error = "complete", None
            else:
                status, error = "complete", None
            with lock:
                update(path, status=status, error=error)
        except Exception as exc:
            with lock:
                update(path, status="failed", error=str(exc))
        finally:
            with lock:
                active = None


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
def configuration():
    return jsonify(
        defaults=DEFAULTS,
        source=LANE_STAMP,
        physics=physics_ready(),
        camera=(SOURCE / "ckpts/vggt-omega/vggt_omega_1b_512.pt").is_file(),
        upload_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024),
    )


@app.get("/api/jobs")
def jobs():
    with lock:
        rows = [
            snapshot(p)
            for p in WORK.iterdir()
            if p.is_dir() and not p.is_symlink() and (p / "job.json").exists()
        ]
    return jsonify(sorted(rows, key=lambda j: j["created"], reverse=True))


def reuse_files(old, new, config):
    previous = read(old / "job.json")["settings"]
    destination = new / "native/input"
    destination.mkdir(parents=True, exist_ok=True)
    # Native caches depend on selection, frame range, camera intrinsics and SAM
    # chunking. Copies are intentional: native invalidation may rewrite caches.
    stable_track = (
        not config["reselect"] and config["max_frames"] == previous["max_frames"]
    )
    stable_camera = stable_track and all(
        config[k] == previous[k] for k in ("static_camera", "camera_stride")
    )
    stable_features = stable_camera and config["sam_chunk"] == previous["sam_chunk"]
    names = {"input_30fps.mp4"}
    if stable_track:
        names.add("input_bbox.npz")
    if stable_camera:
        names |= {"input_vggt_camera.npz", "input_static_camera.npz"}
    if stable_features:
        names |= {"input_sam3d_feat.pt", "input_sam3d_feat_static.pt"}
    for name in names:
        source = old / "native/input" / name
        if source.is_file():
            shutil.copy2(source, destination / name)


@app.post("/api/jobs")
def submit():
    config = settings(json.loads(request.form.get("settings", "{}")))
    if (
        not config["static_camera"]
        and not (SOURCE / "ckpts/vggt-omega/vggt_omega_1b_512.pt").is_file()
    ):
        raise ValueError(
            "Download the flowhmr camera checkpoint, or choose Static camera"
        )
    if config["physics"] and not physics_ready():
        raise ValueError(
            "Set up the separate PHC environment before enabling Physics tracking"
        )
    key = uuid.uuid4().hex
    path = job_path(key)
    old = None
    with lock:
        if stopping.is_set() or len(pending) + len(preparing) >= 16:
            return jsonify(error="Queue is full or closing"), 409
        reuse = request.form.get("reuse")
        if reuse:
            old = job_path(reuse)
            meta = read(old / "job.json")
            if not meta or meta["status"] not in TERMINAL:
                raise ValueError("Reuse a finished or interrupted job")
            filename = meta["filename"]
            source = old / meta["input"]
            extension = source.suffix
        else:
            upload = request.files.get("video")
            if not upload or not upload.filename:
                raise ValueError("Choose a video")
            filename = Path(upload.filename).name
            extension = Path(filename).suffix.lower()
            if extension not in VIDEO:
                raise ValueError("Choose MP4, MOV, MKV, WebM, AVI or M4V")
        path.mkdir()
        preparing.add(key)
        if old:
            readers[reuse] += 1
    # Uploads/cache copies can be large; leave status and cancellation responsive.
    try:
        target = path / ("input" + extension)
        if old:
            shutil.copy2(source, target)
            reuse_files(old, path, config)
        else:
            upload.save(target)
        if target.stat().st_size == 0:
            raise ValueError("The video is empty")
        with wake:
            closing = stopping.is_set()
            atomic(
                path / "job.json",
                dict(
                    id=key,
                    filename=filename,
                    input=target.name,
                    settings=config,
                    created=time.time(),
                    status="interrupted" if closing else "queued",
                    source=LANE_STAMP,
                    reused_from=reuse,
                ),
            )
            if not closing:
                pending.append(key)
                wake.notify()
            return jsonify(snapshot(path)), 201
    except Exception:
        shutil.rmtree(path)
        raise
    finally:
        with lock:
            preparing.discard(key)
            if old:
                readers[reuse] -= 1
                if not readers[reuse]:
                    del readers[reuse]


@app.get("/api/jobs/<key>")
def job(key):
    with lock:
        return jsonify(snapshot(job_path(key)))


@app.post("/api/jobs/<key>/person")
def choose_person(key):
    with lock:
        path = job_path(key)
        selection = read(path / "selection.json")
        body = request.get_json(silent=True) or {}
        choice = body.get("index")
        if (
            active != key
            or not selection
            or type(choice) is not int
            or not 0 <= choice < len(selection["candidates"])
        ):
            raise ValueError("Choose an offered person track")
        if (path / "choice.json").exists():
            return jsonify(error="Selection already accepted"), 409
        atomic(path / "choice.json", {"index": choice})
    return jsonify(ok=True)


@app.post("/api/jobs/<key>/cancel")
def cancel(key):
    with lock:
        path = job_path(key)
        meta = read(path / "job.json")
        if not meta:
            abort(404)
        (path / "cancel").touch()
        if key in pending:
            pending.remove(key)
            update(path, status="cancelled")
    return jsonify(ok=True)


@app.delete("/api/jobs/<key>")
def delete(key):
    with lock:
        path = job_path(key)
        meta = read(path / "job.json")
        if not meta:
            abort(404)
        if active == key or key in pending or readers[key]:
            return jsonify(
                error="Wait for copying or cancel the job before deleting it"
            ), 409
        shutil.rmtree(path)
    return jsonify(ok=True)


@app.get("/api/jobs/<key>/log")
def log(key):
    path = job_path(key) / "worker.log"
    if not path.is_file():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 64000))
        content = handle.read().decode("utf-8", errors="replace")
    return content, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<path:name>")
def files(key, name):
    root = job_path(key)
    file = (root / name).resolve()
    if not file.is_relative_to(root.resolve()) or file.suffix.lower() not in PUBLIC:
        abort(404)
    return send_from_directory(
        root, name, conditional=True, as_attachment=request.args.get("download") == "1"
    )


def close():
    stopping.set()
    with wake:
        for key in pending:
            update(
                job_path(key),
                status="interrupted",
                error="Studio closed before this job started",
            )
        pending.clear()
        wake.notify_all()
    if thread:
        thread.join(timeout=8)
    stop_group(process)


if __name__ == "__main__":
    WORK.mkdir(exist_ok=True)
    instance = (WORK / ".server.lock").open("a")
    try:
        fcntl.flock(instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This studio workspace is already serving")
    for path in WORK.iterdir():
        meta = read(path / "job.json") if path.is_dir() else None
        if meta and meta["status"] in ("queued", "running"):
            update(
                path,
                status="interrupted",
                error="Previous session ended; reuse this job to continue",
            )
    thread = threading.Thread(target=run_queue, daemon=True)
    thread.start()
    atexit.register(close)

    def interrupted(signum, frame):
        raise SystemExit(0)

    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    port = int(os.environ.get("FLOWHMR_PORT", "8281"))
    http = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=8,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"FlowHMR studio: http://127.0.0.1:{port}", flush=True)
    try:
        http.run()
    finally:
        http.close()
        close()
