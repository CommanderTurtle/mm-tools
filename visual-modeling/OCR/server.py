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

from state import (
    ROOT,
    WORK,
    RUNS,
    PIN,
    DEFAULTS,
    atomic,
    read,
    document,
    options,
    require_models,
    offline_env,
)
from state import LANE_STAMP

app = Flask(__name__, static_folder=None)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024**3
lock = threading.RLock()
closing = threading.Event()
active = child = runner = None
VENDOR = {
    "marked": "marked/lib",
    "purify": "dompurify/dist",
    "katex": "katex/dist",
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
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' blob: data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
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


def snapshot(job):
    meta = read(job / "run.json")
    if not meta:
        abort(404)
    return dict(
        meta,
        pages=read(job / "pages.json", []),
        progress=read(job / "progress.json", {}),
        result=read(job / "result.json"),
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


def execute(job, stage):
    global active, child
    process = None
    try:
        with (job / "worker.log").open("ab", buffering=0) as log:
            with lock:
                if closing.is_set() or (job / "cancel").exists():
                    update(job, status="cancelled")
                    return
                process = child = subprocess.Popen(
                    [
                        sys.executable,
                        str(ROOT / "supervisor.py"),
                        sys.executable,
                        str(ROOT / "worker.py"),
                        job.name,
                        stage,
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
                    else ("ready" if stage == "prepare" else "complete")
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
            active = child = None


def launch(job, stage):
    global active, runner
    active = job.name
    update(job, status="preparing" if stage == "prepare" else "running", error=None)
    runner = threading.Thread(target=execute, args=(job, stage), daemon=True)
    runner.start()


@app.get("/api/config")
def config():
    with lock:
        return jsonify(defaults=DEFAULTS, active=active, source=LANE_STAMP)


@app.get("/api/documents")
def documents():
    with lock:
        rows = [
            read(p / "run.json")
            for p in RUNS.iterdir()
            if p.is_dir() and not p.is_symlink()
        ]
        return jsonify(
            sorted((r for r in rows if r), key=lambda r: r["created"], reverse=True)
        )


@app.get("/api/documents/<key>")
def get_document(key):
    with lock:
        return jsonify(snapshot(document(key)))


@app.post("/api/documents")
def upload():
    cfg = options(json.loads(request.form.get("settings", "{}")))
    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active operation or stop it"), 409
        inputs = request.files.getlist("files")
        if not inputs:
            raise ValueError("Choose page images or PDFs")
        job = document(uuid.uuid4().hex)
        (job / "uploads").mkdir(parents=True)
        try:
            uploads = []
            for i, file in enumerate(inputs):
                name = Path(file.filename or "Page").name[:160]
                kind = "pdf" if name.lower().endswith(".pdf") else "image"
                identity = f"{i:06d}.pdf" if kind == "pdf" else f"{i:06d}.image"
                file.save(job / "uploads" / identity)
                uploads.append(dict(id=identity, title=name, kind=kind))
            atomic(
                job / "run.json",
                dict(
                    id=job.name,
                    title=uploads[0]["title"],
                    created=time.time(),
                    settings=cfg,
                    uploads=uploads,
                    order=[],
                    source=LANE_STAMP,
                ),
            )
        except Exception:
            shutil.rmtree(job)
            raise
        launch(job, "prepare")
        return jsonify(snapshot(job)), 202


@app.post("/api/documents/<key>/parse")
def parse(key):
    data = request.get_json()
    if not isinstance(data, dict):
        raise ValueError("Send a settings and page-order object")
    cfg = options(data.get("settings", {}))
    require_models()
    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active operation or stop it"), 409
        job = document(key)
        current = snapshot(job)
        allowed = {page["id"] for page in current["pages"]}
        order = data.get("order")
        if (
            not isinstance(order, list)
            or not order
            or any(not isinstance(p, str) or p not in allowed for p in order)
            or len(set(order)) != len(order)
        ):
            raise ValueError("Choose an ordered, non-empty set of pages")
        # Each parse gets its own receipt and outputs; a new parse never destroys
        # the previous one. Inputs/pages are immutable and linked locally.
        run = document(uuid.uuid4().hex)
        run.mkdir()
        for folder in ("pages", "thumbs"):
            (run / folder).mkdir()
            for page in order:
                os.link(job / folder / page, run / folder / page)
        atomic(run / "pages.json", [p for p in current["pages"] if p["id"] in order])
        atomic(
            run / "run.json",
            dict(
                id=run.name,
                title=current["title"],
                created=time.time(),
                settings=cfg,
                order=order,
                source=LANE_STAMP,
                uploads=[],
            ),
        )
        launch(run, "parse")
        return jsonify(snapshot(run)), 202


@app.post("/api/example")
def example():
    from PIL import Image, ImageDraw, ImageFont

    with lock:
        if active or closing.is_set():
            return jsonify(error="Wait for the active operation or stop it"), 409
        job = document(uuid.uuid4().hex)
        (job / "uploads").mkdir(parents=True)
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        title = ImageFont.load_default(size=48)
        body = ImageFont.load_default(size=26)
        draw.text((100, 100), "Field report / 2026", fill="#183d34", font=title)
        draw.multiline_text(
            (100, 220),
            "A compact document with headings, a table and a figure.\n\nThe sampling stations record temperature and rainfall.\nMeasurements are collected at regular intervals.",
            fill="#202722",
            font=body,
            spacing=16,
        )
        rows = [
            ("Station", "Temperature", "Rainfall"),
            ("North", "18.5 C", "12 mm"),
            ("East", "21.2 C", "8 mm"),
            ("West", "19.7 C", "15 mm"),
        ]
        for r, values in enumerate(rows):
            y = 500 + r * 70
            draw.line((100, y, 1100, y), fill="#669583", width=2)
            for x, value in zip((120, 480, 840), values):
                draw.text((x, y + 18), value, font=body, fill="#202722")
        draw.line((100, 780, 1100, 780), fill="#669583", width=2)
        draw.text(
            (100, 870), "Figure 1. Rainfall by station", font=body, fill="#202722"
        )
        for x, height, name in (
            (200, 240, "North"),
            (500, 160, "East"),
            (800, 300, "West"),
        ):
            draw.rectangle((x, 1300 - height, x + 140, 1300), fill="#438a73")
            draw.text((x, 1330), name, font=body, fill="#202722")
        draw.text(
            (100, 1480), "Local example document  |  1", font=body, fill="#666666"
        )
        image.save(job / "uploads/000000.image", format="PNG")
        atomic(
            job / "run.json",
            dict(
                id=job.name,
                title="Field report",
                created=time.time(),
                settings=DEFAULTS,
                uploads=[dict(id="000000.image", title="Field report", kind="image")],
                order=[],
                source=LANE_STAMP,
            ),
        )
        launch(job, "prepare")
        return jsonify(snapshot(job)), 202


@app.post("/api/documents/<key>/cancel")
def cancel(key):
    with lock:
        job = document(key)
        if active == key:
            (job / "cancel").touch()
    return jsonify(ok=True)


@app.delete("/api/documents/<key>")
def delete(key):
    with lock:
        job = document(key)
        snapshot(job)
        if active == key:
            return jsonify(error="Stop the document first"), 409
        shutil.rmtree(job)
    return jsonify(ok=True)


@app.get("/api/documents/<key>/stream")
def stream(key):
    job = document(key)
    snapshot(job)
    offset = request.args.get("offset", 0, type=int)
    if offset is None or offset < 0:
        raise ValueError("Invalid stream offset")
    path = job / "stream.txt"
    if not path.exists():
        return jsonify(text="", offset=0, more=False)
    with path.open("rb") as source:
        source.seek(min(offset, path.stat().st_size))
        start = source.tell()
        value = source.read(65536)
        # Never split a UTF-8 code point between polls.
        while value:
            try:
                text = value.decode("utf-8")
                break
            except UnicodeDecodeError as error:
                if error.start < len(value) - 4:
                    raise
                value = value[: error.start]
        else:
            text = ""
        offset = start + len(value)
    return jsonify(text=text, offset=offset, more=path.stat().st_size > offset)


@app.get("/api/documents/<key>/log")
def log(key):
    job = document(key)
    snapshot(job)
    path = job / "worker.log"
    if not path.exists():
        return "", 200, {"Content-Type": "text/plain; charset=utf-8"}
    with path.open("rb") as source:
        source.seek(max(0, path.stat().st_size - 64000))
        text = source.read().decode("utf-8", errors="replace")
    return text, 200, {"Content-Type": "text/plain; charset=utf-8"}


@app.get("/files/<key>/<folder>/<name>")
def files(key, folder, name):
    job = document(key)
    snapshot(job)
    if folder in ("pages", "thumbs"):
        allowed = bool(re.fullmatch(r"\d{6}\.png", name))
    elif folder == "output":
        allowed = name in {
            "document.md",
            "raw.txt",
            "layout.json",
            "settings.json",
        } or bool(re.fullmatch(r"page\d+_\d+_\d+_\d+_\d+\.jpg", name))
    else:
        abort(404)
    if not allowed or (job / folder).is_symlink() or (job / folder / name).is_symlink():
        abort(404)
    return send_from_directory(
        job / folder, name, as_attachment=request.args.get("download") == "1"
    )


@app.get("/files/<key>/document.zip")
def bundle(key):
    job = document(key)
    snapshot(job)
    return send_from_directory(job, "document.zip", as_attachment=True)


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
        raise SystemExit("This studio workspace is already serving")
    for job in RUNS.iterdir():
        if (
            job.is_dir()
            and not job.is_symlink()
            and read(job / "run.json", {}).get("status") in ("running", "preparing")
        ):
            update(
                job,
                status="interrupted",
                error="Studio stopped; uploaded pages and partial output remain saved",
            )
    atexit.register(close)

    def interrupted(*_):
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    port = int(os.environ.get("OCR_PORT", "8287"))
    server = create_server(
        app,
        host="127.0.0.1",
        port=port,
        threads=6,
        max_request_body_size=app.config["MAX_CONTENT_LENGTH"],
    )
    print(f"OCR 2 studio: http://127.0.0.1:{port}", flush=True)
    try:
        server.run()
    finally:
        server.close()
        close()
