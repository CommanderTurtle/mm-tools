"""Owned, offline Comfy subprocess. Never connects to an external Comfy service."""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.getenv("MULTIMEDIA_MODELS", "~/multimedia/models")).expanduser()


class PrivateComfy:
    def __init__(self):
        self.runtime = Path(os.getenv("IDEOGRAM_COMFY_RUNTIME", str(ROOT.parent / "minimax/runtime"))).expanduser()
        bundled_python = ROOT.parent / "minimax/.venv/bin/python"
        self.python = Path(os.getenv("IDEOGRAM_COMFY_PYTHON", str(bundled_python) if bundled_python.is_file() else sys.executable)).expanduser()
        self.state = Path(os.getenv("IDEOGRAM_COMFY_STATE", str(ROOT / ".runtime/editing"))).expanduser()
        self.model = Path(os.getenv("IDEOGRAM4_FP8_MODEL", str(MODELS / "ideogram-ai--ideogram-4-fp8"))).expanduser()
        self.caption_model = Path(os.getenv("IDEOGRAM_CAPTION_MODEL", str(MODELS / "qwen/text-encoder-vl-nvfp4/qwen3_vl_4b_nvfp4_full.safetensors"))).expanduser()
        self.port = int(os.getenv("IDEOGRAM_PRIVATE_PORT", "8175"))
        if not 1024 <= self.port <= 65535:
            raise ValueError("Invalid IDEOGRAM_PRIVATE_PORT")
        self.url = f"http://127.0.0.1:{self.port}"
        self.process = None
        self.log = None
        self.lock = threading.RLock()
        self.http = requests.Session()
        self.http.trust_env = False  # local private traffic must not use a user's HTTP proxy
        atexit.register(self.stop)

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def status(self):
        required = [self.runtime / "main.py", self.python,
                    self.model / "transformer/diffusion_pytorch_model.safetensors",
                    self.model / "unconditional_transformer/diffusion_pytorch_model.safetensors",
                    self.model / "text_encoder/model.safetensors",
                    self.model / "vae/diffusion_pytorch_model.safetensors"]
        missing = [str(p) for p in required if not p.is_file()]
        return {"loaded": self.running, "available": not missing, "missing": missing,
                "model_path": str(self.model), "caption_available": self.caption_model.is_file(),
                "caption_model": str(self.caption_model), "log": str(self.state / "engine.log")}

    def request(self, method, path, **kwargs):
        response = self.http.request(method, self.url + path, timeout=kwargs.pop("timeout", 15), **kwargs)
        if not response.ok:
            raise RuntimeError(f"Private Comfy {path}: {response.status_code} {response.text[:2000]}")
        return response

    def start(self):
        with self.lock:
            if self.running:
                return
            missing = self.status()["missing"]
            if missing:
                raise RuntimeError("Missing local files: " + ", ".join(missing))
            with socket.socket() as probe:
                if probe.connect_ex(("127.0.0.1", self.port)) == 0:
                    raise RuntimeError(f"Port {self.port} is occupied; refusing to use another Comfy process.")
            for name in ("input", "output", "temp", "user", "custom_nodes"):
                (self.state / name).mkdir(parents=True, exist_ok=True)
            import yaml
            config = self.state / "paths.yaml"
            config.write_text(yaml.safe_dump({"mmtools_ideogram": {
                "custom_nodes": str(ROOT / "object_remover"),
                "text_encoders": str(self.caption_model.parent),
            }}), encoding="utf-8")
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                       DO_NOT_TRACK="1", TOKENIZERS_PARALLELISM="false", IDEOGRAM4_FP8_MODEL=str(self.model))
            command = [str(self.python), str(self.runtime / "main.py"), "--listen", "127.0.0.1", "--port", str(self.port),
                       "--base-directory", str(self.state),
                       "--extra-model-paths-config", str(config), "--disable-auto-launch", "--disable-api-nodes",
                       "--disable-metadata", "--disable-all-custom-nodes", "--whitelist-custom-nodes", "comfy_nodes",
                       "--cache-none", "--database-url", f"sqlite:///{self.state / 'user/comfyui.db'}"]
            if os.getenv("IDEOGRAM_COMFY_DEVICE", "auto") == "cpu":
                command += ["--cpu"]
            for kind in ("input", "output", "temp", "user"):
                command += [f"--{kind}-directory", str(self.state / kind)]
            self.log = (self.state / "engine.log").open("ab")
            self.process = subprocess.Popen(command, cwd=self.state, env=env, stdin=subprocess.DEVNULL,
                                            stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 120
                while time.monotonic() < deadline:
                    if not self.running:
                        raise RuntimeError(f"Private engine exited. See {self.state / 'engine.log'}")
                    try:
                        info = self.request("GET", "/object_info", timeout=2).json()
                        if "MMToolsIdeogramLocal" not in info or "Ideogram4Scheduler" not in info:
                            raise RuntimeError("Private engine is missing its Ideogram nodes; check engine.log")
                        return
                    except requests.RequestException:
                        time.sleep(.25)
                raise RuntimeError("Private engine startup timed out; check engine.log")
            except Exception:
                self.stop()
                raise

    def stop(self):
        # Stop only our process group, never by port, process name or external PID.
        with self.lock:
            if self.running:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait(timeout=5)
            self.process = None
            if self.log:
                self.log.close()
                self.log = None

    def run(self, graph, files):
        with self.lock:
            self.start()
            # Server-generated names only; never accept a user-supplied path or graph.
            uploaded = []
            prompt_id = None
            try:
                for name, data in files.items():
                    path = self.state / "input" / name
                    if Path(name).name != name:
                        raise ValueError("Invalid input filename")
                    path.write_bytes(data)
                    uploaded.append(path)
                result = self.request("POST", "/prompt", json={"prompt": graph, "client_id": uuid.uuid4().hex}).json()
                prompt_id = result["prompt_id"]
                deadline = time.monotonic() + float(os.getenv("IDEOGRAM_EDIT_TIMEOUT", "1800"))
                while time.monotonic() < deadline:
                    if not self.running:
                        raise RuntimeError("Private engine stopped during the edit; check engine.log")
                    item = self.request("GET", f"/history/{prompt_id}").json().get(prompt_id)
                    if item:
                        status = item.get("status", {})
                        if status.get("status_str") == "error":
                            errors = [v.get("exception_message", str(v)) for k, v in status.get("messages", [])
                                      if k in {"execution_error", "execution_interrupted"}]
                            raise RuntimeError("Ideogram execution failed: " + " ".join(errors))
                        if status.get("completed"):
                            return item.get("outputs", {})
                    time.sleep(.3)
                raise RuntimeError("Ideogram edit timed out")
            except Exception:
                # A timed-out/failed request must not leave a hidden GPU job alive.
                self.stop()
                raise
            finally:
                for path in uploaded:
                    path.unlink(missing_ok=True)
                if prompt_id and self.running:
                    try:
                        self.request("POST", "/history", json={"delete": [prompt_id]})
                    except (requests.RequestException, RuntimeError):
                        pass

    def image_output(self, outputs):
        images = outputs.get("19", {}).get("images", [])
        if not images:
            raise RuntimeError("Ideogram returned no image")
        record = images[0]
        root = (self.state / "output").resolve()
        path = (root / record.get("subfolder", "") / record["filename"]).resolve()
        if not path.is_relative_to(root) or record.get("type") != "output":
            raise RuntimeError("Invalid private output path")
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)


def caption_json(text):
    """Accept plain JSON or one fenced response; never fall back to a different model."""
    value = text.strip()
    if value.startswith("```") and value.endswith("```"):
        value = value.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    obj = json.loads(value)
    if not isinstance(obj, dict) or not isinstance(obj.get("high_level_description"), str):
        raise ValueError("Caption needs a high_level_description string")
    parts = obj.get("compositional_deconstruction")
    if not isinstance(parts, dict) or not isinstance(parts.get("background"), str) or not isinstance(parts.get("elements"), list):
        raise ValueError("Caption needs compositional_deconstruction with background and elements")
    # Magic prompt includes aspect ratio for planning, not the model's caption body.
    obj.pop("aspect_ratio", None)
    # Load the pure validator without importing the native package's GPU pipeline.
    import importlib.util
    spec = importlib.util.spec_from_file_location("ideogram_caption_validator", ROOT / "src/ideogram4/caption_verifier.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    issues = module.CaptionVerifier().verify(obj)
    if issues:
        raise ValueError("Caption validation: " + "; ".join(issues))
    return json.dumps(obj, ensure_ascii=False)
