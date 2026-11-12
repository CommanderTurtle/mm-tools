import json
import subprocess
import threading
import uuid

from apps.k3_image_captioner import (
    I2V_PROMPT_PROFILE,
    NomasImageCaptioner,
    _reference_image_url,
    normalize_caption,
    selected_perspective,
)
from apps.kimi_k3_prompt_enhancer import PromptEnhancementError, _extract_json
from paths import ROOT


class LocalEnhancer:
    model = "Nomas-PE"
    configured = True

    def __init__(self, processes):
        self.processes = processes
        self.lock = threading.Lock()

    def complete(
        self, messages, *, stream_callback=None, reasoning_effort=None, seed=None
    ):
        epoch = self.processes.epoch
        with self.lock:
            if epoch != self.processes.epoch:
                raise PromptEnhancementError("Studio stopped this request")
            child = self.processes.spawn(
                [str(ROOT / ".pe-venv/bin/python"), str(ROOT / "pe_worker.py")],
                epoch=epoch,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                text=True,
                encoding="utf-8",
            )
            try:
                output, _ = child.communicate(
                    json.dumps({"messages": messages, "seed": seed}), timeout=3600
                )
                if child.returncode or epoch != self.processes.epoch:
                    raise PromptEnhancementError(
                        "Local prompt enhancer stopped; see runtime log"
                    )
                reply = json.loads(output)
                if stream_callback:
                    stream_callback("content", reply["content"])
                return reply["content"], str(uuid.uuid4())
            except subprocess.TimeoutExpired as exc:
                self.processes.terminate(child)
                raise PromptEnhancementError(
                    "Local prompt enhancer exceeded one hour"
                ) from exc
            finally:
                self.processes.reap(child)


class ImageCaptioner(NomasImageCaptioner):
    def __init__(self, client):
        self.client = client
        self.model = "Nomas-PE"

    @property
    def configured(self):
        return self.client.configured

    def caption(self, image_path, *, fps=24, perspective="third-person follow"):
        perspective = selected_perspective(perspective)
        messages = self._messages(_reference_image_url(image_path), perspective)
        raw, request_id = self.client.complete(messages, reasoning_effort="none")
        result = normalize_caption(
            _extract_json(raw),
            fps=fps,
            model=self.model,
            request_id=request_id,
            perspective=perspective,
        )
        result.update(prompt_profile=I2V_PROMPT_PROFILE.name, user_message=messages[-1])
        return result


class DisabledCloud:
    configured = False
    model = "disabled"
