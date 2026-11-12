"""Narrow I/O adapters; keep upstream prompts, samplers and geometry intact."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from state import ROOT, NATIVE, MODELS


def local_vision(messages, *, model=None, max_tokens=2048, temperature=0.0):
    with tempfile.TemporaryDirectory(prefix="scenebloom-vl-") as folder:
        request, response = (
            Path(folder) / "request.json",
            Path(folder) / "response.json",
        )
        request.write_text(
            json.dumps(
                dict(messages=messages, max_tokens=max_tokens, temperature=temperature)
            )
        )
        subprocess.run(
            [
                str(ROOT / ".vl-venv/bin/python"),
                str(ROOT / "vision.py"),
                str(request),
                str(response),
                str(os.getpid()),
            ],
            check=True,
        )
        return json.loads(response.read_text())["text"]


def module(module_name, spec, ok_codes=(0,), no_user_site=False):
    with tempfile.TemporaryDirectory(prefix="scenebloom-module-") as folder:
        request, response = Path(folder) / "spec.json", Path(folder) / "response.json"
        request.write_text(json.dumps(spec))
        result = subprocess.run(
            [
                sys.executable,
                str(ROOT / "module.py"),
                module_name,
                str(os.getpid()),
                "--spec",
                str(request),
                "--output",
                str(response),
            ],
            env=dict(os.environ, PYTHONNOUSERSITE="1"),
        )
        if result.returncode not in ok_codes or not response.is_file():
            raise RuntimeError(
                f"{module_name} stopped with code {result.returncode}; see run log"
            )
        return json.loads(response.read_text())


def install():
    sys.path.insert(0, str(NATIVE))
    os.chdir(NATIVE)
    from scenebloom.utils import vlm_openrouter

    vlm_openrouter.call_openrouter = local_vision
    from scenebloom.sampling.zoning.llm_zone_decomposer import LLMZoneDecomposer

    # Zoning uses a separate SDK call upstream; preserve its conversation/re-read
    # logic while making its one provider call use the same local image transport.
    LLMZoneDecomposer._read_once = lambda self, messages: local_vision(
        messages, max_tokens=2048, temperature=1.0
    )
    from scenebloom.sampling.proposals import strategy

    strategy._run_module = module


def local_dino():
    import torch

    load = torch.hub.load

    def offline(repository, name, *args, **kwargs):
        if repository == "facebookresearch/dinov2" and name == "dinov2_vitb14":
            network = load(
                str(ROOT / "dinov2"), name, source="local", pretrained=False
            )
            weights = torch.load(
                MODELS / "checkpoints/dinov2/dinov2_vitb14_pretrain.pth",
                map_location="cpu",
                weights_only=True,
            )
            network.load_state_dict(weights, strict=True)
            return network
        if kwargs.get("source") != "local":
            raise RuntimeError(f"Unregistered offline backbone: {repository}/{name}")
        return load(repository, name, *args, **kwargs)

    torch.hub.load = offline
