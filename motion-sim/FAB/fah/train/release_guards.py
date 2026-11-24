"""Checkpoint and camera-split validation shared by the release training entry points."""
import json
import os
import zlib
from functools import lru_cache
from pathlib import Path


def finetuned_teacher_state(checkpoint, model_keys):
    """Extract SAM hand-model tensors from a train.finetune checkpoint, before student initialisation.

    Teacher fine-tuning stores model.* and may also store an unused encoder.*. Reject a
    checkpoint without backbone updates: silently using base features would change distillation.
    """
    if checkpoint.get("student_name"):
        raise ValueError("The teacher must be a SAM hand-model fine-tune, not a student checkpoint")
    trainable = checkpoint.get("trainable", {})
    state = {k[len("model."):]: v for k, v in trainable.items() if k.startswith("model.")}
    if not any(k.startswith("backbone.") for k in state):
        raise ValueError("Teacher checkpoint has no fine-tuned backbone tensors")
    unexpected = set(state) - set(model_keys)
    if unexpected:
        raise ValueError(f"Teacher checkpoint has unsupported tensors: {sorted(unexpected)[:5]}")
    return state


def egodex_episode_key(episode):
    """Canonical part2/task/id from source paths or cached task__id directory names."""
    parts = str(episode).replace("\\", "/").replace("__", "/").rstrip("/").split("/")
    if len(parts) < 2 or not all(parts[-2:]):
        raise ValueError(f"Expected an EgoDex task/episode: {episode}")
    task, name = parts[-2:]
    return f"part2/{task}/{os.path.splitext(name)[0]}"


@lru_cache(maxsize=1)
def _listed_egodex_holdouts():
    split = Path(__file__).resolve().parents[1] / "bench/splits/egodex_world_p2.json"
    return frozenset(egodex_episode_key(e) for e in json.loads(split.read_text())["episodes"])


def egodex_camera_holdout(episode):
    """Reserve the manifest's entire 5% candidate pool, including all 80 scored episodes."""
    key = egodex_episode_key(episode)
    return key in _listed_egodex_holdouts() or zlib.crc32(key.encode()) % 1000 < 50
