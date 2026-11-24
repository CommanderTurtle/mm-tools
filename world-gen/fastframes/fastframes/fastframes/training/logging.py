#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Rank-0 metric logging to ``<run_dir>/metrics.jsonl`` and, optionally, Weights & Biases."""
from __future__ import annotations

import json
import time
from pathlib import Path


class MetricLogger:
    """``config.wandb``: ``{project, entity, name, id}``; ``project: null`` disables W&B.

    Pass a stable ``id`` to continue the same W&B run after a resume.
    """

    def __init__(self, config, run_dir, is_main=True):
        self.is_main = is_main
        self.path = Path(run_dir) / "metrics.jsonl"
        self.wandb = None
        if not is_main:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        settings = dict(config.get("wandb") or {})
        if settings.get("project"):
            import wandb
            self.wandb = wandb.init(project=settings["project"], entity=settings.get("entity"),
                                    name=settings.get("name"), id=settings.get("id"), resume="allow",
                                    config=dict(config))

    def log(self, metrics, step):
        if not self.is_main:
            return
        values = {k: (v.item() if hasattr(v, "item") else v) for k, v in metrics.items()}
        with self.path.open("a") as f:
            f.write(json.dumps({"step": step, "time": time.time(), **values}) + "\n")
        if self.wandb is not None:
            self.wandb.log(values, step=step)

    def finish(self):
        if self.wandb is not None:
            self.wandb.finish()
