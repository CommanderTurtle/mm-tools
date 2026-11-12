"""Public inference runtime for WeVisOne-VL."""

from .engine import InferenceResult, WeVisOneVL
from .prompts import TASKS, build_prompt

__all__ = ["InferenceResult", "TASKS", "WeVisOneVL", "build_prompt"]

__version__ = "1.0.0"
