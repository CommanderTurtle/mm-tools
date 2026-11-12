"""
SVS data preprocessing package.

Provides reusable library code for converting raw SVS data (folder-based or
JSON-based annotations) into preprocessed Arrow format for training.

Key public API:

* :class:`SVSPreprocessor` / :func:`create_lightweight_preprocessor` —
  audio VAE encoding and SVS token sequence construction.
* :func:`rebuild_svs_prompt` — reconstruct SVS prompt text from metadata.
* :func:`load_config`, :func:`load_all_datasets` — dataset loading.
* :func:`build_text_tensor`, :func:`estimate_duration_from_notes` — text
  tensor construction and duration estimation.
* :func:`process_and_save`, :func:`process_and_save_multigpu` — Arrow
  dataset writing (single-GPU and multi-GPU).
"""

from importlib import import_module

from .svs_preprocessor import SVSPreprocessor, create_lightweight_preprocessor
from .svs_prompt import rebuild_svs_prompt

_LAZY_EXPORTS = {
    "load_config": (".data_loaders", "load_config"),
    "load_samples_from_folder": (".data_loaders", "load_samples_from_folder"),
    "load_samples_from_json_file": (".data_loaders", "load_samples_from_json_file"),
    "load_samples_from_weak_json_file": (".data_loaders", "load_samples_from_weak_json_file"),
    "load_all_datasets": (".data_loaders", "load_all_datasets"),
    "reconstruct_lyric_text": (".data_loaders", "reconstruct_lyric_text"),
    "build_text_tensor": (".text_tensor", "build_text_tensor"),
    "estimate_duration_from_notes": (".text_tensor", "estimate_duration_from_notes"),
    "process_and_save": (".arrow_writer", "process_and_save"),
    "process_and_save_multigpu": (".arrow_writer", "process_and_save_multigpu"),
}


def __getattr__(name):
    """Keep training/data dependencies lazy for inference-only installations."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

__all__ = [
    "SVSPreprocessor",
    "create_lightweight_preprocessor",
    "rebuild_svs_prompt",
    "load_config",
    "load_samples_from_folder",
    "load_samples_from_json_file",
    "load_samples_from_weak_json_file",
    "load_all_datasets",
    "reconstruct_lyric_text",
    "build_text_tensor",
    "estimate_duration_from_notes",
    "process_and_save",
    "process_and_save_multigpu",
]
