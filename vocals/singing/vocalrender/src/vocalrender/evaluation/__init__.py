"""
vocalrender.evaluation — Shared evaluation modules for SVS test and validation.

Provides unified interfaces for:
- Batch/single inference
- Audio utilities (normalization, reference audio decoding)
- Metrics (SingMOS, AES)
- Visualization (score condition figures, alignment/note plots)
- SVSEvaluator: single source of truth for training validation + standalone inference
"""

from importlib import import_module

from vocalrender.evaluation.audio_utils import normalize_audio, decode_reference_audio

_LAZY_EXPORTS = {
    "run_inference_single": ("vocalrender.evaluation.inference", "run_inference_single"),
    "run_inference_batch": ("vocalrender.evaluation.inference", "run_inference_batch"),
    "load_singmos_predictor": ("vocalrender.evaluation.metrics", "load_singmos_predictor"),
    "SingMOSFrameCapture": ("vocalrender.evaluation.metrics", "SingMOSFrameCapture"),
    "compute_singmos_score": ("vocalrender.evaluation.metrics", "compute_singmos_score"),
    "compute_batch_singmos_scores": ("vocalrender.evaluation.metrics", "compute_batch_singmos_scores"),
    "SVSEvaluator": ("vocalrender.evaluation.svs_metrics", "SVSEvaluator"),
    "EvalItem": ("vocalrender.evaluation.svs_metrics", "EvalItem"),
    "EvalResult": ("vocalrender.evaluation.svs_metrics", "EvalResult"),
    "items_from_infer_results": ("vocalrender.evaluation.svs_metrics", "items_from_infer_results"),
    "items_from_train_buffers": ("vocalrender.evaluation.svs_metrics", "items_from_train_buffers"),
    "create_score_condition_figure": ("vocalrender.evaluation.visualization", "create_score_condition_figure"),
}


def __getattr__(name):
    """Load validation metrics only when training/evaluation explicitly asks."""
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _LAZY_EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value

__all__ = [
    "normalize_audio",
    "decode_reference_audio",
    "run_inference_single",
    "run_inference_batch",
    "load_singmos_predictor",
    "SingMOSFrameCapture",
    "compute_singmos_score",
    "compute_batch_singmos_scores",
    "create_score_condition_figure",
    "SVSEvaluator",
    "EvalItem",
    "EvalResult",
    "items_from_infer_results",
    "items_from_train_buffers",
]
