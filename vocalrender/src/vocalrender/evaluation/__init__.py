"""
vocalrender.evaluation — Shared evaluation modules for SVS test and validation.

Provides unified interfaces for:
- Batch/single inference
- Audio utilities (normalization, reference audio decoding)
- Metrics (SingMOS, AES)
- Visualization (score condition figures, alignment/note plots)
- SVSEvaluator: single source of truth for training validation + standalone inference
"""

from vocalrender.evaluation.audio_utils import normalize_audio, decode_reference_audio
from vocalrender.evaluation.inference import run_inference_single, run_inference_batch
from vocalrender.evaluation.metrics import (
    load_singmos_predictor,
    SingMOSFrameCapture,
    compute_singmos_score,
    compute_batch_singmos_scores,
)
from vocalrender.evaluation.svs_metrics import (
    SVSEvaluator,
    EvalItem,
    EvalResult,
    items_from_infer_results,
    items_from_train_buffers,
)
from vocalrender.evaluation.visualization import create_score_condition_figure

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
