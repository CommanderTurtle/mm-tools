"""Choose the execution mode for a complete offline sample, never model loading."""
from contextlib import contextmanager
import torch


@contextmanager
def offline_sample_mode(enabled, *, preserve_kv_cache=False, segment_callback=None,
                        interactive=False):
    if enabled and (preserve_kv_cache or segment_callback is not None or interactive):
        raise ValueError("inference_mode requires a complete offline sample")
    context = torch.inference_mode if enabled else torch.no_grad
    with context():
        yield
