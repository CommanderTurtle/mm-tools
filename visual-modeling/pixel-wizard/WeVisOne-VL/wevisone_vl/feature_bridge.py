"""Capture Qwen3-VL features before every spatial merger.

Qwen orders patches as ``[block_y, block_x, intra_y, intra_x]`` so each group of
``merge_size**2`` adjacent tokens can be consumed by its merger.  The helpers in
this file capture the inputs of the three DeepStack mergers and the final merger,
then undo that shuffle before exposing ordinary row-major 2-D feature maps.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch


def premerge_grid_hw(image_grid_thw: torch.Tensor) -> List[Tuple[int, int]]:
    """Per-image patch grid before Qwen's spatial merger."""
    grids = []
    for t, h, w in image_grid_thw.tolist():
        if int(t) != 1:
            raise ValueError(f"video not supported by segmentation decoder, got t={t}")
        grids.append((int(h), int(w)))
    return grids


def split_concat_tokens(tokens: torch.Tensor, counts: Sequence[int]) -> List[torch.Tensor]:
    """Split a concatenated [sum N_i, C] tensor into per-image [N_i, C]."""
    if tokens is None:
        return []
    if sum(counts) != tokens.shape[0]:
        raise ValueError(
            f"token count mismatch: sum(counts)={sum(counts)} vs tokens={tokens.shape[0]}"
        )
    return list(torch.split(tokens, list(counts), dim=0))


def assert_cnn_rgb(
    image: torch.Tensor,
    cnn_image_size: int,
    sample_idx: int = 0,
) -> torch.Tensor:
    """Require decoder CNN RGB ``[1,3,S,S]`` / ``[3,S,S]`` (anisotropic orig resize).

    This canvas is **not** the Qwen smart-resize used by the vision encoder.
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.shape[1] != 3:
        raise ValueError(
            f"sample {sample_idx}: decoder image must be [1,3,H,W] or [3,H,W], "
            f"got {tuple(image.shape)}"
        )
    got = (int(image.shape[-2]), int(image.shape[-1]))
    expect = (int(cnn_image_size), int(cnn_image_size))
    if got != expect:
        raise ValueError(
            f"sample {sample_idx}: decoder CNN image HW {got} != {expect}. "
            "Resize the original photo to cnn_image_size x cnn_image_size "
            "(not Qwen smart-resize)."
        )
    return image


def premerge_tokens_to_2d(
    feature_tokens: torch.Tensor,
    grid_hw: Tuple[int, int],
    spatial_merge_size: int,
) -> torch.Tensor:
    """Undo Qwen's merge-local patch shuffle and return ``[1,C,H,W]``.

    The processor flattens patches in ``[H/m,W/m,m,m]`` order.  A direct
    ``reshape(H,W)`` silently scrambles every ``m x m`` neighbourhood.
    """
    h, w = (int(grid_hw[0]), int(grid_hw[1]))
    m = int(spatial_merge_size)
    if h % m or w % m:
        raise ValueError(f"pre-merge grid {(h, w)} is not divisible by merge size {m}")
    if feature_tokens.dim() != 2 or int(feature_tokens.shape[0]) != h * w:
        raise ValueError(
            f"pre-merge tokens {tuple(feature_tokens.shape)} do not match grid {h}x{w}"
        )
    c = int(feature_tokens.shape[-1])
    x = feature_tokens.reshape(h // m, w // m, m, m, c)
    x = x.permute(0, 2, 1, 3, 4).contiguous().reshape(h, w, c)
    return x.permute(2, 0, 1).unsqueeze(0)


def extract_t_seg(
    last_hidden_state: torch.Tensor,
    input_ids: torch.Tensor,
    seg_token_id: int,
) -> List[torch.Tensor]:
    """Per-sample LM hidden states at ``<OBJ>`` positions."""
    result = []
    for b in range(input_ids.shape[0]):
        pos = (input_ids[b] == seg_token_id).nonzero(as_tuple=False).flatten()
        result.append(last_hidden_state[b, pos])
    return result


class VisualFeatureCapture:
    """Capture layers 5/11/17/23 exactly at their merger inputs."""

    def __init__(self, qwen, feature_layers: Sequence[int] = (5, 11, 17, 23)):
        self.qwen = qwen
        self.feature_layers = tuple(int(x) for x in feature_layers)
        self.premerge_features = {}
        self._handles = []

    def _capture_premerge(self, layer_idx: int):
        def hook(_module, inputs):
            if not inputs:
                raise RuntimeError(f"empty merger input for ViT layer {layer_idx}")
            self.premerge_features[int(layer_idx)] = inputs[0]

        return hook

    def __enter__(self):
        visual = self._get_visual()
        self.premerge_features.clear()
        deep_indexes = tuple(int(x) for x in visual.deepstack_visual_indexes)
        mergers = visual.deepstack_merger_list
        final_idx = len(visual.blocks) - 1
        available = set(deep_indexes) | {final_idx}
        missing = [x for x in self.feature_layers if x not in available]
        if missing:
            raise RuntimeError(
                f"cannot capture pre-merge ViT layers {missing}; merger inputs exist at "
                f"{sorted(available)}"
            )
        for idx, merger in zip(deep_indexes, mergers):
            if idx in self.feature_layers:
                self._handles.append(
                    merger.register_forward_pre_hook(self._capture_premerge(idx))
                )
        if final_idx in self.feature_layers:
            self._handles.append(
                visual.merger.register_forward_pre_hook(
                    self._capture_premerge(final_idx)
                )
            )
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return False

    def _get_visual(self):
        root = self.qwen
        base = getattr(root, "model", root)
        visual = getattr(base, "visual", None) or getattr(root, "visual", None)
        if visual is None:
            raise RuntimeError("cannot locate qwen visual module for feature capture")
        return visual


class LastHiddenCapture:
    """Hook the last LM decoder layer instead of using ``output_hidden_states=True``.

    ``output_hidden_states=True`` materialises every layer's [B,L,H] while the
    mask decoder only needs the final hidden state.
    """

    def __init__(self, qwen):
        self.qwen = qwen
        self.hidden = None
        self._handle = None

    def _hook(self, _module, _inputs, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        self.hidden = h

    def __enter__(self):
        layer = self._get_last_layer()
        self._handle = layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None
        return False

    def _get_last_layer(self):
        root = self.qwen
        # Qwen3-VL / Qwen2-VL common layouts
        candidates = []
        model = getattr(root, "model", root)
        candidates.append(getattr(getattr(model, "language_model", None), "layers", None))
        candidates.append(getattr(model, "layers", None))
        inner = getattr(model, "model", None)
        if inner is not None:
            candidates.append(getattr(getattr(inner, "language_model", None), "layers", None))
            candidates.append(getattr(inner, "layers", None))
        for layers in candidates:
            if layers is not None and len(layers) > 0:
                return layers[-1]
        raise RuntimeError("cannot locate last LM decoder layer for LastHiddenCapture")


def per_sample_premerge_feature_maps(
    premerge_features: Dict[int, torch.Tensor],
    feature_layers: Sequence[int],
    image_grid_thw: torch.Tensor,
    spatial_merge_size: int,
) -> List[List[torch.Tensor]]:
    """Split concatenated raw ViT tokens and restore row-major feature maps."""
    grids = premerge_grid_hw(image_grid_thw)
    counts = [h * w for h, w in grids]
    ordered = []
    for idx in feature_layers:
        feat = premerge_features.get(int(idx))
        if feat is None:
            raise RuntimeError(f"ViT layer {idx} was not captured before its merger")
        ordered.append(split_concat_tokens(feat, counts))

    per_sample: List[List[torch.Tensor]] = []
    for sample_idx, grid in enumerate(grids):
        per_sample.append(
            [
                premerge_tokens_to_2d(level[sample_idx], grid, spatial_merge_size)
                for level in ordered
            ]
        )
    return per_sample
