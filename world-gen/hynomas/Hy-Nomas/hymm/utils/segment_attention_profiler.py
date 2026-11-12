"""
Segment Attention Profiler — visualize per-head attention distributions
across layers and denoising steps for segment-level causal attention.

Focuses on Segment 1's FIRST and LAST latent frame with 9 decomposed scores:

  First-frame group (softmax-normalised, sum to 1):
    0: first_frame → seg0_vid
    1: first_frame → seg0_txt
    2: first_frame → seg1_txt
    3: first_frame → seg1_vid

  Last-frame group (softmax-normalised, sum to 1):
    4: last_frame → seg0_vid
    5: last_frame → seg0_txt
    6: last_frame → seg1_txt
    7: last_frame → seg1_vid

  Text group:
    8: seg1_txt → seg1_txt  (sigmoid self-fraction)

Usage:
    from hymm.utils.segment_attention_profiler import seg_attn_profiler

    seg_attn_profiler.set_total_steps(num_inference_steps)

    for i, t in enumerate(timesteps):
        seg_attn_profiler.new_step(i, t.item())
        output = model(...)

    seg_attn_profiler.plot_and_save('output/seg_attn_profile.png')

Enable via: PROFILE_SEGMENT_ATTENTION_VIS=1
"""

import os
import torch
from typing import Dict, List, Optional

NUM_SCORES = 9

SCORE_LABELS = [
    "1st→s0v",
    "1st→s0t",
    "1st→s1t",
    "1st→s1v",
    "lst→s0v",
    "lst→s0t",
    "lst→s1t",
    "lst→s1v",
    "txt→txt",
]

# Major group boundaries for drawing thick separators on heatmap
# (after first-frame block and after last-frame block)
_GROUP_BOUNDARIES = [4, 8]


class SegmentAttentionProfiler:
    """Collects per-layer, per-step segment-1 attention scores for visualization."""

    def __init__(self):
        self.enabled = os.environ.get("PROFILE_SEGMENT_ATTENTION_VIS", "") == "1"
        self.data: Dict[int, Dict[int, torch.Tensor]] = {}
        self.timesteps: Dict[int, float] = {}
        self._current_step = -1
        self._layer_counter = 0
        self._total_steps = -1
        self._steps_to_profile: Optional[set] = None
        self._profiling_active = True
        self.tokens_per_frame: int = 0

    def set_total_steps(self, total_steps: int):
        """Pre-select 3 steps to profile (high/mid/low noise). Call before the loop."""
        if not self.enabled:
            return
        self._total_steps = total_steps
        if total_steps >= 3:
            self._steps_to_profile = {0, total_steps // 2, total_steps - 1}
        else:
            self._steps_to_profile = set(range(total_steps))
        rank = int(os.environ.get("RANK", 0))
        if rank <= 0:
            print(
                f"[SegAttnProfiler] Enabled. Will profile steps: "
                f"{sorted(self._steps_to_profile)}"
            )

    def new_step(self, step_idx: int, timestep: float = 0.0):
        if not self.enabled:
            return
        self._current_step = step_idx
        self._layer_counter = 0
        self.timesteps[step_idx] = timestep

    def should_profile(self) -> bool:
        if not self.enabled or not self._profiling_active or self._current_step < 0:
            return False
        if self._steps_to_profile is not None:
            return self._current_step in self._steps_to_profile
        return True

    def record(self, scores: torch.Tensor):
        """Record (NUM_SCORES, H) scores for current step / auto-incremented layer."""
        if not self.should_profile():
            return
        if self._current_step not in self.data:
            self.data[self._current_step] = {}
        self.data[self._current_step][self._layer_counter] = scores.detach().cpu()
        self._layer_counter += 1

    def reset(self):
        self.data.clear()
        self.timesteps.clear()
        self._current_step = -1
        self._layer_counter = 0

    def plot_and_save(self, save_path: str, steps_to_plot: Optional[List[int]] = None):
        if not self.data:
            print("[SegAttnProfiler] No data collected, skipping plot.")
            return
        _plot_heatmaps(self.data, self.timesteps, save_path, steps_to_plot)


seg_attn_profiler = SegmentAttentionProfiler()


# ---------------------------------------------------------------------------
# Score computation  (10 flash-attn calls under no_grad)
# ---------------------------------------------------------------------------


def compute_seg1_attention_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    encoder_query: torch.Tensor,
    encoder_key: torch.Tensor,
    encoder_value: torch.Tensor,
    prompt_video_boundaries: List[int],
    txt_tokens_per_block: int,
    softmax_scale: float,
    fa_func,
    tokens_per_frame: int = 0,
) -> Optional[torch.Tensor]:
    """
    Compute 9 LSE-based attention scores for segment 1.

    Returns (9, H) tensor or None.
      [0-3] first-frame video Q scores  (softmax over 4 sources, sum=1)
            →seg0_vid, →seg0_txt, →seg1_txt, →seg1_vid
      [4-7] last-frame  video Q scores  (softmax over 4 sources, sum=1)
            →seg0_vid, →seg0_txt, →seg1_txt, →seg1_vid
      [8]   seg1 text self-fraction      (sigmoid)
    """
    if fa_func is None or len(prompt_video_boundaries) < 3:
        return None

    vid_s0, vid_e0 = prompt_video_boundaries[0], prompt_video_boundaries[1]
    vid_s1, vid_e1 = prompt_video_boundaries[1], prompt_video_boundaries[2]
    L = txt_tokens_per_block
    tpf = tokens_per_frame

    if (vid_e0 - vid_s0) == 0 or (vid_e1 - vid_s1) == 0 or L == 0 or tpf <= 0:
        return None

    first_q = query[:, vid_s1 : vid_s1 + tpf]  # (B, tpf, H, D)
    last_q = query[:, vid_e1 - tpf : vid_e1]  # (B, tpf, H, D)
    txt1_q = encoder_query[:, L : 2 * L]  # (B, L,   H, D)

    kv_seg0_vid = (key[:, vid_s0:vid_e0], value[:, vid_s0:vid_e0])
    kv_seg0_txt = (encoder_key[:, :L], encoder_value[:, :L])
    kv_seg1_txt = (encoder_key[:, L : 2 * L], encoder_value[:, L : 2 * L])
    kv_seg1_vid = (key[:, vid_s1:vid_e1], value[:, vid_s1:vid_e1])

    other_k = torch.cat([key[:, :vid_e1], encoder_key[:, :L]], dim=1)
    other_v = torch.cat([value[:, :vid_e1], encoder_value[:, :L]], dim=1)

    def _lse(q, kv_pair):
        r = fa_func(
            q,
            kv_pair[0],
            kv_pair[1],
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        return r[1]  # (B, H, S_q)

    with torch.no_grad():
        # --- first frame: 4 sources ---
        f_lse = torch.stack(
            [
                _lse(first_q, kv_seg0_vid),
                _lse(first_q, kv_seg0_txt),
                _lse(first_q, kv_seg1_txt),
                _lse(first_q, kv_seg1_vid),
            ],
            dim=0,
        )  # (4, B, H, tpf)
        f_w = torch.softmax(f_lse.float(), dim=0).mean(dim=(1, 3))  # (4, H)

        # --- last frame: 4 sources ---
        l_lse = torch.stack(
            [
                _lse(last_q, kv_seg0_vid),
                _lse(last_q, kv_seg0_txt),
                _lse(last_q, kv_seg1_txt),
                _lse(last_q, kv_seg1_vid),
            ],
            dim=0,
        )
        l_w = torch.softmax(l_lse.float(), dim=0).mean(dim=(1, 3))  # (4, H)

        # --- text self-fraction ---
        lse_self = _lse(txt1_q, kv_seg1_txt)
        lse_other = fa_func(
            txt1_q,
            other_k,
            other_v,
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )[1]
        t_w = torch.sigmoid((lse_self - lse_other).float()).mean(dim=(0, 2))  # (H,)

    return torch.cat([f_w, l_w, t_w.unsqueeze(0)], dim=0)  # (9, H)


# ---------------------------------------------------------------------------
# AR pipeline variant  (KV-cache autoregressive inference)
# ---------------------------------------------------------------------------


def compute_seg1_attention_scores_ar(
    img_q: torch.Tensor,
    txt_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    history_k: torch.Tensor,
    history_v: torch.Tensor,
    tokens_per_frame: int,
    fa_func,
) -> Optional[torch.Tensor]:
    """
    AR-pipeline version: compute 9 attention scores for the current segment
    when exactly one previous segment is in the KV cache.

    Cache layout:  [seg0_img | seg0_txt]
    Current:       img = seg1_vid, txt = seg1_txt

    Returns (9, H) tensor — same layout as compute_seg1_attention_scores.
    """
    tpf = tokens_per_frame
    L = txt_k.size(1)
    cache_len = history_k.size(1)

    if tpf <= 0 or cache_len <= L or img_q.size(1) < tpf:
        return None

    head_dim = img_q.shape[-1]
    softmax_scale = head_dim**-0.5

    seg0_vid_len = cache_len - L
    kv_seg0_vid = (history_k[:, :seg0_vid_len], history_v[:, :seg0_vid_len])
    kv_seg0_txt = (history_k[:, seg0_vid_len:], history_v[:, seg0_vid_len:])
    kv_seg1_txt = (txt_k, txt_v)
    kv_seg1_vid = (img_k, img_v)

    first_q = img_q[:, :tpf]
    last_q = img_q[:, -tpf:]

    other_k = torch.cat([history_k, img_k], dim=1)
    other_v = torch.cat([history_v, img_v], dim=1)

    def _lse(q, kv_pair):
        r = fa_func(
            q,
            kv_pair[0],
            kv_pair[1],
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )
        return r[1]

    with torch.no_grad():
        f_lse = torch.stack(
            [
                _lse(first_q, kv_seg0_vid),
                _lse(first_q, kv_seg0_txt),
                _lse(first_q, kv_seg1_txt),
                _lse(first_q, kv_seg1_vid),
            ],
            dim=0,
        )
        f_w = torch.softmax(f_lse.float(), dim=0).mean(dim=(1, 3))

        l_lse = torch.stack(
            [
                _lse(last_q, kv_seg0_vid),
                _lse(last_q, kv_seg0_txt),
                _lse(last_q, kv_seg1_txt),
                _lse(last_q, kv_seg1_vid),
            ],
            dim=0,
        )
        l_w = torch.softmax(l_lse.float(), dim=0).mean(dim=(1, 3))

        lse_self = _lse(txt_q, kv_seg1_txt)
        lse_other = fa_func(
            txt_q,
            other_k,
            other_v,
            softmax_scale=softmax_scale,
            causal=False,
            return_attn_probs=True,
        )[1]
        t_w = torch.sigmoid((lse_self - lse_other).float()).mean(dim=(0, 2))

    return torch.cat([f_w, l_w, t_w.unsqueeze(0)], dim=0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _plot_heatmaps(
    data: Dict[int, Dict[int, torch.Tensor]],
    timesteps: Dict[int, float],
    save_path: str,
    steps_to_plot: Optional[List[int]] = None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import numpy as np

    all_steps = sorted(data.keys())
    n = len(all_steps)

    if steps_to_plot is None:
        if n >= 3:
            steps_to_plot = [all_steps[0], all_steps[n // 2], all_steps[-1]]
        elif n == 2:
            steps_to_plot = [all_steps[0], all_steps[-1]]
        else:
            steps_to_plot = list(all_steps)

    num_plots = len(steps_to_plot)
    noise_tags = ["High Noise (early)", "Mid Noise", "Low Noise (late)"]

    fig = plt.figure(figsize=(8 * num_plots + 1.5, 10))
    gs = gridspec.GridSpec(1, num_plots + 1, width_ratios=[1] * num_plots + [0.05])
    axes = [fig.add_subplot(gs[0, i]) for i in range(num_plots)]
    cax = fig.add_subplot(gs[0, num_plots])

    im_ref = None
    for ax_idx, step in enumerate(steps_to_plot):
        ax = axes[ax_idx]
        if step not in data:
            ax.set_title(f"Step {step}: no data")
            continue

        layers = sorted(data[step].keys())
        num_layers = len(layers)
        H = data[step][layers[0]].shape[1]

        matrix = np.zeros((num_layers, NUM_SCORES * H))
        for li, layer in enumerate(layers):
            matrix[li] = data[step][layer].numpy().flatten()

        im = ax.imshow(
            matrix,
            aspect="auto",
            cmap="RdYlBu_r",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        im_ref = im

        ts = timesteps.get(step, float(step))
        tag = noise_tags[ax_idx] if ax_idx < len(noise_tags) else f"Step {step}"
        ax.set_title(f"{tag}\nstep={step}, t={ts:.4f}", fontsize=11)
        ax.set_ylabel("Layer" if ax_idx == 0 else "", fontsize=10)

        # Major separators (first-frame | last-frame | text)
        for g in _GROUP_BOUNDARIES:
            ax.axvline(x=g * H - 0.5, color="white", linewidth=2.0)
        # Minor separators within each group of 3
        for g in range(1, NUM_SCORES):
            if g not in _GROUP_BOUNDARIES:
                ax.axvline(x=g * H - 0.5, color="white", linewidth=0.5, linestyle="--")

        # X-axis: one label per score group, centered
        centres = [H * i + H / 2 - 0.5 for i in range(NUM_SCORES)]
        ax.set_xticks(centres)
        ax.set_xticklabels(SCORE_LABELS, fontsize=7.5)

        # Add "First Frame" / "Last Frame" / "Txt" span labels
        y_off = num_layers + 0.8
        for label, x0, x1 in [
            ("First Frame", 0, 4 * H - 1),
            ("Last Frame", 4 * H, 8 * H - 1),
            ("Txt", 8 * H, 9 * H - 1),
        ]:
            cx = (x0 + x1) / 2
            ax.text(
                cx,
                y_off,
                label,
                ha="center",
                va="top",
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="gray", alpha=0.8),
            )

        ax.set_yticks(range(0, num_layers, max(1, num_layers // 10)))

    if im_ref is not None:
        fig.colorbar(im_ref, cax=cax, label="Attention Weight")

    fig.suptitle(
        "Seg1 Attention: First Frame vs Last Frame  (per-head × per-layer)",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SegAttnProfiler] Saved to {save_path}")
