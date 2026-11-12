"""Small text and camera helpers for segment self-forcing SFT."""

import json
import random
import re
from typing import List, Optional


def _sample_segment_camera_keep(
    *,
    num_segments: int,
    dropout_p: float,
    min_on: int,
    global_step: int,
    sample_idx: int,
) -> List[bool]:
    """Decide which segments stay camera-ON for one multi-segment camera sample.

    Returns a length-``num_segments`` bool list (True = keep camera, False =
    drop camera / treat as prompt-controlled). Decisions are made with a
    deterministic per-(global_step, sample_idx) RNG so that **all SP ranks
    within a group derive the identical mask** (they share global_step and the
    packed batch order), avoiding a desync where ranks gate different tokens.

    Each segment is dropped *independently* with probability ``dropout_p``
    (Bernoulli per segment), then two hard constraints are enforced:
      - at least one segment is dropped, and
      - at least ``min_on`` (>= 1) segments are kept.

    No-op cases return all-True (the constraints cannot be satisfied):
      - ``dropout_p <= 0`` or ``num_segments <= 1``
      - ``num_segments <= min_on`` (can't keep ``min_on`` and still drop one)
    """
    if num_segments <= 1 or dropout_p <= 0.0:
        return [True] * max(num_segments, 1)
    min_on = max(1, int(min_on))
    if num_segments <= min_on:
        return [True] * num_segments

    rng = random.Random()
    rng.seed(
        (int(global_step) * 1_000_003 + int(sample_idx) * 9_176 + 12_345) & 0x7FFFFFFF
    )

    # Independent per-segment Bernoulli drop: keep with prob (1 - dropout_p).
    keep = [rng.random() >= dropout_p for _ in range(num_segments)]

    # Enforce "at least one dropped": if everything was kept, drop one at random.
    if all(keep):
        keep[rng.randrange(num_segments)] = False

    # Enforce "at least min_on kept": if too few survived, turn some back on.
    # Since num_segments > min_on, this still leaves >= 1 dropped segment.
    n_on = sum(keep)
    if n_on < min_on:
        off_idx = [j for j, k in enumerate(keep) if not k]
        for j in rng.sample(off_idx, min_on - n_on):
            keep[j] = True

    return keep


_CAMERA_TEXT_BLOCK_RE = re.compile(
    r"\s*<Camera\d+>.*?</Camera\d+>\s*",
    flags=re.DOTALL,
)


def _strip_camera_text_blocks(text: str) -> str:
    """Remove complete camera-caption blocks from one serialized prompt."""
    if not isinstance(text, str) or "<Camera" not in text:
        return text
    cleaned = _CAMERA_TEXT_BLOCK_RE.sub(" ", text)
    return re.sub(r"\s+", " ", cleaned).strip()


def _text_tokens(text_encoder, text: str, text_len: int):
    text_inputs = text_encoder.text2tokens(text, data_type="video", max_length=text_len)
    return (
        text_inputs["input_ids"].squeeze(0),
        text_inputs["attention_mask"].squeeze(0),
    )


def _text_crop_start(text_encoder) -> int:
    try:
        tmpl = getattr(text_encoder, "prompt_template_video", None)
        if isinstance(tmpl, dict):
            cs = int(tmpl.get("crop_start", 0) or 0)
            return cs if cs > 0 else 0
    except Exception:
        pass
    return 0


def _action_prefix_prompts(prompt: str) -> Optional[List[str]]:
    """Build JSON prompt variants with 0..A-1 contained actions kept."""
    try:
        obj = json.loads(prompt)
    except Exception:
        return None
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segments, list) or len(segments) != 1:
        return None
    seg = segments[0]
    if not isinstance(seg, dict):
        return None
    actions = seg.get("contained_actions")
    if not isinstance(actions, list) or not actions:
        return None

    def _dump(seg_obj: dict) -> str:
        out = dict(obj)
        out["segments"] = [seg_obj]
        return json.dumps(out, ensure_ascii=False)

    seg_no_actions = {k: v for k, v in seg.items() if k != "contained_actions"}
    prefixes = [_dump(seg_no_actions)]
    for k in range(1, len(actions)):
        seg_k = dict(seg)
        seg_k["contained_actions"] = actions[:k]
        prefixes.append(_dump(seg_k))
    return prefixes


def _compute_action_offsets_for_prompt(
    text_encoder,
    full_text: str,
    text_len: int,
) -> Optional[List[int]]:
    prefixes = _action_prefix_prompts(full_text)
    if not prefixes:
        return None

    full_ids, full_mask = _text_tokens(text_encoder, full_text, text_len)
    full_list = full_ids.tolist()
    crop_start = _text_crop_start(text_encoder)
    valid = int(full_mask.sum().item())

    offsets = []
    for variant in prefixes:
        v_ids, _ = _text_tokens(text_encoder, variant, text_len)
        v_list = v_ids.tolist()
        n = min(len(full_list), len(v_list))
        common = 0
        while common < n and full_list[common] == v_list[common]:
            common += 1
        off = max(0, common - crop_start)
        offsets.append(min(int(off), text_len - 1))

    cleaned = []
    prev = 0
    for off in offsets:
        off = min(max(off, prev), valid)
        cleaned.append(off)
        prev = off
    cleaned.append(min(valid, text_len))
    return cleaned
