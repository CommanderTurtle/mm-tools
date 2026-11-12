"""User prompts for all text-prompted benchmarks, selected by ``dataset_name``.

Each template below defines the standard evaluation prompt protocol for its task.
"""

from __future__ import annotations

import re

TMPL_DETECTION = "Locate all the instances that match the following description: {labels}."
TMPL_REFERRING_SINGLE = (
    "Locate a single instance that best matches the following description: {phrase}"
)
TMPL_REFERRING_MULTI = "Locate the instances that match the following description: {phrase}"
TMPL_OCR = "detect all the {noun} in the box format"
TMPL_COUNT = "How many {label} in the image? Please count them with bounding boxes."
TMPL_POINT = "point to: {labels}"
TMPL_MATTE = {
    "single": (
        "\nLocate a single instance that best matches the following description and "
        "provide the alpha mattes: {expression}"
    ),
    "instances": (
        "\nLocate the instances that match the following description and provide "
        "the alpha mattes: {expression}"
    ),
}

# Referring benchmarks with at most one target per expression use the
# single-instance template. The other referring benchmarks may have several
# targets (HumanRef, Ref-Adv-s) or none (PR-Bench, RefCOCOg-rej-val).
REFERRING_SINGLE = {"RefCOCOg_val", "RefCOCOg_test", "DVGBench", "Ref-L4-all"}
REFERRING_MULTI = {"HumanRef", "Ref-Adv-s", "PR-Bench", "RefCOCOg-rej-val"}
OCR_DATASETS = {"HierText", "IC15", "TotalText", "SROIE"}
COUNTING_PREFIXES = ("CountQA", "pixmo-count", "cloc-", "KubriCount", "MixCount")

# Detection benchmarks that are prompted one category at a time.
PER_CATEGORY_DATASETS = {"COCO", "LVIS", "Dense200", "VisDrone"}

# Native pointing benchmarks: the query is the benchmark's own question.
POINTING_BENCHMARKS = {"PointBench", "PixMo-Points-Eval"}


def is_counting(dataset: str) -> bool:
    return dataset.startswith(COUNTING_PREFIXES)


def build_prompt(dataset: str, categories: list[str], *, pointing: bool = False) -> str:
    """Prompt for one annotation row whose targets are ``categories``."""
    labels = ". ".join(str(category) for category in categories)
    first = str(categories[0]) if categories else ""
    if is_counting(dataset):
        return TMPL_COUNT.format(label=first)
    if pointing:
        return TMPL_POINT.format(labels=labels)
    if dataset in REFERRING_SINGLE or dataset in REFERRING_MULTI:
        if first[-1:] not in ".?!":
            first += "."
        template = TMPL_REFERRING_SINGLE if dataset in REFERRING_SINGLE else TMPL_REFERRING_MULTI
        return template.format(phrase=first)
    if dataset in OCR_DATASETS:
        return TMPL_OCR.format(noun=first or "text")
    return TMPL_DETECTION.format(labels=labels)


def pointing_bench_prompt(question: str) -> str:
    """``point to: <query>`` with any leading pointing imperative removed."""
    query = re.sub(
        r"^(?:please\s+)?point(?:\s+out)?(?:\s+to)?\s*:?\s*",
        "",
        question.strip(),
        count=1,
        flags=re.IGNORECASE,
    ).strip()
    return TMPL_POINT.format(labels=query or question.strip())
