#!/usr/bin/env python3
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""VBench sampling protocol: prompt suites, sample jobs, seeds, and MP4 contract.

The paper protocol (``fastframes_extended_vbench_v1``) conditions the
generator on Self-Forcing's row-paired extended prompts, names every video by
its original VBench prompt, and generates 5 samples per prompt (25 candidates
for the 75 temporal-flickering prompts, later reduced to at most 5 by the RAFT
static filter). This yields 6,220 raw videos and 4,720 scored slots.

Usage:
    python protocol.py summary                 # counts and fingerprints
    python protocol.py jobs --output jobs.jsonl
    python protocol.py verify --mode raw --videos-dir RUN/raw_videos --output RUN/raw_corpus_verification.json
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HERE = Path(__file__).resolve().parent
try:
    from fastframes import mp4
except ImportError:  # the VBench scorer environment does not install FastFrames; mp4 is standard library only
    sys.path.insert(0, str(HERE.parents[1]))
    from fastframes import mp4
DEFAULT_FULL_INFO = HERE / "prompts" / "VBench_full_info.json"
DEFAULT_ORIGINAL_PROMPTS = HERE / "prompts" / "all_dimension.txt"
DEFAULT_EXTENDED_PROMPTS = HERE / "prompts" / "all_dimension_extended.txt"
CAUSVID_PROMPTS = HERE / "prompts" / "all_dimension_longer.txt"

RAW_PROTOCOL = "fastframes_original_vbench_v1"
EXTENDED_PROTOCOL = "fastframes_extended_vbench_v1"
CAUSVID_PROTOCOL = "fastframes_causvid_extended_vbench_v1"

EXPECTED_FULL_INFO_SHA256 = "5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd"
EXPECTED_ORIGINAL_PROMPTS_SHA256 = "f9b50654f81ab732b9235d5bb91f0c3a4d152fac19249ce3a3a9185b2d80d8c3"
EXPECTED_EXTENDED_PROMPTS_SHA256 = "c78f1ad32974503efaa8cc935347ace88f3a8b571adde6687bc8425bbf417cf3"
EXPECTED_RESOLVED_PROMPT_PAIR_SHA256 = "89b1fa74afe3f90eca8ff18f912853e7018a32370b40d98f6f3d72e504a6954e"
EXPECTED_SEED_MAP_SHA256 = "ff2e8b97f5e125a0728672d76776453618db35711919c6e651ef1213a937b528"
# Conditioning variants: protocol -> (default prompt file, its sha256, resolved prompt-pair sha256).
# CausVid uses its VBench GPT-enhanced prompts.
CONDITIONING = {
    EXTENDED_PROTOCOL: (DEFAULT_EXTENDED_PROMPTS, EXPECTED_EXTENDED_PROMPTS_SHA256,
                        EXPECTED_RESOLVED_PROMPT_PAIR_SHA256),
    CAUSVID_PROTOCOL: (CAUSVID_PROMPTS, "58ed0665de52c1bdb4820af01327f5cc1138bedbeedcc63902751056f6bf5768",
                       "8ee29b863b8c17e722e0afc82d5d56300d63cdbd8a685983439d9d32f619aaa8"),
}

EXPECTED_METADATA_ROWS = 946
EXPECTED_UNIQUE_PROMPTS = 944
EXPECTED_TEMPORAL_PROMPTS = 75
BASE_SAMPLES = 5
TEMPORAL_CANDIDATES = 25
EXPECTED_RAW_VIDEOS = 6_220
EXPECTED_SCORING_VIDEOS = 4_720
EXPECTED_TEMPORAL_FILTERED_VIDEOS = 375
EXPECTED_LATENT_FRAMES = 21
EXPECTED_RGB_FRAMES = 81
EXPECTED_WIDTH = 832
EXPECTED_HEIGHT = 480
EXPECTED_FPS = 16

# Two original prompts occur twice with different rewrites. Filenames collapse
# each pair to one video, so the last occurrence (one-based row) is used.
DUPLICATE_ROWS = {
    "A beautiful coastal beach in spring, waves lapping on sand by Hokusai, in the style of Ukiyo": (496, 749),
    "The bund Shanghai, oil painting": (504, 747),
}


@dataclass(frozen=True)
class PromptSuite:
    rows: int
    prompts: tuple[str, ...]
    temporal_prompts: frozenset[str]
    full_info_path: Path
    full_info_sha256: str
    protocol_identifier: str = RAW_PROTOCOL
    conditioning_prompts: tuple[str, ...] | None = None
    original_prompt_sha256: str | None = None
    conditioning_prompt_sha256: str | None = None
    resolved_source_rows_one_based: tuple[int, ...] | None = None
    prompt_pair_sha256: str | None = None


@dataclass(frozen=True)
class SampleSpec:
    prompt: str
    conditioning_prompt: str
    prompt_ordinal: int
    sample_index: int
    seed: int
    temporal_flickering: bool

    @property
    def filename(self) -> str:
        return f"{self.prompt}-{self.sample_index}.mp4"


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _check_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash changed: {actual} != {expected}")
    return actual


def load_prompt_suite(full_info_path: Path = DEFAULT_FULL_INFO) -> PromptSuite:
    """Load the 944 unique VBench prompts and the 75 temporal-flickering prompts."""
    full_info_path = Path(full_info_path).resolve()
    actual_hash = _check_hash(full_info_path, EXPECTED_FULL_INFO_SHA256, "VBench_full_info.json")
    rows = json.loads(full_info_path.read_text(encoding="utf-8"))
    if len(rows) != EXPECTED_METADATA_ROWS:
        raise ValueError(f"VBench metadata rows changed: {len(rows)} != {EXPECTED_METADATA_ROWS}")

    prompts: dict[str, None] = {}
    temporal: set[str] = set()
    non_temporal: set[str] = set()
    for row in rows:
        prompt = str(row["prompt_en"])
        if not prompt or "\x00" in prompt or os.sep in prompt:
            raise ValueError(f"unsafe VBench prompt filename: {prompt!r}")
        dimensions = set(row["dimension"])
        prompts.setdefault(prompt)
        if "temporal_flickering" in dimensions:
            temporal.add(prompt)
        if dimensions - {"temporal_flickering"}:
            non_temporal.add(prompt)

    if len(prompts) != EXPECTED_UNIQUE_PROMPTS:
        raise ValueError(f"unique VBench prompts changed: {len(prompts)} != {EXPECTED_UNIQUE_PROMPTS}")
    if len(temporal) != EXPECTED_TEMPORAL_PROMPTS:
        raise ValueError(f"temporal prompt count changed: {len(temporal)} != {EXPECTED_TEMPORAL_PROMPTS}")
    if temporal & non_temporal:
        raise ValueError("temporal-flickering prompts unexpectedly overlap other dimensions")
    return PromptSuite(
        rows=len(rows),
        prompts=tuple(prompts),
        temporal_prompts=frozenset(temporal),
        full_info_path=full_info_path,
        full_info_sha256=actual_hash,
    )


def prompt_pair_fingerprint(
    prompts: Sequence[str], conditioning_prompts: Sequence[str], rows_one_based: Sequence[int]
) -> str:
    if not len(prompts) == len(conditioning_prompts) == len(rows_one_based):
        raise ValueError("prompt-pair fingerprint inputs differ in length")
    digest = hashlib.sha256()
    for prompt, conditioning_prompt, row in zip(prompts, conditioning_prompts, rows_one_based):
        digest.update(prompt.encode("utf-8") + b"\0")
        digest.update(conditioning_prompt.encode("utf-8") + b"\0")
        digest.update(str(row).encode("ascii") + b"\n")
    return digest.hexdigest()


def _prompt_lines(path: Path) -> tuple[str, ...]:
    with path.open(encoding="utf-8") as handle:
        return tuple(line.rstrip("\r\n") for line in handle)


def load_extended_prompt_suite(
    full_info_path: Path = DEFAULT_FULL_INFO,
    original_prompt_path: Path = DEFAULT_ORIGINAL_PROMPTS,
    extended_prompt_path: Path | None = None,
    protocol_identifier: str = EXTENDED_PROTOCOL,
) -> PromptSuite:
    """Pair each original VBench prompt with its extended rewrite (``CONDITIONING[protocol_identifier]``).

    Rows are paired by ordinal. The original prompt names the video and is used
    for scoring; the extended prompt conditions the generator. For the two
    duplicated originals, the last occurrence wins.
    """
    default_path, expected_extended_hash, expected_pair_hash = CONDITIONING[protocol_identifier]
    base = load_prompt_suite(full_info_path)
    original_prompt_path = Path(original_prompt_path).resolve()
    extended_prompt_path = Path(extended_prompt_path or default_path).resolve()
    original_hash = _check_hash(original_prompt_path, EXPECTED_ORIGINAL_PROMPTS_SHA256, "original prompt")
    extended_hash = _check_hash(extended_prompt_path, expected_extended_hash, "extended prompt")
    originals = _prompt_lines(original_prompt_path)
    extended = _prompt_lines(extended_prompt_path)
    if len(originals) != EXPECTED_METADATA_ROWS or len(extended) != EXPECTED_METADATA_ROWS:
        raise ValueError(f"prompt row counts changed: original={len(originals)} extended={len(extended)}")
    rows = json.loads(base.full_info_path.read_text(encoding="utf-8"))
    if originals != tuple(str(row["prompt_en"]) for row in rows):
        raise ValueError("original prompt file differs from VBench metadata")

    conditioning_by_prompt: dict[str, str] = {}
    source_row_by_prompt: dict[str, int] = {}
    counts: dict[str, int] = {}
    for row, (prompt, conditioning_prompt) in enumerate(zip(originals, extended), start=1):
        if not conditioning_prompt:
            raise ValueError(f"empty extended prompt at row {row}")
        conditioning_by_prompt[prompt] = conditioning_prompt
        source_row_by_prompt[prompt] = row
        counts[prompt] = counts.get(prompt, 0) + 1
    duplicates = {prompt: count for prompt, count in counts.items() if count > 1}
    if duplicates != {prompt: len(rows) for prompt, rows in DUPLICATE_ROWS.items()}:
        raise ValueError(f"duplicate-prompt set changed: {duplicates}")
    for prompt, duplicate_rows in DUPLICATE_ROWS.items():
        if source_row_by_prompt[prompt] != duplicate_rows[-1]:
            raise AssertionError(f"unexpected last-row resolution for {prompt!r}")

    conditioning_prompts = tuple(conditioning_by_prompt[prompt] for prompt in base.prompts)
    resolved_rows = tuple(source_row_by_prompt[prompt] for prompt in base.prompts)
    pair_hash = prompt_pair_fingerprint(base.prompts, conditioning_prompts, resolved_rows)
    if pair_hash != expected_pair_hash:
        raise ValueError(f"resolved prompt-pair hash changed: {pair_hash}")
    return PromptSuite(
        rows=base.rows,
        prompts=base.prompts,
        temporal_prompts=base.temporal_prompts,
        full_info_path=base.full_info_path,
        full_info_sha256=base.full_info_sha256,
        protocol_identifier=protocol_identifier,
        conditioning_prompts=conditioning_prompts,
        original_prompt_sha256=original_hash,
        conditioning_prompt_sha256=extended_hash,
        resolved_source_rows_one_based=resolved_rows,
        prompt_pair_sha256=pair_hash,
    )


def load_suite(
    protocol_identifier: str = EXTENDED_PROTOCOL,
    full_info_path: Path = DEFAULT_FULL_INFO,
    original_prompt_path: Path = DEFAULT_ORIGINAL_PROMPTS,
    extended_prompt_path: Path | None = None,
) -> PromptSuite:
    if protocol_identifier not in CONDITIONING:
        raise ValueError(f"unsupported VBench protocol: {protocol_identifier}")
    return load_extended_prompt_suite(full_info_path, original_prompt_path, extended_prompt_path, protocol_identifier)


def build_sample_specs(suite: PromptSuite, *, base_seed: int = 0) -> list[SampleSpec]:
    """Build the 6,220 generation jobs with globally unique, sharding-independent seeds."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    conditioning_prompts = suite.conditioning_prompts or suite.prompts
    jobs: list[SampleSpec] = []
    for ordinal, (prompt, conditioning_prompt) in enumerate(zip(suite.prompts, conditioning_prompts)):
        temporal = prompt in suite.temporal_prompts
        for sample_index in range(TEMPORAL_CANDIDATES if temporal else BASE_SAMPLES):
            jobs.append(SampleSpec(
                prompt=prompt,
                conditioning_prompt=conditioning_prompt,
                prompt_ordinal=ordinal,
                sample_index=sample_index,
                seed=base_seed + ordinal * TEMPORAL_CANDIDATES + sample_index,
                temporal_flickering=temporal,
            ))
    if len(jobs) != EXPECTED_RAW_VIDEOS:
        raise AssertionError(f"raw job count {len(jobs)} != {EXPECTED_RAW_VIDEOS}")
    if len({job.seed for job in jobs}) != len(jobs) or len({job.filename for job in jobs}) != len(jobs):
        raise AssertionError("seeds or filenames are not unique")
    return jobs


def seed_map_fingerprint(jobs: Sequence[SampleSpec]) -> str:
    digest = hashlib.sha256()
    for job in jobs:
        digest.update(job.filename.encode("utf-8") + b"\0" + str(job.seed).encode("ascii") + b"\n")
    return digest.hexdigest()


def protocol_summary(suite: PromptSuite, *, base_seed: int = 0) -> dict[str, Any]:
    jobs = build_sample_specs(suite, base_seed=base_seed)
    payload = {
        "protocol": suite.protocol_identifier,
        "full_info_sha256": suite.full_info_sha256,
        "metadata_rows": suite.rows,
        "unique_prompts": len(suite.prompts),
        "temporal_flickering_prompts": len(suite.temporal_prompts),
        "base_samples_per_prompt": BASE_SAMPLES,
        "temporal_candidates_per_prompt": TEMPORAL_CANDIDATES,
        "raw_video_count": len(jobs),
        "post_filter_scoring_video_count": EXPECTED_SCORING_VIDEOS,
        "seed_formula": "base_seed + prompt_ordinal * 25 + sample_index",
        "base_seed": base_seed,
        "seed_map_sha256": seed_map_fingerprint(jobs),
        "latent_frames": EXPECTED_LATENT_FRAMES,
        "rgb_frames": EXPECTED_RGB_FRAMES,
        "width": EXPECTED_WIDTH,
        "height": EXPECTED_HEIGHT,
        "fps": EXPECTED_FPS,
    }
    if suite.protocol_identifier in CONDITIONING:
        payload["prompt_conditioning"] = {
            "original_prompts_sha256": suite.original_prompt_sha256,
            "extended_prompts_sha256": suite.conditioning_prompt_sha256,
            "pairing": "strict_row_ordinal",
            "duplicate_policy": "last_occurrence_wins",
            "selected_duplicate_rows_one_based": {p: rows[-1] for p, rows in DUPLICATE_ROWS.items()},
            "resolved_prompt_pair_sha256": suite.prompt_pair_sha256,
        }
    return payload


def require_encoder() -> None:
    """Fail early when the OpenH264 library (see ``fastframes.openh264``) cannot be loaded."""
    from fastframes import openh264

    try:
        openh264.load_library()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None


def _top_level_boxes(path: Path) -> list[str]:
    try:
        return mp4.top_level_boxes(path)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from None


def is_valid_video(path: Path) -> bool:
    """Whether ``path`` exists and passes ``verify_video`` (used to skip finished jobs)."""
    try:
        verify_video(path, hash_file=False)
        return True
    except Exception:
        return False


def verify_video(path: Path, *, hash_file: bool = True) -> dict[str, Any]:
    """Check codec, geometry, frame count, frame rate, and faststart layout."""
    stream = mp4.probe(path)
    observed = {key: stream[key] for key in ("codec", "width", "height", "frames")}
    observed["fps"] = float(stream["fps"])
    expected = {"codec": "h264", "width": EXPECTED_WIDTH, "height": EXPECTED_HEIGHT,
                "frames": EXPECTED_RGB_FRAMES, "fps": float(EXPECTED_FPS)}
    if observed != expected:
        raise RuntimeError(f"media contract mismatch for {path}: {observed} != {expected}")
    boxes = _top_level_boxes(path)
    if "moov" not in boxes or "mdat" not in boxes or boxes.index("moov") > boxes.index("mdat"):
        raise RuntimeError(f"MP4 is not faststart: {path}; boxes={boxes}")
    return {**observed, "faststart": True, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path) if hash_file else None}


def write_faststart_video(output_path: Path, frames_uint8: Any, *, fps: int = EXPECTED_FPS) -> dict[str, Any]:
    """Encode ``frames_uint8`` ([T, H, W, 3] uint8) to a verified faststart H.264 MP4 (OpenH264)."""
    from fastframes.video import save_video

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tag = f"{os.getpid()}-{hashlib.sha256(os.fsencode(output_path.name)).hexdigest()[:16]}"
    staged_path = output_path.parent / f".vbench-{tag}.mp4"
    try:
        save_video(staged_path, frames_uint8, fps=fps)
        verified = verify_video(staged_path)
        os.replace(staged_path, output_path)
        published = verify_video(output_path)
        if published["sha256"] != verified["sha256"]:
            raise RuntimeError(f"MP4 changed during rename: {output_path}")
        return published
    finally:
        staged_path.unlink(missing_ok=True)


def expected_filtered_names(suite: PromptSuite, filter_contract: Mapping[str, Any]) -> set[str]:
    """Filenames produced by the temporal filter, validated against the prompt suite."""
    selections = filter_contract.get("selection_by_prompt", {})
    if set(selections) != set(suite.temporal_prompts):
        raise RuntimeError("filter contract prompt set mismatch")
    expected = set()
    for prompt, selected in selections.items():
        if not isinstance(selected, list) or len(selected) > BASE_SAMPLES:
            raise RuntimeError(f"invalid filter selection count for {prompt}")
        for index, row in enumerate(selected):
            if row.get("output_filename") != f"{prompt}-{index}.mp4":
                raise RuntimeError(f"invalid filtered output sequence for {prompt}")
            expected.add(row["output_filename"])
    if not expected or len(expected) != filter_contract.get("output_video_count"):
        raise RuntimeError("filter contract output count mismatch")
    return expected


def corpus_fingerprint(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["filename"]):
        digest.update(f"{row['filename']}\0{row['sha256']}\0{row['size_bytes']}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_corpus(
    videos_dir: Path, expected: set[str], *, workers: int = 16
) -> dict[str, Any]:
    """Require exactly the expected MP4 filenames and check every file's media contract."""
    entries = sorted(videos_dir.iterdir(), key=lambda path: path.name)
    unexpected = [p.name for p in entries if not p.is_file() or p.suffix.lower() != ".mp4"]
    if unexpected:
        raise RuntimeError(f"corpus directory must contain only MP4 files: {unexpected[:8]}")
    observed = {p.name for p in entries}
    missing, extra = sorted(expected - observed), sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"corpus filenames differ: missing={len(missing)} {missing[:5]} "
                           f"extra={len(extra)} {extra[:5]}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(
            lambda name: {"filename": name, **verify_video(videos_dir / name)},
            sorted(expected),
        ))
    return {"count": len(rows), "total_size_bytes": sum(r["size_bytes"] for r in rows),
            "corpus_sha256": corpus_fingerprint(rows), "files": rows}


def _suite_from_args(args: argparse.Namespace) -> PromptSuite:
    return load_suite(args.protocol, args.full_info, args.original_prompts, args.extended_prompts)


def add_suite_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--protocol", choices=tuple(CONDITIONING), default=EXTENDED_PROTOCOL)
    parser.add_argument("--full-info", type=Path, default=DEFAULT_FULL_INFO)
    parser.add_argument("--original-prompts", type=Path, default=DEFAULT_ORIGINAL_PROMPTS)
    parser.add_argument("--extended-prompts", type=Path, help="default: the protocol's pinned prompt file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    summary = sub.add_parser("summary", help="print protocol counts and fingerprints")
    jobs = sub.add_parser("jobs", help="write one JSON line per generation job")
    jobs.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify", help="verify a raw or filtered corpus")
    verify.add_argument("--mode", choices=("raw", "filtered_temporal"), required=True)
    verify.add_argument("--videos-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--filter-contract", type=Path)
    verify.add_argument("--workers", type=int, default=16)
    for command in (summary, jobs, verify):
        add_suite_arguments(command)
    args = parser.parse_args()
    suite = _suite_from_args(args)

    if args.command == "summary":
        print(json.dumps(protocol_summary(suite), indent=2))
    elif args.command == "jobs":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            for job in build_sample_specs(suite):
                handle.write(json.dumps({
                    "filename": job.filename, "prompt": job.prompt,
                    "conditioning_prompt": job.conditioning_prompt, "seed": job.seed,
                    "prompt_ordinal": job.prompt_ordinal, "sample_index": job.sample_index,
                    "temporal_flickering": job.temporal_flickering,
                }, ensure_ascii=False) + "\n")
        print(f"wrote {EXPECTED_RAW_VIDEOS} jobs to {args.output}")
    else:
        if args.mode == "raw":
            expected = {job.filename for job in build_sample_specs(suite)}
        else:
            if args.filter_contract is None:
                parser.error("--mode filtered_temporal requires --filter-contract")
            expected = expected_filtered_names(suite, json.loads(args.filter_contract.read_text("utf-8")))
        result = verify_corpus(args.videos_dir.resolve(), expected, workers=args.workers)
        atomic_json(args.output, {"mode": args.mode,
                                  "vbench": protocol_summary(suite), **result})
        print(f"PASS: {args.mode} count={result['count']} corpus_sha256={result['corpus_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
