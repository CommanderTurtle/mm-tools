#!/usr/bin/env python3
"""Local video -> multimodal multi-turn annotations -> single-file SFT dataset.

This is the small, local-file counterpart of large_scale_captioning's
``k3_openai_chat_arrow_multiturn_load_aware.py``. Its five 0725 prompt fragments
are included; cluster storage, VJEPA, caches, and DP routing are omitted.
The ``segment`` task fills an Arrow ``segments`` column with model-predicted
event segments using the structure-analysis prompt.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

PROMPT_DIR = Path(__file__).with_name("prompts")
PROMPT_FILES = (
    "prompt_0725_system.md",
    "prompt_0725_icl2.md",
    "prompt_0725_first_turn.md",
    "prompt_0725_intermediate_turn.md",
    "prompt_0725_final_summary.md",
)
SEGMENTATION_PROMPT_FILE = "prompt_segmentation.md"
EVENT_KEYS = {"start_time", "end_time", "label", "description", "contained_actions"}
EVENT_ACTION_KEYS = {"actor", "action_label", "start_time", "end_time"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
PERSPECTIVES = {
    "first-person",
    "second-person",
    "third-person follow",
    "third-person observer",
}
STYLE_EN_TO_CN = {
    "real-world footage": "真实视频",
    "animation/anime": "动画/动漫",
    "CG": "CG",
    "video game": "游戏",
    "software/web/app/computer screen recording": "软件/网页/App/电脑录屏",
    "slideshow": "幻灯片",
}
ACTION_WRAPPER = re.compile(
    r"\s*<Action([1-9][0-9]*)>(.*?)</Action\1>\s*[.!?。！？]?\s*", re.DOTALL
)
SUBJECT_TOKEN = re.compile(r"<\|Subject([1-9][0-9]*)\|>")


def prompt_bundle() -> tuple[str, str, str, str, str]:
    parts = [(PROMPT_DIR / name).read_text(encoding="utf-8") for name in PROMPT_FILES]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return parts[0] + "\n\n" + parts[1], *parts[2:], digest


def timestamp(seconds: float) -> str:
    minutes = int(seconds // 60)
    return f"{minutes:02d}:{seconds - minutes * 60:06.3f}"


def segments_text(segments: list[dict[str, Any]]) -> str:
    lines = ["PRECOMPUTED_EVENT_SEGMENTS:"]
    for index, segment in enumerate(segments, 1):
        lines.append(
            f"{index}) {timestamp(segment['start_sec'])} – {timestamp(segment['end_sec'])}"
        )
    return "\n".join(lines)


def render_prompt(
    template: str,
    segments: list[dict[str, Any]],
    duration: float,
    fps: int,
    segment_index: int | None = None,
) -> str:
    replacements = {
        "{duration:.2f}": f"{duration:.2f}",
        "{num_frames}": str(round(duration * fps) + 1),
        "{fps}": str(fps),
        "{precomputed_event_segments}": segments_text(segments),
        "{segment_count}": str(len(segments)),
    }
    if segment_index is not None:
        segment = segments[segment_index - 1]
        replacements.update(
            {
                "{segment_index}": str(segment_index),
                "{segment_start_time}": timestamp(segment["start_sec"]),
                "{segment_end_time}": timestamp(segment["end_sec"]),
            }
        )
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def sample_video(path: Path, fps: int, size: int, max_seconds: float):
    """Decode once, sample on an exact FPS grid, and center-crop to VAE size."""
    import av
    from PIL import Image, ImageOps

    sampled = []
    next_time = 0.0
    first_time = None
    last_time = 0.0
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        fallback_fps = float(stream.average_rate or fps)
        for frame_index, frame in enumerate(container.decode(video=0)):
            frame_time = frame.time
            if frame_time is None:
                frame_time = frame_index / fallback_fps
            if first_time is None:
                first_time = float(frame_time)
            elapsed = max(0.0, float(frame_time) - first_time)
            if elapsed > max_seconds + 1 / fps:
                raise ValueError(
                    f"{path}: longer than --max-seconds={max_seconds}; clip the video first"
                )
            last_time = elapsed
            if elapsed + 1e-6 < next_time:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            image = ImageOps.fit(image, (size, size), method=Image.Resampling.LANCZOS)
            while next_time <= elapsed + 1e-6:
                sampled.append(image)
                next_time = len(sampled) / fps
    if not sampled:
        raise ValueError(f"{path}: no decodable video frames")
    # DCAE-V's temporal ratio is 4 and its first latent represents frame zero.
    frame_count = ((len(sampled) - 1) // 4) * 4 + 1
    sampled = sampled[:frame_count]
    if len(sampled) < 2 * fps + 1:
        raise ValueError(
            f"{path}: needs at least 2 seconds after sampling at {fps} FPS"
        )
    if last_time + 1 / fps < (len(sampled) - 1) / fps:
        raise ValueError(f"{path}: unstable video timestamps")
    return sampled


def make_segments(frame_count: int, fps: int, segment_seconds: float):
    latent_frames = (frame_count - 1) // 4 + 1
    target_latents = max(2, round(segment_seconds * fps / 4))
    boundaries = list(range(0, latent_frames, target_latents)) + [latent_frames]
    if len(boundaries) == 2:
        boundaries.insert(1, latent_frames // 2)
    elif len(boundaries) == 3 and boundaries[-1] - boundaries[-2] < max(
        2, target_latents // 4
    ):
        boundaries[1] = latent_frames // 2
    # Merge a tiny tail; the first and last event both need meaningful evidence.
    if len(boundaries) > 3 and boundaries[-1] - boundaries[-2] < max(
        2, target_latents // 4
    ):
        boundaries.pop(-2)
    duration = (frame_count - 1) / fps
    segments = [
        {
            "start_sec": min(start * 4 / fps, duration),
            "end_sec": min(end * 4 / fps, duration),
        }
        for start, end in zip(boundaries, boundaries[1:])
    ]
    if any(s["start_sec"] >= s["end_sec"] for s in segments):
        raise ValueError("video is too short for two nonempty segments")
    return boundaries, segments


def jpeg_part(image) -> dict[str, Any]:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/jpeg;base64,"
            + base64.b64encode(buffer.getvalue()).decode("ascii")
        },
    }


def image_content(prompt: str, images, start: float, fps: int, max_frames: int):
    count = min(len(images), max_frames)
    positions = (
        sorted({round(i * (len(images) - 1) / (count - 1)) for i in range(count)})
        if count > 1
        else [0]
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": f"The following {len(positions)} frames are ordered "
            "chronologically and belong only to the current event segment.",
        },
    ]
    for number, position in enumerate(positions, 1):
        relative = position / fps
        content.append(
            {
                "type": "text",
                "text": f"Frame {number}/{len(positions)} | "
                f"video_timestamp={timestamp(start + relative)} | "
                f"segment_relative_time={timestamp(relative)}",
            }
        )
        content.append(jpeg_part(images[position]))
    return content


def parse_json_reply(content: Any) -> dict[str, Any]:
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    if not isinstance(content, str):
        raise ValueError("assistant content is not text")
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"assistant did not return JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("assistant JSON must be an object")
    return parsed


def validate_draft(parsed: dict[str, Any], index: int, segment: dict[str, Any]):
    draft = parsed.get("segment_draft")
    if parsed.get("segment_index") != index or not isinstance(draft, dict):
        raise ValueError("segment_index or segment_draft is missing")
    if draft.get("start_time") != timestamp(segment["start_sec"]) or draft.get(
        "end_time"
    ) != timestamp(segment["end_sec"]):
        raise ValueError("draft timestamps differ from the fixed segment")
    if not isinstance(draft.get("contained_actions"), list):
        raise ValueError("draft contained_actions must be a list")
    if set(draft) != {
        "start_time",
        "end_time",
        "visual_description_long",
        "description_change",
        "contained_actions",
    }:
        raise ValueError("draft has missing or extra segment fields")
    if index == 1:
        if (
            not isinstance(draft["visual_description_long"], str)
            or not draft["visual_description_long"].strip()
            or draft["description_change"] is not None
        ):
            raise ValueError("first draft needs a long description and null change")
    elif (
        draft["visual_description_long"] is not None
        or not isinstance(draft["description_change"], str)
        or not draft["description_change"].strip()
    ):
        raise ValueError(
            "later draft needs null long description and a change description"
        )


def seconds_from_timestamp(value: str) -> float:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"\d{2,}:[0-5]\d\.\d{3}", value) is None
    ):
        raise ValueError(f"invalid timestamp: {value!r}")
    minutes, seconds = value.split(":")
    return int(minutes) * 60 + float(seconds)


def validate_final(parsed: dict[str, Any], segments: list[dict[str, Any]]):
    if set(parsed) != {"english_json", "chinese_json"}:
        raise ValueError("final response needs only english_json and chinese_json")
    subject_ids = None
    action_times = None
    for language in ("english_json", "chinese_json"):
        item = parsed[language]
        if not isinstance(item, dict) or set(item) != {
            "perspective_view",
            "Style",
            "subject_list",
            "segments",
        }:
            raise ValueError(f"{language}: missing or extra top-level fields")
        if item.get("perspective_view") not in PERSPECTIVES:
            raise ValueError(f"{language}: invalid perspective_view")
        if language == "english_json":
            if item.get("Style") not in STYLE_EN_TO_CN:
                raise ValueError("english_json: invalid Style")
        elif item.get("Style") != STYLE_EN_TO_CN[parsed["english_json"]["Style"]]:
            raise ValueError("chinese_json: Style does not match English Style")
        if (
            language == "chinese_json"
            and item["perspective_view"] != parsed["english_json"]["perspective_view"]
        ):
            raise ValueError("bilingual perspective_view values differ")
        subjects = item["subject_list"]
        if not isinstance(subjects, list):
            raise ValueError(f"{language}: subject_list must be a list")
        ids = []
        for number, subject in enumerate(subjects, 1):
            if (
                not isinstance(subject, dict)
                or set(subject) != {"subject_id", "display_name", "subject_attribute"}
                or subject["subject_id"] != f"<|Subject{number}|>"
                or not isinstance(subject["display_name"], str)
                or not subject["display_name"].strip()
            ):
                raise ValueError(f"{language}: subject {number} has an invalid schema")
            ids.append(subject["subject_id"])
        if subject_ids is not None and ids != subject_ids:
            raise ValueError("bilingual subject IDs differ")
        subject_ids = ids
        captions = item.get("segments")
        if not isinstance(captions, list) or len(captions) != len(segments):
            raise ValueError(f"{language}: segment count mismatch")
        language_action_times = []
        for index, (caption, segment) in enumerate(zip(captions, segments), 1):
            if (
                not isinstance(caption, dict)
                or caption.get("start_time") != timestamp(segment["start_sec"])
                or caption.get("end_time") != timestamp(segment["end_sec"])
            ):
                raise ValueError(
                    f"{language} segment {index}: fixed timestamps changed"
                )
            if set(caption) != {
                "start_time",
                "end_time",
                "visual_description_long",
                "description_change",
                "contained_actions",
            }:
                raise ValueError(f"{language} segment {index}: missing or extra fields")
            if index == 1:
                if (
                    not isinstance(caption["visual_description_long"], str)
                    or not caption["visual_description_long"].strip()
                    or caption["description_change"] is not None
                ):
                    raise ValueError(
                        f"{language} segment {index}: invalid first description"
                    )
            elif (
                caption["visual_description_long"] is not None
                or not isinstance(caption["description_change"], str)
                or not caption["description_change"].strip()
            ):
                raise ValueError(
                    f"{language} segment {index}: invalid change description"
                )
            if not isinstance(caption.get("contained_actions"), list):
                raise ValueError(
                    f"{language} segment {index}: contained_actions missing"
                )
            segment_action_times = []
            prior_end = segment["start_sec"]
            for action_number, action in enumerate(caption["contained_actions"], 1):
                if not isinstance(action, dict) or set(action) != {
                    "start_time",
                    "end_time",
                    "action_description",
                }:
                    raise ValueError(f"{language} segment {index}: malformed action")
                start = seconds_from_timestamp(action["start_time"])
                end = seconds_from_timestamp(action["end_time"])
                if (
                    start < prior_end - 0.001
                    or end <= start
                    or end > segment["end_sec"] + 0.001
                ):
                    raise ValueError(
                        f"{language} segment {index}: invalid action timeline"
                    )
                description = action["action_description"]
                match = (
                    ACTION_WRAPPER.fullmatch(description)
                    if isinstance(description, str)
                    else None
                )
                if match is None or int(match.group(1)) != action_number:
                    raise ValueError(
                        f"{language} segment {index}: invalid Action wrapper"
                    )
                mentioned = {
                    f"<|Subject{number}|>"
                    for number in SUBJECT_TOKEN.findall(match.group(2))
                }
                if not mentioned or not mentioned.issubset(ids):
                    raise ValueError(
                        f"{language} segment {index}: Action has no valid Subject token"
                    )
                segment_action_times.append((action["start_time"], action["end_time"]))
                prior_end = end
            language_action_times.append(segment_action_times)
        if action_times is not None and action_times != language_action_times:
            raise ValueError("bilingual action timestamps differ")
        action_times = language_action_times


def chat(
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    timeout: int,
    thinking: bool,
    validator,
    retries: int,
):
    import requests

    body = {
        "model": model,
        "messages": list(messages),
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "low" if thinking else "none",
        "chat_template_kwargs": {"enable_thinking": thinking, "thinking": thinking},
    }
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    for attempt in range(retries + 1):
        response = requests.post(
            api_base.rstrip("/") + "/chat/completions",
            json=body,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        answer = choice["message"].get("content")
        if not answer:
            raise ValueError(
                f"empty assistant response (finish_reason={choice.get('finish_reason')})"
            )
        try:
            parsed = parse_json_reply(answer)
            validator(parsed)
            return answer, parsed
        except ValueError as exc:
            if attempt == retries:
                raise ValueError(
                    f"Model response failed validation after {retries + 1} attempts: {exc}"
                ) from exc
            body["messages"].extend(
                [
                    {"role": "assistant", "content": answer},
                    {
                        "role": "user",
                        "content": f"Correct the JSON: {exc}. Output only the corrected JSON object.",
                    },
                ]
            )
    raise AssertionError("unreachable")


def prompts_from_caption(english: dict[str, Any]) -> list[str]:
    context = {
        key: english[key] for key in ("perspective_view", "Style", "subject_list")
    }
    return [
        json.dumps({**context, "segments": [segment]}, ensure_ascii=False)
        for segment in english["segments"]
    ]


def video_inputs(path: Path):
    if path.is_file():
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"unsupported video extension: {path}")
        return [path]
    if not path.is_dir():
        raise ValueError(f"video input does not exist: {path}")
    files = sorted(
        p
        for p in path.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not files:
        raise ValueError(f"no videos found under {path}")
    return files


def annotate(args):
    system, first, intermediate, final, digest = prompt_bundle()
    inputs = video_inputs(Path(args.input).resolve())
    output = Path(args.annotations).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["video_path"]] = record
    for path in inputs:
        stat = path.stat()
        old = completed.get(str(path))
        if old is not None:
            if (
                old["video_size"] != stat.st_size
                or old["video_mtime_ns"] != stat.st_mtime_ns
                or old["prompt_sha256"] != digest
                or old["fps"] != args.fps
                or old["size"] != args.size
                or old["segment_seconds"] != args.segment_seconds
                or old["model"] != args.model
                or old["max_frames_per_segment"] != args.max_frames_per_segment
                or old["thinking"] != args.thinking
            ):
                raise ValueError(
                    f"{path}: input or settings changed since existing annotation; use a new output file"
                )
            print(f"skip annotated video: {path}", flush=True)
            continue
        images = sample_video(path, args.fps, args.size, args.max_seconds)
        boundaries, segments = make_segments(
            len(images), args.fps, args.segment_seconds
        )
        if len(segments) > args.max_segments:
            raise ValueError(
                f"{path}: {len(segments)} segments exceeds --max-segments={args.max_segments}"
            )
        duration = (len(images) - 1) / args.fps
        messages = [{"role": "system", "content": system}]
        for index, segment in enumerate(segments, 1):
            start_frame = boundaries[index - 1] * 4
            end_frame = min(boundaries[index] * 4, len(images) - 1)
            clip = (
                images[start_frame:end_frame]
                if index < len(segments)
                else images[start_frame:]
            )
            prompt = render_prompt(
                first if index == 1 else intermediate,
                segments,
                duration,
                args.fps,
                index,
            )
            messages.append(
                {
                    "role": "user",
                    "content": image_content(
                        prompt,
                        clip,
                        segment["start_sec"],
                        args.fps,
                        args.max_frames_per_segment,
                    ),
                }
            )
            answer, _ = chat(
                args.api_base,
                args.api_key,
                args.model,
                messages,
                args.segment_max_tokens,
                args.timeout,
                args.thinking,
                lambda obj, i=index, s=segment: validate_draft(obj, i, s),
                args.retries,
            )
            messages.append({"role": "assistant", "content": answer})
            print(f"{path.name}: segment {index}/{len(segments)} annotated", flush=True)
        messages.append(
            {
                "role": "user",
                "content": render_prompt(final, segments, duration, args.fps),
            }
        )
        _, result = chat(
            args.api_base,
            args.api_key,
            args.model,
            messages,
            args.final_max_tokens,
            args.timeout,
            args.thinking,
            lambda obj: validate_final(obj, segments),
            args.retries,
        )
        record = {
            "sample_id": path.stem
            + "-"
            + hashlib.sha256(str(path).encode()).hexdigest()[:10],
            "video_path": str(path),
            "video_size": stat.st_size,
            "video_mtime_ns": stat.st_mtime_ns,
            "fps": args.fps,
            "size": args.size,
            "segment_seconds": args.segment_seconds,
            "model": args.model,
            "max_seconds": args.max_seconds,
            "max_frames_per_segment": args.max_frames_per_segment,
            "thinking": args.thinking,
            "frame_count": len(images),
            "segment_boundaries": boundaries,
            "prompt_sha256": digest,
            "annotation": result,
            "prompts": prompts_from_caption(result["english_json"]),
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        print(f"{path.name}: saved to {output}", flush=True)


def sample_clip_frames(
    path: Path, fps: int, size: int, start: float, duration: float | None
):
    """Sample a whole clip on an FPS grid, keeping aspect ratio (no crop)."""
    import av
    from PIL import Image, ImageOps

    frames, times = [], []
    next_time = 0.0
    first_time = None
    last_time = None
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError(f"{path}: no video stream")
        stream = container.streams.video[0]
        source_fps = float(stream.average_rate or 30)
        for frame_index, frame in enumerate(container.decode(video=0)):
            frame_time = frame.time
            if frame_time is None:
                frame_time = frame_index / source_fps
            if first_time is None:
                first_time = float(frame_time)
            elapsed = float(frame_time) - first_time - start
            if elapsed < -1e-6:
                continue
            if duration is not None and elapsed >= duration - 1e-6:
                break
            last_time = max(elapsed, 0.0)
            if elapsed + 1e-6 < next_time:
                continue
            image = Image.fromarray(frame.to_ndarray(format="rgb24"))
            frames.append(ImageOps.contain(image, (size, size)))
            times.append(last_time)
            next_time = len(frames) / fps
    if len(frames) < 2 or last_time is None:
        raise ValueError(f"{path}: needs at least two sampled frames")
    clip_duration = last_time + 1 / source_fps
    if duration is not None:
        clip_duration = min(duration, clip_duration)
    return frames, times, clip_duration


def validate_segmentation(
    parsed: dict[str, Any], duration: float, tolerance: float
):
    events = parsed.get("events")
    if set(parsed) != {"events"} or not isinstance(events, list) or not events:
        raise ValueError("response needs only a nonempty events list")
    previous_end = None
    for number, event in enumerate(events, 1):
        if not isinstance(event, dict) or set(event) != EVENT_KEYS:
            raise ValueError(f"event {number}: missing or extra fields")
        start = seconds_from_timestamp(event["start_time"])
        end = seconds_from_timestamp(event["end_time"])
        if number == 1 and start > 0.001:
            raise ValueError("event 1 must start at 00:00.000")
        if previous_end is not None and event["start_time"] != previous_end:
            raise ValueError(
                f"event {number}: start_time must equal the previous end_time"
            )
        if end <= start or end > duration + tolerance:
            raise ValueError(
                f"event {number}: invalid time range for a {timestamp(duration)} video"
            )
        for key in ("label", "description"):
            if not isinstance(event[key], str) or not event[key].strip():
                raise ValueError(f"event {number}: {key} must be a nonempty string")
        if not isinstance(event["contained_actions"], list):
            raise ValueError(f"event {number}: contained_actions must be a list")
        for action_number, action in enumerate(event["contained_actions"], 1):
            if not isinstance(action, dict) or set(action) != EVENT_ACTION_KEYS:
                raise ValueError(
                    f"event {number} action {action_number}: missing or extra fields"
                )
            action_start = seconds_from_timestamp(action["start_time"])
            action_end = seconds_from_timestamp(action["end_time"])
            if (
                action_start < start - 0.001
                or action_end > end + 0.001
                or action_end <= action_start
            ):
                raise ValueError(
                    f"event {number} action {action_number}: must lie inside its event"
                )
            for key in ("actor", "action_label"):
                if not isinstance(action[key], str) or not action[key].strip():
                    raise ValueError(
                        f"event {number} action {action_number}: {key} must be nonempty"
                    )
        previous_end = event["end_time"]
    if abs(seconds_from_timestamp(previous_end) - duration) > tolerance:
        raise ValueError(
            f"last event must end at the video end ({timestamp(duration)})"
        )


def segments_from_events(events: list[dict[str, Any]], duration: float):
    segments = []
    for index, event in enumerate(events):
        start = 0.0 if index == 0 else segments[-1]["end_sec"]
        end = (
            duration
            if index == len(events) - 1
            else seconds_from_timestamp(event["end_time"])
        )
        segments.append(
            {
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "label": event["label"].strip(),
                "description": event["description"].strip(),
                "contained_actions": [
                    {
                        "actor": action["actor"].strip(),
                        "action_label": action["action_label"].strip(),
                        "start_sec": round(
                            max(start, seconds_from_timestamp(action["start_time"])), 3
                        ),
                        "end_sec": round(
                            min(end, seconds_from_timestamp(action["end_time"])), 3
                        ),
                    }
                    for action in event["contained_actions"]
                ],
            }
        )
    return segments


def read_arrow_table(path: Path):
    import pyarrow as pa

    with pa.memory_map(str(path), "r") as source:
        try:
            return pa.ipc.open_file(source).read_all()
        except pa.ArrowInvalid:
            source.seek(0)
            return pa.ipc.open_stream(source).read_all()


def segment_row(index: int, row: dict[str, Any], template: str, args):
    import requests

    video_path = row.get("video_path")
    if not isinstance(video_path, str) or not video_path.strip():
        raise ValueError("missing video_path")
    start = float(row.get("video_start_sec") or 0)
    duration = row.get("video_duration_sec")
    duration = None if duration is None else float(duration)
    if start < 0 or (duration is not None and duration <= 0):
        raise ValueError("invalid video_start_sec or video_duration_sec")
    parsed_url = urlsplit(video_path)
    with tempfile.TemporaryDirectory(prefix="segment_") as workdir:
        if parsed_url.scheme in {"http", "https"}:
            local = Path(workdir) / "video"
            with requests.get(video_path, stream=True, timeout=(15, 90)) as response:
                response.raise_for_status()
                with local.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
        elif parsed_url.scheme in {"", "file"}:
            local = Path(
                unquote(parsed_url.path) if parsed_url.scheme == "file" else video_path
            ).expanduser()
            if not local.is_absolute():
                local = args.input.parent / local
            if not local.is_file():
                raise ValueError(f"local video does not exist: {local}")
        else:
            raise ValueError(f"unsupported video_path scheme: {parsed_url.scheme}")
        frames, times, clip_duration = sample_clip_frames(
            local, args.fps, args.size, start, duration
        )
    if len(frames) > args.max_frames:
        positions = sorted(
            {
                round(i * (len(frames) - 1) / (args.max_frames - 1))
                for i in range(args.max_frames)
            }
        )
        frames = [frames[p] for p in positions]
        times = [times[p] for p in positions]
    effective_fps = min(args.fps, (len(frames) - 1) / max(times[-1], 1e-6))
    prompt = (
        template.replace("{duration:.2f}", f"{clip_duration:.2f}")
        .replace("{num_frames}", str(len(frames)))
        .replace("{fps}", f"{effective_fps:.3g}")
    )
    content = [
        {"type": "text", "text": prompt},
        {
            "type": "text",
            "text": f"The following {len(frames)} frames are ordered "
            "chronologically and cover the whole video.",
        },
    ]
    for number, (image, frame_time) in enumerate(zip(frames, times), 1):
        content.append(
            {
                "type": "text",
                "text": f"Frame {number}/{len(frames)} | "
                f"video_timestamp={timestamp(frame_time)}",
            }
        )
        content.append(jpeg_part(image))
    tolerance = max(1 / effective_fps, 0.25)
    _, parsed = chat(
        args.api_base,
        args.api_key,
        args.model,
        [{"role": "user", "content": content}],
        args.max_tokens,
        args.timeout,
        args.thinking,
        lambda obj: validate_segmentation(obj, clip_duration, tolerance),
        args.retries,
    )
    segments = segments_from_events(parsed["events"], clip_duration)
    print(
        f"row {index}: {len(segments)} segments over {timestamp(clip_duration)}",
        flush=True,
    )
    return segments


def segment(args):
    import pyarrow as pa
    from concurrent.futures import ThreadPoolExecutor, as_completed

    template = (PROMPT_DIR / SEGMENTATION_PROMPT_FILE).read_text(encoding="utf-8")
    args.input = Path(args.input).resolve()
    output = Path(args.output).resolve()
    if output.suffix != ".arrow":
        raise ValueError("--output must end in .arrow")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    table = read_arrow_table(args.input)
    names = table.schema.names
    if "video_path" not in names:
        raise ValueError("input Arrow is missing video_path (URL or local path)")
    if "segments" in names:
        existing = table.column("segments")
        if existing.null_count != len(table):
            filled = next(
                i for i, value in enumerate(existing.is_null().to_pylist()) if not value
            )
            raise ValueError(
                f"segments must be null in every input row; row {filled} is filled"
            )
    columns = [
        name
        for name in ("video_path", "video_start_sec", "video_duration_sec")
        if name in names
    ]
    rows = table.select(columns).to_pylist()
    results: list[str | None] = [None] * len(rows)
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(segment_row, index, row, template, args): index
            for index, row in enumerate(rows)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                results[index] = json.dumps(future.result(), ensure_ascii=False)
            except Exception as exc:
                failures.append({"row_index": index, "error": str(exc)})
                print(f"row {index}: FAILED: {exc}", flush=True)
    column = pa.array(results, type=pa.string())
    if "segments" in names:
        table = table.set_column(names.index("segments"), "segments", column)
    else:
        table = table.append_column("segments", column)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with pa.ipc.new_file(str(temporary), table.schema) as writer:
            writer.write_table(table)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    print(
        f"segmented {len(rows) - len(failures)}/{len(rows)} rows: {output}",
        flush=True,
    )
    if failures:
        failure_path = output.with_name(output.name + ".failures.jsonl")
        failure_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in failures),
            encoding="utf-8",
        )
        raise SystemExit(
            f"{len(failures)} rows failed and keep null segments; see {failure_path}"
        )


def camera_payload(camera_dir: Path | None, sample_id: str, frame_count: int):
    if camera_dir is None:
        return None, None, False
    import numpy as np

    path = camera_dir / f"{sample_id}.npz"
    if not path.is_file():
        return None, None, False
    with np.load(path, allow_pickle=False) as arrays:
        result = []
        for name, shape in (
            ("camera_extrinsics", (frame_count, 4, 4)),
            ("camera_intrinsics", (frame_count, 3, 3)),
        ):
            values = np.asarray(arrays[name], dtype="<f4")
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(
                    f"{path}: {name} must have shape {shape} and finite values"
                )
            result.append(values.tobytes(order="C"))
    return *result, True


def pack(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch

    if sys.byteorder != "little":
        raise RuntimeError("BF16 Arrow payload requires a little-endian host")
    output = Path(args.output).resolve()
    if output.suffix not in {".arrow", ".parquet"}:
        raise ValueError("--output must end in .arrow or .parquet")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    records = [
        json.loads(line)
        for line in Path(args.annotations).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError("annotation file has no completed videos")
    from hymm.models.vae.autoencoder_dcae_v import AutoencoderDCAEV

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    vae = AutoencoderDCAEV.from_checkpoint(
        args.vae_checkpoint, dtype=dtype, device=device
    ).eval()
    camera_dir = Path(args.camera_dir).resolve() if args.camera_dir else None
    schema = pa.schema(
        [
            ("sample_id", pa.string()),
            ("latent", pa.binary()),
            ("latent_shape", pa.list_(pa.int32())),
            ("prompts", pa.list_(pa.string())),
            ("segment_boundaries", pa.list_(pa.int32())),
            ("camera_extrinsics", pa.binary()),
            ("camera_intrinsics", pa.binary()),
            ("has_camera", pa.bool_()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        if output.suffix == ".arrow":
            writer = pa.ipc.new_file(str(temporary), schema)
        else:
            writer = pq.ParquetWriter(str(temporary), schema, compression="zstd")
        for record in records:
            path = Path(record["video_path"])
            stat = path.stat()
            if (
                stat.st_size != record["video_size"]
                or stat.st_mtime_ns != record["video_mtime_ns"]
            ):
                raise ValueError(f"{path}: video changed after annotation")
            images = sample_video(
                path,
                record["fps"],
                record["size"],
                max(args.max_seconds, record.get("max_seconds", 0)),
            )
            if len(images) != record["frame_count"]:
                raise ValueError(
                    f"{path}: sampled frame count changed after annotation"
                )
            import numpy as np

            video = np.stack([np.asarray(image) for image in images])
            tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)
            tensor = tensor.to(device=device, dtype=dtype).div_(127.5).sub_(1)
            with torch.no_grad():
                latent = (
                    vae.encode(tensor)
                    .latent_dist.mode()[0]
                    .to(device="cpu", dtype=torch.bfloat16)
                    .contiguous()
                )
            del tensor, video
            shape = list(latent.shape)
            if (
                shape[0] != 32
                or shape[1] != record["segment_boundaries"][-1]
                or shape[2] != record["size"] // 32
                or shape[3] != record["size"] // 32
            ):
                raise ValueError(f"{path}: unexpected DCAE latent shape {shape}")
            prompts = record["prompts"]
            if len(prompts) != len(record["segment_boundaries"]) - 1:
                raise ValueError(f"{path}: prompt/segment count mismatch")
            camera_ext, camera_int, has_camera = camera_payload(
                camera_dir, record["sample_id"], len(images)
            )
            row = {
                "sample_id": record["sample_id"],
                "latent": latent.view(torch.uint16)
                .numpy()
                .astype("<u2", copy=False)
                .tobytes(),
                "latent_shape": shape,
                "prompts": prompts,
                "segment_boundaries": record["segment_boundaries"],
                "camera_extrinsics": camera_ext,
                "camera_intrinsics": camera_int,
                "has_camera": has_camera,
            }
            writer.write_table(pa.Table.from_pylist([row], schema=schema))
            print(
                f"packed {path.name}: latent {shape}, camera={has_camera}", flush=True
            )
        writer.close()
        writer = None
        if output.exists():
            raise FileExistsError(f"output already exists: {output}")
        os.replace(temporary, output)
        temporary = None
    finally:
        if writer is not None:
            writer.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(f"SFT data ready: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    annotation = subcommands.add_parser(
        "annotate", help="Annotate local videos with an OpenAI-compatible multimodal API"
    )
    annotation.add_argument(
        "--input", required=True, help="One video or a directory of short videos"
    )
    annotation.add_argument(
        "--annotations", required=True, help="Append validated JSONL annotations here"
    )
    annotation.add_argument(
        "--api-base", default=os.getenv("CAPTION_API_BASE", "http://127.0.0.1:30000/v1")
    )
    annotation.add_argument(
        "--api-key", default=os.getenv("CAPTION_API_KEY", os.getenv("PE_API_KEY", ""))
    )
    annotation.add_argument("--model", default=os.getenv("CAPTION_MODEL", "kimi-k3"))
    annotation.add_argument("--fps", type=int, default=16)
    annotation.add_argument("--size", type=int, default=512)
    annotation.add_argument("--segment-seconds", type=float, default=4.0)
    annotation.add_argument("--max-seconds", type=float, default=32.0)
    annotation.add_argument("--max-segments", type=int, default=8)
    annotation.add_argument("--max-frames-per-segment", type=int, default=24)
    annotation.add_argument("--segment-max-tokens", type=int, default=8192)
    annotation.add_argument("--final-max-tokens", type=int, default=16384)
    annotation.add_argument("--timeout", type=int, default=1200)
    annotation.add_argument("--retries", type=int, default=2)
    annotation.add_argument("--no-thinking", action="store_false", dest="thinking")
    packing = subcommands.add_parser(
        "pack", help="Encode annotated videos to one SFT Arrow/Parquet file"
    )
    packing.add_argument("--annotations", required=True)
    packing.add_argument("--vae-checkpoint", required=True)
    packing.add_argument("--output", required=True)
    packing.add_argument("--device", default="cuda")
    packing.add_argument(
        "--camera-dir", help="Optional directory of normalized camera .npz files"
    )
    packing.add_argument("--max-seconds", type=float, default=32.0)
    segmentation = subcommands.add_parser(
        "segment",
        help="Fill the null segments column of an Arrow file via event segmentation",
    )
    segmentation.add_argument(
        "--input",
        required=True,
        help="Arrow with video_path and null (or absent) segments; "
        "optional video_start_sec / video_duration_sec",
    )
    segmentation.add_argument(
        "--output", required=True, help="New .arrow with segments filled as JSON"
    )
    segmentation.add_argument(
        "--api-base", default=os.getenv("CAPTION_API_BASE", "http://127.0.0.1:30000/v1")
    )
    segmentation.add_argument(
        "--api-key", default=os.getenv("CAPTION_API_KEY", os.getenv("PE_API_KEY", ""))
    )
    segmentation.add_argument("--model", default=os.getenv("CAPTION_MODEL", "kimi-k3"))
    segmentation.add_argument("--fps", type=int, default=2)
    segmentation.add_argument("--size", type=int, default=512)
    segmentation.add_argument("--max-frames", type=int, default=256)
    segmentation.add_argument("--max-tokens", type=int, default=11520)
    segmentation.add_argument("--timeout", type=int, default=1200)
    segmentation.add_argument("--retries", type=int, default=2)
    segmentation.add_argument("--workers", type=int, default=4)
    segmentation.add_argument("--no-thinking", action="store_false", dest="thinking")
    args = parser.parse_args()
    if args.command == "segment":
        if (
            min(args.fps, args.size, args.workers, args.max_tokens) <= 0
            or args.max_frames < 2
            or args.retries < 0
        ):
            parser.error(
                "fps, size, workers and max tokens must be positive; max frames >= 2"
            )
        segment(args)
    elif args.command == "annotate":
        if (
            args.fps <= 0
            or args.size <= 0
            or args.size % 32
            or args.segment_seconds <= 0
            or args.max_seconds <= 0
            or args.max_segments < 2
            or args.max_frames_per_segment < 1
            or args.retries < 0
        ):
            parser.error(
                "fps, segment seconds, max seconds, and size must be positive; size must be a multiple of 32"
            )
        annotate(args)
    else:
        pack(args)


if __name__ == "__main__":
    main()
