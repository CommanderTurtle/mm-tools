"""Dependency-light, recorded-parameter replay for the demo backend.

This reuses archived *model-ready* params, including per-turn shaped prompts,
inline camera JSON, frame counts, and session/reset flags. It never concatenates
camera tracks, reruns prompt enhancement, or changes normalization settings.
Historical checkpoint/config/runtime identity is NOT contained in these exports;
matching those remains a prerequisite for reproducing historical pixels.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path


PARAM_KEYS = (
    "prompt",
    "prompt_file",
    "subject",
    "prompts",
    "negative_prompt",
    "height",
    "width",
    "num_frames",
    "timing_source",
    "segment_durations_seconds",
    "video_fps",
    "seed",
    "guidance_scale",
    "num_inference_steps",
    "flow_shift",
    "solver_type",
    "segment_boundaries",
    "segment_boundaries_from_request",
    "camera_trajectory",
    "segment_has_camera",
    "user_prompts",
    "stable_prefix_carry_frames",
    "stable_prefix_carry_total_frames",
    "carry_camera_trajectory",
    "carry_user_prompt",
    "carry_has_camera",
    "cfg_schedule",
    "cfg_skip_steps",
    "use_dynamic_cfg_scale",
    "cfg_scale_min",
    "cfg_scale_max",
    "cfg_scale_max_step",
    "camera_guidance_scale",
    "session_id",
    "reset_session",
    "preserve_kv_cache",
    "continue_kv_cache",
)


OPTIONAL_PARAM_KEYS = ("trajectory_translation_scale", "trajectory_rotation_scale")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_params(params, session_id, first):
    missing = set(PARAM_KEYS) - params.keys()
    if missing:
        raise ValueError(
            f"{session_id}: missing archived parameters: {sorted(missing)}"
        )
    if params["solver_type"] != "euler":
        raise ValueError(f"{session_id}: only Euler inference can be replayed")
    if params["session_id"] != session_id or params["preserve_kv_cache"] is not True:
        raise ValueError(
            f"{session_id}: archived params are not a persistent demo session"
        )
    if first and (not params["reset_session"] or params["continue_kv_cache"]):
        raise ValueError(
            f"{session_id}: export starts mid-session; cannot replay strictly"
        )
    if not first and (params["reset_session"] or not params["continue_kv_cache"]):
        raise ValueError(
            f"{session_id}: reset/retry history needs explicit reconstruction"
        )
    bounds = params["segment_boundaries"]
    if (
        not isinstance(bounds, list)
        or len(bounds) != len(params["prompts"]) + 1
        or bounds[0] != 0
        or bounds[-1] != params["num_frames"]
        or any(a >= b for a, b in zip(bounds[:-1], bounds[1:]))
    ):
        raise ValueError(f"{session_id}: invalid archived per-turn frame boundaries")
    camera = params["camera_trajectory"]
    if camera:
        value = json.loads(camera)
        if not isinstance(value, dict) or not value.get("frames"):
            raise ValueError(
                f"{session_id}: replay requires original inline camera JSON"
            )


def build_manifest(source):
    """Read all archives up front and fail before GPU loading on missing history."""
    source = Path(source).resolve()
    sessions = read_json(source)
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Replay input must be a nonempty demo-export JSON array")
    output, ids = [], set()
    for session in sessions:
        session_id = session["session_id"]
        if session_id in ids:
            raise ValueError(f"Duplicate session: {session_id}")
        ids.add(session_id)
        index = sorted(session["turn_index"], key=lambda t: t["turn_number"])
        if [t["turn_number"] for t in index] != list(range(1, len(index) + 1)):
            raise ValueError(f"{session_id}: non-contiguous turn history")
        if len(index) != session["turn_count"]:
            raise ValueError(f"{session_id}: incomplete turn history")
        turns = []
        previous_end = 0
        for item in index:
            attempt_path = Path(item["source_attempt_path"])
            attempt = read_json(attempt_path)
            if attempt["status"] != "succeeded":
                raise ValueError(f"{attempt_path}: attempt did not succeed")
            metadata_path = Path(attempt["artifacts"]["metadata_path"])
            metadata = read_json(metadata_path)
            if metadata["request_id"] != item["result_id"]:
                raise ValueError(f"{metadata_path}: mismatched generation ID")
            params = {
                key: deepcopy(metadata[key])
                for key in PARAM_KEYS + OPTIONAL_PARAM_KEYS if key in metadata
            }
            validate_params(params, session_id, first=not turns)
            # Reject replacement/rollback histories rather than appending a
            # retried turn to the wrong KV state. Camera export has no retries.
            next_end = round(float(item["duration_seconds"]) * params["video_fps"])
            if next_end - previous_end != params["num_frames"]:
                raise ValueError(
                    f"{session_id} turn {item['turn_number']}: non-append history "
                    "(rollback/retry or missing turn); strict replay needs its event log"
                )
            previous_end = next_end
            turns.append(
                {
                    "turn_number": item["turn_number"],
                    "original_request_id": metadata["request_id"],
                    "metadata_path": str(metadata_path),
                    "metadata_sha256": hashlib.sha256(
                        metadata_path.read_bytes()
                    ).hexdigest(),
                    "params": params,
                }
            )
        if not turns:
            raise ValueError(f"{session_id}: empty session")
        output.append({"session_id": session_id, "turns": turns})
    return {
        "schema": "seg_ar_recorded_params_replay_v1",
        "source": str(source),
        "historical_runtime_verified": False,
        "limitations": [
            "Export does not identify historical DiT/PixelUMM/text-encoder weights, full config, code or SP topology.",
            "Recorded-parameter replay is not proof of historical bit-exact pixel reproduction.",
        ],
        "sessions": output,
    }


def replay_sessions(service, manifest, group_index=0, group_count=1):
    """Rank-zero only; normal demo worker loop executes broadcast parameters."""
    if not 0 <= group_index < group_count:
        raise ValueError("Invalid replay group index/count")
    inference = service.config.inference
    if (
        not inference.seg_ar_segment_vae_decode
        or not service.config.vae.stable_prefix_streaming
    ):
        raise ValueError(
            "Demo replay requires segment decoding and PixelUMM stable-prefix streaming"
        )
    if getattr(getattr(service.config, "vae", None), "name", None) != "pixelumm":
        raise ValueError("Demo replay requires the PixelUMM decoder")
    sessions = manifest["sessions"][group_index::group_count]
    for session in sessions:
        for turn in session["turns"]:
            if turn["params"]["video_fps"] != int(inference.video_fps):
                raise ValueError("Configured FPS differs from archived per-turn FPS")
    # Exclusive creation prevents accidental resume with empty KV history.
    journal_path = service.output_dir / f"replay_group_{group_index:03d}.jsonl"
    with journal_path.open("x", encoding="utf-8") as journal:
        journal.write(
            json.dumps(
                {
                    "event": "runtime",
                    "runtime": service.public_config(),
                    "historical_runtime_verified": False,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        journal.flush()
        for session in sessions:
            for turn in session["turns"]:
                params = deepcopy(turn["params"])
                # Namespace output/stream names to avoid touching historical data.
                suffix = hashlib.sha256(session["session_id"].encode()).hexdigest()[:16]
                params["session_id"] = f"replay-{suffix}"
                params["request_id"] = f"replay-{suffix}-turn{turn['turn_number']:03d}"
                params["command"] = "generate"
                result = service._collective_generate(params)
                journal.write(
                    json.dumps(
                        {
                            "event": "turn_succeeded",
                            "original_session_id": session["session_id"],
                            "turn_number": turn["turn_number"],
                            "original_request_id": turn["original_request_id"],
                            "result": result,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                journal.flush()
            service.release_session(params["session_id"])
