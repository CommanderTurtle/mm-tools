"""Single-file training data for segment self-forcing SFT.

Arrow IPC and Parquet use the same row schema.  Each row contains the complete
unscaled VAE latent, so training never needs a dataset directory or sidecars.
"""

import json
import math
import random
from bisect import bisect_right
from itertools import islice
from math import prod
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from hymm.trainer.transport.segment_sft_helpers import (
    _compute_action_offsets_for_prompt,
)


REQUIRED_COLUMNS = frozenset(
    {
        "latent",
        "latent_shape",
        "prompts",
        "segment_boundaries",
        "camera_extrinsics",
        "camera_intrinsics",
    }
)
OPTIONAL_COLUMNS = frozenset({"sample_id", "has_camera", "video_path"})


class SingleFileSFTDataset(Dataset):
    """Read one Arrow IPC or Parquet file with a shared, strict row schema."""

    def __init__(self, path):
        if path is None:
            raise ValueError(
                "dataset.data_file must be set to one .arrow or .parquet file"
            )
        self.path = Path(path)
        if self.path.suffix not in (".arrow", ".parquet") or not self.path.is_file():
            raise ValueError(
                "dataset.data_file must be one existing .arrow or .parquet file"
            )
        self._source = None
        self._reader = None
        self._cached_group = None
        self._cached_table = None
        self._open()
        schema = (
            self._reader.schema
            if self.path.suffix == ".arrow"
            else self._reader.schema_arrow
        )
        missing = REQUIRED_COLUMNS - set(schema.names)
        if missing:
            raise ValueError(f"SFT data file is missing columns: {sorted(missing)}")
        unexpected = set(schema.names) - REQUIRED_COLUMNS - OPTIONAL_COLUMNS
        if unexpected:
            raise ValueError(
                f"SFT data file has unsupported columns: {sorted(unexpected)}"
            )
        if self.path.suffix == ".arrow":
            counts = [
                self._reader.get_batch(i).num_rows
                for i in range(self._reader.num_record_batches)
            ]
        else:
            counts = [
                self._reader.metadata.row_group(i).num_rows
                for i in range(self._reader.num_row_groups)
            ]
        self._ends = np.cumsum(counts).tolist()
        if not self._ends or self._ends[-1] == 0:
            raise ValueError("SFT data file has no rows")

    def _open(self):
        if self.path.suffix == ".arrow":
            self._source = pa.memory_map(str(self.path), "r")
            self._reader = pa.ipc.open_file(self._source)
        else:
            self._reader = pq.ParquetFile(str(self.path))

    def __getstate__(self):
        state = self.__dict__.copy()
        state.update(_source=None, _reader=None, _cached_group=None, _cached_table=None)
        return state

    def __len__(self):
        return self._ends[-1]

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._reader is None:
            self._open()
        group = bisect_right(self._ends, index)
        if group != self._cached_group:
            if self.path.suffix == ".arrow":
                self._cached_table = pa.Table.from_batches(
                    [self._reader.get_batch(group)]
                )
            else:
                self._cached_table = self._reader.read_row_group(group)
            self._cached_group = group
        local_index = index - (self._ends[group - 1] if group else 0)
        row = self._cached_table.slice(local_index, 1).to_pylist()[0]
        return self._decode_row(row, index)

    @staticmethod
    def _decode_row(row, index):
        shape = row["latent_shape"]
        prompts = row["prompts"]
        boundaries = row["segment_boundaries"]
        payload = row["latent"]
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(not isinstance(value, int) or value <= 0 for value in shape)
        ):
            raise ValueError(
                f"row {index}: latent_shape must be [C,T,H,W] with positive integers"
            )
        if (
            not isinstance(prompts, list)
            or len(prompts) < 1
            or any(
                not isinstance(prompt, str) or not prompt.strip() for prompt in prompts
            )
        ):
            raise ValueError(
                f"row {index}: prompts must contain at least one nonempty segment"
            )
        if (
            not isinstance(boundaries, list)
            or len(boundaries) != len(prompts) + 1
            or (boundaries[0] != 0 or boundaries[-1] != shape[1])
            or any(a >= b for a, b in zip(boundaries, boundaries[1:]))
        ):
            raise ValueError(f"row {index}: segment_boundaries must partition latent T")
        expected_bytes = prod(shape) * 2
        if not isinstance(payload, bytes) or len(payload) != expected_bytes:
            raise ValueError(
                f"row {index}: latent must contain {expected_bytes} raw BF16 bytes"
            )
        # A copy gives PyTorch writable storage and keeps it independent of the
        # Arrow record batch cache.  The file stores little-endian BF16 bits.
        bits = np.frombuffer(payload, dtype="<u2").copy()
        latent = torch.from_numpy(bits).view(torch.bfloat16).reshape(shape)
        has_camera = row.get("has_camera", True)
        if not isinstance(has_camera, bool):
            raise ValueError(f"row {index}: has_camera must be a boolean")
        camera_frames = (shape[1] - 1) * 4 + 1
        camera_tensors = {}
        for column, dimensions in (
            ("camera_extrinsics", (camera_frames, 4, 4)),
            ("camera_intrinsics", (camera_frames, 3, 3)),
        ):
            camera_payload = row[column]
            camera_bytes = prod(dimensions) * 4
            if camera_payload is None and not has_camera:
                camera_tensors[column] = torch.zeros(dimensions, dtype=torch.float32)
            elif (
                not isinstance(camera_payload, bytes)
                or len(camera_payload) != camera_bytes
            ):
                raise ValueError(
                    f"row {index}: {column} must contain {camera_bytes} raw FP32 bytes"
                )
            else:
                values = np.frombuffer(camera_payload, dtype="<f4").copy()
                if not np.isfinite(values).all():
                    raise ValueError(f"row {index}: {column} contains nonfinite values")
                camera_tensors[column] = torch.from_numpy(values).reshape(dimensions)
        return {
            "latent": latent,
            "prompts": prompts,
            "segment_boundaries": boundaries,
            "row_index": index,
            "sample_id": row.get("sample_id") or str(index),
            "has_camera": has_camera,
            **camera_tensors,
        }


def _seconds(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        parts = value.replace(",", ".").split(":")
        try:
            if len(parts) == 2:
                return 60 * float(parts[0]) + float(parts[1])
            if len(parts) == 3:
                return 3600 * float(parts[0]) + 60 * float(parts[1]) + float(parts[2])
            return float(value)
        except ValueError:
            pass
    raise ValueError(f"invalid action timestamp: {value!r}")


def _stage7_prompt(
    prompt,
    segment_index,
    rng,
    *,
    desc_mid_prob,
    no_desc_long_after_first,
    normalize_action_timestamp,
    global_only_in_first_segment,
    keep_keys,
):
    """Shape one raw or already-shaped JSON prompt like the Stage7 loader."""
    obj = json.loads(prompt)
    segments = obj.get("segments") if isinstance(obj, dict) else None
    if (
        not isinstance(segments, list)
        or len(segments) != 1
        or not isinstance(segments[0], dict)
    ):
        raise ValueError("SFT prompts must be JSON objects with exactly one segment")
    seg = dict(segments[0])
    absolute_start = seg.get("start_time")
    actions = seg.get("contained_actions")
    if actions is not None and not isinstance(actions, list):
        raise ValueError("contained_actions must be a list")

    if (
        "visual_description_long" in seg
        and "visual_description_mid" in seg
        and rng.random() < desc_mid_prob
    ):
        seg["visual_description_long"] = seg["visual_description_mid"]
    if no_desc_long_after_first:
        if segment_index > 0:
            seg.pop("visual_description_long", None)
        # The public Arrow schema has one prompt per event. Each later event
        # is therefore its own first segment, as in the Stage7 run.
        if segment_index == 0:
            seg.pop("description_change", None)
    if normalize_action_timestamp and absolute_start is not None and actions:
        origin = _seconds(absolute_start)
        normalized = []
        for action in actions:
            item = dict(action)
            for key in ("start_time", "end_time"):
                if key in item:
                    relative = max(0.0, _seconds(item[key]) - origin)
                    item[key] = f"{int(relative // 60):02d}:{relative % 60:06.3f}"
            normalized.append(item)
        seg["contained_actions"] = normalized
    # An already-shaped Stage7 prompt has no event start_time and its action
    # timestamps are already local. Never subtract the video boundary twice.
    if "contained_actions" in seg:
        seg["contained_actions"] = seg.pop("contained_actions")
    obj = (
        {"segments": [seg]}
        if global_only_in_first_segment and segment_index > 0
        else {
            **{key: value for key, value in obj.items() if key != "segments"},
            "segments": [seg],
        }
    )

    if keep_keys:
        keys = set(keep_keys)

        def filter_value(value, path=""):
            if isinstance(value, list):
                return [filter_value(item, path) for item in value]
            if not isinstance(value, dict):
                return value
            out = {}
            for key, item in value.items():
                child = f"{path}.{key}" if path else key
                if child not in keys:
                    continue
                out[key] = (
                    filter_value(item, child)
                    if any(name.startswith(child + ".") for name in keys)
                    else item
                )
            return out

        obj = filter_value(obj)
    return json.dumps(obj, ensure_ascii=False), actions, absolute_start


def _action_windows(actions, absolute_start, frame_start, frame_end, fps):
    """Tile one latent segment with action spans and gaps for the Magi mask."""
    if not actions:
        return None
    origin = _seconds(absolute_start) if absolute_start is not None else 0.0
    spans = []
    for index, action in enumerate(actions):
        if not isinstance(action, dict) or not all(
            k in action for k in ("start_time", "end_time")
        ):
            return None
        start = frame_start + int(
            max(0.0, _seconds(action["start_time"]) - origin) * fps / 4
        )
        end = frame_start + int(
            max(0.0, _seconds(action["end_time"]) - origin) * fps / 4
        )
        start, end = (
            max(frame_start, min(start, frame_end)),
            max(frame_start, min(end, frame_end)),
        )
        if end > start:
            spans.append((start, end, index))
    if not spans:
        return None
    spans.sort()
    if any(spans[i][0] < spans[i - 1][1] for i in range(1, len(spans))):
        raise ValueError(
            "overlapping action windows are unsupported by the intra-segment mask"
        )
    windows = []
    cursor = frame_start
    for start, end, index in spans:
        if start > cursor:
            windows.append([cursor, start, -1])
        windows.append([start, end, index])
        cursor = end
    if cursor < frame_end:
        windows.append([cursor, frame_end, -1])
    if len(windows) == 1 and len(actions) == 1:
        return None
    return windows


class SegmentSFTCollator:
    """Build the transport's packed tensor contract without a packing plan."""

    def __init__(self, text_encoder, text_len, dataset_args):
        self.text_encoder = text_encoder
        self.text_len = int(text_len)
        self.uncond_p = float(getattr(dataset_args, "uncond_p", 0.0))
        self.desc_mid_prob = float(
            getattr(dataset_args, "json_prompt_desc_mid_prob", 0.0)
        )
        self.no_desc_long_after_first = bool(
            getattr(dataset_args, "json_prompt_no_desc_long_after_first", False)
        )
        self.normalize_action_timestamp = bool(
            getattr(dataset_args, "json_prompt_normalize_action_timestamp", False)
        )
        self.keep_keys = list(getattr(dataset_args, "json_prompt_keep_keys", []) or [])
        self.global_only_in_first_segment = bool(
            getattr(dataset_args, "global_only_in_first_segment", False)
        )
        self.fps = float(getattr(dataset_args, "fps", 16))
        if not (
            0 <= self.uncond_p <= 1 and 0 <= self.desc_mid_prob <= 1 and self.fps > 0
        ):
            raise ValueError(
                "uncond_p and desc_mid_prob must be in [0,1], fps must be positive"
            )

    def __call__(self, samples):
        channels = {sample["latent"].shape[0] for sample in samples}
        if len(channels) != 1:
            raise ValueError(
                "all samples in a batch must have the same latent channel count"
            )
        max_segments = max(len(sample["prompts"]) for sample in samples)
        encoded_segments = []
        shaped_prompts = []
        action_metadata = []
        max_encoded_length = 0
        for sample in samples:
            encoded_sample = []
            prompts = []
            actions_per_segment = []
            # The historical loader keyed prompt randomization and whole-sample
            # CFG dropout to the row index, independent of worker/resume order.
            row_index = int(sample["row_index"])
            prompt_rng = random.Random(row_index * 1_000_003 + 100)
            drop_text = (
                random.Random(row_index * 1_000_003 + 2).random() < self.uncond_p
            )
            for segment_index, prompt in enumerate(sample["prompts"]):
                prompt, actions, absolute_start = _stage7_prompt(
                    prompt,
                    segment_index,
                    prompt_rng,
                    desc_mid_prob=self.desc_mid_prob,
                    no_desc_long_after_first=self.no_desc_long_after_first,
                    normalize_action_timestamp=self.normalize_action_timestamp,
                    global_only_in_first_segment=self.global_only_in_first_segment,
                    keep_keys=self.keep_keys,
                )
                if drop_text:
                    prompt = ""
                    actions = None
                prompts.append(prompt)
                actions_per_segment.append((actions, absolute_start))
                encoded = self.text_encoder.text2tokens(
                    prompt,
                    data_type="video",
                    max_length=self.text_len,
                )
                token_ids = encoded["input_ids"].reshape(-1)
                token_mask = encoded["attention_mask"].reshape(-1)
                if token_ids.numel() != token_mask.numel():
                    raise ValueError("text encoder returned mismatched ids and mask")
                max_encoded_length = max(max_encoded_length, token_ids.numel())
                encoded_sample.append((token_ids, token_mask))
            encoded_segments.append(encoded_sample)
            shaped_prompts.append(prompts)
            action_metadata.append(actions_per_segment)
        ids = torch.zeros(
            (len(samples), max_segments, max_encoded_length), dtype=torch.long
        )
        masks = torch.zeros_like(ids)
        flat_latents = []
        shapes = []
        spans = []
        kwargs_list = []
        camera_extrinsics = []
        camera_intrinsics = []
        camera_actions = []
        has_camera = []
        offset = 0
        for sample_index, sample in enumerate(samples):
            latent = sample["latent"]
            _, frames, height, width = latent.shape
            tokens = frames * height * width
            flat_latents.append(latent.reshape(latent.shape[0], tokens))
            shapes.append((frames, height, width))
            spans.append((offset, offset + tokens))
            offset += tokens
            for segment_index, (token_ids, token_mask) in enumerate(
                encoded_segments[sample_index]
            ):
                ids[sample_index, segment_index, : token_ids.numel()] = token_ids
                masks[sample_index, segment_index, : token_mask.numel()] = token_mask
            offsets = [
                _compute_action_offsets_for_prompt(
                    self.text_encoder, prompt, self.text_len
                )
                if prompt
                else None
                for prompt in shaped_prompts[sample_index]
            ]
            windows = [
                _action_windows(
                    actions,
                    absolute_start,
                    sample["segment_boundaries"][j],
                    sample["segment_boundaries"][j + 1],
                    self.fps,
                )
                if offsets[j] is not None
                else None
                for j, (actions, absolute_start) in enumerate(
                    action_metadata[sample_index]
                )
            ]
            kwargs_list.append(
                {
                    "text": "|||".join(shaped_prompts[sample_index]),
                    "text_encoder_input": shaped_prompts[sample_index],
                    "num_text_segments": len(shaped_prompts[sample_index]),
                    "prompt_video_boundaries": list(sample["segment_boundaries"]),
                    "text_action_start_index": [o[0] if o else -1 for o in offsets],
                    "text_action_offsets": offsets,
                    "action_video_boundaries": windows,
                    "videoid": sample["sample_id"],
                }
            )
            camera_extrinsics.append(sample["camera_extrinsics"])
            camera_intrinsics.append(sample["camera_intrinsics"])
            camera_actions.append(
                torch.zeros(
                    (sample["camera_extrinsics"].shape[0], 8),
                    dtype=torch.float32,
                )
            )
            has_camera.append(sample["has_camera"])
        return {
            "is_token_packed": True,
            "num_packed_samples": len(samples),
            "sample_boundaries": spans,
            "sample_shapes": shapes,
            "packed_latent": torch.cat(flat_latents, dim=1).unsqueeze(0),
            "packed_text_ids": ids,
            "packed_text_mask": masks,
            "kwargs_list": kwargs_list,
            "total_packed_tokens": offset,
            "packed_norm_ext": camera_extrinsics,
            "packed_norm_int": camera_intrinsics,
            "packed_action_oh": camera_actions,
            "packed_has_camera": has_camera,
        }


class _OffsetDistributedSampler(DistributedSampler):
    """Skip already-consumed per-rank samples when resuming an epoch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def __iter__(self):
        return islice(super().__iter__(), self.start_index, None)

    def __len__(self):
        return max(0, super().__len__() - self.start_index)


class _PackingBatchSampler:
    """Yield precomputed sample-index groups, sharded across DP ranks."""

    offset_unit = "groups"

    def __init__(
        self,
        groups,
        *,
        dataset_size,
        dp_rank=0,
        dp_degree=1,
        shuffle_chunks_each_epoch=False,
        seed=42,
        drop_last=True,
    ):
        if not groups:
            raise ValueError("token packing groups must be nonempty")
        if dp_degree <= 0 or not 0 <= dp_rank < dp_degree:
            raise ValueError("invalid data-parallel rank/degree for token packing")
        normalized = []
        for group_index, group in enumerate(groups):
            if not isinstance(group, list) or not group:
                raise ValueError(
                    f"token packing group {group_index} must be a nonempty list"
                )
            if any(
                not isinstance(index, int) or isinstance(index, bool) for index in group
            ):
                raise ValueError(
                    f"token packing group {group_index} contains a non-integer index"
                )
            if any(index < 0 or index >= dataset_size for index in group):
                raise ValueError(
                    f"token packing group {group_index} contains an index outside "
                    f"[0, {dataset_size})"
                )
            normalized.append(list(group))
        self.all_groups = normalized
        self.dp_rank = int(dp_rank)
        self.dp_degree = int(dp_degree)
        self.num_replicas = self.dp_degree
        self.shuffle_chunks_each_epoch = bool(shuffle_chunks_each_epoch)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.start_index = 0
        self.num_groups_per_rank = (
            len(self.all_groups) // self.dp_degree
            if self.drop_last
            else math.ceil(len(self.all_groups) / self.dp_degree)
        )

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return max(0, self.num_groups_per_rank - self.start_index)

    def __iter__(self):
        usable = self.num_groups_per_rank * self.dp_degree
        indices = list(range(usable if self.drop_last else len(self.all_groups)))
        if self.shuffle_chunks_each_epoch:
            rng = random.Random(self.seed + self.epoch)
            blocks = [
                indices[start : start + self.dp_degree]
                for start in range(0, usable, self.dp_degree)
            ]
            for block in blocks:
                rng.shuffle(block)
            rng.shuffle(blocks)
            indices = [index for block in blocks for index in block]
        rank_indices = indices[self.dp_rank :: self.dp_degree][
            : self.num_groups_per_rank
        ]
        if self.start_index:
            rank_indices = rank_indices[self.start_index :]
            self.start_index = 0
        for group_index in rank_indices:
            yield self.all_groups[group_index]


class RepeatingSFTIterator:
    """Repeat one DataLoader while keeping trainer checkpoint cursors."""

    def __init__(self, loader, sampler, scalar_states):
        self.loader = loader
        self.sampler = sampler
        self.scalar_states = scalar_states
        state = scalar_states
        state.epoch.setdefault("video", 0)
        state.consumed_samples_per_dp.setdefault("video", 0)
        state.consumed_groups_per_dp.setdefault("video", 0)
        state.epoch_consumed_samples.setdefault("video", 0)
        sampler.set_epoch(state.epoch["video"])
        sampler.start_index = (
            state.consumed_groups_per_dp["video"]
            if getattr(sampler, "offset_unit", "samples") == "groups"
            else state.epoch_consumed_samples["video"]
        )
        self._iterator = iter(loader)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self._iterator)
        except StopIteration:
            state = self.scalar_states
            state.epoch["video"] += 1
            state.consumed_samples_per_dp["video"] = 0
            state.consumed_groups_per_dp["video"] = 0
            state.epoch_consumed_samples["video"] = 0
            self.sampler.start_index = 0
            self.sampler.set_epoch(state.epoch["video"])
            self._iterator = iter(self.loader)
            batch = next(self._iterator)
        count = batch["num_packed_samples"]
        self.scalar_states.consumed_samples_per_dp["video"] += count
        self.scalar_states.consumed_groups_per_dp["video"] += 1
        self.scalar_states.epoch_consumed_samples["video"] += count
        return batch


def create_single_file_sft_dataloader(args, text_encoder, dp_rank=0, dp_degree=1):
    path = getattr(args.dataset, "data_file", None)
    dataset = SingleFileSFTDataset(path)
    packing = bool(getattr(args.dataset, "token_packing", False))
    groups_path = getattr(args.dataset, "token_packing_sorted_groups_path", None)
    if packing:
        if not groups_path:
            raise ValueError(
                "dataset.token_packing=true requires "
                "dataset.token_packing_sorted_groups_path"
            )
        groups_path = Path(groups_path).expanduser()
        if not groups_path.is_file():
            raise FileNotFoundError(f"token packing groups not found: {groups_path}")
        with groups_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        groups = payload.get("groups") if isinstance(payload, dict) else None
        saved_num_samples = (
            payload.get("num_samples") if isinstance(payload, dict) else None
        )
        if saved_num_samples is not None and int(saved_num_samples) != len(dataset):
            raise ValueError(
                f"token packing groups were built for {saved_num_samples} samples, "
                f"but dataset has {len(dataset)}"
            )
        sampler = _PackingBatchSampler(
            groups,
            dataset_size=len(dataset),
            dp_rank=dp_rank,
            dp_degree=dp_degree,
            shuffle_chunks_each_epoch=bool(
                getattr(args.dataset, "token_packing_shuffle_chunks_each_epoch", False)
            ),
            seed=args.global_seed,
            drop_last=True,
        )
        if len(sampler) == 0:
            raise ValueError(
                "token packing plan needs at least one group per data-parallel rank"
            )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=0,
            collate_fn=SegmentSFTCollator(text_encoder, args.text_len, args.dataset),
        )
        return dataset, sampler, loader

    sampler = _OffsetDistributedSampler(
        dataset,
        num_replicas=dp_degree,
        rank=dp_rank,
        shuffle=True,
        seed=args.global_seed,
        drop_last=True,
    )
    if len(sampler) == 0:
        raise ValueError("SFT data file needs at least one row per data-parallel rank")
    loader = DataLoader(
        dataset,
        batch_size=args.dataset.video_batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
        collate_fn=SegmentSFTCollator(text_encoder, args.text_len, args.dataset),
    )
    return dataset, sampler, loader
