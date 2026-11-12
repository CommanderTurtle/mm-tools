"""Preprocessing, serialization and single-file output contract tests."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vae_latent_encoding import (
    PreparedVideo, encode_row, normalize_pixels, prepare_video, read_arrow,
    crop_window, sample_indices, select_rows, serialize_latent,
    update_boundaries, video_path,
)
from hymm.dataset.single_file_sft import SingleFileSFTDataset


SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("latent", pa.binary()),
    ("latent_shape", pa.list_(pa.int32())), ("prompts", pa.list_(pa.string())),
    ("segment_boundaries", pa.list_(pa.int32())), ("camera_extrinsics", pa.binary()),
    ("camera_intrinsics", pa.binary()), ("has_camera", pa.bool_()), ("video_path", pa.string()),
])


def row():
    return {
        "sample_id": "test", "latent": None, "latent_shape": None,
        "prompts": [json.dumps({"segments": [{"contained_actions": []}]})],
        "segment_boundaries": [0, 4], "camera_extrinsics": None,
        "camera_intrinsics": None, "has_camera": False, "video_path": "original.mp4",
    }


def write_source(path, height=64, width=96):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.thread_count = 1
        for index in range(15):
            frame = av.VideoFrame.from_ndarray(
                np.broadcast_to(
                    (np.arange(height, dtype=np.uint8) + index * 10)[:, None, None],
                    (height, width, 3),
                ).copy(), format="rgb24"
            )
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_center_crop_720p_to_704p_without_padding_or_resizing():
    assert crop_window(720, 1280, 704, 1280) == (8, 0)
    assert crop_window(64, 96, 64, 96) == (0, 0)
    with pytest.raises(ValueError, match="multiples of 32"):
        crop_window(720, 1280, 720, 1280)
    with pytest.raises(ValueError, match="smaller than crop"):
        crop_window(64, 96, 704, 1280)


def test_sample_indices_use_exact_24fps_grid():
    np.testing.assert_array_equal(
        sample_indices(15, 30, 24), np.rint(np.arange(12) * 30 / 24).astype(np.int64)
    )
    assert len(sample_indices(15, 12, 24)) == 29
    assert len(sample_indices(1797, 30, 24)) == 1437


def test_prepare_video_decodes_original_without_rewriting_it(tmp_path):
    source = tmp_path / "original.mp4"
    write_source(source)
    original = source.read_bytes()
    args = SimpleNamespace(input_arrow=tmp_path / "input.arrow", fps=24, height=64, width=96)
    prepared = prepare_video(row(), args)
    assert source.read_bytes() == original
    assert prepared.pixels.shape == (12, 64, 96, 3)
    assert prepared.source_frames == 15 and prepared.source_fps == 30
    assert prepared.sampled_frames == 12 and prepared.output_fps == 24


def test_cropped_pixels_are_identical_to_original_center_region(tmp_path):
    source = tmp_path / "original.mp4"
    write_source(source, height=720, width=1280)
    args = SimpleNamespace(input_arrow=tmp_path / "input.arrow", fps=24, height=704, width=1280)
    prepared = prepare_video(row(), args)
    assert prepared.pixels.shape == (12, 704, 1280, 3)
    assert (prepared.sampled_height, prepared.sampled_width) == (704, 1280)
    with av.open(str(source)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    expected = np.stack([frames[index][8:712] for index in sample_indices(15, 30, 24)])
    np.testing.assert_array_equal(prepared.pixels, expected)


def test_temporal_alignment_uses_4k_frames_without_padding(tmp_path):
    source = tmp_path / "original.mp4"
    write_source(source)
    args = SimpleNamespace(input_arrow=tmp_path / "input.arrow", fps=24, height=64, width=96)
    import vae_latent_encoding
    original = vae_latent_encoding.sample_indices
    try:
        vae_latent_encoding.sample_indices = lambda *args: original(*args)[:-1]
        prepared = prepare_video(row(), args)
    finally:
        vae_latent_encoding.sample_indices = original
    assert prepared.sampled_frames == 11
    assert len(prepared.pixels) == 8


def test_normalization_is_float32_before_bf16_rounding():
    values = np.arange(256, dtype=np.uint8).reshape(4, 4, 16, 1).repeat(3, axis=-1)
    expected = torch.from_numpy(values).float().mul_(1 / 127.5).sub_(1).permute(3, 0, 1, 2).unsqueeze(0)
    actual = normalize_pixels(values, torch.device("cpu"), torch.bfloat16)
    assert torch.equal(actual, expected.to(torch.bfloat16))


def test_boundaries_update_final_T_without_changing_internal_prompt_times():
    source = row()
    source["prompts"] *= 2
    source["segment_boundaries"] = [0, 2, 7]
    assert update_boundaries(source, 6, 24, 24) == [0, 2, 6]
    assert source["segment_boundaries"] == [0, 2, 7]
    with pytest.raises(ValueError, match="FPS"):
        update_boundaries(source, 6, 16, 24)
    with pytest.raises(ValueError, match="too short"):
        update_boundaries(source, 2, 24, 24)


def test_bf16_bytes_are_exact_and_decodable_by_training_loader():
    tensor = torch.arange(32 * 3 * 2 * 3, dtype=torch.float32).reshape(32, 3, 2, 3).bfloat16()
    payload, shape = serialize_latent(tensor)
    assert len(payload) == tensor.numel() * 2
    source = row()
    source.update(latent=payload, latent_shape=shape, segment_boundaries=[0, 3])
    decoded = SingleFileSFTDataset._decode_row(source, 0)
    assert torch.equal(decoded["latent"], tensor)


def test_nonfinite_latents_are_rejected():
    tensor = torch.zeros((32, 3, 2, 3))
    tensor[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        serialize_latent(tensor)


def test_encode_row_whole_video_and_preserves_other_columns():
    class FakeVAE:
        def encode(self, tensor):
            assert tensor.shape == (1, 3, 12, 64, 96)
            return SimpleNamespace(latent_dist=SimpleNamespace(
                mode=lambda: torch.zeros((1, 32, 3, 2, 3), dtype=torch.bfloat16)
            ))

    args = SimpleNamespace(vae_precision="bf16", boundary_fps=24)
    prepared = PreparedVideo(np.zeros((12, 64, 96, 3), dtype=np.uint8), 15, 30, 12, 24, 64, 96, 64, 96)
    original = row()
    output, stats = encode_row(original, prepared, FakeVAE(), args, torch.device("cpu"))
    assert output["latent_shape"] == [32, 3, 2, 3]
    assert output["segment_boundaries"] == [0, 3]
    assert original["latent"] is None
    for key in ("sample_id", "video_path", "prompts", "has_camera", "camera_extrinsics", "camera_intrinsics"):
        assert output[key] == original[key]
    assert stats["source_frames"] == 15


def test_local_video_paths_and_rows_are_validated(tmp_path):
    path = tmp_path / "original.mp4"
    path.touch()
    assert video_path("original.mp4", tmp_path / "data.arrow") == path
    with pytest.raises(ValueError, match="local"):
        video_path("https://example.com/video.mp4", tmp_path / "data.arrow")
    assert select_rows("all", 3) == [0, 1, 2]
    assert select_rows("2,0", 3) == [0, 2]
    with pytest.raises(ValueError):
        select_rows("0,0", 3)
    with pytest.raises(ValueError):
        select_rows("3", 3)


def test_single_file_pass_through_and_refuses_overwrites(tmp_path):
    source = row()
    source["latent"], source["latent_shape"] = serialize_latent(torch.zeros((32, 3, 2, 3)))
    source["segment_boundaries"] = [0, 3]
    input_path = tmp_path / "input.arrow"
    with pa.ipc.new_file(str(input_path), SCHEMA) as writer:
        writer.write_table(pa.Table.from_pylist([source], schema=SCHEMA))
    output_path = tmp_path / "output.arrow"
    command = [
        sys.executable, str(Path(__file__).with_name("vae_latent_encoding.py")),
        "--input-arrow", str(input_path), "--output-arrow", str(output_path),
        "--device", "cpu", "--vae-checkpoint", str(tmp_path / "missing.pt"),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)
    schema, rows = read_arrow(output_path)
    assert schema.equals(SCHEMA, check_metadata=True) and rows == [source]
    assert not list(tmp_path.glob(".*.tmp"))
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    assert result.returncode != 0 and "Refusing to overwrite" in result.stderr


def test_failed_encoding_does_not_publish_output(tmp_path):
    source = row()
    input_path = tmp_path / "input.arrow"
    with pa.ipc.new_file(str(input_path), SCHEMA) as writer:
        writer.write_table(pa.Table.from_pylist([source], schema=SCHEMA))
    output = tmp_path / "output.arrow"
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("vae_latent_encoding.py")),
         "--input-arrow", str(input_path), "--output-arrow", str(output),
         "--device", "cpu", "--vae-checkpoint", str(tmp_path / "missing.pt")],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode != 0 and not output.exists()
    assert not list(tmp_path.glob(".*.tmp"))
