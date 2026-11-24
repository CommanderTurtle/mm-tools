"""Opt-in photographic inference, using existing local weights only."""
import argparse
import io
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image
from object_remover.model_backend import EditingModels
from object_remover.native import read_source, align_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("mask", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--background-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = args.image.read_bytes()
    report = {"input_size": read_source(data).size}
    model = EditingModels()
    try:
        if not args.background_only:
            start = time.monotonic()
            data = model.remove_object(data, args.mask.read_bytes(), steps=20, guidance=2.5, seed=42)
            (args.output / "objectclear-native.png").write_bytes(data)
            report["objectclear_seconds"] = round(time.monotonic() - start, 2)
            report["objectclear_size"] = Image.open(io.BytesIO(data)).size
            print(json.dumps(report), flush=True)
            model.unload("all")
        start = time.monotonic()
        result = model.remove_background(data)
        (args.output / "background-native.png").write_bytes(result)
        before = align_source(read_source(data), 32).convert("RGB")
        after = Image.open(io.BytesIO(result))
        assert after.size == before.size and after.mode == "RGBA"
        assert after.convert("RGB").tobytes() == before.tobytes(), "matting changed source RGB"
        report.update(background_seconds=round(time.monotonic() - start, 2), background_size=after.size,
                      alpha_range=after.getchannel("A").getextrema(), background_rgb_unchanged=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
    finally:
        model.unload("all")


if __name__ == "__main__":
    main()
