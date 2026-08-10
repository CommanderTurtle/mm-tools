from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "outputs" / "cli"


def _load_local_config() -> None:
    load_dotenv(ROOT / ".env", override=False)
    load_dotenv(ROOT / ".env.local", override=True)
    home = Path.home()
    qwen = home / "multimedia" / "models" / "qwen"
    os.environ.setdefault("REDESIGN_QWEN_BACKEND", "native-fp8")
    os.environ.setdefault(
        "REDESIGN_QWEN_MODEL",
        str(qwen / "T5B--qwen-image-layered-fp8" / "qwen_image_layered_fp8_e4m3fn.safetensors"),
    )
    os.environ.setdefault(
        "REDESIGN_QWEN_COMPONENTS",
        str(qwen / "diffusers--hfstaff--Qwen-Image-Layered-modular"),
    )
    os.environ.setdefault(
        "REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS",
        str(
            qwen
            / "suzukimain--extraint4stuff--Qwen-Image-Layered-Control-SDNQ-int4"
            / "text_encoder"
        ),
    )
    os.environ.setdefault("REDESIGN_QWEN_DTYPE", "bfloat16")
    os.environ.setdefault("REDESIGN_NATIVE_OFFLOAD", "model")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")


def _input(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise SystemExit(f"Input file not found: {path}")
    return path


def _output(input_path: Path, command: str, value: str | None) -> Path:
    path = Path(value).expanduser().resolve() if value else DEFAULT_OUTPUT / input_path.stem / command
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path)


def _rgba_copy(source: Path, output: Path) -> Path:
    with Image.open(source) as opened:
        opened.convert("RGBA").save(output)
    return output


def _extract_masked_layers(
    image_path: Path,
    masks: dict[str, str],
    output_dir: Path,
) -> list[dict[str, Any]]:
    source = np.asarray(Image.open(image_path).convert("RGBA"))
    result = []
    for index, (name, mask_path) in enumerate(masks.items()):
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if mask.shape != source.shape[:2]:
            mask = np.asarray(
                Image.fromarray(mask).resize((source.shape[1], source.shape[0]), Image.Resampling.NEAREST)
            )
        rgba = source.copy()
        rgba[:, :, 3] = np.minimum(rgba[:, :, 3], mask)
        rows, cols = np.where(mask > 0)
        bbox = (
            [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
            if len(rows)
            else [0, 0, 0, 0]
        )
        path = output_dir / f"{index:03d}-{name}.png"
        Image.fromarray(rgba).save(path)
        result.append({"id": name, "path": str(path), "mask_path": mask_path, "bbox": bbox})
    return result


def cmd_doctor(_args: argparse.Namespace) -> int:
    checks = {
        "venv": ROOT / ".venv" / "bin" / "python",
        "fp8_transformer": Path(os.environ["REDESIGN_QWEN_MODEL"]).expanduser(),
        "diffusers_components": Path(os.environ["REDESIGN_QWEN_COMPONENTS"]).expanduser()
        / "modular_model_index.json",
        "caption_encoder": Path(os.environ["REDESIGN_QWEN_TEXT_ENCODER_COMPONENTS"]).expanduser()
        / "config.json",
        "grounding_dino": ROOT / "weights" / "groundingdino_swinb_cogcoor.pth",
        "sam2": ROOT / "weights" / "sam2.1_hiera_large.pt",
        "hisam": ROOT / "weights" / "sam_tss_h_textseg.pth",
        "lama": ROOT / "weights" / "big-lama.pt",
    }
    failed = False
    for name, path in checks.items():
        present = path.is_file()
        failed |= not present
        print(f"{'ok' if present else 'MISSING':7} {name:22} {path}")
    print(f"profile native-fp8 / BF16 compute / model CPU offload / GPU 0")
    print(f"controller {os.environ.get('OPENAI_BASE_URL', 'unset')} :: {os.environ.get('VLM_MODEL', 'unset')}")
    return 1 if failed else 0


def cmd_decompose(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "decompose", args.output)
    command = [
        sys.executable,
        "-m",
        "ReDesign.run_single_image",
        "--image",
        str(source),
        "--output_dir",
        str(destination),
        "--qwen_gpus",
        args.qwen_gpus,
        "--qwen_pair_size",
        str(args.qwen_pair_size),
        "--tool_gpus",
        args.tool_gpus,
        "--workers",
        str(args.workers),
    ]
    print(" ".join(command), flush=True)
    return subprocess.call(command, cwd=ROOT, env=os.environ.copy())


def cmd_layers(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "layers", args.output)
    from ReDesign.tools.qwen_layered_tool import run_qwen_layered

    result = run_qwen_layered(
        str(source),
        str(destination),
        num_layers=args.count,
        seed=args.seed,
        resolution=args.resolution,
        num_inference_steps=args.steps,
        true_cfg_scale=args.cfg,
        strict_alpha_threshold=args.alpha_threshold,
    )
    _write(destination / "layers.json", result)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "split", args.output)
    working = _rgba_copy(source, destination / "source.png")
    from ReDesign.tools.cca_tool import run_split_cca

    result = run_split_cca(
        str(working),
        min_area=args.min_area,
        alpha_threshold=args.alpha_threshold,
        connectivity=args.connectivity,
    )
    _write(destination / "components.json", result)
    return 0


def cmd_vectorize(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "vectorize", args.output)
    from ReDesign.tools.vtracer_tool import run_vtracer

    result = run_vtracer(str(source), str(destination / f"{source.stem}.svg"))
    _write(destination / "vector.json", result)
    return 0


def cmd_detect(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "detect", args.output)
    working = _rgba_copy(source, destination / "source.png")
    labels = [value.strip() for value in args.labels.split(",") if value.strip()]
    if not labels:
        raise SystemExit("--labels requires at least one comma-separated object description.")
    from ReDesign.tools.dino_tool import run_dino_batch_all
    from ReDesign.tools.objectclear_tool import run_objectclear
    from ReDesign.tools.sam2_tool import run_sam2_union

    detections = run_dino_batch_all(
        str(working),
        labels,
        vis_dir=destination,
        score_min=args.score,
        area_max=args.max_area,
        top_k_per_label=args.top_k,
    )
    ids = [f"{label.replace(' ', '-')}-{index:03d}" for index, label in enumerate(detections["labels"])]
    segmentation = run_sam2_union(str(working), detections["boxes"], ids)
    foreground = _extract_masked_layers(working, segmentation["masks_by_id"], destination)
    background = run_objectclear(str(working), segmentation["mask_union"])
    _write(
        destination / "detected-layers.json",
        {
            "detections": detections,
            "segmentation": segmentation,
            "foreground": foreground,
            "background": background,
        },
    )
    return 0


def cmd_text(args: argparse.Namespace) -> int:
    source = _input(args.input)
    destination = _output(source, "text", args.output)
    working = _rgba_copy(source, destination / "source.png")
    from ReDesign.nodes.fontstyle import _estimate_font_size_from_bbox, _extract_color_from_image, _rgb_to_hex
    from ReDesign.tools.hisam_tool import run_hisam_union
    from ReDesign.tools.lama_tool import run_lama
    from ReDesign.tools.ocr_tool import _quad_to_aabb, run_ocr

    ocr = run_ocr(str(working), vis_dir=destination)
    boxes = [_quad_to_aabb(box) for box in ocr.get("boxes", [])]
    ids = [f"text-{index:03d}" for index in range(len(boxes))]
    segmentation = run_hisam_union(str(working), boxes, ids, vis_dir=destination)
    foreground = _extract_masked_layers(working, segmentation["masks_by_id"], destination)
    for layer in foreground:
        index = int(layer["id"].rsplit("-", 1)[-1])
        bbox = layer["bbox"]
        color = _extract_color_from_image(layer["path"], bbox)
        layer.update(
            content=(ocr.get("texts") or [""] * len(boxes))[index],
            confidence=(ocr.get("scores") or [None] * len(boxes))[index],
            font_family="__UNRESOLVED__",
            font_size_px=_estimate_font_size_from_bbox(bbox),
            font_color={"rgb": list(color), "hex": _rgb_to_hex(color)},
        )
    background = run_lama(str(working), segmentation["mask_union"])
    _write(
        destination / "text-layers.json",
        {"ocr": ocr, "segmentation": segmentation, "text_layers": foreground, "background": background},
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    parse_path = _input(args.parse)
    output_root = parse_path.parent
    source = _input(args.source) if args.source else None
    if source:
        with Image.open(source) as image:
            width, height = image.size
    else:
        raw = json.loads(parse_path.read_text(encoding="utf-8"))
        root_image = raw.get("root_image") or raw.get("root_image_path") if isinstance(raw, dict) else None
        if not root_image:
            raise SystemExit("Pass --source when parse.json does not identify its root image.")
        with Image.open(root_image) as image:
            width, height = image.size
    from local_app.editor import build_editor_document, export_document

    document = build_editor_document(
        root=output_root,
        parse_path=parse_path,
        canvas_width=width,
        canvas_height=height,
    )
    result = export_document(output_root, document)
    _write(output_root / "editable-export.json", result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="redesign",
        description="Local, reproducible flat-image reconstruction toolkit.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="Check the local offline runtime and model files.")
    doctor.set_defaults(func=cmd_doctor)

    decompose = sub.add_parser("decompose", help="Run the complete controller + verifier reconstruction.")
    decompose.add_argument("input")
    decompose.add_argument("-o", "--output")
    decompose.add_argument("--qwen-gpus", default="0")
    decompose.add_argument("--qwen-pair-size", type=int, default=1)
    decompose.add_argument("--tool-gpus", default="0")
    decompose.add_argument("--workers", type=int, default=1)
    decompose.set_defaults(func=cmd_decompose)

    layers = sub.add_parser("layers", help="Fork one image into z-ordered Qwen RGBA layers.")
    layers.add_argument("input")
    layers.add_argument("-o", "--output")
    layers.add_argument("--count", type=int, default=4)
    layers.add_argument("--steps", type=int, default=50)
    layers.add_argument("--resolution", type=int, choices=(640, 1024), default=640)
    layers.add_argument("--cfg", type=float, default=4.0)
    layers.add_argument("--seed", type=int, default=777)
    layers.add_argument("--alpha-threshold", type=int, default=240)
    layers.set_defaults(func=cmd_layers)

    split = sub.add_parser("split", help="Split an RGBA layer into connected components.")
    split.add_argument("input")
    split.add_argument("-o", "--output")
    split.add_argument("--min-area", type=int, default=100)
    split.add_argument("--alpha-threshold", type=int, default=10)
    split.add_argument("--connectivity", type=int, choices=(4, 8), default=8)
    split.set_defaults(func=cmd_split)

    vector = sub.add_parser("vectorize", help="Convert one raster leaf into a stacked color SVG.")
    vector.add_argument("input")
    vector.add_argument("-o", "--output")
    vector.set_defaults(func=cmd_vectorize)

    detect = sub.add_parser("detect", help="Detect, segment, extract, and remove named objects.")
    detect.add_argument("input")
    detect.add_argument("--labels", required=True, help="Comma-separated object descriptions.")
    detect.add_argument("-o", "--output")
    detect.add_argument("--score", type=float, default=0.10)
    detect.add_argument("--max-area", type=float, default=0.80)
    detect.add_argument("--top-k", type=int, default=1)
    detect.set_defaults(func=cmd_detect)

    text = sub.add_parser("text", help="OCR, segment, extract, and inpaint editable text regions.")
    text.add_argument("input")
    text.add_argument("-o", "--output")
    text.set_defaults(func=cmd_text)

    export = sub.add_parser("export", help="Build aligned PNG layers and an editable package from parse.json.")
    export.add_argument("parse")
    export.add_argument("--source", help="Original image when parse.json does not contain its path.")
    export.set_defaults(func=cmd_export)
    return parser


def main() -> None:
    _load_local_config()
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
