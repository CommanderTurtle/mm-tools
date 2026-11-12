import re
import sys
import time
import traceback
import zipfile

from state import MODELS, PIN, atomic, options, read, require_models, run_path


def main(job):
    from PIL import Image, ImageOps

    cfg = options(read(job / "run.json")["settings"])

    def progress(stage):
        atomic(job / "progress.json", dict(stage=stage))
        print(stage, flush=True)

    progress("Preparing image")
    with Image.open(job / "upload") as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.save(job / "input.png")
    atomic(job / "image.json", dict(width=image.width, height=image.height))
    require_models()
    progress("Loading WeVisOne-VL-4B · BF16")
    from wevisone_vl import WeVisOneVL
    from wevisone_vl.engine import InferenceResult
    from wevisone_vl.prompts import build_prompt
    from wevisone_vl.visualization import parse_predictions, save_result

    engine = WeVisOneVL(str(MODELS), backend="transformers", device="cuda", dtype="bfloat16")
    started = time.perf_counter()
    if cfg["task"] == "box_matting":
        from evaluation.utils.prompt_builder import TMPL_MATTE

        # This is the released box-conditioned decoder route: the box is an input,
        # not a fabricated model prediction. Keep that distinction in the receipt.
        label = re.sub(r"[<>]", "", cfg["query"].strip()) or "object"
        prompt = cfg["prompt"].strip() or TMPL_MATTE["single"].format(expression=label)
        coords = "".join(f"<{round(v * 1000)}>" for v in cfg["boxes"][0])
        response = f"<ref>{label}</ref><box>{coords}<OBJ></box>"
        progress("Decoding reference-box alpha")
        boxes, masks = engine.decode_masks(image, prompt, response, boxes=cfg["boxes"])
    else:
        reference = [[b[0] * image.width, b[1] * image.height,
                      b[2] * image.width, b[3] * image.height] for b in cfg["boxes"]]
        prompt = cfg["prompt"].strip() or build_prompt(
            cfg["task"], cfg["query"], reference_boxes=reference, image_size=image.size)
        progress("Generating structured response")
        response = engine.generate_text(image, prompt, **{k: cfg[k] for k in (
            "max_new_tokens", "temperature", "top_p", "top_k", "repetition_penalty")})
        (job / "response.txt").write_text(response, encoding="utf-8")
        boxes, masks = [], []
        if cfg["task"] in ("segmentation", "matting") and re.search(r"<OBJ>", response, re.I):
            progress("Decoding instance masks")
            boxes, masks = engine.decode_masks(image, prompt, response)
    result = InferenceResult(response=response, prompt=prompt, boxes=boxes,
                             masks=masks, latency_s=time.perf_counter() - started)
    progress("Saving annotations and alpha")
    output = job / "output"
    save_result(output, image, result)
    predictions = parse_predictions(response, image.width, image.height)
    instances = [dict(label=label, **item) for label, items in predictions.items() for item in items]
    # Match decoder boxes geometrically, not by category dictionary order: a
    # repeated category may occur on both sides of another category in the text.
    used = set()
    for i, box in enumerate(boxes):
        pixel_box = [box[0] * image.width, box[1] * image.height,
                     box[2] * image.width, box[3] * image.height]
        matches = [(sum(abs(a - b) for a, b in zip(item["coords"], pixel_box)), j)
                   for j, item in enumerate(instances) if item["type"] == "box" and j not in used]
        match = min(matches, default=None)
        if match and match[0] <= (image.width + image.height) * 0.004:
            j = match[1]
            used.add(j)
            instances[j]["mask"] = f"masks/mask_{i:03d}.png"
        else:
            instances.append(dict(label=f"Mask {i + 1}", type="box", coords=pixel_box,
                                  mask=f"masks/mask_{i:03d}.png"))
            used.add(len(instances) - 1)
    count = re.search(r"Total number:\s*(\d+)", response, re.I)
    if count:
        count = int(count[1])
    elif cfg["task"] in ("count_box", "count_point"):
        count = len(instances) if instances or response.strip() == "None" else None
        if count is None:
            numbers = re.findall(r"-?\d+", response)
            count = int(numbers[-1]) if numbers else None
    else:
        count = None
    manifest = dict(width=image.width, height=image.height, instances=instances,
                    masks=len(masks), count=count, prompt=prompt, response=response,
                    latency_s=result.latency_s, task=cfg["task"],
                    supplied_box=cfg["task"] == "box_matting", source=PIN)
    atomic(output / "annotations.json", manifest)
    atomic(output / "settings.json", cfg)
    with zipfile.ZipFile(job / "result.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(job / "input.png", "input.png")
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, str(path.relative_to(output)))
    atomic(job / "result.json", manifest)
    progress("Complete")


if __name__ == "__main__":
    job = run_path(sys.argv[1])
    try:
        main(job)
    except Exception as error:
        atomic(job / "error.json", dict(error=str(error)))
        traceback.print_exc()
        raise SystemExit(1) from None
