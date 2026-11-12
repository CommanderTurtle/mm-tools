import importlib
import re
import sys
import time
import traceback
import warnings
import zipfile

from PIL import Image, ImageOps

from state import MODELS, NATIVE, atomic, document, read, require_models


def prepare(job, meta):
    import fitz

    pages = []
    dest = job / "pages"
    dest.mkdir(exist_ok=True)
    thumbs = job / "thumbs"
    thumbs.mkdir(exist_ok=True)

    def save(image, title):
        image = ImageOps.exif_transpose(image).convert("RGBA")
        if image.width * image.height > 80_000_000:
            raise ValueError("Page exceeds 80 megapixels; lower the PDF DPI")
        white = Image.new("RGBA", image.size, "white")
        white.alpha_composite(image)
        page = white.convert("RGB")
        key = f"{len(pages) + 1:06d}.png"
        page.save(dest / key)
        width, height = page.size
        page.thumbnail((240, 320))
        page.save(thumbs / key)
        pages.append(dict(id=key, title=title, width=width, height=height))
        atomic(job / "pages.json", pages)
        atomic(
            job / "progress.json", dict(stage="preparing", pages=len(pages), text=title)
        )

    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        for item in meta["uploads"]:
            path = job / "uploads" / item["id"]
            if item["kind"] == "pdf":
                with fitz.open(path) as pdf:
                    if pdf.needs_pass:
                        raise ValueError("Unlock the PDF before uploading it")
                    for i, page in enumerate(pdf):
                        scale = meta["settings"]["dpi"] / 72
                        if (
                            page.rect.width * page.rect.height * scale * scale
                            > 80_000_000
                        ):
                            raise ValueError(
                                "PDF page exceeds 80 megapixels; lower the DPI"
                            )
                        pix = page.get_pixmap(
                            matrix=fitz.Matrix(scale, scale),
                            colorspace=fitz.csRGB,
                            alpha=False,
                        )
                        save(
                            Image.frombytes(
                                "RGB", (pix.width, pix.height), pix.samples
                            ),
                            f"{item['title']} · {i + 1}",
                        )
            else:
                with Image.open(path) as image:
                    for i in range(getattr(image, "n_frames", 1)):
                        image.seek(i)
                        save(image.copy(), item["title"] + (f" · {i + 1}" if i else ""))
    if not pages:
        raise ValueError("The document contains no pages")
    atomic(job / "pages.json", pages)


def export(job, result, ordered):
    sys.path.insert(0, str(NATIVE / "utils"))
    processor = importlib.import_module("process_unlimitedOCR2")
    raw = result["text"]
    blocks = processor._parse_blocks(raw)
    output = job / "output"
    output.mkdir(exist_ok=True)
    markdown = processor.convert(raw)
    # Convert with the version-2 layout parser, not the model's legacy
    # grounding saver. Its figure filenames retain source-page coordinates.
    figure = re.compile(r"page(\d+)_(\d+)_(\d+)_(\d+)_(\d+)\.jpg")
    for name in dict.fromkeys(re.findall(r"!\[\]\(([^)]+)\)", markdown)):
        match = figure.fullmatch(name)
        if not match:
            continue
        page, x1, y1, x2, y2 = map(int, match.groups())
        if not 1 <= page <= len(ordered) or x2 <= x1 or y2 <= y1:
            continue
        with Image.open(job / "pages" / ordered[page - 1]) as image:
            box = (
                int(image.width * x1 / 1000),
                int(image.height * y1 / 1000),
                int(image.width * x2 / 1000),
                int(image.height * y2 / 1000),
            )
            if box[2] > box[0] and box[3] > box[1]:
                image.crop(box).save(output / name, quality=95)
    for block in blocks:
        if block["bbox"] is not None:
            block["bbox"] = [max(0, min(1000, v)) for v in block["bbox"]]
    payload = dict(
        blocks=blocks,
        markdown=markdown,
        pages=ordered,
        tokens=len(result["token_ids"]),
        finish_reason=result["finish_reason"],
        expected_pages=result["expected_pages"],
        pages_out=result["num_pages_out"],
        elapsed=round(result["time"], 2),
        prefill_time=result["prefill_time"],
        capped_pages=result["capped_pages"],
    )
    (output / "document.md").write_text(markdown, encoding="utf-8")
    (output / "raw.txt").write_text(raw, encoding="utf-8")
    atomic(output / "layout.json", payload)
    atomic(output / "settings.json", read(job / "run.json"))
    with zipfile.ZipFile(
        job / "document.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for path in sorted(output.iterdir()):
            bundle.write(path, path.name)
    atomic(
        job / "result.json",
        {k: v for k, v in payload.items() if k != "blocks" and k != "markdown"},
    )


def parse(job, meta):
    require_models()
    import torch
    from transformers import AutoModel, AutoTokenizer, TextStreamer

    atomic(
        job / "progress.json",
        dict(stage="loading", text="Loading Unlimited OCR 2 · BF16"),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODELS), trust_remote_code=True, local_files_only=True
    )
    model = (
        AutoModel.from_pretrained(
            str(MODELS),
            trust_remote_code=True,
            local_files_only=True,
            use_safetensors=True,
            torch_dtype=torch.bfloat16,
        )
        .eval()
        .cuda()
    )
    module = sys.modules[type(model).__module__]
    sys.path.insert(0, str(NATIVE / "utils"))
    processor = importlib.import_module("process_unlimitedOCR2")
    stream_path = job / "stream.txt"
    with stream_path.open("w", encoding="utf-8", buffering=1) as stream:

        class StudioStreamer(TextStreamer):
            tail = ""
            page_base = 0
            updated = 0

            def on_finalized_text(self, text, stream_end=False):
                stream.write(text)
                stream.flush()
                self.tail += text
                parts = processor.PAGE_SPLIT_RE.split(self.tail)
                if len(parts) > 1:
                    self.page_base += len(parts) - 1
                    self.tail = parts[-1]
                if stream_end or time.monotonic() - self.updated >= 0.5:
                    blocks = processor._parse_blocks(
                        self.tail[-100000:], page_base=max(1, self.page_base)
                    )
                    atomic(
                        job / "progress.json",
                        dict(
                            stage="parsing",
                            text=f"Parsing page {max(1, self.page_base)}",
                            blocks=blocks,
                        ),
                    )
                    self.updated = time.monotonic()

        original = module.TextStreamer
        module.TextStreamer = StudioStreamer
        try:
            atomic(
                job / "progress.json",
                dict(stage="parsing", text="Parsing ordered pages in one request"),
            )
            cfg = meta["settings"]
            result = model.infer(
                tokenizer,
                image_file=[str(job / "pages" / page) for page in meta["order"]],
                prompt=cfg["instruction"],
                crop_mode={"auto": None, "gundam": True, "base": False}[cfg["mode"]],
                max_length=cfg["max_length"],
                max_page_tokens=cfg["max_page_tokens"],
                no_repeat_ngram_size=35,
                ngram_window=1024 if len(meta["order"]) > 1 else 128,
                temperature=0.0,
                page_count_guard=True,
                save_results=False,
                eval_mode=False,
                return_dict=True,
            )
        finally:
            module.TextStreamer = original
    sparse = getattr(model.config, "sparse_attention_config", None)
    page_id = (
        int((sparse or {}).get("page_token_id", 128827))
        if isinstance(sparse, dict)
        else 128827
    )
    page_number = page_tokens = 0
    capped = set()
    for token in result["token_ids"]:
        if token == page_id:
            page_number += 1
            page_tokens = 0
        else:
            page_tokens += 1
            if page_tokens >= cfg["max_page_tokens"]:
                capped.add(max(1, page_number))
    result["capped_pages"] = sorted(capped)
    atomic(
        job / "progress.json",
        dict(stage="exporting", text="Merging document blocks and cropping figures"),
    )
    export(job, result, meta["order"])


def main():
    job = document(sys.argv[1])
    try:
        meta = read(job / "run.json")
        if sys.argv[2] == "prepare":
            prepare(job, meta)
        else:
            parse(job, meta)
    except Exception as error:
        atomic(job / "error.json", dict(error=str(error)))
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
