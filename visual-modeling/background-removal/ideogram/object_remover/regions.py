"""Source-sized masked editing; trim only the model's alignment remainder."""
from __future__ import annotations

from dataclasses import dataclass
import io
import os
import base64
from PIL import Image, ImageFilter, ImageOps


def check_size(image):
    if image.width * image.height > int(os.getenv("OBJECT_REMOVER_MAX_PIXELS", "100000000")):
        raise ValueError("Image exceeds OBJECT_REMOVER_MAX_PIXELS (default 100 million pixels).")


def png(image: Image.Image) -> bytes:
    out = io.BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


@dataclass
class Region:
    original: Image.Image
    box: tuple[int, int, int, int]
    blend: Image.Image
    image: Image.Image
    mask: Image.Image
    content_box: tuple[int, int, int, int]
    selection_box: tuple[int, int, int, int]
    input_size: tuple[int, int]

    def composite(self, generated: bytes) -> bytes:
        with Image.open(io.BytesIO(generated)) as output:
            if output.size != self.image.size:
                raise ValueError("Generated image dimensions do not match the prepared edit.")
            crop = output.convert("RGB").crop(self.content_box)
        if crop.size != self.blend.size:
            raise ValueError("Native edit geometry changed; output resizing is disabled.")
        original = self.original.crop(self.box)
        merged = Image.composite(crop, original.convert("RGB"), self.blend)
        if self.original.mode == "RGBA":
            merged.putalpha(original.getchannel("A"))
        result = self.original.copy()
        result.paste(merged, self.box[:2])
        return png(result)

    def review(self) -> dict:
        """Small inspection previews and exact geometry; never starts model inference."""
        preview = self.image.copy()
        preview.thumbnail((768, 768), Image.Resampling.LANCZOS)
        selected = self.mask.resize(preview.size, Image.Resampling.BILINEAR)
        overlay = Image.composite(Image.new("RGB", preview.size, "#ff3b30"), preview,
                                  selected.point(lambda a: round(a * .45)))
        data_url = lambda image: "data:image/png;base64," + base64.b64encode(png(image)).decode("ascii")
        content_size = (self.content_box[2] - self.content_box[0], self.content_box[3] - self.content_box[1])
        return {"source_size": self.input_size, "crop_box": self.box,
                "selection_box": self.selection_box, "processing_size": self.image.size,
                "content_box": self.content_box, "downscaled": content_size != self.blend.size,
                "trimmed_pixels": [self.input_size[i] - self.original.size[i] for i in range(2)],
                "processing_megapixels": round(self.image.width * self.image.height / 1e6, 3),
                "source_preview": data_url(preview), "mask_preview": data_url(overlay),
                "native_alpha_generation": False}


def prepare_region(image_bytes: bytes, mask_bytes: bytes, *,
                   feather: int = 8, invert: bool = False) -> Region:
    from .native import read_source, read_mask, align_source
    if not 0 <= feather <= 128:
        raise ValueError("Invalid mask feather.")
    source = read_source(image_bytes)
    mask = read_mask(mask_bytes, source.size)
    original = align_source(source, 16)
    box = (0, 0, original.width, original.height)
    mask = mask.crop(box)
    if invert:
        mask = ImageOps.invert(mask)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Paint a selection first.")
    blend = mask.filter(ImageFilter.GaussianBlur(feather)) if feather else mask
    return Region(original, box, blend, original.convert("RGB"), blend.copy(), box, bounds, source.size)


def prepare_review(image_bytes: bytes, mask_bytes: bytes, **options) -> dict:
    return prepare_region(image_bytes, mask_bytes, **options).review()


def cutout(image_bytes: bytes, mask_bytes: bytes, invert: bool = False) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        check_size(img)
        source = ImageOps.exif_transpose(img).convert("RGBA")
    with Image.open(io.BytesIO(mask_bytes)) as img:
        check_size(img)
        mask = img.convert("L")
    if source.size != mask.size:
        raise ValueError("The selection mask must match the original image dimensions.")
    if not mask.getbbox():
        raise ValueError("Paint the foreground selection first.")
    from PIL import ImageChops
    alpha = ImageOps.invert(mask) if invert else mask
    source.putalpha(ImageChops.multiply(source.getchannel("A"), alpha))
    return png(source)
