"""High-resolution editing: transform a crop, preserve every pixel outside its mask."""
from __future__ import annotations

from dataclasses import dataclass
import io
import os
import base64
import numpy as np
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

    def composite(self, generated: bytes) -> bytes:
        with Image.open(io.BytesIO(generated)) as output:
            if output.size != self.image.size:
                raise ValueError("Generated crop dimensions do not match the prepared edit.")
            crop = output.convert("RGB").crop(self.content_box).resize(self.blend.size, Image.Resampling.LANCZOS)
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
        return {"source_size": self.original.size, "crop_box": self.box,
                "selection_box": self.selection_box, "processing_size": self.image.size,
                "content_box": self.content_box, "downscaled": content_size != self.blend.size,
                "processing_megapixels": round(self.image.width * self.image.height / 1e6, 3),
                "source_preview": data_url(preview), "mask_preview": data_url(overlay),
                "native_alpha_generation": False}


def prepare_region(image_bytes: bytes, mask_bytes: bytes, *, resolution: int = 1024,
                   padding: int = 128, feather: int = 8, invert: bool = False) -> Region:
    if resolution not in {512, 768, 1024, 1536, 2048, 4096, 8192}:
        raise ValueError("Choose a listed processing resolution.")
    if not 0 <= padding <= 2048 or not 0 <= feather <= 128:
        raise ValueError("Invalid context padding or mask feather.")
    with Image.open(io.BytesIO(image_bytes)) as img:
        check_size(img)
        original = ImageOps.exif_transpose(img).convert("RGBA" if "A" in img.getbands() or "transparency" in img.info else "RGB")
    with Image.open(io.BytesIO(mask_bytes)) as img:
        check_size(img)
        mask = img.convert("L")
    if mask.size != original.size:
        raise ValueError("The selection mask must match the original image dimensions.")
    if invert:
        mask = ImageOps.invert(mask)
    bounds = mask.getbbox()
    if bounds is None:
        raise ValueError("Paint a selection first.")
    # Include feather support in the crop; there is no implicit whole-image resize.
    margin = padding + feather * 4
    box = (max(0, bounds[0]-margin), max(0, bounds[1]-margin),
           min(original.width, bounds[2]+margin), min(original.height, bounds[3]+margin))
    selected = mask.crop(box)
    blend = selected.filter(ImageFilter.GaussianBlur(feather)) if feather else selected
    crop = original.crop(box).convert("RGB")
    scale = min(1.0, resolution / max(crop.size))
    # Fit pixels uniformly, then pad to the model grid. Independently clamping
    # width/height to 256 used to stretch narrow selections and extreme ratios.
    content_size = tuple(max(1, round(n * scale)) for n in crop.size)
    size = tuple(max(256, ((n + 15) // 16) * 16) for n in content_size)
    left, top = ((size[i] - content_size[i]) // 2 for i in range(2))
    right, bottom = size[0] - content_size[0] - left, size[1] - content_size[1] - top
    fitted = crop.resize(content_size, Image.Resampling.LANCZOS)
    pixels = np.pad(np.asarray(fitted), ((top, bottom), (left, right), (0, 0)), mode="edge")
    fitted_mask = Image.new("L", size)
    fitted_mask.paste(blend.resize(content_size, Image.Resampling.BILINEAR), (left, top))
    content_box = (left, top, left + content_size[0], top + content_size[1])
    return Region(original, box, blend, Image.fromarray(pixels), fitted_mask, content_box, bounds)


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
