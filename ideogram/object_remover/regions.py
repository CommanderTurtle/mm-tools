"""High-resolution editing: transform a crop, preserve every pixel outside its mask."""
from __future__ import annotations

from dataclasses import dataclass
import io
import os
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

    def composite(self, generated: bytes) -> bytes:
        with Image.open(io.BytesIO(generated)) as output:
            crop = output.convert("RGB").resize(self.blend.size, Image.Resampling.LANCZOS)
        original = self.original.crop(self.box)
        merged = Image.composite(crop, original.convert("RGB"), self.blend)
        if self.original.mode == "RGBA":
            merged.putalpha(original.getchannel("A"))
        result = self.original.copy()
        result.paste(merged, self.box[:2])
        return png(result)


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
    size = tuple(max(256, round(n * scale / 16) * 16) for n in crop.size)
    return Region(original, box, blend,
                  crop.resize(size, Image.Resampling.LANCZOS), blend.resize(size, Image.Resampling.BILINEAR))


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
