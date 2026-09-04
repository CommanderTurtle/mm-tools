"""Full-resolution alpha edge refinement; never thresholds the source's black pixels."""
import io

import cv2
import numpy as np
from PIL import Image, ImageOps

from .regions import check_size, png


def refine_alpha(source: Image.Image, alpha: Image.Image, radius=8, tile_size=1024) -> Image.Image:
    if source.size != alpha.size:
        raise ValueError("Alpha dimensions must match the original image.")
    if not 1 <= radius <= 32 or tile_size < 64:
        raise ValueError("Invalid alpha refinement settings.")
    # Guided filtering has a two-radius dependency. Haloed tiles reproduce the
    # full-image operation without allocating many full 4K/8K float buffers.
    output = Image.new("L", source.size)
    halo = radius * 2
    kernel = (radius * 2 + 1,) * 2
    mean = lambda array: cv2.boxFilter(array, -1, kernel, borderType=cv2.BORDER_REFLECT)
    for y in range(0, source.height, tile_size):
        for x in range(0, source.width, tile_size):
            x2, y2 = min(x + tile_size, source.width), min(y + tile_size, source.height)
            box = (max(0, x-halo), max(0, y-halo), min(source.width, x2+halo), min(source.height, y2+halo))
            guide = np.asarray(source.crop(box).convert("L"), dtype=np.float32) / 255
            prior = np.asarray(alpha.crop(box).convert("L"), dtype=np.float32) / 255
            mean_i, mean_p = mean(guide), mean(prior)
            var_i = mean(guide * guide) - mean_i * mean_i
            cov = mean(guide * prior) - mean_i * mean_p
            a = cov / (np.maximum(var_i, 0) + .001)
            b = mean_p - a * mean_i
            refined = np.clip(mean(a) * guide + mean(b), 0, 1)
            result = Image.fromarray(np.rint(refined * 255).astype(np.uint8))
            output.paste(result.crop((x-box[0], y-box[1], x2-box[0], y2-box[1])), (x, y))
    return output


def refine_matte(image_bytes: bytes, matte_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as image:
        check_size(image)
        source = ImageOps.exif_transpose(image).convert("RGB")
    with Image.open(io.BytesIO(matte_bytes)) as image:
        check_size(image)
        if "A" not in image.getbands():
            raise ValueError("The alpha stage returned no alpha channel.")
        alpha = image.getchannel("A")
    return png(refine_alpha(source, alpha))
