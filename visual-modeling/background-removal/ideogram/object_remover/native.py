"""Source-sized model inputs. No resizing, padding, selection crops or upscaling."""
import io

from PIL import Image, ImageOps


def read_source(data):
    from .regions import check_size
    with Image.open(io.BytesIO(data)) as image:
        check_size(image)
        return ImageOps.exif_transpose(image).convert(
            "RGBA" if "A" in image.getbands() or "transparency" in image.info else "RGB")


def align_source(source, multiple):
    """Trim only right/bottom remainders, keeping all surviving coordinates unchanged."""
    width = source.width // multiple * multiple
    height = source.height // multiple * multiple
    if not width or not height:
        raise ValueError(f"Native inference needs at least {multiple} pixels on each axis; upscaling is disabled.")
    return source.crop((0, 0, width, height)) if (width, height) != source.size else source.copy()


def read_mask(data, source_size):
    from .regions import check_size
    with Image.open(io.BytesIO(data)) as image:
        check_size(image)
        mask = image.convert("L")
    if mask.size != source_size:
        raise ValueError("The selection mask must match the source image dimensions; masks are not resized.")
    if not mask.getbbox():
        raise ValueError("Paint a selection first.")
    return mask
