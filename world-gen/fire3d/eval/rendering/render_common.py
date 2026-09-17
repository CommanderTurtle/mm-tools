"""Small renderer helpers shared by the public Fire3D interfaces."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def replace_empty_background(
    beauty_path: Path, mask_path: Path, background_rgb: list[int]
) -> None:
    """Composite an opaque beauty render over a solid color using its AA mask."""

    if len(background_rgb) != 3 or any(not 0 <= value <= 255 for value in background_rgb):
        raise ValueError("background_rgb must contain three values in [0, 255]")
    with Image.open(beauty_path) as beauty_image:
        beauty = beauty_image.convert("RGB")
    with Image.open(mask_path) as mask_image:
        mask = mask_image.convert("L")
    if beauty.size != mask.size:
        raise ValueError(
            f"Beauty/mask size mismatch: {beauty_path}={beauty.size}, "
            f"{mask_path}={mask.size}"
        )
    background = Image.new("RGB", beauty.size, tuple(background_rgb))
    Image.composite(beauty, background, mask).save(beauty_path)


def background_exclusion_filters_disagree(
    instance_mesh_names: set[str], named_mesh_names: set[str]
) -> bool:
    """Return whether two nonempty background selections are disjoint."""

    if not instance_mesh_names or not named_mesh_names:
        return False
    return instance_mesh_names.isdisjoint(named_mesh_names)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()


def labeled(image: Image.Image, title: str, subtitle: str) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, 58), fill=(12, 18, 26))
    title_font = _font(20, True)
    while (
        getattr(title_font, "size", 12) > 12
        and draw.textbbox((0, 0), title, font=title_font)[2] > image.width - 28
    ):
        title_font = _font(title_font.size - 1, True)
    draw.text((14, 7), title, font=title_font, fill="white")
    draw.text((14, 33), subtitle, font=_font(13), fill=(215, 224, 234))
    return image


def beauty_on_gray(path: Path) -> Image.Image:
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
    background = Image.new("RGBA", rgba.size, (232, 235, 239, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")
