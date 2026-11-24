"""Image-to-image at 512 px: one forward pass per image.

    structure -> image    --task hed | depth      --image <map or photo> --caption "<what the image should show>"
    colorization          --task colorize         --image <photo>          (read as grey)
    style transfer        --task anime | ghibli | watercolor   --image <photo>

    python image_to_image.py --weights <jeveration.pt> --source-decoder <source_decoder512.pt> \
        --task hed --image edges.png --caption "a red vintage car parked on a cobblestone street" --out car.png

For hed / depth the input is the task's map (soft edges, white on black / a depth map, bright = near); with --from-photo
it is a photo and the map is computed first.
"""
import argparse

from PIL import Image


def main():
    from jeveration import config
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="the Jeveration checkpoint")
    ap.add_argument("--source-decoder", required=True, help="the source-conditioned 512-px decoder checkpoint")
    ap.add_argument("--task", required=True, choices=config.I2I_TASKS)
    ap.add_argument("--image", required=True)
    ap.add_argument("--caption", default="", help="hed / depth: a caption of the wanted image")
    ap.add_argument("--wording", type=int, default=0, choices=[0, 1, 2], help="hed / depth / colorize: which of the task's three "
                    "trained instruction wordings to use (jeveration/config.py)")
    ap.add_argument("--from-photo", action="store_true", help="hed / depth: --image is a photo; compute its map first")
    ap.add_argument("--save-source", default="", help="also save the prepared 512-px input (the map / the grey picture) here")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="output.png")
    ap.add_argument("--backbone", default=None, help="local copy of Qwen3.5-4B (default: the Hugging Face Hub)")
    ap.add_argument("--tatitok", default=None, help="local copy of the TA-TiTok VQ-128 tokenizer (default: the Hub)")
    ap.add_argument("--clip", default=None, help="local folder with CLIP ViT-L/14-336's open_clip_model.safetensors (default: the Hub)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    from jeveration import JeverationPipeline
    from jeveration.images import load_square, to_gray
    source = load_square(a.image, config.IMAGE_SIZE)
    if a.task == "colorize":
        source = to_gray(source)
    elif a.from_photo:
        assert a.task in ("hed", "depth"), "--from-photo applies to hed / depth"
        from jeveration.annotators import structure_map
        source = structure_map(source, a.task, a.device)
    if a.save_source:
        Image.fromarray(source).save(a.save_source)
    pipe = JeverationPipeline(a.weights, source_decoder=a.source_decoder, backbone=a.backbone, tatitok=a.tatitok, clip=a.clip,
                              device=a.device)
    images = pipe.image_to_image(source[None], a.task, [a.caption], a.wording, pipe.generator(a.seed))
    Image.fromarray(images[0]).save(a.out)
    print(a.out)


if __name__ == "__main__":
    main()
