"""Text-to-image at 512 px: one forward pass per image.

    python generate.py --weights <jeveration.pt> --decoder <decoder512.pt> \
        --prompt "An oil painting of sunflowers in a blue ceramic vase" --seed 0 --out samples/

Writes <out>/<index>.png for every prompt; image i is drawn with seed + i (at the default batch size 1).
"""
import argparse
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="the Jeveration checkpoint")
    ap.add_argument("--decoder", required=True, help="the 512-px decoder checkpoint (the source decoder's file works too)")
    ap.add_argument("--prompt", action="append", default=[], help="a prompt; repeat for several")
    ap.add_argument("--prompts-file", default="", help="a text file with one prompt per line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--out", default="samples")
    ap.add_argument("--backbone", default=None, help="local copy of Qwen3.5-4B (default: the Hugging Face Hub)")
    ap.add_argument("--tatitok", default=None, help="local copy of the TA-TiTok VQ-128 tokenizer (default: the Hub)")
    ap.add_argument("--clip", default=None, help="local folder with CLIP ViT-L/14-336's open_clip_model.safetensors (default: the Hub)")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    prompts = list(a.prompt)
    if a.prompts_file:
        prompts += [l.strip() for l in open(a.prompts_file, encoding="utf-8") if l.strip()]
    assert prompts, "give --prompt or --prompts-file"

    from jeveration import JeverationPipeline
    pipe = JeverationPipeline(a.weights, decoder=a.decoder, backbone=a.backbone, tatitok=a.tatitok, clip=a.clip, device=a.device)
    os.makedirs(a.out, exist_ok=True)
    for i in range(0, len(prompts), a.batch_size):
        batch = prompts[i:i + a.batch_size]
        images = pipe.text_to_image(batch, pipe.generator(a.seed + i))
        for j, im in enumerate(images):
            path = os.path.join(a.out, f"{i + j:05d}.png")
            Image.fromarray(im).save(path)
            print(f"{path}: {batch[j]}")


if __name__ == "__main__":
    main()
