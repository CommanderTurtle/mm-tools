"""JeverationPipeline: the three tasks of the released checkpoint, each one forward pass of the same model.

    text_to_image    prompt -> 128 codes -> plain 512-px decoder (reads the prompt)
    image_to_image   source image + instruction -> 128 codes -> source decoder (reads a text and the source image)
    dense_predict    picture + the task's instruction -> spatial queries -> segmentation / depth / surface normals
"""
import warnings

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .backbone import load_backbone, load_text_tokenizer
from .dense import resize_for_vision, to_predictions
from .images import to_uint8
from .kernels import fix_kernel_configs
from .model import Jeveration
from .prompts import PromptTokenizer, SourcePromptTokenizer, build_prefix, vision_tokens
from .tokenizer import ImageCodec


def source_to_unit(sources):
    """uint8 pictures [B, H, W, 3] on the CPU -> the decoder's source input [B, 3, H, W], float32 in [0, 1]"""
    assert sources.device.type == "cpu" and sources.dtype == torch.uint8
    return sources.permute(0, 3, 1, 2).float().div(255)


def i2i_texts(task, caption="", wording=0):
    """an image-to-image request -> (instruction the generator reads, text the decoder reads); wording: which of the task's
    trained instruction wordings to use"""
    if task in config.STYLE_INSTRUCTIONS:
        text = config.STYLE_INSTRUCTIONS[task]
        return config.STYLE_PREFIX + text, text
    ins = config.I2I_INSTRUCTIONS[task][wording]
    if "{cap}" in ins:
        return ins.replace("{cap}", caption), caption
    return ins, ins


class JeverationPipeline:
    def __init__(self, weights, decoder=None, source_decoder=None, backbone=None, tatitok=None, clip=None, device="cuda",
                 fix_kernels=True):
        """weights: the Jeveration checkpoint; decoder / source_decoder: the 512-px decoder checkpoints for text_to_image /
        image_to_image; backbone, tatitok, clip: local copies of the public models (None: the Hugging Face Hub);
        fix_kernels: hold the linear-attention kernels at one configuration, so that a seed is reproducible (kernels.py)."""
        self.device = torch.device(device)
        cuda = self.device.type == "cuda"
        if cuda:
            # TF32 on, as when the checkpoint was evaluated
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        lm, visual = load_backbone(backbone, torch.bfloat16 if cuda else torch.float32)
        self.model = Jeveration(lm, visual).to(self.device).eval()
        if fix_kernels:
            left = fix_kernel_configs()
            if left:
                warnings.warn(f"{len(left)} linear-attention kernel(s) keep their autotuning, because the installed "
                              f"flash-linear-attention or this GPU does not offer the configuration of jeveration/config.py: "
                              f"{', '.join(n.rsplit('.', 1)[-1] for n in left)}. One seed can then give slightly different "
                              f"results in different processes.")
        lat_mean, lat_std = self.model.load_weights(weights)
        tok = load_text_tokenizer(backbone)
        self.ptok, self.sptok = PromptTokenizer(tok), SourcePromptTokenizer(tok)
        self.codec = None
        if decoder or source_decoder:
            self.codec = ImageCodec(tatitok, clip).to(self.device)
            self.model.set_codebook(self.codec.codebook, lat_mean, lat_std)
            if decoder:
                self.codec.load_decoder(decoder)
            if source_decoder:
                self.codec.load_source_decoder(source_decoder)

    def generator(self, seed):
        """a random generator on the pipeline's device; one generator drives all draws of a call"""
        return torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def text_to_image(self, prompts, generator=None, return_codes=False):
        """list of B prompts -> images uint8 [B, 512, 512, 3]; return_codes: (images, codes [B, 128])"""
        assert self.codec is not None and self.codec.decoder is not None, "text_to_image needs the plain decoder"
        ids, mask = self.ptok(prompts)
        codes = self.model.sample_codes(ids.to(self.device), mask.to(self.device), generator, heads_in_autocast=True)
        images = to_uint8(self.codec.decode(codes, prompts))
        return (images, codes) if return_codes else images

    @torch.no_grad()
    def image_to_image(self, sources, task, captions=None, wording=0, generator=None, return_codes=False):
        """sources: uint8 [B, 512, 512, 3], the task's input (an edge map, a depth map, a grey picture or a photo); task: one
        of config.I2I_TASKS; captions: of the wanted images (hed / depth) -> images uint8 [B, 512, 512, 3]; return_codes:
        (images, codes [B, 128])"""
        captions = captions if captions is not None else [""] * len(sources)
        texts = [i2i_texts(task, c, wording) for c in captions]
        return self.source_to_image(sources, [t[0] for t in texts], [t[1] for t in texts],
                                    [config.DECODER_TASK_ID[task]] * len(sources), generator, return_codes)

    @torch.no_grad()
    def source_to_image(self, sources, instructions, decoder_texts, decoder_tasks, generator=None, return_codes=False):
        """image_to_image with explicit texts: per row the generator's instruction, the decoder's text and its task id"""
        assert self.codec is not None and self.codec.source_decoder is not None, "image_to_image needs the source decoder"
        src_cpu = torch.from_numpy(np.asarray(sources, dtype=np.uint8))
        src = src_cpu.to(self.device)
        assert src.dim() == 4 and tuple(src.shape[1:]) == (config.IMAGE_SIZE, config.IMAGE_SIZE, 3), tuple(src.shape)
        vis, grid = vision_tokens(self.model.visual, src)
        prefix = build_prefix(self.model.lm, self.sptok, vis, grid, [self.sptok.encode(s) for s in instructions])
        codes = self.model.sample_codes(None, None, generator, prefix=prefix)
        cond = source_to_unit(src_cpu).to(self.device)
        task = torch.tensor(decoder_tasks, dtype=torch.long, device=self.device)
        images = to_uint8(self.codec.decode(codes, decoder_texts, cond, task))
        return (images, codes) if return_codes else images

    @torch.no_grad()
    def dense_predict(self, images, tasks=config.DENSE_TASKS, return_raw=False):
        """images: uint8 [B, H, W, 3] -> {task: prediction at H x W}: seg class indices [B, H, W] (config.NYU13), depth in
        meters [B, H, W], normal unit vectors [B, H, W, 3]. return_raw: the maps before the final decision."""
        im = torch.from_numpy(np.asarray(images, dtype=np.uint8)).to(self.device)
        size, k = tuple(im.shape[1:3]), im.shape[0]
        x = resize_for_vision(im).repeat(len(tasks), 1, 1, 1)
        rows = [t for t in tasks for _ in range(k)]
        out = {}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            vis, grid = vision_tokens(self.model.visual, x)
            prefix = build_prefix(self.model.lm, self.sptok, vis, grid, [self.sptok.encode(config.DENSE_INSTRUCTIONS[t], cut=False) for t in rows])
            hq = self.model.dense_states(prefix, vis, grid, len(self.sptok.head_ids))
            for j, t in enumerate(tasks):
                o = self.model.dense_readout(t, hq[j * k:(j + 1) * k])
                out[t] = F.interpolate(o.float(), size=size, mode="bilinear", align_corners=False)
        return out if return_raw else {t: to_predictions(t, o) for t, o in out.items()}
