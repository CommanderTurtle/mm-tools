import inspect
import numpy as np
import torch
import torch.cuda.amp as amp

from .utils import HuggingfaceTokenizer
from .utils.utils import to_
import wanxiang.models as models
from .models.clip import clip_xlm_roberta_vit_h_14
import torch.nn.functional as F


class T5Encoder:

    def __init__(
        self,
        name,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
    ):
        self.name = name
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        model = (
            getattr(models, name)(
                encoder_only=True, return_tokenizer=False, dtype=dtype, device=device
            )
            .eval()
            .requires_grad_(False)
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model = model

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=text_len, clean="whitespace"
        )

    def __call__(self, texts):
        ids, mask = to_(
            self.tokenizer(texts, return_mask=True, add_special_tokens=True),
            self.device,
        )
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]


class CLIP:

    def __init__(self, name, dtype, device, checkpoint_path, tokenizer_path):
        self.name = name
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        # init model
        self.model, self.transforms = clip_xlm_roberta_vit_h_14(
            pretrained=False,
            return_transforms=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device,
        )
        self.model = self.model.eval().requires_grad_(False)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        # init tokenizer
        self.tokenizer = HuggingfaceTokenizer(
            name=tokenizer_path, seq_len=self.model.max_text_len - 2, clean="whitespace"
        )

    def visual(self, videos):
        video_lens = [u.size(1) for u in videos]

        # preprocess
        size = (self.model.image_size,) * 2
        videos = torch.cat(
            [
                F.interpolate(
                    u.transpose(0, 1), size=size, mode="bicubic", align_corners=False
                )
                for u in videos
            ]
        )
        videos = self.transforms.transforms[-1](videos.mul_(0.5).add_(0.5))

        # forward
        with amp.autocast(dtype=self.dtype):
            out = self.model.visual(videos, use_31_block=True)
            return out


def get_sampling_sigmas(sampling_steps, shift):
    sigma = np.linspace(1, 0, sampling_steps + 1)[:sampling_steps]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)

    return sigma


def retrieve_timesteps(
    scheduler,
    num_inference_steps=None,
    device=None,
    timesteps=None,
    sigmas=None,
    **kwargs,
):
    if timesteps is not None and sigmas is not None:
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def get_i2v_mask(lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
    """注意，这里传入的mask_len是在原始像素空间的。

    Args:
        lat_t (_type_): _description_
        lat_h (_type_): _description_
        lat_w (_type_): _description_
        mask_len (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    msk = torch.zeros(1, (lat_t-1) * 4 + 1, lat_h, lat_w, device=device)
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk

def get_i2vmask_mask(mask_pixel_values, lat_t, lat_h, lat_w, mask_len=1, device="cuda"):
    """注意，这里传入的mask_len是在原始像素空间的。

    Args:
        lat_t (_type_): _description_
        lat_h (_type_): _description_
        lat_w (_type_): _description_
        mask_len (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    msk = mask_pixel_values.clone()
    msk[:, :mask_len] = 1
    msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
    msk = msk.transpose(1, 2)[0]
    return msk

