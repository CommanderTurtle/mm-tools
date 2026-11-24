# Vendored from FunAudioLLM/CosyVoice (Apache-2.0, see LICENSE in this directory); inference-only subset, imports adjusted.
import random
import numpy as np
import torch


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m, mean=0.0, std=0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


# Repetition Aware Sampling in VALL-E 2


def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


