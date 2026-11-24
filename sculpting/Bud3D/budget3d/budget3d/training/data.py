"""Training images: 1024x1024 renders of one object on black, Idol's own framing.

Dome-Objaverse publishes 512x512 RGBA renders, 48 dome views per object, one parquet per
object (``image_png`` holds the RGBA view).  Because the alpha is real, Idol's
preprocessing runs without BiRefNet: erode the alpha by a pixel, crop the square that
leaves a 1.2x margin around the object's bounding box, resize to 1024 and composite on
black -- the same function the released pipeline calls, so the student and the teacher see
exactly the kind of image Idol was built for.

No label, caption or camera is needed: the frozen teacher generates every training target
from the image alone.
"""
import io
import os
import random

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from ..upstream import IDOL_ROOT  # noqa: F401
from idol.preprocess import preprocess_image

CROP_MARGIN = 1.2          # Idol's 'synthetic' mode: renders of a single object


def list_objects(root):
    """Every object parquet under ``root``, sorted for reproducibility."""
    files = []
    for dirpath, _, names in os.walk(root):
        files += [os.path.join(dirpath, n) for n in names if n.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"no .parquet under {root}; run scripts/fetch_dome.py first")
    return sorted(files)


def render_to_tensor(png_bytes, image_size=1024, crop_margin=CROP_MARGIN):
    """One RGBA render -> [3, S, S] float in [0, 1], object centred on black."""
    image = Image.open(io.BytesIO(png_bytes))
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    prepared = preprocess_image(image, background_remover=None, image_size=image_size,
                                crop_margin=crop_margin)
    return torch.from_numpy(np.array(prepared)).permute(2, 0, 1).float().div_(255.0)


class DomeObjaverse(IterableDataset):
    """Streams object views, sharded over ranks and workers, reshuffled every pass.

    Each parquet is read once and ``views_per_object`` of its 48 views are emitted, which
    keeps the 13 MB read amortised while still moving on to a new object quickly.
    """

    def __init__(self, root, image_size=1024, views_per_object=6, crop_margin=CROP_MARGIN,
                 rank=0, world_size=1, seed=0):
        self.files = list_objects(root)
        self.image_size, self.views = image_size, views_per_object
        self.crop_margin = crop_margin
        self.rank, self.world_size, self.seed = rank, world_size, seed

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        shard = self.files[self.rank::self.world_size][wid::nw]
        rng = random.Random(self.seed * 9176 + self.rank * 131 + wid)
        epoch = 0
        while True:
            order = list(shard)
            rng.shuffle(order)
            for path in order:
                try:
                    table = pq.ParquetFile(path).read(columns=["image_png"]).column("image_png")
                except Exception:
                    continue
                n = len(table)
                for j in rng.sample(range(n), min(self.views, n)):
                    try:
                        yield render_to_tensor(table[j].as_py(), self.image_size, self.crop_margin)
                    except Exception:
                        continue
            epoch += 1


def image_stream(root, batch, image_size=1024, views_per_object=6, rank=0, world_size=1, seed=0,
                 num_workers=4):
    """An endless iterator of [batch, 3, S, S] image tensors."""
    ds = DomeObjaverse(root, image_size, views_per_object, rank=rank, world_size=world_size, seed=seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=num_workers,
                                         pin_memory=True, drop_last=True,
                                         persistent_workers=num_workers > 0,
                                         prefetch_factor=4 if num_workers > 0 else None)
    return iter(loader)
