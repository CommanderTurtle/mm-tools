"""Minimal uncompressed run-length encoding for binary masks (column-major, COCO-style counts:
alternating background / foreground run lengths, starting with background)."""
import numpy as np


def encode(mask):
    m = np.asarray(mask, dtype=bool).ravel(order="F")
    change = np.flatnonzero(m[1:] != m[:-1]) + 1
    bounds = np.concatenate([[0], change, [m.size]])
    counts = np.diff(bounds).tolist()
    if m.size and m[0]:
        counts = [0] + counts
    return {"size": list(np.asarray(mask).shape), "counts": counts}


def encode_crop(crop, x0, y0, size):
    """encode() of the (H, W) = size mask that is `crop` at (x0, y0) and False elsewhere, from the crop only."""
    H, W = int(size[0]), int(size[1])
    c = np.asarray(crop, dtype=bool)
    h, w = c.shape
    p = np.zeros((w, h + 2), bool)
    p[:, 1:-1] = c.T
    col, row = np.nonzero(p[:, 1:] != p[:, :-1])  # run boundaries, column-major: before crop row `row` of column `col`
    b = (x0 + col) * H + y0 + row
    dup = np.flatnonzero(b[1:] == b[:-1])  # a full-height box: a run through the bottom of one column into the next
    if len(dup):
        b = np.delete(b, np.concatenate([dup, dup + 1]))
    b = b[b < H * W]
    return {"size": [H, W], "counts": np.diff(np.concatenate([[0], b, [H * W]])).tolist()}


def paste_crop(crop, x0, y0, size):
    """The (H, W) = size mask that is `crop` at (x0, y0) and False elsewhere."""
    out = np.zeros((int(size[0]), int(size[1])), bool)
    out[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return out


def decode(rle):
    h, w = rle["size"]
    out = np.zeros(h * w, dtype=bool)
    pos, val = 0, False
    for c in rle["counts"]:
        if val:
            out[pos : pos + c] = True
        pos += c
        val = not val
    return out.reshape((h, w), order="F")


def area(rle):
    return int(sum(rle["counts"][1::2]))
