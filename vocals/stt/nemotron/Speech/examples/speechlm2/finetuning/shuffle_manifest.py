# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Shuffle a training manifest on disk (seeded), in place.

Lhotse streams a NeMo manifest sequentially through a `shuffle_buffer_size` (10,000) buffer.
A manifest sorted by label and larger than the buffer is therefore trained in label-sorted
stretches, and the model collapses to whatever it saw last (CommonLanguage: every answer
"Maltese"). Sets smaller than the buffer are shuffled fully and are safe.

Usage: shuffle_manifest.py <train.json> [...]
"""
import json
import os
import random
import sys
import tempfile


def atomic_write(path, text):
    """Write to a sibling temporary file, fsync it, then atomically replace `path`.

    An interruption leaves either the old manifest or the new one, never a truncated training set.
    """
    d = os.path.dirname(os.path.abspath(path))
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".shuffle.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


for p in sys.argv[1:]:
    with open(p) as f:
        rows = f.read().splitlines()
    before = sum(json.loads(a)["text"] != json.loads(b)["text"] for a, b in zip(rows, rows[1:])) / max(
        1, len(rows) - 1
    )
    random.Random(0).shuffle(rows)
    after = sum(json.loads(a)["text"] != json.loads(b)["text"] for a, b in zip(rows, rows[1:])) / max(1, len(rows) - 1)
    atomic_write(p, "\n".join(rows) + "\n")
    print(f"{p}: {len(rows)} rows, label-change rate {before:.3f} -> {after:.3f}")
