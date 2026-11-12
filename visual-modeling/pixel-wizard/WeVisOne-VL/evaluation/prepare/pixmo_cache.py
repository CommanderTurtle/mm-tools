"""Download PixMo images from their source URLs, or from the ``UWGZQ/pixmo_images`` cache.

The PixMo releases (pixmo-count and pixmo-points-eval) provide image URLs
instead of images, and some URLs are no longer available. ``UWGZQ/pixmo_images``
stores the original images, keyed by ``sha256(image_url)``, in 525 parquet
files sorted by that key (about 266 GB in total). Each file has one row group
with uncompressed pages, so the reader below downloads only the requested
images with HTTP range requests instead of whole files.
"""

from __future__ import annotations

import hashlib
import io
import os
import struct
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import requests

CACHE_REPO = "UWGZQ/pixmo_images"
CACHE_SHARDS = 525
USER_AGENT = "Mozilla/5.0 (WeVisOne-VL eval data preparation)"


def cache_key(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def download(url: str) -> bytes | None:
    try:
        response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(15, 60))
        response.raise_for_status()
        return response.content
    except Exception:
        return None


def fetch(urls: Iterable[str], accept: Callable[[str, bytes], bool] = lambda url, payload: True,
          workers: int = 32) -> dict[str, bytes]:
    """Return ``{url: bytes}`` for every URL whose bytes pass ``accept``.

    Source URLs are tried first; URLs that fail (or are rejected) are looked up in the cache.
    """
    urls = sorted(set(urls))
    found: dict[str, bytes] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for url, payload in zip(urls, pool.map(download, urls)):
            if payload is not None and accept(url, payload):
                found[url] = payload
    pending = [url for url in urls if url not in found]
    print(f"source URLs: {len(found)}/{len(urls)}; looking up {len(pending)} in {CACHE_REPO}", flush=True)
    if pending:
        cached = _from_cache({cache_key(url): url for url in pending}, workers)
        for url, payload in cached.items():
            if accept(url, payload):
                found[url] = payload
    return found


def _range_get(session: requests.Session, url: str, start: int, end: int) -> tuple[bytes, str, int]:
    """Inclusive byte range; returns (payload, final URL after redirects, file size)."""
    error: Exception | None = None
    for attempt in range(5):
        try:
            response = session.get(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
                                   timeout=(20, 120))
            response.raise_for_status()
            content_range = response.headers.get("Content-Range", "")
            if response.status_code != 206 or "/" not in content_range:
                raise IOError(f"server ignored byte range: status={response.status_code}")
            if len(response.content) != end - start + 1:
                raise IOError("short range response")
            return response.content, response.url, int(content_range.rsplit("/", 1)[1])
        except Exception as exc:
            error = exc
            time.sleep(min(8.0, 0.5 * 2**attempt))
    raise error


class _HTTPFile(io.RawIOBase):
    """Seekable HTTP file, enough for pyarrow to read parquet metadata and small columns."""

    def __init__(self, url: str):
        self.session = requests.Session()
        _, self.url, self.size = _range_get(self.session, url, 0, 0)
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = os.SEEK_SET) -> int:
        base = {os.SEEK_SET: 0, os.SEEK_CUR: self.position, os.SEEK_END: self.size}[whence]
        self.position = base + offset
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size:
            return 0
        end = min(self.size - 1, self.position + len(buffer) - 1)
        payload, self.url, _ = _range_get(self.session, self.url, self.position, end)
        buffer[: len(payload)] = payload
        self.position += len(payload)
        return len(payload)


# Minimal Thrift compact-protocol reader for parquet page headers.

def _varint(data: bytes, position: int) -> tuple[int, int]:
    value = shift = 0
    while True:
        byte = data[position]
        position += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, position
        shift += 7


def _zigzag(value: int) -> int:
    return (value >> 1) ^ -(value & 1)


def _compact_value(data: bytes, position: int, kind: int) -> tuple[Any, int]:
    if kind in (1, 2):
        return kind == 1, position
    if kind == 3:
        return data[position], position + 1
    if kind in (4, 5, 6):
        value, position = _varint(data, position)
        return _zigzag(value), position
    if kind == 7:
        return struct.unpack_from("<d", data, position)[0], position + 8
    if kind == 8:
        length, position = _varint(data, position)
        return data[position: position + length], position + length
    if kind in (9, 10):
        header = data[position]
        position += 1
        length, element = header >> 4, header & 0x0F
        if length == 15:
            length, position = _varint(data, position)
        values = []
        for _ in range(length):
            value, position = _compact_value(data, position, element)
            values.append(value)
        return values, position
    if kind == 11:
        length, position = _varint(data, position)
        if length == 0:
            return {}, position
        key_kind, item_kind = data[position] >> 4, data[position] & 0x0F
        position += 1
        values = {}
        for _ in range(length):
            key, position = _compact_value(data, position, key_kind)
            values[key], position = _compact_value(data, position, item_kind)
        return values, position
    if kind == 12:
        return _compact_struct(data, position)
    raise ValueError(f"unsupported compact-protocol type {kind}")


def _compact_struct(data: bytes, position: int = 0) -> tuple[dict[int, Any], int]:
    fields: dict[int, Any] = {}
    field_id = 0
    while True:
        header = data[position]
        position += 1
        if header == 0:
            return fields, position
        delta, kind = header >> 4, header & 0x0F
        if delta:
            field_id += delta
        else:
            encoded, position = _varint(data, position)
            field_id = _zigzag(encoded)
        fields[field_id], position = _compact_value(data, position, kind)


def _dictionary_indices(payload: bytes, count: int) -> list[int]:
    """Decode a v1 data page of RLE/bit-packed dictionary indices (after definition levels)."""
    position = 4 + struct.unpack_from("<I", payload, 0)[0]
    bit_width = payload[position]
    position += 1
    output: list[int] = []
    while len(output) < count:
        header, position = _varint(payload, position)
        if header & 1:
            groups = header >> 1
            packed = int.from_bytes(payload[position: position + groups * bit_width], "little")
            position += groups * bit_width
            mask = (1 << bit_width) - 1
            output.extend((packed >> (index * bit_width)) & mask for index in range(groups * 8))
        else:
            width = (bit_width + 7) // 8
            output.extend([int.from_bytes(payload[position: position + width], "little")] * (header >> 1))
            position += width
    return output[:count]


@dataclass(frozen=True)
class _Page:
    page_type: int  # 0 data v1, 2 dictionary, 3 data v2
    payload_offset: int
    payload_size: int
    num_values: int
    encoding: int | None


@dataclass
class _Shard:
    url: str
    minimum: str
    maximum: str
    bytes_start: int
    bytes_size: int


def _shard(index: int) -> _Shard:
    import pyarrow.parquet as pq

    url = (f"https://huggingface.co/datasets/{CACHE_REPO}/resolve/main/"
           f"data/train-{index:05d}-of-{CACHE_SHARDS:05d}.parquet")
    handle = _HTTPFile(url)
    group = pq.ParquetFile(handle).metadata.row_group(0)
    ids, payloads = group.column(0), group.column(2)
    if payloads.compression != "UNCOMPRESSED":
        raise NotImplementedError(f"{url}: image column is {payloads.compression}")
    start = payloads.data_page_offset
    if payloads.dictionary_page_offset:
        start = min(start, payloads.dictionary_page_offset)
    return _Shard(handle.url, str(ids.statistics.min), str(ids.statistics.max), start,
                  payloads.total_compressed_size)


def _row_indices(shard: _Shard, keys: set[str]) -> dict[int, str]:
    import pyarrow.parquet as pq

    ids = pq.ParquetFile(_HTTPFile(shard.url)).read(columns=["image_id"])["image_id"].to_pylist()
    return {index: key for index, key in enumerate(ids) if key in keys}


def _pages(session: requests.Session, shard: _Shard) -> list[_Page]:
    pages = []
    position, end = shard.bytes_start, shard.bytes_start + shard.bytes_size
    while position < end:
        prefix, shard.url, _ = _range_get(session, shard.url, position, min(position + 4095, end - 1))
        header, header_size = _compact_struct(prefix)
        page_type = int(header[1])
        nested = {0: header.get(5), 2: header.get(7), 3: header.get(8)}.get(page_type)
        if nested is None:
            num_values, encoding = 0, None
        else:
            num_values, encoding = int(nested[1]), int(nested[4 if page_type == 3 else 2])
        payload_size = int(header[3])
        pages.append(_Page(page_type, position + header_size, payload_size, num_values, encoding))
        position += header_size + payload_size
    return pages


def _byte_arrays(session: requests.Session, shard: _Shard, start: int, wanted: set[int]) -> dict[int, bytes]:
    """Walk a PLAIN BYTE_ARRAY run, downloading only the wanted values."""
    output = {}
    position = start
    for index in range(max(wanted) + 1):
        length_bytes, shard.url, _ = _range_get(session, shard.url, position, position + 3)
        length = struct.unpack("<I", length_bytes)[0]
        if index in wanted:
            output[index], shard.url, _ = _range_get(session, shard.url, position + 4, position + 3 + length)
        position += 4 + length
    return output


def _extract(shard: _Shard, rows: dict[int, str]) -> dict[str, bytes]:
    session = requests.Session()
    pages = _pages(session, shard)
    dictionary = next((page for page in pages if page.page_type == 2), None)
    dictionary_targets: dict[int, str] = {}
    plain_targets: dict[_Page, dict[int, str]] = defaultdict(dict)
    row_start = 0
    for page in pages:
        if page.page_type not in (0, 3):
            continue
        hits = {row: key for row, key in rows.items() if row_start <= row < row_start + page.num_values}
        if hits:
            if page.page_type != 0:
                raise NotImplementedError("DATA_PAGE_V2 image pages are not supported")
            if page.encoding in (2, 8):
                payload, shard.url, _ = _range_get(session, shard.url, page.payload_offset,
                                                   page.payload_offset + page.payload_size - 1)
                indices = _dictionary_indices(payload, page.num_values)
                dictionary_targets.update({indices[row - row_start]: key for row, key in hits.items()})
            elif page.encoding == 0:
                plain_targets[page].update({row - row_start: key for row, key in hits.items()})
            else:
                raise NotImplementedError(f"unsupported image page encoding {page.encoding}")
        row_start += page.num_values

    output: dict[str, bytes] = {}
    if dictionary_targets:
        values = _byte_arrays(session, shard, dictionary.payload_offset, set(dictionary_targets))
        output.update({dictionary_targets[index]: payload for index, payload in values.items()})
    for page, targets in plain_targets.items():
        prefix, shard.url, _ = _range_get(session, shard.url, page.payload_offset, page.payload_offset + 3)
        start = page.payload_offset + 4 + struct.unpack("<I", prefix)[0]
        values = _byte_arrays(session, shard, start, set(targets))
        output.update({targets[index]: payload for index, payload in values.items()})
    return output


def _from_cache(url_of_key: dict[str, str], workers: int) -> dict[str, bytes]:
    candidates = set()
    for key in url_of_key:
        predicted = int(key, 16) * CACHE_SHARDS >> 256
        candidates.update(range(max(0, predicted - 3), min(CACHE_SHARDS, predicted + 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        shards = list(pool.map(_shard, sorted(candidates)))
    keys_by_shard: dict[int, set[str]] = defaultdict(set)
    for key in url_of_key:
        owners = [index for index, shard in enumerate(shards) if shard.minimum <= key <= shard.maximum]
        if len(owners) == 1:
            keys_by_shard[owners[0]].add(key)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        located = dict(zip(keys_by_shard, pool.map(lambda item: _row_indices(shards[item[0]], item[1]),
                                                   keys_by_shard.items())))
        extracted = pool.map(lambda item: _extract(shards[item[0]], item[1]),
                             [(index, rows) for index, rows in located.items() if rows])
        payloads = {key: payload for part in extracted for key, payload in part.items()}
    print(f"cache: recovered {len(payloads)}/{len(url_of_key)}", flush=True)
    return {url_of_key[key]: payload for key, payload in payloads.items()}
