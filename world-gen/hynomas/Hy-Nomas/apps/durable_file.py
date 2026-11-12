"""Helpers for publishing files reliably on local and FUSE-backed storage."""

from __future__ import annotations

import errno
import hashlib
import os
import time
from pathlib import Path


_RETRYABLE_REPLACE_ERRNOS = {
    errno.EBUSY,
    errno.EIO,
    getattr(errno, "ESTALE", 116),
}


def replace_or_overwrite(
    temporary: Path,
    destination: Path,
    *,
    attempts: int = 4,
) -> None:
    """Publish ``temporary`` at ``destination``, tolerating FUSE rename errors.

    Local filesystems use atomic ``os.replace``. Some COS/FUSE mounts
    intermittently reject replacement of an existing object with EBUSY/EIO.
    After bounded retries, overwrite the destination while the caller's lock
    is still held, flush it, and verify the byte count.
    """

    payload = temporary.read_bytes()
    expected_size = len(payload)
    expected_sha256 = hashlib.sha256(payload).digest()

    def destination_matches() -> bool:
        try:
            if destination.stat().st_size != expected_size:
                return False
            return hashlib.sha256(destination.read_bytes()).digest() == (
                expected_sha256
            )
        except OSError:
            return False

    last_error: OSError | None = None
    for attempt in range(max(1, attempts)):
        try:
            os.replace(temporary, destination)
            for verification_attempt in range(max(1, attempts)):
                if destination_matches():
                    return
                time.sleep(0.1 * (verification_attempt + 1))
            break
        except OSError as exc:
            if exc.errno not in _RETRYABLE_REPLACE_ERRNOS:
                raise
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.1 * (attempt + 1))

    for attempt in range(max(1, attempts)):
        try:
            with destination.open("wb") as target:
                target.write(payload)
                target.flush()
                os.fsync(target.fileno())
            if not destination_matches():
                raise OSError(
                    errno.EIO,
                    "published file verification failed",
                    str(destination),
                )
            return
        except OSError as exc:
            last_error = exc
            if attempt + 1 < attempts:
                time.sleep(0.25 * (attempt + 1))

    assert last_error is not None
    raise last_error
