# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import logging
import os
import tarfile
from pathlib import PurePosixPath
from typing import Iterable, Optional, Union


class TarPathTraversalError(ValueError):
    """Raised when a tar member would extract outside the target directory."""


def is_safe_tar_member(member: tarfile.TarInfo, extract_to: str) -> bool:
    member_path = PurePosixPath(member.name)
    if member_path.is_absolute() or ".." in member_path.parts:
        return False

    if member.issym() or member.islnk():
        return False

    destination = os.path.realpath(os.path.join(extract_to, *member_path.parts))
    return os.path.commonpath([destination, extract_to]) == extract_to


def safe_extract(
    tar: tarfile.TarFile,
    path: str,
    members: Optional[Iterable[Union[tarfile.TarInfo, str]]] = None,
    *,
    skip_unsafe: bool = False,
) -> list[tarfile.TarInfo]:
    """Extract tar members after validating their destination paths."""
    extract_to = os.path.realpath(path)
    os.makedirs(extract_to, exist_ok=True)
    if members is None:
        members = tar.getmembers()

    extracted_members = []
    for member in members:
        member = tar.getmember(member) if isinstance(member, str) else member
        if is_safe_tar_member(member, extract_to):
            tar.extract(member, extract_to, filter="data")
            extracted_members.append(member)
            continue

        message = f"Skipping potentially unsafe tar member: {member.name}"
        if skip_unsafe:
            logging.warning(message)
            continue
        raise TarPathTraversalError(message)

    return extracted_members
