# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


class Stream:
    """
    Minimal interface for a stream
    """

    def __init__(self, stream_id: int):
        """
        Args:
            stream_id: (int) The id of the stream.
        """
        self._stream_id = stream_id
        self.frame_count = 0

    def __iter__(self):
        """Returns the iterator object"""
        return self

    def __next__(self):
        """Get the next frame in the stream"""
        raise NotImplementedError("Subclasses must implement __next__ method")

    @property
    def stream_id(self) -> int:
        """Get the stream id"""
        return self._stream_id
