# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


from abc import ABC, abstractmethod

from nemo.collections.asr.inference.streaming.framing.request import Request
from nemo.collections.asr.inference.streaming.framing.request_options import ASRRequestOptions


class PipelineInterface(ABC):
    """
    The base interface for streaming speech pipelines
    Base usage for all pipelines:
        pipeline.start_session()
        for requests in request_generator:
            pipeline.transcribe_step(requests)
        pipeline.close_session()
    """

    @abstractmethod
    def open_session(self):
        """
        Open a new session
        """
        raise NotImplementedError

    @abstractmethod
    def close_session(self):
        """
        End the current session
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self, stream_id: int):
        """
        Get the state of the stream
        """
        raise NotImplementedError

    @abstractmethod
    def delete_state(self, stream_id: int):
        """
        Delete the state of the stream
        """
        raise NotImplementedError

    @abstractmethod
    def create_state(self, options: ASRRequestOptions):
        """
        Create a new empty state
        """
        raise NotImplementedError

    @abstractmethod
    def init_state(self, stream_id: int, options: ASRRequestOptions):
        """
        Initialize the state of the stream
        """
        raise NotImplementedError

    @abstractmethod
    def transcribe_step(self, requests: list[Request]):
        """
        Transcribe a step
        """
        raise NotImplementedError
