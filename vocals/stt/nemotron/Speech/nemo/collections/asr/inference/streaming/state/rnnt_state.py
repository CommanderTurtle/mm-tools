# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.asr.inference.streaming.state.state import StreamingState


class RNNTStreamingState(StreamingState):
    """
    State of the streaming RNNT pipeline
    """

    def __init__(self):
        """
        Initialize the RNNTStreamingState
        """
        super().__init__()
        self._additional_params_reset()

    def reset(self) -> None:
        """
        Reset the state
        """
        super().reset()
        self._additional_params_reset()

    def _additional_params_reset(self) -> None:
        """
        Reset non-inherited parameters
        """
        self.timestamp_offset = 0
        self.hyp_decoding_state = None
