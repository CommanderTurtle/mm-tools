# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""
Implemented following the guide at https://www.promptingguide.ai/models/phi-2#phi-2-usage
"""

from nemo.collections.common.prompts.formatter import Modality, PromptFormatter


class ExamplePromptFormatter(PromptFormatter):
    """
    The simplest possible prompt formatter implementation.

    It defines a dialog of the form:

        User: Hi.
        Assistant: Hi, how can I help you?
        User: What's the time?
        Assistant: It's 9 o'clock.

    """

    NAME = "example_prompt_format"
    OUTPUT_ROLE = "assistant"
    TEMPLATE = {
        "user": {
            "template": "User: |message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
        OUTPUT_ROLE: {
            "template": "Assistant: |message|\n",
            "slots": {
                "message": Modality.Text,
            },
        },
    }
