# SPDX-FileCopyrightText: Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.common.data.dataset import CodeSwitchedDataset, ConcatDataset, ConcatMapDataset
from nemo.collections.common.data.prompt_fn import apply_prompt_format_fn, get_prompt_format_fn
