# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from nemo.collections.tts.g2p.models.en_us_arpabet import EnglishG2p
from nemo.collections.tts.g2p.models.i18n_ipa import IpaG2p as IPAG2P
from nemo.collections.tts.g2p.models.zh_cn_pinyin import ChineseG2p

__all__ = ['ChineseG2p', 'EnglishG2p', 'IPAG2P']

# TODO @xueyang: This file is kept for backward-compatibility purposes since all older NGC models that were trained on
#  and before NeMo 1.16.0 used this import path. We will remove this file soon; `IPAG2P` will be also renamed as
#  `IpaG2p`. Please start using new import path and the new `IpaG2p` name from NeMo 1.16.0.
