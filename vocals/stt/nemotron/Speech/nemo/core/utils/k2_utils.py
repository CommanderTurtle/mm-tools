# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.


K2_INSTALLATION_MESSAGE = (
    "Could not import `k2`.\n"
    "Please install k2 in one of the following ways:\n"
    "1) (recommended) Run `bash scripts/installers/install_k2.sh`\n"
    "2) Use any approach from https://k2-fsa.github.io/k2/installation/index.html "
    "if your your cuda and pytorch versions are supported.\n"
    "It is advised to always install k2 using setup.sh only, "
    "as different versions of k2 may not interact with the NeMo code as expected."
)
