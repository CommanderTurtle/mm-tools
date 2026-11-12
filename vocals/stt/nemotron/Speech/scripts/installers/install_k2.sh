#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

K2_REPO=https://github.com/k2-fsa/k2
LATEST_RELEASE=0a09f67 # fix for PyTorch 2.6.0
# uncomment the following line after the next k2 version is released (>1.24.4)
#LATEST_RELEASE=$(git -c 'versionsort.suffix=-' \
#    ls-remote --exit-code --refs --sort='version:refname' --tags ${K2_REPO} '*.*' \
#    | tail --lines=1 \
#    | cut -d '/' -f 3)
# "cut --delimiter '/' --fields 3" doesn't work on macOS, use "-d ... -f ..." instead

pip install wheel setuptools cmake
K2_MAKE_ARGS="-j" pip install -v "git+${K2_REPO}@${LATEST_RELEASE}#egg=k2" || { echo "k2 could not be installed!"; exit 1; }
python3 -m k2.version > /dev/null || { echo "k2 installed with errors! Please check installation manually."; exit 1; }
echo "k2 installed successfully!"
