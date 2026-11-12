#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

echo "Install latest AIS CLI"
AIS_CLI_URL=https://github.com/NVIDIA/aistore/releases/latest/download/ais-linux-amd64.tar.gz

echo "Download AIS CLI from ${AIS_CLI_URL}"
curl -LO ${AIS_CLI_URL}

echo "Extract"
tar -xzvf ais-linux-amd64.tar.gz

echo "Move to /usr/local/bin/"
mv ./ais /usr/local/bin/. 

echo "Cleanup"
rm ais-linux-amd64.tar.gz
