#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#

set -ex
sudo chown -R "$USER" output
python publish.py
