# SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import json


def write_manifest_out_line(
    f_manifest_out,
    utt_obj,
):

    data = {"audio_filepath": utt_obj.audio_filepath}
    if not utt_obj.text is None:
        data["text"] = utt_obj.text

    if not utt_obj.pred_text is None:
        data["pred_text"] = utt_obj.pred_text

    for key, val in utt_obj.saved_output_files.items():
        data[key] = val

    new_line = json.dumps(data)
    f_manifest_out.write(f"{new_line}\n")

    return None
