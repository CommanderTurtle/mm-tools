# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
#
#
# Unless required by applicable law or agreed to in writing, software
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
"""Write a copy of a NeMo manifest with the `context` (prompt) field set.

The chosen prompt comes from the dev prompt sweep; writing it into the manifest
makes it the training prompt *and* the evaluation prompt (vllm_task_eval.py reads
the same field). `--template` formats per-row fields, e.g. '{context} Reply with
the label only.' or 'Passage: {passage}\\n\\nQuestion...'.

Usage: set_context.py --template "<fmt>" <in.json> <out.json> [<in.json> <out.json> ...]
"""
import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--template", required=True)
p.add_argument("pairs", nargs="+")
a = p.parse_args()
tpl = a.template.replace("\\n", "\n")
assert len(a.pairs) % 2 == 0
for src, dst in zip(a.pairs[::2], a.pairs[1::2]):
    assert dst.endswith(".json") and src != dst
    n = 0
    with open(src) as fi, open(dst, "w") as fo:
        for line in fi:
            r = json.loads(line)
            r["context"] = tpl.format(**r)
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    with open(dst) as f:
        print(f"{dst}: {n} rows; context={json.loads(f.readline())['context'][:100]!r}")
