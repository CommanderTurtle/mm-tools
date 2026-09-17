from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    request_path = Path(sys.argv[1]).resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    project = Path(payload["project_root"])
    output = Path(payload["output"])
    command = [
        sys.executable,
        str(project / "generate.py"),
        "--task", "i2v-1.3B",
        "--infer_mode", "causal_fast",
        "--ckpt_dir", payload["checkpoint"],
        "--assets_dir", payload["assets"],
        "--image", payload["image"],
        "--action_path", payload["action_path"],
        "--prompt", payload["prompt"],
        "--size", payload["size"],
        "--frame_num", str(payload["frame_num"]),
        "--chunk_size", str(payload["chunk_size"]),
        "--base_seed", str(payload["seed"]),
        "--sample_shift", str(payload["sample_shift"]),
        "--local_attn_size", str(payload["local_attn_size"]),
        "--sink_size", str(payload["sink_size"]),
        "--ulysses_size", "1",
        "--offload_model", "false",
        "--save_file", str(output),
    ]
    if payload.get("max_attention_size"):
        command.extend(["--max_attention_size", str(payload["max_attention_size"])])
    if payload.get("convert_model_dtype"):
        command.append("--convert_model_dtype")
    print("MM_PROGRESS 0.06 Loading the 1.3B causal world model", flush=True)
    process = subprocess.Popen(command, cwd=project, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
        if "Generating video" in line:
            print("MM_PROGRESS 0.18 Building the causal world timeline", flush=True)
        elif "infer chunk" in line:
            print("MM_PROGRESS 0.52 Extending the cached world", flush=True)
        elif "Saving generated video" in line:
            print("MM_PROGRESS 0.94 Encoding the local preview", flush=True)
    code = process.wait()
    if code:
        raise SystemExit(code)
    if not output.is_file():
        raise RuntimeError(f"LingBot did not create {output}")
    print("MM_PROGRESS 0.99 World capture complete", flush=True)

if __name__ == "__main__":
    main()
