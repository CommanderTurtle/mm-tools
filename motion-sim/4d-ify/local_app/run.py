from __future__ import annotations

import json
import logging
import sys
from pathlib import Path


class Progress(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        fraction = float(getattr(record, "fraction", 0.0))
        print(f"MM_PROGRESS {fraction:.6f} {record.getMessage()}", flush=True)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: run.py REQUEST.json")
    request_path = Path(sys.argv[1]).resolve()
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    from fdanyone.pipeline import PROGRESS
    from inference import inference

    handler = Progress()
    PROGRESS.addHandler(handler)
    PROGRESS.propagate = False
    try:
        result = inference(**payload)
    finally:
        PROGRESS.removeHandler(handler)
    print("MM_RESULT " + json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
