import json
import sys
from pathlib import Path

from paths import NATIVE, parent_guard


def main():
    parent_guard()
    sys.path.insert(0, str(NATIVE))
    from apps.camera_video_overlay import burn_shared_overlay_video

    job = json.load(sys.stdin)
    burn_shared_overlay_video(
        Path(job["source"]), Path(job["destination"]), job["overlay"]
    )


if __name__ == "__main__":
    main()
