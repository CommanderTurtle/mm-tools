"""Import Idol's released model code without modifying it (see IDOL_ROOT)."""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDOL_ROOT = os.environ.get("IDOL_ROOT", os.path.join(REPO_ROOT, "third_party", "idol-sdk"))
if not os.path.isdir(os.path.join(IDOL_ROOT, "idol")):
    raise ImportError(f"Idol checkout not found at {IDOL_ROOT}; set IDOL_ROOT or run "
                      f"`git clone https://github.com/speridlabs/idol-sdk third_party/idol-sdk && "
                      f"git -C third_party/idol-sdk checkout 956439f`")
if IDOL_ROOT not in sys.path:
    sys.path.insert(0, IDOL_ROOT)

from idol.models.dit import LatentSeqMMFlowModel          # noqa: E402
from idol.models import layers as idol_layers             # noqa: E402
