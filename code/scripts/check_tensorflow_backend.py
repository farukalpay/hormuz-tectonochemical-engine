from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hte.backends import backend_payload  # noqa: E402


if __name__ == "__main__":
    print(json.dumps(backend_payload(), indent=2))
