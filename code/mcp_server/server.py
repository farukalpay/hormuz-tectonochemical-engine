from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = str(Path(__file__).resolve().parents[1] / "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from hte.mcp_app import main  # noqa: E402


if __name__ == "__main__":
    main()
