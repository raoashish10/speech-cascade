"""Root conftest: makes `streaming_gateway` (a top-level package in this
repo, not under src/) importable from anywhere the suite is invoked from,
without requiring `pip install -e .` or a packaging change."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
