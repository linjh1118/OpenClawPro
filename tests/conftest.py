"""
Pytest configuration for OpenClawPro tests.

Adds the repo root to sys.path so that both `nanobot` and `harness`
modules can be imported.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add OpenClawPro repo root to sys.path so nanobot and harness are importable
_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
