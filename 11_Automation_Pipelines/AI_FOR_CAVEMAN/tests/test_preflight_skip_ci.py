from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("CI", "").lower() == "true", reason="Manual-only preflight test")
def test_preflight_script_exists_only() -> None:
    root = Path(__file__).resolve().parents[1]
    script = root / "tools" / "preflight_qwen_check.py"
    assert script.exists()
