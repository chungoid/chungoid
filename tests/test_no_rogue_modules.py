import os
from pathlib import Path


ALLOWED_TOP_LEVEL_PY_FILES = {
    "chungoidmcp.py",  # Back-compat CLI stub
}


def test_no_rogue_python_modules():
    """Fail if stray *.py files live at repo root (outside src/)."""
    repo_root = Path(__file__).resolve().parent.parent  # <repo>/chungoid-core

    for py_file in repo_root.glob("*.py"):
        if py_file.name in ALLOWED_TOP_LEVEL_PY_FILES:
            continue
        # If we reach here we have an unexpected python file
        raise AssertionError(
            f"Unexpected top-level python module detected: {py_file.name}. "
            "Move it under src/chungoid/ or delete the duplicate."
        ) 