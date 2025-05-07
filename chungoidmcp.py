"""Deprecated launcher – kept for backward-compat.

Users should now run:

    chungoid-server <project_dir>

or

    python -m chungoid.mcp <project_dir>

This stub issues a warning and forwards execution to the new entry-point.
"""

import warnings
import sys


def _forward():
    warnings.warn(
        "`chungoidmcp.py` is deprecated; use `chungoid-server` instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    from importlib import import_module

    mcp = import_module("chungoid.mcp")
    # Pass along CLI arguments (excluding script path)
    mcp.main(sys.argv[1:])


if __name__ == "__main__":  # pragma: no cover
    _forward()
