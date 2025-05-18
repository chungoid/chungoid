"""check_metrics_store – CI helper verifying metrics Chroma collection is readable.

Exit code 0 if query succeeds; non-zero if collection missing or corrupt.

Used by GitHub workflow `metrics-health.yml`.
"""
from __future__ import annotations

import sys
from pathlib import Path

from chungoid.utils.metrics_store import MetricsStore

# When this script is in chungoid-core/scripts, parents[2] will be chungoid-mcp root.
# This assumes the MetricsStore is at the chungoid-mcp project level.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    # Attempt to peek into collection – we don't care about content yet.
    # The chroma_mode selection based on "CI" in sys.argv seems specific to its original context.
    # For chungoid-core, this might need to be more explicit or configurable.
    store = MetricsStore(project_root=PROJECT_ROOT, chroma_mode="http" if "CI" in (sys.argv or []) else "persistent")
    try:
        _ = store.query(limit=1)
    except Exception as exc:
        print(f"[error] MetricsStore query failed: {exc}", file=sys.stderr)
        sys.exit(1)
    print("MetricsStore healthy ✔")


if __name__ == "__main__":
    main() 