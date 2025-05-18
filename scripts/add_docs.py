#!/usr/bin/env python
"""Helper CLI to register manually-provided offline docs.

Usage:
    python dev/scripts/add_docs.py --lib fastapi==0.115.9

Assumes you have already placed a plain-text documentation dump at:
    offline_library_docs/<lib>/<version>/raw.txt

The script simply forwards the call to `sync_library_docs.py` so that the
manifest is written, a project-local copy is made under `dev/llms-txt/`,
and the text is embedded into Chroma.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Assuming this script is in chungoid-core/scripts/
# Path(__file__).resolve().parent is chungoid-core/scripts/
# Path(__file__).resolve().parents[0] is chungoid-core/scripts/
# Path(__file__).resolve().parents[1] is chungoid-core/
# Path(__file__).resolve().parents[2] is the workspace root (e.g., chungoid-mcp/)
ROOT = Path(__file__).resolve().parents[2] 
SCRIPT = Path(__file__).resolve().parent / "sync_library_docs.py"


def main():
    parser = argparse.ArgumentParser(description="Register offline docs for a library")
    parser.add_argument("--lib", required=True, help="lib==version to ingest (must match directory name)")
    args = parser.parse_args()

    if "==" not in args.lib:
        parser.error("--lib must be in the form 'name==version'")

    cmd = [sys.executable, str(SCRIPT), "--lib", args.lib]
    print("[add-docs] Delegating to sync_library_docs.py →", " ".join(cmd))
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main() 