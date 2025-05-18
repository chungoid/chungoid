#!/usr/bin/env python3
"""Embed only the planning / thought files that changed in a commit range.

Used by CI to keep Chroma collections in sync without re-embedding the
entire corpus.  The script expects two commit SHAs via CLI args or env:
    --base <sha> --head <sha>
If omitted it defaults to ``$GITHUB_SHA^`` and ``$GITHUB_SHA`` so that
GitHub Actions can call it directly.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EMBED_META = ROOT / "dev" / "scripts" / "embed_meta.py"

PLANNING_PREFIX = "dev/planning/"
THOUGHT_PREFIX = "dev/thoughts/"


def _git_diff_names(base: str, head: str) -> list[str]:
    cmd = ["git", "diff", "--name-only", base, head]
    out = subprocess.check_output(cmd, text=True).strip()
    return [line for line in out.splitlines() if line]


def _embed(file_path: Path) -> None:
    if file_path.as_posix().startswith(PLANNING_PREFIX) and file_path.suffix == ".md":
        doc_type = "documentation"
    elif file_path.as_posix().startswith(THOUGHT_PREFIX) and file_path.suffix == ".yaml":
        doc_type = "thought"
    else:
        return  # ignore others

    subprocess.check_call([
        sys.executable,
        str(EMBED_META),
        "add",
        str(file_path),
        "-t",
        doc_type,
        "--collection",
        "meta_planning" if doc_type == "documentation" else "meta_thoughts",
    ])


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Embed changed planning/thought files between two commits")
    parser.add_argument("--base", default=os.environ.get("BASE_SHA"), help="Base commit SHA (older)")
    parser.add_argument("--head", default=os.environ.get("GITHUB_SHA"), help="Head commit SHA (newer)")
    args = parser.parse_args()

    if not args.base or not args.head:
        print("[embed-changed] Both --base and --head (or env) must be set", file=sys.stderr)
        sys.exit(1)

    changed = _git_diff_names(args.base, args.head)
    candidates = [Path(p) for p in changed if p.startswith((PLANNING_PREFIX, THOUGHT_PREFIX))]
    if not candidates:
        print("[embed-changed] No relevant changed files – nothing to embed")
        return

    print(f"[embed-changed] Embedding {len(candidates)} changed files …")
    for path in candidates:
        try:
            _embed(path)
        except subprocess.CalledProcessError as exc:
            print(f"[embed-changed] ERROR embedding {path}: {exc}", file=sys.stderr)
            sys.exit(exc.returncode)

    print("[embed-changed] Done.")


if __name__ == "__main__":
    main() 