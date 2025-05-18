#!/usr/bin/env python3
"""migrate_stage_flows.py – upgrade Stage-Flow YAMLs to use `agent_id`.

Usage::

    python dev/scripts/migrate_stage_flows.py path/to/flow.yaml  # prints diff
    python dev/scripts/migrate_stage_flows.py path/to/flows/ --in-place  # bulk migrate

The script walks files ending with .yml or .yaml, loads YAML, converts each
stage that uses legacy `agent` field to `agent_id`, removes `agent`, and writes
back (if --in-place).  Otherwise prints the would-be changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml


def _convert(data: Dict[str, Any]) -> bool:
    """Return True if mutation happened."""
    changed = False
    stages = data.get("stages", {})
    for stage in stages.values():
        if "agent" in stage and "agent_id" not in stage:
            stage["agent_id"] = stage.pop("agent")
            changed = True
    return changed


def _process_file(path: Path, in_place: bool) -> bool:
    orig_text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(orig_text)
    mutated = _convert(data)
    if not mutated:
        return False

    new_text = yaml.safe_dump(data, sort_keys=False)
    if in_place:
        path.write_text(new_text, encoding="utf-8")
        try:
            rel = path.relative_to(Path.cwd())
        except ValueError:
            rel = path
        print(f"[UPDATED] {rel}")
    else:
        print(f"--- {path} (original)\n+++ (migrated)")
        for line in _unified_diff(orig_text.splitlines(), new_text.splitlines()):
            print(line)
    return True


def _unified_diff(a: List[str], b: List[str]) -> List[str]:
    import difflib

    return list(difflib.unified_diff(a, b, lineterm=""))


def _iter_files(root: Path) -> List[Path]:
    if root.is_file():
        return [root]
    return [p for p in root.rglob("*.yml") if p.is_file()] + [p for p in root.rglob("*.yaml") if p.is_file()]


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Migrate Stage-Flow YAMLs to agent_id")
    parser.add_argument("path", type=str, help="Path to YAML file or directory")
    parser.add_argument("--in-place", action="store_true", help="Write changes back to file(s)")
    args = parser.parse_args()

    root = Path(args.path).expanduser().resolve()
    if not root.exists():
        print(f"Path not found: {root}", file=sys.stderr)
        sys.exit(1)

    mutated_any = False
    for file_path in _iter_files(root):
        mutated_any |= _process_file(file_path, args.in_place)

    if not mutated_any:
        print("No migration needed.")


if __name__ == "__main__":
    main() 