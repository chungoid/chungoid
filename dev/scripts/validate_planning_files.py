#!/usr/bin/env python3
"""CI helper: validate planning markdown files.

Copied from meta repo so core unit tests can load it locally.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import yaml

ROOT = Path(__file__).resolve().parents[2]
PLANNING_DIR = ROOT / "dev" / "planning"

REQUIRED_KEYS = {"title", "category", "owner", "created", "status"}

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---", re.DOTALL)


def load_front_matter(md_text: str) -> Dict:
    match = FRONT_MATTER_RE.match(md_text)
    if not match:
        raise ValueError("Missing YAML front matter (--- block)")
    try:
        return yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML front matter: {exc}")


def validate_file(path: Path) -> List[str]:
    if path.name.lower() == "readme.md":
        return []
    errors: List[str] = []
    text = path.read_text(encoding="utf-8", errors="ignore")
    try:
        fm = load_front_matter(text)
    except ValueError as e:
        errors.append(f"{path.name}: {e}")
        return errors
    missing = REQUIRED_KEYS - fm.keys()
    if missing:
        errors.append(f"{path.name}: missing keys {sorted(missing)}")
    return errors


def check_pairs(files: List[Path]) -> List[str]:
    errors: List[str] = []
    prefixes: Dict[str, Set[str]] = {}
    for p in files:
        if p.name.count("_") < 1:
            continue
        prefix, suffix = p.name.rsplit("_", 1)
        kind = None
        if suffix.startswith("roadmap"):
            kind = "roadmap"
        elif suffix.startswith("blueprint"):
            kind = "blueprint"
        if kind:
            prefixes.setdefault(prefix, set()).add(kind)
    for pref, kinds in prefixes.items():
        if kinds != {"roadmap", "blueprint"}:
            missing = {"roadmap", "blueprint"} - kinds
            errors.append(f"Planning pair missing for '{pref}': {', '.join(sorted(missing))}")
    return errors


def main() -> None:
    md_files = list(PLANNING_DIR.glob("*.md"))
    all_errors: List[str] = []
    for f in md_files:
        all_errors.extend(validate_file(f))
    all_errors.extend(check_pairs(md_files))

    if all_errors:
        print("Planning validation failed:\n", file=sys.stderr)
        for e in all_errors:
            print(f" - {e}", file=sys.stderr)
        sys.exit(1)
    else:
        print("Planning docs validation passed.")


if __name__ == "__main__":
    main() 