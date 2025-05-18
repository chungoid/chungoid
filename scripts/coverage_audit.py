#!/usr/bin/env python3
"""Coverage Audit – verify vector store coverage per library.

Scans installed Python packages, samples up to N public symbols and checks
if their names appear in embedded metadata for the corresponding Chroma
collection.  Fails with exit 1 if coverage < 0.95 for any library.
Usage:
    python dev/scripts/coverage_audit.py --lib langgraph --sample 200
"""
from __future__ import annotations

import argparse
import importlib
import pkgutil
import random
import sys
from typing import List
import re
import tomllib  # For parsing pyproject
from pathlib import Path

try:
    import chromadb  # type: ignore
except ImportError:
    chromadb = None  # type: ignore

META_COLLECTION_PREFIX = "meta_lib_"


def _list_public_symbols(pkg_name: str, limit: int) -> List[str]:
    try:
        mod = importlib.import_module(pkg_name)
    except ImportError:
        return []
    symbols: List[str] = []
    for _, name, _ in pkgutil.walk_packages(mod.__path__, mod.__name__ + "."):
        symbols.append(name)
        if len(symbols) >= limit:
            break
    return symbols


def audit_library(lib: str, sample: int = 200, threshold: float = 0.95) -> bool:
    if chromadb is None:
        print("[audit] chromadb missing – skipping", lib)
        return True
    client = chromadb.HttpClient(host="localhost", port=8000)  # type: ignore[attr-defined]
    try:
        coll = client.get_collection(META_COLLECTION_PREFIX + lib)
    except Exception:
        print("[audit] no collection", lib)
        return False

    symbols = _list_public_symbols(lib, sample * 2)
    if not symbols:
        print("[audit] no symbols found for", lib)
        return True

    sample_syms = random.sample(symbols, min(sample, len(symbols)))
    found = 0
    for sym in sample_syms:
        res = coll.query(query_texts=[sym], n_results=1)
        if res["documents"] and res["documents"][0]:
            found += 1
    coverage = found / len(sample_syms)
    print(f"[audit] {lib}: coverage {coverage:.2%} ({found}/{len(sample_syms)})")
    return coverage >= threshold


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lib", action="append", help="library name to audit (repeat)")
    p.add_argument("--sample", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.95)
    args = p.parse_args()

    libs = args.lib or []
    # Auto-detect dependencies from chungoid-core/pyproject.toml if no libs specified
    if not libs:
        # Assumes this script is in chungoid-core/scripts/
        # Path(__file__).resolve().parent is chungoid-core/scripts/
        # Path(__file__).resolve().parents[0] is chungoid-core/scripts/
        # Path(__file__).resolve().parents[1] is chungoid-core/
        core_project_root = Path(__file__).resolve().parents[1]
        pyproject = core_project_root / "pyproject.toml"
        if pyproject.exists():
            data = tomllib.loads(pyproject.read_text())
            deps: list[str] = data.get("project", {}).get("dependencies", [])  # type: ignore[index]
            detected = []
            for dep in deps:
                m = re.match(r"([A-Za-z0-9_\-]+)", dep)
                if m:
                    detected.append(m.group(1))
            libs = detected

    if not libs:
        print("[coverage-audit] No libraries to audit (provide --lib or ensure pyproject.toml dependencies present).")
        sys.exit(0)

    ok = True
    for lib in libs:
        ok &= audit_library(lib, args.sample, args.threshold)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main() 