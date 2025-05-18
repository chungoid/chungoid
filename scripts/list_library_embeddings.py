#!/usr/bin/env python
"""List all library documentation embeddings currently stored in Chroma.

Prints a neat table of `<lib> : <chunk_count>` for every collection whose name
matches the convention `meta_lib_<lib>`.

Usage:
    python dev/scripts/list_library_embeddings.py [--sort size|name] [--limit N]

Requires a running Chroma server reachable via CHROMA_HOST/CHROMA_PORT env vars
(or defaults localhost:8000).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from operator import itemgetter
from typing import List, Tuple

try:
    import chromadb  # type: ignore
except ImportError as exc:  # pragma: no cover
    sys.exit("chromadb package not found – pip install chromadb first")

COLL_PREFIX = "meta_lib_"


def list_library_collections() -> List[Tuple[str, int]]:
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )
    libs: List[Tuple[str, int]] = []
    for coll_meta in client.list_collections():  # type: ignore[attr-defined]
        # Recent chromadb returns Collection objects, older versions dictionaries
        name = getattr(coll_meta, "name", None)
        if name is None and isinstance(coll_meta, dict):
            name = coll_meta.get("name")
        if name is None:
            continue
        if not name.startswith(COLL_PREFIX):
            continue
        lib = name[len(COLL_PREFIX) :]
        coll = client.get_collection(name)
        count = coll.count()
        libs.append((lib, count))
    return libs


def main():
    parser = argparse.ArgumentParser(description="Show embedded doc chunk counts")
    parser.add_argument("--sort", choices=["size", "name"], default="name")
    parser.add_argument("--limit", type=int, help="Show only top N (after sorting)")
    args = parser.parse_args()

    rows = list_library_collections()

    if args.sort == "size":
        rows.sort(key=itemgetter(1), reverse=True)
    else:
        rows.sort(key=itemgetter(0))

    if args.limit is not None:
        rows = rows[: args.limit]

    width = max((len(lib) for lib, _ in rows), default=0)
    print("=" * (width + 20))
    for lib, cnt in rows:
        print(f"{lib.ljust(width)} : {cnt:5} chunks")
    print("=" * (width + 20))
    print(f"Total libraries: {len(rows)}")


if __name__ == "__main__":
    main() 