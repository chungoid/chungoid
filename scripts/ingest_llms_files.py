#!/usr/bin/env python
"""Convert raw library documentation text files into the canonical
`chungoid-core/offline_library_docs/<lib>/<version>/raw.txt` layout and then trigger
`chungoid-core/scripts/sync_library_docs.py --offline-prefetch` so the chunks get embedded.

Versioning: uses a specified version folder name (default: "latest").
"""
from __future__ import annotations

import shutil
from pathlib import Path
import subprocess
import sys
import argparse

try:
    import chromadb  # type: ignore
except ImportError:
    chromadb = None  # type: ignore

# When in chungoid-core/scripts, parents[1] is chungoid-core root.
CORE_ROOT = Path(__file__).resolve().parents[1]

# Default source for raw text files (e.g., project_root/manual_docs_input/)
# This should be a path provided by the user, not hardcoded to dev layer.
# For DEST_ROOT, we align with sync_library_docs.py in chungoid-core.
DEST_ROOT = CORE_ROOT / "offline_library_docs"
SYNC_SCRIPT = Path(__file__).resolve().parent / "sync_library_docs.py"


def main():
    parser = argparse.ArgumentParser(
        description="Prepare and embed local raw library documentation files for chungoid-core."
    )
    parser.add_argument(
        "--src",
        type=str,
        required=True,
        help="Path to a directory containing raw <library_name>_docs.txt files.",
    )
    parser.add_argument(
        "--version",
        default="latest",
        help="Version folder name to use inside offline_library_docs/<library_name>/ (default: latest)",
    )
    args = parser.parse_args()

    src_dir = Path(args.src).expanduser().resolve()

    if not src_dir.exists() or not src_dir.is_dir():
        print(f"[ingest-core-docs] Source directory not found or not a directory: {src_dir}", file=sys.stderr)
        sys.exit(1)
    
    if not SYNC_SCRIPT.exists():
        print(f"[ingest-core-docs] Prerequisite script not found: {SYNC_SCRIPT}", file=sys.stderr)
        sys.exit(1)

    moved_count = 0
    processed_libs = []
    for f_path in src_dir.iterdir():
        if not f_path.is_file():
            continue
        if not f_path.name.endswith("_docs.txt"):
            # Allow flexibility, could also be just <libname>.txt
            if not f_path.name.endswith(".txt"):
                continue
            lib_name = f_path.stem # e.g. "fastapi" from "fastapi.txt"
        else:
            lib_name = f_path.name.rsplit("_docs.txt", 1)[0]
        
        dest_path = DEST_ROOT / lib_name / args.version / "raw.txt"
        if dest_path.exists():
            print(f"[ingest-core-docs] Destination exists, skipping {lib_name} ({dest_path.relative_to(CORE_ROOT.parent)})")
            continue
        
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f_path, dest_path)
        moved_count += 1
        processed_libs.append(lib_name)
        print(f"[ingest-core-docs] Copied {f_path.name} → {dest_path.relative_to(CORE_ROOT.parent)}")

    if moved_count == 0:
        print(f"[ingest-core-docs] No suitable .txt or _docs.txt files found in {src_dir}.")
    else:
        print(f"[ingest-core-docs] {moved_count} file(s) copied. Running sync_library_docs.py --offline-prefetch ...")
        try:
            subprocess.check_call([sys.executable, str(SYNC_SCRIPT), "--offline-prefetch"])
        except subprocess.CalledProcessError as e:
            print(f"[ingest-core-docs] Error running sync_library_docs.py: {e}", file=sys.stderr)
            sys.exit(1)

        if chromadb is None:
            print("[ingest-core-docs] chromadb not installed; skipping verification.")
            return

        # Verification requires knowing the ChromaDB connection details used by sync_library_docs.py.
        # Assuming it uses a local HTTP client or a persistent store within the CORE_ROOT/.chungoid context.
        # For simplicity, this verification part might need to align with how chroma_client_factory works.
        print("[ingest-core-docs] Verification (Note: uses default HttpClient localhost:8000 or chungoid-core context):")
        try:
            # Attempt to use chungoid's own client factory if available for consistency
            from chungoid.utils.chroma_client_factory import get_client
            client = get_client(project_root=CORE_ROOT, mode="persistent") # Or http if that's the default for sync
        except ImportError:
            print("[ingest-core-docs] chungoid.utils.chroma_client_factory not found, using default HttpClient.")
            try:
                client = chromadb.HttpClient(host="localhost", port=8000)
            except Exception as e_http:
                print(f"[ingest-core-docs] Default HttpClient failed: {e_http}. Verification may be incomplete.")
                return
        except Exception as e_factory:
            print(f"[ingest-core-docs] Error using get_client: {e_factory}. Verification may be incomplete.")
            return

        for lib_name_verify in processed_libs:
            # Collection name is defined in sync_library_docs.py as META_COLLECTION_PREFIX + lib_name
            coll_name = f"meta_lib_{lib_name_verify}" 
            try:
                collection = client.get_collection(coll_name)
                print(f"[ingest-core-docs] {lib_name_verify}: {collection.count()} chunks embedded in {coll_name}.")
            except Exception:
                print(f"[ingest-core-docs] Collection {coll_name} not found or error during verification.")
                continue

if __name__ == "__main__":
    main() 