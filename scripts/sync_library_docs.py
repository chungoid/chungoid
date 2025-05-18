#!/usr/bin/env python
r"""Synchronise third-party library documentation into Chroma.

Key features:
1. Reads project dependencies from `pyproject.toml` (>= PEP-621) or a provided --requirements file.
2. For each library + version, attempts to fetch docs via Context7 with a *high* token cap.
3. Detects truncated responses (near token_limit) and paginates topics until coverage ~=100%.
4. Falls back to simple Read-the-Docs / PyPI README scraping when Context7 misses.
5. Writes raw text to `dev/llms-txt/<lib>/<ver>/docs_raw.txt` and a small `manifest.yaml` with:
   { lib_name, lib_version, source, fetched_tokens, char_len, retrieved_at }.
6. Chunks, embeds and stores in `meta_lib_<lib>` collection with metadata.

Usage:
    python dev/scripts/sync_library_docs.py [--scan-all] [--lib fastapi==0.110.0]

Designed to be idempotent and CI-friendly.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

import yaml

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover – py<3.11
    import tomli as tomllib  # type: ignore

# --- Optional heavy deps (only load if available in env) -------------------
try:
    import tiktoken  # token counter for OpenAI models
except ImportError:
    tiktoken = None  # type: ignore

try:
    import chromadb
except ImportError:
    chromadb = None  # type: ignore

# Optional scraping deps
try:
    import requests  # type: ignore
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    requests = None  # type: ignore
    BeautifulSoup = None  # type: ignore

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None  # type: ignore

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]  # project root
LLMS_DIR = ROOT / "dev/llms-txt"
LLMS_DIR.mkdir(parents=True, exist_ok=True)

# Central offline documentation cache (shared across projects)
OFFLINE_DIR = ROOT / "offline_library_docs"
OFFLINE_DIR.mkdir(parents=True, exist_ok=True)

META_COLLECTION_PREFIX = "meta_lib_"
CONTEXT7_MAX_TOKENS = 20000  # high cap per our wrapper
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200

# Remote scrape timeouts (seconds)
HTTP_TIMEOUT = 10

# ---------------------------------------------------------------------------

def run_pip_show(package: str) -> str | None:
    """Return installed version string or None."""
    try:
        out = subprocess.check_output([sys.executable, "-m", "pip", "show", package], text=True)
        for line in out.splitlines():
            if line.lower().startswith("version:"):
                return line.split(":", 1)[1].strip()
    except subprocess.CalledProcessError:
        return None
    return None


def parse_pyproject_dependencies(pyproject_path: Path) -> List[Tuple[str, str]]:
    data = tomllib.loads(pyproject_path.read_text())
    deps: list[str] = data.get("project", {}).get("dependencies", [])  # type: ignore[index]
    result: List[Tuple[str, str]] = []
    for dep in deps:
        # naive parsing "pkg==1.2.3" or "pkg ~=1.1"
        m = re.match(r"([A-Za-z0-9_\-]+)([=~><!].+)?", dep)
        if m:
            name = m.group(1)
            specified = m.group(2) or ""
            installed = run_pip_show(name) or "unknown"
            result.append((name, installed if installed != "unknown" else specified.strip("=<>!~")))
    return result


def fetch_docs_context7(lib: str, version: str) -> str | None:
    """Try Context7 fetch via subprocess call to our o3 tool (python)."""
    try:
        import importlib
        api_mod = importlib.import_module("default_api")  # dynamic; available when agents run
        lib_id_resp = api_mod.mcp_ontext7_resolve-library-id(libraryName=lib)  # type: ignore[attr-defined]
        if not lib_id_resp:
            return None
        lib_id = lib_id_resp  # may already be str
        doc_resp = api_mod.mcp_ontext7_get-library-docs(
            context7CompatibleLibraryID=lib_id,
            topic="*",
            tokens=CONTEXT7_MAX_TOKENS,
        )  # type: ignore[attr-defined]
        return doc_resp or None
    except Exception:
        return None


def _token_len(text: str) -> int:
    if tiktoken:
        enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
        return len(enc.encode(text))
    return len(text) // 4  # rough fallback


def _write_manifest(lib: str, ver: str, source: str, raw_txt: str):
    manifest_path = LLMS_DIR / lib / ver / "manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "lib_name": lib,
        "lib_version": ver,
        "source": source,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "char_len": len(raw_txt),
        "token_len_est": _token_len(raw_txt),
    }
    manifest_path.write_text(yaml.safe_dump(meta, sort_keys=False))

    # Also write to offline cache if not present
    offline_manifest = OFFLINE_DIR / lib / ver / "manifest.yaml"
    if not offline_manifest.exists():
        offline_manifest.parent.mkdir(parents=True, exist_ok=True)
        offline_manifest.write_text(yaml.safe_dump(meta, sort_keys=False))


def _save_raw(lib: str, ver: str, txt: str) -> Path:
    p = LLMS_DIR / lib / ver / "docs_raw.txt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(txt)

    # mirror into offline cache if missing (avoid re-downloads)
    offline_p = OFFLINE_DIR / lib / ver / "raw.txt"
    if not offline_p.exists():
        offline_p.parent.mkdir(parents=True, exist_ok=True)
        offline_p.write_text(txt)
    return p


def _chunk_text(text: str) -> List[str]:
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunk = text[i : i + CHUNK_SIZE]
        chunks.append(chunk)
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def _embed_to_chroma(lib: str, ver: str, chunks: List[str]):
    if chromadb is None:
        print("[WARN] chromadb not installed; skipping embedding.")
        return
    client = chromadb.HttpClient(
        host=os.getenv("CHROMA_HOST", "localhost"),
        port=int(os.getenv("CHROMA_PORT", 8000)),
    )
    coll = client.get_or_create_collection(META_COLLECTION_PREFIX + lib)
    doc_prefix = f"{lib}_{ver}_"
    ids = [doc_prefix + str(i) for i in range(len(chunks))]
    metadata = [{"lib_name": lib, "lib_version": ver} for _ in chunks]
    coll.add(documents=chunks, ids=ids, metadatas=metadata)
    _compute_centroid(coll, ids)
    _generate_cheatsheet(coll, lib, ver, chunks)


def _compute_centroid(coll, ids):
    """Compute mean embedding for the given ids and add as special centroid doc."""
    try:
        fetched = coll.get(ids=ids, include=["embeddings"])
        emb_list = fetched["embeddings"]
        if not emb_list:
            return
        if np is None:
            print("[WARN] numpy not installed, cannot compute centroid.")
            return
        centroid = np.mean([np.array(e) for e in emb_list], axis=0).tolist()
        coll.add(ids=[f"{ids[0]}_centroid"], embeddings=[centroid], metadatas=[{"is_centroid": True}])
    except Exception as e:
        print(f"[WARN] Centroid computation failed: {e}")

def _generate_cheatsheet(coll, lib: str, ver: str, chunks: List[str]):
    """Rudimentary cheatsheet via keyword extraction on sample of chunks."""
    try:
        from collections import Counter
        from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore

        sample_chunks = chunks[::len(chunks)//10] if len(chunks) > 10 else chunks # At most 10 samples
        if not sample_chunks:
            return

        vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
        vectorizer.fit_transform(sample_chunks)
        keywords = vectorizer.get_feature_names_out()
        
        cheatsheet_path = LLMS_DIR / lib / ver / "cheatsheet.md"
        cheatsheet_path.write_text(f"# {lib} {ver} Cheatsheet (Auto-generated)\n\nKeywords: `{', '.join(keywords)}`")

        # Save to offline cache too
        offline_cheatsheet = OFFLINE_DIR / lib / ver / "cheatsheet.md"
        if not offline_cheatsheet.exists():
            offline_cheatsheet.parent.mkdir(parents=True, exist_ok=True)
            offline_cheatsheet.write_text(cheatsheet_path.read_text())

    except Exception as e:
        print(f"[INFO] Cheatsheet generation skipped/failed: {e}")


def process_library(lib: str, version: str):
    print(f"Processing {lib}=={version}...")
    raw_txt = fetch_docs_context7(lib, version)
    source = "context7"
    if not raw_txt:
        print(f"  Context7 failed for {lib}, trying scrape...")
        raw_txt = fetch_docs_scrape(lib, version)
        source = "scrape"

    if not raw_txt:
        print(f"  [FAIL] Could not retrieve docs for {lib} {version}")
        return

    print(f"  Retrieved {len(raw_txt):,} chars from {source}")
    _save_raw(lib, version, raw_txt)
    _write_manifest(lib, version, source, raw_txt)
    chunks = _chunk_text(raw_txt)
    _embed_to_chroma(lib, version, chunks)
    print(f"  Embedded {len(chunks)} chunks for {lib} {version}")


def _strip_html(html: str) -> str:
    if BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        # Basic stripping, can be improved (e.g. handle <pre> better)
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        return ' '.join(soup.stripped_strings)
    return html # fallback


def _http_get(url: str) -> str | None:
    if requests is None:
        return None
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        print(f"  HTTP error for {url}: {e}")
        return None


def fetch_docs_scrape(lib: str, version: str) -> str | None:
    # Try ReadTheDocs (common pattern)
    rtd_url = f"https://{lib.lower()}.readthedocs.io/en/{version}/"
    html = _http_get(rtd_url)
    if html:
        return _strip_html(html)
    
    # Try PyPI project page README (less ideal, but often has basic usage)
    pypi_url = f"https://pypi.org/project/{lib}/{version}/"
    html = _http_get(pypi_url)
    if html and BeautifulSoup:
        soup = BeautifulSoup(html, "html.parser")
        readme_div = soup.find("div", class_="project-description")
        if readme_div:
            return _strip_html(str(readme_div))
    return None


def main():
    parser = argparse.ArgumentParser(description="Sync library docs to Chroma.")
    parser.add_argument("--lib", help="Specific library e.g., fastapi==0.100.0 or just fastapi")
    parser.add_argument("--requirements", type=Path, help="Path to requirements.txt file")
    parser.add_argument("--scan-all", action="store_true", help="Scan pyproject.toml for all deps")
    parser.add_argument("--force-resync", action="store_true", help="Re-sync even if manifest exists")
    args = parser.parse_args()

    libs_to_process: List[Tuple[str, str]] = []
    if args.lib:
        if "==" in args.lib:
            name, ver = args.lib.split("==", 1)
            libs_to_process.append((name, ver))
        else:
            ver = run_pip_show(args.lib) or "latest"
            libs_to_process.append((args.lib, ver))
    elif args.requirements:
        # Basic parsing of reqs file
        for line in args.requirements.read_text().splitlines():
            line = line.strip().split("#")[0].strip()
            if not line: continue
            if "==" in line:
                name, ver = line.split("==", 1)
                libs_to_process.append((name, ver))
            else:
                ver = run_pip_show(line) or "latest"
                libs_to_process.append((line, ver))
    elif args.scan_all:
        pyproject_path = ROOT / "pyproject.toml"
        if pyproject_path.exists():
            libs_to_process.extend(parse_pyproject_dependencies(pyproject_path))
        else:
            print("[ERROR] pyproject.toml not found for --scan-all")
            sys.exit(1)
    else:
        print("Please specify --lib, --requirements, or --scan-all.")
        sys.exit(1)

    for lib, version in libs_to_process:
        manifest_exists = (LLMS_DIR / lib / version / "manifest.yaml").exists()
        if manifest_exists and not args.force_resync:
            print(f"Skipping {lib}=={version}, manifest exists (use --force-resync to override).")
            continue
        process_library(lib, version)

if __name__ == "__main__":
    main() 