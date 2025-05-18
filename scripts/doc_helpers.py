#!/usr/bin/env python3
"""Doc-Helpers – ergonomic access layer over Chroma library docs.

This is **v0.1** scaffolding that fulfils the public API described in the
Doc-Helpers blueprint.  Most heavy-lifting helpers are implemented with safe
fallbacks so unit-tests can already import the functions without a running
Chroma server.  Subsequent roadmap IDs (DH10-DH15) will add the smart search,
retrieval-gating, re-ranking and auto-patch logic.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path
from typing import List, Sequence
import os
from collections import defaultdict

# Optional deps – lazy import so that CI passes even if chromadb or transformers
# are absent in the environment.
try:
    import chromadb  # type: ignore
except ImportError:  # pragma: no cover – chroma optional in tests
    chromadb = None  # type: ignore

# Optional transformers for rerank
try:
    from sentence_transformers import CrossEncoder  # type: ignore
except ImportError:  # pragma: no cover
    CrossEncoder = None  # type: ignore

try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants & utility
# ---------------------------------------------------------------------------
# When in chungoid-core/scripts, parents[1] is chungoid-core, parents[2] is chungoid-mcp root.
# This configuration might need to be more flexible if chungoid-core is used standalone.
ROOT = Path(__file__).resolve().parents[2]
META_COLLECTION_PREFIX = "meta_lib_"
# TODO: Make ChromaDB path/connection configurable, potentially using chungoid.utils.chroma_client_factory
DEFAULT_CHROMA_HOST_PATH = ROOT / "dev_chroma_db" # Default for dev, might not exist for core-only use.

# ---------------------------------------------------------------------------
# Internal client factory (DH3)
# ---------------------------------------------------------------------------

def _ensure_client():
    """Return a Chroma client instance or raise RuntimeError if unavailable."""
    if chromadb is None:
        raise RuntimeError("chromadb not installed – Doc-Helpers requires it for runtime queries")
    
    # Attempt persistent client first, fallback to in-memory.
    # This path needs to be robust or configurable for chungoid-core deployment.
    host_path_str = DEFAULT_CHROMA_HOST_PATH.as_posix()
    try:
        # Check if the directory exists and is accessible for persistent client
        if DEFAULT_CHROMA_HOST_PATH.exists() and DEFAULT_CHROMA_HOST_PATH.is_dir():
            # print(f"[doc_helpers] Using PersistentClient with path: {host_path_str}") # Debug
            return chromadb.PersistentClient(path=host_path_str)
        else:
            # print(f"[doc_helpers] Path {host_path_str} for PersistentClient not found or not a dir. Falling back.") # Debug
            pass # Fall through to in-memory
    except Exception as e_persist:  # pylint: disable=broad-except
        # print(f"[doc_helpers] PersistentClient failed: {e_persist}. Falling back.") # Debug
        pass # Fall through to in-memory

    # print("[doc_helpers] Using in-memory Client as fallback.") # Debug
    return chromadb.Client()  # type: ignore[attr-defined]

# ---------------------------------------------------------------------------
# Helper: Reciprocal-Rank Fusion (simple)
# ---------------------------------------------------------------------------

def _rrf_merge(results_per_variant: List[List[dict]], k: int) -> List[dict]:
    """Merge ranked lists using RRF and return top-k docs (stable order)."""
    scores = defaultdict(float)
    doc_map: dict[str, dict] = {}
    for ranked in results_per_variant:
        for rank, item in enumerate(ranked):
            doc_id = item.get("id") or item.get("document")[:40]  # fallback hash
            scores[doc_id] += 1.0 / (60 + rank)  # typical RRF constant
            if doc_id not in doc_map:
                doc_map[doc_id] = item
    # sort by descending score
    merged = sorted(doc_map.items(), key=lambda kv: scores[kv[0]], reverse=True)
    return [doc_map[k] for k, _ in merged[:k]]

# ---------------------------------------------------------------------------
# Helper: Paraphrase generator (cheap)
# ---------------------------------------------------------------------------

def _paraphrase_variants(query: str, n: int = 3) -> List[str]:
    if openai and os.getenv("OPENAI_API_KEY"):
        try:
            # Ensure openai client is initialized if version >= 1.0
            client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai
            resp = client.chat.completions.create(
                model="gpt-3.5-turbo-1106",
                messages=[
                    {
                        "role": "system",
                        "content": f"Generate exactly {n} alternative phrasings for the following search query without changing its meaning. Return them as a JSON list of strings.",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.7,
                max_tokens=150,
                response_format={"type": "json_object"} # For newer OpenAI API versions
            )
            import json as _json
            # Accessing response content might differ based on OpenAI client version
            if hasattr(resp, 'choices') and resp.choices:
                message_content = resp.choices[0].message.content
                if message_content:
                    variants_data = _json.loads(message_content)
                    # Expecting structure like {"variants": ["var1", "var2", ...]}
                    # or just ["var1", "var2", ...]
                    if isinstance(variants_data, dict) and "variants" in variants_data and isinstance(variants_data["variants"], list):
                         variants = variants_data["variants"]
                    elif isinstance(variants_data, list):
                        variants = variants_data
                    else:
                        variants = [] # Unexpected JSON structure
                    
                    if isinstance(variants, list):
                        return [v.strip() for v in variants if isinstance(v, str)]
        except Exception as e:  # pragma: no cover
            # print(f"[doc_helpers] OpenAI paraphrase failed: {e}") # Debug
            pass
    # Fallback: simple heuristic variations
    return [query, query.replace(" ", "") + " example", "Guide to " + query]

# ---------------------------------------------------------------------------
# Smart search implementation (now DH12-15 features)
# ---------------------------------------------------------------------------

def _smart_search(lib: str | None, query: str, k: int = 3, *, mode: str = "balanced", expand: bool = False) -> List[dict]:
    # Retrieval gate
    if not should_retrieve(query):
        return []

    client = _ensure_client()

    # Determine candidate collections
    candidate_colls_typed: List[chromadb.api.models.Collection.Collection] = []
    if lib:
        try:
            candidate_colls_typed = [client.get_or_create_collection(META_COLLECTION_PREFIX + lib)]
        except Exception as e:
            # print(f"[doc_helpers] Error getting collection for lib '{lib}': {e}") # Debug
            return [] # If specific lib collection fails, return empty
    else:
        try:
            all_colls = client.list_collections()
            # Filter for actual collections if needed, or assume all are valid
            candidate_colls_typed = [c for c in all_colls if c.name.startswith(META_COLLECTION_PREFIX)]
        except Exception as e:
            # print(f"[doc_helpers] Error listing collections: {e}") # Debug
            return []

    if not lib and candidate_colls_typed: # Only do centroid if multiple collections considered
        scored: List[tuple[float, chromadb.api.models.Collection.Collection]] = []
        for coll_typed in candidate_colls_typed:
            try:
                res = coll_typed.query(query_texts=[query], n_results=1, where={"kind": "centroid"}, include=["distances"])
                dist = res["distances"][0][0] if res.get("distances") and res["distances"][0] else 1.0
                scored.append((dist, coll_typed))
            except Exception:
                scored.append((1.0, coll_typed))
        scored.sort(key=lambda t: t[0])
        if mode == "balanced":
            candidate_colls_typed = [c for _, c in scored[:5]]
        elif mode == "precision":
            candidate_colls_typed = [c for d, c in scored if d < 0.3][:5] or ([scored[0][1]] if scored else [])
        # elif mode == "recall": # default is all relevant candidate_colls_typed
        #     pass 

    # Build query variants
    variants = [query]
    if expand:
        variants.extend(_paraphrase_variants(query, n=3))

    per_variant_hits: List[List[dict]] = []
    for q_variant in variants:
        hits = []
        for coll_typed in candidate_colls_typed:
            try:
                out = coll_typed.query(query_texts=[q_variant], n_results=min(20, k * 4), include=["documents", "metadatas", "ids"])  # type: ignore[index]
                if out.get("documents") and out["documents"][0] and \
                   out.get("metadatas") and out["metadatas"][0] and \
                   out.get("ids") and out["ids"][0]:
                    for doc, meta, _id in zip(out["documents"][0], out["metadatas"][0], out["ids"][0]):
                        hits.append({"document": doc, **meta, "id": _id})
            except Exception:
                continue
        # naive length sort placeholder score
        hits.sort(key=lambda d: len(d.get("document", "")))
        per_variant_hits.append(hits[:k * 4])

    merged = _rrf_merge(per_variant_hits, k * 4)

    # Optional cross-encoder rerank
    if CrossEncoder is not None:
        try:
            model = CrossEncoder("BAAI/bge-reranker-base")
            pairs = [(query, h.get("document", "")) for h in merged]
            rerank_scores = model.predict(pairs)  # type: ignore[attr-defined]
            for h, s_rerank in zip(merged, rerank_scores):
                h["score"] = float(s_rerank)
            merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
        except Exception:  # pragma: no cover
            pass

    return merged[:k]

# ---------------------------------------------------------------------------
# Public helper APIs
# ---------------------------------------------------------------------------

def get_library_passage(lib: str | None, query: str, k: int = 3, *, mode: str = "balanced", expand: bool = False) -> List[str]:
    """Retrieve up-to-`k` passages that best match the query.

    Args:
        lib: library name (e.g. "langgraph") or None for any.
        query: natural-language search string.
        k: max number of passages to return.
        mode: balanced / recall / precision (DH12).
        expand: whether to use query expansion + RRF (DH15).
    """
    hits = _smart_search(lib, query, k=k, mode=mode, expand=expand)
    return [h.get("document", "") for h in hits]


def explain_symbol(symbol: str, *, lib_hint: str | None = None) -> str:
    """Return a plain-English explanation for `symbol` (class/function).

    Current implementation tries vector search, then falls back to live
    `inspect.getdoc`.  DH10 will add dynamic autopatch on miss.
    """
    # 1) vector search attempt
    passages = get_library_passage(lib_hint, symbol, k=1, mode="precision") # Use precision for direct symbol lookup
    if passages and passages[0]: # Ensure passage is not empty
        return passages[0]

    # 2) fallback to inspect
    if lib_hint:
        try:
            mod = importlib.import_module(lib_hint)
            # Safer way to access attributes if possible, or ensure symbol is well-formed
            obj_to_inspect = mod
            for part in symbol.split('.'):
                if hasattr(obj_to_inspect, part):
                    obj_to_inspect = getattr(obj_to_inspect, part)
                else:
                    raise AttributeError(f"Symbol part '{part}' not found in module/object")
            
            doc = inspect.getdoc(obj_to_inspect) or "(no docstring)"
            # DH10 dynamic autopatch – embed docstring back to collection
            try:
                client = _ensure_client()
                coll_name = META_COLLECTION_PREFIX + lib_hint
                coll = client.get_or_create_collection(coll_name)
                chunk_id = f"live_{lib_hint.replace('.', '_')}_{symbol.replace('.', '_')}"
                coll.add(documents=[doc], ids=[chunk_id], metadatas=[{"lib_name": lib_hint, "kind": "live_doc", "symbol": symbol}])
            except Exception: # pragma: no cover
                pass # Best effort autopatch
            return doc
        except Exception: # pragma: no cover
            pass # Fall through if inspect fails
    return f"No documentation found for '{symbol}'."


def lib_quickstart(lib: str) -> str:
    """Return pre-generated cheat-sheet chunk if present."""
    passages = get_library_passage(lib, f"{lib} quickstart", k=1, mode="precision")
    if passages and passages[0]:
        return passages[0]
    # try cheatsheet metadata only
    hits = _smart_search(lib, "cheatsheet", k=1, mode="precision", expand=False)
    return hits[0].get("document", "") if hits and hits[0].get("document") else "(quick-start not available)"


def list_library_symbols(lib: str, prefix: str | None = None) -> List[str]:
    """Enumerate public symbols in the installed package via pkgutil."""
    try:
        mod = importlib.import_module(lib)
    except ImportError:
        return []
    symbols: List[str] = []
    # Ensure mod.__path__ is valid and iterable
    mod_path = getattr(mod, '__path__', [])
    if not isinstance(mod_path, list):
        mod_path = list(mod_path)

    for _, name, _ in pkgutil.walk_packages(mod_path, mod.__name__ + "."):
        if prefix and not name.startswith(prefix):
            continue
        symbols.append(name)
    return symbols


def should_retrieve(query: str, model: str = "gpt-3.5-turbo-0125") -> bool:  # noqa: D401 (imperative mood)
    """DH14 retrieval gate. True if query seems doc-related."""
    if openai and os.getenv("OPENAI_API_KEY"):
        try:
            client = openai.OpenAI() if hasattr(openai, "OpenAI") else openai
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Does the following user query seem like a request for library documentation, code examples, or technical explanation? Respond YES or NO.",
                    },
                    {"role": "user", "content": query},
                ],
                temperature=0.1,
                max_tokens=5,
            )
            if hasattr(resp, 'choices') and resp.choices:
                answer = resp.choices[0].message.content
                return "yes" in answer.lower()
        except Exception:  # pragma: no cover
            pass  # default to retrieve on error
    return True  # default: retrieve


def _cli(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Query library docs via Chroma.")
    parser.add_argument("command", choices=["get", "explain", "symbols", "quickstart", "should_retrieve"])
    parser.add_argument("--lib", help="Library name (e.g. langgraph)")
    parser.add_argument("--query", help="Query string or symbol name")
    parser.add_argument("-k", type=int, default=3, help="Number of results for get")
    parser.add_argument("--mode", choices=["balanced", "recall", "precision"], default="balanced")
    parser.add_argument("--expand", action="store_true", help="Enable query expansion")

    args = parser.parse_args(argv)

    if args.command == "get":
        if not args.query:
            parser.error("--query is required for 'get'")
        passages = get_library_passage(args.lib, args.query, args.k, mode=args.mode, expand=args.expand)
        print(f"Found {len(passages)} passages:")
        for i, p in enumerate(passages):
            print(f"\n--- Passage {i+1} ({len(p.split())} words) ---")
            print(p)
    elif args.command == "explain":
        if not args.query:
            parser.error("--query (symbol name) is required for 'explain'")
        print(explain_symbol(args.query, lib_hint=args.lib))
    elif args.command == "symbols":
        if not args.lib:
            parser.error("--lib is required for 'symbols'")
        print(list_library_symbols(args.lib, prefix=args.query))
    elif args.command == "quickstart":
        if not args.lib:
            parser.error("--lib is required for 'quickstart'")
        print(lib_quickstart(args.lib))
    elif args.command == "should_retrieve":
        if not args.query:
            parser.error("--query is required for 'should_retrieve'")
        print(should_retrieve(args.query))


if __name__ == "__main__":
    _cli() 