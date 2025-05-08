"""Very small in-memory stub mimicking Chroma Client & Collection.

Only implements the subset of API used by FeedbackStore & ReflectionStore:
• get_or_create_collection
• Collection.add, .count, .peek, .get

Not thread-safe and not intended for production – **tests only**.
"""
from __future__ import annotations

from typing import Dict, List, Any

class _MemoryCollection:
    def __init__(self, name: str):
        self.name = name
        self._docs: Dict[str, str] = {}
        self._metas: Dict[str, Any] = {}

    # Thin wrappers -------------------------------------------------
    def add(self, *, documents: List[str], ids: List[str], metadatas: List[dict]):  # noqa: D401
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            self._docs[doc_id] = doc
            self._metas[doc_id] = meta

    def count(self) -> int:  # noqa: D401
        return len(self._docs)

    def peek(self, limit: int = 100) -> dict:  # noqa: D401
        ids = list(self._docs.keys())[:limit]
        return {
            "ids": ids,
            "documents": [self._docs[i] for i in ids],
            "metadatas": [self._metas[i] for i in ids],
        }

    def get(self, ids: List[str]):  # noqa: D401
        docs = [self._docs.get(i) for i in ids if i in self._docs]
        metas = [self._metas.get(i) for i in ids if i in self._metas]
        return {"ids": ids, "documents": docs, "metadatas": metas}


class MemoryClient:
    """Singleton-ish client that holds collections in a dict."""

    def __init__(self):
        self._colls: Dict[str, _MemoryCollection] = {}

    def get_or_create_collection(self, name: str):  # noqa: D401
        if name not in self._colls:
            self._colls[name] = _MemoryCollection(name)
        return self._colls[name] 