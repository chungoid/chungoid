"""A2A Reflection storage helper (Chroma-backed).

This module provides a thin wrapper around a Chroma collection that
stores agent-to-agent (A2A) reflections, enabling fast retrieval during
stage execution and for auditing/debugging tools.

Design goals
------------
* **Simple schema** – a single Pydantic model representing one
  reflection message.  Extra, un-structured keys go into the `extra`
  dict so we can evolve the schema without breakage.
* **Database-agnostic callers** – callers never see Chroma directly; they
  interact with `ReflectionStore` which exposes CRUD helpers.
* **Low coupling** – no implicit globals.  The caller passes the project
  root path *or* an explicit `chromadb.ClientAPI`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb is required for ReflectionStore") from exc

from .chroma_client_factory import get_client

__all__ = [
    "Reflection",
    "ReflectionStore",
]


class Reflection(BaseModel):
    """Represents one atomic agent reflection / message."""

    conversation_id: str = Field(..., description="Logical thread ID")
    message_id: str = Field(..., description="Primary key / UUID")
    parent_message_id: Optional[str] = Field(
        None, description="Parent msg id if this reflection is a reply"
    )
    agent_id: str = Field(..., description="Agent (tool) that produced the content")
    stage: Optional[str] = Field(None, description="Stage name, if applicable")
    content_type: str = Field(..., description="e.g. 'thought', 'result', 'error'")
    content: Any = Field(..., description="Raw JSON-serialisable payload")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp ISO format",
    )
    extra: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary additional metadata"
    )

    class Config:
        # Allow extra keys at top-level so callers can evolve gradually.
        extra = "allow"


class ReflectionStore:
    """High-level interface for reading/writing *Reflection*s in Chroma."""

    COLLECTION_NAME = "a2a_reflection"

    def __init__(
        self,
        *,
        project_root: Optional[Path] = None,
        chroma_client: Optional[ClientAPI] = None,
        chroma_mode: str = "persistent",
    ) -> None:
        if chroma_client is None:
            if project_root is None:
                raise ValueError("Either project_root or chroma_client must be supplied")
            chroma_client = get_client(chroma_mode, project_root)
        self._client: ClientAPI = chroma_client
        self._coll: Collection = self._client.get_or_create_collection(self.COLLECTION_NAME)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def add(self, reflection: Reflection) -> None:
        """Insert one reflection (overwrites on duplicate message_id)."""
        doc = reflection.json()
        # We use message_id as the Chroma id (unique).
        meta = reflection.dict(exclude={"content"})
        self._coll.add(documents=[reflection.content], ids=[reflection.message_id], metadatas=[meta])

    def add_many(self, reflections: Sequence[Reflection]) -> None:
        """Bulk-insert reflections."""
        if not reflections:
            return
        ids = [r.message_id for r in reflections]
        docs = [r.content for r in reflections]
        metas = [r.dict(exclude={"content"}) for r in reflections]
        self._coll.add(documents=docs, ids=ids, metadatas=metas)

    def get(self, message_id: str) -> Optional[Reflection]:
        """Return one reflection by primary key or *None* if absent."""
        try:
            res = self._coll.get(ids=[message_id])
        except Exception:
            return None
        if not res or not res.get("ids"):
            return None
        return self._reconstruct_single(res)

    def query(
        self,
        *,
        conversation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Reflection]:
        """Return reflections filtered by conversation_id and/or agent_id."""
        # Since Chroma's filtering is basic, we just fetch all & filter client-side.
        # For large corpora we could add more granular filters.
        all_meta = self._coll.peek(limit)  # returns dict with ids, metadatas, documents
        results: List[Reflection] = []
        for i, rid in enumerate(all_meta.get("ids", [])):
            meta = all_meta["metadatas"][i]
            if conversation_id and meta.get("conversation_id") != conversation_id:
                continue
            if agent_id and meta.get("agent_id") != agent_id:
                continue
            doc_text = all_meta["documents"][i]
            meta["content"] = doc_text
            results.append(Reflection.parse_obj(meta))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _reconstruct_single(self, chroma_payload: dict) -> Reflection:
        meta = chroma_payload["metadatas"][0]
        doc = chroma_payload["documents"][0]
        meta["content"] = doc
        return Reflection.parse_obj(meta) 