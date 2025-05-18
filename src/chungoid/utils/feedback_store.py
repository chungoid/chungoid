"""Process feedback storage helper (Chroma-backed).

Captures subjective/affective feedback left by agents or humans while
using the system.  Unlike *Reflection* (which stores structured
conversation content), *ProcessFeedback* records how smooth or painful a
step felt so that maintainers can discover recurring friction points.

Design mirrors `ReflectionStore` so callers can swap easily.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence
import json

from pydantic import BaseModel, Field

try:
    import chromadb
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb is required for FeedbackStore") from exc

from .chroma_client_factory import get_client

__all__ = [
    "ProcessFeedback",
    "FeedbackStore",
]


class ProcessFeedback(BaseModel):
    """One qualitative feedback entry about meta-layer usage."""

    conversation_id: str = Field(..., description="Logical thread ID (optional if not chat-based)")
    agent_id: str = Field(..., description="Agent or user who logged the feedback")
    stage: Optional[str] = Field(None, description="Stage / component in focus")
    sentiment: str = Field(
        ..., description="Quick emoji or short token: '👍', '😐', '👎', or free-form tag"
    )
    comment: str = Field(..., description="Free-text explanation of experience")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra fields")

    class Config:
        extra = "allow"


class FeedbackStore:
    """High-level interface for CRUD on *ProcessFeedback* in Chroma."""

    COLLECTION_NAME = "a2a_process_feedback"

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def add(self, fb: ProcessFeedback) -> None:
        meta = fb.model_dump(exclude={"comment"})
        # Convert datetime and dict to string for Chroma compatibility
        if isinstance(meta.get("timestamp"), datetime):
            meta["timestamp"] = meta["timestamp"].isoformat()
        if isinstance(meta.get("extra"), dict):
            meta["extra"] = json.dumps(meta["extra"])
            
        self._coll.add(documents=[fb.comment], ids=[self._make_id(fb)], metadatas=[meta])

    def add_many(self, fbs: Sequence[ProcessFeedback]) -> None:
        if not fbs:
            return
        ids = [self._make_id(f) for f in fbs]
        docs = [f.comment for f in fbs]
        metas_raw = [f.model_dump(exclude={"comment"}) for f in fbs]
        
        metas_processed = []
        for meta_item in metas_raw:
            # Convert datetime and dict to string for Chroma compatibility
            if isinstance(meta_item.get("timestamp"), datetime):
                meta_item["timestamp"] = meta_item["timestamp"].isoformat()
            if isinstance(meta_item.get("extra"), dict):
                meta_item["extra"] = json.dumps(meta_item["extra"])
            metas_processed.append(meta_item)
            
        self._coll.add(documents=docs, ids=ids, metadatas=metas_processed)

    def query(
        self,
        *,
        conversation_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        stage: Optional[str] = None,
        limit: int = 100,
    ) -> List[ProcessFeedback]:
        all_meta = self._coll.peek(limit)
        results: List[ProcessFeedback] = []
        for i, _ in enumerate(all_meta.get("ids", [])):
            meta = all_meta["metadatas"][i]
            if conversation_id and meta.get("conversation_id") != conversation_id:
                continue
            if agent_id and meta.get("agent_id") != agent_id:
                continue
            if stage and meta.get("stage") != stage:
                continue
            comment = all_meta["documents"][i]
            meta["comment"] = comment
            results.append(ProcessFeedback.model_validate(meta))
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_id(self, fb: ProcessFeedback) -> str:
        """Stable-ish unique id (timestamp—agent combo)"""
        ts = int(fb.timestamp.timestamp() * 1000)
        return f"{fb.agent_id}.{ts}" 