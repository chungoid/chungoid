"""Runtime metrics storage helper (Chroma-backed).

Stores *MetricEvent*s emitted by the execution runtime so that agents and
humans can query performance and reliability information.

Design goals
------------
* **Light-weight** – write-once documents; no embeddings required.
* **Schema-evolvable** – extra keys allowed so we can extend metrics later.
* **Drop-in** – mirrors `ReflectionStore` API for familiarity.
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
    raise ImportError("chromadb is required for MetricsStore") from exc

from .chroma_client_factory import get_client

__all__ = ["MetricEvent", "MetricsStore"]


class MetricEvent(BaseModel):
    """Represents a single stage/run metric sample."""

    run_id: str = Field(..., description="UUID for the overall flow run")
    stage_id: str = Field(..., description="Identifier of the stage")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="UTC timestamp when the metric was recorded",
    )
    duration_ms: int = Field(..., description="Stage runtime in milliseconds")
    status: str = Field(..., description="success | error | skipped")
    error_message: Optional[str] = Field(None, description="Error summary if status == error")
    tags: dict[str, Any] = Field(default_factory=dict, description="Arbitrary extra dimensions")

    class Config:
        extra = "allow"


class MetricsStore:
    """High-level API for persisting and querying `MetricEvent`s."""

    COLLECTION_NAME = "a2a_metrics"

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
    # Write helpers
    # ------------------------------------------------------------------
    def add(self, event: MetricEvent) -> None:
        """Add one metric event (upsert by composite id)."""
        event_id = f"{event.run_id}:{event.stage_id}:{int(event.timestamp.timestamp()*1000)}"
        meta = event.dict()
        # Store an empty document (no embedding); Chroma requires a doc string.
        self._coll.add(documents=[""], ids=[event_id], metadatas=[meta])

    def add_many(self, events: Sequence[MetricEvent]) -> None:
        if not events:
            return
        ids = [f"{e.run_id}:{e.stage_id}:{int(e.timestamp.timestamp()*1000)}" for e in events]
        metas = [e.dict() for e in events]
        self._coll.add(documents=["" for _ in events], ids=ids, metadatas=metas)

    # ------------------------------------------------------------------
    # Query helpers (basic metadata filters only)
    # ------------------------------------------------------------------
    def query(
        self,
        *,
        run_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[MetricEvent]:
        """Return metric events filtered by simple metadata fields.

        This is **not** full-text search – we rely only on stored metadata.
        For large datasets callers should page via `since` + `limit`.
        """
        payload = self._coll.peek(limit)
        results: List[MetricEvent] = []
        for i, _ in enumerate(payload.get("ids", [])):
            meta = payload["metadatas"][i]
            if run_id and meta.get("run_id") != run_id:
                continue
            if stage_id and meta.get("stage_id") != stage_id:
                continue
            if status and meta.get("status") != status:
                continue
            if since:
                ts: datetime = meta.get("timestamp")  # type: ignore[assignment]
                if ts < since:
                    continue
            results.append(MetricEvent.parse_obj(meta))
        # sort ascending by timestamp for convenience
        results.sort(key=lambda e: e.timestamp)
        return results 