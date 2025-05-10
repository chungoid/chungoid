"""Flow registry backed by Chroma collection `core_flow_registry`.

This is **Phase-5, Task P5.1** — minimal CRUD implementation and Pydantic
schema definition required by upcoming CLI, tests and CI gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from .chroma_client_factory import get_client

try:
    import chromadb  # noqa: F401 — imported for side-effects / typing
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb required for FlowRegistry") from exc


class FlowCard(BaseModel):
    """Metadata + YAML document for a Stage-Flow definition."""

    flow_id: str = Field(..., description="Unique slug/uuid identifying this flow")
    name: str = Field(..., description="Human-readable title")
    yaml_text: str = Field(..., description="Raw YAML document — single source of truth")
    description: Optional[str] = None
    version: str = Field("0.1", description="Semantic version string")
    owner: Optional[str] = Field(None, description="Maintainer or team")
    tags: List[str] = Field(default_factory=list)
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Convenience: extract stage names directly from YAML (lazy) --------
    @property
    def stage_names(self) -> List[str]:
        try:
            import yaml

            data = yaml.safe_load(self.yaml_text)
            return list(data.get("stages", {}).keys()) if isinstance(data, dict) else []
        except Exception:  # pragma: no cover – defensive
            return []


class FlowRegistry:
    """CRUD wrapper for Chroma collection holding *FlowCard*s."""

    COLLECTION = "core_flow_registry"

    def __init__(self, *, project_root: Path, chroma_mode: str = "persistent") -> None:
        self._client: ClientAPI = get_client(chroma_mode, project_root)
        self._coll: Collection = self._client.get_or_create_collection(self.COLLECTION)

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------
    def add(self, card: FlowCard, *, overwrite: bool = False) -> None:
        if self._exists(card.flow_id):
            if not overwrite:
                raise ValueError(f"Flow {card.flow_id} already exists")
            # If overwriting, remove first to avoid duplicate ids error
            self.remove(card.flow_id)
        meta = card.model_dump(exclude={"yaml_text"})
        self._coll.add(ids=[card.flow_id], documents=[card.yaml_text], metadatas=[meta])

    def get(self, flow_id: str) -> Optional[FlowCard]:
        res = self._coll.get(ids=[flow_id])
        if not res["ids"]:
            return None
        doc = res["documents"][0]
        meta = res["metadatas"][0]
        meta["yaml_text"] = doc
        return FlowCard.model_validate(meta)

    def list(self, limit: int = 100) -> List[FlowCard]:
        peek = self._coll.peek(limit)
        cards: List[FlowCard] = []
        for i, fid in enumerate(peek.get("ids", [])):
            meta = peek["metadatas"][i]
            meta["yaml_text"] = peek["documents"][i]
            cards.append(FlowCard.model_validate(meta))
        return cards

    def remove(self, flow_id: str) -> None:
        if hasattr(self._coll, "delete"):
            # Newer Chroma clients implement delete()
            self._coll.delete(ids=[flow_id])  # type: ignore[attr-defined]
            return

        # Fallback for in-memory stub used in tests – simply ignore if not present.
        try:
            # Some in-memory stubs expose internal `_docs` mapping
            if hasattr(self._coll, "_docs") and flow_id in self._coll._docs:  # type: ignore[attr-defined]
                del self._coll._docs[flow_id]  # type: ignore[attr-defined]
        except Exception:  # pragma: no cover
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _exists(self, flow_id: str) -> bool:
        res = self._coll.get(ids=[flow_id])
        # Chroma returns empty lists when ID is unknown. Our in-memory stub previously
        # returned the *requested* ids irrespective of existence, so rely on the
        # presence of a corresponding document to decide.
        return bool(res["documents"]) 