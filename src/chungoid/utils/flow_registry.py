"""Flow registry backed by Chroma collection `core_flow_registry`.

This is **Phase-5, Task P5.1** — minimal CRUD implementation and Pydantic
schema definition required by upcoming CLI, tests and CI gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
import json

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

        # Define fields to exclude from direct dump IF they are handled separately
        # or are not part of metadata (like yaml_text).
        # `tags` is a list and will be serialized to _tags_str.
        excluded_fields_for_dump = {"yaml_text", "tags"}

        meta_from_model = card.model_dump(exclude=excluded_fields_for_dump)

        final_chroma_meta = {}
        for key, value in meta_from_model.items():
            if value is None: # Exclude keys if value is None (e.g., description, owner)
                continue
            if isinstance(value, datetime): # Explicitly convert datetime to ISO string
                final_chroma_meta[key] = value.isoformat()
            else:
                # Assumes str, int, float, bool are Chroma-compatible
                final_chroma_meta[key] = value

        # Serialize 'tags' list to string
        # Store as empty string if card.tags is None (it's default_factory=list, so not None) or empty.
        final_chroma_meta["_tags_str"] = ",".join(card.tags or [])

        self._coll.add(ids=[card.flow_id], documents=[card.yaml_text], metadatas=[final_chroma_meta])

    def get(self, flow_id: str) -> Optional[FlowCard]:
        res = self._coll.get(ids=[flow_id])
        # Chroma may return empty lists if *flow_id* was not found or after deletion.
        if not res["documents"]:
            return None
        
        doc = res["documents"][0]
        retrieved_meta_from_chroma = res["metadatas"][0].copy()

        # Data to build FlowCard
        card_data = retrieved_meta_from_chroma # Start with what Chroma gave us
        card_data["yaml_text"] = doc # Add document back

        # Deserialize 'tags'
        tags_str = card_data.pop("_tags_str", "") # Default to empty string if key missing
        card_data["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
        
        # Pydantic will handle parsing ISO strings back to datetime for 'created' and 'updated'.
        # Optional fields like 'description', 'owner' will be None if they were excluded during add and thus missing.
        return FlowCard.model_validate(card_data)

    def list(self, limit: int = 100) -> List[FlowCard]:
        peek_results = self._coll.peek(limit=limit) # Pass limit
        cards: List[FlowCard] = []
        
        retrieved_ids = peek_results.get("ids", []) 
        retrieved_metadatas = peek_results.get("metadatas", [])
        retrieved_documents = peek_results.get("documents", [])

        for i in range(len(retrieved_ids)):
            current_chroma_meta = retrieved_metadatas[i].copy()
            
            card_data = current_chroma_meta
            card_data["yaml_text"] = retrieved_documents[i]

            tags_str = card_data.pop("_tags_str", "")
            card_data["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            
            cards.append(FlowCard.model_validate(card_data))
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