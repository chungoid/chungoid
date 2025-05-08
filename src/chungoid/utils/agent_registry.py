from __future__ import annotations

"""Agent registry backed by Chroma collection `a2a_agent_registry`."""

from typing import List, Optional
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field

from .chroma_client_factory import get_client

try:
    import chromadb
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb required for AgentRegistry") from exc


class AgentCard(BaseModel):
    agent_id: str = Field(..., description="Unique slug/uuid")
    name: str
    description: Optional[str] = None
    stage_focus: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    tool_names: List[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRegistry:
    COLLECTION = "a2a_agent_registry"

    def __init__(self, *, project_root: Path, chroma_mode: str = "persistent"):
        self._client: ClientAPI = get_client(chroma_mode, project_root)
        self._coll: Collection = self._client.get_or_create_collection(self.COLLECTION)

    # CRUD -------------------------------------------------------------
    def add(self, card: AgentCard, *, overwrite: bool = False):
        if self._exists(card.agent_id):
            if not overwrite:
                raise ValueError(f"Agent {card.agent_id} already exists")
        meta = card.model_dump(exclude={"description"})
        self._coll.add(ids=[card.agent_id], documents=[card.description or ""], metadatas=[meta])

    def get(self, agent_id: str) -> Optional[AgentCard]:
        res = self._coll.get(ids=[agent_id])
        if not res["ids"]:
            return None
        doc = res["documents"][0]
        meta = res["metadatas"][0]
        meta["description"] = doc
        return AgentCard.model_validate(meta)

    def list(self, limit: int = 100) -> List[AgentCard]:
        peek = self._coll.peek(limit)
        cards: List[AgentCard] = []
        for i, aid in enumerate(peek.get("ids", [])):
            meta = peek["metadatas"][i]
            meta["description"] = peek["documents"][i]
            cards.append(AgentCard.model_validate(meta))
        return cards

    # Helpers -----------------------------------------------------------
    def _exists(self, agent_id: str) -> bool:
        return bool(self._coll.get(ids=[agent_id])["ids"]) 