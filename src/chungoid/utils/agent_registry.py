from __future__ import annotations

"""Agent registry backed by Chroma collection `a2a_agent_registry`."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
import json

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
    mcp_tool_input_schemas: Optional[Dict[str, Any]] = Field(None, description="Summarized input schemas for exposed MCP tools.")
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
        
        # Use the helper method to prepare metadata
        final_chroma_meta = self._agent_card_to_chroma_metadata(card)

        self._coll.add(ids=[card.agent_id], documents=[card.description or ""], metadatas=[final_chroma_meta])

    def get(self, agent_id: str) -> Optional[AgentCard]:
        res = self._coll.get(ids=[agent_id])
        if not res["ids"]:
            return None
        
        retrieved_meta_from_chroma = res["metadatas"][0].copy() 
        
        card_data = retrieved_meta_from_chroma.copy() 
        card_data["description"] = res["documents"][0]

        capabilities_str = card_data.pop("_capabilities_str", "")
        card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]
            
        tool_names_str = card_data.pop("_tool_names_str", "")
        card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

        metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}")
        card_data["metadata"] = json.loads(metadata_json_str)
        
        # Deserialize the new mcp_tool_input_schemas field
        # Default to "null" so json.loads results in None if key is missing
        mcp_schemas_json_str = card_data.pop("_mcp_tool_input_schemas_json", "null") 
        card_data["mcp_tool_input_schemas"] = json.loads(mcp_schemas_json_str)

        return AgentCard.model_validate(card_data)

    def list(self, limit: int = 100) -> List[AgentCard]:
        peek_results = self._coll.peek(limit=limit)
        cards: List[AgentCard] = []
        
        retrieved_ids = peek_results.get("ids", [])
        retrieved_metadatas = peek_results.get("metadatas", [])
        retrieved_documents = peek_results.get("documents", [])

        for i in range(len(retrieved_ids)):
            current_chroma_meta = retrieved_metadatas[i].copy()
            
            card_data = current_chroma_meta.copy()
            card_data["description"] = retrieved_documents[i]

            capabilities_str = card_data.pop("_capabilities_str", "")
            card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]

            tool_names_str = card_data.pop("_tool_names_str", "")
            card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

            metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}")
            card_data["metadata"] = json.loads(metadata_json_str)
            
            # Deserialize the new mcp_tool_input_schemas field
            # Default to "null" so json.loads results in None if key is missing
            mcp_schemas_json_str = card_data.pop("_mcp_tool_input_schemas_json", "null")
            card_data["mcp_tool_input_schemas"] = json.loads(mcp_schemas_json_str)

            cards.append(AgentCard.model_validate(card_data))
        return cards

    # Helpers -----------------------------------------------------------
    def _exists(self, agent_id: str) -> bool:
        return bool(self._coll.get(ids=[agent_id])["ids"]) 

    def _agent_card_to_chroma_metadata(self, agent_card: AgentCard) -> Dict[str, Any]:
        """Converts AgentCard fields to a dictionary suitable for ChromaDB metadata."""
        metadata = {
            "agent_id": agent_card.agent_id,
            "name": agent_card.name,
            "stage_focus": agent_card.stage_focus if agent_card.stage_focus is not None else "",
            # Store datetime as ISO 8601 string (ChromaDB compatibility)
            "created": agent_card.created.isoformat(),
            # Store lists as comma-separated strings (simple approach)
            "_capabilities_str": ",".join(agent_card.capabilities or []),
            "_tool_names_str": ",".join(agent_card.tool_names or []),
            # Store complex metadata dictionary as a JSON string
            "_agent_card_metadata_json": json.dumps(agent_card.metadata or {}),
            # Store tool input schemas as JSON string, ONLY IF PRESENT
        }
        # Only add schemas if they exist and are not None/empty
        if agent_card.mcp_tool_input_schemas:
            metadata["_mcp_tool_input_schemas_json"] = json.dumps(agent_card.mcp_tool_input_schemas)
        
        return metadata 