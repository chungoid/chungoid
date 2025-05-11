from __future__ import annotations

"""Agent registry backed by Chroma collection `a2a_agent_registry`."""

from typing import List, Optional
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
        
        excluded_fields_for_dump = {"description", "capabilities", "tool_names", "metadata"}
        # Dump the model. Datetimes will be included here as datetime objects by default
        # if mode='python' (default for model_dump).
        meta_from_model = card.model_dump(exclude=excluded_fields_for_dump) 

        final_chroma_meta = {}
        # Process fields from the model dump, ensuring correct types for ChromaDB
        for key, value in meta_from_model.items():
            if value is None:
                continue # Exclude keys with None values (Strategy from Thought 2)
            
            if isinstance(value, datetime):
                # Explicitly convert datetime to ISO format string
                final_chroma_meta[key] = value.isoformat()
            else:
                # This assumes other types (str, int, float, bool) are already Chroma-compatible
                # or will be handled by subsequent specific serialization if they were complex types
                # not caught by initial exclusion (though they should be excluded).
                final_chroma_meta[key] = value

        # Serialize list fields (Strategy from Thought 3)
        # Ensure these are always present as strings (empty if original is None/empty)
        final_chroma_meta["_capabilities_str"] = ",".join(card.capabilities or [])
        final_chroma_meta["_tool_names_str"] = ",".join(card.tool_names or [])
        
        # Serialize the AgentCard's 'metadata' dict field to a JSON string
        # Ensure it's always present as a string (empty JSON object if original is None/empty)
        final_chroma_meta["_agent_card_metadata_json"] = json.dumps(card.metadata or {})

        self._coll.add(ids=[card.agent_id], documents=[card.description or ""], metadatas=[final_chroma_meta])

    def get(self, agent_id: str) -> Optional[AgentCard]:
        res = self._coll.get(ids=[agent_id])
        if not res["ids"]:
            return None
        
        retrieved_meta_from_chroma = res["metadatas"][0].copy() 
        
        # Start building the dictionary for AgentCard.model_validate
        # Pydantic will handle parsing ISO string back to datetime for 'created' field
        card_data = retrieved_meta_from_chroma.copy() 
        card_data["description"] = res["documents"][0] # Add document field

        # Deserialize list fields
        # Pop the key to avoid passing it to model_validate if it's not part of the model
        capabilities_str = card_data.pop("_capabilities_str", "") # Default to empty string if key missing
        card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]
            
        tool_names_str = card_data.pop("_tool_names_str", "") # Default to empty string
        card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

        # Deserialize the AgentCard's 'metadata' field from JSON string
        metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}") # Default to empty JSON object string
        card_data["metadata"] = json.loads(metadata_json_str)
        
        # Fields like 'agent_id', 'name', 'stage_focus', 'created' (as ISO string)
        # are expected to be directly in card_data and will be validated by Pydantic.
        # Pydantic automatically converts ISO strings to datetime objects for datetime fields.

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
            
            # Pydantic will handle 'created' (from ISO string in card_data) and other direct fields.
            cards.append(AgentCard.model_validate(card_data))
        return cards

    # Helpers -----------------------------------------------------------
    def _exists(self, agent_id: str) -> bool:
        return bool(self._coll.get(ids=[agent_id])["ids"]) 