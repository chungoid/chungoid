from __future__ import annotations

"""Agent registry backed by Chroma collection `a2a_agent_registry`."""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
import json
import logging

from .chroma_client_factory import get_client
from .agent_registry_meta import AgentCategory, AgentVisibility

try:
    import chromadb
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb required for AgentRegistry") from exc


class AgentCard(BaseModel):
    agent_id: str = Field(..., description="Unique slug/uuid")
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    categories: Optional[List[str]] = Field(default_factory=list, description="Functional categories the agent belongs to (e.g., ['CodeGeneration', 'Python']).")
    capability_profile: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Key-value pairs describing specific capabilities (e.g., {'language': 'python', 'version_semver': '3.9.1', 'framework': 'typer'}).")
    priority: Optional[int] = Field(default=0, description="Default priority for selection within a category (higher wins).")
    visibility: Optional[AgentVisibility] = None
    stage_focus: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    tool_names: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for the agent\'s direct input if it\'s a callable.")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for the agent\'s direct output if it\'s a callable.")
    mcp_tool_input_schemas: Optional[Dict[str, Any]] = Field(None, description="Summarized input schemas for MCP tools this agent EXPOSES.")
    correlation_id: Optional[str] = Field(None, description="Identifier to correlate related agent invocations or tasks. Can be inherited or generated.")
    metadata: dict = Field(default_factory=dict)
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentRegistry:
    COLLECTION = "a2a_agent_registry"

    def __init__(self, *, project_root: Path, chroma_mode: str = "persistent"):
        self._client: ClientAPI = get_client(chroma_mode, project_root)
        self._coll: Collection = self._client.get_or_create_collection(self.COLLECTION)

    def _create_searchable_document_for_agent_card(self, card: AgentCard) -> str:
        """Constructs a single string from AgentCard fields for semantic search."""
        doc_parts = []
        if card.name:
            doc_parts.append(f"Agent Name: {card.name}")
        if card.description:
            doc_parts.append(f"Description: {card.description}")
        
        if card.categories:
            doc_parts.append(f"Categories: {', '.join(card.categories)}")
        if card.capability_profile:
            profile_summary = json.dumps(card.capability_profile)
            doc_parts.append(f"Capability Profile: {profile_summary}")
        if card.priority is not None:
            doc_parts.append(f"Priority: {card.priority}")

        if card.capabilities:
            doc_parts.append(f"Capabilities: {', '.join(card.capabilities)}")
        if card.tags:
            doc_parts.append(f"Tags: {', '.join(card.tags)}")
        if card.tool_names: # These are tools the agent *uses* or has affinity with
            doc_parts.append(f"Relevant Tools: {', '.join(card.tool_names)}")

        # Summarize direct input/output schemas of the agent itself
        if card.input_schema:
            input_summary = card.input_schema.get('title', card.input_schema.get('description', 'No summary'))
            doc_parts.append(f"Agent Input: {input_summary}")
        if card.output_schema:
            output_summary = card.output_schema.get('title', card.output_schema.get('description', 'No summary'))
            doc_parts.append(f"Agent Output: {output_summary}")

        # Summarize schemas of MCP tools the agent *exposes* (if any)
        if card.mcp_tool_input_schemas:
            for tool_name, schema in card.mcp_tool_input_schemas.items():
                if isinstance(schema, dict):
                    schema_summary = schema.get('title', schema.get('description', 'No summary'))
                    doc_parts.append(f"Exposed MCP Tool '{tool_name}' Input: {schema_summary}")
        
        return "\n".join(doc_parts)

    # CRUD -------------------------------------------------------------
    def add(self, card: AgentCard, *, overwrite: bool = False):
        if self._exists(card.agent_id):
            if not overwrite:
                raise ValueError(f"Agent {card.agent_id} already exists")
        
        logger = logging.getLogger(__name__)

        # Attempt to re-validate/finalize card if it's CoreTestGeneratorAgent_v1
        # This was a workaround for a Pydantic issue where model_dump() was losing fields.
        # Keeping this specific workaround for now as it proved necessary.
        if card.agent_id == "CoreTestGeneratorAgent_v1":
            try:
                card_dict = card.model_dump() 
                card = AgentCard.model_validate(card_dict)
            except Exception as e_reval:
                logger.warning(f"AGENT_REGISTRY_ADD: Warning during CoreTestGeneratorAgent_v1 re-validation: {e_reval}")

        # Standard INFO log for specific agent if needed (example)
        if card.agent_id == "CoreCodeGeneratorAgent_v1":
            logger.info(f"AgentRegistry.add: Received card CoreCodeGeneratorAgent_v1. Categories: {card.categories}, Profile: {card.capability_profile}")

        final_chroma_meta = self._agent_card_to_chroma_metadata(card)
        searchable_doc = self._create_searchable_document_for_agent_card(card)

        if card.agent_id == "CoreCodeGeneratorAgent_v1": # Example of a standard INFO log
            logger.info(f"AgentRegistry.add: Storing Chroma metadata for CoreCodeGeneratorAgent_v1: categories_str='{final_chroma_meta.get('categories_str')}', capability_profile_json='{final_chroma_meta.get('capability_profile_json')}'")

        self._coll.add(ids=[card.agent_id], documents=[searchable_doc], metadatas=[final_chroma_meta])

    def get(self, agent_id: str) -> Optional[AgentCard]:
        res = self._coll.get(ids=[agent_id])
        
        if not res["ids"]:
            # AGENT NOT FOUND - TRY REFRESHING COLLECTION AND RETRYING ONCE
            logger.warning(f"AgentRegistry.get: Agent ID '{agent_id}' not found on first try. Attempting collection refresh.")
            try:
                self._coll = self._client.get_or_create_collection(self.COLLECTION) # Re-fetch collection
                res = self._coll.get(ids=[agent_id]) # Retry the get
                if not res["ids"]:
                    logger.warning(f"AgentRegistry.get: Agent ID '{agent_id}' still not found after collection refresh.")
                    return None # Still not found
                else:
                    logger.info(f"AgentRegistry.get: Agent ID '{agent_id}' found after collection refresh.")
            except Exception as e:
                logger.error(f"AgentRegistry.get: Error during collection refresh attempt for '{agent_id}': {e}")
                return None # Error during refresh, treat as not found

        if not res["ids"]:
            return None # Should be redundant if above logic is correct, but as a safeguard
        
        retrieved_meta_from_chroma = res["metadatas"][0].copy()
        # The document is now the full searchable_doc. The original description is part of it.
        # For AgentCard reconstruction, we rely on metadata for most fields.
        # If a direct 'description' field is needed on AgentCard separate from the searchable doc, 
        # it should primarily come from metadata if stored there, or be reconstructed carefully.
        # Currently, AgentCard.description is populated from metadata if _agent_card_to_chroma_metadata includes it.
        # Let's ensure _agent_card_to_chroma_metadata stores the original description.
        
        card_data = retrieved_meta_from_chroma.copy()

        # Deserialize category and visibility from string to enum
        # Convert empty strings back to None before validation
        if "visibility" in card_data and card_data["visibility"] == "":
            card_data["visibility"] = None
        elif "visibility" in card_data and card_data["visibility"] is not None: # Ensure it's not already None
            try:
                card_data["visibility"] = AgentVisibility(card_data["visibility"])
            except ValueError:
                # Let Pydantic handle it if it's a non-empty, invalid string
                pass
        
        categories_str = card_data.pop("categories_str", "")
        card_data["categories"] = [cat.strip() for cat in categories_str.split(",") if cat.strip()]

        capability_profile_json_str = card_data.pop("capability_profile_json", "{}")
        card_data["capability_profile"] = json.loads(capability_profile_json_str)
        
        # Ensure 'priority' is present and a valid number, defaulting to 0 if missing or None.
        if "priority" not in card_data or card_data["priority"] is None:
            card_data["priority"] = 0

        capabilities_str = card_data.pop("_capabilities_str", "")
        card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]
        
        tags_str = card_data.pop("_tags_str", "") # Added for tags
        card_data["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            
        tool_names_str = card_data.pop("_tool_names_str", "")
        card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

        metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}")
        card_data["metadata"] = json.loads(metadata_json_str)
        
        input_schema_json_str = card_data.pop("_input_schema_json", "null")
        card_data["input_schema"] = json.loads(input_schema_json_str)

        output_schema_json_str = card_data.pop("_output_schema_json", "null")
        card_data["output_schema"] = json.loads(output_schema_json_str)
        
        mcp_schemas_json_str = card_data.pop("_mcp_tool_input_schemas_json", "null") 
        card_data["mcp_tool_input_schemas"] = json.loads(mcp_schemas_json_str)

        # correlation_id will be directly in card_data if it was stored, no special deserialization needed.
        # If description is not directly in metadata (because it was part of the searchable doc),
        # we might need to extract it or accept that AgentCard.description will be None
        # if not reconstructed from the searchable doc (which is complex).
        # For now, AgentCard construction will use whatever `description` is in `card_data` (from metadata).

        return AgentCard.model_validate(card_data)

    def list(self, limit: int = 100) -> List[AgentCard]:
        peek_results = self._coll.peek(limit=limit)
        cards: List[AgentCard] = []
        
        logger = logging.getLogger(__name__) 

        retrieved_ids_list = peek_results.get("ids", [])
        retrieved_metadatas_list = peek_results.get("metadatas", [])

        for i in range(len(retrieved_ids_list)):
            current_chroma_meta = retrieved_metadatas_list[i].copy()
            current_agent_id_from_peek = retrieved_ids_list[i]

            # Example of a standard conditional INFO log during list, if necessary for a specific agent.
            # if current_agent_id_from_peek == "SomeOtherAgent_v1":
            #    logger.info(f"Processing {current_agent_id_from_peek} in list, raw meta: {current_chroma_meta}")

            card_data = current_chroma_meta.copy()

            if "visibility" in card_data and card_data["visibility"] == "":
                card_data["visibility"] = None
            elif "visibility" in card_data and card_data["visibility"] is not None: 
                try:
                    card_data["visibility"] = AgentVisibility(card_data["visibility"])
                except ValueError:
                    pass

            categories_str = card_data.pop("categories_str", "") 
            card_data["categories"] = [cat.strip() for cat in categories_str.split(",") if cat.strip()]
            
            capability_profile_json_str = card_data.pop("capability_profile_json", "{}")
            card_data["capability_profile"] = json.loads(capability_profile_json_str)
            
            if "priority" not in card_data or card_data["priority"] is None:
                card_data["priority"] = 0

            capabilities_str = card_data.pop("_capabilities_str", "")
            card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]

            tags_str = card_data.pop("_tags_str", "") 
            card_data["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]
            
            tool_names_str = card_data.pop("_tool_names_str", "")
            card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

            metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}")
            card_data["metadata"] = json.loads(metadata_json_str)
            
            input_schema_json_str = card_data.pop("_input_schema_json", "null")
            card_data["input_schema"] = json.loads(input_schema_json_str)

            output_schema_json_str = card_data.pop("_output_schema_json", "null")
            card_data["output_schema"] = json.loads(output_schema_json_str)
            
            mcp_schemas_json_str = card_data.pop("_mcp_tool_input_schemas_json", "null")
            card_data["mcp_tool_input_schemas"] = json.loads(mcp_schemas_json_str)
            
            validated_card = AgentCard.model_validate(card_data)
            
            # Append a deep copy to isolate the object from potential external modifications
            cards.append(validated_card.model_copy(deep=True))
        return cards

    def search_agents(self, query_text: str, n_results: int = 3, where_filter: Optional[Dict[str, Any]] = None) -> List[AgentCard]:
        """Performs semantic search for agents based on query_text."""
        cards: List[AgentCard] = []
        if not query_text:
            return cards

        try:
            query_results = self._coll.query(
                query_texts=[query_text.strip()], 
                n_results=n_results, 
                where=where_filter, 
                include=["metadatas"] # Only metadatas needed for reconstruction
            )
        except Exception as e:
            # Log error appropriately, e.g., self.logger.error(...) if logger is available
            print(f"Error during ChromaDB query in search_agents: {e}") # Basic print for now
            return cards

        # query_results structure for a single query_text: {"ids": [[id1, id2]], "metadatas": [[meta1, meta2]], ...}
        if query_results and query_results.get("ids") and query_results["ids"][0]:
            retrieved_metadatas_list = query_results.get("metadatas", [[]])[0]
            
            for i in range(len(retrieved_metadatas_list)):
                current_chroma_meta = retrieved_metadatas_list[i]
                if not isinstance(current_chroma_meta, dict):
                    # Log warning or skip if metadata is not as expected
                    print(f"Warning: Expected dict for metadata, got {type(current_chroma_meta)}. Skipping.")
                    continue
                
                card_data = current_chroma_meta.copy()

                # Deserialize fields from metadata
                categories_str = card_data.pop("categories_str", "")
                card_data["categories"] = [cat.strip() for cat in categories_str.split(",") if cat.strip()]

                capability_profile_json_str = card_data.pop("capability_profile_json", "{}")
                card_data["capability_profile"] = json.loads(capability_profile_json_str)
                
                # Ensure 'priority' is present and a valid number, defaulting to 0 if missing or None.
                if "priority" not in card_data or card_data["priority"] is None:
                    card_data["priority"] = 0

                capabilities_str = card_data.pop("_capabilities_str", "")
                card_data["capabilities"] = [cap.strip() for cap in capabilities_str.split(",") if cap.strip()]

                tags_str = card_data.pop("_tags_str", "")
                card_data["tags"] = [tag.strip() for tag in tags_str.split(",") if tag.strip()]

                tool_names_str = card_data.pop("_tool_names_str", "")
                card_data["tool_names"] = [name.strip() for name in tool_names_str.split(",") if name.strip()]

                metadata_json_str = card_data.pop("_agent_card_metadata_json", "{}")
                try:
                    card_data["metadata"] = json.loads(metadata_json_str)
                except json.JSONDecodeError:
                    card_data["metadata"] = {} # Default on error
                
                input_schema_json_str = card_data.pop("_input_schema_json", "null")
                try:
                    card_data["input_schema"] = json.loads(input_schema_json_str)
                except json.JSONDecodeError:
                    card_data["input_schema"] = None # Default on error

                output_schema_json_str = card_data.pop("_output_schema_json", "null")
                try:
                    card_data["output_schema"] = json.loads(output_schema_json_str)
                except json.JSONDecodeError:
                    card_data["output_schema"] = None # Default on error
                
                mcp_schemas_json_str = card_data.pop("_mcp_tool_input_schemas_json", "null")
                try:
                    card_data["mcp_tool_input_schemas"] = json.loads(mcp_schemas_json_str)
                except json.JSONDecodeError:
                    card_data["mcp_tool_input_schemas"] = None # Default on error
                
                # The 'description' field should already be in card_data directly from metadata
                # Ensure 'agent_id' and 'name' are present as they are mandatory for AgentCard
                if not card_data.get("agent_id") or not card_data.get("name"):
                    # Log warning or skip
                    print(f"Warning: Missing agent_id or name in metadata for card reconstruction. Skipping. Data: {card_data}")
                    continue

                # correlation_id will be directly in card_data if stored.
                try:
                    cards.append(AgentCard.model_validate(card_data))
                except Exception as model_val_err: # Catch Pydantic ValidationError or other issues
                    # Log error with card_data for debugging
                    print(f"Error validating AgentCard from metadata: {model_val_err}. Data: {card_data}")
        return cards

    # Helpers -----------------------------------------------------------
    def _exists(self, agent_id: str) -> bool:
        return bool(self._coll.get(ids=[agent_id])["ids"]) 

    def _agent_card_to_chroma_metadata(self, agent_card: AgentCard) -> Dict[str, Any]:
        """Converts AgentCard to a flat dictionary suitable for ChromaDB metadata.
        Complex types are serialized to strings. Optional simple types that are None
        have their keys excluded.
        """
        # Logger for this method if specific logging is ever needed here.
        # logger = logging.getLogger(__name__) 

        meta: Dict[str, Any] = {
            "agent_id": agent_card.agent_id,
            "name": agent_card.name,
        }
        if agent_card.version is not None:
            meta["version"] = agent_card.version
        if agent_card.description is not None:
            meta["description"] = agent_card.description
        
        # Restore original logic for categories and capability_profile
        if agent_card.categories: 
            meta["categories_str"] = ",".join(agent_card.categories)
        else:
            meta["categories_str"] = ""

        if agent_card.capability_profile: 
            meta["capability_profile_json"] = json.dumps(agent_card.capability_profile)
        else:
            meta["capability_profile_json"] = "{}"
        
        if agent_card.priority is not None:
            meta["priority"] = agent_card.priority

        # Handle existing optional fields
        if agent_card.visibility is not None:
            meta["visibility"] = agent_card.visibility.value
        else:
            meta["visibility"] = "" # Store empty string if None

        if agent_card.stage_focus is not None:
            meta["stage_focus"] = agent_card.stage_focus
        
        # Handle existing list fields (store as comma-separated strings)
        meta["_capabilities_str"] = ",".join(agent_card.capabilities) if agent_card.capabilities else ""
        meta["_tags_str"] = ",".join(agent_card.tags) if agent_card.tags else ""
        meta["_tool_names_str"] = ",".join(agent_card.tool_names) if agent_card.tool_names else ""

        # Handle existing dict fields (store as JSON strings)
        meta["_input_schema_json"] = json.dumps(agent_card.input_schema) if agent_card.input_schema is not None else "null"
        meta["_output_schema_json"] = json.dumps(agent_card.output_schema) if agent_card.output_schema is not None else "null"
        meta["_mcp_tool_input_schemas_json"] = json.dumps(agent_card.mcp_tool_input_schemas) if agent_card.mcp_tool_input_schemas is not None else "null"
        
        if agent_card.correlation_id is not None:
            meta["correlation_id"] = agent_card.correlation_id
        
        # Store other metadata as JSON string
        meta["_agent_card_metadata_json"] = json.dumps(agent_card.metadata) if agent_card.metadata else "{}"
        
        # created field is a datetime object, convert to ISO format string
        meta["created"] = agent_card.created.isoformat()
        
        return meta

    def delete(self, agent_id: str) -> bool:
        # Implement delete logic here
        try:
            self._coll.delete(ids=[agent_id])
            return True
        except Exception: # Broad exception, refine if ChromaDB has specific ones
            return False 