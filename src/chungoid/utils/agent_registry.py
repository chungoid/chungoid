from __future__ import annotations

"""Agent registry backed by Chroma collection `a2a_agent_registry`."""

from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
from pathlib import Path
from pydantic import BaseModel, Field
import json
import logging

# Define logger at module level
logger = logging.getLogger(__name__)

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
    input_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for the agent's direct input if it's a callable.")
    output_schema: Optional[Dict[str, Any]] = Field(default=None, description="JSON schema for the agent's direct output if it's a callable.")
    mcp_tool_input_schemas: Optional[Dict[str, Any]] = Field(None, description="Summarized input schemas for MCP tools this agent EXPOSES.")
    correlation_id: Optional[str] = Field(None, description="Identifier to correlate related agent invocations or tasks. Can be inherited or generated.")
    metadata: dict = Field(default_factory=dict)
    created: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # New fields for External MCP Tool Proxy
    agent_type: Optional[Literal["Internal", "ExternalMCPProxy"]] = Field(default="Internal", description="Type of agent: Internal or a proxy for an external MCP tool.")
    external_tool_info: Optional[Dict[str, Any]] = Field(default=None, description="If agent_type is ExternalMCPProxy, this holds info like base_url, specific_mcp_tool_spec, auth.")


class AgentRegistry:
    COLLECTION = "a2a_agent_registry"
    DEFAULT_EXTERNAL_TOOLS_CONFIG = "config/external_mcp_servers.json"

    def __init__(self, *, project_root: Path, chroma_mode: str = "persistent"):
        self._client: ClientAPI = get_client(chroma_mode, project_root)
        self._coll: Collection = self._client.get_or_create_collection(self.COLLECTION)
        self._project_root = project_root # Store project_root

        # Load external tools if config exists
        external_tools_config_path = self._project_root / self.DEFAULT_EXTERNAL_TOOLS_CONFIG
        if external_tools_config_path.exists():
            logger.info(f"Attempting to load external MCP tools from {external_tools_config_path}")
            try:
                self.load_external_mcp_tools(external_tools_config_path)
            except Exception as e_load_ext:
                logger.error(f"Failed to load external MCP tools from {external_tools_config_path}: {e_load_ext}")
        else:
            logger.info(f"External MCP tools config file not found at {external_tools_config_path}, skipping load.")

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

        # Add info for external tools
        if card.agent_type == "ExternalMCPProxy" and card.external_tool_info:
            doc_parts.append(f"Agent Type: External MCP Tool Proxy")
            tool_spec = card.external_tool_info.get('specific_mcp_tool_spec', {})
            base_url = card.external_tool_info.get('base_url', 'N/A')
            auth_type = card.external_tool_info.get('authentication', {}).get('type', 'None')
            
            doc_parts.append(f"External Tool Action: {tool_spec.get('tool_name', 'Unknown Action')}")
            doc_parts.append(f"External Tool Base URL: {base_url}")
            doc_parts.append(f"External Tool Authentication: {auth_type}")
            if tool_spec.get('description'):
                 doc_parts.append(f"External Tool Action Description: {tool_spec.get('description')}")

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
        
        # AGENT_NOT_FOUND_HANDLING_START
        if not res["ids"]:
            logger.warning(f"AgentRegistry.get: Agent ID '{agent_id}' not found on first try. Attempting collection refresh and checking external tools load status.")
            # Attempt to reload external tools if the default config file exists
            # This is a simple heuristic: if an agent is not found, maybe external tools weren't loaded.
            external_tools_config_path = self._project_root / self.DEFAULT_EXTERNAL_TOOLS_CONFIG
            if external_tools_config_path.exists():
                logger.info(f"Re-attempting to load external MCP tools from {external_tools_config_path} as part of get() fallback.")
                try:
                    self.load_external_mcp_tools(external_tools_config_path, force_reload=True)
                except Exception as e_load_ext_retry:
                    logger.error(f"Failed to reload external MCP tools during get() fallback: {e_load_ext_retry}")
            else:
                logger.info("External tools config not found, cannot reload them during get() fallback.")
            
            try:
                self._coll = self._client.get_or_create_collection(self.COLLECTION) # Re-fetch collection
                res = self._coll.get(ids=[agent_id]) # Retry the get
                if not res["ids"]:
                    logger.warning(f"AgentRegistry.get: Agent ID '{agent_id}' still not found after collection refresh and potential external tool reload.")
                    return None # Still not found
                else:
                    logger.info(f"AgentRegistry.get: Agent ID '{agent_id}' found after collection refresh and potential external tool reload.")
            except Exception as e:
                logger.error(f"AgentRegistry.get: Error during collection refresh attempt for '{agent_id}': {e}")
                return None # Error during refresh, treat as not found
        # AGENT_NOT_FOUND_HANDLING_END

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

        # Deserialize new fields
        agent_type_str = card_data.pop("agent_type", "Internal") # Default to Internal if missing for backward compatibility
        card_data["agent_type"] = agent_type_str

        external_tool_info_json_str = card_data.pop("external_tool_info_json", "null")
        card_data["external_tool_info"] = json.loads(external_tool_info_json_str)

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

            # TEMPORARY_LOG_START - Remove after debugging
            # if "HumanFeedbackService_v1" in current_agent_id_from_peek:
            #     logger.info(f"AgentRegistry.list: Processing external tool candidate: {current_agent_id_from_peek}")
            #     logger.info(f"Raw Chroma Meta for {current_agent_id_from_peek}: {current_chroma_meta}")
            # TEMPORARY_LOG_END

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
            
            # Deserialize new fields for list
            agent_type_str = card_data.pop("agent_type", "Internal")
            card_data["agent_type"] = agent_type_str

            external_tool_info_json_str = card_data.pop("external_tool_info_json", "null")
            card_data["external_tool_info"] = json.loads(external_tool_info_json_str)
            
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

                # Deserialize new fields
                agent_type_str = card_data.pop("agent_type", "Internal")
                card_data["agent_type"] = agent_type_str

                external_tool_info_json_str = card_data.pop("external_tool_info_json", "null")
                card_data["external_tool_info"] = json.loads(external_tool_info_json_str)
                
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
        """Converts an AgentCard to a flat dictionary suitable for Chroma metadata."""
        # Start with a model dump, then process fields that need special handling
        # Using exclude_none=True to avoid storing lots of nulls if Pydantic adds them by default
        # However, be careful if some fields *should* be None vs not present. Chroma converts None to string "None".
        # Best to handle conversions explicitly.
        meta = agent_card.model_dump(mode='json', exclude_none=False) # Get all fields, including None

        # Required fields that are directly usable or become string representations
        # Ensure all AgentCard fields are represented here or handled below
        chroma_meta = {
            "agent_id": agent_card.agent_id,
            "name": agent_card.name,
            "version": agent_card.version if agent_card.version is not None else "",
            "description": agent_card.description if agent_card.description is not None else "",
            "stage_focus": agent_card.stage_focus if agent_card.stage_focus is not None else "",
            "correlation_id": agent_card.correlation_id if agent_card.correlation_id is not None else "",
            "created": agent_card.created.isoformat(),
            "priority": agent_card.priority if agent_card.priority is not None else 0,
             # New fields serialization
            "agent_type": agent_card.agent_type if agent_card.agent_type is not None else "Internal",
            "external_tool_info_json": json.dumps(agent_card.external_tool_info) if agent_card.external_tool_info is not None else "null",
        }

        # Handle enum fields by storing their string value or empty string for None
        if agent_card.visibility is not None:
            chroma_meta["visibility"] = agent_card.visibility.value
        else:
            chroma_meta["visibility"] = "" # Store empty string if None

        # Handle existing list fields (store as comma-separated strings)
        chroma_meta["_capabilities_str"] = ",".join(agent_card.capabilities) if agent_card.capabilities else ""
        chroma_meta["_tags_str"] = ",".join(agent_card.tags) if agent_card.tags else ""
        chroma_meta["_tool_names_str"] = ",".join(agent_card.tool_names) if agent_card.tool_names else ""

        # Handle existing dict fields (store as JSON strings)
        chroma_meta["_input_schema_json"] = json.dumps(agent_card.input_schema) if agent_card.input_schema is not None else "null"
        chroma_meta["_output_schema_json"] = json.dumps(agent_card.output_schema) if agent_card.output_schema is not None else "null"
        chroma_meta["_mcp_tool_input_schemas_json"] = json.dumps(agent_card.mcp_tool_input_schemas) if agent_card.mcp_tool_input_schemas is not None else "null"
        
        # Restore original logic for categories and capability_profile
        if agent_card.categories: 
            chroma_meta["categories_str"] = ",".join(agent_card.categories)
        else:
            chroma_meta["categories_str"] = ""

        if agent_card.capability_profile: 
            chroma_meta["capability_profile_json"] = json.dumps(agent_card.capability_profile)
        else:
            chroma_meta["capability_profile_json"] = "{}"
        
        return chroma_meta

    def load_external_mcp_tools(self, config_file_path: Path, force_reload: bool = False):
        """Loads external MCP tool definitions from a JSON config file and registers them."""
        if not config_file_path.exists():
            logger.warning(f"External MCP tools config file not found: {config_file_path}")
            return

        # Simple check to avoid reloading if not forced (can be made more sophisticated)
        # This check is basic. A more robust check might involve checking a timestamp or a specific flag.
        # For now, if 'force_reload' is False, and we find any ExternalMCPProxy, we assume they are loaded.
        # This is primarily to prevent redundant loads during the `get` fallback.
        if not force_reload:
            existing_proxies = self.search_agents(query_text="External MCP Tool Proxy", n_results=1, where_filter={"agent_type": "ExternalMCPProxy"})
            if existing_proxies:
                logger.info(f"External MCP tools seem to be loaded already (found {existing_proxies[0].agent_id}). Skipping reload unless forced.")
                return

        logger.info(f"Loading external MCP tools from: {config_file_path}")
        try:
            with open(config_file_path, 'r') as f:
                external_tools_data = json.load(f)
        except Exception as e:
            logger.error(f"Error reading or parsing external MCP tools config {config_file_path}: {e}")
            return

        for tool_entry in external_tools_data:
            tool_id = tool_entry.get("tool_id")
            tool_name_overall = tool_entry.get("name", tool_id)
            base_url = tool_entry.get("base_url")
            authentication = tool_entry.get("authentication")
            
            if not tool_id or not base_url:
                logger.warning(f"Skipping external tool entry due to missing 'tool_id' or 'base_url': {tool_entry}")
                continue

            for mcp_spec in tool_entry.get("mcp_tool_specs", []):
                spec_tool_name = mcp_spec.get("tool_name")
                if not spec_tool_name:
                    logger.warning(f"Skipping mcp_tool_spec in {tool_id} due to missing 'tool_name': {mcp_spec}")
                    continue

                agent_id = f"{tool_id}::{spec_tool_name}"
                agent_name = f"{tool_name_overall} - {spec_tool_name}"
                description = mcp_spec.get("description", f"External MCP tool action {spec_tool_name} for {tool_name_overall}")

                # TODO: Resolve input_schema_ref and output_schema_ref to actual schemas
                # For now, storing refs in external_tool_info.
                # Actual schemas on AgentCard can be generic or None for proxies initially.
                
                external_tool_info_payload = {
                    "base_url": base_url,
                    "specific_mcp_tool_spec": mcp_spec, # Contains tool_name, description, schema_refs
                    "authentication": authentication,
                    "parent_tool_id": tool_id
                }

                card = AgentCard(
                    agent_id=agent_id,
                    name=agent_name,
                    description=description,
                    agent_type="ExternalMCPProxy",
                    categories=["ExternalTool", "MCPProxy", tool_name_overall.replace(" ", "")], # Basic categories
                    capability_profile={"external_action": spec_tool_name, "proxy_for": tool_id},
                    external_tool_info=external_tool_info_payload,
                    # input_schema, output_schema could be generic or loaded from refs if implemented
                )
                try:
                    logger.info(f"Registering external MCP tool proxy: {agent_id}")
                    self.add(card, overwrite=True) # Overwrite to allow updates if config changes
                except Exception as e_add_card:
                    logger.error(f"Failed to register external MCP tool proxy {agent_id}: {e_add_card}")

    def delete(self, agent_id: str) -> bool:
        # Implement delete logic here
        try:
            self._coll.delete(ids=[agent_id])
            logger.info(f"Agent {agent_id} deleted successfully from registry.")
            return True
        except Exception as e:
            logger.error(f"Error deleting agent {agent_id}: {e}")
            return False 