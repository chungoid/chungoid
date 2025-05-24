"""Core execution engine for the Chungoid agent."""

import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable, Tuple, Union
from datetime import datetime, timezone
import inspect
import json
import uuid
from pydantic import BaseModel, Field, ValidationError
import asyncio # Added asyncio
from pydantic.json import pydantic_encoder # <<< ADD IMPORT

# Absolute imports – sys.path hacks removed
from chungoid.utils.state_manager import StateManager, StatusFileError, ChromaOperationError
from chungoid.utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError
from chungoid.utils.config_manager import get_config, ConfigurationError
from chungoid.utils.llm_provider import LLMProvider # Added LLMProvider
from chungoid.schemas.user_goal_schemas import UserGoalRequest # Added UserGoalRequest
from chungoid.schemas.master_flow import MasterExecutionPlan # Added MasterExecutionPlan
from chungoid.utils.agent_resolver import AgentProvider # Added AgentProvider (assuming engine will manage this)

# <<< ADDED LOGGING FOR IMPORT PATH >>>
try:
    pm_file_path = inspect.getfile(PromptManager)
    logging.getLogger(__name__).info(f"DEBUG ENGINE: Imported PromptManager class from: {pm_file_path}")
except Exception as inspect_err:
    logging.getLogger(__name__).error(f"DEBUG ENGINE: Error inspecting PromptManager file path: {inspect_err}")
# <<< END ADDED LOGGING >>>

class ChungoidEngine:
    """Manages the execution lifecycle of a Chungoid project."""

    def __init__(self, project_directory: str, llm_provider: LLMProvider, agent_provider: AgentProvider): # Added llm_provider and agent_provider
        """
        Initializes the engine for a specific project directory.

        Args:
            project_directory: The absolute path to the target project directory.
        """
        self.project_dir = Path(project_directory).resolve()
        if not self.project_dir.is_dir():
            # Allow engine to proceed if project_dir is a placeholder like ${workspaceFolder}
            # The actual check for its validity can be deferred or handled by tools that need it.
            # Or, if we know this is from MCP, we might have a default cwd already set.
            logger.warning(f"Project directory '{self.project_dir}' not found or not a directory. This might be a placeholder.")
            # For now, let's not raise an error here, assuming it might be resolved by CWD or specific tool logic
            # raise ValueError(f"Project directory not found: {self.project_dir}")

        logger.info(f"Initializing ChungoidEngine for project: {self.project_dir}")

        # --- Load Configuration ---
        try:
            self.config = get_config() # Load central config
        except ConfigurationError as e:
            logger.exception("Failed to load configuration.")
            raise RuntimeError("Configuration error prevented engine initialization.") from e

        self.llm_provider = llm_provider # Store LLMProvider
        self.agent_provider = agent_provider # Store AgentProvider (assuming it's passed in)
        
        # --- Initialize Core Components ---
        try:
            # CORRECTED: core_root should be project root, not src/chungoid
            # Path(__file__) is .../chungoid-core/src/chungoid/engine.py
            # .parent is .../chungoid-core/src/chungoid/
            # .parent.parent is .../chungoid-core/src/
            # .parent.parent.parent is .../chungoid-core/
            core_root = Path(__file__).parent.parent.parent 
            stages_dir_path = core_root / 'server_prompts' / 'stages'
            common_prompt_path = core_root / 'server_prompts' / 'common.yaml'

            logger.info(f"Calculated stages_dir_path: {stages_dir_path}")
            logger.info(f"Calculated common_prompt_path: {common_prompt_path}")

            self.state_manager = StateManager(
                target_directory=str(self.project_dir),
                server_stages_dir=str(stages_dir_path) 
            )
            self.prompt_manager = PromptManager(
                server_stages_dir=str(stages_dir_path),
                common_template_path=str(common_prompt_path)
            )
            logger.info("StateManager and PromptManager initialized successfully.")
        except (StatusFileError, PromptLoadError, ValueError, FileNotFoundError) as e:
            logger.exception("Failed to initialize StateManager or PromptManager.")
            raise RuntimeError("Core component initialization failed.") from e
        except Exception as e:
            logger.exception("An unexpected error occurred during engine initialization.")
            raise RuntimeError("Unexpected initialization error.") from e

        # === Tool Handling Components ===
        # Define tool argument schemas (can be moved to schemas module if complex)
        class MCPToolArgsInitializeProject(BaseModel):
            # No specific arguments needed, uses engine context
            pass
        
        class MCPToolArgsSetProjectContext(BaseModel):
            project_directory: str = Field(..., description="The absolute or relative path to the new project directory.")

        class MCPToolArgsGetProjectStatus(BaseModel):
            pass
        
        class MCPToolArgsPrepareNextStage(BaseModel):
            pass
        
        class MCPToolArgsSubmitStageArtifacts(BaseModel):
            stage_number: float = Field(..., description="The number identifying the completed stage.")
            stage_result_status: str = Field(..., description="Status of the stage (e.g., 'PASS', 'FAIL').")
            generated_artifacts: Dict[str, str] = Field(default={}, description="Dictionary mapping relative artifact paths to their string content.")
            reflection_text: Optional[str] = Field(None, description="Optional reflection text generated by the stage.")
        
        class MCPToolArgsGetFile(BaseModel):
            relative_path: str = Field(..., description="The relative path to the file within the project directory.")
            
        class MCPToolArgsLoadReflections(BaseModel):
            query_text: str = Field(..., description="The query text to search for relevant reflections.")
            n_results: int = Field(default=5, description="Number of reflection results to retrieve.")

        class MCPToolArgsSetPendingReflection(BaseModel):
             reflection_text: str = Field(..., description="The reflection text to store temporarily.")
             
        class MCPToolArgsLoadPendingReflection(BaseModel):
             pass

        class MCPToolArgsExportCursorRule(BaseModel):
            pass # No args needed

        class MCPToolArgsCreateMasterPlan(BaseModel): # New schema
            user_goal: Dict[str, Any] = Field(description="A dictionary representing the UserGoalRequest schema.")

        # ChromaDB MCP Tools Schemas
        class MCPToolArgsChromaListCollections(BaseModel):
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")
            limit: Optional[int] = Field(None, description="Maximum number of collections to return")
            offset: Optional[int] = Field(None, description="Number of collections to skip")
            include_metadata: bool = Field(True, description="Whether to include collection metadata")

        class MCPToolArgsChromaCreateCollection(BaseModel):
            collection_name: str = Field(..., description="Name of the collection to create")
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")
            embedding_function: Optional[str] = Field(None, description="Embedding function to use (optional)")
            metadata: Optional[Dict[str, Any]] = Field(None, description="Collection metadata (optional)")
            get_or_create: bool = Field(True, description="Whether to get existing collection if it exists")

        class MCPToolArgsChromaQueryDocuments(BaseModel):
            collection_name: str = Field(..., description="Name of the collection to query")
            query_texts: Optional[List[str]] = Field(None, description="List of query texts for semantic search")
            n_results: int = Field(10, description="Maximum number of results to return")
            where: Optional[Dict[str, Any]] = Field(None, description="Metadata filter conditions")
            where_document: Optional[Dict[str, Any]] = Field(None, description="Document content filter conditions")
            include: Optional[List[str]] = Field(None, description="What to include in results")
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")

        class MCPToolArgsChromaAddDocuments(BaseModel):
            collection_name: str = Field(..., description="Name of the collection")
            documents: List[str] = Field(..., description="List of document contents to add")
            metadatas: Optional[List[Dict[str, Any]]] = Field(None, description="Optional list of metadata dictionaries")
            ids: Optional[List[str]] = Field(None, description="Optional list of document IDs")
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")
            auto_generate_ids: bool = Field(True, description="Whether to auto-generate IDs if not provided")

        class MCPToolArgsChromaInitializeProjectCollections(BaseModel):
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")
            collections: Optional[List[str]] = Field(None, description="Optional list of specific collections to initialize")
            embedding_function: Optional[str] = Field(None, description="Embedding function to use for collections (optional)")
            force_recreate: bool = Field(False, description="Whether to recreate existing collections")

        class MCPToolArgsChromaGetProjectStatus(BaseModel):
            project_path: Optional[str] = Field(None, description="Path to project directory (optional)")
            project_id: Optional[str] = Field(None, description="Project identifier for context (optional)")
            include_collection_details: bool = Field(True, description="Whether to include detailed collection information")
            include_document_counts: bool = Field(True, description="Whether to include document counts")

        # Define synchronous wrappers for StateManager/other methods called by tools
        # These handle calling the potentially async StateManager methods appropriately
        # Note: If StateManager methods become sync, these wrappers might simplify.
        def _initialize_project_sync_wrapper():
            # Assuming StateManager.initialize_project is synchronous
            self.state_manager.initialize_project()
            return {"message": f"Project initialized at {self.project_dir}"}

        def _get_project_status_sync_wrapper():
            # Assuming StateManager.get_full_status is synchronous
            return self.state_manager.get_full_status() # Returns dict, will be JSON serialized by execute_mcp_tool
        
        def _prepare_next_stage_sync_wrapper():
            # Assuming self.run_next_stage is synchronous
            # run_next_stage currently returns a dict like {"status": ..., "next_stage": ..., "prompt": ...}
            # This should be directly usable as JSON content
            return self.run_next_stage()
        
        def _submit_stage_artifacts_sync_wrapper(**kwargs):
            # Need to save artifact content first, then call update_status
            stage_number = kwargs.get("stage_number")
            status_str = kwargs.get("stage_result_status")
            artifacts_dict = kwargs.get("generated_artifacts", {})
            reflection = kwargs.get("reflection_text")
            error_details_dict = kwargs.get("error_details")

            saved_artifact_paths = []
            if artifacts_dict:
                for rel_path, content in artifacts_dict.items():
                    try:
                        full_path = self.project_dir / rel_path
                        full_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        saved_artifact_paths.append(rel_path)
                        logger.info(f"Saved artifact: {full_path}")
                    except IOError as e:
                        logger.error(f"Failed to save artifact {rel_path}: {e}")
                        # Decide if this is a critical failure for the tool
                        # For now, log and continue, but status update might reflect partial success/failure
            
            # Update status
            success = self.state_manager.update_status(
                stage=stage_number,
                status=status_str,
                artifacts=saved_artifact_paths,
                reflection_text=reflection,
                error_details=error_details_dict
            )
            if not success:
                 logger.error(f"Failed to update status for stage {stage_number} via state_manager.")
                 raise RuntimeError(f"Failed to update status for stage {stage_number}") # Propagate as tool error

            return {"message": f"Stage {stage_number} submitted with status {status_str} and {len(saved_artifact_paths)} artifacts."}

        def _get_file_sync_wrapper(relative_path: str):
            full_path = (self.project_dir / relative_path).resolve()
            if not full_path.is_file():
                raise FileNotFoundError(f"File not found: {full_path}")
            if not full_path.is_relative_to(self.project_dir): # Security check
                raise ValueError(f"File path {full_path} is outside the project directory {self.project_dir}")
            return f"Content of {full_path}:\n\n{full_path.read_text(encoding='utf-8')}"

        def _load_reflections_sync_wrapper(query_text: str, n_results: int):
            # Assuming StateManager.get_reflection_context_from_chroma is synchronous
            # (If it becomes async, need asyncio.run or similar here, carefully)
            try:
                reflections = self.state_manager.get_reflection_context_from_chroma(query=query_text, n_results=n_results)
                return reflections # Already a list of dicts (JSON serializable)
            except ChromaOperationError as e:
                logger.error(f"ChromaDB operation failed during load_reflections: {e}")
                raise # Re-raise to be caught by execute_mcp_tool error handler

        def _set_pending_reflection_sync_wrapper(reflection_text: str):
            self.state_manager.set_pending_reflection(reflection_text)
            return {"message": "Pending reflection stored."}

        def _load_pending_reflection_sync_wrapper():
            text = self.state_manager.load_pending_reflection()
            return {"reflection_text": text if text else ""}

        def _export_cursor_rule_sync_wrapper():
            return self.state_manager.export_cursor_rule()

        def _create_master_plan_sync_wrapper(**kwargs): # New handler
            try:
                user_goal_dict = kwargs['user_goal']
                user_goal_request_obj = UserGoalRequest(**user_goal_dict)
                
                # planner = MasterPlannerAgent(
                #     agent_provider=self.agent_provider, 
                #     llm_provider=self.llm_provider
                # )
                
                # # MasterPlannerAgent.execute is async, so we need to run it in an event loop
                # loop = asyncio.new_event_loop()
                # asyncio.set_event_loop(loop)
                # try:
                #     plan = loop.run_until_complete(planner.execute(user_goal_request_obj))
                # finally:
                #     loop.close()

                # # Save the plan before returning its model dump
                # if not plan.id.startswith("error_plan_"): # Only save successful plans
                #     save_success = self.state_manager.save_master_execution_plan(plan)
                #     if not save_success:
                #         logger.error(f"Failed to save MasterExecutionPlan ID: {plan.id} to state file.")
                #         # Decide if this should alter the tool's success response or just log
                
                # # The wrapper needs to return a dict, not the Pydantic model directly,
                # # because the main execute_mcp_tool tries to json.dumps it.
                # if isinstance(plan, MasterExecutionPlan):
                #     return plan.model_dump() # <<< REVERTED TO THIS
                # elif isinstance(plan, dict): # Error plan already a dict
                #     return plan
                # For now, return an error or placeholder since MasterPlannerAgent is removed
                logger.error("MasterPlannerAgent related code in _create_master_plan_sync_wrapper is commented out due to deprecation.")
                return {"error": "MasterPlannerAgent is deprecated and its usage here is commented out."}
            except ValidationError as ve:
                logger.error(f"UserGoalRequest validation error for create_master_plan: {ve}")
                raise # Re-raise to be caught by execute_mcp_tool error handler
            except Exception as e:
                logger.exception(f"Error executing MasterPlannerAgent for create_master_plan: {e}")
                # Return a structured error that execute_mcp_tool will handle
                raise RuntimeError(f"MasterPlannerAgent execution failed: {e}")

        # ChromaDB MCP Tools Handlers
        def _chroma_list_collections_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_list_collections
            return asyncio.run(chroma_list_collections(**kwargs))

        def _chroma_create_collection_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_create_collection
            return asyncio.run(chroma_create_collection(**kwargs))

        def _chroma_query_documents_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_query_documents
            return asyncio.run(chroma_query_documents(**kwargs))

        def _chroma_add_documents_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_add_documents
            return asyncio.run(chroma_add_documents(**kwargs))

        def _chroma_initialize_project_collections_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_initialize_project_collections
            return asyncio.run(chroma_initialize_project_collections(**kwargs))

        def _chroma_get_project_status_sync_wrapper(**kwargs):
            import asyncio
            from chungoid.mcp_tools.chromadb import chroma_get_project_status
            return asyncio.run(chroma_get_project_status(**kwargs))

        # === TOOL REGISTRY ===
        # Moved registry initialization to __init__ and assigned to self.TOOL_REGISTRY
        self.TOOL_REGISTRY = {
            "initialize_project": {
                "description": "Initializes a new Chungoid project structure and status file if they don't exist.",
                "args_schema": MCPToolArgsInitializeProject,
                "handler_sync": _initialize_project_sync_wrapper,
            },
            "set_project_context": {
                "description": "Sets the active project directory context for subsequent commands within the engine instance.",
                "args_schema": MCPToolArgsSetProjectContext,
                "handler_sync": None, # Handled directly in execute_mcp_tool due to re-init need
            },
             "get_project_status": {
                "description": "Retrieves the current project status.",
                "args_schema": MCPToolArgsGetProjectStatus,
                "handler_sync": _get_project_status_sync_wrapper,
            },
            "prepare_next_stage": {
                "description": "Determines and prepares the prompt for the next stage of the project.",
                "args_schema": MCPToolArgsPrepareNextStage,
                "handler_sync": _prepare_next_stage_sync_wrapper,
            },
            "submit_stage_artifacts": {
                "description": "Submits artifacts from a completed stage, updating status and optionally adding a reflection.",
                "args_schema": MCPToolArgsSubmitStageArtifacts,
                "handler_sync": _submit_stage_artifacts_sync_wrapper,
            },
            "get_file": {
                "description": "Retrieves the content of a specified file within the project.",
                "args_schema": MCPToolArgsGetFile,
                "handler_sync": _get_file_sync_wrapper,
            },
            "load_reflections": {
                "description": "Queries and loads reflections (contextual memories) from ChromaDB based on a query.",
                "args_schema": MCPToolArgsLoadReflections,
                "handler_sync": _load_reflections_sync_wrapper,
            },
            "set_pending_reflection": {
                 "description": "Stores a piece of text temporarily as a pending reflection.",
                 "args_schema": MCPToolArgsSetPendingReflection,
                 "handler_sync": _set_pending_reflection_sync_wrapper,
             },
             "load_pending_reflection": {
                 "description": "Retrieves the currently stored pending reflection text.",
                 "args_schema": MCPToolArgsLoadPendingReflection,
                 "handler_sync": _load_pending_reflection_sync_wrapper,
             },
             "mcp_chungoid_export_cursor_rule": {
                 "description": "Exports the StateManager's internal cursor rule.",
                 "args_schema": MCPToolArgsExportCursorRule,
                 "handler_sync": _export_cursor_rule_sync_wrapper,
             },
            "create_master_plan": { # New tool registration
                "description": "Generates a multi-stage MasterExecutionPlan from a user goal.",
                "args_schema": MCPToolArgsCreateMasterPlan,
                "handler_sync": _create_master_plan_sync_wrapper,
            },
            # ChromaDB MCP Tools Registry Entries
            "chroma_list_collections": {
                "description": "Lists all collections in the project-specific ChromaDB instance with optional filtering.",
                "args_schema": MCPToolArgsChromaListCollections,
                "handler_sync": _chroma_list_collections_sync_wrapper,
            },
            "chroma_create_collection": {
                "description": "Creates a new collection in the project-specific ChromaDB instance with project context awareness.",
                "args_schema": MCPToolArgsChromaCreateCollection,
                "handler_sync": _chroma_create_collection_sync_wrapper,
            },
            "chroma_query_documents": {
                "description": "Queries documents in a collection using semantic search and filtering with project context awareness.",
                "args_schema": MCPToolArgsChromaQueryDocuments,
                "handler_sync": _chroma_query_documents_sync_wrapper,
            },
            "chroma_add_documents": {
                "description": "Adds multiple documents to a collection in the project-specific ChromaDB instance with automatic project metadata.",
                "args_schema": MCPToolArgsChromaAddDocuments,
                "handler_sync": _chroma_add_documents_sync_wrapper,
            },
            "chroma_initialize_project_collections": {
                "description": "Initializes all standard project collections for a Chungoid project, replacing ProjectChromaManagerAgent_v1 functionality.",
                "args_schema": MCPToolArgsChromaInitializeProjectCollections,
                "handler_sync": _chroma_initialize_project_collections_sync_wrapper,
            },
            "chroma_get_project_status": {
                "description": "Gets comprehensive status information for a project's ChromaDB setup including health score and missing collections.",
                "args_schema": MCPToolArgsChromaGetProjectStatus,
                "handler_sync": _chroma_get_project_status_sync_wrapper,
            },
        }
        # === End Tool Handling Components ===

        logger.info(f"ChungoidEngine initialized for project: {self.project_dir}")

    def get_mcp_tools(self) -> list[dict[str, Any]]:
        """Returns a list of tool definitions for MCP."""
        tools = [
            {
                "name": "initialize_project",
                "description": "Initializes a new Chungoid project in the specified directory. Creates .chungoid folder and project_status.json.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_directory": {
                            "type": "string",
                            "description": "The absolute path to the project directory to initialize. If not provided, uses the engine's current project directory.",
                        }
                    },
                    "required": [],
                },
            },
            {
                "name": "set_project_context",
                "description": "Sets the active project directory for the Chungoid engine. Future tool calls will operate on this project.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_directory": {
                            "type": "string",
                            "description": "The absolute path to the project directory.",
                        }
                    },
                    "required": ["project_directory"],
                },
            },
            {
                "name": "get_project_status",
                "description": "Retrieves the current status of the Chungoid project, including completed stages and run history.",
                "inputSchema": {"type": "object", "properties": {}}, # No specific params needed beyond current context
            },
            {
                "name": "prepare_next_stage",
                "description": "Determines the next stage in the workflow and returns its prompt and relevant context.",
                "inputSchema": {"type": "object", "properties": {}}, # No specific params needed
            },
            {
                "name": "submit_stage_artifacts",
                "description": "Submits the artifacts and reflection for a completed stage, updating the project status.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "stage_number": {"type": "number", "description": "The stage number being submitted (e.g., 0.0, 1.0)."},
                        "stage_result_status": {"type": "string", "enum": ["PASS", "FAIL", "DONE", "UNKNOWN"], "description": "The result status of the stage."},
                        "generated_artifacts": {
                            "type": "object",
                            "description": "A dictionary where keys are relative file paths and values are the full file content (as strings).",
                            "additionalProperties": {"type": "string"}
                        },
                        "reflection_text": {"type": ["string", "null"], "description": "Optional reflection text for the stage."}
                    },
                    "required": ["stage_number", "stage_result_status", "generated_artifacts"]
                }
            },
            {
                "name": "get_file",
                "description": "Reads the content of a file within the currently set project context.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "relative_path": {"type": "string", "description": "The relative path to the file from the project root."}
                    },
                    "required": ["relative_path"]
                }
            },
            {
                "name": "load_reflections", # Corresponds to state_manager.get_reflection_context_from_chroma
                "description": "Loads and searches stored reflections from ChromaDB based on a query.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The query string to search reflections."},
                        "n_results": {"type": "integer", "default": 3, "description": "Number of results to return."}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "set_pending_reflection", # Maps to state_manager.set_pending_reflection_text
                "description": "Stages reflection text temporarily before it's submitted with artifacts. Overwrites previous pending reflection.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "reflection_text": {"type": "string", "description": "The reflection text to stage."}
                    },
                    "required": ["reflection_text"]
                }
            },
            {
                "name": "load_pending_reflection", # Added new tool definition
                "description": "Retrieves the currently set pending reflection text that has not yet been submitted.",
                "inputSchema": {"type": "object", "properties": {}} # No input arguments needed
            },
            {
                "name": "mcp_chungoid_export_cursor_rule",
                "description": "Exports the chungoid.mdc rule to the user's global Cursor rules directory (~/.cursor/rules/chungoid.mdc).",
                 "inputSchema": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            # ChromaDB MCP Tools - Enhanced project-aware versions
            {
                "name": "chroma_list_collections",
                "description": "Lists all collections in the project-specific ChromaDB instance with optional filtering.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"},
                        "limit": {"type": "integer", "description": "Maximum number of collections to return"},
                        "offset": {"type": "integer", "description": "Number of collections to skip"},
                        "include_metadata": {"type": "boolean", "default": True, "description": "Whether to include collection metadata"}
                    },
                    "required": []
                }
            },
            {
                "name": "chroma_create_collection",
                "description": "Creates a new collection in the project-specific ChromaDB instance with project context awareness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {"type": "string", "description": "Name of the collection to create"},
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"},
                        "embedding_function": {"type": "string", "description": "Embedding function to use (optional)"},
                        "metadata": {"type": "object", "description": "Collection metadata (optional)"},
                        "get_or_create": {"type": "boolean", "default": True, "description": "Whether to get existing collection if it exists"}
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "chroma_query_documents",
                "description": "Queries documents in a collection using semantic search and filtering with project context awareness.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {"type": "string", "description": "Name of the collection to query"},
                        "query_texts": {"type": "array", "items": {"type": "string"}, "description": "List of query texts for semantic search"},
                        "n_results": {"type": "integer", "default": 10, "description": "Maximum number of results to return"},
                        "where": {"type": "object", "description": "Metadata filter conditions"},
                        "where_document": {"type": "object", "description": "Document content filter conditions"},
                        "include": {"type": "array", "items": {"type": "string"}, "description": "What to include in results"},
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"}
                    },
                    "required": ["collection_name"]
                }
            },
            {
                "name": "chroma_add_documents",
                "description": "Adds multiple documents to a collection in the project-specific ChromaDB instance with automatic project metadata.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "collection_name": {"type": "string", "description": "Name of the collection"},
                        "documents": {"type": "array", "items": {"type": "string"}, "description": "List of document contents to add"},
                        "metadatas": {"type": "array", "items": {"type": "object"}, "description": "Optional list of metadata dictionaries"},
                        "ids": {"type": "array", "items": {"type": "string"}, "description": "Optional list of document IDs"},
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"},
                        "auto_generate_ids": {"type": "boolean", "default": True, "description": "Whether to auto-generate IDs if not provided"}
                    },
                    "required": ["collection_name", "documents"]
                }
            },
            {
                "name": "chroma_initialize_project_collections",
                "description": "Initializes all standard project collections for a Chungoid project, replacing ProjectChromaManagerAgent_v1 functionality.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"},
                        "collections": {"type": "array", "items": {"type": "string"}, "description": "Optional list of specific collections to initialize"},
                        "embedding_function": {"type": "string", "description": "Embedding function to use for collections (optional)"},
                        "force_recreate": {"type": "boolean", "default": False, "description": "Whether to recreate existing collections"}
                    },
                    "required": []
                }
            },
            {
                "name": "chroma_get_project_status",
                "description": "Gets comprehensive status information for a project's ChromaDB setup including health score and missing collections.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "project_path": {"type": "string", "description": "Path to project directory (optional)"},
                        "project_id": {"type": "string", "description": "Project identifier for context (optional)"},
                        "include_collection_details": {"type": "boolean", "default": True, "description": "Whether to include detailed collection information"},
                        "include_document_counts": {"type": "boolean", "default": True, "description": "Whether to include document counts"}
                    },
                    "required": []
                }
            }
            # TODO: Add retrieve_reflections if it's distinct from load_reflections
            # TODO: Add other tools as needed
        ]
        logger.info(f"MCP Tool List generated with {len(tools)} tools.")
        return tools

    def execute_mcp_tool(self, tool_name: str, tool_arguments: dict, tool_call_id: Optional[str] = None) -> Any:
        """
        Executes the specified MCP tool with the given arguments.

        Args:
            tool_name: The name of the tool to execute.
            tool_arguments: A dictionary of arguments for the tool.
            tool_call_id: Optional ID for the specific tool call (from MCP).

        Returns:
            The result of the tool execution. Structure depends on the tool.
        """
        logger.info(f"Executing MCP tool '{tool_name}' with args: {tool_arguments}")
        current_project_dir = Path(self.project_dir).resolve()
        tool_call_id = tool_call_id or f"tool-{uuid.uuid4()}"

        # Get tool specification from registry
        tool_spec = self.TOOL_REGISTRY.get(tool_name)
        if not tool_spec:
            logger.error(f"Unknown MCP tool requested: {tool_name}")
            raise NotImplementedError(f"Tool '{tool_name}' is not implemented.")

        try:
            # Validate arguments against the tool's Pydantic schema
            validated_args = tool_spec["args_schema"](**tool_arguments)
            validated_args_dict = validated_args.model_dump(exclude_unset=True) # Use validated dict
            logger.debug(f"Validated tool arguments for {tool_name}: {validated_args_dict}")

            # Get the handler (sync wrapper)
            handler = tool_spec["handler_sync"]
            if not handler or not callable(handler):
                 raise NotImplementedError(f"Handler not found or not callable for tool '{tool_name}'.")
            
            # Execute the handler with validated arguments
            result_data = handler(**validated_args_dict)
            
            # Format the successful result according to MCP spec (basic structure)
            # Specific tools might return different structures, handlers should ideally conform
            # For now, assuming handler returns data to be put in JSON text content.
            if result_data is not None:
                if isinstance(result_data, str): # If handler already returned a string
                    text_content = result_data
                else:
                    text_content = json.dumps(result_data, indent=2, default=pydantic_encoder) # <<< NEW LINE

            response = {
                "toolCallId": tool_call_id,
                "content": [
                    {
                        "type": "text",
                        "text": text_content
                    }
                ]
            }
            return response
            
        except (ValidationError) as e_val:
            logger.error(f"Tool argument validation failed for {tool_name} (ToolCallID: {tool_call_id}): {e_val}", exc_info=True)
            return {
                "toolCallId": tool_call_id,
                "isError": True,
                "error": {
                    "code": -32602, # Invalid params
                    "message": f"Invalid arguments for tool {tool_name}: {e_val}",
                    "tool_name": tool_name,
                },
                "content": [{"type": "text", "text": f"Invalid arguments for tool {tool_name}: {str(e_val)}"}]
            }

        except (ValueError, FileNotFoundError, NotImplementedError, IOError, RuntimeError, StatusFileError, ChromaOperationError) as e_tool:
           logger.error(f"Error executing tool {tool_name} (ToolCallID: {tool_call_id}): {e_tool}", exc_info=True)
           # MCP spec suggests returning error within the result structure
           # This structure might vary based on client expectations for tool errors
           return {
               "toolCallId": tool_call_id, # Echo back the ID
               "isError": True,
               "error": { # This is a custom error structure, MCP spec might have a more defined one.
                   "code": -32001, # Generic tool error
                   "message": str(e_tool),
                   "tool_name": tool_name,
               },
               "content": [{"type": "text", "text": f"Error in tool {tool_name}: {str(e_tool)}"}] # MCP-style content
           }
        except Exception as e_unhandled: # Catch-all for unexpected errors
            logger.error(f"Unexpected error executing tool {tool_name} (ToolCallID: {tool_call_id}): {e_unhandled}", exc_info=True)
            return {
                "toolCallId": tool_call_id,
                "isError": True,
                 "error": {
                    "code": -32000, # Generic server error during tool execution
                    "message": f"Unexpected server error: {str(e_unhandled)}",
                    "tool_name": tool_name,
                },
                "content": [{"type": "text", "text": f"Unexpected server error in tool {tool_name}: {str(e_unhandled)}"}]
            }

    def run_next_stage(self) -> Dict[str, Any]:
        """
        Determines, prepares, and simulates the execution of the next stage.

        Returns:
            A dictionary containing the status and outcome (e.g., prompt for the stage).
        """
        logger.info(f"Attempting to run next stage for project: {self.project_dir}")
        result: Dict[str, Any] = {"status": "error", "message": "Unknown error"}

        try:
            # 1. Determine the next stage
            next_stage = self.state_manager.get_next_stage()
            if next_stage is None:
                # This condition might mean the project is complete or stalled
                logger.info("StateManager indicated no next stage available (project likely complete or failed).")
                result = {"status": "complete", "message": "No further stages defined or last stage failed.", "next_stage": None}
                return result # Or handle differently if needed

            logger.info(f"Determined next stage: {next_stage}")

            # 2. Get Context (Placeholder - Future Step)
            context_data = self._gather_context(next_stage)
            logger.debug(f"Context gathered for stage {next_stage}: {str(context_data)[:100]}...")


            # 3. Get the Prompt for the stage
            rendered_prompt = self.prompt_manager.get_rendered_prompt(
                stage_number=next_stage,
                context_data=context_data # Pass gathered context
            )
            logger.info(f"Successfully rendered prompt for stage {next_stage}.")

            # 4. Execute the Prompt (Placeholder - Future Step)
            # --- REMOVED LLM Execution Placeholder ---
            # # In a real scenario, this would involve calling an LLM
            # # For now, we just return the prompt itself as the main result.
            # logger.warning("Placeholder for actual LLM execution.")
            # llm_response = f"(Simulated LLM Execution for Stage {next_stage})"


            # 5. Update Status (Placeholder - Requires results/artifacts)
            # --- REMOVED Status Update Placeholder (Handled by submit_stage_artifacts tool) ---
            # # This would happen *after* processing the LLM response
            # # self.state_manager.update_status(stage=next_stage, status="DONE", artifacts=[...])
            # logger.warning("Placeholder for updating status based on LLM execution.")


            # --- Prepare successful result for the MCP tool ---
            # Return the stage number, prompt, and context for the agent to use
            result = {
                "status": "success",
                "message": f"Prepared prompt and context for stage {next_stage}.",
                "next_stage": next_stage,
                "prompt": rendered_prompt,
                # "simulated_llm_response": llm_response, # Removed
                "gathered_context": context_data # Return context for visibility/debugging
            }
            return result

        except (StatusFileError, PromptLoadError, PromptRenderError, ChromaOperationError) as e:
            logger.exception(f"A controlled error occurred during stage execution: {e}")
            result = {"status": "error", "message": f"Error during stage {next_stage if 'next_stage' in locals() else 'determination'}: {e}"}
            return result
        except RuntimeError as e:
             # Catch runtime errors from get_next_stage (e.g., no stages defined)
             logger.exception(f"Runtime error during stage execution: {e}")
             result = {"status": "error", "message": f"Runtime error: {e}"}
             return result
        except Exception as e:
            logger.exception("An unexpected error occurred during run_next_stage.")
            result = {"status": "error", "message": f"Unexpected error: {e}"}
            return result

    def _gather_context(self, stage_number: float) -> Dict[str, Any]:
        """
        Gathers necessary context for a given stage.

        This method will evolve to query StateManager (ChromaDB) for relevant
        past reflections, artifact contents, etc., based on the stage definition.

        Args:
            stage_number: The stage for which to gather context.

        Returns:
            A dictionary containing context data for the prompt template.
        """
        logger.info(f"Gathering context for stage {stage_number}...")
        context = {}

        # --- Static Context --- (Always include)
        context["project_directory"] = str(self.project_dir)
        context["current_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        context["current_stage"] = stage_number # Add current stage to context

        # --- Dynamic Context --- (Based on stage number for now)
        try:
            # Always try to get the last status update
            last_status = self.state_manager.get_last_status()
            context['last_status'] = last_status if last_status else "No previous status found."
            logger.debug(f"Gathered last_status for stage {stage_number}.")

            # Add more complex context gathering for later stages
            if stage_number > 0:
                 # Example: Get relevant reflections for stages > 0
                 reflection_query = f"Context or reflections relevant to starting or executing stage {stage_number}"
                 reflections = self.state_manager.get_reflection_context_from_chroma(
                     query=reflection_query, n_results=3
                 )
                 context['relevant_reflections'] = reflections # reflections is already [] if none found
                 logger.debug(f"Gathered {len(reflections)} reflections for stage {stage_number}.")

                 # Example: Get some recent artifact metadata
                 # artifact_metadata = self.state_manager.list_artifact_metadata(limit=5)
                 # context['recent_artifact_metadata'] = artifact_metadata
                 # logger.debug(f"Gathered {len(artifact_metadata)} artifact metadata entries.")

            # Add more stage-specific context logic here...
            # E.g., for stage 3 (implementation), maybe query for specific artifacts from stage 2
            # if stage_number == 3:
            #     planning_docs = self.state_manager.get_artifact_context_from_chroma(
            #         query="implementation plan or detailed interfaces",
            #         n_results=2,
            #         where_filter={"stage_number": 2.0, "artifact_type": "planning"} # Example filter
            #     )
            #     context['planning_documents'] = planning_docs

        except ChromaOperationError as e:
            logger.warning(
                f"ChromaDB operation failed during context gathering for stage {stage_number}: {e}. Proceeding with potentially limited context."
            )
            context['chroma_error'] = f"ChromaDB Error: {e}" # Add error info to context
        except Exception as e:
            logger.warning(f"Unexpected error during context gathering for stage {stage_number}: {e}")
            context['gathering_error'] = f"Unexpected Context Error: {e}" # Add error info

        logger.info(f"Finished gathering context for stage {stage_number}.")
        return context


# Example of how to potentially run this engine (e.g., from a server endpoint or CLI)
if __name__ == "__main__":
    # Basic logging setup for direct script execution
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Example: Replace with actual project path or command-line argument
    # Ensure the test project exists and is initialized if needed
    test_project_path = Path("./test_engine_project").resolve()

    if not test_project_path.exists():
         print(f"Creating and initializing test project directory: {test_project_path}")
         test_project_path.mkdir()
         # We need to ensure it's initialized for StateManager to work
         # This requires calling the init logic, potentially from a separate script/tool
         # For now, manually create .chungoid if testing directly
         (test_project_path / ".chungoid").mkdir(exist_ok=True)
         # Need dummy stage files too
         dummy_stages_path = Path(__file__).parent / 'server_prompts' / 'stages'
         dummy_stages_path.mkdir(parents=True, exist_ok=True)
         (dummy_stages_path / "stage0.yaml").write_text("prompt_details: Stage 0 details\nuser_prompt: Execute stage 0")
         (dummy_stages_path / "stage1.yaml").write_text("prompt_details: Stage 1 details\nuser_prompt: Execute stage 1")
         (Path(__file__).parent / 'server_prompts' / "common.yaml").write_text("preamble: Common Preamble")


         print("NOTE: Project might need manual initialization steps (e.g., status file creation) if not done previously.")


    print(f"Running engine for project: {test_project_path}")
    try:
        engine = ChungoidEngine(str(test_project_path), None, None)
        stage_result = engine.run_next_stage()

        print("\n--- Engine Result ---")
        print(json.dumps(stage_result, indent=2))

        # Example: Simulate submitting results (if stage run was successful)
        if stage_result.get("status") == "success" and stage_result.get("next_stage") is not None:
             current_stage = stage_result["next_stage"]
             print(f"\n--- Simulating Status Update for Stage {current_stage} ---")
             # Normally, artifacts would come from processing LLM response
             dummy_artifacts = [f"output/stage_{current_stage}_result.txt"]
             update_success = engine.state_manager.update_status(
                 stage=current_stage,
                 status="PASS", # Assuming success for demo
                 artifacts=dummy_artifacts
             )
             print(f"Status update successful: {update_success}")

             # Run next stage again
             print("\n--- Running Engine Again ---")
             stage_result_2 = engine.run_next_stage()
             print(json.dumps(stage_result_2, indent=2))


    except (ValueError, RuntimeError, ConfigurationError) as e:
        print(f"\nEngine Initialization/Runtime Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 