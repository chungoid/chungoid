"""Core execution engine for the Chungoid agent."""

import logging
logger = logging.getLogger(__name__)
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import inspect
import json

# Absolute imports – sys.path hacks removed
from chungoid.utils.state_manager import StateManager, StatusFileError, ChromaOperationError
from chungoid.utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError
from chungoid.utils.config_loader import get_config, ConfigError

# <<< ADDED LOGGING FOR IMPORT PATH >>>
try:
    pm_file_path = inspect.getfile(PromptManager)
    logging.getLogger(__name__).info(f"DEBUG ENGINE: Imported PromptManager class from: {pm_file_path}")
except Exception as inspect_err:
    logging.getLogger(__name__).error(f"DEBUG ENGINE: Error inspecting PromptManager file path: {inspect_err}")
# <<< END ADDED LOGGING >>>

class ChungoidEngine:
    """Manages the execution lifecycle of a Chungoid project."""

    def __init__(self, project_directory: str):
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
        except ConfigError as e:
            logger.exception("Failed to load configuration.")
            raise RuntimeError("Configuration error prevented engine initialization.") from e

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
                "name": "mcp_chungoid_export_cursor_rule", # Direct mapping to existing state_manager method
                "description": "Exports the chungoid_bootstrap.mdc rule to the specified destination path within the project.",
                 "inputSchema": {
                    "type": "object",
                    "properties": {
                        "dest_path": {"type": "string", "default": ".cursor/rules", "description": "Relative destination path for the rule file."}
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
        current_project_dir = self.project_dir # Always use the engine's current project directory
        logger.info(f"Executing MCP tool: {tool_name} with args: {tool_arguments} (ToolCallID: {tool_call_id}) for project: {current_project_dir}")
        
        try:
            if tool_name == "initialize_project":
                # Engine already re-initializes StateManager if project_directory is passed and different
                # This tool can be used to explicitly ensure directories and status file are set up.
                self.state_manager.initialize_project() # Operates on self.state_manager.target_dir_path
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Project initialized at {str(self.state_manager.target_dir_path)}"
                        }
                    ]
                }

            elif tool_name == "set_project_context":
                new_project_dir_str = tool_arguments.get("project_directory")
                if not new_project_dir_str:
                    raise ValueError("Missing 'project_directory' argument for set_project_context")
                
                new_project_path = Path(new_project_dir_str).resolve()
                if not new_project_path.is_dir():
                    # It's okay if it doesn't exist yet, might be initialized next.
                    logger.warning(f"New project context directory {new_project_path} does not exist or is not a directory. It might be created by 'initialize_project'.")

                self.project_dir = new_project_path
                # IMPORTANT: StateManager and PromptManager are tied to the initial project_directory.
                # To truly switch context, these would need to be re-initialized.
                # This is a significant change. For now, this tool will update self.project_dir,
                # but StateManager will still point to the original.
                # This implies mcp.py should re-instantiate ChungoidEngine or pass project_dir per call for true context switching.
                # Given mcp.py *does* pass project_dir to execute_mcp_tool, we should prioritize that.
                
                # Re-initialize components with the new project directory
                # This assumes that mcp.py would create a new engine instance or this engine instance is for this context.
                # The current structure of mcp.py passes the resolved project_directory_path to engine constructor.
                # So, this set_project_context might be more for an *interactive session* where the *same engine instance*
                # is used for multiple commands and needs its internal context changed.
                # If mcp.py makes a new engine per high-level command, this tool is less critical.
                # Let's assume this changes the *current* engine instance's context:
                
                logger.info(f"Attempting to re-initialize StateManager and PromptManager for new context: {new_project_path}")
                core_root = Path(__file__).parent.parent.parent
                stages_dir_path = core_root / 'server_prompts' / 'stages'
                common_prompt_path = core_root / 'server_prompts' / 'common.yaml'
                
                self.state_manager = StateManager(
                    target_directory=str(new_project_path),
                    server_stages_dir=str(stages_dir_path)
                )
                self.prompt_manager = PromptManager(
                    server_stages_dir=str(stages_dir_path),
                    common_template_path=str(common_prompt_path)
                )
                logger.info(f"ChungoidEngine context switched to: {new_project_path}. StateManager and PromptManager re-initialized.")
                # Return structure expected by MCP client for successful tool call
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": f"Project context set to {str(new_project_path)}"
                        }
                    ]
                }

            elif tool_name == "get_project_status":
                # Ensure StateManager is operating on the correct project_dir if passed by mcp.py
                # This requires StateManager to be flexible or re-instantiated.
                # For now, assume StateManager associated with this engine instance is used.
                status_content = self.state_manager.get_full_status() # get_full_status to get the whole dict
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text", 
                            "text": json.dumps(status_content, indent=2)
                        }
                    ]
                }

            elif tool_name == "prepare_next_stage":
                # run_next_stage uses self.project_dir and self.state_manager implicitly
                # It already returns a dict suitable for MCP.
                next_stage_data = self.run_next_stage()
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(next_stage_data, indent=2)
                        }
                    ]
                }

            elif tool_name == "submit_stage_artifacts":
                stage_num = tool_arguments.get("stage_number")
                status_str = tool_arguments.get("stage_result_status")
                artifacts_dict = tool_arguments.get("generated_artifacts")
                reflection = tool_arguments.get("reflection_text") # Optional

                if stage_num is None or status_str is None or artifacts_dict is None:
                    raise ValueError("Missing required arguments for submit_stage_artifacts (stage_number, stage_result_status, generated_artifacts)")
                
                # The StateManager.update_status expects artifact *paths*, not content.
                # The MCP spec implies content is sent. This is a mismatch.
                # For now, let's assume the agent will create files and then pass paths.
                # OR, this tool needs to save the content to files first.
                # The prompt examples show sending content. Let's adapt StateManager or this tool.
                
                # Option 1: Tool saves content to files, then calls SM.update_status with paths.
                # This seems more aligned with how StateManager currently works with artifacts.
                saved_artifact_paths = []
                if isinstance(artifacts_dict, dict):
                    for rel_path, content_str in artifacts_dict.items():
                        if not isinstance(content_str, str):
                            logger.warning(f"Artifact content for {rel_path} is not a string, skipping.")
                            continue
                        try:
                            # Ensure project_dir for the tool call is used
                            artifact_abs_path = Path(current_project_dir) / rel_path
                            artifact_abs_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(artifact_abs_path, "w", encoding="utf-8") as f:
                                f.write(content_str)
                            saved_artifact_paths.append(rel_path) # SM expects relative paths
                            logger.info(f"Saved artifact content to: {artifact_abs_path}")
                        except Exception as e_save:
                            logger.error(f"Failed to save artifact {rel_path}: {e_save}", exc_info=True)
                            # Decide if we should continue or fail the tool call
                            raise IOError(f"Failed to save artifact {rel_path}") from e_save
                else:
                    logger.warning(f"generated_artifacts was not a dictionary: {type(artifacts_dict)}. No files saved.")

                # Handle reflection: StateManager's update_status can take reflection_text
                success = self.state_manager.update_status(
                    stage=float(stage_num),
                    status=status_str.upper(), # Ensure uppercase (PASS, FAIL, etc.)
                    artifacts=saved_artifact_paths, # List of relative paths
                    reflection_text=reflection
                )
                if success:
                    return {
                        "toolCallId": tool_call_id,
                        "content": [
                            {
                                "type": "text",
                                "text": f"Stage {stage_num} submitted with status {status_str} and {len(saved_artifact_paths)} artifacts."
                            }
                        ]
                    }
                else:
                    # update_status currently returns bool, might want more detailed error
                    raise RuntimeError(f"Failed to update status for stage {stage_num}")
            
            elif tool_name == "get_file":
                relative_path = tool_arguments.get("relative_path")
                if not relative_path:
                    raise ValueError("Missing 'relative_path' for get_file tool.")
                
                file_path = Path(current_project_dir) / relative_path
                if not file_path.is_file():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                try:
                    content = file_path.read_text(encoding="utf-8")
                    return {
                        "toolCallId": tool_call_id,
                        "content": [
                            {"type": "text", "text": f"Content of {str(file_path)}:\\n\\n{content}"}
                        ]
                    }
                except Exception as e_read:
                    logger.error(f"Error reading file {file_path}: {e_read}", exc_info=True)
                    raise IOError(f"Could not read file: {file_path}") from e_read

            elif tool_name == "load_reflections": # Corresponds to get_reflection_context_from_chroma
                query = tool_arguments.get("query")
                n_results = tool_arguments.get("n_results", 3)
                if not query:
                    raise ValueError("Missing 'query' for load_reflections tool.")
                # Ensure StateManager's Chroma client is set to the correct project context
                # This might be redundant if set_project_context was called correctly and SM was re-initialized
                if self.state_manager.target_dir_path != current_project_dir:
                     logger.warning(f"load_reflections: StateManager context {self.state_manager.target_dir_path} differs from tool context {current_project_dir}. Re-initializing SM for tool's context.")
                     core_root = Path(__file__).parent.parent.parent
                     stages_dir_path = core_root / 'server_prompts' / 'stages'
                     common_prompt_path = core_root / 'server_prompts' / 'common.yaml'
                     self.state_manager = StateManager(str(current_project_dir), str(stages_dir_path))
                
                reflection_results = self.state_manager.get_reflection_context_from_chroma(query=query, n_results=n_results)
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(reflection_results, indent=2)
                        }
                    ]
                }

            elif tool_name == "set_pending_reflection":
                reflection_text = tool_arguments.get("reflection_text")
                if reflection_text is None: # Allow empty string, but not missing key
                    raise ValueError("Missing 'reflection_text' for set_pending_reflection tool.")
                self.state_manager.set_pending_reflection_text(reflection_text)
                return {
                    "toolCallId": tool_call_id,
                    "content": [
                        {
                            "type": "text",
                            "text": "Pending reflection text has been set."
                        }
                    ]
                }

            elif tool_name == "mcp_chungoid_export_cursor_rule":
                dest_path_str = tool_arguments.get("dest_path", ".cursor/rules") 
                
                if self.state_manager.target_dir_path != current_project_dir:
                     logger.warning(f"mcp_chungoid_export_cursor_rule: StateManager context {self.state_manager.target_dir_path} differs from tool context {current_project_dir}. Re-initializing SM for tool's context.")
                     core_root = Path(__file__).parent.parent.parent
                     stages_dir_path = core_root / 'server_prompts' / 'stages'
                     common_prompt_path = core_root / 'server_prompts' / 'common.yaml'
                     self.state_manager = StateManager(str(current_project_dir), str(stages_dir_path))

                exported_path = self.state_manager.export_cursor_rule(dest_path=dest_path_str)
                if exported_path:
                    return {
                        "toolCallId": tool_call_id,
                        "content": [
                            {
                                "type": "text",
                                "text": f"Cursor rule exported to {str(exported_path)}"
                            }
                        ]
                    }
                else:
                    raise RuntimeError("Failed to export cursor rule, path might be None.")


            else:
                logger.error(f"Unknown MCP tool requested: {tool_name}")
                raise NotImplementedError(f"Tool '{tool_name}' is not implemented.")

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
        engine = ChungoidEngine(str(test_project_path))
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


    except (ValueError, RuntimeError, ConfigError) as e:
        print(f"\nEngine Initialization/Runtime Error: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc() 