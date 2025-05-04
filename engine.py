"""Core execution engine for the Chungoid agent."""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import inspect

# Add project root to path to allow importing project modules
# Adjust based on actual execution context if needed
project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root)) # Commented out for now, assume module structure handles it

try:
    from .utils.state_manager import StateManager, StatusFileError, ChromaOperationError
    from .utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError
    from .utils.config_loader import get_config, ConfigError
except ImportError as e:
    print(f"Error importing Chungoid utils: {e}. Ensure paths are correct or run as module.")
    # Handle path issues more gracefully if running as script vs module
    if Path(__file__).parent.name == 'chungoid-core': # Simple check if running from root
         sys.path.insert(0, str(Path(__file__).parent))
         from utils.state_manager import StateManager, StatusFileError, ChromaOperationError
         from utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError
         from utils.config_loader import get_config, ConfigError
    else:
        raise

logger = logging.getLogger(__name__)
# TODO: Configure logging properly (e.g., inheriting from a central config)

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
            raise ValueError(f"Project directory not found: {self.project_dir}")

        logger.info(f"Initializing ChungoidEngine for project: {self.project_dir}")

        # --- Load Configuration ---
        try:
            self.config = get_config() # Load central config
            # TODO: Potentially pass specific config sections if needed
        except ConfigError as e:
            logger.exception("Failed to load configuration.")
            raise RuntimeError("Configuration error prevented engine initialization.") from e

        # --- Initialize Core Components ---
        try:
            # TODO: Get stages dir path from config or make it relative/standard?
            # Assuming server_stages_dir is findable relative to this file or configured
            # For now, let's assume it's './server_prompts/stages' relative to core root
            core_root = Path(__file__).parent
            stages_dir_path = core_root / 'server_prompts' / 'stages' # Example path
            common_prompt_path = core_root / 'server_prompts' / 'common.yaml' # Example path

            self.state_manager = StateManager(
                target_directory=str(self.project_dir),
                server_stages_dir=str(stages_dir_path) # Pass the stages dir path
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
        import json
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