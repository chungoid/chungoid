#!/usr/bin/env python
import asyncio
import os
import shutil
import tempfile
from pathlib import Path
import logging
import sys
import json

# Add chungoid-core/src to sys.path to allow imports
# Assuming this script is run from chungoid-mcp root
# or chungoid-mcp/dev/scripts or chungoid-mcp/chungoid-core/scripts
SCRIPT_DIR = Path(__file__).parent.resolve()
# Adjust project root detection for flexibility
if SCRIPT_DIR.name == "scripts" and SCRIPT_DIR.parent.name == "dev" and SCRIPT_DIR.parent.parent.name == "chungoid-mcp":
    # Running from chungoid-mcp/dev/scripts
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
elif SCRIPT_DIR.name == "scripts" and SCRIPT_DIR.parent.name == "chungoid-core":
    # Running from chungoid-mcp/chungoid-core/scripts
    PROJECT_ROOT = SCRIPT_DIR.parent.parent # This would be chungoid-mcp
else:
    # Default or fallback if structure is different (e.g. running from chungoid-mcp root directly)
    PROJECT_ROOT = Path.cwd()

CORE_SRC_PATH = PROJECT_ROOT / "chungoid-core" / "src"
# Ensure this path is valid if running from within chungoid-core directly
# For a standalone chungoid-core, src would be PROJECT_ROOT / "src"
if not CORE_SRC_PATH.exists() and (PROJECT_ROOT / "src").exists():
   CORE_SRC_PATH = PROJECT_ROOT / "src"

sys.path.insert(0, str(CORE_SRC_PATH))

try:
    from chungoid.engine import ChungoidEngine
    from chungoid.schemas.user_goal_schemas import UserGoalRequest
    from chungoid.utils.llm_provider import MockLLMProvider, OpenAILLMProvider
    from chungoid.utils.agent_resolver import RegistryAgentProvider
    from chungoid.utils.agent_registry import AgentRegistry
    # Import the specific MasterPlannerAgent the engine tool uses
    from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent as SystemMasterPlanner
    from chungoid.schemas.master_flow import MasterExecutionPlan # For parsing the result
    # Corrected import path for MasterPlannerInput and MasterPlannerOutput
    from chungoid.schemas.agent_master_planner import MasterPlannerInput, MasterPlannerOutput
except ImportError as e:
    print(f"Error importing Chungoid components: {e}")
    print(f"Ensure CORE_SRC_PATH is correct: {CORE_SRC_PATH}")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TempBuildScript")

async def main():
    # 1. Create a temporary directory for the build
    temp_build_dir = Path(tempfile.mkdtemp(prefix="chungoid_temp_build_"))
    logger.info(f"Temporary build directory created at: {temp_build_dir}")

    # 2. Set up OpenAI API Key (even if using MockLLM, good practice)
    #    User provided path: /home/flip/.openaikey # TODO: Make this configurable or use env vars
    api_key_path = Path(os.getenv("OPENAI_API_KEY_PATH", "/home/flip/.openaikey")) 
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key and api_key_path.exists():
        with open(api_key_path, "r") as f:
            api_key = f.read().strip()
        logger.info("OpenAI API key found from path.")
    elif api_key:
        logger.info("OpenAI API key found from environment variable.")
    else:
        logger.warning(f"API key file not found at {api_key_path} and OPENAI_API_KEY env var not set. Will use MockLLMProvider.")

    try:
        # 3. Instantiate Providers
        if api_key:
            try:
                llm_provider = OpenAILLMProvider(api_key=api_key)
                logger.info("OpenAILLMProvider instantiated.")
            except Exception as e:
                logger.error(f"Failed to instantiate OpenAILLMProvider: {e}. Falling back to MockLLMProvider.")
                llm_provider = MockLLMProvider()
                logger.info("MockLLMProvider instantiated as fallback.")
        else:
            llm_provider = MockLLMProvider()
            logger.info("MockLLMProvider instantiated.")

        # RegistryAgentProvider needs access to the main project's agent registry (ChromaDB)
        # It takes project_root which is where .chungoid (for agent registry) is expected.
        # TODO: This should be configurable, or determine if a local/temp registry is more appropriate for this script.
        # For chungoid-core tests, it might point to a test project or a temp registry.
        main_project_root_for_registry = PROJECT_ROOT 
        
        # First, create an AgentRegistry instance
        agent_registry_instance = AgentRegistry(
            project_root=main_project_root_for_registry,
            chroma_mode="persistent" # Or "memory" if preferred for this script
        )
        logger.info(f"AgentRegistry instantiated with project_root: {main_project_root_for_registry}")

        # Then, pass this instance to RegistryAgentProvider
        agent_provider = RegistryAgentProvider(registry=agent_registry_instance)
        logger.info(f"RegistryAgentProvider instantiated with AgentRegistry instance.")

        # 4. Instantiate ChungoidEngine for the temporary build directory
        engine = ChungoidEngine(
            project_directory=str(temp_build_dir),
            llm_provider=llm_provider,
            agent_provider=agent_provider
        )
        logger.info(f"ChungoidEngine instantiated for temporary directory: {temp_build_dir}")

        # 5. Define the User Goal
        user_goal = UserGoalRequest(
            goal_description="Create a 3-page website. "
                             "Page 1 (Home): Static content, navigation bar. "
                             "Page 2 (Blog): Dynamically loaded blog posts (mocked data is fine), basic list and detail view. "
                             "Page 3 (Contact): A simple contact form (front-end only). "
                             "Use a common CSS framework like Bootstrap or Tailwind CSS. Basic responsive design.",
            target_platform="web",
            key_constraints={ 
                "technologies": ["HTML, CSS, JavaScript", "Bootstrap or Tailwind CSS"],
                "notes": "No backend for the contact form submission needed. Focus on structure and front-end elements.",
                "style_preference": "modern and clean"
            }
        )
        logger.info(f"UserGoalRequest defined: {user_goal.goal_description[:100]}...")

        # 6. Execute the User Goal Request
        logger.info("Executing user goal request...")
        
        logger.info("Directly instantiating and calling SystemMasterPlanner...")
        system_planner = SystemMasterPlanner(llm_provider=llm_provider)
        
        master_planner_input = MasterPlannerInput(
            user_goal=user_goal.goal_description,
            original_request=user_goal
        )

        master_plan_output: MasterPlannerOutput = await system_planner.invoke_async(inputs=master_planner_input)
        
        if not master_plan_output or not master_plan_output.master_plan_json:
            logger.error(f"Failed to generate a valid master plan. Output: {master_plan_output}")
            return

        master_plan_dict = json.loads(master_plan_output.master_plan_json)
        master_plan_obj = MasterExecutionPlan.model_validate(master_plan_dict)

        logger.info(f"Master plan generated directly: {master_plan_obj.id}")
        
        engine_for_state = ChungoidEngine(
            project_directory=str(temp_build_dir),
            llm_provider=llm_provider,
            agent_provider=agent_provider
        )
        engine_for_state.state_manager.save_master_execution_plan(master_plan_obj)
        logger.info("Master plan saved to state manager in temp directory.")

        logger.info("Build process initiated (plan generation).")
        logger.info(f"Inspect the temporary directory for any created files: {temp_build_dir}")
        logger.info("To run the full flow, a more complex setup involving an orchestrator would be needed, "
                    "or adapting flow_executor.py to target this temp directory and plan.")

    except Exception as e:
        logger.exception(f"An error occurred during the temporary build process: {e}")
    finally:
        logger.info(f"Temporary build directory {temp_build_dir} was NOT automatically cleaned up for inspection.")

if __name__ == "__main__":
    asyncio.run(main()) 