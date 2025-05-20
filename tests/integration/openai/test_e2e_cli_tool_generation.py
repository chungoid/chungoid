import pytest
import uuid
import json
import os
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from datetime import datetime
import stat
import sys
import logging
import subprocess
import re # Added for regex parsing of CLI output

# Core Utilities
from chungoid.utils.llm_provider import LLMManager, OpenAILLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.state_manager import StateManager
from chungoid.utils.logger_setup import setup_logging
from chungoid.schemas.project_status_schema import (
    ProjectOverallStatus,
    CycleStatus,
    ProjectStateV2
)
# from chungoid.schemas.common_types import AgentOutputStatus, Artifact, ArtifactType # Try removing these
from chungoid.schemas.arca_request_and_response import ARCAReviewArtifactType
from chungoid.schemas.code_debugging_agent_schemas import FailedTestReport
from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput, RetrieveArtifactOutput

# Agent Imports
from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import (
    AutomatedRefinementCoordinatorAgent_v1,
    ARCAReviewInput,
    ARCAOutput
)
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    EXECUTION_PLANS_COLLECTION,
    GENERATED_CODE_ARTIFACTS_COLLECTION,
    LIVE_CODEBASE_COLLECTION,
    PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION
)
from chungoid.runtime.agents.system_master_planner_agent import (
    MasterPlannerAgent,
    MasterPlannerInput,
    MasterPlannerOutput
)
from chungoid.runtime.agents.core_code_generator_agent import (
    CoreCodeGeneratorAgent_v1,
    SmartCodeGeneratorAgentInput,
    SmartCodeGeneratorAgentOutput
)
from chungoid.schemas.master_flow import MasterExecutionPlan

# Constants for test project/cycle
TEST_PROJECT_ID_E2E_CLI = f"test_e2e_cli_project_{uuid.uuid4().hex[:8]}"
TEST_CYCLE_ID_E2E_CLI_INITIAL = "cycle_001_e2e_cli_initial"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PROJECT_ID = "e2e_cli_tool_gen_project"

# Setup logging
setup_logging(level="DEBUG") # Configure root logger, use DEBUG for more test output
logger = logging.getLogger("test_e2e_line_counter_cli") # Get a specific logger instance

@pytest.fixture(scope="session")
def openai_api_key_e2e():
    if not OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not found in environment variables")
    return OPENAI_API_KEY

@pytest.fixture(scope="module")
def project_id_e2e():
    return f"test_e2e_cli_{uuid.uuid4().hex[:8]}"

@pytest.fixture(scope="module")
def temp_project_dir_e2e(project_id_e2e):
    base_dir = Path(tempfile.gettempdir()) / "chungoid_e2e_tests"
    project_dir = base_dir / project_id_e2e
    project_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created temporary project directory for E2E test: {project_dir}")
    yield project_dir
    # shutil.rmtree(project_dir) # Keep for inspection for now
    # logger.info(f"Removed temporary project directory: {project_dir}")

@pytest.fixture(scope="module")
def state_manager_e2e(temp_project_dir_e2e: Path):
    # Determine server_stages_dir relative to the test file's location
    # test_e2e_cli_tool_generation.py is in chungoid-core/tests/integration/openai/
    # server_prompts/stages/ is in chungoid-core/server_prompts/stages/
    current_file_path = Path(__file__).resolve()
    # chungoid-core/tests/integration/openai/ -> chungoid-core/tests/integration/ -> chungoid-core/tests/ -> chungoid-core/
    chungoid_core_root = current_file_path.parent.parent.parent.parent
    server_stages_path = chungoid_core_root / "server_prompts" / "stages"

    logger.info(f"StateManager using server_stages_dir: {server_stages_path}")
    assert server_stages_path.is_dir(), f"Server stages directory not found: {server_stages_path}"

    sm = StateManager(
        target_directory=str(temp_project_dir_e2e),
        server_stages_dir=str(server_stages_path)
    )
    # project_id is not passed to constructor, it's handled by initialize_project()
    return sm

@pytest.fixture(scope="module")
def llm_manager_e2e(openai_api_key_e2e, prompt_manager_e2e: PromptManager):
    provider = OpenAILLMProvider(api_key=openai_api_key_e2e, default_model="gpt-4o")
    manager = LLMManager(llm_provider_instance=provider, prompt_manager=prompt_manager_e2e)
    yield manager

@pytest.fixture(scope="module")
def prompt_manager_e2e():
    # Assuming prompts are in 'chungoid-core/server_prompts' relative to workspace root
    # Path(__file__) is chungoid-core/tests/integration/openai/test_e2e_cli_tool_generation.py
    # .parent.parent.parent.parent gives chungoid-core/
    prompt_dir = Path(__file__).resolve().parent.parent.parent.parent / "server_prompts"
    logger.info(f"PromptManager using prompt_directory_paths: {[str(prompt_dir)]}")
    assert prompt_dir.is_dir(), f"Prompt directory not found: {prompt_dir}"
    return PromptManager(prompt_directory_paths=[str(prompt_dir)]) # Corrected argument

@pytest.fixture(scope="function")
async def project_chroma_manager_agent_e2e(temp_project_dir_e2e: Path, project_id_e2e: str):
    # Ensure the parent directory for chroma_db for this project is managed by the agent's __init__ logic now.
    # The agent itself constructs paths like: 
    # self._project_root_workspace_path / self.DEFAULT_DB_SUBDIR / self.project_id
    # So, no need to create chroma_base_path here in the fixture beforehand.

    agent = ProjectChromaManagerAgent_v1(
        project_id=project_id_e2e,
        project_root_workspace_path=str(temp_project_dir_e2e), # Root for this specific test project
    )
    
    logger.info(f"PCMA Initialized. Attempting to initialize all project collections for {project_id_e2e}.")
    init_collections_success = await agent.initialize_project_collections()
    assert init_collections_success, "Failed to initialize project collections in PCMA."
    logger.info(f"PCMA project collections initialized successfully for {project_id_e2e}.")
    
    yield agent
    # Add any teardown for PCMA if necessary, e.g., clearing context
    # from chungoid.utils.chroma_utils import clear_chroma_project_context
    # clear_chroma_project_context() # If context is sticky per process/thread

# --- Agent Fixtures ---

@pytest.fixture(scope="function")
def master_planner_agent_e2e(
    llm_manager_e2e: LLMManager,
    prompt_manager_e2e: PromptManager,
    project_chroma_manager_agent_e2e: ProjectChromaManagerAgent_v1,
    state_manager_e2e: StateManager
):
    return MasterPlannerAgent(
        llm_provider=llm_manager_e2e._llm_provider,
        prompt_manager=prompt_manager_e2e,
        project_chroma_manager=project_chroma_manager_agent_e2e,
    )

@pytest.fixture(scope="function")
def code_generator_agent_e2e(
    llm_manager_e2e: LLMManager,
    prompt_manager_e2e: PromptManager,
    project_chroma_manager_agent_e2e: ProjectChromaManagerAgent_v1
):
    return CoreCodeGeneratorAgent_v1(
        llm_provider=llm_manager_e2e._llm_provider,
        prompt_manager=prompt_manager_e2e,
        pcma_agent=project_chroma_manager_agent_e2e
    )

@pytest.fixture(scope="function")
def arca_agent_e2e(
    llm_manager_e2e: LLMManager,
    prompt_manager_e2e: PromptManager,
    project_chroma_manager_agent_e2e: ProjectChromaManagerAgent_v1,
    state_manager_e2e: StateManager
):
    return AutomatedRefinementCoordinatorAgent_v1(
        llm_provider=llm_manager_e2e._llm_provider,
        prompt_manager=prompt_manager_e2e,
        project_chroma_manager=project_chroma_manager_agent_e2e,
        state_manager=state_manager_e2e
    )

# --- Test Case ---

@pytest.mark.integration
@pytest.mark.openai
async def test_e2e_generate_line_counter_cli(
    project_id_e2e: str,
    temp_project_dir_e2e: Path,
    state_manager_e2e: StateManager,
    master_planner_agent_e2e: MasterPlannerAgent,
    code_generator_agent_e2e: CoreCodeGeneratorAgent_v1,
    arca_agent_e2e: AutomatedRefinementCoordinatorAgent_v1,
    project_chroma_manager_agent_e2e: ProjectChromaManagerAgent_v1
):
    """
    End-to-end test to generate a simple CLI tool that counts lines in a file.
    Flow:
    1. Master Planner creates an initial plan.
    2. Code Generator attempts to write the code based on a task from the plan.
    3. (Optional/Iterative) ARCA reviews outputs and suggests refinements or new tasks.
    4. Verify the CLI tool is created and functions correctly.
    """
    logger.info(f"Starting E2E test for CLI tool generation in {temp_project_dir_e2e}")

    # 1. Initial Goal Setting and Project Initialization
    initial_goal = "Create a Python CLI tool that takes a file path as an argument and prints the number of lines in that file. The tool should be named 'linecounter.py'."
    
    # Initialize the project with StateManager
    try:
        initialized_state = state_manager_e2e.initialize_project(
            project_id=project_id_e2e,
            project_name="E2E CLI Test Project", # A descriptive name for the test project
            initial_user_goal_summary=initial_goal,
            initial_user_goal_doc_id=None # Assuming no pre-existing goal document
        )
        logger.info(f"StateManager initialized project: {initialized_state.project_id} with goal: {initialized_state.initial_user_goal_summary}")
    except Exception as e:
        logger.error(f"Failed to initialize project with StateManager: {e}", exc_info=True)
        pytest.fail(f"StateManager project initialization failed: {e}")

    current_state = state_manager_e2e.get_project_state() # Get the now initialized state
    assert current_state.project_id == project_id_e2e, "Project ID in state manager does not match test project ID after initialization."
    assert current_state.initial_user_goal_summary == initial_goal, "Project goal in state manager does not match test initial goal."

    # 2. Master Planner creates an initial plan
    logger.info("Invoking Master Planner...")
    planner_input = MasterPlannerInput(
        project_id=project_id_e2e,
        user_goal=initial_goal,
        project_context_artifacts=[],
    )
    try:
        planner_output: MasterPlannerOutput = await master_planner_agent_e2e.invoke_async(planner_input)
    except Exception as e:
        logger.error(f"Master Planner execution failed: {e}", exc_info=True)
        pytest.fail(f"Master Planner execution failed: {e}")

    assert planner_output.error_message is None, f"Master Planner failed: {planner_output.error_message}"
    assert planner_output.generated_plan_artifact_id is not None, "Master Planner did not return a generated_plan_artifact_id"
    
    logger.info(f"Master Planner created plan: {planner_output.generated_plan_artifact_id}")
    
    # --- Update Project State After Planning ---
    current_state: ProjectStateV2 = state_manager_e2e.get_project_state() # Refresh state
    current_state.latest_accepted_master_plan_doc_id = planner_output.generated_plan_artifact_id # Corrected field name
    
    # Start a new cycle for the code generation phase
    # The start_new_cycle method returns a CycleInfo object, which we are not actively managing further in this simplified test flow yet.
    new_cycle_objective = f"Initial code generation for CLI tool based on plan {planner_output.generated_plan_artifact_id}"
    current_cycle_info = current_state.start_new_cycle(
        cycle_id=f"cycle_{current_state.current_cycle_number:03d}_{uuid.uuid4().hex[:8]}",
        cycle_objective=new_cycle_objective
    )
    # start_new_cycle updates current_cycle_id, current_cycle_number, overall_status, and last_updated_utc internally.
    # It also sets the new cycle's status to IN_PROGRESS.
    logger.info(f"Started new cycle: ID={current_state.current_cycle_id}, Number={current_state.current_cycle_number}, Objective='{new_cycle_objective}'")
    logger.info(f"Project overall status: {current_state.overall_status}")

    # The active CycleInfo object (current_cycle_info) would be updated by an orchestrator.
    # For this test, we'll assume the cycle proceeds. When it completes (or if ARCA intervenes),
    # state_manager.complete_cycle() would be called with the updated CycleInfo.

    state_manager_e2e._write_status_file(current_state)
    # --- End State Update ---

    # Retrieve the actual plan content to find a coding task
    plan_artifact_response = await project_chroma_manager_agent_e2e.retrieve_artifact(
        base_collection_name=EXECUTION_PLANS_COLLECTION,
        document_id=planner_output.generated_plan_artifact_id
    )
    assert plan_artifact_response.status == "SUCCESS", f"Failed to retrieve master plan: {plan_artifact_response.error_message}"
    assert plan_artifact_response.content, "Retrieved master plan content is empty"
    
    master_plan_json_from_chroma = plan_artifact_response.content

    # Log what we got from Chroma
    logger.debug(f"Type of master_plan_json_from_chroma: {type(master_plan_json_from_chroma)}")
    logger.debug(f"Value of master_plan_json_from_chroma (first 500 chars): '{str(master_plan_json_from_chroma)[:500]}...'")

    # The plan_data should be a dictionary representing the MasterExecutionPlan
    plan_data: Dict[str, Any]

    # Extract a specific coding task from the master_plan_content.
    coding_task_id: Optional[str] = None
    coding_task_description: Optional[str] = None
    coding_task_inputs: Optional[Dict[str, Any]] = None
    target_file_name_from_plan: Optional[str] = None

    if "stages" in master_plan_json_from_chroma and isinstance(master_plan_json_from_chroma["stages"], dict):
        for stage_id, stage_spec in master_plan_json_from_chroma["stages"].items():
            if isinstance(stage_spec, dict):
                agent_id_in_plan = stage_spec.get("agent_id")
                # Check if this stage is for the CoreCodeGeneratorAgent
                if agent_id_in_plan == CoreCodeGeneratorAgent_v1.AGENT_ID:
                    coding_task_id = stage_id # The stage's key/id is the task_id
                    coding_task_description = stage_spec.get("description") or stage_spec.get("name")
                    coding_task_inputs = stage_spec.get("inputs")
                    
                    # Try to get target_file_path from plan inputs if available
                    if coding_task_inputs and "target_file_path" in coding_task_inputs:
                        target_file_name_from_plan = coding_task_inputs.get("target_file_path")
                    elif coding_task_inputs and "target_filename" in coding_task_inputs: # another common name
                        target_file_name_from_plan = coding_task_inputs.get("target_filename")

                    logger.info(f"Found coding stage: ID='{coding_task_id}', Description='{coding_task_description}'")
                    if coding_task_inputs:
                        logger.info(f"Coding stage inputs from plan: {coding_task_inputs}")
                    if target_file_name_from_plan:
                        logger.info(f"Target file name from plan: {target_file_name_from_plan}")
                    break # Found a suitable task
                # Fallback: check description if agent_id is not specific enough or missing
                elif not agent_id_in_plan and (
                    "generate code" in (stage_spec.get("description","") or "").lower() or \
                    "implement" in (stage_spec.get("description","") or "").lower() or \
                    "write code" in (stage_spec.get("description","") or "").lower() or \
                    "cli tool" in (stage_spec.get("description","") or "").lower() # Specific to our goal
                ):
                    coding_task_id = stage_id
                    coding_task_description = stage_spec.get("description") or stage_spec.get("name")
                    coding_task_inputs = stage_spec.get("inputs")
                    if coding_task_inputs and "target_file_path" in coding_task_inputs:
                        target_file_name_from_plan = coding_task_inputs.get("target_file_path")
                    elif coding_task_inputs and "target_filename" in coding_task_inputs: # another common name
                        target_file_name_from_plan = coding_task_inputs.get("target_filename")
                        
                    logger.info(f"Found potential coding stage by description: ID='{coding_task_id}', Description='{coding_task_description}'")
                    if coding_task_inputs:
                        logger.info(f"Coding stage inputs from plan: {coding_task_inputs}")
                    if target_file_name_from_plan:
                        logger.info(f"Target file name from plan: {target_file_name_from_plan}")
                    break # Found a suitable task
    
    if not coding_task_id or not coding_task_description:
        pytest.fail("Could not find a suitable coding task in the master plan for CoreCodeGeneratorAgent_v1 or by description.")

    logger.info(f"Selected coding task: '{coding_task_description}' (ID: {coding_task_id})")

    # 4. Code Generator attempts to write the code
    logger.info("Invoking Code Generator...")

    # Prefer target_file_name from plan if specified, otherwise use default for this test
    final_target_file_path = target_file_name_from_plan or "linecounter.py"

    # Store the coding_task_description as a specification artifact
    specification_artifact_id = f"spec_for_{coding_task_id}_{uuid.uuid4()}"
    store_spec_input = StoreArtifactInput(
        project_id=project_id_e2e,
        document_id=specification_artifact_id,
        artifact_content=coding_task_description,
        artifact_type="CodeGenerationSpecification",
        content_type="text/plain",
        base_collection_name=PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION,
        metadata={"source": "e2e_test_master_plan_task", "task_id": coding_task_id, "original_content_type": "text/plain"}
    )
    store_spec_output = await project_chroma_manager_agent_e2e.store_artifact(store_spec_input)
    assert store_spec_output and store_spec_output.status == "SUCCESS", \
        f"Failed to store code specification artifact: {store_spec_output.message if store_spec_output else 'Unknown error'}"
    logger.info(f"Stored code specification artifact with ID: {specification_artifact_id}")

    # Prepare input for the Code Generator
    code_gen_input = SmartCodeGeneratorAgentInput(
        task_id=f"codegen_task_{uuid.uuid4()}",
        project_id=project_id_e2e,
        code_specification_doc_id=specification_artifact_id, # Use the stored spec ID
        target_file_path=target_file_name_from_plan or "linecounter.py",
        programming_language="python",
        context_artifacts=[]
    )
    
    try:
        code_gen_output: SmartCodeGeneratorAgentOutput = await code_generator_agent_e2e.invoke_async(code_gen_input)
        logger.info(f"Code Generator finished. Status: {code_gen_output.status}")
        
        # Check for successful code generation
        assert code_gen_output and code_gen_output.status == "SUCCESS" and code_gen_output.generated_code_artifact_doc_id, \
            f"Code Generator failed: {code_gen_output.failure_reason if code_gen_output and hasattr(code_gen_output, 'failure_reason') else (code_gen_output.error_message if code_gen_output else 'None')}"
        
        logger.info(f"Code Generator produced artifact: {code_gen_output.generated_code_artifact_doc_id}")

        # 5. Retrieve the generated code artifact from Chroma
        generated_code_artifact_id = code_gen_output.generated_code_artifact_doc_id
        assert generated_code_artifact_id, "Code generator did not return an artifact ID."
        logger.info(f"Retrieving generated code artifact: {generated_code_artifact_id} from {code_gen_output.stored_in_collection}")

        # Retrieve the artifact content
        retrieved_code_artifact: RetrieveArtifactOutput = await project_chroma_manager_agent_e2e.retrieve_artifact(
            base_collection_name=code_gen_output.stored_in_collection or GENERATED_CODE_ARTIFACTS_COLLECTION, # Use the collection name from output if available
            document_id=generated_code_artifact_id
        )
        assert retrieved_code_artifact and retrieved_code_artifact.status == "SUCCESS" and retrieved_code_artifact.content, \
            f"Failed to retrieve generated code artifact: {retrieved_code_artifact.message if retrieved_code_artifact else 'None'}"
        
        generated_code_str = str(retrieved_code_artifact.content)
        logger.info(f"Successfully retrieved generated code. Length: {len(generated_code_str)}")
        # logger.debug(f"Retrieved generated code:\n{generated_code_str}")

        # 6. Save the generated code to a file
        cli_tool_filename = "linecounter.py" # As per original goal
        cli_tool_path = temp_project_dir_e2e / cli_tool_filename # Use the determined file path
        # Verify file was created by the agent (or save it if not)
        if not cli_tool_path.exists():
            with open(cli_tool_path, "w") as f:
                f.write(generated_code_str)
            logger.info(f"Manually saved generated code to {cli_tool_path}")
        
        # Make the script executable
        current_permissions = cli_tool_path.stat().st_mode
        cli_tool_path.chmod(current_permissions | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info(f"Made CLI tool {cli_tool_path} executable.")

        assert cli_tool_path.exists(), f"CLI tool {cli_tool_path} was not created."
        assert cli_tool_path.is_file()

        # 7. Test the generated CLI tool
        logger.info(f"Testing the generated CLI tool at {cli_tool_path}")
        
        # Create a dummy file for the CLI tool to count
        dummy_file_content = "Hello\nWorld\nThis is a test file."
        expected_line_count = len(dummy_file_content.splitlines())
        dummy_file_path = temp_project_dir_e2e / "test_input.txt"
        with open(dummy_file_path, "w") as f:
            f.write(dummy_file_content)

        try:
            # Run the generated CLI tool directly
            process = subprocess.run(
                [str(cli_tool_path), str(dummy_file_path)], 
                capture_output=True, text=True, check=False, timeout=30
            )
            output = process.stdout.strip()
            stderr_output = process.stderr.strip()
            logger.info(f"CLI tool stdout: {output}")
            if stderr_output:
                logger.warning(f"CLI tool stderr: {stderr_output}")

            if process.returncode != 0:
                logger.error(f"CLI tool failed with return code {process.returncode}")
                # ... (ARCA invocation logic remains here if needed) ...
                pytest.fail(f"CLI tool execution failed. STDERR: {stderr_output}\nSTDOUT: {output}")
            
            # --- Corrected Assertion Logic --- 
            # Extract the number from the output string
            # Assuming output format like "Number of lines: X" or "The file ... contains X lines."
            # Find all numbers in the string and assume the last one is the line count.
            all_numbers = re.findall(r"\d+", output)
            
            if all_numbers:
                extracted_number_str = all_numbers[-1] # Take the last number found
                assert extracted_number_str.isdigit(), f"Extracted part '{extracted_number_str}' from CLI output '{output}' is not a digit."
                actual_line_count = int(extracted_number_str)
                assert actual_line_count == expected_line_count, \
                    f"CLI tool output {actual_line_count} does not match expected {expected_line_count}. Full output: '{output}'"
                logger.info(f"CLI tool successfully counted {actual_line_count} lines, matching expected {expected_line_count}.")
            else:
                # If no digits found, try to see if the output itself is the number (e.g. if tool just prints "3")
                assert output.isdigit(), f"CLI output '{output}' does not contain a number and is not a digit itself."
                actual_line_count = int(output)
                assert actual_line_count == expected_line_count, \
                    f"CLI tool output {actual_line_count} (parsed as digit) does not match expected {expected_line_count}. Full output: '{output}'"
                logger.info(f"CLI tool successfully counted {actual_line_count} lines (direct digit), matching expected {expected_line_count}.")
            # --- End Corrected Assertion Logic ---

        except subprocess.TimeoutExpired:
            logger.error("CLI tool execution timed out.")
            pytest.fail("CLI tool execution timed out.")
        except Exception as e:
            logger.error(f"An unexpected error occurred while testing CLI tool: {e}", exc_info=True)
            pytest.fail(f"An unexpected error occurred while testing CLI tool: {e}")

    except Exception as e:
        logger.error(f"An unexpected error occurred while generating CLI tool: {e}", exc_info=True)
        pytest.fail(f"An unexpected error occurred while generating CLI tool: {e}")

    logger.info("E2E test for CLI tool generation completed successfully.")
    # Add more assertions as needed, e.g., checking for documentation, tests, etc.
    # if those were part of the initial goal. 