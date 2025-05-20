import pytest
import uuid
import json
import os
import yaml
import tempfile
import shutil
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import Optional
from unittest.mock import patch, AsyncMock
import inspect

from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import (
    AutomatedRefinementCoordinatorAgent_v1,
    ARCAReviewInput,
    ARCAReviewArtifactType,
    ARCAOutput,
    ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME
)
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    StoreArtifactInput,
    EXECUTION_PLANS_COLLECTION,
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
    ARTIFACT_TYPE_MASTER_EXECUTION_PLAN_YAML
)
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.project_status_schema import ProjectStateV2, ProjectOverallStatus
from chungoid.schemas.common import ConfidenceScore
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.utils.llm_provider import OpenAILLMProvider, LLMManager
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.agent_registry_meta import AgentCategory

# Constants for test project/cycle
TEST_PROJECT_ID_ARCA_NEW_TASK = f"test_arca_new_task_project_{uuid.uuid4().hex[:8]}"
TEST_CYCLE_ID_ARCA_NEW_TASK = "cycle_001_new_task_test"
INITIAL_MASTER_PLAN_DOC_ID = f"initial_master_plan_{uuid.uuid4().hex[:8]}.yaml"
MOCK_OPTIMIZATION_REPORT_DOC_ID = f"mock_opt_report_{uuid.uuid4().hex[:8]}.json"

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

print(f"DEBUG: ProjectChromaManagerAgent_v1 imported from module: {ProjectChromaManagerAgent_v1.__module__} which is file: {inspect.getfile(ProjectChromaManagerAgent_v1)}")

@pytest.fixture(scope="module")
def openai_api_key():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        pytest.skip("OPENAI_API_KEY not found in environment variables.")
    return key

@pytest.fixture(scope="function")
def temp_test_dir_arca_new_task():
    temp_dir = Path(tempfile.gettempdir()) / f"arca_new_task_test_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture(scope="function")
def temp_log_dir(temp_test_dir_arca_new_task: Path) -> Path:
    log_dir = temp_test_dir_arca_new_task / "test_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

@pytest.fixture(scope="function")
def initial_master_plan_content() -> MasterExecutionPlan:
    """Defines the content of an initial MasterExecutionPlan using MasterStageSpec."""
    stage1_id = "stage_1_setup"
    stage2_id = "stage_2_dummy_work"

    return MasterExecutionPlan(
        id=f"test_plan_{uuid.uuid4().hex[:8]}",
        name="Test Initial Master Plan for ARCA New Task",
        version="1.0",
        start_stage=stage1_id,
        stages={
            stage1_id: MasterStageSpec(
                id=stage1_id, # id field is part of MasterStageSpec
                name="Project Setup",
                description="Initial project setup and configuration.",
                agent_category=AgentCategory.TESTING_MOCK, # Changed from NO_OP
                # inputs={}, # MasterStageSpec might not have 'outputs' field directly
                next_stage=stage2_id
            ),
            stage2_id: MasterStageSpec(
                id=stage2_id,
                name="Dummy Work",
                description="A placeholder stage for work.",
                agent_category=AgentCategory.TESTING_MOCK, # Changed from NO_OP
                inputs={"input_A": "value_A"},
                # dependencies=[stage1_id], # MasterStageSpec does not have 'dependencies' field. Links are via next_stage.
                next_stage=None # End of this initial plan path
            )
        },
        # llm_provider_config field is not in MasterExecutionPlan from master_flow.py
        # If needed, it should be added to the MasterExecutionPlan Pydantic model.
        # For now, omitting it to match the schema.
        # llm_provider_config=LLMProviderConfig(provider_type=LLMProviderType.OPENAI, model_name="gpt-4-turbo-preview")
    )

@pytest.fixture(scope="function")
def initial_master_plan_path(temp_test_dir_arca_new_task, initial_master_plan_content: MasterExecutionPlan) -> Path:
    plan_path = temp_test_dir_arca_new_task / INITIAL_MASTER_PLAN_DOC_ID
    # MasterExecutionPlan has a to_yaml method
    yaml_content = initial_master_plan_content.to_yaml()
    with open(plan_path, "w") as f:
        f.write(yaml_content)
    return plan_path

@pytest.fixture(scope="function")
def project_chroma_manager_agent_arca_new_task(temp_test_dir_arca_new_task: Path) -> ProjectChromaManagerAgent_v1:
    # The ProjectChromaManagerAgent_v1 expects the root of the entire workspace,
    # and it will create project-specific subdirectories within that.
    # temp_test_dir_arca_new_task serves as a temporary, isolated workspace root for this test.
    return ProjectChromaManagerAgent_v1(
        project_root_workspace_path=temp_test_dir_arca_new_task, 
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK
    )

@pytest.fixture(scope="function")
def state_manager_arca_new_task(temp_test_dir_arca_new_task) -> StateManager:
    dummy_server_stages_dir = temp_test_dir_arca_new_task / "dummy_server_stages"
    dummy_server_stages_dir.mkdir(exist_ok=True)
    return StateManager(target_directory=str(temp_test_dir_arca_new_task), server_stages_dir=str(dummy_server_stages_dir))

@pytest.fixture(scope="module")
def prompt_manager_arca_new_task() -> PromptManager:
    # current_file_path = Path(__file__)
    # Adjust path to be relative to the workspace root for consistency
    # Assuming the test is run from the workspace root (chungoid-mcp)
    # prompts_dir = Path("chungoid-core/server_prompts")
    # Make path relative to this test file, going up to chungoid-core/server_prompts
    prompts_dir = Path(__file__).resolve().parent.parent.parent.parent / "server_prompts"
    # Ensure the directory exists for the test, though PromptManager should handle non-existent gracefully with logging
    # prompts_dir.mkdir(parents=True, exist_ok=True) # Not strictly needed if PM handles it
    return PromptManager([str(prompts_dir)]) # Pass as a list of strings

@pytest.fixture(scope="module")
def llm_manager_for_arca(
    openai_api_key, # Ensures API key is present or test is skipped
    prompt_manager_arca_new_task: PromptManager
) -> LLMManager:
    """Provides the LLMManager configured with OpenAILLMProvider for ARCA."""
    # OpenAILLMProvider is designed to load the API key from env vars if not provided,
    # or can take it as an argument. Here we explicitly pass it to ensure test isolation if needed,
    # though OpenAILLMProvider itself might still fall back to env vars if api_key is None.
    underlying_provider = OpenAILLMProvider(api_key=openai_api_key)
    return LLMManager(
        llm_provider_instance=underlying_provider,
        prompt_manager=prompt_manager_arca_new_task
    )

@pytest.fixture(scope="function")
def arca_instance_new_task(
    project_chroma_manager_agent_arca_new_task: ProjectChromaManagerAgent_v1,
    state_manager_arca_new_task: StateManager,
    prompt_manager_arca_new_task: PromptManager, # ARCA still needs its own PromptManager for other things
    llm_manager_for_arca: LLMManager # UPDATED PARAMETER NAME AND TYPE
) -> AutomatedRefinementCoordinatorAgent_v1:
    return AutomatedRefinementCoordinatorAgent_v1(
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK,
        cycle_id=TEST_CYCLE_ID_ARCA_NEW_TASK,
        state_manager=state_manager_arca_new_task,
        project_chroma_manager=project_chroma_manager_agent_arca_new_task,
        prompt_manager=prompt_manager_arca_new_task, # ARCA's own prompt manager
        llm_provider=llm_manager_for_arca, # UPDATED ARGUMENT
        optimization_evaluator_prompt_name=ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME
        # Note: The OpenAILLMProvider fixture is no longer directly used by ARCA's instantiation
        # but other parts of the test or other tests might still use it if it was designed for more general OpenAI calls.
        # For this specific ARCA test, llm_manager_for_arca is the correct one.
    )

@pytest.fixture(scope="function")
def mock_optimization_report_for_new_task_content() -> dict:
    # This mock report needs to suggest adding a new stage.
    # The 'details' should align with what MasterStageSpec expects if ARCA creates one.
    return {
        "report_id": f"opt_report_{uuid.uuid4().hex[:8]}",
        "target_artifact_type": "REQUIREMENTS_DOCUMENT", # Could be 'MasterExecutionPlan' if reviewing the plan itself
        "target_artifact_reference": { # For ARCA's input, this is artifact_doc_id
            "document_id": "some_requirements_doc.md", # Or INITIAL_MASTER_PLAN_DOC_ID if reviewing plan
            "version": "1.0"
        },
        "suggestions": [
            {
                "suggestion_id": "sugg_new_task_001",
                "description": "The current plan is missing a critical security audit stage. "
                               "This stage should involve static analysis, dynamic analysis, and penetration testing. "
                               "It must be performed after 'stage_2_dummy_work' and before any deployment.",
                "category": "Missing Feature",
                "severity": "CRITICAL",
                "confidence": 0.95,
                "suggested_action_type": "ADD_NEW_CAPABILITY", # This should guide LLM
                "details": { # These details should map to MasterStageSpec fields
                    "name": "Comprehensive Security Audit", # For MasterStageSpec.name
                    "description": "Conduct thorough static and dynamic analysis, and penetration testing to ensure application robustness.", # For MasterStageSpec.description
                    "agent_category": "SECURITY_TOOLING", # Example, for MasterStageSpec.agent_category
                    "inputs": {"scan_level": "deep"},
                    # "dependencies": ["stage_2_dummy_work"], # Not directly in MasterStageSpec, linking is via next_stage
                    "placement_hint": "after_stage:stage_2_dummy_work", # ARCA's prompt for evaluator needs to parse this
                                                                      # to set up next_stage relationships.
                    "estimated_effort": "Significant" # Informational for LLM
                },
                "justification": "Without this stage, the product could be vulnerable to major security threats, leading to data breaches and loss of trust."
            }
        ]
    }

@pytest.fixture(scope="function")
def mock_openai_chat_completion_v2_eval_standard(
    llm_manager_for_arca: LLMManager, # UPDATED PARAMETER TO USE THE NEW FIXTURE
    mock_optimization_report_for_new_task_content: dict
):
    """
    Mocks the LLM response for the ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME
    to simulate a successful evaluation that recommends adding a new task.
    """
    suggestion_details = mock_optimization_report_for_new_task_content["suggestions"][0]

    mock_llm_response_content = {
        "evaluated_optimizations": [
            {
                "optimization_id": suggestion_details["suggestion_id"],
                "source_report": "mock_test_agent", # Matching the source in the mock report
                "original_suggestion_summary": suggestion_details["description"],
                "is_actionable_and_relevant": True,
                "assessment_rationale": "The suggested security audit is critical and relevant.",
                "potential_impact_assessment": "Positive: Improved security. Negative: Time/resource cost.",
                "recommendation": "NEW_TASK_FOR_PLAN", # Key for this test
                "incorporation_instructions_for_next_agent": None,
                "clarification_query_for_generator": None,
                "new_task_details_for_plan": { # Details for ARCA to create a new MasterStageSpec
                    "task_description": suggestion_details["details"]["description"],
                    "suggested_agent_id_or_category": suggestion_details["details"]["agent_category"],
                    "placement_hint": suggestion_details["details"]["placement_hint"],
                    # "dependency_hint": suggestion_details["details"].get("dependencies", []), # Assuming dependencies might be in details
                    "initial_inputs": suggestion_details["details"].get("inputs", {}),
                    "success_criteria_suggestions": ["Security audit completed and report generated."],
                    # "output_context_path_suggestion": "stage_outputs.security_audit.report_doc_id" # Optional
                },
                "confidence_in_recommendation": 0.98
            }
        ],
        "overall_summary_of_actions": "Recommend adding a new security audit task to the plan."
    }
    mock_llm_json_response_str = json.dumps(mock_llm_response_content)

    # ARCA calls llm_provider.instruct_direct_async
    # We need to patch this method on the *instance* of LLMManager that ARCA uses.
    # The llm_manager_for_arca fixture provides this instance.
    
    # Use a context manager for patching if this fixture is function-scoped
    # If it were module/session scoped and shared, direct patching might be okay,
    # but function scope is safer to avoid test interference.
    patcher = patch.object(llm_manager_for_arca, 'generate_text_async_with_prompt_manager', new_callable=AsyncMock)
    mocked_method = patcher.start()

    async def mock_llm_call_side_effect(*args, **kwargs):
        # Check if the prompt_name and prompt_version match the one ARCA uses for optimization evaluation
        if (kwargs.get('prompt_name') == ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME and
            kwargs.get('prompt_version') == "v1"):
            return mock_llm_json_response_str
        # Fallback for other prompts
        return json.dumps({
            "error": "Mock not configured for this prompt/version combination",
            "received_prompt_name": kwargs.get('prompt_name'),
            "received_prompt_version": kwargs.get('prompt_version')
        })

    mocked_method.side_effect = mock_llm_call_side_effect
    
    yield mocked_method # The test doesn't strictly need the mock, but can be useful for assertions

    patcher.stop()

@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.skipif_llm_providers_not_configured
async def test_arca_adds_new_task_to_plan_via_llm_eval(
    event_loop,  # Pytest fixture for asyncio
    mock_openai_chat_completion_v2_eval_standard, # Simulate LLM accepting the task
    project_chroma_manager_agent_arca_new_task: ProjectChromaManagerAgent_v1,
    temp_test_dir_arca_new_task: Path, # Using this existing fixture instead
    temp_log_dir, # Fixture for temporary logging directory
    openai_api_key, 
    initial_master_plan_path: Path, # This path points to a YAML consistent with master_flow.MasterExecutionPlan
    initial_master_plan_content: MasterExecutionPlan, # This is an instance of master_flow.MasterExecutionPlan
    state_manager_arca_new_task: StateManager,
    arca_instance_new_task: AutomatedRefinementCoordinatorAgent_v1,
    mock_optimization_report_for_new_task_content: dict
):
    """
    Test that ARCA can successfully add a new task to an existing plan
    by using an LLM evaluation that approves the task.
    """
    import chromadb
    import sys
    print("\\n--- PYTEST ChromaDB Library Diagnosis ---")
    print(f"PYTEST ChromaDB version: {getattr(chromadb, '__version__', 'N/A')}")
    print(f"PYTEST ChromaDB library loaded from: {getattr(chromadb, '__file__', 'N/A')}")
    print(f"PYTEST PersistentClient class exists: {hasattr(chromadb, 'PersistentClient')}")
    print("\\n--- PYTEST sys.path ---")
    for p in sys.path:
        print(p)
    print("\\n--- PYTEST Diagnosis Complete ---\\n")

    # 1. Seed PCMA with the initial master plan
    store_plan_input = StoreArtifactInput(
        base_collection_name=EXECUTION_PLANS_COLLECTION,
        document_id=INITIAL_MASTER_PLAN_DOC_ID,
        artifact_content=initial_master_plan_path.read_text(),
        metadata={
            "version": initial_master_plan_content.version, 
            "project_id": TEST_PROJECT_ID_ARCA_NEW_TASK,
            "artifact_type": ARTIFACT_TYPE_MASTER_EXECUTION_PLAN_YAML # CORRECTED METADATA
        }
    )
    initial_plan_doc_ref = await project_chroma_manager_agent_arca_new_task.store_artifact(args=store_plan_input)
    assert initial_plan_doc_ref.document_id == INITIAL_MASTER_PLAN_DOC_ID
    assert initial_plan_doc_ref.status == "SUCCESS", f"Failed to store initial plan: {initial_plan_doc_ref.error_message}"

    # 2. Seed StateManager with initial state
    current_stage_id_before_arca = "stage_2_dummy_work" # This is a key in initial_master_plan_content.stages
    
    assert current_stage_id_before_arca in initial_master_plan_content.stages, \
        f"Initial current_stage_id {current_stage_id_before_arca} not found in plan stages dict."

    state_manager_arca_new_task.initialize_project(
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK,
        project_name="ARCA New Task Test Project",
        initial_user_goal_summary="Test goal for ARCA new task generation."
    )
    current_state = state_manager_arca_new_task.get_project_state()
    print(f"DEBUG: type(current_state) = {type(current_state)}")
    if hasattr(current_state, 'model_fields'):
        print(f"DEBUG: current_state.model_fields.keys() = {current_state.model_fields.keys()}")
    elif hasattr(current_state, '__fields__'): # Fallback for Pydantic V1 just in case
        print(f"DEBUG: current_state.__fields__.keys() = {current_state.__fields__.keys()}")
    else:
        print("DEBUG: current_state has neither model_fields nor __fields__")
    current_state.latest_accepted_master_plan_doc_id = initial_plan_doc_ref.document_id
    # current_state.current_stage_id = current_stage_id_before_arca # Field does not exist on ProjectStateV2 from project_status_schema
    current_state.overall_status = ProjectOverallStatus.REFINEMENT_CYCLE_IN_PROGRESS # CORRECTED ENUM MEMBER
    current_state.historical_cycles = [] # Correct field name for ProjectStateV2 from project_status_schema

    # ARCA needs a current cycle to operate on. StateManager.initialize_project might set up an initial cycle_0, 
    # or it might need to be explicitly started. For this test, let's assume initialize_project sets current_cycle_number = 0
    # and we might need to set current_cycle_id if ARCA expects it.
    # The schema ProjectStateV2 has current_cycle_id and current_cycle_number.
    # initialize_project in StateManager sets current_project_state with default ProjectStateV2 which has current_cycle_number = 0
    # but current_cycle_id is None. ARCA might need a current_cycle_id.
    current_state.current_cycle_id = "cycle_0_initial_setup_for_arca_test"
    current_state.current_cycle_number = 0 # Matches default from ProjectStateV2 and initialize_project

    state_manager_arca_new_task._write_status_file(current_state) # Persist these initial settings using the private write method

    # 3. "Upload" the mock optimization report to PCMA
    mock_report_str = json.dumps(mock_optimization_report_for_new_task_content)
    store_report_input = StoreArtifactInput(
        base_collection_name=OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION,
        document_id=MOCK_OPTIMIZATION_REPORT_DOC_ID,
        artifact_content=mock_report_str, # Content is a string (JSON string)
        metadata={
            "source": "mock_test_agent", 
            "project_id": TEST_PROJECT_ID_ARCA_NEW_TASK,
            "artifact_type": ARCAReviewArtifactType.OPTIMIZATION_SUGGESTION_REPORT.value # CORRECTED METADATA
        }
    )
    mock_report_doc_ref = await project_chroma_manager_agent_arca_new_task.store_artifact(args=store_report_input)
    assert mock_report_doc_ref.document_id == MOCK_OPTIMIZATION_REPORT_DOC_ID
    assert mock_report_doc_ref.status == "SUCCESS", f"Failed to store mock report: {mock_report_doc_ref.error_message}"

    # 4. Construct ARCAReviewInput
    # ARCAReviewInput uses artifact_doc_id for the suggestion report itself

    # <START DIAGNOSTIC PRINT>
    print(f"DIAGNOSTIC: ARCAReviewArtifactType module: {ARCAReviewArtifactType.__module__}")
    print(f"DIAGNOSTIC: ARCAReviewArtifactType members: {[member.value for member in ARCAReviewArtifactType]}")
    # <END DIAGNOSTIC PRINT>

    arca_input = ARCAReviewInput(
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK,
        cycle_id=TEST_CYCLE_ID_ARCA_NEW_TASK,
        artifact_type=ARCAReviewArtifactType.OPTIMIZATION_SUGGESTION_REPORT,
        artifact_doc_id=mock_report_doc_ref.document_id, 
        generator_agent_id="mock_suggestion_generator_agent", # Agent that generated the suggestion report
        # generator_agent_confidence is for the artifact being reviewed, not the suggestion report itself.
        # If OPTIMIZATION_SUGGESTION_REPORT is the artifact, its confidence might not be relevant here.
        # Let's assume it's not strictly needed or ARCA handles its absence for this artifact_type.
    )

    # 5. Invoke ARCA
    print(f"Invoking ARCA for project {TEST_PROJECT_ID_ARCA_NEW_TASK}, cycle {TEST_CYCLE_ID_ARCA_NEW_TASK}...")
    arca_output: ARCAOutput = await arca_instance_new_task.invoke_async(arca_input)
    print(f"ARCA output: {arca_output.model_dump_json(indent=2)}")

    # 6. Assertions
    assert arca_output is not None
    assert arca_output.decision == "PLAN_MODIFIED_NEW_TASKS_ADDED", \
        f"ARCA decision was {arca_output.decision}, expected PLAN_MODIFIED_NEW_TASKS_ADDED. Reasoning: {arca_output.reasoning}"

    assert arca_output.new_master_plan_doc_id is not None, "new_master_plan_doc_id should be set for PLAN_MODIFIED_NEW_TASKS_ADDED"
    assert arca_output.new_master_plan_doc_id != initial_plan_doc_ref.document_id, \
        "New master plan ID should be different from the initial one."
    
    # The reviewed_artifact_doc_id in output should be the optimization report ARCA processed
    assert arca_output.reviewed_artifact_doc_id == mock_report_doc_ref.document_id, \
        f"ARCA output reviewed_artifact_doc_id ({arca_output.reviewed_artifact_doc_id}) does not match input optimization report id ({mock_report_doc_ref.document_id})."


    # 7. Verify the new master plan in PCMA
    # retrieve_artifact needs base_collection_name and document_id
    new_plan_doc_output = await project_chroma_manager_agent_arca_new_task.retrieve_artifact(
        base_collection_name=EXECUTION_PLANS_COLLECTION, # Specify the correct collection
        document_id=arca_output.new_master_plan_doc_id
    )
    assert new_plan_doc_output is not None, "New master plan RetrieveArtifactOutput is None"
    assert new_plan_doc_output.status == "SUCCESS", f"Failed to retrieve new master plan: {new_plan_doc_output.error_message}"
    assert new_plan_doc_output.content is not None, "New master plan content not found in PCMA via retrieve_artifact"
    
    # The new plan should be parsable by MasterExecutionPlan.from_yaml()
    # new_plan_doc_output.content should be the YAML string
    new_master_plan = MasterExecutionPlan.from_yaml(new_plan_doc_output.content)

    initial_total_stages = len(initial_master_plan_content.stages)
    new_total_stages = len(new_master_plan.stages)
    
    assert new_total_stages == initial_total_stages + 1, \
        f"Expected {initial_total_stages + 1} stages in the new plan's stages dict, found {new_total_stages}."

    # Find the newly added stage
    newly_added_stage_details_as_spec: Optional[MasterStageSpec] = None
    for stage_id, stage_spec in new_master_plan.stages.items():
        if stage_id not in initial_master_plan_content.stages:
            newly_added_stage_details_as_spec = stage_spec
            break
    
    assert newly_added_stage_details_as_spec is not None, "Could not find the newly added stage in the new master plan's stages dictionary."
    print(f"Newly added stage details: {newly_added_stage_details_as_spec.model_dump_json(indent=2)}")

    # Check characteristics of the new MasterStageSpec
    # These assertions depend on how ARCA's LLM prompt for arca_optimization_evaluator_v1_prompt.yaml
    # translates the suggestion into new_task_details, and how _apply_new_tasks_to_parsed_plan uses them.
    # Assuming new_task_details from LLM directly map to MasterStageSpec fields.
    assert "audit" in newly_added_stage_details_as_spec.name.lower() or \
           "security" in newly_added_stage_details_as_spec.name.lower(), \
        "New stage name doesn't seem to reflect 'security' or 'audit'."
    assert "static analysis" in newly_added_stage_details_as_spec.description.lower() or \
           "penetration testing" in newly_added_stage_details_as_spec.description.lower(), \
        "New stage description doesn't seem to reflect security testing activities."
    
    # Verify linking: The mock report suggests placement after "stage_2_dummy_work".
    # ARCA's logic in _apply_new_tasks_to_parsed_plan (if it were to work with this flat structure)
    # would need to update stage_2_dummy_work.next_stage to point to the new stage's ID,
    # and the new stage's next_stage would point to what stage_2_dummy_work originally pointed to.
    
    # For this test, let's check if stage_2_dummy_work now points to the new stage
    predecessor_stage_in_new_plan = new_master_plan.stages.get(current_stage_id_before_arca)
    assert predecessor_stage_in_new_plan is not None
    assert predecessor_stage_in_new_plan.next_stage == newly_added_stage_details_as_spec.id, \
        f"Predecessor stage '{current_stage_id_before_arca}' should now link to new stage '{newly_added_stage_details_as_spec.id}', but links to '{predecessor_stage_in_new_plan.next_stage}'."

    # And the new stage should link to what stage_2_dummy_work previously linked to (None in this case)
    # This depends on ARCA correctly preserving the chain.
    original_next_of_predecessor = initial_master_plan_content.stages[current_stage_id_before_arca].next_stage
    assert newly_added_stage_details_as_spec.next_stage == original_next_of_predecessor, \
        f"New stage '{newly_added_stage_details_as_spec.id}' should link to original next stage '{original_next_of_predecessor}', but links to '{newly_added_stage_details_as_spec.next_stage}'."


    # 8. Verify StateManager update
    updated_state = state_manager_arca_new_task.get_project_state()
    assert updated_state.latest_accepted_master_plan_doc_id == arca_output.new_master_plan_doc_id, \
        "StateManager latest_accepted_master_plan_doc_id was not updated."
    
    # current_stage_id related assertions are commented out as the field doesn't exist directly on ProjectStateV2
    # assert updated_state.current_stage_id == arca_output.next_stage_id, \
    #     "StateManager current_stage_id was not updated to the new stage ID."
    
    # Verify that a new cycle may have been added to historical_cycles or current_cycle_id updated
    # This depends on ARCA's exact behavior with StateManager, which is TBD from this test's perspective
    # For now, we are primarily testing the plan modification part.

    # 9. Verify the new plan content (Optional but good)
    # Also, looking at the schema, ProjectStateV2 (from project_status_schema) has 'historical_cycles', not 'cycle_history'.
    # Let's assume the intent was to clear historical cycles for this test setup.
    current_state.historical_cycles = []

    print(f"Test test_arca_adds_new_task_to_plan_via_llm_eval completed successfully.")

# Placeholder for the original test if needed, or can be removed.
# @pytest.mark.skip(reason="Test implementation pending details and fixtures.")
# @pytest.mark.integration_openai 
# async def test_arca_adds_new_task_to_plan_via_llm_eval_placeholder():
#    assert True 