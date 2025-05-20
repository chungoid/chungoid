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

from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import (
    AutomatedRefinementCoordinatorAgent_v1,
    ARCAReviewInput,
    ARCAReviewArtifactType,
    ARCAOutput,
    ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME
)
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.project_state import ProjectStateV2, CycleHistoryItem
from chungoid.schemas.common import ConfidenceScore
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.utils.llm_provider import OpenAILLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.agent_registry_meta import AgentCategory

# Constants for test project/cycle
TEST_PROJECT_ID_ARCA_NEW_TASK = f"test_arca_new_task_project_{uuid.uuid4().hex[:8]}"
TEST_CYCLE_ID_ARCA_NEW_TASK = "cycle_001_new_task_test"
INITIAL_MASTER_PLAN_DOC_ID = f"initial_master_plan_{uuid.uuid4().hex[:8]}.yaml"
MOCK_OPTIMIZATION_REPORT_DOC_ID = f"mock_opt_report_{uuid.uuid4().hex[:8]}.json"

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

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
                agent_category=AgentCategory.NO_OP, # Example, adjust as per actual NoOpAgent registration
                # inputs={}, # MasterStageSpec might not have 'outputs' field directly
                next_stage=stage2_id
            ),
            stage2_id: MasterStageSpec(
                id=stage2_id,
                name="Dummy Work",
                description="A placeholder stage for work.",
                agent_category=AgentCategory.NO_OP, # Example
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
def project_chroma_manager_agent_arca_new_task(temp_test_dir_arca_new_task) -> ProjectChromaManagerAgent_v1:
    pcma_root_dir = temp_test_dir_arca_new_task / "pcma_data"
    pcma_root_dir.mkdir(parents=True, exist_ok=True)
    return ProjectChromaManagerAgent_v1(project_id=TEST_PROJECT_ID_ARCA_NEW_TASK, base_project_dir=str(pcma_root_dir))

@pytest.fixture(scope="function")
def state_manager_arca_new_task(temp_test_dir_arca_new_task) -> StateManager:
    dummy_server_stages_dir = temp_test_dir_arca_new_task / "dummy_server_stages"
    dummy_server_stages_dir.mkdir(exist_ok=True)
    return StateManager(target_directory=str(temp_test_dir_arca_new_task), server_stages_dir=str(dummy_server_stages_dir))

@pytest.fixture(scope="module")
def prompt_manager_arca_new_task() -> PromptManager:
    current_file_path = Path(__file__)
    core_project_root = current_file_path.parent.parent.parent.parent
    prompts_dir = core_project_root / "server_prompts"
    
    if not prompts_dir.is_dir():
        prompts_dir = Path("chungoid-core/server_prompts") 
        if not prompts_dir.is_dir():
             pytest.skip(f"Prompts directory not found at {core_project_root / 'server_prompts'} or {Path('chungoid-core/server_prompts')}. Skipping ARCA tests.")
    return PromptManager(str(prompts_dir))


@pytest.fixture(scope="module")
def openai_provider_arca_new_task(openai_api_key) -> OpenAILLMProvider:
    return OpenAILLMProvider(api_key=openai_api_key, default_model="gpt-4-turbo-preview")

@pytest.fixture(scope="function")
def arca_instance_new_task(
    project_chroma_manager_agent_arca_new_task: ProjectChromaManagerAgent_v1,
    state_manager_arca_new_task: StateManager,
    prompt_manager_arca_new_task: PromptManager,
    openai_provider_arca_new_task: OpenAILLMProvider
) -> AutomatedRefinementCoordinatorAgent_v1:
    return AutomatedRefinementCoordinatorAgent_v1(
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK,
        cycle_id=TEST_CYCLE_ID_ARCA_NEW_TASK,
        state_manager=state_manager_arca_new_task,
        project_chroma_manager=project_chroma_manager_agent_arca_new_task,
        prompt_manager=prompt_manager_arca_new_task,
        llm_provider=openai_provider_arca_new_task,
        optimization_evaluator_prompt_name=ARCA_OPTIMIZATION_EVALUATOR_PROMPT_NAME
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

@pytest.mark.integration_openai
async def test_arca_adds_new_task_to_plan_via_llm_eval(
    openai_api_key, 
    temp_test_dir_arca_new_task: Path, 
    initial_master_plan_path: Path, # This path points to a YAML consistent with master_flow.MasterExecutionPlan
    initial_master_plan_content: MasterExecutionPlan, # This is an instance of master_flow.MasterExecutionPlan
    project_chroma_manager_agent_arca_new_task: ProjectChromaManagerAgent_v1,
    state_manager_arca_new_task: StateManager,
    arca_instance_new_task: AutomatedRefinementCoordinatorAgent_v1,
    mock_optimization_report_for_new_task_content: dict
):
    # 1. Seed PCMA with the initial master plan
    initial_plan_doc_ref = await project_chroma_manager_agent_arca_new_task.store_document_async(
        document_id=INITIAL_MASTER_PLAN_DOC_ID,
        document_content=initial_master_plan_path.read_text(), # Reads the YAML created by initial_master_plan_content.to_yaml()
        document_type="master_execution_plan",
        metadata={"version": initial_master_plan_content.version}
    )
    assert initial_plan_doc_ref.document_id == INITIAL_MASTER_PLAN_DOC_ID

    # 2. Seed StateManager with initial state
    current_stage_id_before_arca = "stage_2_dummy_work" # This is a key in initial_master_plan_content.stages
    
    assert current_stage_id_before_arca in initial_master_plan_content.stages, \
        f"Initial current_stage_id {current_stage_id_before_arca} not found in plan stages dict."

    await state_manager_arca_new_task.initialize_project(
        project_id=TEST_PROJECT_ID_ARCA_NEW_TASK,
        project_name="ARCA New Task Test Project",
        initial_user_goal_summary="Test goal for ARCA new task generation."
    )
    current_state = await state_manager_arca_new_task.get_project_state()
    current_state.master_plan_doc_id = initial_plan_doc_ref.document_id
    current_state.current_stage_id = current_stage_id_before_arca 
    current_state.overall_status = "IN_PROGRESS" 
    current_state.cycle_history = [] 
    current_state.project_id = TEST_PROJECT_ID_ARCA_NEW_TASK
    current_active_cycle = CycleHistoryItem(
        cycle_id=TEST_CYCLE_ID_ARCA_NEW_TASK,
        objective="Initial cycle for ARCA new task test",
        start_time=datetime.now(timezone.utc),
        status="IN_PROGRESS"
    )
    current_state.cycle_history.append(current_active_cycle)
    current_state.current_cycle_id = TEST_CYCLE_ID_ARCA_NEW_TASK

    await state_manager_arca_new_task._write_status_file(current_state) 
    await state_manager_arca_new_task._load_or_initialize_project_state()

    # 3. "Upload" the mock optimization report to PCMA
    mock_report_str = json.dumps(mock_optimization_report_for_new_task_content)
    mock_report_doc_ref = await project_chroma_manager_agent_arca_new_task.store_document_async(
        document_id=MOCK_OPTIMIZATION_REPORT_DOC_ID,
        document_content=mock_report_str,
        document_type="optimization_suggestion_report",
        metadata={"source": "mock_test_agent"}
    )
    assert mock_report_doc_ref.document_id == MOCK_OPTIMIZATION_REPORT_DOC_ID

    # 4. Construct ARCAReviewInput
    # ARCAReviewInput uses artifact_doc_id for the suggestion report itself
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
        f"ARCA decision was {arca_output.decision}, expected PLAN_MODIFIED_NEW_TASKS_ADDED. LLM Output: {arca_output.evaluation_summary}"

    assert arca_output.new_master_plan_doc_id is not None, "new_master_plan_doc_id should be set for PLAN_MODIFIED_NEW_TASKS_ADDED"
    assert arca_output.new_master_plan_doc_id != initial_plan_doc_ref.document_id, \
        "New master plan ID should be different from the initial one."
    
    # The reviewed_artifact_doc_id in output should be the optimization report ARCA processed
    assert arca_output.reviewed_artifact_doc_id == mock_report_doc_ref.document_id, \
        f"ARCA output reviewed_artifact_doc_id ({arca_output.reviewed_artifact_doc_id}) does not match input optimization report id ({mock_report_doc_ref.document_id})."


    # 7. Verify the new master plan in PCMA
    new_plan_raw_content = await project_chroma_manager_agent_arca_new_task.retrieve_document_async(
        document_id=arca_output.new_master_plan_doc_id
    )
    assert new_plan_raw_content is not None, "New master plan content not found in PCMA"
    
    # The new plan should be parsable by MasterExecutionPlan.from_yaml()
    new_master_plan = MasterExecutionPlan.from_yaml(new_plan_raw_content.content)

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
    updated_state = await state_manager_arca_new_task.get_project_state()
    assert updated_state.master_plan_doc_id == arca_output.new_master_plan_doc_id, \
        "StateManager master_plan_doc_id was not updated."
    
    # current_stage_id in StateManager should be updated to the new stage ID
    assert updated_state.current_stage_id == newly_added_stage_details_as_spec.id, \
        f"StateManager current_stage_id should be '{newly_added_stage_details_as_spec.id}', but is '{updated_state.current_stage_id}'"

    print(f"Test test_arca_adds_new_task_to_plan_via_llm_eval completed successfully.")

# Placeholder for the original test if needed, or can be removed.
# @pytest.mark.skip(reason="Test implementation pending details and fixtures.")
# @pytest.mark.integration_openai 
# async def test_arca_adds_new_task_to_plan_via_llm_eval_placeholder():
#    assert True 