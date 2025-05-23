import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import logging
from pathlib import Path
from typing import Optional, Dict, Any

from chungoid.runtime.orchestrator import AsyncOrchestrator, NEXT_STAGE_END_SUCCESS, NEXT_STAGE_END_FAILURE
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, MasterStageFailurePolicy
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType, ModifyMasterPlanDetails
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus, OnFailureAction
from chungoid.schemas.metrics import MetricEventType # Corrected import
from chungoid.utils.agent_resolver import AgentProvider, RegistryAgentProvider
from chungoid.utils.state_manager import StateManager # For type hinting
from chungoid.utils.metrics_store import MetricsStore # For type hinting
from chungoid.schemas.user_goal_schemas import UserGoalRequest # CORRECTED
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1 # MODIFIED
from chungoid.schemas.orchestration import SharedContext # ADDED for full_context typing
from chungoid.schemas.project_state import RunRecord, StageRecord, ProjectStateV2
from chungoid.schemas.flows import PausedRunDetails


# --- Mock Classes ---

class MockMasterPlannerReviewerAgent:
    def __init__(self, output_to_return: Optional[MasterPlannerReviewerOutput] = None, raise_exception: Optional[Exception] = None):
        self.output_to_return = output_to_return
        self.raise_exception = raise_exception
        
        async def internal_side_effect_func(inputs: MasterPlannerReviewerInput, full_context: Optional[SharedContext] = None):
            if self.raise_exception:
                raise self.raise_exception
            if self.output_to_return is not None:
                return self.output_to_return
            return MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
                reasoning="Default mock response (output_to_return was None) from internal_side_effect_func"
            )
        
        self._internal_async_mock = AsyncMock(side_effect=internal_side_effect_func)

    async def __call__(self, inputs: MasterPlannerReviewerInput, full_context: Optional[SharedContext] = None):
        return await self._internal_async_mock(inputs=inputs, full_context=full_context)

    # Helper to access call_count of the internal mock for assertions
    @property
    def call_count(self):
        return self._internal_async_mock.call_count

    @property
    def call_args_list(self):
        return self._internal_async_mock.call_args_list

class MockStageAgent:
    def __init__(self, agent_id: str, output_to_return: any = None, raise_exception: Optional[AgentErrorDetails] = None, call_count_reset: bool = True):
        self.agent_id = agent_id
        self.output_to_return = output_to_return
        self.raise_exception_on_invoke = raise_exception
        self.async_call_mock = AsyncMock(side_effect=self._callable_side_effect)
        if call_count_reset:
            self.reset_call_count()

    async def _callable_side_effect(self, inputs: dict, full_context: Optional[SharedContext] = None):
        if self.raise_exception_on_invoke:
            if isinstance(self.raise_exception_on_invoke, list):
                if not self.raise_exception_on_invoke:
                    pass 
                else:
                    exc_to_raise = self.raise_exception_on_invoke.pop(0)
                    if exc_to_raise:
                        raise exc_to_raise
            elif self.raise_exception_on_invoke:
                 raise self.raise_exception_on_invoke

        if isinstance(self.output_to_return, list):
            if not self.output_to_return:
                return None 
            return self.output_to_return.pop(0)
        return self.output_to_return


    async def __call__(self, inputs: dict, full_context: Optional[SharedContext] = None):
        return await self.async_call_mock(inputs, full_context=full_context)

    @property
    def call_count(self):
        return self.async_call_mock.call_count
        
    def reset_call_count(self):
        self.async_call_mock.reset_mock()


# --- Pytest Fixtures ---

@pytest.fixture
def mock_agent_provider():
    provider = MagicMock(spec=AgentProvider)
    # provider.get = MagicMock() # We will set this per test or with a default mock
    # provider.resolve_agent_by_category = AsyncMock()
    provider.get_project_chroma_manager = MagicMock(return_value=MagicMock(spec=ProjectChromaManagerAgent_v1))
    return provider

@pytest.fixture
def mock_state_manager():
    manager = MagicMock(spec=StateManager)
    manager.save_paused_flow_state = AsyncMock()
    manager.get_or_create_current_run_id = MagicMock(return_value="test_run_id_123")
    manager.update_status = MagicMock()
    manager.load_master_execution_plan = AsyncMock()
    manager.get_project_id = MagicMock(return_value="test_project_id")
    manager.get_project_root_path = MagicMock(return_value=Path("/tmp/test_project_root"))
    manager.record_stage_run = AsyncMock()
    manager.record_stage_end = AsyncMock()

    # Added missing mocks based on orchestrator usage
    manager.create_run_record = AsyncMock()
    manager.update_run_record_status = AsyncMock()
    manager.load_master_plan = AsyncMock()  # Already used in some tests, ensuring it's here
    manager.get_master_plan_by_id = AsyncMock()
    manager.save_paused_run = AsyncMock()
    manager.load_context_snapshot = AsyncMock()
    manager.get_run_record = AsyncMock()
    manager.record_stage_start = AsyncMock()
    manager.update_run_status = AsyncMock()
    return manager

@pytest.fixture
def mock_metrics_store():
    store = MagicMock(spec=MetricsStore)
    store.add_event = MagicMock()
    return store

@pytest.fixture
def orchestrator(mock_agent_provider, mock_state_manager, mock_metrics_store):
    # Default config, can be overridden in tests
    config = {
        "project_name": "test_project", 
        "project_id": "test_project_id",
        "project_root_path": "/tmp/test_project_root",
        "master_planner_reviewer_agent_id": "system.mock_reviewer_agent", # MOVED to config
        "default_on_failure_action": OnFailureAction.INVOKE_REVIEWER.value # ADDED
    }
    return AsyncOrchestrator(
        config=config,
        agent_provider=mock_agent_provider,
        state_manager=mock_state_manager,
        metrics_store=mock_metrics_store
        # master_planner_reviewer_agent_id parameter removed here
    )

# --- Test Plan Helper ---
def create_simple_plan(
    plan_id: str = "test_plan_reviewer", 
    stage1_id: str = "stage1", 
    stage1_name: str = "Test Stage 1",
    stage1_agent_id: str = "mock_agent_s1",
    stage1_inputs: Optional[Dict[str, Any]] = None,
    stage1_on_failure: Optional[MasterStageFailurePolicy] = None
) -> MasterExecutionPlan:
    return MasterExecutionPlan(
        id=plan_id,
        name="Test Plan for Reviewer",
        original_request=UserGoalRequest(goal_description="Test goal"),
        start_stage=stage1_id,
        stages={
            stage1_id: MasterStageSpec(
                id=stage1_id,
                name=stage1_name,
                agent_id=stage1_agent_id,
                inputs=stage1_inputs if stage1_inputs is not None else {"data": "initial"},
                on_failure=stage1_on_failure if stage1_on_failure is not None else MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER)
            )
        }
    )

# --- Test Cases ---

@pytest.mark.asyncio
async def test_reviewer_suggests_retry_stage_as_is(
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock
):
    """
    Tests Scenario 1.1: Reviewer suggests RETRY_STAGE_AS_IS.
    Orchestrator should retry the failed stage.
    """
    failing_agent_id = "agent_fails_then_succeeds"
    
    # Agent will fail first time, succeed second time
    mock_failing_agent = MockStageAgent(
        agent_id=failing_agent_id,
        raise_exception=[AgentErrorDetails(error_type="TestError", message="First call fails"), None], # Fail first, then no error
        output_to_return=[None, {"result": "success_on_retry"}] # Output for second call
    )

    # Reviewer agent setup
    reviewer_output = MasterPlannerReviewerOutput(
        suggestion_type=ReviewerActionType.RETRY_STAGE_AS_IS,
        suggestion_details={"target_stage_id": "stage1"},
        reasoning="Suggesting retry as is for test."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider to return mocks
    def agent_provider_get_side_effect(agent_id_requested, shared_context: Optional[SharedContext] = None):
        if agent_id_requested == failing_agent_id:
            return mock_failing_agent # Instance
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer # Instance
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get_agent_callable = AsyncMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=failing_agent_id)
    initial_context = {"global_data": "test"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_execution_plan = AsyncMock(return_value=plan)

    # Execute by passing master_plan_id
    final_context = await orchestrator.run(master_plan_or_path=plan)

    # Assertions
    # 1. Failing agent was called twice (initial fail, then retry)
    assert mock_failing_agent.call_count == 2, "Failing agent should have been called twice"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.call_count == 1, "Reviewer should be called once"
    
    # 3. Flow should not have paused (no call to save_paused_flow_state)
    mock_state_manager.save_paused_flow_state.assert_not_called()
    
    # 4. Final context should indicate success (no _flow_error) and contain retry output
    assert "_flow_error" not in final_context, "Flow should not end in error"
    assert final_context.get("outputs", {}).get("stage1", {}).get("result") == "success_on_retry"
    
    # 5. Verify context passed to reviewer
    reviewer_call_args = mock_reviewer.call_args_list[0][0] # (inputs_arg, full_context_arg)
    reviewer_input_arg: MasterPlannerReviewerInput = reviewer_call_args[0]
    assert isinstance(reviewer_input_arg.triggering_error_details, AgentErrorDetails)
    assert reviewer_input_arg.triggering_error_details.message == "First call fails"
    assert reviewer_input_arg.paused_stage_id == "stage1"

    # 6. StateManager should record initial failure then final success for the stage
    # This requires checking calls to mock_state_manager.update_status
    # First call (failure)
    update_status_calls = mock_state_manager.update_status.call_args_list
    assert len(update_status_calls) >= 2 # At least one for fail, one for success
    
    # Check initial failure status update
    # Note: The exact number and order of update_status calls can be complex due to emit_metric and direct updates.
    # We are looking for a call that records the failure of stage1.
    failure_status_call = next((call for call in update_status_calls if call.kwargs.get('stage') == plan.stages["stage1"].number and call.kwargs.get('status') == StageStatus.FAILURE.value), None)
    assert failure_status_call is not None, "Stage failure status not updated"
    assert failure_status_call.kwargs['error_details'].message == "First call fails"

    # Check final success status update for stage1
    success_status_call = next((call for call in update_status_calls if call.kwargs.get('stage') == plan.stages["stage1"].number and call.kwargs.get('status') == StageStatus.SUCCESS.value), None)
    assert success_status_call is not None, "Stage success status not updated after retry"

@pytest.mark.asyncio
async def test_reviewer_suggests_retry_with_valid_new_inputs(
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock
):
    """
    Tests Scenario 2.1: Reviewer suggests RETRY_STAGE_WITH_CHANGES 
    and provides valid new_inputs.
    Orchestrator should update stage inputs and retry successfully.
    """
    stage_agent_id = "agent_needs_new_inputs"
    original_inputs = {"param": "original_value"}
    corrected_inputs = {"param": "corrected_value", "new_param": True}
    
    # Stage agent will fail with original_inputs, succeed with corrected_inputs
    # For simplicity, let's make it check the input directly in its mock logic
    async def stage_agent_callable(inputs: dict):
        mock_stage_agent.async_call_mock(inputs) # Track call for count and args
        if inputs == corrected_inputs:
            return {"result": "success_with_corrected_inputs"}
        else:
            # Simulate an error if inputs are not the corrected ones
            raise AgentErrorDetails(error_type="InputValidationError", message="Inputs were not corrected")

    mock_stage_agent = MockStageAgent(agent_id=stage_agent_id) # Main mock for call_count
    # We assign the custom callable to the side_effect of the internal AsyncMock
    mock_stage_agent.async_call_mock.side_effect = stage_agent_callable
    
    # Reviewer agent setup
    reviewer_output = MasterPlannerReviewerOutput(
        suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES,
        suggestion_details={
            "target_stage_id": "stage1",
            "changes_to_stage_spec": corrected_inputs,
            "modification_needed": "Original inputs were incorrect."
        },
        reasoning="Suggesting retry with corrected inputs."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider
    def agent_provider_get_side_effect(agent_id_requested, shared_context: Optional[SharedContext] = None):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent # INSTANCE
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer # INSTANCE
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get_agent_callable = AsyncMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_retry_modified"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_execution_plan = AsyncMock(return_value=plan)

    # Execute
    final_context = await orchestrator.run(master_plan_or_path=plan)

    # Assertions
    # 1. Stage agent was called twice
    assert mock_stage_agent.call_count == 2, "Stage agent should be called twice (fail then success)"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.call_count == 1, "Reviewer should be called once"
    
    # 3. Flow should not have paused
    mock_state_manager.save_paused_flow_state.assert_not_called()
    
    # 4. Final context should indicate success and contain output from corrected run
    assert "_flow_error" not in final_context, "Flow should not end in error after retry"
    assert final_context.get("outputs", {}).get("stage1", {}).get("result") == "success_with_corrected_inputs"
    
    # 5. Verify inputs to the stage agent on the second call were the corrected_inputs
    second_call_args = mock_stage_agent.async_call_mock.call_args_list[1][0][0] # args[0] of first arg of call
    assert second_call_args == corrected_inputs, "Stage agent not called with corrected inputs on retry"

    # 6. Verify the plan's stage inputs were modified *in the orchestrator's current_plan instance* 
    #    (this is an internal check of the orchestrator's behavior for this run)
    assert orchestrator.current_plan.stages["stage1"].inputs == corrected_inputs, "Plan inputs not updated in orchestrator"

    # 7. StateManager status updates (similar to previous test)
    update_status_calls = mock_state_manager.update_status.call_args_list
    failure_status_call = next((call for call in update_status_calls if call.kwargs.get('stage') == plan.stages["stage1"].number and call.kwargs.get('status') == StageStatus.FAILURE.value), None)
    assert failure_status_call is not None, "Stage failure status not updated for initial call"
    assert failure_status_call.kwargs['error_details'].message == "Inputs were not corrected"

    success_status_call = next((call for call in update_status_calls if call.kwargs.get('stage') == plan.stages["stage1"].number and call.kwargs.get('status') == StageStatus.SUCCESS.value), None)
    assert success_status_call is not None, "Stage success status not updated after retry with new inputs"

@pytest.mark.parametrize(
    "reviewer_suggestion_details_variant, expected_log_warning_fragment",
    [
        pytest.param(
            {
                "target_stage_id": "stage1",
                "changes_to_stage_spec": {},
                "modification_needed": "Inputs require user correction."
            },
            "'new_inputs' were missing",
            id="missing_new_inputs"
        ),
        pytest.param(
            {
                "target_stage_id": "stage1",
                "changes_to_stage_spec": {"__this_is_not_valid_spec__": True},
                "modification_needed": "Inputs require user correction."
            },
            "'new_inputs' were missing, malformed, or for the wrong stage", # Covers malformed
            id="malformed_new_inputs_not_a_dict"
        ),
        pytest.param(
            {
                "target_stage_id": "wrong_stage", # MISMATCHED target_stage_id
                "changes_to_stage_spec": {"param": "some_value"},
                "modification_needed": "Inputs require user correction."
            },
            "'new_inputs' were missing, malformed, or for the wrong stage", # Covers wrong stage
            id="mismatched_target_stage_id"
        ),
    ]
)
@pytest.mark.asyncio
async def test_reviewer_retry_modified_input_pause_variations(
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock,
    mock_metrics_store: MagicMock,
    reviewer_suggestion_details_variant: dict,
    expected_log_warning_fragment: str,
    caplog: pytest.LogCaptureFixture
):
    """
    Tests Scenarios 2.2, 2.3, 2.4: Reviewer suggests RETRY_STAGE_WITH_CHANGES 
    but new_inputs are missing, malformed, or for the wrong stage.
    Orchestrator should pause for user input in all these cases.
    """
    stage_agent_id = "agent_for_pause_variations"
    original_inputs = {"param": "original_value"}
    
    mock_stage_agent = MockStageAgent(
        agent_id=stage_agent_id,
        raise_exception=AgentErrorDetails(error_type="InputValidationError", message="Original inputs failed as expected for pause test")
    )
    
    # Reviewer agent setup based on parametrized variant
    reviewer_output = MasterPlannerReviewerOutput(
        suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES,
        suggestion_details=reviewer_suggestion_details_variant,
        reasoning="Suggesting retry, but user intervention needed for inputs."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider
    def agent_provider_get_side_effect(agent_id_requested, shared_context: Optional[SharedContext] = None):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent # INSTANCE
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer # INSTANCE
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get_agent_callable = AsyncMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_retry_pause_variations"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_execution_plan = AsyncMock(return_value=plan)
    
    caplog.clear() # Clear previous logs for this test
    with caplog.at_level(logging.WARNING, logger="chungoid.runtime.orchestrator"):
        final_context = await orchestrator.run(master_plan_or_path=plan)

    # Assertions (mostly same as the original test_reviewer_retry_modified_input_missing_new_inputs)
    # 1. Stage agent was called once
    assert mock_stage_agent.call_count == 1, "Stage agent should only be called once before pause"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.call_count == 1, "Reviewer agent should have been called once"
    
    # 3. Flow should have paused
    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == orchestrator.state_manager.get_or_create_current_run_id()
    assert pause_details_call.pause_status == FlowPauseStatus.USER_INTERVENTION_REQUIRED
    assert pause_details_call.clarification_request["type"] == "INPUT_MODIFICATION_REQUIRED"
    assert pause_details_call.clarification_request["reviewer_analysis"] == reviewer_suggestion_details_variant.get("modification_needed", "No specific analysis provided by reviewer.")
    assert pause_details_call.clarification_request["original_stage_inputs"] == original_inputs
    
    # 4. Check for expected log warning (covers why it paused)
    assert any(expected_log_warning_fragment in record.message for record in caplog.records), \
        f"Expected log fragment '{expected_log_warning_fragment}' not found in orchestrator warnings."

    # 5. Final context returned by orchestrator should indicate pause
    assert final_context.get("_autonomous_flow_paused_state_saved") is True
    assert "_flow_error" in final_context

    # 6. Metrics store should have a FLOW_PAUSED event
    flow_paused_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.FLOW_PAUSED), None)
    assert flow_paused_metric_call is not None, "FLOW_PAUSED metric not emitted"
    assert flow_paused_metric_call[0][0].data["reason"] == "Awaiting user input for RETRY_STAGE_WITH_CHANGES"
    assert flow_paused_metric_call[0][0].data["stage_id"] == "stage1"

# Note: Ensuring a blank line above this decorator for proper separation.
@pytest.mark.parametrize(
    "action_type, suggestion_details, expected_pause_status, expected_clarification_type, expected_metric_reason_fragment",
    [
        pytest.param(
            ReviewerActionType.MODIFY_MASTER_PLAN,
            lambda mock_reviewer_cls: mock_reviewer_cls(output_to_return=MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.MODIFY_MASTER_PLAN,
                suggestion_details=ModifyMasterPlanDetails( 
                    new_plan_description="Plan needs changes for stage1.", 
                    new_plan_content="Replace agent X with Y" # Example content
                ),
                reasoning="Suggesting plan modification by test"
            )),
            FlowPauseStatus.USER_INTERVENTION_REQUIRED,
            "PLAN_MODIFICATION_REVIEW",
            "Reviewer suggested MODIFY_MASTER_PLAN",
            id="MODIFY_MASTER_PLAN_escalation"
        ),
        pytest.param(
            ReviewerActionType.ESCALATE_TO_USER,
            {"message_to_user": "User, please check stage1 output."},
            FlowPauseStatus.USER_INTERVENTION_REQUIRED,
            "USER_ESCALATION",
            "Reviewer suggested ESCALATE_TO_USER",
            id="ESCALATE_TO_USER_direct"
        ),
        pytest.param(
            ReviewerActionType.NO_ACTION_SUGGESTED,
            {},
            FlowPauseStatus.USER_INTERVENTION_REQUIRED, # Should pause for user if reviewer has no idea
            "NO_REVIEWER_SUGGESTION",
            "Reviewer provided NO_ACTION_SUGGESTED",
            id="NO_ACTION_SUGGESTED_escalation"
        ),
    ]
)
@pytest.mark.asyncio
async def test_reviewer_escalation_scenarios(
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock,
    mock_metrics_store: MagicMock,
    action_type: ReviewerActionType,
    suggestion_details: dict,
    expected_pause_status: FlowPauseStatus,
    expected_clarification_type: str,
    expected_metric_reason_fragment: str
):
    """
    Tests escalation scenarios: MODIFY_MASTER_PLAN (P2F.1 simple escalation), 
    ESCALATE_TO_USER, and NO_ACTION_SUGGESTED.
    All should pause the flow with appropriate details.
    """
    stage_agent_id = "agent_for_escalation_tests"
    original_inputs = {"data": "escalation_test_input"}
    error_message = "Escalation test: initial agent failure."

    mock_stage_agent = MockStageAgent(
        agent_id=stage_agent_id,
        raise_exception=AgentErrorDetails(error_type="TestEscalationError", message=error_message)
    )
    
    reviewer_output = MasterPlannerReviewerOutput(
        suggestion_type=action_type,
        suggestion_details=suggestion_details,
        reasoning=f"Test reasoning for {action_type.value}"
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent # INSTANCE
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer # INSTANCE
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get_agent_callable = AsyncMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_escalations"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_execution_plan = AsyncMock(return_value=plan)

    final_context = await orchestrator.run(master_plan_or_path=plan)

    # Assertions
    assert mock_stage_agent.call_count == 1
    assert mock_reviewer.call_count == 1
    
    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == orchestrator.state_manager.get_or_create_current_run_id()
    assert pause_details_call.flow_id == plan.id
    assert pause_details_call.paused_at_stage_id == "stage1"
    assert pause_details_call.pause_status == expected_pause_status
    assert pause_details_call.triggering_error_details.message == error_message
    
    clarification_req = pause_details_call.clarification_request
    assert clarification_req["type"] == expected_clarification_type
    assert clarification_req["reviewer_output"] == reviewer_output.model_dump()
    if "message_to_user" in suggestion_details:
        assert suggestion_details["message_to_user"] in clarification_req["message"]
    elif action_type == ReviewerActionType.NO_ACTION_SUGGESTED:
        assert f"Reviewer was invoked for stage 'stage1' but provided no specific recovery action." in clarification_req["message"]

    assert final_context.get("_autonomous_flow_paused_state_saved") is True
    assert "_flow_error" in final_context
    assert final_context["_flow_error"]["reviewer_suggestion_type"] == action_type.value

    flow_paused_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.FLOW_PAUSED), None)
    assert flow_paused_metric_call is not None, "FLOW_PAUSED metric not emitted"
    assert expected_metric_reason_fragment in flow_paused_metric_call[0][0].data["reason"]
    assert flow_paused_metric_call[0][0].data["stage_id"] == "stage1"

@pytest.mark.asyncio
async def test_reviewer_suggests_proceed_as_is(
    mock_state_manager: MagicMock, 
    mock_metrics_store: MagicMock
):
    """
    Tests Scenario 5.1: Reviewer suggests PROCEED_AS_IS after a stage failure.
    Orchestrator should mark stage1 as COMPLETED_WITH_WARNINGS and proceed to stage2.
    """
    failing_agent_id = "agent_always_fails_for_proceed_as_is"
    reviewer_agent_id = "system.mock_reviewer_agent_proceed_as_is"

    # Config for this specific test orchestrator
    config_for_test = {
        "project_name": "test_proceed_as_is",
        "project_id": "test_project_id_proceed",
        "project_root_path": "/tmp/test_project_proceed",
        "master_planner_reviewer_agent_id": reviewer_agent_id, # Use specific ID for this test
        "default_on_failure_action": OnFailureAction.INVOKE_REVIEWER.value
    }

    # Mock for the agent that will always fail
    mock_failing_agent = MockStageAgent(
        agent_id=failing_agent_id,
        raise_exception=AgentErrorDetails(error_type="RecoverableError", message="Stage1 failed but reviewer said proceed.")
    )

    # Initialize orchestrator directly for this test for more control over agent provider
    isolated_agent_provider = MagicMock(spec=AgentProvider)
    isolated_agent_provider.get_agent_callable = AsyncMock(side_effect=lambda agent_id_requested, shared_context: mock_failing_agent)

    # Initialize orchestrator instance for this specific test run
    isolated_orchestrator = AsyncOrchestrator(
        config=config_for_test, # USE UPDATED CONFIG
        agent_provider=isolated_agent_provider,
        state_manager=mock_state_manager,
        metrics_store=mock_metrics_store
        # master_planner_reviewer_agent_id parameter removed here
    )

    plan = create_simple_plan(stage1_agent_id=failing_agent_id)
    initial_context = {"global_data": "test_proceed"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_plan = AsyncMock(return_value=plan)

    # Execute
    final_context = await isolated_orchestrator.run(master_plan_or_path=plan)

    # Assertions
    # 1. Stage 1 agent was called once (and failed)
    assert mock_failing_agent.call_count == 1
    
    # 2. Flow should NOT have paused
    mock_state_manager.save_paused_flow_state.assert_not_called()
    
    # 3. Final context should indicate overall success (as stage2 completed)
    #    and contain outputs from both stages.
    assert "_flow_error" not in final_context, "Flow should not end in overall error"
    # Check stage1 output in the shared_context.outputs collection
    assert isolated_orchestrator.shared_context.outputs.get("stage1", {}).get("output_from_failed_stage") == "some_partial_data", "Stage 1 output not found in shared_context.outputs"
    # Check stage2 output in the shared_context.previous_stage_outputs
    # final_context is the SharedContext object, its previous_stage_outputs field holds outputs keyed by stage name.
    assert isolated_orchestrator.shared_context.previous_stage_outputs.get("stage2", {}).get("result_stage2") == "stage2_success", \
        f"Stage 2 output not found. previous_stage_outputs: {isolated_orchestrator.shared_context.previous_stage_outputs}"

    # 4. StateManager status updates: stage1 COMPLETED_WITH_WARNINGS, stage2 SUCCESS
    # mock_state_manager.record_stage_run is called for stage1 with COMPLETED_WITH_WARNINGS
    stage1_run_call = None
    stage1_end_call = None
    stage2_end_call = None

    for call_args in mock_state_manager.record_stage_run.call_args_list:
        if call_args.kwargs.get('stage_id') == "stage1":
            stage1_run_call = call_args
            break
    
    for call_args in mock_state_manager.record_stage_end.call_args_list:
        if call_args.kwargs.get('stage_name') == "stage1":
            stage1_end_call = call_args
        elif call_args.kwargs.get('stage_name') == "stage2":
            stage2_end_call = call_args
        if stage1_end_call and stage2_end_call:
            break

    assert stage1_run_call is not None, "StateManager.record_stage_run not called for stage1 initial attempt"
    assert stage1_run_call.kwargs.get('status') == StageStatus.COMPLETED_FAILURE, "Stage1 initial attempt should be COMPLETED_FAILURE"
    assert stage1_run_call.kwargs.get('error_details')['message'] == "Stage1 failed but reviewer said proceed."

    assert stage1_end_call is not None, "StateManager.record_stage_end not called for stage1 with COMPLETED_WITH_WARNINGS"
    assert stage1_end_call.kwargs.get('status') == StageStatus.COMPLETED_WITH_WARNINGS, "Stage1 final status should be COMPLETED_WITH_WARNINGS"
    assert stage1_end_call.kwargs.get('output') == {"output_from_failed_stage": "some_partial_data"}
    assert stage1_end_call.kwargs.get('error_details').message == "Stage1 failed but reviewer said proceed."

    assert stage2_end_call is not None, "StateManager.record_stage_end not called for stage2"

    # 5. Metrics: MASTER_STAGE_END for stage1 with COMPLETED_WITH_WARNINGS, MASTER_STAGE_END for stage2 with COMPLETED_SUCCESS
    stage1_warning_metric_call = next((
        call for call in mock_metrics_store.add_event.call_args_list 
        if call[0][0].event_type == MetricEventType.MASTER_STAGE_END and 
           call[0][0].stage_id == "stage1" and 
           call[0][0].data.get("status") == StageStatus.COMPLETED_WITH_WARNINGS.value
    ), None)
    assert stage1_warning_metric_call is not None, "MASTER_STAGE_END metric with COMPLETED_WITH_WARNINGS not emitted for stage1"
    assert stage1_warning_metric_call[0][0].data["original_error"]["message"] == "Stage1 failed but reviewer said proceed."
    assert stage1_warning_metric_call[0][0].data["output"] == {"output_from_failed_stage": "some_partial_data"}

    stage2_success_metric_call = next((
        call for call in mock_metrics_store.add_event.call_args_list 
        if call[0][0].event_type == MetricEventType.MASTER_STAGE_END and 
           call[0][0].stage_id == "stage2" and 
           call[0][0].data.get("status") == StageStatus.COMPLETED_SUCCESS.value
    ), None)
    assert stage2_success_metric_call is not None, "MASTER_STAGE_END metric with COMPLETED_SUCCESS not emitted for stage2"

@pytest.mark.parametrize(
    "test_id, reviewer_setup_action, orchestrator_reviewer_agent_id_override, expected_pause_status, expected_clarification_type, expected_metric_reason_fragment, expected_log_message_fragment",
    [
        pytest.param(
            "unhandled_suggestion_type",
            lambda mock_reviewer_cls: mock_reviewer_cls(output_to_return=MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
                reasoning="Testing unhandled type effectively becomes NO_ACTION_SUGGESTED path"
            )),
            "system.mock_reviewer_agent", # Default, reviewer is configured
            FlowPauseStatus.USER_INTERVENTION_REQUIRED, # Fallback pause
            "UNHANDLED_REVIEWER_SUGGESTION",
            "Reviewer agent failed or returned no actionable output",
            "Reviewer suggested 'NO_ACTION_SUGGESTED'",
            id="unhandled_reviewer_action_type"
        ),
        pytest.param(
            "reviewer_raises_exception",
            lambda mock_reviewer_cls: mock_reviewer_cls(raise_exception=ValueError("Reviewer mock failed!")),
            "system.mock_reviewer_agent",
            FlowPauseStatus.USER_INTERVENTION_REQUIRED, # Fallback pause due to reviewer error
            "NO_REVIEWER_SUGGESTION_HANDLER_DEFAULT_PAUSE", # Generic pause reason
            "Reviewer agent failed or returned no actionable output",
            "Error invoking reviewer agent: Reviewer mock failed!",
            id="reviewer_agent_fails_exception"
        ),
        pytest.param(
            "reviewer_returns_none",
            lambda mock_reviewer_cls: mock_reviewer_cls(output_to_return=None), # Explicitly return None
            "system.mock_reviewer_agent",
            FlowPauseStatus.USER_INTERVENTION_REQUIRED,
            "NO_REVIEWER_SUGGESTION_HANDLER_DEFAULT_PAUSE",
            "Reviewer agent failed or returned no actionable output",
            "Reviewer agent returned None or no actionable suggestion",
            id="reviewer_agent_returns_none"
        ),
        pytest.param(
            "reviewer_not_configured", # test_id
            None,                      # reviewer_setup_action
            None,                      # orchestrator_reviewer_agent_id_override
            FlowPauseStatus.USER_INTERVENTION_REQUIRED, # expected_pause_status
            "NO_REVIEWER_CONFIGURED_PAUSE", # expected_clarification_type
            "No reviewer agent configured, or reviewer failed", # expected_metric_reason_fragment
            "No master_planner_reviewer_agent_id configured", # expected_log_message_fragment
            id="reviewer_not_configured"
        )
    ]
)
@pytest.mark.asyncio
async def test_reviewer_fallback_and_error_scenarios(
    mock_state_manager: MagicMock,
    mock_metrics_store: MagicMock,
    caplog: pytest.LogCaptureFixture,
    test_id: str,
    reviewer_setup_action: callable, # Lambda to setup the mock_reviewer or None
    orchestrator_reviewer_agent_id_override: str, # To set orchestrator.config['master_planner_reviewer_agent_id']
    expected_pause_status: FlowPauseStatus,
    expected_clarification_type: str,
    expected_metric_reason_fragment: str,
    expected_log_message_fragment: str
):
    """
    Tests fallback scenarios: Unhandled suggestion, Reviewer failure/None, Reviewer not configured.
    All should pause the flow with appropriate details and logs.
    """
    failing_agent_id = f"failing_agent_for_{test_id}"
    
    config_for_fallback_test = {
        "project_name": f"test_fallback_{test_id}",
        "project_id": f"project_fallback_{test_id}",
        "project_root_path": f"/tmp/project_fallback_{test_id}",
        "master_planner_reviewer_agent_id": orchestrator_reviewer_agent_id_override, # Set from param
        "default_on_failure_action": OnFailureAction.INVOKE_REVIEWER.value # ADDED
    }

    # Mock for the agent that will always fail, triggering reviewer logic
    mock_failing_agent = MockStageAgent(
        agent_id=failing_agent_id,
        raise_exception=AgentErrorDetails(error_type="TestFallbackError", message="Fallback test: initial agent failure.")
    )

    # Create orchestrator instance for this specific test run
    isolated_orchestrator = AsyncOrchestrator(
        config=config_for_fallback_test, # USE UPDATED CONFIG
        agent_provider=MagicMock(spec=AgentProvider),
        state_manager=mock_state_manager,
        metrics_store=mock_metrics_store
        # master_planner_reviewer_agent_id removed here
    )

    # Create a simple plan that will use the failing agent
    plan = create_simple_plan(stage1_agent_id=failing_agent_id)
    plan.stages["stage1"].on_failure = MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER) # MODIFIED ensure reviewer is called if configured
    initial_context = {"global_data": "test_fallbacks"}

    # Mock state_manager to return the plan by ID
    mock_state_manager.load_master_plan = AsyncMock(return_value=plan)

    caplog.clear()
    log_level_to_capture = logging.ERROR if "reviewer_raises_exception" in test_id or "reviewer_not_configured" in test_id or "reviewer_returns_none" in test_id else logging.WARNING
    
    with caplog.at_level(log_level_to_capture, logger="chungoid.runtime.orchestrator"):
        final_context = await isolated_orchestrator.run(master_plan_or_path=plan)
    
    # Assertions
    assert mock_failing_agent.call_count == 1

    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == isolated_orchestrator.state_manager.get_or_create_current_run_id()
    assert pause_details_call.pause_status == expected_pause_status
    assert pause_details_call.triggering_error_details.message == "Fallback test: initial agent failure."
    
    clarification_req = pause_details_call.clarification_request
    assert clarification_req["type"] == expected_clarification_type
    if test_id == "unhandled_suggestion_type":
        assert clarification_req["reviewer_output"]["suggestion_type"] == "NO_ACTION_SUGGESTED"
    
    assert final_context.get("_autonomous_flow_paused_state_saved") is True
    assert "_flow_error" in final_context
    if "reviewer_suggestion_type" in final_context["_flow_error"]:
         assert final_context["_flow_error"]["reviewer_suggestion_type"] == ("NO_ACTION_SUGGESTED" if test_id == "unhandled_suggestion_type" else None) # Adjust if other types populate this

    # Check logs
    assert any(expected_log_message_fragment in record.message for record in caplog.records), \
        f"Expected log fragment '{expected_log_message_fragment}' not found. Logs: {[r.message for r in caplog.records]}"

    # Check metrics
    flow_paused_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.FLOW_PAUSED), None)
    assert flow_paused_metric_call is not None, "FLOW_PAUSED metric not emitted"
    assert expected_metric_reason_fragment in flow_paused_metric_call[0][0].data["reason"]
    assert flow_paused_metric_call[0][0].data["stage_id"] == "stage1"