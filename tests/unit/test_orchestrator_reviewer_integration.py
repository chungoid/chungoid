import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
import logging

from chungoid.runtime.orchestrator import AsyncOrchestrator
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, StageFailurePolicy
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.agent_master_planner_reviewer import MasterPlannerReviewerInput, MasterPlannerReviewerOutput, ReviewerActionType
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus # Assuming StageStatus is used by StateManager
from chungoid.schemas.metrics import MetricEventType # Corrected import
from chungoid.utils.agent_provider import AgentProvider # For type hinting
from chungoid.utils.state_manager import StateManager # For type hinting
from chungoid.utils.metrics_store import MetricsStore # For type hinting


# --- Mock Classes ---

class MockMasterPlannerReviewerAgent:
    def __init__(self, output_to_return: MasterPlannerReviewerOutput = None, raise_exception: Exception = None):
        self.output_to_return = output_to_return
        self.raise_exception = raise_exception
        self.invoke_async_mock = AsyncMock() # To track calls and args

    async def __call__(self, inputs: MasterPlannerReviewerInput, full_context: dict = None):
        self.invoke_async_mock(inputs, full_context)
        if self.raise_exception:
            raise self.raise_exception
        if self.output_to_return:
            return self.output_to_return
        # Default minimal response if nothing specific is set, to avoid None issues if not expected
        return MasterPlannerReviewerOutput(
            suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
            reasoning="Default mock response"
        )

class MockStageAgent:
    def __init__(self, agent_id: str, output_to_return: any = None, raise_exception: AgentErrorDetails = None, call_count_reset: bool = True):
        self.agent_id = agent_id
        self.output_to_return = output_to_return
        self.raise_exception = raise_exception
        self._call_count = 0
        # Use AsyncMock for the call itself to easily track calls and arguments
        self.async_call_mock = AsyncMock(side_effect=self._callable_side_effect)
        if call_count_reset: # For tests that re-invoke
             self.reset_call_count()

    async def _callable_side_effect(self, inputs: dict):
        self._call_count += 1
        if self.raise_exception:
            # If it's the first call and we need to raise, or if we always raise
            if isinstance(self.raise_exception, list): # Cycle through exceptions
                exc_to_raise = self.raise_exception[(self._call_count -1) % len(self.raise_exception)]
                if exc_to_raise: # if not None
                    raise exc_to_raise 
            elif self.raise_exception: # Single exception to raise
                 raise self.raise_exception
        
        if isinstance(self.output_to_return, list): # Cycle through outputs
            return self.output_to_return[(self._call_count -1) % len(self.output_to_return)]
        return self.output_to_return


    async def __call__(self, inputs: dict):
        return await self.async_call_mock(inputs)

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
    return provider

@pytest.fixture
def mock_state_manager():
    manager = MagicMock(spec=StateManager)
    manager.save_paused_flow_state = AsyncMock()
    manager.get_or_create_current_run_id = MagicMock(return_value="test_run_id_123")
    manager.update_status = MagicMock()
    return manager

@pytest.fixture
def mock_metrics_store():
    store = MagicMock(spec=MetricsStore)
    store.add_event = MagicMock()
    return store

@pytest.fixture
def orchestrator(mock_agent_provider, mock_state_manager, mock_metrics_store):
    # Default config, can be overridden in tests
    config = {"project_name": "test_project"} 
    return AsyncOrchestrator(
        config=config,
        agent_provider=mock_agent_provider,
        state_manager=mock_state_manager,
        metrics_store=mock_metrics_store,
        master_planner_reviewer_agent_id="system.mock_reviewer_agent" # Consistent ID for mocking
    )

# --- Test Plan Helper ---
def create_simple_plan(stage1_agent_id: str, stage1_inputs: dict = None) -> MasterExecutionPlan:
    return MasterExecutionPlan(
        id="test_plan_reviewer",
        name="Test Plan for Reviewer",
        original_request="Test goal",
        start_stage="stage1",
        stages={
            "stage1": MasterStageSpec(
                agent_id=stage1_agent_id,
                inputs=stage1_inputs if stage1_inputs else {"data": "initial"},
                on_failure=StageFailurePolicy(action="PAUSE_FOR_INTERVENTION") # Default, can override
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
    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == failing_agent_id:
            return mock_failing_agent
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=failing_agent_id)
    initial_context = {"global_data": "test"}

    # Execute
    final_context = await orchestrator.run(plan, initial_context)

    # Assertions
    # 1. Failing agent was called twice (initial fail, then retry)
    assert mock_failing_agent.call_count == 2, "Failing agent should have been called twice"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.invoke_async_mock.call_count == 1, "Reviewer agent should have been called once"
    
    # 3. Flow should not have paused (no call to save_paused_flow_state)
    mock_state_manager.save_paused_flow_state.assert_not_called()
    
    # 4. Final context should indicate success (no _flow_error) and contain retry output
    assert "_flow_error" not in final_context, "Flow should not end in error"
    assert final_context.get("outputs", {}).get("stage1", {}).get("result") == "success_on_retry"
    
    # 5. Verify context passed to reviewer
    reviewer_call_args = mock_reviewer.invoke_async_mock.call_args_list[0][0] # (inputs_arg, full_context_arg)
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
    Tests Scenario 2.1: Reviewer suggests RETRY_STAGE_WITH_MODIFIED_INPUT 
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
        suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_MODIFIED_INPUT,
        suggestion_details={
            "target_stage_id": "stage1",
            "new_inputs": corrected_inputs,
            "modification_needed": "Original inputs were incorrect."
        },
        reasoning="Suggesting retry with corrected inputs."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider
    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent # The callable itself is the mock_stage_agent instance
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_retry_modified"}

    # Execute
    final_context = await orchestrator.run(plan, initial_context)

    # Assertions
    # 1. Stage agent was called twice
    assert mock_stage_agent.call_count == 2, "Stage agent should be called twice (fail then success)"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.invoke_async_mock.call_count == 1, "Reviewer agent should be called once"
    
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
                # "new_inputs": {...}, // INTENTIONALLY MISSING
                "modification_needed": "Inputs require user correction."
            },
            "'new_inputs' were missing",
            id="missing_new_inputs"
        ),
        pytest.param(
            {
                "target_stage_id": "stage1",
                "new_inputs": "not_a_dict", # MALFORMED
                "modification_needed": "Inputs require user correction."
            },
            "'new_inputs' were missing, malformed, or for the wrong stage", # Covers malformed
            id="malformed_new_inputs_not_a_dict"
        ),
        pytest.param(
            {
                "target_stage_id": "wrong_stage", # MISMATCHED target_stage_id
                "new_inputs": {"param": "some_value"},
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
    Tests Scenarios 2.2, 2.3, 2.4: Reviewer suggests RETRY_STAGE_WITH_MODIFIED_INPUT 
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
        suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_MODIFIED_INPUT,
        suggestion_details=reviewer_suggestion_details_variant,
        reasoning="Suggesting retry, but user intervention needed for inputs."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider
    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_retry_pause_variations"}
    run_id = orchestrator.state_manager.get_or_create_current_run_id()
    
    caplog.clear() # Clear previous logs for this test
    with caplog.at_level(logging.WARNING, logger="chungoid.runtime.orchestrator"):
        final_context = await orchestrator.run(plan, initial_context)

    # Assertions (mostly same as the original test_reviewer_retry_modified_input_missing_new_inputs)
    # 1. Stage agent was called once
    assert mock_stage_agent.call_count == 1, "Stage agent should only be called once before pause"
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.invoke_async_mock.call_count == 1, "Reviewer agent should have been called once"
    
    # 3. Flow should have paused
    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == run_id
    assert pause_details_call.pause_status == FlowPauseStatus.USER_INPUT_REQUIRED
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
    assert flow_paused_metric_call[0][0].data["reason"] == "Awaiting user input for RETRY_STAGE_WITH_MODIFIED_INPUT"
    assert flow_paused_metric_call[0][0].data["stage_id"] == "stage1"

# Note: Ensuring a blank line above this decorator for proper separation.
@pytest.mark.parametrize(
    "action_type, suggestion_details, expected_pause_status, expected_clarification_type, expected_metric_reason_fragment",
    [
        pytest.param(
            ReviewerActionType.MODIFY_MASTER_PLAN,
            {"message_to_user": "Plan needs changes for stage1.", "suggested_plan_change": "Replace agent X with Y"},
            FlowPauseStatus.REVIEWER_ACTION_REQUIRED,
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
            FlowPauseStatus.REVIEWER_ACTION_REQUIRED, # Should pause for user if reviewer has no idea
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
            return mock_stage_agent
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id, stage1_inputs=original_inputs)
    initial_context = {"global_data": "test_escalations"}
    run_id = orchestrator.state_manager.get_or_create_current_run_id()

    final_context = await orchestrator.run(plan, initial_context)

    # Assertions
    assert mock_stage_agent.call_count == 1
    assert mock_reviewer.invoke_async_mock.call_count == 1
    
    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == run_id
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
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock,
    mock_metrics_store: MagicMock # Added metrics store for completeness
):
    """
    Tests Scenario 5.1: Reviewer suggests PROCEED_AS_IS after a stage failure.
    Orchestrator should mark stage1 as COMPLETED_WITH_WARNINGS and proceed to stage2.
    """
    failing_stage_agent_id = "agent_fails_but_proceed"
    succeeding_stage_agent_id = "agent_succeeds_after_proceed"
    error_message_stage1 = "Stage1 failed but reviewer said proceed."

    # Stage 1 agent will fail
    mock_failing_agent_stage1 = MockStageAgent(
        agent_id=failing_stage_agent_id,
        raise_exception=AgentErrorDetails(error_type="RecoverableError", message=error_message_stage1),
        output_to_return={"output_from_failed_stage": "some_partial_data"} # Output even on failure
    )
    # Stage 2 agent will succeed
    mock_succeeding_agent_stage2 = MockStageAgent(
        agent_id=succeeding_stage_agent_id,
        output_to_return={"result_stage2": "stage2_success"}
    )

    # Reviewer agent setup
    reviewer_output = MasterPlannerReviewerOutput(
        suggestion_type=ReviewerActionType.PROCEED_AS_IS,
        suggestion_details={"target_stage_id": "stage1", "message_to_log": "Proceeding past stage1 failure as per reviewer."},
        reasoning="Reviewer deems stage1 failure acceptable to proceed."
    )
    mock_reviewer = MockMasterPlannerReviewerAgent(output_to_return=reviewer_output)

    # Configure agent provider
    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == failing_stage_agent_id:
            return mock_failing_agent_stage1
        if agent_id_requested == succeeding_stage_agent_id:
            return mock_succeeding_agent_stage2
        if agent_id_requested == "system.mock_reviewer_agent":
            return mock_reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    # Two-stage plan
    plan = MasterExecutionPlan(
        id="test_plan_proceed_as_is",
        name="Test Plan for Proceed As Is",
        original_request="Test goal for proceed",
        start_stage="stage1",
        stages={
            "stage1": MasterStageSpec(
                agent_id=failing_stage_agent_id,
                inputs={"data": "input_stage1"},
                on_failure=StageFailurePolicy(action="CALL_REVIEWER_AGENT"), # Ensure reviewer is called
                next_master_stage_key="stage2"
            ),
            "stage2": MasterStageSpec(
                agent_id=succeeding_stage_agent_id,
                inputs={"data_from_stage1": "{{ outputs.stage1.output_from_failed_stage }}"},
                on_failure=StageFailurePolicy(action="TERMINATE_FAILURE")
            )
        }
    )
    initial_context = {"global_data": "test_proceed"}

    # Execute
    final_context = await orchestrator.run(plan, initial_context)

    # Assertions
    # 1. Stage 1 agent was called once (and failed)
    assert mock_failing_agent_stage1.call_count == 1
    
    # 2. Reviewer agent was called once
    assert mock_reviewer.invoke_async_mock.call_count == 1
    
    # 3. Stage 2 agent was called once (flow proceeded)
    assert mock_succeeding_agent_stage2.call_count == 1
    
    # 4. Flow should NOT have paused
    mock_state_manager.save_paused_flow_state.assert_not_called()
    
    # 5. Final context should indicate overall success (as stage2 completed)
    #    and contain outputs from both stages.
    assert "_flow_error" not in final_context, "Flow should not end in overall error"
    assert final_context.get("outputs", {}).get("stage1", {}).get("output_from_failed_stage") == "some_partial_data"
    assert final_context.get("outputs", {}).get("stage2", {}).get("result_stage2") == "stage2_success"

    # 6. StateManager status updates: stage1 COMPLETED_WITH_WARNINGS, stage2 SUCCESS
    update_status_calls = mock_state_manager.update_status.call_args_list
    
    stage1_warning_status_call = next((
        call for call in update_status_calls 
        if call.kwargs.get('stage') == plan.stages["stage1"].number and 
           call.kwargs.get('status') == StageStatus.COMPLETED_WITH_WARNINGS.value
    ), None)
    assert stage1_warning_status_call is not None, "Stage1 status not updated to COMPLETED_WITH_WARNINGS"
    assert stage1_warning_status_call.kwargs['error_details'].message == error_message_stage1
    assert "Proceeding past stage1 failure as per reviewer." in stage1_warning_status_call.kwargs['message_override']

    stage2_success_status_call = next((
        call for call in update_status_calls 
        if call.kwargs.get('stage') == plan.stages["stage2"].number and 
           call.kwargs.get('status') == StageStatus.SUCCESS.value
    ), None)
    assert stage2_success_status_call is not None, "Stage2 status not updated to SUCCESS"

    # 7. Metrics: STAGE_COMPLETED_WITH_WARNINGS for stage1, STAGE_COMPLETED for stage2
    stage1_warning_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.STAGE_COMPLETED_WITH_WARNINGS and call[0][0].data.get("stage_id") == "stage1"), None)
    assert stage1_warning_metric_call is not None, "STAGE_COMPLETED_WITH_WARNINGS metric not emitted for stage1"
    assert stage1_warning_metric_call[0][0].data["details"] == error_message_stage1
    assert stage1_warning_metric_call[0][0].data["reviewer_action"] == ReviewerActionType.PROCEED_AS_IS.value

    stage2_success_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.STAGE_COMPLETED and call[0][0].data.get("stage_id") == "stage2"), None)
    assert stage2_success_metric_call is not None, "STAGE_COMPLETED metric not emitted for stage2"

@pytest.mark.parametrize(
    "test_id, reviewer_setup_action, orchestrator_reviewer_agent_id_override, expected_pause_status, expected_clarification_type, expected_metric_reason_fragment, expected_log_message_fragment",
    [
        pytest.param(
            "unhandled_suggestion_type",
            lambda mock_reviewer_cls: mock_reviewer_cls(output_to_return=MasterPlannerReviewerOutput(
                suggestion_type="UNKNOWN_ACTION_TYPE_XYZ", # Not a valid ReviewerActionType
                reasoning="Testing unhandled type"
            )),
            "system.mock_reviewer_agent", # Default, reviewer is configured
            FlowPauseStatus.PAUSED_FOR_INTERVENTION, # Fallback pause
            "UNHANDLED_REVIEWER_SUGGESTION",
            "Unhandled reviewer suggestion type: UNKNOWN_ACTION_TYPE_XYZ",
            "Received unhandled suggestion type UNKNOWN_ACTION_TYPE_XYZ",
            id="unhandled_reviewer_action_type"
        ),
        pytest.param(
            "reviewer_raises_exception",
            lambda mock_reviewer_cls: mock_reviewer_cls(raise_exception=ValueError("Reviewer mock failed!")),
            "system.mock_reviewer_agent",
            FlowPauseStatus.PAUSED_FOR_INTERVENTION, # Fallback pause due to reviewer error
            "NO_REVIEWER_SUGGESTION_HANDLER_DEFAULT_PAUSE", # Generic pause reason
            "Reviewer agent failed or returned no actionable output",
            "Error invoking reviewer agent: Reviewer mock failed!",
            id="reviewer_agent_fails_exception"
        ),
        pytest.param(
            "reviewer_returns_none",
            lambda mock_reviewer_cls: mock_reviewer_cls(output_to_return=None), # Explicitly return None
            "system.mock_reviewer_agent",
            FlowPauseStatus.PAUSED_FOR_INTERVENTION,
            "NO_REVIEWER_SUGGESTION_HANDLER_DEFAULT_PAUSE",
            "Reviewer agent failed or returned no actionable output",
            "Reviewer agent returned None or no actionable suggestion",
            id="reviewer_agent_returns_none"
        ),
        pytest.param(
            "reviewer_not_configured",
            None, # Reviewer mock won't be called
            None, # Override orchestrator config to have no reviewer
            FlowPauseStatus.PAUSED_FOR_INTERVENTION,
            "NO_REVIEWER_CONFIGURED_PAUSE",
            "No reviewer agent configured, or reviewer failed", # Metric is a bit generic here
            "No master_planner_reviewer_agent_id configured",
            id="reviewer_not_configured"
        ),
    ]
)
@pytest.mark.asyncio
async def test_reviewer_fallback_and_error_scenarios(
    orchestrator: AsyncOrchestrator, 
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock,
    mock_metrics_store: MagicMock,
    caplog: pytest.LogCaptureFixture,
    test_id: str,
    reviewer_setup_action: callable, # Lambda to setup the mock_reviewer or None
    orchestrator_reviewer_agent_id_override: str, # To set orchestrator.master_planner_reviewer_agent_id
    expected_pause_status: FlowPauseStatus,
    expected_clarification_type: str,
    expected_metric_reason_fragment: str,
    expected_log_message_fragment: str
):
    """
    Tests fallback scenarios: Unhandled suggestion, Reviewer failure/None, Reviewer not configured.
    All should pause the flow with appropriate details and logs.
    """
    stage_agent_id = "agent_for_fallback_tests"
    error_message = "Fallback test: initial agent failure."

    mock_stage_agent = MockStageAgent(
        agent_id=stage_agent_id,
        raise_exception=AgentErrorDetails(error_type="TestFallbackError", message=error_message)
    )
    
    mock_reviewer = None
    if reviewer_setup_action:
        mock_reviewer = reviewer_setup_action(MockMasterPlannerReviewerAgent)
    
    # Override orchestrator's reviewer agent ID if needed for the test case
    original_reviewer_id = orchestrator.master_planner_reviewer_agent_id
    orchestrator.master_planner_reviewer_agent_id = orchestrator_reviewer_agent_id_override

    def agent_provider_get_side_effect(agent_id_requested):
        if agent_id_requested == stage_agent_id:
            return mock_stage_agent
        if agent_id_requested == "system.mock_reviewer_agent" and mock_reviewer: # Only return if mock_reviewer is set up
            return mock_reviewer
        # If reviewer not configured for test, or different ID, this shouldn't be hit for reviewer
        raise KeyError(f"Mock agent not found for {agent_id_requested}")
    mock_agent_provider.get = MagicMock(side_effect=agent_provider_get_side_effect)

    plan = create_simple_plan(stage1_agent_id=stage_agent_id)
    plan.stages["stage1"].on_failure = StageFailurePolicy(action="CALL_REVIEWER_AGENT") # Ensure reviewer is called if configured
    initial_context = {"global_data": "test_fallbacks"}
    run_id = orchestrator.state_manager.get_or_create_current_run_id()

    caplog.clear()
    log_level_to_capture = logging.ERROR if "reviewer_raises_exception" in test_id or "reviewer_not_configured" in test_id or "reviewer_returns_none" in test_id else logging.WARNING
    
    with caplog.at_level(log_level_to_capture, logger="chungoid.runtime.orchestrator"):
        final_context = await orchestrator.run(plan, initial_context)
    
    # Restore original reviewer ID for other tests
    orchestrator.master_planner_reviewer_agent_id = original_reviewer_id

    # Assertions
    assert mock_stage_agent.call_count == 1
    if mock_reviewer and hasattr(mock_reviewer, 'invoke_async_mock'): # Reviewer should be called unless not configured or returns None early
        if test_id != "reviewer_returns_none": # invoke_async_mock not called if __call__ returns None directly from output_to_return=None
             assert mock_reviewer.invoke_async_mock.call_count == 1
    
    mock_state_manager.save_paused_flow_state.assert_called_once()
    pause_details_call = mock_state_manager.save_paused_flow_state.call_args[0][1]

    assert pause_details_call.run_id == run_id
    assert pause_details_call.pause_status == expected_pause_status
    assert pause_details_call.triggering_error_details.message == error_message
    
    clarification_req = pause_details_call.clarification_request
    assert clarification_req["type"] == expected_clarification_type
    if test_id == "unhandled_suggestion_type":
        assert clarification_req["reviewer_output"]["suggestion_type"] == "UNKNOWN_ACTION_TYPE_XYZ"
    
    assert final_context.get("_autonomous_flow_paused_state_saved") is True
    assert "_flow_error" in final_context
    if "reviewer_suggestion_type" in final_context["_flow_error"]:
         assert final_context["_flow_error"]["reviewer_suggestion_type"] == ("UNKNOWN_ACTION_TYPE_XYZ" if test_id == "unhandled_suggestion_type" else None) # Adjust if other types populate this

    # Check logs
    assert any(expected_log_message_fragment in record.message for record in caplog.records), \
        f"Expected log fragment '{expected_log_message_fragment}' not found. Logs: {[r.message for r in caplog.records]}"

    # Check metrics
    flow_paused_metric_call = next((call for call in mock_metrics_store.add_event.call_args_list if call[0][0].event_type == MetricEventType.FLOW_PAUSED), None)
    assert flow_paused_metric_call is not None, "FLOW_PAUSED metric not emitted"
    assert expected_metric_reason_fragment in flow_paused_metric_call[0][0].data["reason"]
    assert flow_paused_metric_call[0][0].data["stage_id"] == "stage1"