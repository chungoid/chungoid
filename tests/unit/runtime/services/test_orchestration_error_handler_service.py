import pytest
import logging
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import Optional, Dict, Any

from pydantic import ValidationError

from chungoid.runtime.services.orchestration_error_handler_service import (
    OrchestrationErrorHandlerService,
    OrchestrationErrorResult,
    NEXT_STAGE_END_FAILURE,
    NEXT_STAGE_END_SUCCESS # Though not directly used by error handler, good for consistency if plan gives it
)
from chungoid.schemas.common_enums import StageStatus, FlowPauseStatus, OnFailureAction, ReviewerActionType
from chungoid.schemas.errors import AgentErrorDetails, OrchestratorError
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, MasterStageFailurePolicy, ConditionalTransition
from chungoid.schemas.orchestration import SharedContext
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    RetryStageWithChangesDetails,
)
from chungoid.schemas.metrics import MetricEvent, MetricEventType

from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore


@pytest.fixture
def mock_logger():
    logger = MagicMock(spec=logging.Logger)
    logger.warning = MagicMock()
    logger.info = MagicMock()
    logger.error = MagicMock()
    logger.debug = MagicMock()
    return logger

@pytest.fixture
def mock_agent_provider():
    provider = AsyncMock(spec=AgentProvider)
    provider.get_agent_callable = AsyncMock()
    return provider

@pytest.fixture
def mock_state_manager():
    sm = AsyncMock(spec=StateManager)
    sm.record_stage_run = AsyncMock()
    # Add other methods if directly called by service, though most are orchestrator's concern
    return sm

@pytest.fixture
def mock_metrics_store():
    ms = MagicMock(spec=MetricsStore)
    ms.add_event = MagicMock()
    return ms

@pytest.fixture
def default_config_params():
    return {
        "master_planner_reviewer_agent_id": "system.mock_reviewer",
        "default_on_failure_action": OnFailureAction.PAUSE_FOR_INTERVENTION,
        "default_agent_retries": 1
    }

@pytest.fixture
def error_handler_service(
    mock_logger, mock_agent_provider, mock_state_manager, mock_metrics_store, default_config_params
):
    return OrchestrationErrorHandlerService(
        logger=mock_logger,
        agent_provider=mock_agent_provider,
        state_manager=mock_state_manager,
        metrics_store=mock_metrics_store,
        **default_config_params
    )

@pytest.fixture
def minimal_stage_spec(request):
    # Allow overriding via request.param
    details = getattr(request, "param", {})
    return MasterStageSpec(
        agent_id=details.get("agent_id", "test_agent"),
        inputs=details.get("inputs", {"input1": "value1"}),
        output_context_path=details.get("output_context_path", "test_output"),
        on_failure=details.get("on_failure", MasterStageFailurePolicy(action=OnFailureAction.PAUSE_FOR_INTERVENTION)),
        max_retries=details.get("max_retries", 1)
    )

@pytest.fixture
def minimal_plan(minimal_stage_spec):
    return MasterExecutionPlan(
        id="test_plan_id",
        flow_id="test_flow_id",
        name="Test Plan",
        description="A plan for testing",
        start_stage_id="stage1",
        stages={"stage1": minimal_stage_spec}
    )

@pytest.fixture
def minimal_shared_context():
    return SharedContext(run_id="test_run_id", flow_id="test_flow_id")


class TestOrchestrationErrorHandlerService:

    class TestCreateAgentErrorDetails:
        def test_with_raw_exception(self, error_handler_service):
            raw_exc = ValueError("A raw error")
            details = error_handler_service._create_agent_error_details(
                error=raw_exc,
                agent_id="agent_x",
                stage_id="stage_y",
                resolved_inputs_at_failure={"input": "val"},
                default_can_retry=True
            )
            assert isinstance(details, AgentErrorDetails)
            assert details.agent_id == "agent_x"
            assert details.stage_id == "stage_y"
            assert details.error_type == "ValueError"
            assert details.message == "A raw error"
            assert details.traceback is not None
            assert details.can_retry is False # Generic exceptions default can_retry to False from the method
            assert details.resolved_inputs_at_failure == {"input": "val"}

        def test_with_existing_agent_error_details(self, error_handler_service):
            existing_details = AgentErrorDetails(
                agent_id="orig_agent",
                stage_id="orig_stage",
                error_type="OriginalError",
                message="Original message",
                can_retry=True,
                resolved_inputs_at_failure={"orig_input": "orig_val"}
            )
            details = error_handler_service._create_agent_error_details(
                error=existing_details,
                agent_id="new_agent", # Should not override if already present
                stage_id="new_stage",   # Should not override
                resolved_inputs_at_failure={"new_input": "new_val"}, # Should augment if None
                default_can_retry=False
            )
            assert details.agent_id == "orig_agent"
            assert details.stage_id == "orig_stage"
            assert details.error_type == "OriginalError"
            assert details.can_retry is True
            assert details.resolved_inputs_at_failure == {"orig_input": "orig_val"} # Not overridden if present
            assert details.traceback is not None # Should be added

        def test_with_orchestrator_error_wrapping_agent_error(self, error_handler_service):
            inner_agent_error = AgentErrorDetails(agent_id="core_agent", stage_id="core_stage", error_type="CoreError", message="Core problem")
            orch_error = OrchestratorError("Wrapper error", stage_name="wrapper_stage")
            orch_error.__cause__ = inner_agent_error
            
            details = error_handler_service._create_agent_error_details(
                error=orch_error,
                agent_id="outer_agent",
                stage_id="outer_stage",
                resolved_inputs_at_failure=None
            )
            assert details.agent_id == "core_agent" # Should use the innermost AgentErrorDetails
            assert details.stage_id == "core_stage"
            assert details.error_type == "CoreError"

        def test_with_orchestrator_error_not_wrapping(self, error_handler_service):
            orch_error = OrchestratorError("Standalone orchestrator issue", stage_name="setup_stage")
            details = error_handler_service._create_agent_error_details(
                error=orch_error,
                agent_id="specific_agent_if_known", # Passed agent_id
                stage_id="setup_stage",
                resolved_inputs_at_failure=None
            )
            assert details.agent_id == "specific_agent_if_known" # If OrchestratorError, uses passed agent_id or "Orchestrator"
            assert details.stage_id == "setup_stage"
            assert details.error_type == "OrchestratorError" # The type of OrchestratorError itself
            assert details.can_retry is False # Orchestrator errors default to not retriable

        def test_default_can_retry_behavior(self, error_handler_service):
            raw_exc = ValueError("A raw error")
            # getattr(raw_exc, 'can_retry', False) will be False
            details_no_default_retry = error_handler_service._create_agent_error_details(
                error=raw_exc, agent_id="a", stage_id="s", resolved_inputs_at_failure={}, default_can_retry=False
            )
            assert details_no_default_retry.can_retry is False
            
            # Test with an error that *has* can_retry = True
            class RetriableError(Exception):
                can_retry = True
            retriable_exc = RetriableError("I am retriable")
            details_retriable = error_handler_service._create_agent_error_details(
                 error=retriable_exc, agent_id="a", stage_id="s", resolved_inputs_at_failure={}, default_can_retry=False
            )
            assert details_retriable.can_retry is True


    class TestInvokeReviewer:
        @pytest.mark.asyncio
        async def test_reviewer_not_configured(
            self, mock_logger, mock_agent_provider, mock_state_manager, mock_metrics_store, default_config_params
        ):
            config = default_config_params.copy()
            config["master_planner_reviewer_agent_id"] = "" # Not configured
            service_no_reviewer = OrchestrationErrorHandlerService(
                logger=mock_logger, agent_provider=mock_agent_provider, state_manager=mock_state_manager,
                metrics_store=mock_metrics_store, **config
            )
            result = await service_no_reviewer._invoke_reviewer(
                run_id="r1", flow_id="f1", current_stage_name="s1",
                agent_error_details_obj=MagicMock(spec=AgentErrorDetails),
                current_shared_context=MagicMock(spec=SharedContext),
                current_plan=MagicMock(spec=MasterExecutionPlan)
            )
            assert result is None
            mock_logger.warning.assert_called_once()

        @pytest.mark.asyncio
        async def test_agent_provider_get_fails(self, error_handler_service, mock_agent_provider, mock_logger):
            mock_agent_provider.get_agent_callable.side_effect = Exception("Provider boom")
            result = await error_handler_service._invoke_reviewer("r1", "f1", "s1", MagicMock(), MagicMock(), MagicMock())
            assert result is None
            mock_logger.error.assert_called_once()

        @pytest.mark.asyncio
        async def test_reviewer_agent_call_raises_exception(self, error_handler_service, mock_agent_provider, mock_logger, mock_metrics_store):
            mock_reviewer_callable = AsyncMock(side_effect=Exception("Agent boom"))
            mock_agent_provider.get_agent_callable.return_value = mock_reviewer_callable
            
            result = await error_handler_service._invoke_reviewer(
                "r1", "f1", "s1", 
                AgentErrorDetails(agent_id="a", stage_id="s", error_type="t", message="m"), # Valid AED
                SharedContext(run_id="r1", flow_id="f1"), # Valid context
                MasterExecutionPlan(id="p1", name="p", description="d", start_stage_id="s1", stages={}) # Valid plan
            )
            assert result is None
            mock_logger.error.assert_called_once()
            # Check metrics for start and error end
            assert mock_metrics_store.add_event.call_count == 2


        @pytest.mark.asyncio
        async def test_reviewer_returns_unexpected_type(self, error_handler_service, mock_agent_provider, mock_logger, mock_metrics_store):
            mock_reviewer_callable = AsyncMock(return_value="not a ReviewerOutput")
            mock_agent_provider.get_agent_callable.return_value = mock_reviewer_callable
            result = await error_handler_service._invoke_reviewer(
                 "r1", "f1", "s1", 
                AgentErrorDetails(agent_id="a", stage_id="s", error_type="t", message="m"),
                SharedContext(run_id="r1", flow_id="f1"),
                MasterExecutionPlan(id="p1", name="p", description="d", start_stage_id="s1", stages={})
            )
            assert result is None
            mock_logger.error.assert_called_once()
            assert mock_metrics_store.add_event.call_count == 2 # Start and error end

        @pytest.mark.asyncio
        async def test_reviewer_returns_valid_output(self, error_handler_service, mock_agent_provider, mock_metrics_store):
            expected_output = MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.PROCEED_AS_IS)
            mock_reviewer_callable = AsyncMock(return_value=expected_output)
            mock_agent_provider.get_agent_callable.return_value = mock_reviewer_callable
            
            result = await error_handler_service._invoke_reviewer(
                "r1", "f1", "s1",
                AgentErrorDetails(agent_id="a", stage_id="s", error_type="t", message="m"),
                SharedContext(run_id="r1", flow_id="f1"),
                MasterExecutionPlan(id="p1", name="p", description="d", start_stage_id="s1", stages={})
            )
            assert result == expected_output
            assert mock_metrics_store.add_event.call_count == 2 # Start and success end
            
        @pytest.mark.asyncio
        async def test_reviewer_input_construction(self, error_handler_service, mock_agent_provider):
            # Test that MasterPlannerReviewerInput is constructed correctly
            mock_reviewer_callable = AsyncMock(return_value=MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.PROCEED_AS_IS))
            mock_agent_provider.get_agent_callable.return_value = mock_reviewer_callable

            error_details = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="TestError", message="Test msg")
            shared_ctx = SharedContext(run_id="r1", flow_id="f1", current_stage_id="s1")
            plan = MasterExecutionPlan(id="p1", name="p", description="d", start_stage_id="s1", stages={})

            await error_handler_service._invoke_reviewer(
                run_id="r1", flow_id="f1", current_stage_name="s1",
                agent_error_details_obj=error_details,
                current_shared_context=shared_ctx,
                current_plan=plan
            )
            
            # Assert that the callable was called with an instance of MasterPlannerReviewerInput
            assert mock_reviewer_callable.call_args is not None
            called_with_input = mock_reviewer_callable.call_args[1].get('task_input') # Or 'inputs' depending on agent signature
            if called_with_input is None: # try other common kwarg name
                 called_with_input = mock_reviewer_callable.call_args[1].get('inputs')

            assert isinstance(called_with_input, MasterPlannerReviewerInput)
            assert called_with_input.current_master_plan == plan
            assert called_with_input.paused_stage_id == "s1"
            assert called_with_input.triggering_error_details == error_details
            assert called_with_input.paused_run_details.run_id == "r1"


    class TestHandleStageExecutionError:

        @pytest.mark.asyncio
        @pytest.mark.parametrize("minimal_stage_spec", [{"max_retries": 2}], indirect=True)
        async def test_retriable_error_within_limits(self, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context, mock_state_manager, mock_metrics_store):
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Retriable", message="Can retry", can_retry=True)
            
            result = await error_handler_service.handle_stage_execution_error(
                current_stage_name="stage1", flow_id="f1", run_id="r1",
                current_plan=minimal_plan, error=error, agent_id_that_erred="a",
                attempt_number=1, # First attempt failed
                shared_context_at_error=minimal_shared_context,
                resolved_inputs_at_failure={"in": "val"}
            )
            assert result.next_stage_to_execute == "stage1" # Retry current
            assert result.flow_pause_status == FlowPauseStatus.NOT_PAUSED
            assert result.reviewer_output is None
            assert result.handled_attempt_number == 1
            mock_state_manager.record_stage_run.assert_called_once()
            # Metrics: STAGE_ERROR_ENCOUNTERED, STAGE_RETRY_ATTEMPT
            assert mock_metrics_store.add_event.call_count == 2


        @pytest.mark.asyncio
        @pytest.mark.parametrize("minimal_stage_spec", [{"max_retries": 0}], indirect=True) # Max 0 retries (so only 1 attempt allowed)
        async def test_retriable_error_retries_exhausted(self, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context, default_config_params):
            # on_failure is PAUSE_FOR_USER_REVIEW by default from minimal_stage_spec via default_config_params
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Retriable", message="Can retry", can_retry=True)
            minimal_plan.stages["stage1"].on_failure = MasterStageFailurePolicy(action=OnFailureAction.PAUSE_FOR_USER_REVIEW)

            result = await error_handler_service.handle_stage_execution_error(
                current_stage_name="stage1", flow_id="f1", run_id="r1",
                current_plan=minimal_plan, error=error, agent_id_that_erred="a",
                attempt_number=1, # This attempt fails, max_retries is 0, so no more retries
                shared_context_at_error=minimal_shared_context,
                resolved_inputs_at_failure={}
            )
            assert result.next_stage_to_execute is None # Pause
            assert result.flow_pause_status == FlowPauseStatus.USER_INTERVENTION_REQUIRED


        @pytest.mark.asyncio
        async def test_non_retriable_error(self, error_handler_service, minimal_plan, minimal_shared_context):
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Fatal", message="Cannot retry", can_retry=False)
            minimal_plan.stages["stage1"].on_failure = MasterStageFailurePolicy(action=OnFailureAction.PAUSE_FOR_USER_REVIEW)
            
            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            assert result.next_stage_to_execute is None # Pause due to default on_failure
            assert result.flow_pause_status == FlowPauseStatus.USER_INTERVENTION_REQUIRED

        @pytest.mark.asyncio
        @pytest.mark.parametrize("minimal_stage_spec", [{"on_failure": MasterStageFailurePolicy(action=OnFailureAction.FAIL_MASTER_FLOW)}], indirect=True)
        async def test_on_failure_fail_master_flow(self, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context):
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Fatal", message="Fail flow", can_retry=False)
            
            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            assert result.next_stage_to_execute == NEXT_STAGE_END_FAILURE
            assert result.flow_pause_status == FlowPauseStatus.NOT_PAUSED
        
        @pytest.mark.asyncio
        @patch.object(OrchestrationErrorHandlerService, '_invoke_reviewer', new_callable=AsyncMock)
        @pytest.mark.parametrize("minimal_stage_spec", [{"on_failure": MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER)}], indirect=True)
        async def test_on_failure_invoke_reviewer_suggests_retry_as_is(self, mock_invoke_reviewer, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context):
            mock_invoke_reviewer.return_value = MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.RETRY_STAGE_AS_IS)
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Reviewed", message="Review me", can_retry=False) # can_retry=False to skip initial retry logic

            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            assert result.next_stage_to_execute == "stage1" # Retry current
            assert result.flow_pause_status == FlowPauseStatus.NOT_PAUSED
            assert result.reviewer_output is not None
            assert result.reviewer_output.suggestion_type == ReviewerActionType.RETRY_STAGE_AS_IS
            mock_invoke_reviewer.assert_called_once()

        @pytest.mark.asyncio
        @patch.object(OrchestrationErrorHandlerService, '_invoke_reviewer', new_callable=AsyncMock)
        @pytest.mark.parametrize("minimal_stage_spec", [{"on_failure": MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER)}], indirect=True)
        async def test_on_failure_invoke_reviewer_suggests_retry_with_changes(self, mock_invoke_reviewer, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context):
            new_inputs = {"new_input": "new_value"}
            suggestion_details = RetryStageWithChangesDetails(target_stage_id="stage1", changes_to_stage_spec=new_inputs)
            mock_invoke_reviewer.return_value = MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES,
                suggestion_details=suggestion_details
            )
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="ChangeMe", message="Needs changes", can_retry=False)

            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            assert result.next_stage_to_execute == "stage1"
            assert result.flow_pause_status == FlowPauseStatus.NOT_PAUSED
            assert result.modified_stage_inputs == new_inputs
            assert result.reviewer_output is not None


        @pytest.mark.asyncio
        @patch.object(OrchestrationErrorHandlerService, '_invoke_reviewer', new_callable=AsyncMock)
        @pytest.mark.parametrize("minimal_stage_spec", [{"on_failure": MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER)}], indirect=True)
        async def test_on_failure_invoke_reviewer_suggests_proceed_as_is(self, mock_invoke_reviewer, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context):
            mock_invoke_reviewer.return_value = MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.PROCEED_AS_IS)
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="AcceptableError", message="Proceed anyway", can_retry=False)

            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            # For PROCEED_AS_IS, the handler returns current_stage_name, orchestrator interprets
            assert result.next_stage_to_execute == "stage1" 
            assert result.flow_pause_status == FlowPauseStatus.NOT_PAUSED
            assert result.reviewer_output is not None
            assert result.reviewer_output.suggestion_type == ReviewerActionType.PROCEED_AS_IS


        @pytest.mark.asyncio
        @patch.object(OrchestrationErrorHandlerService, '_invoke_reviewer', new_callable=AsyncMock)
        @pytest.mark.parametrize("minimal_stage_spec", [{"on_failure": MasterStageFailurePolicy(action=OnFailureAction.INVOKE_REVIEWER)}], indirect=True)
        async def test_on_failure_invoke_reviewer_fails_or_no_suggestion(self, mock_invoke_reviewer, error_handler_service, minimal_plan, minimal_stage_spec, minimal_shared_context):
            mock_invoke_reviewer.return_value = None # Reviewer fails
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Reviewed", message="Review me", can_retry=False)

            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            assert result.next_stage_to_execute is None # Pause
            assert result.flow_pause_status == FlowPauseStatus.USER_INTERVENTION_REQUIRED


        @pytest.mark.asyncio
        async def test_stage_spec_not_found(self, error_handler_service, minimal_plan, minimal_shared_context, mock_logger):
            # Tamper with plan so stage "stage1" is not found
            plan_no_stage1 = minimal_plan.model_copy(deep=True)
            plan_no_stage1.stages = {"stage2": MasterStageSpec(agent_id="other")} # Remove stage1

            error = ValueError("Some error")
            result = await error_handler_service.handle_stage_execution_error(
                current_stage_name="stage1", # This stage is missing
                flow_id="f1", run_id="r1",
                current_plan=plan_no_stage1, 
                error=error, agent_id_that_erred="a",
                attempt_number=1, shared_context_at_error=minimal_shared_context,
                resolved_inputs_at_failure={}
            )
            assert result.next_stage_to_execute == NEXT_STAGE_END_FAILURE
            assert result.updated_agent_error_details.error_type == "StageSpecNotFound"
            mock_logger.error.assert_any_call(
                "Run r1, Flow f1: Stage spec for 'stage1' not found in _handle_stage_error. Cannot determine retry/failure policy."
            )


        @pytest.mark.asyncio
        async def test_state_manager_record_stage_run_failure(self, error_handler_service, minimal_plan, minimal_shared_context, mock_state_manager, mock_logger):
            # Setup StateManager to fail on record_stage_run
            mock_state_manager.record_stage_run.side_effect = Exception("StateManager down")
            error = AgentErrorDetails(agent_id="a", stage_id="s1", error_type="Retriable", message="Can retry", can_retry=True)
            minimal_plan.stages["stage1"].max_retries = 1 # Ensure it attempts retry

            result = await error_handler_service.handle_stage_execution_error(
                "stage1", "f1", "r1", minimal_plan, error, "a", 1, minimal_shared_context, {}
            )
            # Service should still proceed and attempt retry logic despite SM failure
            assert result.next_stage_to_execute == "stage1" 
            mock_logger.error.assert_any_call(
                "Run r1, Flow f1: Failed to record stage run with StateManager for stage 'stage1': StateManager down",
                exc_info=True
            )

        # Add more tests for other specific reviewer suggestions (ESCALATE_TO_USER, etc.)
        # Add tests for default pause behavior if on_failure is PAUSE_FOR_USER_REVIEW
