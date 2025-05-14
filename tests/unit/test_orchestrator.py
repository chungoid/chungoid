#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call, ANY
from typing import Dict, Any, List, Union, Optional
import traceback
from datetime import datetime, timezone
import copy

from chungoid.runtime.orchestrator import AsyncOrchestrator
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.flows import PausedRunDetails
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.schemas.user_goal_schemas import UserGoalRequest # Delayed import


# --- Fixtures ---

@pytest.fixture
def mock_agent_provider() -> MagicMock:
    provider = MagicMock(spec=AgentProvider)
    # Setup a default async mock for the get method
    provider.get = AsyncMock()
    return provider

@pytest.fixture
def mock_state_manager() -> MagicMock:
    manager = MagicMock(spec=StateManager)
    # Setup default async mocks for methods called by orchestrator
    manager.update_status = MagicMock(return_value=True)
    manager.get_or_create_current_run_id = MagicMock(return_value="0") # Return string "0"
    manager.save_paused_flow_state = MagicMock(return_value=True) # Sync method
    return manager

@pytest.fixture
def mock_project_config() -> Dict[str, Any]:
    # In the current orchestrator, config is passed but not used directly in run
    # Provide a basic dictionary.
    return {"some_config_key": "some_value"}

@pytest.fixture
def basic_plan() -> MasterExecutionPlan:
    """A simple linear execution plan."""
    return MasterExecutionPlan(
        id="test_plan_basic",
        name="Test Basic Plan",
        start_stage="stage0",
        stages={
            "stage0": MasterStageSpec(agent_id="agent_a", next_stage="stage1", number=0.0),
            "stage1": MasterStageSpec(agent_id="agent_b", next_stage=None, number=1.0),
        }
    )

@pytest.fixture
def conditional_plan() -> MasterExecutionPlan:
    """A plan with conditional branching."""
    return MasterExecutionPlan(
        id="test_plan_conditional",
        name="Test Conditional Plan",
        start_stage="stage_input",
        stages={
            "stage_input": MasterStageSpec(
                agent_id="input_agent",
                next_stage="stage_cond",
                number=-1.0 # Assign arbitrary number for testing if needed
            ),
            "stage_cond": MasterStageSpec(
                agent_id="condition_checker", # Not actually called, condition uses context
                condition="outputs.stage_input.result == 'go_left'",
                next_stage_true="stage_left",
                next_stage_false="stage_right",
                number=-2.0
            ),
            "stage_left": MasterStageSpec(agent_id="left_agent", next_stage=None, number=-3.0),
            "stage_right": MasterStageSpec(agent_id="right_agent", next_stage=None, number=-4.0),
        }
    )

@pytest.fixture
def orchestrator_for_resume(mock_agent_provider, mock_state_manager, mock_project_config, basic_plan) -> AsyncOrchestrator:
    """Creates an AsyncOrchestrator specifically for resume tests, maybe pre-configured."""
    # Use basic_plan for simplicity, can override in specific tests if needed
    instance = AsyncOrchestrator(
        pipeline_def=basic_plan,
        config=mock_project_config,
        agent_provider=mock_agent_provider,
        state_manager=mock_state_manager
    )
    # Ensure _execute_loop is always an AsyncMock for consistent assertion capability
    instance._execute_loop = AsyncMock(return_value={"final_context_from_fixture_loop_mock": True})
    return instance


# --- Test Class ---

@pytest.mark.asyncio
class TestAsyncOrchestrator:

    @pytest.fixture
    def orchestrator(self, mock_agent_provider, mock_state_manager, mock_project_config, basic_plan) -> AsyncOrchestrator:
        """Creates an AsyncOrchestrator instance for testing."""
        return AsyncOrchestrator(
            pipeline_def=basic_plan,
            config=mock_project_config,
            agent_provider=mock_agent_provider,
            state_manager=mock_state_manager
        )

    async def test_run_basic_linear_flow(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test a simple plan executes stages sequentially."""
        agent_a_mock = AsyncMock(return_value={"output_a": "value_a"})
        agent_b_mock = AsyncMock(return_value={"output_b": "value_b"})
        mock_agent_provider.get.side_effect = [agent_a_mock, agent_b_mock]

        initial_context = {"global_input": "start_value"}
        final_context = await orchestrator.run(basic_plan, initial_context.copy())

        mock_agent_provider.get.assert_has_calls([call("agent_a"), call("agent_b")])
        agent_a_mock.assert_awaited_once()
        agent_b_mock.assert_awaited_once()
        assert final_context['outputs']['stage0'] == {"output_a": "value_a"}
        assert final_context['outputs']['stage1'] == {"output_b": "value_b"}
        assert final_context['global_input'] == "start_value"
        mock_state_manager.update_status.assert_has_calls([
            call(stage=0.0, status='PASS', artifacts=[]),
            call(stage=1.0, status='PASS', artifacts=[]),
        ])

    async def test_run_context_passing_and_merging(self, orchestrator, mock_agent_provider, mock_state_manager):
        """Test context is passed correctly and outputs merged."""
        plan = MasterExecutionPlan(
            id="test_plan_context",
            name="Test Context Plan",
            start_stage="stage_x",
            stages={
                "stage_x": MasterStageSpec(
                    agent_id="agent_x",
                    inputs={"param1": "literal", "param2": "context.global_val"},
                    next_stage="stage_y",
                    number=10.0
                ),
                "stage_y": MasterStageSpec(
                    agent_id="agent_y",
                    inputs={"input_from_x": "context.outputs.stage_x.data"},
                    next_stage=None,
                    number=11.0
                ),
            }
        )

        agent_x_mock = AsyncMock(return_value={"data": "output_from_x"})
        agent_y_mock = AsyncMock(return_value={"final_result": "output_from_y"})
        mock_agent_provider.get.side_effect = [agent_x_mock, agent_y_mock]

        initial_context = {"global_val": "hello"}
        final_context = await orchestrator.run(plan, initial_context.copy())

        agent_x_mock.assert_awaited_once_with(
            {"global_val": "hello", "outputs": {}, "visited": ["stage_x"], "param1": "literal", "param2": "hello"}
        )
        agent_y_mock.assert_awaited_once_with(
            {"global_val": "hello", "outputs": {"stage_x": {"data": "output_from_x"}}, "visited": ["stage_x", "stage_y"], "input_from_x": "output_from_x"}
        )
        assert final_context['outputs']['stage_x'] == {"data": "output_from_x"}
        assert final_context['outputs']['stage_y'] == {"final_result": "output_from_y"}
        assert final_context['global_val'] == "hello"

    # Helper to get stage number, handling potential errors
    def _get_stage_num(self, plan: MasterExecutionPlan, stage_name: str) -> float:
        # Attempt to parse stage number from name first (e.g., "stage1.5")
        try:
            num_str = stage_name.replace("stage", "")
            return float(num_str)
        except ValueError:
            # Fallback to using the 'number' field if parsing fails or it's not numeric
            try:
                 stage_spec = plan.stages.get(stage_name)
                 if stage_spec and stage_spec.number is not None:
                     return stage_spec.number
            except KeyError:
                 pass # Stage not found in plan stages dict
        # Default error value if neither method works
        print(f"Warning: Could not determine numeric stage for '{stage_name}'. Using -99.9") # Added print for debugging
        return -99.9

    async def test_run_conditional_branching_true(self, orchestrator, mock_agent_provider, mock_state_manager, conditional_plan):
        """Test conditional branching when condition is true."""
        input_agent_mock = AsyncMock(return_value={"result": "go_left"})
        condition_checker_mock = AsyncMock(return_value={"cond_check": "done"}) # Mock for the condition stage
        left_agent_mock = AsyncMock(return_value={"left_out": "ok"})
        right_agent_mock = AsyncMock() # Should not be called
        # Order matters: input_agent, then condition_checker, then left_agent (if true)
        mock_agent_provider.get.side_effect = [input_agent_mock, condition_checker_mock, left_agent_mock]
    
        initial_context = {}
        await orchestrator.run(conditional_plan, initial_context.copy())

        mock_agent_provider.get.assert_has_calls([call("input_agent"), call("condition_checker"), call("left_agent")])
        input_agent_mock.assert_awaited_once()
        condition_checker_mock.assert_awaited_once()
        left_agent_mock.assert_awaited_once()
        right_agent_mock.assert_not_called()

        stage_input_num = self._get_stage_num(conditional_plan, "stage_input")
        stage_cond_num = self._get_stage_num(conditional_plan, "stage_cond") # Added for the conditional stage
        stage_left_num = self._get_stage_num(conditional_plan, "stage_left")
    
        mock_state_manager.update_status.assert_has_calls([
            call(stage=stage_input_num, status='PASS', artifacts=[]),
            call(stage=stage_cond_num, status='PASS', artifacts=[]), # Added call for conditional stage
            call(stage=stage_left_num, status='PASS', artifacts=[]),
        ], any_order=False) # Order should be deterministic here

    async def test_run_conditional_branching_false(self, orchestrator, mock_agent_provider, mock_state_manager, conditional_plan):
        """Test conditional branching when condition is false."""
        input_agent_mock = AsyncMock(return_value={"result": "go_right"})
        condition_checker_mock = AsyncMock(return_value={"cond_check": "done"}) # Mock for the condition stage
        left_agent_mock = AsyncMock() # Should not be called
        right_agent_mock = AsyncMock(return_value={"right_out": "ok"})
        # Order matters: input_agent, then condition_checker, then right_agent (if false)
        mock_agent_provider.get.side_effect = [input_agent_mock, condition_checker_mock, right_agent_mock]
    
        initial_context = {}
        await orchestrator.run(conditional_plan, initial_context.copy())

        mock_agent_provider.get.assert_has_calls([call("input_agent"), call("condition_checker"), call("right_agent")])
        input_agent_mock.assert_awaited_once()
        condition_checker_mock.assert_awaited_once()
        right_agent_mock.assert_awaited_once()
        left_agent_mock.assert_not_called()

        stage_input_num = self._get_stage_num(conditional_plan, "stage_input")
        stage_cond_num = self._get_stage_num(conditional_plan, "stage_cond") # Added for the conditional stage
        stage_right_num = self._get_stage_num(conditional_plan, "stage_right")

        mock_state_manager.update_status.assert_has_calls([
             call(stage=stage_input_num, status='PASS', artifacts=[]),
             call(stage=stage_cond_num, status='PASS', artifacts=[]), # Added call for conditional stage
             call(stage=stage_right_num, status='PASS', artifacts=[]),
        ], any_order=False) # Order should be deterministic

    async def test_run_agent_not_found(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test behavior when an agent cannot be resolved."""
        mock_agent_provider.get.return_value = None

        initial_context = {}
        await orchestrator.run(basic_plan, initial_context.copy())

        mock_agent_provider.get.assert_awaited_once_with("agent_a")
        mock_state_manager.update_status.assert_called_once_with(
            stage=0.0,
            status=StageStatus.FAILURE.value,
            artifacts=[],
            reason=f"Agent 'agent_a' resolved to None"
        )
        mock_state_manager.save_paused_flow_state.assert_not_called()

    async def test_run_max_hops_reached(self, orchestrator, mock_agent_provider, mock_state_manager):
        """Test execution stops if max hops are reached (potential loop)."""
        plan = MasterExecutionPlan(
            id="test_plan_loop",
            name="Test Loop Plan",
            start_stage="stage_loop1",
            stages={
                "stage_loop1": MasterStageSpec(agent_id="looper", next_stage="stage_loop2", number=1.0),
                "stage_loop2": MasterStageSpec(agent_id="looper", next_stage="stage_loop1", number=2.0),
            }
        )
        looper_agent_mock = AsyncMock(return_value={})
        mock_agent_provider.get.return_value = looper_agent_mock

        initial_context = {}
        await orchestrator.run(plan, initial_context.copy())

        max_hops = len(plan.stages) + 5 # 7
        # The agent provider should be called 2 times before the simple loop detection for stage_loop1 stops it.
        assert mock_agent_provider.get.await_count == 2

        # Check that the error logged is for loop detection
        # The failure occurs when trying to re-execute stage_loop1 (number 1.0)

        # Assert that update_status was called 3 times (2 SUCCESS, 1 FAILURE)
        assert mock_state_manager.update_status.call_count == 3

        # Get the last call arguments
        # call_args_list contains Call objects. call_args is a deprecated alias for call_args_list[0].
        # Each element is a Call object which is like a tuple: first element is positional args (a tuple), second is keyword args (a dict).
        # Since update_status is called with keyword arguments, we inspect the kwargs of the last call.
        last_call_kwargs = mock_state_manager.update_status.call_args_list[-1].kwargs

        expected_error_info_dict = {
            "short_error": "Loop detected",
            "details": "Loop detected: Stage 'stage_loop1' visited again. Aborting flow."
        }

        assert last_call_kwargs.get("pipeline_run_id") == "0"
        assert last_call_kwargs.get("stage_name") == "stage_loop1"
        assert last_call_kwargs.get("stage_number") == 1.0
        assert last_call_kwargs.get("status") == StageStatus.FAILURE.value
        assert last_call_kwargs.get("artifacts") == []
        assert last_call_kwargs.get("reason") == "Loop detected: Stage 'stage_loop1' visited again. Aborting flow."
        assert last_call_kwargs.get("error_info") == expected_error_info_dict

        mock_state_manager.save_paused_flow_state.assert_not_called()

        # Print the call list for debugging
        print("DEBUG: mock_state_manager.update_status.call_args_list:")
        for c_args in mock_state_manager.update_status.call_args_list:
            print(f"  {c_args}")

    # --- Tests for AW1.5 (Exception Handling) & AW1.12 (Pause Behavior) ---

    async def test_run_agent_exception_handling_and_pause(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test agent exception triggers error details creation and pause state saving."""
        agent_a_mock = AsyncMock(side_effect=ValueError("Agent A failed spectacularly!"))
        agent_b_mock = AsyncMock()
        mock_agent_provider.get.side_effect = [agent_a_mock, agent_b_mock]

        test_run_id = 123
        mock_state_manager.get_or_create_current_run_id.return_value = test_run_id

        initial_context = {"input": "value"}
        # Make sure the orchestrator uses the basic_plan with its ID
        orchestrator.pipeline_def = basic_plan 
        final_context = await orchestrator.run(basic_plan, initial_context.copy()) # Pass plan here too

        mock_agent_provider.get.assert_awaited_once_with("agent_a")
        agent_a_mock.assert_awaited_once()
        agent_b_mock.assert_not_awaited()
        mock_state_manager.get_or_create_current_run_id.assert_called_once()
        mock_state_manager.save_paused_flow_state.assert_called_once()

        saved_paused_details: PausedRunDetails = mock_state_manager.save_paused_flow_state.call_args[0][0]
        assert isinstance(saved_paused_details, PausedRunDetails)
        assert saved_paused_details.run_id == str(test_run_id)
        assert saved_paused_details.flow_id == basic_plan.id # Verify flow_id is saved
        assert saved_paused_details.paused_at_stage_id == "stage0"
        assert saved_paused_details.reason == "Paused due to agent error in master stage"
        assert saved_paused_details.context_snapshot == {"input": "value", "outputs": {}, "visited": ["stage0"]} # Initial context + empty outputs
        assert isinstance(saved_paused_details.error_details, AgentErrorDetails)
        assert saved_paused_details.error_details.error_type == "ValueError"
        assert saved_paused_details.error_details.message == "Agent A failed spectacularly!"
        assert saved_paused_details.error_details.agent_id == "agent_a"
        assert saved_paused_details.error_details.stage_id == "stage0"
        assert "Traceback (most recent call last):" in saved_paused_details.error_details.traceback

        mock_state_manager.update_status.assert_called_once()
        update_call_args = mock_state_manager.update_status.call_args
        # Check args passed to update_status
        passed_kwargs = update_call_args[1]
        assert passed_kwargs['stage'] == 0.0 # Check stage number passed correctly
        assert passed_kwargs['status'] == StageStatus.FAILURE.value
        # Reason includes agent ID and specific error
        expected_reason_substr = f"PAUSED_ON_ERROR: Agent 'agent_a' failed: ValueError"
        assert expected_reason_substr in passed_kwargs['reason']
        assert isinstance(passed_kwargs['error_details'], AgentErrorDetails)

    async def test_run_pause_state_save_failure(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test behavior when saving pause state fails after an agent exception."""
        agent_a_mock = AsyncMock(side_effect=RuntimeError("Agent A Error"))
        mock_agent_provider.get.return_value = agent_a_mock

        test_run_id = 456
        mock_state_manager.get_or_create_current_run_id.return_value = test_run_id
        mock_state_manager.save_paused_flow_state.return_value = False

        initial_context = {}
        await orchestrator.run(basic_plan, initial_context.copy())

        mock_state_manager.save_paused_flow_state.assert_called_once()
        mock_state_manager.update_status.assert_called_once()

        update_call_args = mock_state_manager.update_status.call_args
        passed_kwargs = update_call_args[1]
        assert passed_kwargs['stage'] == 0.0
        assert passed_kwargs['status'] == StageStatus.FAILURE.value
        # Updated assertion to check for substring containing agent_id
        assert "PAUSED_ON_ERROR (Save Failed): Agent 'agent_a' failed: RuntimeError" in passed_kwargs['reason']
        assert isinstance(passed_kwargs['error_details'], AgentErrorDetails)
        assert passed_kwargs['error_details'].error_type == "RuntimeError"

    # --- Tests for AW1.5 (Resumption Logic) ---

    @pytest.fixture
    def paused_run_details(self) -> PausedRunDetails:
        """Fixture for a sample PausedRunDetails object."""
        return PausedRunDetails(
            run_id="test_run_123",
            flow_id="test_plan_basic", # Add the flow_id here
            paused_at_stage_id="stage1",
            timestamp=datetime.now(timezone.utc),
            context_snapshot={"input": "original", "outputs": {"stage0": {"data": "from_stage0"}}},
            error_details=AgentErrorDetails(
                error_type="ValueError",
                message="Something failed",
                traceback="Traceback...",
                agent_id="agent_b",
                stage_id="stage1"
            ),
            reason="Paused due to agent error"
        )

    async def test_resume_flow_retry(self, orchestrator_for_resume, mock_state_manager, paused_run_details, basic_plan):
        """Test resuming with 'retry' action."""
        run_id = paused_run_details.run_id
        paused_stage_id = paused_run_details.paused_at_stage_id # "stage1"
        # original_context = paused_run_details.context_snapshot # Context comes from details directly

        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed load_context mock
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed from clear_paused_run_details

        # Mock the execute_loop to verify it's called correctly
        orchestrator_for_resume._execute_loop = AsyncMock(return_value={"final": "context"})

        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="retry")

        mock_state_manager.load_paused_flow_state.assert_called_once_with(run_id)
        # mock_state_manager.load_context.assert_called_once_with(run_id) # Removed load_context assertion
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed from clear_paused_run_details
        orchestrator_for_resume._execute_loop.assert_awaited_once_with(
            start_stage_name=paused_stage_id, 
            context=paused_run_details.context_snapshot # Expect context from details
        )
        assert result == {"final": "context"} # Check result from _execute_loop mock

    async def test_resume_flow_retry_with_inputs_valid(self, orchestrator_for_resume, mock_state_manager, paused_run_details):
        """Test resuming with 'retry_with_inputs' action and valid inputs."""
        run_id = paused_run_details.run_id
        paused_stage_id = paused_run_details.paused_at_stage_id
        original_context = paused_run_details.context_snapshot
        new_inputs = {"input": "updated", "extra_param": True}
        expected_context = copy.deepcopy(original_context) # Use deepcopy if context might be mutated
        expected_context.update(new_inputs) # Simple dict update logic

        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed
        orchestrator_for_resume._execute_loop = AsyncMock(return_value={"final": "context_updated"})

        result = await orchestrator_for_resume.resume_flow(
            run_id=run_id, action="retry_with_inputs", action_data={"inputs": new_inputs}
        )

        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed
        orchestrator_for_resume._execute_loop.assert_awaited_once_with(
            start_stage_name=paused_stage_id,
            context=expected_context # Check context with updated inputs
        )
        assert result == {"final": "context_updated"}

    async def test_resume_flow_retry_with_inputs_invalid(self, orchestrator_for_resume, mock_state_manager, paused_run_details):
        """Test resuming with 'retry_with_inputs' action but invalid/missing inputs data."""
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock() # Changed
        orchestrator_for_resume._execute_loop = AsyncMock() # ADDED: Ensure it's mocked for assertion
    
        # Case 1: Missing 'inputs' key
        result_missing = await orchestrator_for_resume.resume_flow(
            run_id=run_id, action="retry_with_inputs", action_data={"other_key": "value"}
        )
        assert "error" in result_missing
        assert "requires a dictionary under the 'inputs' key" in result_missing["error"]
        mock_state_manager.delete_paused_flow_state.assert_not_called() # Changed
        orchestrator_for_resume._execute_loop.assert_not_awaited()
    
        # Case 2: 'inputs' key is not a dictionary
        result_wrong_type = await orchestrator_for_resume.resume_flow(
            run_id=run_id, action="retry_with_inputs", action_data={"inputs": "not_a_dict"}
        )
        assert "error" in result_wrong_type
        assert "requires a dictionary under the 'inputs' key" in result_wrong_type["error"]
        # Ensure mocks weren't called from previous failure if state wasn't reset (it shouldn't be)
        mock_state_manager.delete_paused_flow_state.assert_not_called() # Changed 
        orchestrator_for_resume._execute_loop.assert_not_awaited()


    async def test_resume_flow_skip_stage_to_next(self, orchestrator_for_resume, mock_state_manager, paused_run_details, basic_plan):
        """Test resuming with 'skip_stage' where there is a next stage."""
        # Modify paused details to pause at stage0, which has a next stage (stage1)
        paused_run_details.paused_at_stage_id = "stage0"
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
        expected_next_stage = "stage1" # From basic_plan
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed
        orchestrator_for_resume._execute_loop = AsyncMock(return_value={"final": "context_skipped"})
    
        # Inject basic_plan into the orchestrator instance for this test
        orchestrator_for_resume.pipeline_def = basic_plan
    
        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="skip_stage")
    
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed
        orchestrator_for_resume._execute_loop.assert_awaited_once_with(
            start_stage_name=expected_next_stage,
            context=original_context
        )
        assert result == {"final": "context_skipped"}


    async def test_resume_flow_skip_stage_to_end(self, orchestrator_for_resume, mock_state_manager, paused_run_details, basic_plan):
        """Test resuming with 'skip_stage' where the skipped stage is the last one."""
        # Paused at stage1 (last stage in basic_plan)
        paused_run_details.paused_at_stage_id = "stage1"
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed
        # orchestrator_for_resume._execute_loop = AsyncMock() # Loop not called
    
        # Inject basic_plan
        orchestrator_for_resume.pipeline_def = basic_plan
    
        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="skip_stage")
    
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed
        orchestrator_for_resume._execute_loop.assert_not_awaited() # Loop should not run
        # The function should return the context directly when skipping the last stage
        assert result == original_context


    async def test_resume_flow_force_branch_valid(self, orchestrator_for_resume, mock_state_manager, paused_run_details, basic_plan):
        """Test resuming with 'force_branch' to a valid stage."""
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
        target_stage = "stage0" # Branch back to the start (or any valid stage)
    
        assert target_stage in basic_plan.stages # Ensure target is actually in the plan
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed
        orchestrator_for_resume._execute_loop = AsyncMock(return_value={"final": "context_branched"})
    
        # Inject basic_plan
        orchestrator_for_resume.pipeline_def = basic_plan
    
        result = await orchestrator_for_resume.resume_flow(
            run_id=run_id, action="force_branch", action_data={"target_stage_id": target_stage}
        )
    
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed
        orchestrator_for_resume._execute_loop.assert_awaited_once_with(
            start_stage_name=target_stage,
            context=original_context
        )
        assert result == {"final": "context_branched"}


    async def test_resume_flow_force_branch_invalid(self, orchestrator_for_resume, mock_state_manager, paused_run_details, basic_plan):
        """Test resuming with 'force_branch' to an invalid/non-existent stage."""
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
        invalid_target = "stage_does_not_exist"
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock() # Changed
        # orchestrator_for_resume._execute_loop = AsyncMock() # Loop not called
    
        # Inject basic_plan
        orchestrator_for_resume.pipeline_def = basic_plan
    
        result = await orchestrator_for_resume.resume_flow(
            run_id=run_id, action="force_branch", action_data={"target_stage_id": invalid_target}
        )
    
        assert "error" in result
        assert "Invalid target_stage_id" in result["error"]
        mock_state_manager.delete_paused_flow_state.assert_not_called() # Changed
        orchestrator_for_resume._execute_loop.assert_not_awaited()


    async def test_resume_flow_abort(self, orchestrator_for_resume, mock_state_manager, paused_run_details):
        """Test resuming with 'abort' action."""
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
        expected_context = original_context.copy()
        expected_context["status"] = "ABORTED" # Check for the status marker
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(return_value=True) # Changed
        # orchestrator_for_resume._execute_loop = AsyncMock() # Loop not called
    
        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="abort")
    
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id) # Changed
        orchestrator_for_resume._execute_loop.assert_not_awaited() # Loop should not run
        assert result == expected_context


    async def test_resume_flow_run_not_found(self, orchestrator_for_resume, mock_state_manager):
        """Test attempting to resume a run_id that doesn't exist or isn't paused."""
        run_id = "non_existent_run"
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=None)
        # mock_state_manager.load_context = MagicMock() # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock() # Changed
        # orchestrator_for_resume._execute_loop = AsyncMock() # Loop not called
    
        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="retry")
    
        assert "error" in result
        assert "No paused run found" in result["error"]
        # mock_state_manager.load_context.assert_not_called() # Removed
        mock_state_manager.delete_paused_flow_state.assert_not_called() # Changed
        orchestrator_for_resume._execute_loop.assert_not_awaited()


    async def test_resume_flow_context_load_failure(self, orchestrator_for_resume, mock_state_manager, paused_run_details):
        """Test failure during context loading after finding paused details.
           NOTE: This case is currently impossible as context comes directly from paused_details.
                 Keeping the test structure in case behavior changes.
        """
        run_id = paused_run_details.run_id
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # Simulate context being None *within* the paused_details, although load_context isn't called
        paused_run_details.context_snapshot = None 
        mock_state_manager.delete_paused_flow_state = MagicMock() # Changed
        # orchestrator_for_resume._execute_loop = AsyncMock() # Explicitly mock _execute_loop to not return a value (override fixture)
        orchestrator_for_resume._execute_loop = AsyncMock() # Loop not reached anyway

        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="retry")

        # The code currently WARNS and uses empty context, it does not error out.
        # Asserting the warning would require capturing logs.
        # For now, check that it proceeds and tries to delete state/call loop
        # assert "error" in result # No error is returned
        # assert "Context not found for paused run_id" in result["error"]
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id)
        orchestrator_for_resume._execute_loop.assert_awaited_once_with(
            start_stage_name=paused_run_details.paused_at_stage_id,
            context={}
        )


    async def test_resume_flow_clear_state_failure(self, orchestrator_for_resume, mock_state_manager, paused_run_details):
        """Test failure during clearing of the paused state."""
        run_id = paused_run_details.run_id
        original_context = paused_run_details.context_snapshot
    
        mock_state_manager.load_paused_flow_state = MagicMock(return_value=paused_run_details)
        # mock_state_manager.load_context = MagicMock(return_value=original_context) # Removed
        mock_state_manager.delete_paused_flow_state = MagicMock(side_effect=RuntimeError("DB connection lost"), return_value=False) # Changed & set side_effect
        orchestrator_for_resume._execute_loop = AsyncMock() # Mock loop, it might be called depending on error handling
    
        result = await orchestrator_for_resume.resume_flow(run_id=run_id, action="retry")
    
        # Check that the error from delete_paused_flow_state is caught and returned
        assert "error" in result
        assert "Failed to clear paused state" in result["error"]
        mock_state_manager.delete_paused_flow_state.assert_called_once_with(run_id)
        orchestrator_for_resume._execute_loop.assert_not_awaited() # Loop should not be reached if clear fails

    # --- New Tests for MasterExecutionPlan Context Passing Enhancements (P2.3.2) ---

    async def test_mep_original_request_in_context(self, orchestrator, mock_agent_provider, basic_plan):
        """Test that MasterExecutionPlan.original_request is injected into the context."""
        original_req_obj = UserGoalRequest(goal_id="goal123", goal_description="Test Goal")
        mep = MasterExecutionPlan(
            id="mep_orig_req_test",
            name="Test Original Request",
            start_stage="stage_A",
            stages={
                "stage_A": MasterStageSpec(
                    agent_id="agent_check_orig_req", 
                    inputs={"some_input": "context.original_request.goal_description"},
                    next_stage=None, 
                    number=0.0
                )
            },
            original_request=original_req_obj
        )
        orchestrator.pipeline_def = mep # Set the MEP for the orchestrator

        agent_mock = AsyncMock(return_value={"result": "ok"})
        mock_agent_provider.get.return_value = agent_mock

        initial_context = {"user_param": "initial_user_data"}
        final_context = await orchestrator.run(mep, initial_context.copy())

        agent_mock.assert_awaited_once()
        # The agent_call_context includes resolved inputs overlaid on the base context for resolution.
        # base_context_for_stage_resolution contains 'original_request'.
        # stage_specific_resolved_inputs contains {"some_input": "Test Goal"}
        # So, agent_call_context will have both.
        
        call_args_list = agent_mock.call_args_list
        assert len(call_args_list) == 1
        agent_call_context = call_args_list[0][0][0] # First arg of first call

        assert agent_call_context.get("original_request") == original_req_obj
        assert agent_call_context.get("some_input") == "Test Goal"
        assert agent_call_context.get("user_param") == "initial_user_data"
        assert final_context['outputs']['stage_A'] == {"result": "ok"}

    async def test_mep_previous_stage_outputs_special_key(self, orchestrator, mock_agent_provider):
        """Test the special 'previous_stage_outputs': '<stage_id>' mechanism."""
        mep = MasterExecutionPlan(
            id="mep_prev_outputs_special",
            name="Test Previous Outputs Special",
            start_stage="stage_P1",
            stages={
                "stage_P1": MasterStageSpec(
                    agent_id="agent_p1", 
                    inputs={"task": "generate data"}, 
                    next_stage="stage_P2", 
                    number=0.0
                ),
                "stage_P2": MasterStageSpec(
                    agent_id="agent_p2", 
                    inputs={
                        "previous_stage_outputs": "stage_P1", # Special key
                        "another_param": "context.resolved_previous_stage_output_data.key1" 
                    },
                    next_stage=None, 
                    number=1.0
                )
            }
        )
        orchestrator.pipeline_def = mep

        agent_p1_mock = AsyncMock(return_value={"key1": "value1", "key2": "value2"})
        agent_p2_mock = AsyncMock(return_value={"processed": True})
        mock_agent_provider.get.side_effect = [agent_p1_mock, agent_p2_mock]

        await orchestrator.run(mep, {})

        agent_p1_mock.assert_awaited_once_with( # Context for P1
            {
                'original_request': None, # default from orchestrator.run if plan has no original_request
                'outputs': {},
                'errors': {},
                'status_updates': [],
                'visited': ['stage_P1'],
                'run_id': '0', # from mock_state_manager
                'flow_id': 'mep_prev_outputs_special',
                'task': 'generate data'
            }
        )
        
        # Check context for P2
        call_args_list_p2 = agent_p2_mock.call_args_list
        assert len(call_args_list_p2) == 1
        agent_p2_call_context = call_args_list_p2[0][0][0]

        assert agent_p2_call_context.get('resolved_previous_stage_output_data') == {"key1": "value1", "key2": "value2"}
        assert agent_p2_call_context.get('another_param') == "value1"
        # 'previous_stage_outputs' key itself should be removed from direct inputs if it was an ID.
        assert "previous_stage_outputs" not in agent_p2_call_context 
        # ... unless it was defined as a context path, which is tested elsewhere.

    async def test_mep_context_path_outputs_resolution(self, orchestrator, mock_agent_provider):
        """Test resolving general 'context.outputs.<stage>.<key>'."""
        mep = MasterExecutionPlan(
            id="mep_ctx_path_outputs",
            name="Test Context Path Outputs",
            start_stage="stage_C1",
            stages={
                "stage_C1": MasterStageSpec(agent_id="agent_c1", inputs={"data": "c1_data"}, next_stage="stage_C2", number=0.0),
                "stage_C2": MasterStageSpec(
                    agent_id="agent_c2", 
                    inputs={"input_from_c1": "context.outputs.stage_C1.output_val"},
                    next_stage=None, 
                    number=1.0
                )
            }
        )
        orchestrator.pipeline_def = mep

        agent_c1_mock = AsyncMock(return_value={"output_val": "data_from_c1"})
        agent_c2_mock = AsyncMock(return_value={})
        mock_agent_provider.get.side_effect = [agent_c1_mock, agent_c2_mock]

        await orchestrator.run(mep, {})

        # Check context for C2
        call_args_list_c2 = agent_c2_mock.call_args_list
        assert len(call_args_list_c2) == 1
        agent_c2_call_context = call_args_list_c2[0][0][0]
        assert agent_c2_call_context.get('input_from_c1') == "data_from_c1"

    async def test_mep_mixed_literal_and_context_inputs(self, orchestrator, mock_agent_provider):
        """Test a stage with mixed literal values and context path lookups in inputs."""
        original_req = UserGoalRequest(goal_id="mixed_goal", goal_description="Mixed test")
        mep = MasterExecutionPlan(
            id="mep_mixed_inputs",
            name="Test Mixed Inputs",
            original_request=original_req,
            start_stage="stage_M1",
            stages={
                "stage_M1": MasterStageSpec(agent_id="agent_m1", inputs={"val": 100}, next_stage="stage_M2", number=0.0),
                "stage_M2": MasterStageSpec(
                    agent_id="agent_m2", 
                    inputs={
                        "literal_str": "hello_world",
                        "literal_int": 42,
                        "from_original_req": "context.original_request.goal_id",
                        "from_m1_output": "context.outputs.stage_M1.m1_result",
                        "nested_m1_output": "context.outputs.stage_M1.deep.nested_val"
                    },
                    next_stage=None, 
                    number=1.0
                )
            }
        )
        orchestrator.pipeline_def = mep

        agent_m1_mock = AsyncMock(return_value={"m1_result": "m1_done", "deep": {"nested_val": "deep_value"}})
        agent_m2_mock = AsyncMock(return_value={})
        mock_agent_provider.get.side_effect = [agent_m1_mock, agent_m2_mock]

        await orchestrator.run(mep, {"initial_ctx_val": "start"})

        call_args_list_m2 = agent_m2_mock.call_args_list
        assert len(call_args_list_m2) == 1
        agent_m2_call_context = call_args_list_m2[0][0][0]

        assert agent_m2_call_context.get('literal_str') == "hello_world"
        assert agent_m2_call_context.get('literal_int') == 42
        assert agent_m2_call_context.get('from_original_req') == "mixed_goal"
        assert agent_m2_call_context.get('from_m1_output') == "m1_done"
        assert agent_m2_call_context.get('nested_m1_output') == "deep_value"
        assert agent_m2_call_context.get('initial_ctx_val') == "start" # from initial run context
        assert agent_m2_call_context.get('original_request') == original_req # direct access to original_request object

    async def test_mep_previous_stage_outputs_as_context_path(self, orchestrator, mock_agent_provider):
        """Test 'previous_stage_outputs' used as a context path, not a direct ID."""
        mep = MasterExecutionPlan(
            id="mep_prev_out_path",
            name="Test Prev Output as Path",
            start_stage="stage_S1",
            stages={
                "stage_S1": MasterStageSpec(agent_id="agent_s1", inputs={"param": "stage_id_provider"}, next_stage="stage_S2", number=0.0),
                "stage_S2": MasterStageSpec(
                    agent_id="agent_s2", 
                    inputs={
                        "task_data": "use previous output data", 
                        # This will be resolved as a path to the string 'actual_prev_stage_id'
                        "previous_stage_outputs": "context.outputs.stage_S1.provided_stage_id" 
                    },
                    next_stage="stage_S3", 
                    number=1.0
                ),
                "stage_S3": MasterStageSpec(
                    agent_id="agent_s3",
                    inputs={
                        # This uses the SPECIAL mechanism because 'actual_prev_stage_id' is a direct string ID
                        "previous_stage_outputs": "actual_prev_stage_id",
                        "check_resolved": "context.resolved_previous_stage_output_data.s3_input_key"
                    },
                    next_stage=None,
                    number=2.0
                )
            }
        )
        orchestrator.pipeline_def = mep

        # Mock agent for stage_S1 that will output the ID of another stage
        # (this other stage 'actual_prev_stage_id' is not in the MEP, but its output will be injected into context manually for testing)
        agent_s1_mock = AsyncMock(return_value={"provided_stage_id": "actual_prev_stage_id"})
        agent_s2_mock = AsyncMock(return_value={"s2_done": True})
        agent_s3_mock = AsyncMock(return_value={"s3_done": True})
        mock_agent_provider.get.side_effect = [agent_s1_mock, agent_s2_mock, agent_s3_mock]

        initial_context_for_run = {
            "outputs": {
                "actual_prev_stage_id": {"s3_input_key": "data_for_s3"} # Pre-populate output for the dynamic ID
            }
        }

        await orchestrator.run(mep, initial_context_for_run)

        # Check context for S2
        call_args_list_s2 = agent_s2_mock.call_args_list
        assert len(call_args_list_s2) == 1
        agent_s2_call_context = call_args_list_s2[0][0][0]
        
        # 'previous_stage_outputs' input should be resolved to the string "actual_prev_stage_id"
        assert agent_s2_call_context.get('previous_stage_outputs') == "actual_prev_stage_id"
        # The special 'resolved_previous_stage_output_data' should NOT be populated for S2
        # because 'previous_stage_outputs' was a context path, not a direct ID string.
        assert 'resolved_previous_stage_output_data' not in agent_s2_call_context

        # Check context for S3
        call_args_list_s3 = agent_s3_mock.call_args_list
        assert len(call_args_list_s3) == 1
        agent_s3_call_context = call_args_list_s3[0][0][0]

        # For S3, 'previous_stage_outputs' was a direct ID, so special mechanism applies.
        assert agent_s3_call_context.get('resolved_previous_stage_output_data') == {"s3_input_key": "data_for_s3"}
        assert agent_s3_call_context.get('check_resolved') == "data_for_s3"
        # The key 'previous_stage_outputs' itself (value: 'actual_prev_stage_id') should be removed from s3's direct inputs
        assert "previous_stage_outputs" not in agent_s3_call_context

    async def test_mep_context_path_resolution_failure_fallback(self, orchestrator, mock_agent_provider):
        """Test that if a context path cannot be resolved, it falls back to the literal string path."""
        mep = MasterExecutionPlan(
            id="mep_path_fail",
            name="Test Path Fail Fallback",
            start_stage="stage_F1",
            stages={
                "stage_F1": MasterStageSpec(
                    agent_id="agent_f1", 
                    inputs={
                        "non_existent_path": "context.outputs.no_such_stage.no_such_key",
                        "partially_valid_path_to_non_attr": "context.outputs.stage_F1.some_list.0.non_existent_attr", # if stage_F1 outputs a list
                        "path_to_primitive": "context.outputs.stage_F1.a_string.length" # access attr on primitive
                    },
                    next_stage=None, 
                    number=0.0
                )
            }
        )
        orchestrator.pipeline_def = mep

        agent_f1_mock = AsyncMock(return_value={"some_list": [{"item_key": "item_val"}], "a_string": "test_string"})
        mock_agent_provider.get.return_value = agent_f1_mock

        # Mock logger to check for warnings
        with patch.object(orchestrator.logger, 'warning') as mock_log_warning:
            await orchestrator.run(mep, {})

            call_args_list_f1 = agent_f1_mock.call_args_list
            assert len(call_args_list_f1) == 1
            agent_f1_call_context = call_args_list_f1[0][0][0]

            assert agent_f1_call_context.get('non_existent_path') == "context.outputs.no_such_stage.no_such_key"
            assert agent_f1_call_context.get('partially_valid_path_to_non_attr') == "context.outputs.stage_F1.some_list.0.non_existent_attr"
            assert agent_f1_call_context.get('path_to_primitive') == "context.outputs.stage_F1.a_string.length"
            
            # Verify warnings were logged for each failed resolution
            # The number of warnings might vary depending on how deep the path resolution fails.
            # For "context.outputs.no_such_stage.no_such_key", it fails at "no_such_stage".
            # For "context.outputs.stage_F1.some_list.0.non_existent_attr", after resolving "some_list[0]", "non_existent_attr" fails.
            # For "context.outputs.stage_F1.a_string.length", after resolving "a_string", "length" fails (if not a direct attr).
            assert mock_log_warning.call_count >= 3 # Expect at least one warning per failed path
            
            # Example check for specific warning messages (optional, can be brittle)
            # warning_messages = [c[0][0] for c in mock_log_warning.call_args_list]
            # assert any("Could not resolve context path 'context.outputs.no_such_stage.no_such_key'" in msg for msg in warning_messages)
            # assert any("Path part 'non_existent_attr' not found" in msg for msg in warning_messages)
            # assert any("Path part 'length' not found" in msg for msg in warning_messages)

    # --- New Tests for P2.3.2a: Orchestrator Artifact Path Handling ---

    async def test_artifact_path_extraction_valid_paths(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test artifact paths are correctly extracted and passed to state_manager."""
        artifact_paths = ["path/to/artifact1.txt", "another/path/report.pdf"]
        agent_a_mock = AsyncMock(return_value={
            "output_a": "value_a",
            "_mcp_generated_artifacts_relative_paths_": artifact_paths
        })
        agent_b_mock = AsyncMock(return_value={"output_b": "value_b"}) # No artifacts from b
        mock_agent_provider.get.side_effect = [agent_a_mock, agent_b_mock]

        orchestrator.pipeline_def = basic_plan # Ensure the correct plan is set
        await orchestrator.run(basic_plan, {})

        mock_state_manager.update_status.assert_any_call(
            pipeline_run_id=ANY,  # Allow any run ID
            stage_name="stage0",    # As per basic_plan
            stage_number=0.0,     # As per basic_plan
            status=StageStatus.SUCCESS.value,
            reason=ANY,           # Allow any success reason
            error_info=None,      # Expect None for success
            artifacts=artifact_paths
        )

        # Check for the second stage's status update (agent_b, no artifacts)
        mock_state_manager.update_status.assert_any_call(
            stage=1.0, status=StageStatus.SUCCESS.value, artifacts=[] # Agent B produces no artifacts
        )
        # Ensure SUCCESS is used instead of PASS
        assert mock_state_manager.update_status.call_args_list[0][1]['status'] == StageStatus.SUCCESS.value
        assert mock_state_manager.update_status.call_args_list[1][1]['status'] == StageStatus.SUCCESS.value


    async def test_artifact_path_extraction_empty_list(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test extraction when agent returns an empty list for artifacts."""
        agent_a_mock = AsyncMock(return_value={
            "output_a": "value_a",
            "_mcp_generated_artifacts_relative_paths_": []
        })
        mock_agent_provider.get.return_value = agent_a_mock # Only one stage needed

        plan_one_stage = MasterExecutionPlan(
            id="one_stage_plan", start_stage="stage0", stages={
                "stage0": MasterStageSpec(agent_id="agent_a", next_stage=None, number=0.0)
            }
        )
        orchestrator.pipeline_def = plan_one_stage
        await orchestrator.run(plan_one_stage, {})

        mock_state_manager.update_status.assert_called_once_with(
            pipeline_run_id=ANY,
            stage_name="stage0",
            stage_number=0.0,
            status=StageStatus.SUCCESS.value,
            reason=ANY,
            error_info=None,
            artifacts=[]
        )

    async def test_artifact_path_extraction_key_not_a_list(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test extraction when artifact key's value is not a list."""
        agent_a_mock = AsyncMock(return_value={
            "output_a": "value_a",
            "_mcp_generated_artifacts_relative_paths_": "not_a_list_value" # Incorrect type
        })
        mock_agent_provider.get.return_value = agent_a_mock

        plan_one_stage = MasterExecutionPlan(
            id="one_stage_plan_alt", start_stage="stage0", stages={
                "stage0": MasterStageSpec(agent_id="agent_a", next_stage=None, number=0.0)
            }
        )
        orchestrator.pipeline_def = plan_one_stage
        
        with patch.object(orchestrator.logger, 'warning') as mock_log_warning:
            await orchestrator.run(plan_one_stage, {})

            mock_state_manager.update_status.assert_called_once_with(
                stage=0.0, status=StageStatus.SUCCESS.value, artifacts=[] # Should default to empty list
            )
            mock_log_warning.assert_called_once()
            assert "was not a list" in mock_log_warning.call_args[0][0]

    async def test_artifact_path_extraction_list_with_non_strings(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test extraction when artifact list contains non-string items."""
        mixed_artifact_list = ["valid/path.txt", 123, {"obj": "path"}, None, "another/valid.md"]
        expected_filtered_paths = ["valid/path.txt", "another/valid.md"]
        agent_a_mock = AsyncMock(return_value={
            "output_a": "value_a",
            "_mcp_generated_artifacts_relative_paths_": mixed_artifact_list
        })
        mock_agent_provider.get.return_value = agent_a_mock
        plan_one_stage = MasterExecutionPlan(
            id="one_stage_plan_mixed", start_stage="stage0", stages={
                "stage0": MasterStageSpec(agent_id="agent_a", next_stage=None, number=0.0)
            }
        )
        orchestrator.pipeline_def = plan_one_stage

        with patch.object(orchestrator.logger, 'warning') as mock_log_warning:
            await orchestrator.run(plan_one_stage, {})

            mock_state_manager.update_status.assert_called_once_with(
                stage=0.0, status=StageStatus.SUCCESS.value, artifacts=expected_filtered_paths
            )
            # Expect 3 warnings: one for 123, one for {"obj": "path"}, one for None
            assert mock_log_warning.call_count == 3 
            # Check if one of the warnings contains relevant info (optional, can be brittle)
            # warning_messages = [c[0][0] for c in mock_log_warning.call_args_list]
            # assert any("provided a non-string path" in msg for msg in warning_messages)


    async def test_artifact_path_extraction_key_missing(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test extraction when artifact key is missing from agent output."""
        agent_a_mock = AsyncMock(return_value={
            "output_a": "value_a" # No artifact key
        })
        mock_agent_provider.get.return_value = agent_a_mock
        plan_one_stage = MasterExecutionPlan(
            id="one_stage_plan_no_key", start_stage="stage0", stages={
                "stage0": MasterStageSpec(agent_id="agent_a", next_stage=None, number=0.0)
            }
        )
        orchestrator.pipeline_def = plan_one_stage
        
        with patch.object(orchestrator.logger, 'warning') as mock_log_warning: # Ensure no warnings for this valid case
            await orchestrator.run(plan_one_stage, {})

            mock_state_manager.update_status.assert_called_once_with(
                stage=0.0, status=StageStatus.SUCCESS.value, artifacts=[] # Default to empty list
            )
            mock_log_warning.assert_not_called()
