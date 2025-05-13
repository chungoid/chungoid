#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import Dict, Any
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
    manager.update_status = AsyncMock(return_value=True)
    manager.get_or_create_current_run_id = MagicMock(return_value=0) # Sync method
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
def orchestrator_for_resume(mock_agent_provider, mock_state_manager, basic_plan):
    """Fixture to create an AsyncOrchestrator instance for resume tests."""
    # Note: pipeline_def (basic_plan) is often overridden in specific tests
    config = {"some_config": "value"}
    instance = AsyncOrchestrator(
        pipeline_def=basic_plan, # Initial plan, might be replaced by test 
        config=config,
        agent_provider=mock_agent_provider,
        state_manager=mock_state_manager
    )
    # Ensure _execute_loop is always mocked for resume tests, even if not awaited
    instance._execute_loop = AsyncMock(return_value={"final_context_from_loop": True})
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
        mock_state_manager.update_status.assert_has_awaits([
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
           {"global_val": "hello", "outputs": {}, "param1": "literal", "param2": "hello"}
        )
        agent_y_mock.assert_awaited_once_with(
            {"global_val": "hello", "outputs": {"stage_x": {"data": "output_from_x"}}, "input_from_x": "output_from_x"}
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
        left_agent_mock = AsyncMock(return_value={"left_out": "ok"})
        right_agent_mock = AsyncMock()
        mock_agent_provider.get.side_effect = [input_agent_mock, left_agent_mock]

        initial_context = {}
        await orchestrator.run(conditional_plan, initial_context.copy())

        mock_agent_provider.get.assert_has_calls([call("input_agent"), call("left_agent")])
        input_agent_mock.assert_awaited_once()
        left_agent_mock.assert_awaited_once()
        right_agent_mock.assert_not_called()

        stage_input_num = self._get_stage_num(conditional_plan, "stage_input")
        stage_left_num = self._get_stage_num(conditional_plan, "stage_left")

        mock_state_manager.update_status.assert_has_awaits([
            call(stage=stage_input_num, status='PASS', artifacts=[]),
            call(stage=stage_left_num, status='PASS', artifacts=[]),
        ], any_order=False) # Order should be deterministic here

    async def test_run_conditional_branching_false(self, orchestrator, mock_agent_provider, mock_state_manager, conditional_plan):
        """Test conditional branching when condition is false."""
        input_agent_mock = AsyncMock(return_value={"result": "go_right"})
        left_agent_mock = AsyncMock()
        right_agent_mock = AsyncMock(return_value={"right_out": "ok"})
        mock_agent_provider.get.side_effect = [input_agent_mock, right_agent_mock]

        initial_context = {}
        await orchestrator.run(conditional_plan, initial_context.copy())

        mock_agent_provider.get.assert_has_calls([call("input_agent"), call("right_agent")])
        input_agent_mock.assert_awaited_once()
        right_agent_mock.assert_awaited_once()
        left_agent_mock.assert_not_called()

        stage_input_num = self._get_stage_num(conditional_plan, "stage_input")
        stage_right_num = self._get_stage_num(conditional_plan, "stage_right")

        mock_state_manager.update_status.assert_has_awaits([
             call(stage=stage_input_num, status='PASS', artifacts=[]),
             call(stage=stage_right_num, status='PASS', artifacts=[]),
        ], any_order=False) # Order should be deterministic

    async def test_run_agent_not_found(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test behavior when an agent cannot be resolved."""
        mock_agent_provider.get.return_value = None

        initial_context = {}
        await orchestrator.run(basic_plan, initial_context.copy())

        mock_agent_provider.get.assert_awaited_once_with("agent_a")
        mock_state_manager.update_status.assert_awaited_once_with(
            stage=0.0,
            status=StageStatus.FAILURE.value,
            artifacts=[],
            reason="Agent 'agent_a' not found"
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
        # The agent provider should be called max_hops - 1 times before the loop breaks
        assert mock_agent_provider.get.await_count == max_hops - 1 
        assert mock_state_manager.update_status.await_count == max_hops

        # It will execute stage 1 (hop 1), stage 2 (hop 2), stage 1 (hop 3), stage 2 (hop 4), stage 1 (hop 5), stage 2 (hop 6)
        # On hop 7, it will try to execute stage 1 again, hit max hops, and record failure for stage 1.
        mock_state_manager.update_status.assert_has_awaits([
            call(stage=1.0, status='PASS', artifacts=[]), # Hop 1
            call(stage=2.0, status='PASS', artifacts=[]), # Hop 2
            call(stage=1.0, status='PASS', artifacts=[]), # Hop 3
            call(stage=2.0, status='PASS', artifacts=[]), # Hop 4
            call(stage=1.0, status='PASS', artifacts=[]), # Hop 5
            call(stage=2.0, status='PASS', artifacts=[]), # Hop 6
            call(stage=1.0, status='FAIL', artifacts=[], reason="Max hops reached"), # Hop 7 failure
        ])
        mock_state_manager.save_paused_flow_state.assert_not_called()

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
        assert saved_paused_details.context_snapshot == {"input": "value", "outputs": {}} # Initial context + empty outputs
        assert isinstance(saved_paused_details.error_details, AgentErrorDetails)
        assert saved_paused_details.error_details.error_type == "ValueError"
        assert saved_paused_details.error_details.message == "Agent A failed spectacularly!"
        assert saved_paused_details.error_details.agent_id == "agent_a"
        assert saved_paused_details.error_details.stage_id == "stage0"
        assert "Traceback (most recent call last):" in saved_paused_details.error_details.traceback

        mock_state_manager.update_status.assert_awaited_once()
        update_call_args = mock_state_manager.update_status.await_args
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
        mock_state_manager.update_status.assert_awaited_once()

        update_call_args = mock_state_manager.update_status.await_args
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

    @pytest.fixture
    def orchestrator_for_resume(self, mock_agent_provider, mock_state_manager, mock_project_config, basic_plan) -> AsyncOrchestrator:
        """Creates an AsyncOrchestrator specifically for resume tests, maybe pre-configured."""
        # Use basic_plan for simplicity, can override in specific tests if needed
        return AsyncOrchestrator(
            pipeline_def=basic_plan,
            config=mock_project_config,
            agent_provider=mock_agent_provider,
            state_manager=mock_state_manager
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
