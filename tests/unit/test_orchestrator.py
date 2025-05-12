#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from typing import Dict, Any
import traceback
from datetime import datetime, timezone

from chungoid.runtime.orchestrator import AsyncOrchestrator, ExecutionPlan, StageSpec
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.common_enums import StageStatus
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.schemas.flows import PausedRunDetails


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
def basic_plan() -> ExecutionPlan:
    """A simple linear execution plan."""
    return ExecutionPlan(
        id="test_plan_basic",
        start_stage="stage0",
        stages={
            "stage0": StageSpec(agent_id="agent_a", next_stage="stage1", number=0.0),
            "stage1": StageSpec(agent_id="agent_b", next_stage=None, number=1.0),
        }
    )

@pytest.fixture
def conditional_plan() -> ExecutionPlan:
    """A plan with conditional branching."""
    return ExecutionPlan(
        id="test_plan_conditional",
        start_stage="stage_input",
        stages={
            "stage_input": StageSpec(
                agent_id="input_agent",
                next_stage="stage_cond",
                number=-1.0 # Assign arbitrary number for testing if needed
            ),
            "stage_cond": StageSpec(
                agent_id="condition_checker", # Not actually called, condition uses context
                condition="outputs.stage_input.result == 'go_left'",
                next_stage_true="stage_left",
                next_stage_false="stage_right",
                number=-2.0
            ),
            "stage_left": StageSpec(agent_id="left_agent", next_stage=None, number=-3.0),
            "stage_right": StageSpec(agent_id="right_agent", next_stage=None, number=-4.0),
        }
    )


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
            call(stage=0.0, status=StageStatus.SUCCESS, artifacts=[]),
            call(stage=1.0, status=StageStatus.SUCCESS, artifacts=[]),
        ])

    async def test_run_context_passing_and_merging(self, orchestrator, mock_agent_provider, mock_state_manager):
        """Test context is passed correctly and outputs merged."""
        plan = ExecutionPlan(
            id="test_plan_context",
            start_stage="stage_x",
            stages={
                "stage_x": StageSpec(
                    agent_id="agent_x",
                    inputs={"param1": "literal", "param2": "context.global_val"},
                    next_stage="stage_y",
                    number=10.0
                ),
                "stage_y": StageSpec(
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
    def _get_stage_num(self, plan: ExecutionPlan, stage_name: str) -> float:
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
            call(stage=stage_input_num, status=StageStatus.SUCCESS, artifacts=[]),
            call(stage=stage_left_num, status=StageStatus.SUCCESS, artifacts=[]),
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
             call(stage=stage_input_num, status=StageStatus.SUCCESS, artifacts=[]),
             call(stage=stage_right_num, status=StageStatus.SUCCESS, artifacts=[]),
        ], any_order=False) # Order should be deterministic

    async def test_run_agent_not_found(self, orchestrator, mock_agent_provider, mock_state_manager, basic_plan):
        """Test behavior when an agent cannot be resolved."""
        mock_agent_provider.get.return_value = None

        initial_context = {}
        await orchestrator.run(basic_plan, initial_context.copy())

        mock_agent_provider.get.assert_awaited_once_with("agent_a")
        mock_state_manager.update_status.assert_awaited_once_with(
            stage=0.0,
            status=StageStatus.FAILURE,
            artifacts=[],
            reason="Agent not found"
        )
        mock_state_manager.save_paused_flow_state.assert_not_called()

    async def test_run_max_hops_reached(self, orchestrator, mock_agent_provider, mock_state_manager):
        """Test execution stops if max hops are reached (potential loop)."""
        plan = ExecutionPlan(
            id="test_plan_loop",
            start_stage="stage_loop1",
            stages={
                "stage_loop1": StageSpec(agent_id="looper", next_stage="stage_loop2", number=1.0),
                "stage_loop2": StageSpec(agent_id="looper", next_stage="stage_loop1", number=2.0),
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
            call(stage=1.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 1
            call(stage=2.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 2
            call(stage=1.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 3
            call(stage=2.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 4
            call(stage=1.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 5
            call(stage=2.0, status=StageStatus.SUCCESS, artifacts=[]), # Hop 6
            call(stage=1.0, status=StageStatus.FAILURE, artifacts=[], reason="Max hops reached"), # Hop 7 failure
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
        final_context = await orchestrator.run(basic_plan, initial_context.copy())

        mock_agent_provider.get.assert_awaited_once_with("agent_a")
        agent_a_mock.assert_awaited_once()
        agent_b_mock.assert_not_awaited()
        mock_state_manager.get_or_create_current_run_id.assert_called_once()
        mock_state_manager.save_paused_flow_state.assert_called_once()

        saved_paused_details: PausedRunDetails = mock_state_manager.save_paused_flow_state.call_args[0][0]
        assert isinstance(saved_paused_details, PausedRunDetails)
        assert saved_paused_details.run_id == str(test_run_id)
        assert saved_paused_details.paused_at_stage_id == "stage0"
        assert saved_paused_details.reason == "Paused due to agent error"
        assert saved_paused_details.context_snapshot == {"input": "value", "outputs": {}}
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
        assert passed_kwargs['reason'] == "PAUSED_ON_ERROR: Agent execution failed: ValueError"
        assert isinstance(passed_kwargs['error_details'], AgentErrorDetails)
        assert passed_kwargs['error_details'].message == "Agent A failed spectacularly!"

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
        assert "PAUSED_ON_ERROR (Save Failed): Agent execution failed: RuntimeError" in passed_kwargs['reason']
        assert isinstance(passed_kwargs['error_details'], AgentErrorDetails)
        assert passed_kwargs['error_details'].error_type == "RuntimeError"
