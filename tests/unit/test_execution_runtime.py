import textwrap
import pytest
from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator, AsyncOrchestrator, StageSpec
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec


def _linear_yaml():
    return textwrap.dedent(
        """
        start_stage: greet
        stages:
          greet:
            agent_id: system-greeter
            inputs:
              message: "hello"
            next_stage: farewell
          farewell:
            agent_id: system-greeter
            inputs:
              message: "bye"
        """
    )


def _conditional_yaml():
    return textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            condition: "input > 5"
            next_stage_true: s2
            next_stage_false: s3
          s2:
            agent_id: a2
          s3:
            agent_id: a3
        """
    )


def _on_error_yaml():
    return textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            on_error:
              condition: "error_code == 404"
              "true": s2
              "false": s3
          s2:
            agent_id: a2
          s3:
            agent_id: a3
        """
    )


def _plugins_extra_yaml():
    return textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            plugins: ["log", "audit"]
            extra:
              custom: 42
            next_stage: s2
          s2:
            agent_id: a2
        """
    )


def test_execution_plan_parses():
    plan = ExecutionPlan.from_yaml(_linear_yaml(), flow_id="flow-xyz")
    assert plan.start_stage == "greet"
    assert set(plan.stages.keys()) == {"greet", "farewell"}
    assert plan.id == "flow-xyz"


def test_sync_orchestrator_runs_linear_flow():
    plan = ExecutionPlan.from_yaml(_linear_yaml())
    orch = SyncOrchestrator(project_config={})
    visited = orch.run(plan, context={})
    assert visited == ["greet", "farewell"]


def test_sync_orchestrator_conditional_next():
    plan = ExecutionPlan.from_yaml(_conditional_yaml())
    orch = SyncOrchestrator(project_config={})
    # input > 5 (input=10)
    visited = orch.run(plan, context={"input": 10})
    assert visited == ["s1", "s2"]
    # input <= 5 (input=3)
    visited2 = orch.run(plan, context={"input": 3})
    assert visited2 == ["s1", "s3"]


def test_sync_orchestrator_on_error_conditional():
    plan = ExecutionPlan.from_yaml(_on_error_yaml())
    orch = SyncOrchestrator(project_config={})
    # error_code == 404
    visited = orch.run(plan, context={"error_code": 404})
    # Only s1 is visited, as on_error is not triggered in this simple run
    assert visited == ["s1"]


def test_sync_orchestrator_plugins_and_extra():
    plan = ExecutionPlan.from_yaml(_plugins_extra_yaml())
    s1 = plan.stages["s1"]
    assert s1.plugins == ["log", "audit"]
    assert s1.extra == {"custom": 42}


def test_sync_orchestrator_loop_guard():
    yaml_text = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            next_stage: s2
          s2:
            agent_id: a2
            next_stage: s1
        """
    )
    plan = ExecutionPlan.from_yaml(yaml_text)
    orch = SyncOrchestrator(project_config={})
    visited = orch.run(plan, context={})
    # Check against expected visited stages (internal max_hops = len(plan.stages) + 5 = 2 + 5 = 7)
    # Loop should execute 6 times before hops (7) >= max_hops (7) condition stops it *before* 7th execution.
    assert visited == ["s1", "s2", "s1", "s2", "s1", "s2"]


def test_schema_validation_missing_required():
    bad_yaml = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            # agent_id missing
            next_stage: s2
        """
    )
    with pytest.raises(ValueError) as exc:
        ExecutionPlan.from_yaml(bad_yaml)
    assert "agent_id" in str(exc.value)


def test_schema_validation_extra_field():
    bad_yaml = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            not_in_schema: 123
            next_stage: s2
        """
    )
    with pytest.raises(ValueError) as exc:
        ExecutionPlan.from_yaml(bad_yaml)
    assert "not_in_schema" in str(exc.value)


def test_schema_validation_invalid_type():
    bad_yaml = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: 123  # should be string
            next_stage: s2
        """
    )
    with pytest.raises(ValueError) as exc:
        ExecutionPlan.from_yaml(bad_yaml)
    assert "agent_id" in str(exc.value)


def test_sync_orchestrator_next_if():
    plan = ExecutionPlan.from_yaml(_conditional_yaml())
    orch = SyncOrchestrator(project_config={})
    visited = orch.run(plan, context={})
    # No context, so input > 5 is not met; should take 'false' branch
    assert visited == ["s1", "s3"]


@pytest.fixture
def mock_agent_provider() -> MagicMock:
    return MagicMock(spec=AgentProvider)

@pytest.fixture
def mock_state_manager() -> MagicMock:
    manager = MagicMock(spec=StateManager)
    manager.update_status = AsyncMock()
    return manager

@pytest.fixture
def mock_config() -> dict:
    return {"logging": {"level": "INFO"}}

# Example Flow YAML for testing
FLOW_YAML_SIMPLE = """
start_stage: stage1
stages:
  stage1:
    agent_id: agent1
    next_stage: stage2
  stage2:
    agent_id: agent2
    next_stage: null
"""

def test_execution_plan_from_yaml():
    """Verify ExecutionPlan can parse minimal valid YAML."""
    plan = ExecutionPlan.from_yaml(FLOW_YAML_SIMPLE, flow_id="test-flow")
    assert plan.id == "test-flow"
    assert plan.start_stage == "stage1"
    assert "stage1" in plan.stages
    assert "stage2" in plan.stages
    assert plan.stages["stage1"].agent_id == "agent1"
    assert plan.stages["stage2"].agent_id == "agent2"

@pytest.mark.asyncio
async def test_async_orchestrator_runs(
    mock_agent_provider: MagicMock, 
    mock_state_manager: MagicMock, 
    mock_config: dict
): 
    """Test that AsyncOrchestrator runs a basic linear flow (using MasterExecutionPlan)."""
    # Define a plan using MasterExecutionPlan and MasterStageSpec
    master_plan = MasterExecutionPlan(
        id="test_async_linear",
        name="Test Async Linear",
        start_stage="greet_master",
        stages={
            "greet_master": MasterStageSpec(
                agent_id="system-greeter-master", 
                inputs={"message": "hello async"}, 
                next_stage="farewell_master", 
                number=1.0
            ),
            "farewell_master": MasterStageSpec(
                agent_id="system-greeter-master", 
                inputs={"message": "bye async"},
                next_stage=None,
                number=2.0
            ),
        }
    )

    # Mock agents expected by this plan
    greet_agent_mock = AsyncMock(return_value={"greet_output": "ok"})
    farewell_agent_mock = AsyncMock(return_value={"farewell_output": "ok"})
    mock_agent_provider.get = AsyncMock(side_effect=[greet_agent_mock, farewell_agent_mock])
    
    # Instantiate AsyncOrchestrator correctly
    orch = AsyncOrchestrator(
        pipeline_def=master_plan, 
        config=mock_config, 
        agent_provider=mock_agent_provider, 
        state_manager=mock_state_manager
    )
    
    initial_context = {"run_type": "async_test"}
    final_context = await orch.run(master_plan, initial_context.copy()) # Pass plan and context

    # Assertions
    mock_agent_provider.get.assert_has_calls([
        call("system-greeter-master"), 
        call("system-greeter-master")
    ])
    greet_agent_mock.assert_awaited_once()
    farewell_agent_mock.assert_awaited_once()
    assert final_context['outputs']['greet_master'] == {"greet_output": "ok"}
    assert final_context['outputs']['farewell_master'] == {"farewell_output": "ok"}
    assert final_context['run_type'] == "async_test"
    # Use assert_has_calls for AsyncMock, it checks awaited calls
    mock_state_manager.update_status.assert_has_calls([
        call(stage=1.0, status='PASS', artifacts=[]),
        call(stage=2.0, status='PASS', artifacts=[]),
    ])