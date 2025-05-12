import textwrap
import pytest
from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator, AsyncOrchestrator, StageSpec
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager


def _linear_yaml():
    return textwrap.dedent(
        """
        start_stage: greet
        stages:
          greet:
            agent_id: system-greeter
            inputs:
              message: "hello"
            next: farewell
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
            next:
              condition: "input > 5"
              "true": s2
              "false": s3
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
            next: s2
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
    orch = SyncOrchestrator(plan)
    visited = orch.run()
    assert visited == ["greet", "farewell"]


def test_sync_orchestrator_conditional_next():
    plan = ExecutionPlan.from_yaml(_conditional_yaml())
    orch = SyncOrchestrator(plan)
    # input > 5 (input=10)
    visited = orch.run(context={"input": 10})
    assert visited == ["s1", "s2"]
    # input <= 5 (input=3)
    visited2 = orch.run(context={"input": 3})
    assert visited2 == ["s1", "s3"]


def test_sync_orchestrator_on_error_conditional():
    plan = ExecutionPlan.from_yaml(_on_error_yaml())
    orch = SyncOrchestrator(plan)
    # error_code == 404
    visited = orch.run(context={"error_code": 404})
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
            next: s2
          s2:
            agent_id: a2
            next: s1
        """
    )
    plan = ExecutionPlan.from_yaml(yaml_text)
    orch = SyncOrchestrator(plan)
    visited = orch.run(max_hops=5)
    assert visited == ["s1", "s2", "s1", "s2", "s1"]


def test_schema_validation_missing_required():
    bad_yaml = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            # agent_id missing
            next: s2
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
            next: s2
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
            next: s2
        """
    )
    with pytest.raises(ValueError) as exc:
        ExecutionPlan.from_yaml(bad_yaml)
    assert "agent_id" in str(exc.value)


def test_async_orchestrator_runs(event_loop):
    plan = ExecutionPlan.from_yaml(_linear_yaml())
    import asyncio
    async def _run():
        from chungoid.runtime.orchestrator import AsyncOrchestrator
        orch = AsyncOrchestrator(plan)
        return await orch.run()
    visited = event_loop.run_until_complete(_run())
    assert visited == ["greet", "farewell"]


def test_sync_orchestrator_next_if():
    plan = ExecutionPlan.from_yaml(_conditional_yaml())
    orch = SyncOrchestrator(plan)
    visited = orch.run()
    # No context, so input > 5 is not met; should take 'false' branch
    assert visited == ["s1", "s3"]


def test_sync_orchestrator_on_error_conditional():
    plan = ExecutionPlan.from_yaml(_on_error_yaml())
    orch = SyncOrchestrator(plan)
    # error_code == 404
    visited = orch.run(context={"error_code": 404})
    # Only s1 is visited, as on_error is not triggered in this simple run
    assert visited == ["s1"]


@pytest.fixture
def mock_agent_provider() -> MagicMock:
    return MagicMock(spec=AgentProvider)

@pytest.fixture
def mock_state_manager() -> MagicMock:
    manager = MagicMock(spec=StateManager)
    manager.update_status = MagicMock()
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
    next: stage2
  stage2:
    agent_id: agent2
    next: null
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
    """Basic test to ensure AsyncOrchestrator can be instantiated and run."""
    plan = ExecutionPlan.from_yaml(FLOW_YAML_SIMPLE, flow_id="test-async")
    
    # Setup basic mocks for agent calls
    async def dummy_agent(context): return {"output": "ok"}
    mock_agent_provider.get.return_value = AsyncMock(side_effect=dummy_agent)
    
    # Instantiate with required args
    orchestrator = AsyncOrchestrator(plan, mock_config, mock_agent_provider, mock_state_manager)
    
    # Run
    final_context = await orchestrator.run(run_id="async-run-1", context={}, max_hops=5)
    
    # Basic assertions
    assert mock_agent_provider.get.call_count == 2
    mock_agent_provider.get.assert_any_call("agent1")
    mock_agent_provider.get.assert_any_call("agent2")
    assert "outputs" in final_context
    assert "stage1" in final_context["outputs"]
    assert "stage2" in final_context["outputs"]
    assert mock_state_manager.update_status.call_count >= 2 # At least called for each stage success 