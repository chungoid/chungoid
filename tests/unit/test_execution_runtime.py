import textwrap
import pytest
from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator


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