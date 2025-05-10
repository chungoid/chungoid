import textwrap

from chungoid.runtime.orchestrator import ExecutionPlan, SyncOrchestrator


def _sample_yaml():
    return textwrap.dedent(
        """
        start_stage: greet
        stages:
          greet:
            agent_id: system-greeter
            input: "hello"
            next: farewell
          farewell:
            agent_id: system-greeter
            input: "bye"
        """
    )


def _branching_yaml():
    return textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            agent_id: a1
            next_if:
              always: s2
          s2:
            agent_id: a2
        """
    )


def test_execution_plan_parses():
    plan = ExecutionPlan.from_yaml(_sample_yaml(), flow_id="flow-xyz")
    assert plan.start_stage == "greet"
    assert set(plan.stages.keys()) == {"greet", "farewell"}
    assert plan.id == "flow-xyz"


def test_sync_orchestrator_runs_linear_flow():
    plan = ExecutionPlan.from_yaml(_sample_yaml())
    orch = SyncOrchestrator(plan)
    visited = orch.run()
    assert visited == ["greet", "farewell"]


def test_async_orchestrator_runs(event_loop):
    plan = ExecutionPlan.from_yaml(_sample_yaml())
    import asyncio
    async def _run():
        from chungoid.runtime.orchestrator import AsyncOrchestrator
        orch = AsyncOrchestrator(plan)
        return await orch.run()
    visited = event_loop.run_until_complete(_run())
    assert visited == ["greet", "farewell"]


def test_sync_orchestrator_next_if():
    plan = ExecutionPlan.from_yaml(_branching_yaml())
    orch = SyncOrchestrator(plan)
    visited = orch.run()
    assert visited == ["s1", "s2"]


def test_sync_orchestrator_loop_guard():
    # s1 -> s2 -> s1 (cycle)
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
    # Should visit s1, s2, s1, s2, s1
    assert visited == ["s1", "s2", "s1", "s2", "s1"] 