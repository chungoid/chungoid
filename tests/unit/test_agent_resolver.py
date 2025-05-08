from pathlib import Path

import yaml

from chungoid.flow_executor import FlowExecutor
from chungoid.utils.agent_resolver import RegistryAgentProvider
from chungoid.utils.agent_registry import AgentRegistry, AgentCard


def test_flow_executor_with_registry(tmp_path):
    # --- set up in-memory registry ---
    proj_root = Path(__file__).resolve().parents[4]
    registry = AgentRegistry(project_root=proj_root, chroma_mode="memory")
    card = AgentCard(agent_id="goal_analyzer", name="Goal Analyzer")
    registry.add(card, overwrite=True)

    # fallback callable for registry stub
    def goal_analyzer_callable(stage):  # noqa: D401
        return {"ok": True}

    provider = RegistryAgentProvider(registry, fallback={"goal_analyzer": goal_analyzer_callable})

    # create simple flow yaml
    flow_path = tmp_path / "flow.yaml"
    flow_dict = {
        "name": "test_flow",
        "start_stage": "s1",
        "stages": {
            "s1": {
                "agent_id": "goal_analyzer",
                "next": None,
            }
        },
    }
    flow_path.write_text(yaml.safe_dump(flow_dict))

    executor = FlowExecutor(provider)
    executed = executor.run(flow_path)
    assert executed == ["s1"] 