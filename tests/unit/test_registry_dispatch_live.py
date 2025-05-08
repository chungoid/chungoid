import socket
from pathlib import Path
import yaml
import pytest

from chungoid.flow_executor import FlowExecutor
from chungoid.utils.agent_registry import AgentRegistry, AgentCard
from chungoid.utils.agent_resolver import RegistryAgentProvider


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except Exception:
            return False


# Skip entire module if MCP server not running (local dev use)
if not _port_open("localhost", 9000):
    pytest.skip("MCP server not running on localhost:9000", allow_module_level=True)


def test_flow_executor_live_dispatch(tmp_path):
    proj_root = Path(__file__).resolve().parents[4]

    # prepare registry with tool mapping existing on MCP server (load_pending_reflection)
    registry = AgentRegistry(project_root=proj_root, chroma_mode="memory")
    card = AgentCard(
        agent_id="reflection_loader",
        name="Reflection Loader",
        tool_names=["load_pending_reflection"],
    )
    registry.add(card, overwrite=True)

    provider = RegistryAgentProvider(registry)

    # simple flow
    flow_path = tmp_path / "flow.yaml"
    yaml.safe_dump(
        {
            "name": "live_flow",
            "start_stage": "s1",
            "stages": {
                "s1": {"agent_id": "reflection_loader", "next": None}
            },
        },
        flow_path.open("w"),
    )

    executor = FlowExecutor(provider)
    result = executor.run(flow_path)
    assert result == ["s1"] 