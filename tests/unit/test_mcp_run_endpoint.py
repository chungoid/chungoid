import os
os.environ["FLOW_REGISTRY_MODE"] = "memory"

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import textwrap
from unittest.mock import patch, AsyncMock, MagicMock, call

from chungoid.utils.mcp_server import app
# Import getter functions from flow_api for dependency overrides
from chungoid.utils.flow_api import get_agent_provider, get_state_manager, get_config
from chungoid.utils.flow_registry_singleton import _flow_registry
from chungoid.utils.flow_registry import FlowCard

# Assuming AsyncOrchestrator and related classes are available
from chungoid.runtime.orchestrator import AsyncOrchestrator, ExecutionPlan, StageSpec
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager

client = TestClient(app)

@pytest.fixture
def mock_agent_provider() -> MagicMock:
    provider = MagicMock(spec=AgentProvider)

    async def mock_agent_side_effect(context: dict):
        # This is what the agent executor returns
        # For these tests, the content might not be critical, but it must be awaitable
        # and return a dictionary as expected by the orchestrator.
        # The orchestrator updates its internal context with this dict.
        return {"agent_output": "mocked_data", "status_code": 0, "some_key_from_agent": True}

    # AgentProvider.get() should return a callable (the agent executor).
    # This callable needs to be an AsyncMock if it's awaited.
    provider.get.return_value = AsyncMock(side_effect=mock_agent_side_effect)
    return provider

@pytest.fixture
def mock_state_manager() -> MagicMock:
    manager = MagicMock(spec=StateManager)
    manager.update_status = MagicMock()
    return manager

@pytest.fixture
def mock_config() -> dict:
    return {"logging": {"level": "INFO"}} # Simple mock config

@pytest.fixture
def setup_orchestrator_dependencies_override(mock_agent_provider, mock_state_manager, mock_config):
    """Fixture to override orchestrator dependencies for endpoint tests."""
    original_overrides = client.app.dependency_overrides.copy()
    client.app.dependency_overrides[get_agent_provider] = lambda: mock_agent_provider
    client.app.dependency_overrides[get_state_manager] = lambda: mock_state_manager
    client.app.dependency_overrides[get_config] = lambda: mock_config
    yield
    client.app.dependency_overrides = original_overrides

async def run_flow(plan_dict, initial_context, agent_provider, state_manager, config):
    """Helper function to run a flow defined by a dictionary."""
    plan = ExecutionPlan(id="test-flow", start_stage=plan_dict['start_stage'], stages=plan_dict['stages'])
    orchestrator = AsyncOrchestrator(plan, config, agent_provider, state_manager)
    return await orchestrator.run(plan=plan, context=initial_context)

def test_run_endpoint_input_branching():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-branch",
        name="Test Branch",
        yaml_text=textwrap.dedent(
            """
            start_stage: s1
            stages:
              s1:
                agent_id: a1
                next:
                  condition: "input == foo"
                  "true": s2
                  "false": s3
              s2:
                agent_id: a2
              s3:
                agent_id: a3
            """
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # input == foo
        resp = client.post(
            "/run/test-branch",
            headers={"X-API-Key": api_key},
            json={"input": "foo"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["visited"] == ["s1", "s2"]
        # input == bar
        resp2 = client.post(
            "/run/test-branch",
            headers={"X-API-Key": api_key},
            json={"input": "bar"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-branch")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_numeric_branching():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-numeric-branch",
        name="Test Numeric Branch",
        yaml_text=textwrap.dedent(
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
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # input = 10 (should go to s2)
        resp = client.post(
            "/run/test-numeric-branch",
            headers={"X-API-Key": api_key},
            json={"input": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s2"]
        # input = 3 (should go to s3)
        resp2 = client.post(
            "/run/test-numeric-branch",
            headers={"X-API-Key": api_key},
            json={"input": 3},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-numeric-branch")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_string_inequality():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-string-neq",
        name="Test String NEQ",
        yaml_text=textwrap.dedent(
            """
            start_stage: s1
            stages:
              s1:
                agent_id: a1
                next:
                  condition: "input != foo"
                  "true": s2
                  "false": s3
              s2:
                agent_id: a2
              s3:
                agent_id: a3
            """
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # input = bar (should go to s2)
        resp = client.post(
            "/run/test-string-neq",
            headers={"X-API-Key": api_key},
            json={"input": "bar"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s2"]
        # input = foo (should go to s3)
        resp2 = client.post(
            "/run/test-string-neq",
            headers={"X-API-Key": api_key},
            json={"input": "foo"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-string-neq")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_numeric_less_than():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-numeric-lt",
        name="Test Numeric LT",
        yaml_text=textwrap.dedent(
            """
            start_stage: s1
            stages:
              s1:
                agent_id: a1
                next:
                  condition: "input < 5"
                  "true": s2
                  "false": s3
              s2:
                agent_id: a2
              s3:
                agent_id: a3
            """
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # input = 3 (should go to s2)
        resp = client.post(
            "/run/test-numeric-lt",
            headers={"X-API-Key": api_key},
            json={"input": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s2"]
        # input = 7 (should go to s3)
        resp2 = client.post(
            "/run/test-numeric-lt",
            headers={"X-API-Key": api_key},
            json={"input": 7},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-numeric-lt")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_type_mismatch():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-type-mismatch",
        name="Test Type Mismatch",
        yaml_text=textwrap.dedent(
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
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # input = "foo" (should go to s3, not s2)
        resp = client.post(
            "/run/test-type-mismatch",
            headers={"X-API-Key": api_key},
            json={"input": "foo"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-type-mismatch")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_missing_context_key():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-missing-key",
        name="Test Missing Key",
        yaml_text=textwrap.dedent(
            """
            start_stage: s1
            stages:
              s1:
                agent_id: a1
                next:
                  condition: "foo > 5"
                  "true": s2
                  "false": s3
              s2:
                agent_id: a2
              s3:
                agent_id: a3
            """
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # No 'foo' in context, should go to s3
        resp = client.post(
            "/run/test-missing-key",
            headers={"X-API-Key": api_key},
            json={"input": 10},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-missing-key")

@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
def test_run_endpoint_user_context_branching():
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    card = FlowCard(
        flow_id="test-user-branch",
        name="Test User Branch",
        yaml_text=textwrap.dedent(
            """
            start_stage: s1
            stages:
              s1:
                agent_id: a1
                next:
                  condition: "user == admin"
                  "true": s2
                  "false": s3
              s2:
                agent_id: a2
              s3:
                agent_id: a3
            """
        ),
        tags=["test"],
        version="0.1",
    )
    _flow_registry.add(card)
    try:
        # user = admin (should go to s2)
        resp = client.post(
            "/run/test-user-branch",
            headers={"X-API-Key": api_key},
            json={"user": "admin"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["result"]["visited"] == ["s1", "s2"]
        # user = guest (should go to s3)
        resp2 = client.post(
            "/run/test-user-branch",
            headers={"X-API-Key": api_key},
            json={"user": "guest"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["result"]["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-user-branch")

@pytest.mark.asyncio
async def test_run_endpoint_input_branching(mock_agent_provider, mock_state_manager, mock_config):
    """Test branching based on initial context input."""
    # Mock agent behaviour
    async def agent1_effect(context): return {"output1": "done"}
    async def agent_a_effect(context): return {"outputA": "branch A taken"}
    async def agent_b_effect(context): return {"outputB": "branch B taken"}
    mock_agent_provider.get.side_effect = lambda agent_id: {
        "agent1": AsyncMock(side_effect=agent1_effect),
        "agentA": AsyncMock(side_effect=agent_a_effect),
        "agentB": AsyncMock(side_effect=agent_b_effect)
    }[agent_id]

    plan_dict = {
        'start_stage': 'stage1',
        'stages': {
            'stage1': StageSpec(
                agent_id='agent1',
                condition='input_branch == "A"',
                next_stage_true='stageA',
                next_stage_false='stageB',
                next_stage=None
            ),
            'stageA': StageSpec(agent_id='agentA', next_stage=None),
            'stageB': StageSpec(agent_id='agentB', next_stage=None)
        }
    }

    # Test branch A
    initial_context_a = {'input_branch': 'A'}
    result_a = await run_flow(plan_dict, initial_context_a, mock_agent_provider, mock_state_manager, mock_config)
    assert mock_agent_provider.get.call_count == 2
    get_calls_a = mock_agent_provider.get.call_args_list
    assert call('agent1') in get_calls_a
    assert call('agentA') in get_calls_a
    assert 'outputA' in result_a['outputs']['stageA']
    assert result_a['outputs']['stageA']['outputA'] == 'branch A taken'
    assert 'stageB' not in result_a['outputs']

    # Reset mock for next run
    mock_agent_provider.reset_mock()
    mock_state_manager.reset_mock()

    # Test branch B
    initial_context_b = {'input_branch': 'B'}
    result_b = await run_flow(plan_dict, initial_context_b, mock_agent_provider, mock_state_manager, mock_config)
    assert mock_agent_provider.get.call_count == 2
    get_calls_b = mock_agent_provider.get.call_args_list
    assert call('agent1') in get_calls_b
    assert call('agentB') in get_calls_b
    assert 'outputB' in result_b['outputs']['stageB']
    assert result_b['outputs']['stageB']['outputB'] == 'branch B taken'
    assert 'stageA' not in result_b['outputs']

# This test was skipped due to: "Endpoint dependency issue: Tool 'get_file' not found in registry or its handler is not callable/defined."
# This is a DIFFERENT issue than the orchestrator dependency injection.
# I will leave this skip for now and address it later.
# @pytest.mark.skip(reason="Endpoint dependency issue: Tool 'get_file' not found in registry or its handler is not callable/defined.")
@pytest.mark.usefixtures("setup_orchestrator_dependencies_override")
@pytest.mark.asyncio
async def test_run_flow_with_tool_calls(mock_agent_provider, mock_state_manager, mock_config):
    api_key = os.environ.get("MCP_API_KEY", "dev-key")
    flow_id = "test-flow-with-tools"
    card = FlowCard(
        flow_id=flow_id,
        name="Test Flow With Tool Calls",
        yaml_text=textwrap.dedent(
            '''
            start_stage: stage_tool_user
            stages:
              stage_tool_user:
                agent_id: agent_tool_user
                # We'll need to define how an agent specifies it needs to call a tool.
                # This is a placeholder, the actual mechanism might involve 'inputs'
                # or a specific 'tool_calls' section in the StageSpec.
                # For now, let's assume the agent internally knows to call 'get_file'.
                inputs:
                  file_path: "dummy/path.txt"
                next_stage: null
            '''
        ),
        tags=["test", "tools"],
        version="0.1",
    )
    _flow_registry.add(card)

    # --- Mocking AgentProvider for this specific test ---
    # The default mock_agent_provider might not be suitable if tool calls
    # are handled differently or if the agent's response depends on tool output.

    async def agent_tool_user_side_effect(context: dict):
        # This mock agent will simulate calling 'get_file' and returning its output.
        # How it "calls" get_file isn't directly mocked here, but its output
        # should reflect that a tool like get_file was notionally used.
        file_path_from_context = context.get("file_path", "unknown_path_from_context")
        return {
            "status_code": 0,
            "data_from_get_file": f"content of {file_path_from_context}",
            "tool_used": "get_file"
        }

    # Make the mock_agent_provider return this specific side_effect for 'agent_tool_user'
    # We need to be careful if the fixture `mock_agent_provider` is broadly scoped.
    # For now, let's assume we can modify its behavior for this test or refine the fixture.
    # A more robust way would be for the fixture to allow per-test customization.

    # For this test, we expect 'agent_tool_user' to be requested.
    mock_agent_provider.get.side_effect = lambda agent_id: (
        AsyncMock(side_effect=agent_tool_user_side_effect) if agent_id == "agent_tool_user" else AsyncMock(return_value={"status_code": 1, "error": "Unknown agent"})
    )

    try:
        response = client.post(
            f"/run/{flow_id}",
            headers={"X-API-Key": api_key},
            json={"initial_input": "some_value"} # Example initial context
        )
        assert response.status_code == 200
        data = response.json()
        
        assert data["status"] == "ok"
        assert "result" in data
        assert "outputs" in data["result"]
        assert "stage_tool_user" in data["result"]["outputs"]
        stage_output = data["result"]["outputs"]["stage_tool_user"]
        assert stage_output["tool_used"] == "get_file"
        assert stage_output["data_from_get_file"] == "content of dummy/path.txt"
        assert data["result"]["visited"] == ["stage_tool_user"]

    finally:
        _flow_registry.remove(flow_id) 