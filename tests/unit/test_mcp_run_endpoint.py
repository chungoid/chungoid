import os
os.environ["FLOW_REGISTRY_MODE"] = "memory"

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import textwrap
from unittest.mock import patch, AsyncMock, MagicMock, call

from chungoid.utils.mcp_server import app
from chungoid.utils.flow_registry_singleton import _flow_registry
from chungoid.utils.flow_registry import FlowCard

# Assuming AsyncOrchestrator and related classes are available
from chungoid.runtime.orchestrator import AsyncOrchestrator, ExecutionPlan, StageSpec
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager

client = TestClient(app)

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
    return {"logging": {"level": "INFO"}} # Simple mock config

async def run_flow(plan_dict, initial_context, agent_provider, state_manager, config):
    """Helper function to run a flow defined by a dictionary."""
    plan = ExecutionPlan(id="test-flow", start_stage=plan_dict['start_stage'], stages=plan_dict['stages'])
    orchestrator = AsyncOrchestrator(plan, config, agent_provider, state_manager)
    return await orchestrator.run(run_id="mcp-test-run", context=initial_context)

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

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s2"]
        # input = 3 (should go to s3)
        resp2 = client.post(
            "/run/test-numeric-branch",
            headers={"X-API-Key": api_key},
            json={"input": 3},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-numeric-branch")

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s2"]
        # input = foo (should go to s3)
        resp2 = client.post(
            "/run/test-string-neq",
            headers={"X-API-Key": api_key},
            json={"input": "foo"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-string-neq")

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s2"]
        # input = 7 (should go to s3)
        resp2 = client.post(
            "/run/test-numeric-lt",
            headers={"X-API-Key": api_key},
            json={"input": 7},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-numeric-lt")

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-type-mismatch")

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s3"]
    finally:
        _flow_registry.remove("test-missing-key")

@pytest.mark.skip(reason="Endpoint dependency injection for Orchestrator not yet implemented")
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
        assert data["visited"] == ["s1", "s2"]
        # user = guest (should go to s3)
        resp2 = client.post(
            "/run/test-user-branch",
            headers={"X-API-Key": api_key},
            json={"user": "guest"},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["visited"] == ["s1", "s3"]
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
                next={
                    'condition': 'input_branch == "A"',
                    'true': 'stageA',
                    'false': 'stageB'
                }
            ),
            'stageA': StageSpec(agent_id='agentA', next=None),
            'stageB': StageSpec(agent_id='agentB', next=None)
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