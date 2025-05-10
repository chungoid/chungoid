import os
os.environ["FLOW_REGISTRY_MODE"] = "memory"

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import textwrap
from unittest.mock import patch

from chungoid.utils.mcp_server import app
from chungoid.utils.flow_registry_singleton import _flow_registry
from chungoid.utils.flow_registry import FlowCard

client = TestClient(app)

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
                next_if:
                  "input == foo": s2
                  always: s3
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
                next_if:
                  "input > 5": s2
                  always: s3
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
                next_if:
                  "input != foo": s2
                  always: s3
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
                next_if:
                  "input < 5": s2
                  always: s3
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
                next_if:
                  "input > 5": s2
                  always: s3
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
                next_if:
                  "foo > 5": s2
                  always: s3
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
                next_if:
                  "user == admin": s2
                  always: s3
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