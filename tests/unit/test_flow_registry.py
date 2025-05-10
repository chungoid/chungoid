"""Unit tests for FlowRegistry (Phase-5).
"""
from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from chungoid.utils.flow_registry import FlowCard, FlowRegistry


@pytest.fixture()
def registry(tmp_path: Path) -> FlowRegistry:
    """Return a memory-mode registry for tests, isolated per test case."""
    return FlowRegistry(project_root=tmp_path, chroma_mode="memory")


def _sample_yaml() -> str:
    return """
start_stage: greet
stages:
  greet:
    agent_id: greeter
    next: null
"""


def test_add_and_get_flow(registry: FlowRegistry):
    flow_id = f"test-{uuid.uuid4().hex[:8]}"
    card = FlowCard(
        flow_id=flow_id,
        name="Test Flow",
        yaml_text=_sample_yaml(),
        tags=["unit"],
        version="0.1",
    )
    registry.add(card)
    fetched = registry.get(flow_id)
    assert fetched is not None, "Flow should be retrievable"
    assert fetched.flow_id == flow_id
    assert "greet" in fetched.stage_names


def test_list_contains_added_flow(registry: FlowRegistry):
    flow_id = f"list-{uuid.uuid4().hex[:8]}"
    card = FlowCard(flow_id=flow_id, name="List Flow", yaml_text=_sample_yaml())
    registry.add(card)
    items = registry.list(limit=10)
    ids = {c.flow_id for c in items}
    assert flow_id in ids


def test_remove_flow(registry: FlowRegistry):
    flow_id = f"del-{uuid.uuid4().hex[:8]}"
    card = FlowCard(flow_id=flow_id, name="Del Flow", yaml_text=_sample_yaml())
    registry.add(card)
    registry.remove(flow_id)
    assert registry.get(flow_id) is None 