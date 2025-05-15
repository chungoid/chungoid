import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pytest

from chungoid.schemas.metrics import MetricEvent, MetricEventType
from chungoid.utils.metrics_store import MetricsStore


@pytest.fixture
def temp_project_root(tmp_path: Path) -> Path:
    """Create a temporary project root with a .chungoid directory."""
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    (project_root / ".chungoid").mkdir(exist_ok=True)
    # Create a dummy pyproject.toml to satisfy the example in MetricsStore main
    (project_root / "pyproject.toml").touch()
    return project_root


@pytest.fixture
def metrics_store(temp_project_root: Path) -> MetricsStore:
    """Fixture to provide a MetricsStore instance with a clean state."""
    store = MetricsStore(project_root=temp_project_root)
    store.clear_all_events() # Ensure clean state for each test
    return store


def create_dummy_event(
    event_type: MetricEventType,
    flow_id: Optional[str] = "test_flow",
    run_id: Optional[str] = "test_run",
    stage_id: Optional[str] = None,
    master_stage_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    data: Optional[dict] = None,
    timestamp: Optional[datetime] = None
) -> MetricEvent:
    """Helper to create a MetricEvent with some defaults."""
    return MetricEvent(
        event_type=event_type,
        flow_id=flow_id,
        run_id=run_id,
        stage_id=stage_id,
        master_stage_id=master_stage_id,
        agent_id=agent_id,
        data=data if data is not None else {},
        timestamp=timestamp if timestamp is not None else datetime.now(timezone.utc)
    )


def test_metrics_store_initialization(temp_project_root: Path):
    """Test that MetricsStore initializes correctly and creates .chungoid dir."""
    store = MetricsStore(project_root=temp_project_root)
    assert (temp_project_root / ".chungoid").exists()
    assert store._get_metrics_file_path().name == "metrics.jsonl"
    store.clear_all_events() # Cleanup


def test_add_and_get_single_event(metrics_store: MetricsStore):
    """Test adding a single event and retrieving it."""
    event = create_dummy_event(MetricEventType.FLOW_START, run_id="run1")
    metrics_store.add_event(event)

    retrieved_events = metrics_store.get_events(run_id="run1")
    assert len(retrieved_events) == 1
    assert retrieved_events[0].event_id == event.event_id
    assert retrieved_events[0].event_type == MetricEventType.FLOW_START


def test_add_multiple_events_and_get_all(metrics_store: MetricsStore):
    """Test adding multiple events and retrieving all of them."""
    event1 = create_dummy_event(MetricEventType.FLOW_START, run_id="run2")
    time.sleep(0.001) # ensure distinct timestamps
    event2 = create_dummy_event(MetricEventType.STAGE_START, run_id="run2", stage_id="s1")
    time.sleep(0.001)
    event3 = create_dummy_event(MetricEventType.FLOW_END, run_id="run2")

    metrics_store.add_event(event1)
    metrics_store.add_event(event2)
    metrics_store.add_event(event3)

    retrieved_events = metrics_store.get_events(run_id="run2")
    assert len(retrieved_events) == 3
    # Default sort is descending by timestamp
    assert retrieved_events[0].event_id == event3.event_id
    assert retrieved_events[1].event_id == event2.event_id
    assert retrieved_events[2].event_id == event1.event_id


def test_get_events_filtering(metrics_store: MetricsStore):
    """Test various filtering capabilities of get_events."""
    ev1_r1f1s1 = create_dummy_event(MetricEventType.STAGE_START, run_id="r1", flow_id="f1", stage_id="s1")
    time.sleep(0.001)
    ev2_r1f1s2 = create_dummy_event(MetricEventType.STAGE_END, run_id="r1", flow_id="f1", stage_id="s2", data={"status": "SUCCESS"})
    time.sleep(0.001)
    ev3_r2f1s1 = create_dummy_event(MetricEventType.STAGE_START, run_id="r2", flow_id="f1", stage_id="s1") # Different run_id
    time.sleep(0.001)
    ev4_r1f2s1 = create_dummy_event(MetricEventType.STAGE_START, run_id="r1", flow_id="f2", stage_id="s1") # Different flow_id
    time.sleep(0.001)
    ev5_r1f1s1_agent = create_dummy_event(MetricEventType.AGENT_INVOCATION_START, run_id="r1", flow_id="f1", stage_id="s1", agent_id="agent_x")

    all_events = [ev1_r1f1s1, ev2_r1f1s2, ev3_r2f1s1, ev4_r1f2s1, ev5_r1f1s1_agent]
    for ev in all_events:
        metrics_store.add_event(ev)

    # Filter by run_id
    r1_events = metrics_store.get_events(run_id="r1")
    assert len(r1_events) == 4 # ev1, ev2, ev4, ev5
    assert {e.event_id for e in r1_events} == {ev1_r1f1s1.event_id, ev2_r1f1s2.event_id, ev4_r1f2s1.event_id, ev5_r1f1s1_agent.event_id}

    # Filter by flow_id
    f1_events = metrics_store.get_events(flow_id="f1")
    assert len(f1_events) == 3 # ev1, ev2, ev3, ev5 (mistake here, should be ev1,ev2,ev3,ev5)
                                # Correcting: f1_events should be ev1, ev2, ev3, ev5 -> wait, ev3 is r2. so ev1, ev2, ev5_r1f1s1_agent. That's 3.
                                # And also ev3_r2f1s1 is flow_id f1. So 4 events. ev1, ev2, ev3, ev5.
    # Let's re-evaluate: Events with flow_id='f1': ev1_r1f1s1, ev2_r1f1s2, ev3_r2f1s1, ev5_r1f1s1_agent.
    assert len(f1_events) == 4
    assert {e.event_id for e in f1_events} == {ev1_r1f1s1.event_id, ev2_r1f1s2.event_id, ev3_r2f1s1.event_id, ev5_r1f1s1_agent.event_id}

    # Filter by stage_id (across different runs/flows)
    s1_events = metrics_store.get_events(stage_id="s1")
    assert len(s1_events) == 3 # ev1, ev3, ev4, ev5
    assert {e.event_id for e in s1_events} == {ev1_r1f1s1.event_id, ev3_r2f1s1.event_id, ev4_r1f2s1.event_id, ev5_r1f1s1_agent.event_id}

    # Filter by event_type
    stage_start_events = metrics_store.get_events(event_types=[MetricEventType.STAGE_START])
    assert len(stage_start_events) == 3 # ev1, ev3, ev4
    assert {e.event_id for e in stage_start_events} == {ev1_r1f1s1.event_id, ev3_r2f1s1.event_id, ev4_r1f2s1.event_id}

    # Filter by agent_id
    agent_x_events = metrics_store.get_events(agent_id="agent_x")
    assert len(agent_x_events) == 1
    assert agent_x_events[0].event_id == ev5_r1f1s1_agent.event_id

    # Combined filters
    r1f1_stage_starts = metrics_store.get_events(run_id="r1", flow_id="f1", event_types=[MetricEventType.STAGE_START])
    assert len(r1f1_stage_starts) == 1
    assert r1f1_stage_starts[0].event_id == ev1_r1f1s1.event_id

def test_get_events_limit(metrics_store: MetricsStore):
    """Test the limit parameter of get_events."""
    for i in range(5):
        metrics_store.add_event(create_dummy_event(MetricEventType.ORCHESTRATOR_INFO, run_id="limit_test", data={"i": i}))
        if i < 4: time.sleep(0.001) # ensure distinct timestamps except for the last two if range is 5

    limited_events = metrics_store.get_events(run_id="limit_test", limit=3)
    assert len(limited_events) == 3
    # Assuming default sort_desc=True, these should be the 3 most recent
    assert limited_events[0].data["i"] == 4
    assert limited_events[1].data["i"] == 3
    assert limited_events[2].data["i"] == 2

def test_get_events_sorting(metrics_store: MetricsStore):
    """Test sorting by timestamp (ascending and descending)."""
    ev1 = create_dummy_event(MetricEventType.FLOW_START, run_id="sort_test")
    time.sleep(0.002) # Larger sleep to ensure order if system clock resolution is low
    ev2 = create_dummy_event(MetricEventType.FLOW_END, run_id="sort_test")

    metrics_store.add_event(ev1) # Added first, so older
    metrics_store.add_event(ev2) # Added second, so newer

    # Descending (default)
    desc_events = metrics_store.get_events(run_id="sort_test")
    assert len(desc_events) == 2
    assert desc_events[0].event_id == ev2.event_id # Newer first
    assert desc_events[1].event_id == ev1.event_id # Older second

    # Ascending
    asc_events = metrics_store.get_events(run_id="sort_test", sort_desc=False)
    assert len(asc_events) == 2
    assert asc_events[0].event_id == ev1.event_id # Older first
    assert asc_events[1].event_id == ev2.event_id # Newer second

def test_get_events_empty_store(metrics_store: MetricsStore):
    """Test get_events on an empty or non-existent metrics file."""
    # metrics_store fixture already clears, so file might not exist initially
    assert metrics_store.get_events(run_id="non_existent_run") == []

    # Explicitly ensure file doesn't exist by clearing again (if it was created by another test)
    metrics_store.clear_all_events()
    assert metrics_store.get_events(run_id="non_existent_run") == []

def test_clear_all_events(metrics_store: MetricsStore, temp_project_root: Path):
    """Test that clear_all_events removes the metrics file."""
    metrics_file_path = temp_project_root / ".chungoid" / "metrics.jsonl"

    # Add an event to ensure the file is created
    metrics_store.add_event(create_dummy_event(MetricEventType.FLOW_START, run_id="clear_test"))
    assert metrics_file_path.exists()

    metrics_store.clear_all_events()
    assert not metrics_file_path.exists()

    # Calling clear again should not raise an error
    metrics_store.clear_all_events()
    assert not metrics_file_path.exists()

def test_metrics_store_malformed_json_line(metrics_store: MetricsStore, temp_project_root: Path, capsys):
    """Test that the store handles a malformed JSON line gracefully."""
    metrics_file = metrics_store._get_metrics_file_path()
    
    # Add a valid event
    valid_event = create_dummy_event(MetricEventType.FLOW_START, run_id="malformed_test")
    metrics_store.add_event(valid_event)
    
    # Manually append a malformed line
    with open(metrics_file, 'a', encoding='utf-8') as f:
        f.write("this is not json\n")
        
    # Add another valid event after malformed line
    valid_event_after = create_dummy_event(MetricEventType.FLOW_END, run_id="malformed_test")
    metrics_store.add_event(valid_event_after)
    
    retrieved_events = metrics_store.get_events(run_id="malformed_test")
    assert len(retrieved_events) == 2 # Should retrieve the two valid events
    event_ids = {e.event_id for e in retrieved_events}
    assert valid_event.event_id in event_ids
    assert valid_event_after.event_id in event_ids
    
    captured = capsys.readouterr()
    assert "Skipping malformed line" in captured.out or "Skipping malformed line" in captured.err

def test_metrics_store_validation_error_on_load(metrics_store: MetricsStore, temp_project_root: Path, capsys):
    """Test that the store handles a line that is JSON but fails Pydantic validation."""
    metrics_file = metrics_store._get_metrics_file_path()

    # Add a valid event
    valid_event = create_dummy_event(MetricEventType.FLOW_START, run_id="validation_err_test")
    metrics_store.add_event(valid_event)

    # Manually append a line that is JSON but will fail MetricEvent validation (e.g., missing event_type)
    bad_data_event = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(), # Pydantic will parse this from iso str
        # "event_type": MetricEventType.ORCHESTRATOR_INFO.value, # Missing event_type
        "flow_id": "validation_err_test",
        "run_id": "validation_err_test"
    }
    with open(metrics_file, 'a', encoding='utf-8') as f:
        import json
        json.dump(bad_data_event, f)
        f.write("\n")

    retrieved_events = metrics_store.get_events(run_id="validation_err_test")
    assert len(retrieved_events) == 1 # Should retrieve only the valid event
    assert retrieved_events[0].event_id == valid_event.event_id
    
    captured = capsys.readouterr()
    assert "Skipping event due to validation error" in captured.out or "Skipping event due to validation error" in captured.err 