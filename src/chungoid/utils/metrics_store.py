"""Runtime metrics storage helper (Chroma-backed).

Stores *MetricEvent*s emitted by the execution runtime so that agents and
humans can query performance and reliability information.

Design goals
------------
* **Light-weight** – write-once documents; no embeddings required.
* **Schema-evolvable** – extra keys allowed so we can extend metrics later.
* **Drop-in** – mirrors `ReflectionStore` API for familiarity.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence
import json
import os

try:
    import chromadb
    from chromadb.api import ClientAPI, Collection
except ImportError as exc:  # pragma: no cover
    raise ImportError("chromadb is required for MetricsStore") from exc

from .chroma_client_factory import get_client

# Import the canonical MetricEvent and MetricEventType
from chungoid.schemas.metrics import MetricEvent, MetricEventType

__all__ = ["MetricsStore"]


class MetricsStore:
    """
    Handles the storage and retrieval of MetricEvent objects to/from a JSONL file.
    """
    DEFAULT_METRICS_FILENAME = "metrics.jsonl"

    def __init__(self, project_root: Path):
        """
        Initializes the MetricsStore.

        Args:
            project_root: The root directory of the chungoid project.
                          The .chungoid directory (and metrics file) will be relative to this.
        """
        self.project_root = project_root
        self._metrics_dir = self.project_root / ".chungoid"
        self._metrics_file_path = self._metrics_dir / self.DEFAULT_METRICS_FILENAME
        self._ensure_metrics_dir_exists()

    def _ensure_metrics_dir_exists(self):
        """Ensures that the .chungoid directory for metrics exists."""
        os.makedirs(self._metrics_dir, exist_ok=True)

    def _get_metrics_file_path(self) -> Path:
        """Returns the path to the metrics.jsonl file."""
        return self._metrics_file_path

    def add_event(self, event: MetricEvent):
        """
        Adds a MetricEvent to the metrics store (appends to the JSONL file).

        Args:
            event: The MetricEvent object to add.
        """
        self._ensure_metrics_dir_exists() # Ensure dir exists before writing
        try:
            # We need to customize the serialization for datetime
            event_dict = event.model_dump(mode='json') # Pydantic v2
            
            with open(self._get_metrics_file_path(), 'a', encoding='utf-8') as f:
                json.dump(event_dict, f)
                f.write('\n')
        except Exception as e:
            # Basic error handling, consider more robust logging
            print(f"Error writing metric event to {self._get_metrics_file_path()}: {e}")
            # Potentially re-raise or handle more gracefully depending on requirements

    def get_events(
        self,
        run_id: Optional[str] = None,
        flow_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        master_stage_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        event_types: Optional[List[MetricEventType]] = None,
        limit: Optional[int] = None,
        sort_desc: bool = True, # Default to most recent first
    ) -> List[MetricEvent]:
        """
        Retrieves a list of MetricEvents from the store, with optional filtering and limiting.

        Args:
            run_id: Filter by run ID.
            flow_id: Filter by flow ID.
            stage_id: Filter by stage ID.
            master_stage_id: Filter by master stage ID.
            agent_id: Filter by agent ID.
            event_types: Filter by a list of MetricEventTypes.
            limit: Maximum number of events to return.
            sort_desc: If True, sorts events by timestamp descending (most recent first).
                       If False, sorts ascending (oldest first).

        Returns:
            A list of matching MetricEvent objects.
        """
        events: List[MetricEvent] = []
        metrics_file = self._get_metrics_file_path()

        if not metrics_file.exists():
            return events

        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line.strip())
                            event = MetricEvent.model_validate(data) # Pydantic v2

                            # Apply filters
                            if run_id is not None and event.run_id != run_id:
                                continue
                            if flow_id is not None and event.flow_id != flow_id:
                                continue
                            if stage_id is not None and event.stage_id != stage_id:
                                continue
                            if master_stage_id is not None and event.master_stage_id != master_stage_id:
                                continue
                            if agent_id is not None and event.agent_id != agent_id:
                                continue
                            if event_types is not None and event.event_type not in event_types:
                                continue
                            
                            events.append(event)
                        except json.JSONDecodeError as jde:
                            print(f"Skipping malformed line in metrics file {metrics_file}: {jde} - Line: {line.strip()}")
                        except Exception as ve: # Catch Pydantic ValidationError and other model errors
                            print(f"Skipping event due to validation error in metrics file {metrics_file}: {ve} - Data: {line.strip()}")
        except Exception as e:
            print(f"Error reading metrics file {metrics_file}: {e}")
            return [] # Return empty list on general read error

        # Sort events by timestamp
        # The timestamp in MetricEvent is already a datetime object due to Pydantic model.
        if sort_desc:
            events.sort(key=lambda ev: ev.timestamp, reverse=True)
        else:
            events.sort(key=lambda ev: ev.timestamp, reverse=False)

        if limit is not None and limit > 0:
            return events[:limit]
        
        return events

    def clear_all_events(self):
        """
        Deletes the metrics file, effectively clearing all stored events.
        Use with caution.
        """
        metrics_file = self._get_metrics_file_path()
        if metrics_file.exists():
            try:
                os.remove(metrics_file)
                print(f"Metrics file {metrics_file} deleted.")
            except OSError as e:
                print(f"Error deleting metrics file {metrics_file}: {e}")

# Example Usage (for testing or direct script use):
if __name__ == "__main__":
    # Assuming this script is run from a context where project_root can be determined
    # For example, if your project root is the parent of 'src'
    current_script_path = Path(__file__).resolve()
    # Adjust this path according to your project structure
    # This example assumes 'src/chungoid/utils/metrics_store.py'
    # and project root is two levels up from 'utils'
    example_project_root = current_script_path.parent.parent.parent.parent 
    print(f"Using example project root: {example_project_root}")

    if not (example_project_root / "pyproject.toml").exists():
         # Fallback for when script is not in typical place, e.g. during isolated testing
        example_project_root = Path(".") # Current directory
        print(f"pyproject.toml not found at assumed root, falling back to CWD: {example_project_root.resolve()}")


    store = MetricsStore(project_root=example_project_root)
    
    # Clear previous test events
    store.clear_all_events()

    # Create some dummy events
    event1_data = {
        "event_type": MetricEventType.FLOW_START,
        "flow_id": "flow_abc",
        "run_id": "run_123",
        "data": {"flow_name": "Test Flow Alpha"}
    }
    # Pydantic v2 will handle timestamp default factory
    event1 = MetricEvent(**event1_data) 
    store.add_event(event1)

    # Simulate a slight delay for timestamp ordering
    import time
    time.sleep(0.01)

    event2_data = {
        "event_type": MetricEventType.STAGE_START,
        "flow_id": "flow_abc",
        "run_id": "run_123",
        "stage_id": "stage_1",
        "data": {"stage_name": "Initialization"}
    }
    event2 = MetricEvent(**event2_data)
    store.add_event(event2)

    time.sleep(0.01)

    event3_data = {
        "event_type": MetricEventType.STAGE_END,
        "flow_id": "flow_abc",
        "run_id": "run_123",
        "stage_id": "stage_1",
        "data": {"status": "COMPLETED_SUCCESS", "duration_seconds": 1.5}
    }
    event3 = MetricEvent(**event3_data)
    store.add_event(event3)

    time.sleep(0.01)
    
    event4_data = {
        "event_type": MetricEventType.FLOW_END,
        "flow_id": "flow_abc",
        "run_id": "run_123",
        "data": {"final_status": "COMPLETED_SUCCESS", "total_duration_seconds": 2.0}
    }
    event4 = MetricEvent(**event4_data)
    store.add_event(event4)

    print(f"\nAll events for run_123 (most recent first):")
    all_run_events = store.get_events(run_id="run_123")
    for ev in all_run_events:
        print(ev.model_dump_json(indent=2))

    print(f"\nStage end events for run_123:")
    stage_end_events = store.get_events(run_id="run_123", event_types=[MetricEventType.STAGE_END])
    for ev in stage_end_events:
        print(ev.model_dump_json(indent=2))

    print(f"\nOldest 2 events for run_123:")
    oldest_events = store.get_events(run_id="run_123", limit=2, sort_desc=False)
    for ev in oldest_events:
        print(ev.model_dump_json(indent=2))
        
    print(f"\nMetrics file location: {store._get_metrics_file_path()}") 