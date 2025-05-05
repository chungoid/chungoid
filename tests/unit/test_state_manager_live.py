import json
from pathlib import Path

import pytest

from utils.state_manager import StateManager


def create_sample_project(tmp_path: Path) -> Path:
    """Create a minimal Chungoid project directory on the real filesystem.

    Returns the new project dir path.
    """
    project_dir = tmp_path / "my_project"
    project_dir.mkdir()

    # Also create a dummy stages dir with one yaml file so StateManager init passes
    stages_dir = tmp_path / "server_stages"
    stages_dir.mkdir()
    (stages_dir / "stage0.yaml").write_text("system_prompt: foo\nuser_prompt: bar\n")

    return project_dir, stages_dir


@pytest.fixture()
def sm(tmp_path):
    """Return an initialised StateManager backed by a tmp directory."""
    project_dir, stages_dir = create_sample_project(tmp_path)
    manager = StateManager(target_directory=str(project_dir), server_stages_dir=str(stages_dir), use_locking=False)
    return manager


def test_status_file_initialises_empty(sm: StateManager):
    assert sm.get_full_status() == {"runs": []}


def test_update_status_adds_entry(sm: StateManager):
    ok = sm.update_status(stage=0.0, status="DONE", artifacts=["out.txt"])
    assert ok
    status = sm.get_full_status()
    assert status["runs"], "Should have at least one run after update"
    first_run = status["runs"][0]
    assert first_run["status_updates"][0]["status"] == "DONE"


def test_get_next_stage(sm: StateManager):
    # No statuses yet – expect Stage 0 to be next since a stage0.yaml exists
    assert sm.get_next_stage() == 0.0
    sm.update_status(stage=0.0, status="DONE", artifacts=[])
    assert sm.get_next_stage() == 1.0


def test_store_artifact_context(sm: StateManager):
    # Ensure method returns True and does not raise with fake chroma
    assert sm.store_artifact_context_in_chroma(stage_number=1.0, rel_path="a/b.txt", content="hello") 