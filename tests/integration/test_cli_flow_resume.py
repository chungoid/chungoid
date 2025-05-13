import pytest
import subprocess
import json
import os
from pathlib import Path
from typing import Generator, Any
import shutil

from click.testing import CliRunner
from unittest.mock import patch, AsyncMock

# Assuming the main CLI entry point is 'cli' in 'chungoid.cli'
from chungoid.cli import cli 
from chungoid.schemas.flows import PausedRunDetails
from chungoid.schemas.errors import AgentErrorDetails
from datetime import datetime, timezone
from chungoid.utils.state_manager import StateManager
from chungoid.utils.flow_registry import FlowRegistry, FlowCard
from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card 

# TODO: Potentially add imports for FlowRegistry, RegistryAgentProvider if needed for setup helpers

# --- Test Setup Fixtures ---

@pytest.fixture
def setup_paused_project(tmp_path: Path, request) -> Generator[Path, None, None]:
    """Sets up a temporary project directory with a paused flow state.
    Allows modification via request.param.
    Copies the server_prompts directory needed by the CLI.
    Creates the master_flows directory and the flow definition yaml.
    """
    project_dir = tmp_path
    chungoid_dir = project_dir / ".chungoid"
    paused_dir = chungoid_dir / "paused_runs"
    master_flows_dir = project_dir / "master_flows"
    paused_dir.mkdir(parents=True)
    master_flows_dir.mkdir(parents=True)

    # <<< Copy server_prompts from source >>>
    try:
        # Determine source path relative to this test file
        # tests/integration/test_cli_flow_resume.py -> tests/ -> chungoid-core/ -> server_prompts/
        source_prompts_dir = Path(__file__).parent.parent.parent / "server_prompts"
        dest_prompts_dir = project_dir / "server_prompts"
        
        if source_prompts_dir.is_dir():
            shutil.copytree(source_prompts_dir, dest_prompts_dir)
            print(f"Copied {source_prompts_dir} to {dest_prompts_dir}")
            # <<< ADD: Ensure stages subdir exists after copy >>>
            stages_subdir = dest_prompts_dir / "stages"
            stages_subdir.mkdir(exist_ok=True) # Ensure it exists even if source didn't have it
        else:
             pytest.fail(f"Source server_prompts directory not found at {source_prompts_dir}. Check path calculation.")
    except Exception as e_copy:
        pytest.fail(f"Failed to copy server_prompts directory in test setup: {e_copy}")
    # <<< End copy >>>

    # <<< ADD: Create dummy stage definition files >>>
    dummy_yaml_content = "mcp_actions: []" # Minimal valid YAML content
    stages_subdir = dest_prompts_dir / "stages" # Define stages_subdir again for clarity
    (stages_subdir / "stage_a_def.yaml").write_text(dummy_yaml_content, encoding='utf-8')
    (stages_subdir / "stage_b_def.yaml").write_text(dummy_yaml_content, encoding='utf-8')
    # stage_c is only needed for the skip_setup variation, but creating it always is harmless
    (stages_subdir / "stage_c_def.yaml").write_text(dummy_yaml_content, encoding='utf-8') 
    print(f"Created dummy stage defs in {stages_subdir}")
    # <<< END: Create dummy stage definition files >>>

    # Default: Pause at stage_b in a 2-stage flow
    flow_id = "test_flow"
    flow_name = "Test Flow for Resume"
    paused_stage_id = "stage_b"
    agent_id_to_use = core_stage_executor_card.agent_id
    project_dir_str = str(project_dir.resolve()) # Get the resolved path as string for YAML

    flow_yaml_content = f"""
id: {flow_id}
name: {flow_name}
start_stage: stage_a
stages:
  stage_a:
    agent_id: {agent_id_to_use}
    inputs:
      stage_definition_filename: "stage_a_def.yaml"
      current_project_root: "{project_dir_str}"
    next_stage: stage_b
  stage_b: # This stage will be "paused at"
    agent_id: {agent_id_to_use}
    inputs:
      stage_definition_filename: "stage_b_def.yaml"
      current_project_root: "{project_dir_str}"
    next_stage: null
"""

    # Check for test-specific modifications
    if hasattr(request, "param") and request.param == "skip_setup":
        flow_id = "skip_test_flow"
        flow_name = "Flow for Skip Test"
        paused_stage_id = "stage_b" # Pause at B, skip to C
        flow_yaml_content = f"""
id: {flow_id}
name: {flow_name}
start_stage: stage_a
stages:
  stage_a:
    agent_id: {agent_id_to_use}
    inputs:
      stage_definition_filename: "stage_a_def.yaml"
      current_project_root: "{project_dir_str}"
    next_stage: stage_b
  stage_b:
    agent_id: {agent_id_to_use}
    inputs:
      stage_definition_filename: "stage_b_def.yaml"
      current_project_root: "{project_dir_str}"
    next_stage: stage_c # B leads to C
  stage_c:
    agent_id: {agent_id_to_use}
    inputs:
      stage_definition_filename: "stage_c_def.yaml"
      current_project_root: "{project_dir_str}"
    next_stage: null # C is the end
"""

    # 1. Create flow definition file in master_flows
    flow_yaml_path = master_flows_dir / f"{flow_id}.yaml"
    flow_yaml_path.write_text(flow_yaml_content)
    
    # 2. Add Flow to Registry
    # try:
    #     registry = FlowRegistry(project_root=project_dir, chroma_mode="persistent")
    #     flow_card = FlowCard(
    #         flow_id=flow_id,
    #         name=flow_name,
    #         yaml_text=flow_yaml_content,
    #         description="A flow used for testing resume functionality."
    #     )
    #     registry.add(flow_card)
    # except Exception as e:
    #     pytest.fail(f"Failed to add flow card to registry in test setup: {e}")

    # 3. Create the PausedRunDetails file
    run_id = f"{flow_id}-run1"
    paused_file_path = paused_dir / f"{run_id}.json"

    error = AgentErrorDetails(
        error_type="RuntimeError",
        message="Agent B failed!",
        traceback="Traceback...",
        agent_id=agent_id_to_use, # <<< USE CORE AGENT ID for the error details >>>
        stage_id=paused_stage_id
    )
    paused_details = PausedRunDetails(
        run_id=run_id,
        flow_id=flow_id,
        paused_at_stage_id=paused_stage_id,
        timestamp=datetime.now(timezone.utc),
        context_snapshot={"input": "data", "outputs": {"stage_a": {"result": "from A"}}},
        error_details=error,
        reason="Paused due to agent error"
    )
    paused_file_path.write_text(paused_details.model_dump_json(indent=2))
    
    # 4. Dummy status file
    status_file = chungoid_dir / "project_status.json"
    if not status_file.exists():
        status_file.write_text(json.dumps({"runs": []}, indent=2))

    # Yield project dir and the specific run_id created for this variation
    yield project_dir, run_id 

    # Teardown handled by tmp_path
    pass 


# --- Test Cases ---

def test_resume_retry(setup_paused_project):
    """Test resuming a paused flow using the 'retry' action via CLI."""
    project_dir: Path = setup_paused_project[0]
    run_id = setup_paused_project[1] # Must match the one created in the fixture
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"

    # Verify the paused file exists before running the command
    assert paused_file_path.exists()

    runner = CliRunner()
    
    # Store original CWD and change to the temp project directory for the test
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    try:
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'retry'],
            catch_exceptions=False # See full traceback on error
        )
    finally:
        # Restore original CWD
        os.chdir(original_cwd)

    # Assertions
    assert result.exit_code == 0, f"CLI command failed with output:\\\\n{result.output}"
    assert f"Flow resumption initiated for run_id '{run_id}'" in result.output
    assert f"processed with action 'retry'" in result.output
    
    # Verify the paused state file was cleared on successful initiation leading to execution
    assert not paused_file_path.exists(), "Paused run details file was not deleted after successful retry initiation."


def test_resume_retry_with_inputs_invalid(setup_paused_project):
    """Test resuming with 'retry_with_inputs' but providing invalid (non-JSON dict) inputs."""
    project_dir: Path = setup_paused_project[0]
    run_id = setup_paused_project[1]
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    invalid_inputs_json = '["not_a_dict"]' # Provide JSON that isn't a dictionary

    try:
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'retry_with_inputs', '--inputs', invalid_inputs_json],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    # Assertions for failure *before* execution attempt
    assert result.exit_code != 0, f"CLI command unexpectedly succeeded with invalid inputs: {result.output}"
    # Check for the specific error message from the orchestrator
    assert "requires a dictionary under the 'inputs' key" in result.output
    # Paused file should *not* be deleted because the resume command failed validation early
    assert paused_file_path.exists(), "Paused run details file was unexpectedly deleted after invalid input failure."


@pytest.mark.parametrize("setup_paused_project", ["skip_setup"], indirect=True)
def test_resume_skip_stage(setup_paused_project):
    """Test resuming a paused flow with 'skip_stage' action."""
    project_dir, run_id = setup_paused_project # Unpack run_id from fixture
    # Paused file path uses the specific run_id from the fixture
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    try:
        # No mocking needed, expect loop to start at stage_c and potentially fail
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'skip_stage'],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    print(f"CLI exited with code {result.exit_code}. Output:\\\\n{result.output}") 
    # Allow exit code 0 or 1, as actual execution of the next stage might fail
    assert result.exit_code in [0, 1], f"CLI command crashed: {result.output}"
    assert f"Flow resumption initiated for run_id '{run_id}'" in result.output
    assert "processed with action 'skip_stage'" in result.output
    
    # Paused file should be deleted because the skip action itself was valid
    assert not paused_file_path.exists(), "Paused run details file was not deleted after skip stage action."


def test_resume_force_branch_success(setup_paused_project):
    """Test resuming a paused flow with 'force_branch' action to a VALID stage."""
    project_dir, run_id = setup_paused_project # Use default setup
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    target_stage = "stage_a" # Force back to the start (valid stage)

    try:
        # Expect loop to start at stage_a and potentially fail
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'force_branch', '--target-stage', target_stage],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    print(f"CLI exited with code {result.exit_code}. Output:\\\\n{result.output}") 
    # Allow exit code 0 or 1, as actual execution from target stage might fail
    assert result.exit_code in [0, 1], f"CLI command crashed: {result.output}"
    assert f"Flow resumption initiated for run_id '{run_id}'" in result.output
    assert f"processed with action 'force_branch'" in result.output
    
    # Paused file should be deleted because the force branch action itself was valid
    assert not paused_file_path.exists(), "Paused run details file was not deleted after valid force_branch action."


def test_resume_force_branch_invalid_target(setup_paused_project):
    """Test resuming with 'force_branch' action to an INVALID stage."""
    project_dir, run_id = setup_paused_project
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    invalid_target_stage = "stage_does_not_exist"

    try:
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'force_branch', '--target-stage', invalid_target_stage],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    print(f"CLI exited with code {result.exit_code}. Output:\\\\n{result.output}")
    assert result.exit_code != 0, f"CLI command unexpectedly succeeded with invalid target stage: {result.output}"
    # Check for the specific error message from the orchestrator
    assert "Invalid target_stage_id for force_branch" in result.output
    # Paused file should NOT be deleted because the resume command failed validation early
    assert paused_file_path.exists(), "Paused run details file was unexpectedly deleted after invalid target stage failure."

def test_resume_force_branch_missing_target(setup_paused_project):
    """Test resuming with 'force_branch' action but MISSING the --target-stage option."""
    project_dir, run_id = setup_paused_project
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    try:
        # Omit the --target-stage option entirely
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'force_branch'],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    print(f"CLI exited with code {result.exit_code}. Output:\\\\n{result.output}")
    # We expect click to raise an error because --target-stage is likely required by the logic,
    # OR the orchestrator logic should catch the missing value.
    assert result.exit_code != 0, f"CLI command unexpectedly succeeded without target stage: {result.output}"
    # Check for the specific error message (might come from click or orchestrator)
    # Orchestrator returns: "Invalid target_stage_id for force_branch: 'None'"
    assert "Error: --target-stage is required" in result.output
    # Paused file should NOT be deleted
    assert paused_file_path.exists(), "Paused run details file was unexpectedly deleted when target stage was missing."


def test_resume_abort(setup_paused_project):
    """Test resuming a paused flow with 'abort' action."""
    project_dir, run_id = setup_paused_project # Use default setup
    paused_file_path = project_dir / ".chungoid" / "paused_runs" / f"{run_id}.json"
    assert paused_file_path.exists()

    runner = CliRunner()
    original_cwd = Path.cwd()
    os.chdir(project_dir)

    try:
        result = runner.invoke(
            cli,
            ['flow', 'resume', run_id, '--action', 'abort'],
            catch_exceptions=False
        )
    finally:
        os.chdir(original_cwd)

    print(f"CLI exited with code {result.exit_code}. Output:\\\\n{result.output}")
    # Abort should be clean, expect exit code 0
    assert result.exit_code == 0, f"CLI command failed: {result.output}"
    # Check for success message
    assert "successfully aborted" in result.output
    # Paused file should be deleted on successful abort
    assert not paused_file_path.exists(), "Paused run details file was not deleted on abort."


# TODO: Implement tests for error conditions (e.g., run_id not found) 