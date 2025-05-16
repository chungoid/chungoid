from __future__ import annotations
import json
from chungoid.utils.config_loader import get_config
import click

"""Command-line interface for Chungoid-core.

This Click-based CLI provides a friendlier wrapper around the existing
`chungoid.mcp` entry-point, surfacing a few high-level sub-commands that
map to the most common actions developers perform when interacting with
Chungoid from the shell.

Examples
--------
$ chungoid run .            # run next stage for current dir
$ chungoid init my_proj     # initialise a fresh project skeleton
$ chungoid status           # print project status (pretty or JSON)

Auto-completion can be installed via `chungoid --install-completion` once
this script is on your `$PATH` (Click handles the plumbing).
"""

from pathlib import Path
import logging
import sys
import os
from typing import Any, Optional, Dict, List, cast

import click
import rich.traceback
from rich.logging import RichHandler

import chungoid
from chungoid.constants import (DEFAULT_MASTER_FLOWS_DIR, DEFAULT_SERVER_STAGES_DIR, MIN_PYTHON_VERSION,
                              PROJECT_CHUNGOID_DIR, STATE_FILE_NAME)
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus
from chungoid.core_utils import get_project_root_or_raise, init_project_structure
from chungoid.schemas.master_flow import MasterExecutionPlan
from chungoid.schemas.metrics import MetricEventType
from chungoid.schemas.flows import PausedRunDetails
# from chungoid.runtime.agents import ALL_AGENT_CARDS # For programmatic registration example - Commenting out as it's unused and causes import error
from chungoid.runtime.agents.system_master_planner_reviewer_agent import get_agent_card_static as get_reviewer_card
# from chungoid.runtime.agents.system_core_stage_executor_agent import get_agent_card_static as get_executor_card # Removed incorrect import
from chungoid.runtime.orchestrator import AsyncOrchestrator # for type hint
from chungoid.schemas.metrics import MetricEvent # For type hint
from chungoid.utils.agent_registry import AgentRegistry # Corrected: AGENT_REGISTRY is not exported or used
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore # for metrics CLI
from chungoid.utils.llm_provider import MockLLMProvider # <<< ADD THIS IMPORT
# from chungoid.config import ProjectConfig # REMOVED - This was incorrect and caused ModuleNotFoundError / SyntaxError

# Other existing imports ...

# For Agent Cards (used in agent_registry.add())
from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card
# from chungoid.runtime.agents.mock_agents import (  # Removed
#     get_mock_human_input_agent_card,
#     get_mock_code_generator_agent_card,
#     get_mock_test_generator_agent_card
# )
from chungoid.runtime.agents.mock_human_input_agent import get_agent_card_static as get_mock_human_input_agent_card
from chungoid.runtime.agents.mock_code_generator_agent import get_agent_card_static as get_mock_code_generator_agent_card
from chungoid.runtime.agents.mock_test_generator_agent import get_agent_card_static as get_mock_test_generator_agent_card
from chungoid.runtime.agents.mock_system_requirements_gathering_agent import get_agent_card_static as get_mock_system_requirements_gathering_agent_card

from chungoid.runtime.agents.system_master_planner_agent import get_agent_card_static as get_master_planner_agent_card
from chungoid.runtime.agents.system_master_planner_reviewer_agent import get_agent_card_static as get_master_planner_reviewer_agent_card

# For Agent Classes (used in the fallback_agents_map and direct instantiation)
# from chungoid.runtime.agents.core_stage_executor import CoreStageExecutorAgent # Removed - it's a function, not a class
from chungoid.runtime.agents.mock_human_input_agent import MockHumanInputAgent
from chungoid.runtime.agents.mock_code_generator_agent import MockCodeGeneratorAgent
from chungoid.runtime.agents.mock_test_generator_agent import MockTestGeneratorAgent
from chungoid.runtime.agents.mock_system_requirements_gathering_agent import MockSystemRequirementsGatheringAgent

from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent

# Other existing imports ...

# For CLI subcommands that call external scripts
import subprocess

# For JSON output
import json as py_json # Avoid conflict with click.json

# --- DIAGNOSTIC CODE AT THE TOP OF cli.py ---
print("--- DIAGNOSING chungoid.cli (Top of cli.py) ---")
print(f"Python Executable: {sys.executable}")
print(f"Initial sys.path: {sys.path}")
print(f"os.getcwd(): {os.getcwd()}")
print(f"__file__ (cli.py): {__file__}")

# Try to see where 'chungoid' itself is found from
try:
    print("Relevant sys.path entries for 'chungoid' (in CLI):")
    for p in sys.path:
        if 'chungoid' in p.lower() or 'site-packages' in p.lower() or p == os.getcwd() or '.local/pipx/venvs' in p.lower():
            print(f"  - {p}")

    import chungoid
    print(f"Found chungoid (in cli.py): {chungoid.__file__ if hasattr(chungoid, '__file__') else 'Namespace package'}")
    if hasattr(chungoid, '__path__'):
        print(f"chungoid.__path__ (in cli.py): {chungoid.__path__}")
        for p_item_chungoid in chungoid.__path__:
            print(f"  Contents of chungoid path item {p_item_chungoid}: {os.listdir(p_item_chungoid) if os.path.exists(p_item_chungoid) and os.path.isdir(p_item_chungoid) else 'Not a dir or does not exist'}")
            utils_dir_path = Path(p_item_chungoid) / 'utils'
            print(f"    Looking for {utils_dir_path} (from CLI): Exists? {utils_dir_path.exists()}, IsDir? {utils_dir_path.is_dir()}")
            if utils_dir_path.is_dir():
                 print(f"    Contents of {utils_dir_path} (from CLI): {os.listdir(utils_dir_path)}")

    # Now try importing chungoid.utils directly here for diagnostics
    try:
        import chungoid.utils
        print(f"Found chungoid.utils (in cli.py): {chungoid.utils.__file__ if hasattr(chungoid.utils, '__file__') else 'Namespace package'}")
        if hasattr(chungoid.utils, '__path__'):
            print(f"chungoid.utils.__path__ (in cli.py): {chungoid.utils.__path__}")
            for p_item_utils in chungoid.utils.__path__:
                print(f"  Contents of chungoid.utils path item {p_item_utils}: {os.listdir(p_item_utils) if os.path.exists(p_item_utils) and os.path.isdir(p_item_utils) else 'Not a dir or does not exist'}")
    except ModuleNotFoundError as e_utils_diag_cli:
        print(f"DIAGNOSTIC (CLI): Failed to import chungoid.utils in cli.py: {e_utils_diag_cli}")

except ModuleNotFoundError as e_chungoid_diag_cli:
    print(f"DIAGNOSTIC (CLI): Failed to import top-level 'chungoid' in cli.py: {e_chungoid_diag_cli}")
except Exception as e_diag_general_cli:
    print(f"DIAGNOSTIC (CLI): General error during diagnostic imports in cli.py: {e_diag_general_cli}")

print("--- END DIAGNOSTIC (Top of cli.py) ---")
# --- END DIAGNOSTIC CODE ---

# Original application imports
from chungoid.utils.state_manager import StateManager
from chungoid.utils.config_loader import get_config
from chungoid.utils.logger_setup import setup_logging
from chungoid.schemas.user_goal_schemas import UserGoalRequest # <<< ADD THIS IMPORT

# Imports needed for new 'flow resume' command
import asyncio
import json as py_json # Alias to avoid conflict with click option
from chungoid.runtime.orchestrator import AsyncOrchestrator #, ExecutionPlan no longer primary
from chungoid.schemas.master_flow import MasterExecutionPlan # <<< Import MasterExecutionPlan
from chungoid.utils.agent_resolver import RegistryAgentProvider, AgentCallable # MODIFIED: Added AgentCallable
# from chungoid.utils.flow_registry import FlowRegistry # No longer used directly for master plans
from chungoid.utils.master_flow_registry import MasterFlowRegistry
from chungoid.utils.agent_registry import AgentRegistry # Import AgentRegistry
from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card, core_stage_executor_agent # <<< For registration
from chungoid.schemas.flows import PausedRunDetails # Ensure this is imported
from chungoid.utils.config_loader import get_config # For default config
# <<< Import patch and AsyncMock >>>
from unittest.mock import patch, AsyncMock 
# import chungoid.server_prompts as server_prompts_pkg # REMOVED IMPORT

# New imports for metrics CLI
from chungoid.utils.metrics_store import MetricsStore
from chungoid.schemas.metrics import MetricEventType
from datetime import datetime, timezone # For summary display

# Import for MasterPlannerReviewerAgent registration
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent, get_agent_card_static as get_master_planner_reviewer_agent_card
# Import for MasterPlannerAgent registration
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent, get_agent_card_static as get_master_planner_agent_card

# Imports for Mock Agents (P2.6 MVP)
from chungoid.runtime.agents.mock_human_input_agent import get_agent_card_static as get_mock_human_input_agent_card
from chungoid.runtime.agents.mock_code_generator_agent import get_agent_card_static as get_mock_code_generator_agent_card
from chungoid.runtime.agents.mock_test_generator_agent import get_agent_card_static as get_mock_test_generator_agent_card
from chungoid.runtime.agents.mock_system_requirements_gathering_agent import get_agent_card_static as get_mock_system_requirements_gathering_agent_card

# New imports for MasterPlannerInput
from chungoid.schemas.agent_master_planner import MasterPlannerInput # <<< ADD THIS IMPORT

# Import for CoreCodeGeneratorAgent
from chungoid.runtime.agents.core_code_generator_agent import CodeGeneratorAgent, get_agent_card_static as get_code_generator_agent_card
# Import for CoreTestGeneratorAgent
from chungoid.runtime.agents.core_test_generator_agent import TestGeneratorAgent, get_agent_card_static as get_test_generator_agent_card
# Import for CoreCodeIntegrationAgentV1
from chungoid.runtime.agents.core_code_integration_agent import CoreCodeIntegrationAgentV1, get_agent_card_static as get_code_integration_agent_card

# New imports for MockTestGenerationAgentV1
from chungoid.runtime.agents.mock_test_generation_agent import MockTestGenerationAgentV1, get_agent_card_static as get_mock_test_generation_agent_v1_card

# Import the new system_test_runner_agent
from chungoid.runtime.agents import system_test_runner_agent # ADDED

# Ensure AgentID type is available if used for keys, though strings are fine for dict keys.
from chungoid.models import AgentID
# from chungoid.runtime.agents.base import AgentBase # For type hinting if needed # REMOVED

# --- ADDED IMPORTS FOR MOCK SETUP AGENT ---
from chungoid.agents.testing_mock_agents import (
    MockSetupAgentV1,
    MockFailPointAgentV1,
    # Assuming a get_mock_setup_agent_v1_card exists or we use AGENT_ID directly
)
# --- END ADDED IMPORTS ---

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


# ---------------------------------------------------------------------------
# Root Click group
# ---------------------------------------------------------------------------

@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(_LOG_LEVELS, case_sensitive=False),
    help="Logging verbosity (default: INFO)",
    show_default=True,
)
@click.pass_context
def cli(ctx: click.Context, log_level: str) -> None:  # noqa: D401 – imperative mood fine
    """Chungoid MCP CLI."""
    ctx.obj = {"log_level": log_level.upper()}
    _configure_logging(log_level.upper())
    # Perform diagnostic checks only if a specific env var is set
    if os.environ.get("CHUNGOID_CLI_DIAGNOSTICS") == "1":
        _perform_diagnostic_checks()
    
    logger.debug(f"CLI context object initialized: {ctx.obj}")


@cli.group()
@click.pass_context
def utils(ctx: click.Context):
    """Utility commands for Chungoid."""
    pass


# ---------------------------------------------------------------------------
# Sub-command: run (execute next stage)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    default=".",
)
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
@click.pass_context
def run(ctx: click.Context, project_dir: Path, as_json: bool) -> None:  # noqa: D401
    """Execute the next stage for *PROJECT_DIR* (default: current directory)."""
    logger = logging.getLogger("chungoid.cli.run")
    project_path = project_dir.expanduser().resolve()
    logger.info("Running next stage for project: %s", project_path)

    try:
        engine = ChungoidEngine(str(project_path))
        result: dict[str, Any] = engine.run_next_stage()
    except Exception as exc:  # pragma: no cover – surfacing
        logger.exception("Failed to run next stage: %s", exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(result, indent=2))
    else:
        # Pretty print using click's echo and json dumps for deterministic output
        click.echo(json.dumps(result, indent=2))


# ---------------------------------------------------------------------------
# Sub-command: init (bootstrap project skeleton)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    required=True,
)
@click.pass_context
def init(ctx: click.Context, project_dir: Path) -> None:  # noqa: D401
    """Initialise a new Chungoid project in *PROJECT_DIR*."""
    logger = logging.getLogger("chungoid.cli.init")
    project_path = project_dir.expanduser().resolve()

    # User confirmation for non-empty directory remains in CLI
    if project_path.exists() and any(project_path.iterdir()):
        # Check if .chungoid already exists. If so, init_project_structure will handle it idempotently.
        # If .chungoid doesn't exist in a non-empty dir, then we prompt.
        if not (project_path / ".chungoid").exists(): # Check PROJECT_CHUNGOID_DIR from constants ideally
            click.confirm(
                f"Directory {project_path} is not empty and does not contain an initialized '.chungoid' directory. "
                f"Continue and initialize Chungoid structure inside?",
                abort=True,
            )
    
    # Call the centralized init_project_structure function
    try:
        init_project_structure(project_path)
        logger.info(f"Successfully initialized project structure at {project_path}")
    except Exception as e:
        logger.error(f"Failed to initialize project at {project_path}: {e}", exc_info=True)
        click.echo(f"Error during project initialization: {e}", err=True)
        sys.exit(1)

    click.echo(f"✓ Initialised Chungoid project at {project_path}")
    click.echo("You can now run `chungoid run` to start Stage 0")


# ---------------------------------------------------------------------------
# Sub-command: status (print project status)
# ---------------------------------------------------------------------------

@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
)
@click.option("--json", "as_json", is_flag=True, help="Emit raw JSON status")
@click.pass_context
def status(ctx: click.Context, project_dir: Path, as_json: bool) -> None:  # noqa: D401
    """Show full status for *PROJECT_DIR* (default: current directory)."""
    logger = logging.getLogger("chungoid.cli.status")
    project_path = project_dir.expanduser().resolve()

    # Locate stages directory relative to package root
    core_root = Path(__file__).resolve().parent
    stages_dir = core_root / "server_prompts" / "stages"

    try:
        sm = StateManager(
            target_directory=str(project_path),
            server_stages_dir=str(stages_dir),
            use_locking=False,
        )
        status_data = sm.get_full_status()
    except (StatusFileError, ValueError) as exc:
        logger.error("Unable to read status: %s", exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(json.dumps(status_data, indent=2))
    else:
        # Simple human-readable summary
        runs = status_data.get("runs", [])
        if not runs:
            click.echo("(no runs recorded yet)")
            return
        click.echo("Runs (newest first):")
        for run in reversed(runs):
            click.echo(
                f"  • Stage {run.get('stage')} – {run.get('status')} – {run.get('timestamp', '')}"
            )


# ---------------------------------------------------------------------------
# Sub-command group: flow (manage and run MasterExecutionPlans)
# ---------------------------------------------------------------------------

@cli.group()
@click.pass_context
def flow(ctx: click.Context) -> None:  # noqa: D401
    """Commands for managing and executing Master Flows."""
    pass


@flow.command(name="run")
@click.option(
    "--master-flow-id",
    "master_flow_id_opt",
    type=str, 
    default=None, 
    help="ID of the master flow to run. Loaded from <project_dir>/.chungoid/master_flows/<id>.yaml or from persisted state."
)
@click.option(
    "--flow-yaml",
    "flow_yaml_opt",
    type=click.Path(file_okay=True, dir_okay=False, exists=True, path_type=Path),
    default=None,
    help="Path to a specific master flow YAML file to run. Overrides master_flow_id if both are given for loading, but master_flow_id is used as plan ID."
)
@click.option(
    "--goal", 
    type=str, 
    default=None, 
    help="High-level user goal to generate and run a new plan. Mutually exclusive with MASTER_FLOW_ID."
)
@click.option(
    "--project-dir",
    "project_dir_opt",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    help="Project directory containing the '.chungoid' subdirectory (default: current directory)."
)
@click.option(
    "--initial-context",
    type=str,
    default=None,
    help="JSON string containing initial context variables to pass to the flow."
)
@click.option(
    "--run-id",
    "run_id_override_opt",
    type=str,
    default=None,
    help="Specify a custom run ID for this execution."
)
@click.option(
    "--tags",
    type=str,
    default=None,
    help=(
        "Comma-separated tags for this run (e.g., 'test,nightly'). "
        "These are stored in project_status.json and can be used for filtering later."
    )
)
@click.pass_context
def flow_run(ctx: click.Context, 
             master_flow_id_opt: Optional[str], 
             flow_yaml_opt: Optional[Path], 
             goal: Optional[str], 
             project_dir_opt: Path, 
             initial_context: Optional[str],
             run_id_override_opt: Optional[str],
             tags: Optional[str]
             ) -> None:
    logger = logging.getLogger("chungoid.cli.flow_run")
    logger.info(f"'chungoid flow run' invoked. Master Flow ID: {master_flow_id_opt}, YAML: {flow_yaml_opt}, Goal: {goal}, Project Dir: {project_dir_opt}")

    if not master_flow_id_opt and not flow_yaml_opt and not goal:
        logger.error("Either --master-flow-id, --flow-yaml, or --goal must be provided.")
        click.echo("Error: Either --master-flow-id, --flow-yaml, or --goal must be provided.", err=True)
        raise click.exceptions.Exit(1)
    
    if goal and (master_flow_id_opt or flow_yaml_opt):
        logger.error("--goal option is mutually exclusive with --master-flow-id and --flow-yaml.")
        click.echo("Error: --goal is mutually exclusive with --master-flow-id and --flow-yaml.", err=True)
        raise click.exceptions.Exit(1)

    # Use resolved project path for consistency
    project_path = project_dir_opt.resolve()
    # Ensure .chungoid directory exists for StateManager and others
    chungoid_dir = project_path / PROJECT_CHUNGOID_DIR
    if not chungoid_dir.is_dir():
        logger.error(f"Project .chungoid directory not found at {chungoid_dir}. Please initialize the project or specify the correct directory.")
        click.echo(f"Error: Project .chungoid directory not found at {chungoid_dir}.", err=True)
        raise click.exceptions.Exit(1)

    config_file_path = chungoid_dir / "project_config.yaml"
    if not config_file_path.exists():
        logger.warning(f"Project config file not found at {config_file_path}. Using default configuration.")
        project_config = get_config() # Gets default config
    else:
        project_config = get_config(str(config_file_path))
    
    # Ensure project_root_dir is set in the config, as orchestrator might need it
    project_config["project_root_dir"] = str(project_path)

    parsed_initial_context = {}
    if initial_context:
        try:
            parsed_initial_context = py_json.loads(initial_context)
            logger.info(f"Parsed initial context: {parsed_initial_context}")
        except py_json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --initial-context: {e}")
            click.echo(f"Error: Invalid JSON in --initial-context: {e}", err=True)
            raise click.exceptions.Exit(1)

    # Add tags to initial_context if provided
    if tags:
        parsed_initial_context["_run_tags"] = [tag.strip() for tag in tags.split(',')]
        logger.info(f"Added tags to context: {parsed_initial_context['_run_tags']}")

    # Setup components needed for orchestration
    # TODO: Refactor agent registration to be more centralized and less CLI-dependent for common agents
    
    # Ensure project_root_dir from config is a Path object for AgentRegistry
    registry_project_root = Path(project_config["project_root_dir"])
    # Get chroma_mode from project_config, default to "persistent" if not present or None
    registry_chroma_mode = project_config.get("chroma_db_mode") or "persistent" 

    agent_registry = AgentRegistry(
        project_root=registry_project_root, 
        chroma_mode=registry_chroma_mode
    )
    
    # Register some basic system agents if not already (idempotent)
    agent_registry.add(get_master_planner_reviewer_agent_card(), overwrite=True) # Master Planner Reviewer
    agent_registry.add(get_master_planner_agent_card(), overwrite=True) # Master Planner Agent
    # agent_registry.add(core_stage_executor_card(), overwrite=True) # Core Stage Executor - if needed
    
    # Setup for mock agents (example)
    agent_registry.add(get_mock_human_input_agent_card(), overwrite=True)
    agent_registry.add(get_mock_code_generator_agent_card(), overwrite=True)
    agent_registry.add(get_mock_test_generator_agent_card(), overwrite=True)
    agent_registry.add(get_mock_system_requirements_gathering_agent_card(), overwrite=True)
    agent_registry.add(get_code_generator_agent_card(), overwrite=True) # Core Code Generator
    agent_registry.add(get_test_generator_agent_card(), overwrite=True) # Core Test Generator
    agent_registry.add(get_code_integration_agent_card(), overwrite=True)
    agent_registry.add(get_mock_test_generation_agent_v1_card(), overwrite=True)
    # agent_registry.add(system_test_runner_agent.get_agent_card_static(), overwrite=True) # system_test_runner_agent - REMOVED, will be in fallback

    # --- Prepare fallback agents for RegistryAgentProvider ---
    # Based on the pattern in flow_resume, but adapted for flow_run
    fallback_agents_map: Dict[AgentID, AgentCallable] = { # Assuming AgentCallable is defined or imported
        # It's safer to instantiate agents here if they need config or LLM providers
        # For system agents that are classes:
        get_master_planner_agent_card().agent_id: MasterPlannerAgent(llm_provider=MockLLMProvider()), # Example, adjust if LLM needed
        get_master_planner_reviewer_agent_card().agent_id: MasterPlannerReviewerAgent(config=project_config), # Pass config
        # For mock agents that are classes:
        get_mock_human_input_agent_card().agent_id: MockHumanInputAgent(),
        get_mock_code_generator_agent_card().agent_id: MockCodeGeneratorAgent(),
        get_mock_test_generation_agent_v1_card().agent_id: MockTestGenerationAgentV1(),
        get_mock_system_requirements_gathering_agent_card().agent_id: MockSystemRequirementsGatheringAgent(),
        MockSetupAgentV1.AGENT_ID: MockSetupAgentV1(config=project_config), # ADDED MockSetupAgentV1
        MockFailPointAgentV1.AGENT_ID: MockFailPointAgentV1(config=project_config), # ADDED MockFailPointAgentV1
        # For core functional agents that are classes:
        get_code_generator_agent_card().agent_id: CodeGeneratorAgent(),
        get_test_generator_agent_card().agent_id: TestGeneratorAgent(),
        get_code_integration_agent_card().agent_id: CoreCodeIntegrationAgentV1(config=project_config),
        # For module-based agents like system_test_runner_agent:
        system_test_runner_agent.AGENT_ID: system_test_runner_agent.invoke_async,
    }
    # Ensure CoreStageExecutorAgent is available if needed, e.g. via fallback or direct registration
    # For now, assuming it's not critical for this specific flow or is handled by registry if card exists.
    # fallback_agents_map[core_stage_executor_card().agent_id] = core_stage_executor_agent # If it's a direct function

    agent_provider = RegistryAgentProvider(registry=agent_registry, fallback=fallback_agents_map)
    
    # Determine server_stages_dir
    try:
        # Use relative path logic from cli.py location
        cli_file_path = Path(__file__).resolve() # e.g., /path/to/chungoid-core/src/chungoid/cli.py
        # Expected: chungoid-core/server_prompts/stages
        server_stages_dir_path = cli_file_path.parent.parent.parent / "server_prompts" / "stages"
        if not server_stages_dir_path.is_dir():
            logger.error(f"Determined server_stages_dir does not exist or is not a directory: {server_stages_dir_path}")
            # Fallback to a default relative to project_dir if primary derivation fails
            # This is a last resort and might not be correct if project_dir is not the repo root.
            server_stages_dir_path = project_path / "server_prompts" / "stages" 
            logger.warning(f"Attempting fallback server_stages_dir relative to project_dir: {server_stages_dir_path}")
            if not server_stages_dir_path.is_dir():
                 logger.error(f"Fallback server_stages_dir also not found: {server_stages_dir_path}. StateManager might fail if stages are needed from default location.")
                 # Allow it to proceed; StateManager will log warnings if path is invalid and needed.
                 # Set to a string value that StateManager might interpret or error on,
                 # or rely on a default within StateManager if it has one for such cases.
                 # For now, pass what we have, or a known conventional path if all else fails.
                 server_stages_dir_path = project_config.get("server_stages_dir_fallback", "./server_prompts/stages") # Use a config fallback

    except Exception as e: # Catch any unexpected error during path derivation
        logger.error(f"Unexpected error determining server_stages_dir: {e}. Using default fallback.")
        server_stages_dir_path = project_config.get("server_stages_dir_fallback", "./server_prompts/stages") # Ensure it's a string

    logger.info(f"Using server_stages_dir: {server_stages_dir_path}")

    state_manager = StateManager(
        target_directory=project_config["project_root_dir"], 
        server_stages_dir=str(server_stages_dir_path) # Ensure string
    )
    metrics_store = MetricsStore(project_root=Path(project_config["project_root_dir"])) # CORRECTED
    
    orchestrator = AsyncOrchestrator(
        config=project_config,
        agent_provider=agent_provider,
        state_manager=state_manager,
        metrics_store=metrics_store
        # master_planner_reviewer_agent_id is defaulted in orchestrator __init__
    )

    async def do_run():
        final_context = await orchestrator.run(
            flow_yaml_path=str(flow_yaml_opt) if flow_yaml_opt else None,
            master_plan_id=master_flow_id_opt,
            initial_context=parsed_initial_context,
            run_id_override=run_id_override_opt
        )

        if "_flow_error" in final_context:
            error_details = final_context["_flow_error"]
            logger.error(f"Flow execution finished with error: {error_details}")
            click.echo(f"Error during flow execution: {error_details}", err=True)
        else:
            logger.info("Flow execution completed.")
            # logger.debug(f"Final context: {final_context}") # Can be very verbose
            # Optionally print final context summary or specific outputs
            click.echo("Flow run finished.")
            if final_context.get("outputs"):
                 click.echo(f"Final outputs summary: { {k: str(v)[:100] + '...' if len(str(v)) > 100 else v for k,v in final_context['outputs'].items()} }")

    asyncio.run(do_run())


@flow.command(name="resume")
@click.argument("run_id", type=str)
@click.option(
    "--action",
    required=True,
    type=click.Choice(["retry", "retry_with_inputs", "skip_stage", "force_branch", "abort", "provide_clarification"], case_sensitive=False), # Added provide_clarification
    help="The action to perform for resuming the flow."
)
@click.option(
    "--inputs",
    type=str,
    default=None,
    help="JSON string containing inputs to merge into context (for 'retry_with_inputs' or 'provide_clarification' action)."
)
@click.option(
    "--target-stage",
    type=str,
    default=None,
    help="The stage ID to jump to (for 'force_branch' action)."
)
@click.pass_context
def flow_resume(ctx: click.Context, run_id: str, action: str, inputs: Optional[str], target_stage: Optional[str]) -> None:  # noqa: D401
    """Resume a paused flow execution with a specified action."""
    logger = logging.getLogger("chungoid.cli.flow.resume")
    project_root = Path.cwd() # Assuming resume is run from project root for now
    logger.info(f"Attempting to resume flow run_id={run_id} in project {project_root} with action='{action}'")

    # --- Argument Validation for action_data --- 
    action_data: Dict[str, Any] = {}
    if inputs:
        try:
            action_data["inputs"] = py_json.loads(inputs)
        except py_json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON string provided for --inputs: {e}", err=True)
            sys.exit(1)
    if action == "force_branch":
        if not target_stage:
            click.echo("Error: --target-stage is required for 'force_branch' action.", err=True)
            sys.exit(1)
        action_data["target_stage_id"] = target_stage
    elif target_stage:
        # target_stage is only used with force_branch
        click.echo("Warning: --target-stage is only used with the 'force_branch' action and will be ignored.", err=True)

    # --- Async runner --- 
    async def do_resume():
        # --- Find server_stages_dir (for StateManager) ---
        try:
            # Use relative path logic from cli.py location
            cli_file_path = Path(__file__).resolve() # e.g., /path/to/chungoid-core/src/chungoid/cli.py
            # Expected: chungoid-core/server_prompts/stages
            server_stages_dir_path = cli_file_path.parent.parent.parent / "server_prompts" / "stages"
            if not server_stages_dir_path.is_dir():
                logger.error(f"Determined server_stages_dir does not exist or is not a directory: {server_stages_dir_path}")
                # Fallback to a default relative to project_dir if primary derivation fails
                # This is a last resort and might not be correct if project_dir is not the repo root.
                project_path_for_fallback = Path(ctx.params.get('project_dir', '.')).resolve() # Get project_dir from context for resume
                server_stages_dir_path = project_path_for_fallback / "server_prompts" / "stages"
                logger.warning(f"Attempting fallback server_stages_dir relative to project_dir: {server_stages_dir_path}")
                if not server_stages_dir_path.is_dir():
                     logger.error(f"Fallback server_stages_dir also not found: {server_stages_dir_path}. StateManager might fail if stages are needed from default location.")
                     # Allow it to proceed; StateManager will log warnings if path is invalid and needed.
                     # Set to a string value that StateManager might interpret or error on,
                     # or rely on a default within StateManager if it has one for such cases.
                     # For now, pass what we have, or a known conventional path if all else fails.
                     default_config_for_resume = get_config() # Get default config to find fallback
                     server_stages_dir_path = default_config_for_resume.get("server_stages_dir_fallback", "./server_prompts/stages") # Use a config fallback

        except Exception as e: # Catch any unexpected error during path derivation
            logger.error(f"Unexpected error determining server_stages_dir (in resume): {e}. Using default fallback.")
            default_config_for_resume = get_config()
            server_stages_dir_path = default_config_for_resume.get("server_stages_dir_fallback", "./server_prompts/stages") # Ensure it's a string

        logger.info(f"Using server_stages_dir (in resume): {server_stages_dir_path}")

        # project_path is already defined above as Path.cwd() for resume context
        # state_manager = StateManager(
        #     target_directory=str(project_path),
        #     server_stages_dir=str(server_stages_dir_path)
        # ) # This was re-declared, using the one below from original logic for now

        # Initialize MetricsStore (needed by orchestrator)
        # Ensure project_config is loaded or use a default for metrics store path derivation
        # Assuming project_path is the correct root for metrics for resume
        metrics_store = MetricsStore(project_root=project_path) # project_path is Path.cwd() in resume
        
        # Initialize orchestrator
        # Load project_config for the orchestrator, or use a default if not found.
        # This is critical as orchestrator.config is used widely.
        # If resuming, the original project_config might not be easily available via ctx.obj
        # or might not be the one associated with the paused run. StateManager might hold it.
        # For now, using a fresh get_config() which might default or load from project_path/.chungoid/
        # This needs to be robust for resume.
        resumed_project_config = get_config(str(project_path / PROJECT_CHUNGOID_DIR / "project_config.yaml"))
        if "project_root_dir" not in resumed_project_config:
            resumed_project_config["project_root_dir"] = str(project_path) # Ensure it's set

        orchestrator = AsyncOrchestrator(
            config=resumed_project_config, 
            agent_provider=agent_provider,
            state_manager=state_manager, # StateManager now uses server_stages_dir_path correctly
            metrics_store=metrics_store 
        )

        action_data_dict = None
        if action == "retry_with_inputs":
            action_data_dict = action_data["inputs"]
        elif action == "provide_clarification":
            action_data_dict = action_data

        # --- Perform Resume Action --- 
        click.echo(f"Resuming flow run_id={run_id} with action '{action_data_for_resume["action"]}'...")
        try:
            # Call the orchestrator's resume_flow method
            # The orchestrator's resume_flow already knows about master_plan via its pipeline_def
            final_context_or_error = await orchestrator.resume_flow(
                run_id=run_id,
                action=action,
                action_data=action_data_dict
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred during orchestrator.resume_flow: {e}")
            click.echo(f"Error during flow resumption: {e}", err=True)
            sys.exit(1)

        # --- Handle Result --- 
        if isinstance(final_context_or_error, dict) and final_context_or_error.get("error"):
            click.echo(f"Error resuming flow: {final_context_or_error['error']}", err=True)
            # Should we exit with error code? Depends on if error is recoverable or implies resume failed.
            # For now, if orchestrator returns an error dict, treat as CLI failure.
            sys.exit(1)
        elif action == "abort" and isinstance(final_context_or_error, dict) and final_context_or_error.get("status") == "ABORTED":
            click.echo(f"Flow run '{run_id}' successfully aborted.")
        else:
            # Assuming successful resumption that didn't end in abort
            click.echo(f"Flow run '{run_id}' processed with action '{action}'. Check logs and project status for outcome.")
            # Optionally print some output from final_context if desired
            # click.echo(f"Final context (sample): {{k: v for i, (k,v) in enumerate(final_context_or_error.items()) if i < 3}}")

    # Run the async part
    try:
        asyncio.run(do_resume())
    except Exception as e:
        # This catches errors from within do_resume that weren't handled by sys.exit(1) already
        # (e.g., if asyncio.run itself fails, or an unhandled exception before a sys.exit)
        logger.error(f"High-level error during flow_resume execution: {e}")
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: metrics (inspect execution metrics)
# ---------------------------------------------------------------------------

@cli.group()
@click.pass_context
def metrics(ctx: click.Context) -> None:  # noqa: D401
    """Inspect execution metrics recorded by the system."""
    # The MetricsStore will be instantiated by each subcommand using a project_dir.
    pass

@metrics.command(name="list")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    show_default=True,
    help="Project directory where .chungoid/metrics.jsonl is located."
)
@click.option("--run-id", type=str, default=None, help="Filter by run ID.")
@click.option("--flow-id", type=str, default=None, help="Filter by flow ID.")
@click.option("--stage-id", type=str, default=None, help="Filter by stage ID.")
@click.option("--master-stage-id", type=str, default=None, help="Filter by master stage ID.")
@click.option("--agent-id", type=str, default=None, help="Filter by agent ID.")
@click.option(
    "--event-type", "event_types", # click uses the var name from the param list
    type=click.Choice([e.value for e in MetricEventType], case_sensitive=False),
    multiple=True, # Allow multiple --event-type flags
    default=None,
    help="Filter by one or more event types."
)
@click.option("--limit", type=int, default=100, show_default=True, help="Limit the number of events returned.")
@click.option("--sort-asc/--sort-desc", default=False, help="Sort events by timestamp ascending (default is descending).") # False for --sort-asc means sort_desc=True
@click.option("--output-format", type=click.Choice(["table", "json"], case_sensitive=False), default="table", show_default=True)
@click.pass_context
def metrics_list(
    ctx: click.Context,
    project_dir: Path,
    run_id: Optional[str],
    flow_id: Optional[str],
    stage_id: Optional[str],
    master_stage_id: Optional[str],
    agent_id: Optional[str],
    event_types: Optional[list[str]], # This will be list of strings from click.Choice
    limit: int,
    sort_asc: bool, # True if --sort-asc is passed. We want sort_desc for the store.
    output_format: str
) -> None:
    """List recorded metric events with optional filters."""
    logger = logging.getLogger("chungoid.cli.metrics.list")
    project_path = project_dir.expanduser().resolve()
    store = MetricsStore(project_root=project_path)

    # Convert string event types from CLI to MetricEventType enum members if provided
    enum_event_types: Optional[List[MetricEventType]] = None
    if event_types:
        try:
            enum_event_types = [MetricEventType(et_val) for et_val in event_types]
        except ValueError as e:
            click.echo(f"Error: Invalid event type provided: {e}", err=True)
            sys.exit(1)

    # sort_asc is True if --sort-asc is passed. We want sort_desc for the store.
    # So, if sort_asc is False (meaning --sort-desc or default), sort_desc_store should be True.
    # If sort_asc is True, sort_desc_store should be False.
    sort_desc_store = not sort_asc

    try:
        events = store.get_events(
            run_id=run_id,
            flow_id=flow_id,
            stage_id=stage_id,
            master_stage_id=master_stage_id,
            agent_id=agent_id,
            event_types=enum_event_types,
            limit=limit,
            sort_desc=sort_desc_store
        )
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}", exc_info=True)
        click.echo(f"Error retrieving metrics: {e}", err=True)
        sys.exit(1)

    if not events:
        click.echo("No metric events found matching the criteria.")
        return

    if output_format == "json":
        # Pydantic models' .model_dump_json() is convenient
        click.echo(py_json.dumps([event.model_dump() for event in events], indent=2))
    else: # table format
        # Simple textual table
        headers = ["Timestamp", "Type", "RunID", "FlowID", "StageID", "AgentID", "DataSummary"]
        click.echo(" | ".join(headers))
        click.echo("-" * (sum(len(h) for h in headers) + (len(headers) -1) * 3)) # Separator line

        for event in events:
            data_summary = py_json.dumps(event.data) if event.data else "-"
            if len(data_summary) > 50:
                data_summary = data_summary[:47] + "..."
            
            row = [
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                event.event_type.value,
                event.run_id or "-",
                event.flow_id or "-",
                event.stage_id or "-",
                event.agent_id or "-",
                data_summary
            ]
            click.echo(" | ".join(str(x) for x in row))

@metrics.command(name="summary")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    show_default=True,
    help="Project directory where .chungoid/metrics.jsonl is located."
)
@click.argument("run_id", type=str)
@click.pass_context
def metrics_summary(ctx: click.Context, project_dir: Path, run_id: str) -> None:
    """Provide a summary for a specific run_id."""
    logger = logging.getLogger("chungoid.cli.metrics.summary")
    project_path = project_dir.expanduser().resolve()
    store = MetricsStore(project_root=project_path)

    try:
        events = store.get_events(run_id=run_id, sort_desc=False) # Get all events for the run, oldest first
    except Exception as e:
        logger.error(f"Failed to retrieve metrics for run {run_id}: {e}", exc_info=True)
        click.echo(f"Error retrieving metrics for run {run_id}: {e}", err=True)
        sys.exit(1)

    if not events:
        click.echo(f"No metric events found for run_id: {run_id}")
        return

    summary: Dict[str, Any] = {
        "run_id": run_id,
        "flow_id": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "status": "INCOMPLETE", # Default status
        "total_stages_encountered": 0,
        "stages_completed_success": 0,
        "stages_completed_failure": 0,
        "errors": []
    }

    flow_start_event: Optional[MetricEvent] = None
    flow_end_event: Optional[MetricEvent] = None
    stage_ids_encountered = set()

    for event in events:
        if summary["flow_id"] is None and event.flow_id:
            summary["flow_id"] = event.flow_id

        if event.event_type == MetricEventType.FLOW_START:
            if flow_start_event is None: # Take the first one
                flow_start_event = event
                summary["start_time"] = event.timestamp.isoformat()
        
        elif event.event_type == MetricEventType.FLOW_END:
            flow_end_event = event # Take the last one encountered
            summary["end_time"] = event.timestamp.isoformat()
            if "final_status" in event.data:
                summary["status"] = event.data["final_status"]
            if "total_duration_seconds" in event.data:
                 summary["duration_seconds"] = event.data["total_duration_seconds"]

        elif event.event_type == MetricEventType.STAGE_START:
            if event.stage_id:
                stage_ids_encountered.add(event.stage_id)
        
        elif event.event_type == MetricEventType.STAGE_END:
            if event.stage_id:
                stage_ids_encountered.add(event.stage_id)
            if event.data.get("status") == "COMPLETED_SUCCESS":
                summary["stages_completed_success"] += 1
            elif event.data.get("status") == "COMPLETED_FAILURE":
                summary["stages_completed_failure"] += 1
                error_detail = event.data.get("error_details", event.data.get("error_message", "Unknown error"))
                summary["errors"].append(f"Stage {event.stage_id or '?'} failed: {error_detail}")
        
        # If orchestrator reports an error directly
        elif event.event_type == MetricEventType.ORCHESTRATOR_INFO and event.data.get("level") == "ERROR":
            summary["errors"].append(f"Orchestrator error: {event.data.get('message', 'Unknown orchestrator error')}")
    
    summary["total_stages_encountered"] = len(stage_ids_encountered)

    # Calculate duration if not in FLOW_END event
    if summary["duration_seconds"] is None and flow_start_event and flow_end_event:
        duration = flow_end_event.timestamp - flow_start_event.timestamp
        summary["duration_seconds"] = duration.total_seconds()
    elif summary["duration_seconds"] is None and flow_start_event and summary["status"] != "INCOMPLETE":
        # If flow ended but no explicit FLOW_END event with duration, estimate from last event if available
        # This is a fallback, FLOW_END should ideally provide it.
        last_event_ts = events[-1].timestamp
        duration = last_event_ts - flow_start_event.timestamp
        summary["duration_seconds"] = duration.total_seconds()

    # Refine overall status if not set by FLOW_END
    if summary["status"] == "INCOMPLETE" and flow_start_event and not flow_end_event:
        summary["status"] = "RUNNING_OR_ABORTED_MIDFLIGHT" # Or Paused if we can infer that
    elif summary["status"] == "INCOMPLETE" and flow_end_event is None and summary["errors"]:
        summary["status"] = "COMPLETED_WITH_ERRORS_NO_FLOW_END" # Likely an unhandled crash
    elif summary["status"] == "INCOMPLETE" and flow_end_event is None:
        summary["status"] = "UNKNOWN_ENDED_NO_FLOW_END"

    click.echo(f"Summary for Run ID: {summary['run_id']}")
    click.echo(f"  Flow ID: {summary['flow_id'] or 'N/A'}")
    click.echo(f"  Status: {summary['status']}")
    click.echo(f"  Start Time: {summary['start_time'] or 'N/A'}")
    click.echo(f"  End Time: {summary['end_time'] or 'N/A'}")
    click.echo(f"  Duration: {summary['duration_seconds']:.2f}s" if summary['duration_seconds'] is not None else "Duration: N/A")
    click.echo(f"  Total Stages Encountered: {summary['total_stages_encountered']}")
    click.echo(f"  Stages Succeeded: {summary['stages_completed_success']}")
    click.echo(f"  Stages Failed: {summary['stages_completed_failure']}")
    if summary["errors"]:
        click.echo("  Errors:")
        for err in summary["errors"]:
            click.echo(f"    - {err[:200]}{'...' if len(err) > 200 else ''}") # Truncate long errors
    else:
        click.echo("  Errors: None")

@metrics.command(name="report")
@click.option(
    "--project-dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    show_default=True,
    help="Project directory containing the .chungoid folder (default: current directory)."
)
@click.option("--run-id", type=str, default=None, help="Generate report for a specific Run ID.")
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=None, # Defaults to project_dir/.chungoid/reports/ in the script
    help="Directory to save the HTML report."
)
@click.option("--limit", type=int, default=1000, show_default=True, help="Max events in report.")
@click.pass_context
def metrics_report(
    ctx: click.Context,
    project_dir: Path,
    run_id: Optional[str],
    output_dir: Optional[Path],
    limit: int
) -> None:
    """Generate an HTML metrics report."""
    logger = logging.getLogger("chungoid.cli.metrics.report")
    project_path = project_dir.expanduser().resolve()

    # Construct path to the script relative to this CLI file's location
    # This assumes cli.py is in src/chungoid/ and the script is in scripts/
    # Adjust if your directory structure is different.
    cli_dir = Path(__file__).resolve().parent
    script_path = cli_dir.parent.parent / "scripts" / "generate_metrics_report.py"

    if not script_path.exists():
        click.echo(f"Error: Report generation script not found at {script_path}", err=True)
        sys.exit(1)

    cmd = [sys.executable, str(script_path), "--project-dir", str(project_path)]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir)])
    cmd.extend(["--limit", str(limit)])

    try:
        logger.info(f"Executing report generation script: {' '.join(cmd)}")
        # Use subprocess.run, ensure to capture output and check for errors
        result = subprocess.run(cmd, capture_output=True, text=True, check=False) # check=False to handle errors manually
        
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr:
            click.echo(result.stderr, err=True)
        
        if result.returncode != 0:
            click.echo(f"Error: Report generation script failed with exit code {result.returncode}.", err=True)
            # Optionally, could exit here if script failure is critical for CLI command
            # sys.exit(result.returncode)

    except FileNotFoundError:
        click.echo(f"Error: Python interpreter not found or script path incorrect.", err=True)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(f"Report generation script execution failed: {e}", exc_info=True)
        click.echo(f"Error during report generation: {e}. Output:\n{e.stdout}\n{e.stderr}", err=True)
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred while trying to run the report script: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)

# ---------------------------------------------------------------------------
# Entrypoint for `python -m chungoid.cli` (optional)
# ---------------------------------------------------------------------------

@click.command("show-config")
@click.pass_context
def show_config(ctx):
    '''Displays the current Chungoid project configuration.'''
    try:
        config = get_config()
        if not config:
            click.secho("Error: Project configuration could not be loaded.", fg="red")
            click.secho("Please ensure you are in a Chungoid project directory or provide one with --project-dir.", fg="red")
            ctx.exit(1)
        # Updated title to include config_file_loaded, matching test assertion
        config_file_location = config.get('config_file_loaded', 'N/A')
        click.secho(f"Current Project Configuration (from {config_file_location}):", fg="cyan", bold=True)
        click.echo(f"  project_root: {config['project_root']}")
        click.echo(f"  dot_chungoid_path: {config['dot_chungoid_path']}")
        click.echo(f"  state_manager_db_path: {config['state_manager_db_path']}")
        click.echo(f"  master_flows_dir: {config['master_flows_dir']}")
        click.echo(f"  host_system_info: {config['host_system_info']}")
        click.echo(f"  log_level: {config['log_level']}")
        # The problematic example block is now fully removed.
    except Exception as e:
        click.secho(f"An unexpected error occurred while retrieving configuration: {e}", fg="red")
        # Removed: logger.error(f"Error in show_config command: {e}", exc_info=True)
        ctx.exit(1)
        
utils.add_command(show_config)
if __name__ == "__main__":  # pragma: no cover
    cli()





