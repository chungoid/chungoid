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
from chungoid.runtime.agents.mocks.mock_human_input_agent import get_agent_card_static as get_mock_human_input_agent_card
from chungoid.runtime.agents.mocks.mock_code_generator_agent import get_agent_card_static as get_mock_code_generator_agent_card
from chungoid.runtime.agents.mocks.mock_test_generator_agent import get_agent_card_static as get_mock_test_generator_agent_card
from chungoid.runtime.agents.mocks.mock_system_requirements_gathering_agent import get_agent_card_static as get_mock_system_requirements_gathering_agent_card

from chungoid.runtime.agents.system_master_planner_agent import get_agent_card_static as get_master_planner_agent_card
from chungoid.runtime.agents.system_master_planner_reviewer_agent import get_agent_card_static as get_master_planner_reviewer_agent_card

# For Agent Classes (used in the fallback_agents_map and direct instantiation)
# from chungoid.runtime.agents.core_stage_executor import CoreStageExecutorAgent # Removed - it's a function, not a class
from chungoid.runtime.agents.mocks.mock_human_input_agent import MockHumanInputAgent
from chungoid.runtime.agents.mocks.mock_code_generator_agent import MockCodeGeneratorAgent
from chungoid.runtime.agents.mocks.mock_test_generator_agent import MockTestGeneratorAgent
from chungoid.runtime.agents.mocks.mock_system_requirements_gathering_agent import MockSystemRequirementsGatheringAgent

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

print("--- END DIAGNOSTIC (Top of cli.py) ---\n")
# --- END DIAGNOSTIC CODE ---

# Original application imports
from chungoid.utils.state_manager import StateManager # StatusFileError was used here but not defined, assume from StateManager or custom exceptions
from chungoid.utils.config_loader import load_config, ConfigError # MODIFIED_LINE
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
from chungoid.runtime.agents.mocks.mock_human_input_agent import get_agent_card_static as get_mock_human_input_agent_card
from chungoid.runtime.agents.mocks.mock_code_generator_agent import get_agent_card_static as get_mock_code_generator_agent_card
from chungoid.runtime.agents.mocks.mock_test_generator_agent import get_agent_card_static as get_mock_test_generator_agent_card
from chungoid.runtime.agents.mocks.mock_system_requirements_gathering_agent import get_agent_card_static as get_mock_system_requirements_gathering_agent_card

# New imports for MasterPlannerInput
from chungoid.schemas.agent_master_planner import MasterPlannerInput # <<< ADD THIS IMPORT

# Import for CoreCodeGeneratorAgent
from chungoid.runtime.agents.core_code_generator_agent import CodeGeneratorAgent, get_agent_card_static as get_code_generator_agent_card
# Import for CoreTestGeneratorAgent
from chungoid.runtime.agents.core_test_generator_agent import TestGeneratorAgent, get_agent_card_static as get_test_generator_agent_card
# Import for CoreCodeIntegrationAgentV1
from chungoid.runtime.agents.core_code_integration_agent import CoreCodeIntegrationAgentV1, get_agent_card_static as get_code_integration_agent_card

# New imports for MockTestGenerationAgentV1
from chungoid.runtime.agents.mocks.mock_test_generation_agent import MockTestGenerationAgentV1, get_agent_card_static as get_mock_test_generation_agent_v1_card

# Import the new system_test_runner_agent
from chungoid.runtime.agents import system_test_runner_agent # ADDED

# Ensure AgentID type is available if used for keys, though strings are fine for dict keys.
from chungoid.models import AgentID
# from chungoid.runtime.agents.base import AgentBase # For type hinting if needed # REMOVED

# --- ADDED IMPORTS FOR MOCK SETUP AGENT ---
from chungoid.runtime.agents.mocks.testing_mock_agents import (
    MockSetupAgentV1,
    MockFailPointAgentV1,
    get_mock_agent_fallback_map # ADDED THIS IMPORT
    # Assuming a get_mock_setup_agent_v1_card exists or we use AGENT_ID directly
)
# --- END ADDED IMPORTS ---

# Assuming StatusFileError might be a custom exception, if not defined elsewhere, it might need to be.
# For now, let's assume it's imported or defined if critical. If it's from a known module, add import.
# Example: from chungoid.utils.exceptions import StatusFileError (if it exists there)
# If it was a typo and meant something else, that would need correction.
# For now, proceeding as if it will be resolved by existing imports or is not critical path for this edit.

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

def _perform_diagnostic_checks() -> None:
    # This is a placeholder for any future diagnostic checks needed at CLI startup
    logger.debug("Performing CLI diagnostic checks...")
    # Example: Check if essential directories exist or specific config files are readable
    # For now, it does nothing beyond logging.
    pass

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

# This command might be deprecated or less used if `flow run` is the primary way
@cli.command()
@click.argument(
    "project_dir",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path),
    default=".",
)
@click.option("--json", "as_json", is_flag=True, help="Emit result as JSON")
@click.pass_context
def run(ctx: click.Context, project_dir: Path, as_json: bool) -> None:  # noqa: D401
    """Execute the next stage for *PROJECT_DIR* (default: current directory). (May be deprecated)"""
    logger = logging.getLogger("chungoid.cli.run")
    project_path = project_dir.expanduser().resolve()
    logger.warning("The 'chungoid run' command might be deprecated in favor of 'chungoid flow run'.")
    logger.info("Running next stage for project: %s", project_path)

    # This part of 'run' likely needs to be updated to use the new orchestrator
    # or be officially deprecated if `flow run` is the sole execution path.
    # For now, it's left as is but with a warning.
    # Assuming ChungoidEngine might be an older way or needs to be refactored/removed.
    try:
        # engine = ChungoidEngine(str(project_path)) # ChungoidEngine might not exist or be outdated
        # result: dict[str, Any] = engine.run_next_stage()
        click.echo("Error: 'chungoid run' is likely deprecated. Use 'chungoid flow run'.", err=True)
        sys.exit(1)
    except NameError: # If ChungoidEngine is not defined
        logger.error("'ChungoidEngine' not found. 'chungoid run' command is likely non-functional or deprecated.")
        click.echo("Error: 'chungoid run' command is non-functional. Use 'chungoid flow run'.", err=True)
        sys.exit(1)
    except Exception as exc:  # pragma: no cover – surfacing
        logger.exception("Failed to run next stage: %s", exc)
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    # if as_json:
    #     click.echo(json.dumps(result, indent=2))
    # else:
    #     click.echo(json.dumps(result, indent=2))


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

    if project_path.exists() and any(project_path.iterdir()):
        if not (project_path / PROJECT_CHUNGOID_DIR).exists():
            click.confirm(
                f"Directory {project_path} is not empty and does not contain an initialized '{PROJECT_CHUNGOID_DIR}' directory. "
                f"Continue and initialize Chungoid structure inside?",
                abort=True,
            )
    
    try:
        init_project_structure(project_path)
        logger.info(f"Successfully initialized project structure at {project_path}")
    except Exception as e:
        logger.error(f"Failed to initialize project at {project_path}: {e}", exc_info=True)
        click.echo(f"Error during project initialization: {e}", err=True)
        sys.exit(1)

    click.echo(f"✓ Initialised Chungoid project at {project_path}")
    click.echo(f"You can now use 'chungoid flow run' with a flow YAML or goal.")


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

    # Determine server_stages_dir robustly
    try:
        cli_file_path = Path(__file__).resolve()
        server_stages_dir_path = cli_file_path.parent.parent.parent / "server_prompts" / "stages"
        if not server_stages_dir_path.is_dir():
            logger.warning(f"Default server_stages_dir not found at {server_stages_dir_path}, trying relative to project.")
            server_stages_dir_path = project_path / "server_prompts" / "stages"
            if not server_stages_dir_path.is_dir():
                logger.error(f"Fallback server_stages_dir also not found at {server_stages_dir_path}. Status might be incomplete.")
                # Use a placeholder or a known default if StateManager requires a valid path
                server_stages_dir_path = DEFAULT_SERVER_STAGES_DIR # From constants
    except Exception:
        server_stages_dir_path = DEFAULT_SERVER_STAGES_DIR

    logger.debug(f"Using server_stages_dir for StateManager in status: {server_stages_dir_path}")

    try:
        sm = StateManager(
            target_directory=str(project_path),
            server_stages_dir=str(server_stages_dir_path), # Ensure it's a string
            use_locking=False,
        )
        status_data = sm.get_full_status()
    except Exception as exc: # Catch a broader exception if StatusFileError is not specifically defined/imported
        logger.error("Unable to read status: %s", exc)
        click.echo(f"Error reading status: {exc}", err=True)
        sys.exit(1)

    if as_json:
        click.echo(py_json.dumps(status_data, indent=2))
    else:
        runs = status_data.get("runs", [])
        if not runs:
            click.echo("(no runs recorded yet)")
            return
        click.echo("Runs (newest first):")
        for run_info in reversed(runs): # Assuming runs is a list of dicts
            stage_id = run_info.get('stage_id', run_info.get('stage', 'N/A')) # Adapt to actual key
            run_status = run_info.get('status', 'N/A')
            timestamp = run_info.get('timestamp', '')
            click.echo(
                f"  • Stage {stage_id} – {run_status} – {timestamp}"
            )
        
        paused_runs = status_data.get("paused_runs")
        if paused_runs:
            click.echo("\nPaused Runs:")
            for run_id, details in paused_runs.items():
                click.echo(f"  • Run ID: {run_id}")
                click.echo(f"    Flow ID: {details.get('flow_id', 'N/A')}")
                click.echo(f"    Paused at Stage: {details.get('paused_at_stage_id', 'N/A')}")
                click.echo(f"    Status: {details.get('status', 'N/A')}")
                click.echo(f"    Timestamp: {details.get('timestamp', 'N/A')}")


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
    help="High-level user goal to generate and run a new plan. Mutually exclusive with MASTER_FLOW_ID/flow_yaml_opt."
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

    project_path = project_dir_opt.resolve()
    chungoid_dir = project_path / PROJECT_CHUNGOID_DIR
    if not chungoid_dir.is_dir():
        logger.error(f"Project '{PROJECT_CHUNGOID_DIR}' directory not found at {chungoid_dir}. Please initialize the project or specify the correct directory.")
        click.echo(f"Error: Project '{PROJECT_CHUNGOID_DIR}' directory not found at {chungoid_dir}.", err=True)
        raise click.exceptions.Exit(1)

    config_file_path = chungoid_dir / "project_config.yaml"
    project_config = load_config(str(config_file_path) if config_file_path.exists() else None)
    project_config["project_root_dir"] = str(project_path) # Ensure this is always set based on CLI arg

    parsed_initial_context = {}
    if initial_context:
        try:
            parsed_initial_context = py_json.loads(initial_context)
            logger.info(f"Parsed initial context: {parsed_initial_context}")
        except py_json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --initial-context: {e}")
            click.echo(f"Error: Invalid JSON in --initial-context: {e}", err=True)
            raise click.exceptions.Exit(1)

    if tags:
        parsed_initial_context["_run_tags"] = [tag.strip() for tag in tags.split(',')]
        logger.info(f"Added tags to context: {parsed_initial_context['_run_tags']}")

    registry_project_root = Path(project_config["project_root_dir"])
    registry_chroma_mode = project_config.get("chromadb", {}).get("mode", "persistent") # More robust access

    agent_registry = AgentRegistry(
        project_root=registry_project_root, 
        chroma_mode=registry_chroma_mode
    )
    
    # Register system and mock agents
    # These should ideally be registered once globally or loaded dynamically
    # For CLI, ensuring they are available for the run:
    agent_registry.add(get_master_planner_reviewer_agent_card(), overwrite=True)
    agent_registry.add(get_master_planner_agent_card(), overwrite=True)
    agent_registry.add(get_mock_human_input_agent_card(), overwrite=True)
    agent_registry.add(get_mock_code_generator_agent_card(), overwrite=True)
    agent_registry.add(get_mock_test_generator_agent_card(), overwrite=True) # Alias for MockTestGenerationAgentV1 for now
    agent_registry.add(get_mock_system_requirements_gathering_agent_card(), overwrite=True)
    agent_registry.add(get_code_generator_agent_card(), overwrite=True)
    agent_registry.add(get_test_generator_agent_card(), overwrite=True)
    agent_registry.add(get_code_integration_agent_card(), overwrite=True)
    agent_registry.add(get_mock_test_generation_agent_v1_card(), overwrite=True)

    # For simplicity in RegistryAgentProvider, we provide one merged map. Let's ensure system agents are there.
    
    # Explicitly add core system agents to the fallback map if not already covered.
    # These are agents that provide core functionality and might have specific mock behaviors for MVPs.
    core_system_agents = {
        MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
        MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent,
        CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # MVP uses its mocked output
        TestGeneratorAgent.AGENT_ID: TestGeneratorAgent, # MVP uses its mocked output
        CoreCodeIntegrationAgentV1.AGENT_ID: CoreCodeIntegrationAgentV1, # Handles actual file edits
        system_test_runner_agent.AGENT_ID: system_test_runner_agent.invoke_async # Use the AGENT_ID and invoke_async function for functional agents
        # Add other essential system agents here if their local Python class should be directly invokable via fallback
    }
    
    # Start with mock agents from testing_mock_agents.py
    final_fallback_map: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map()
    # Add/override with core system agents
    final_fallback_map.update(core_system_agents)
    
    agent_provider = RegistryAgentProvider(registry=agent_registry, fallback=final_fallback_map)
    
    try:
        cli_file_path = Path(__file__).resolve()
        server_stages_dir_path_str = str(cli_file_path.parent.parent.parent / "server_prompts" / "stages")
        if not Path(server_stages_dir_path_str).is_dir():
            server_stages_dir_path_str = str(project_path / "server_prompts" / "stages")
            if not Path(server_stages_dir_path_str).is_dir():
                server_stages_dir_path_str = project_config.get("server_stages_dir_fallback", DEFAULT_SERVER_STAGES_DIR)
    except Exception:
        server_stages_dir_path_str = project_config.get("server_stages_dir_fallback", DEFAULT_SERVER_STAGES_DIR)

    logger.info(f"Using server_stages_dir for StateManager: {server_stages_dir_path_str}")

    state_manager = StateManager(
        target_directory=project_config["project_root_dir"], 
        server_stages_dir=server_stages_dir_path_str
    )
    metrics_store_root = Path(project_config["project_root_dir"]) # Ensure this is a Path for MetricsStore
    metrics_store = MetricsStore(project_root=metrics_store_root)
    
    orchestrator = AsyncOrchestrator(
        config=project_config,
        agent_provider=agent_provider,
        state_manager=state_manager,
        metrics_store=metrics_store
    )

    async def do_run():
        final_context = await orchestrator.run(
            flow_yaml_path=str(flow_yaml_opt) if flow_yaml_opt else None,
            master_plan_id=master_flow_id_opt, # Will be used as plan_id by orchestrator
            goal_str=goal, # Pass goal to orchestrator
            initial_context=parsed_initial_context,
            run_id_override=run_id_override_opt
        )

        if "_flow_error" in final_context:
            error_details = final_context["_flow_error"]
            logger.error(f"Flow execution finished with error: {error_details}")
            click.echo(f"Error during flow execution: {error_details}", err=True)
            # Consider sys.exit(1) here for scriptability
        else:
            logger.info("Flow execution completed.")
            click.echo("Flow run finished.")
            if final_context.get("outputs"):
                 outputs_summary = {k: str(v)[:100] + '...' if len(str(v)) > 100 else v for k,v in final_context['outputs'].items()}
                 click.echo(f"Final outputs summary: {py_json.dumps(outputs_summary, indent=2)}")


    try:
        asyncio.run(do_run())
    except Exception as e:
        logger.exception("Unhandled exception during 'flow run' execution.")
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


@flow.command(name="resume")
@click.argument("run_id", type=str)
@click.option(
    "--project-dir", # Added for consistency and proper config/state loading
    "project_dir_opt",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    help="Project directory for the paused flow (default: current directory)."
)
@click.option(
    "--action",
    required=True,
    type=click.Choice(["retry", "retry_with_inputs", "skip_stage", "force_branch", "abort", "provide_clarification"], case_sensitive=False),
    help="The action to perform for resuming the flow."
)
@click.option(
    "--inputs",
    type=str,
    default=None,
    help="JSON string containing inputs to merge (for 'retry_with_inputs' or 'provide_clarification')."
)
@click.option(
    "--target-stage",
    type=str,
    default=None,
    help="The stage ID to jump to (for 'force_branch' action)."
)
@click.pass_context
def flow_resume(ctx: click.Context, run_id: str, project_dir_opt: Path, action: str, inputs: Optional[str], target_stage: Optional[str]) -> None:
    logger = logging.getLogger("chungoid.cli.flow.resume")
    project_path = project_dir_opt.resolve()
    logger.info(f"Attempting to resume flow run_id={run_id} in project {project_path} with action='{action}'")

    action_data_for_resume: Dict[str, Any] = {} # Corrected variable name
    if inputs:
        try:
            # For 'provide_clarification', the entire JSON might be the data.
            # For 'retry_with_inputs', it's specifically for 'inputs' key.
            if action.lower() == "provide_clarification":
                 action_data_for_resume = py_json.loads(inputs)
            else:
                 action_data_for_resume["inputs"] = py_json.loads(inputs)
        except py_json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON string provided for --inputs: {e}", err=True)
            sys.exit(1)
            
    if action.lower() == "force_branch":
        if not target_stage:
            click.echo("Error: --target-stage is required for 'force_branch' action.", err=True)
            sys.exit(1)
        action_data_for_resume["target_stage_id"] = target_stage
    elif target_stage:
        click.echo("Warning: --target-stage is only used with the 'force_branch' action and will be ignored.", err=True)


    async def do_resume():
        # Configuration and setup similar to flow_run for consistency
        chungoid_dir = project_path / PROJECT_CHUNGOID_DIR
        if not chungoid_dir.is_dir(): # Should exist if resuming
            logger.error(f"Project '{PROJECT_CHUNGOID_DIR}' directory not found at {chungoid_dir} for resume.")
            click.echo(f"Error: Project '{PROJECT_CHUNGOID_DIR}' directory not found at {chungoid_dir}.", err=True)
            sys.exit(1)

        config_file_path = chungoid_dir / "project_config.yaml"
        resumed_project_config = load_config(str(config_file_path) if config_file_path.exists() else None)
        resumed_project_config["project_root_dir"] = str(project_path)

        registry_project_root = Path(resumed_project_config["project_root_dir"])
        registry_chroma_mode = resumed_project_config.get("chromadb", {}).get("mode", "persistent")

        agent_registry = AgentRegistry(project_root=registry_project_root, chroma_mode=registry_chroma_mode)
        # Ensure agents are registered as in flow_run, or that registry is persistent/shared
        agent_registry.add(get_master_planner_reviewer_agent_card(), overwrite=True)
        agent_registry.add(get_master_planner_agent_card(), overwrite=True)
        # ... (add other necessary agents as in flow_run)
        agent_registry.add(get_mock_human_input_agent_card(), overwrite=True)
        agent_registry.add(get_mock_code_generator_agent_card(), overwrite=True)
        agent_registry.add(get_mock_test_generation_agent_v1_card(), overwrite=True)
        agent_registry.add(get_mock_system_requirements_gathering_agent_card(), overwrite=True)
        agent_registry.add(get_code_generator_agent_card(), overwrite=True)
        agent_registry.add(get_test_generator_agent_card(), overwrite=True)
        agent_registry.add(get_code_integration_agent_card(), overwrite=True)


        fallback_agents_map_resume: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map()
        # Add/override with core system agents
        core_system_agents_resume = {
            MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
            MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent,
            # Add other essential system agents here if needed
        }
        # Merge, ensuring core system agents take precedence if IDs overlap (though unlikely for mocks)
        # Or, decide on a clear priority. For now, let mock map be primary, then add system agents.
        # A more robust way would be to have separate maps and the resolver check them in order.
        # For simplicity in RegistryAgentProvider, we provide one merged map. Let's ensure system agents are there.
        
        # Start with mock agents
        final_fallback_map_resume: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map()
        # Add/override with core system agents
        final_fallback_map_resume.update(core_system_agents_resume)
        
        agent_provider = RegistryAgentProvider(registry=agent_registry, fallback=final_fallback_map_resume)

        try:
            cli_file_path = Path(__file__).resolve()
            server_stages_dir_path_str_resume = str(cli_file_path.parent.parent.parent / "server_prompts" / "stages")
            if not Path(server_stages_dir_path_str_resume).is_dir():
                server_stages_dir_path_str_resume = str(project_path / "server_prompts" / "stages")
                if not Path(server_stages_dir_path_str_resume).is_dir():
                    server_stages_dir_path_str_resume = resumed_project_config.get("server_stages_dir_fallback", DEFAULT_SERVER_STAGES_DIR)
        except Exception:
            server_stages_dir_path_str_resume = resumed_project_config.get("server_stages_dir_fallback", DEFAULT_SERVER_STAGES_DIR)
        
        logger.info(f"Using server_stages_dir for StateManager (resume): {server_stages_dir_path_str_resume}")
        state_manager = StateManager(
            target_directory=resumed_project_config["project_root_dir"],
            server_stages_dir=server_stages_dir_path_str_resume
        )
        metrics_store = MetricsStore(project_root=Path(resumed_project_config["project_root_dir"]))
        
        orchestrator = AsyncOrchestrator(
            config=resumed_project_config, 
            agent_provider=agent_provider,
            state_manager=state_manager,
            metrics_store=metrics_store 
        )
        
        click.echo(f"Resuming flow run_id={run_id} with action '{action}' and data '{action_data_for_resume}'...")
        try:
            final_context_or_error = await orchestrator.resume_flow(
                run_id=run_id,
                action=action, # Pass the string action directly
                action_data=action_data_for_resume # Pass the prepared dictionary
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred during orchestrator.resume_flow: {e}")
            click.echo(f"Error during flow resumption: {e}", err=True)
            sys.exit(1)

        if isinstance(final_context_or_error, dict) and final_context_or_error.get("_flow_error"):
            click.echo(f"Error resuming flow: {final_context_or_error['_flow_error']}", err=True)
            sys.exit(1)
        elif action.lower() == "abort" and isinstance(final_context_or_error, dict) and final_context_or_error.get("status") == FlowPauseStatus.ABORTED_BY_USER.value: # Check against enum value
            click.echo(f"Flow run '{run_id}' successfully aborted.")
        else:
            click.echo(f"Flow run '{run_id}' processed with action '{action}'. Check logs and project status for outcome.")


    try:
        asyncio.run(do_resume())
    except Exception as e:
        logger.error(f"High-level error during flow_resume execution: {e}", exc_info=True)
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Sub-command: metrics (inspect execution metrics)
# ---------------------------------------------------------------------------

@cli.group()
@click.pass_context
def metrics(ctx: click.Context) -> None:  # noqa: D401
    """Inspect execution metrics recorded by the system."""
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
@click.option("--master-stage-id", type=str, default=None, help="Filter by master stage ID.") # For MasterExecutionPlan stages
@click.option("--agent-id", type=str, default=None, help="Filter by agent ID.")
@click.option(
    "--event-type", "event_types", 
    type=click.Choice([e.value for e in MetricEventType], case_sensitive=False),
    multiple=True, 
    default=None,
    help="Filter by one or more event types."
)
@click.option("--limit", type=int, default=100, show_default=True, help="Limit the number of events returned.")
@click.option("--sort-asc/--sort-desc", "sort_ascending", default=False, help="Sort events by timestamp ascending (default is descending).")
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
    event_types: Optional[list[str]], 
    limit: int,
    sort_ascending: bool, 
    output_format: str
) -> None:
    """List recorded metric events with optional filters."""
    logger = logging.getLogger("chungoid.cli.metrics.list")
    project_path = project_dir.expanduser().resolve()
    store = MetricsStore(project_root=project_path)

    enum_event_types_filter: Optional[List[MetricEventType]] = None
    if event_types:
        try:
            enum_event_types_filter = [MetricEventType(et_val) for et_val in event_types]
        except ValueError as e:
            click.echo(f"Error: Invalid event type provided: {e}", err=True)
            sys.exit(1)

    sort_desc_for_store = not sort_ascending

    try:
        events = store.get_events(
            run_id=run_id,
            flow_id=flow_id,
            stage_id=stage_id, # This might be pipeline stage if applicable
            master_stage_id=master_stage_id, # This is for MasterExecutionPlan stage_id
            agent_id=agent_id,
            event_types=enum_event_types_filter,
            limit=limit,
            sort_desc=sort_desc_for_store
        )
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}", exc_info=True)
        click.echo(f"Error retrieving metrics: {e}", err=True)
        sys.exit(1)

    if not events:
        click.echo("No metric events found matching the criteria.")
        return

    if output_format == "json":
        click.echo(py_json.dumps([event.model_dump() for event in events], indent=2))
    else: 
        headers = ["Timestamp", "Type", "RunID", "FlowID", "MStageID", "AgentID", "DataSummary"]
        click.echo(" | ".join(headers))
        click.echo("-" * (sum(len(h) for h in headers) + (len(headers) -1) * 3))

        for event in events:
            data_summary = py_json.dumps(event.data) if event.data else "-"
            if len(data_summary) > 50:
                data_summary = data_summary[:47] + "..."
            
            # Try to get master_stage_id from data if available, else use event.stage_id
            # This depends on how MetricEvent populates stage_id vs master_stage_id
            # For now, assume event.stage_id is the one we want if master_stage_id is not primary on event
            display_stage_id = event.master_stage_id if event.master_stage_id else (event.stage_id or "-")

            row = [
                event.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                event.event_type.value,
                event.run_id or "-",
                event.flow_id or "-",
                display_stage_id, # Use the derived display_stage_id
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
        events = store.get_events(run_id=run_id, sort_desc=False) 
    except Exception as e:
        logger.error(f"Failed to retrieve metrics for run {run_id}: {e}", exc_info=True)
        click.echo(f"Error retrieving metrics for run {run_id}: {e}", err=True)
        sys.exit(1)

    if not events:
        click.echo(f"No metric events found for run_id: {run_id}")
        return

    summary_data: Dict[str, Any] = {
        "run_id": run_id,
        "flow_id": None,
        "start_time": None,
        "end_time": None,
        "duration_seconds": None,
        "status": "INCOMPLETE", 
        "total_master_stages_encountered": 0,
        "master_stages_completed_success": 0,
        "master_stages_completed_failure": 0,
        "errors_reported": []
    }

    flow_start_event: Optional[MetricEvent] = None
    flow_end_event: Optional[MetricEvent] = None
    master_stage_ids_encountered = set()

    for event in events:
        if summary_data["flow_id"] is None and event.flow_id:
            summary_data["flow_id"] = event.flow_id

        current_master_stage_id = event.master_stage_id or event.data.get("master_stage_id") # Check data too

        if event.event_type == MetricEventType.FLOW_START:
            if flow_start_event is None: 
                flow_start_event = event
                summary_data["start_time"] = event.timestamp.isoformat()
        
        elif event.event_type == MetricEventType.FLOW_END:
            flow_end_event = event 
            summary_data["end_time"] = event.timestamp.isoformat()
            if "final_status" in event.data: # Assuming FLOW_END data has this
                summary_data["status"] = event.data["final_status"]
            if "total_duration_seconds" in event.data:
                 summary_data["duration_seconds"] = event.data["total_duration_seconds"]

        elif event.event_type == MetricEventType.MASTER_STAGE_START and current_master_stage_id:
            master_stage_ids_encountered.add(current_master_stage_id)
        
        elif event.event_type == MetricEventType.MASTER_STAGE_END and current_master_stage_id:
            master_stage_ids_encountered.add(current_master_stage_id) # Ensure it's counted if only end event seen
            stage_status = event.data.get("status")
            if stage_status == StageStatus.COMPLETED_SUCCESS.value: # Compare with enum value
                summary_data["master_stages_completed_success"] += 1
            elif stage_status == StageStatus.COMPLETED_FAILURE.value:
                summary_data["master_stages_completed_failure"] += 1
                error_msg = event.data.get("error_details", event.data.get("error_message", "Unknown error"))
                summary_data["errors_reported"].append(f"Master Stage {current_master_stage_id} failed: {error_msg}")
        
        elif event.event_type == MetricEventType.ORCHESTRATOR_ERROR: # Generic orchestrator error
             err_msg = event.data.get("message", "Unknown orchestrator error")
             summary_data["errors_reported"].append(f"Orchestrator error: {err_msg}")
             if summary_data["status"] == "INCOMPLETE": # If an error occurs, mark flow as failed unless already ended
                 summary_data["status"] = StageStatus.COMPLETED_FAILURE.value # Generic failure status for flow


    summary_data["total_master_stages_encountered"] = len(master_stage_ids_encountered)

    if summary_data["duration_seconds"] is None and flow_start_event and flow_end_event:
        duration = flow_end_event.timestamp - flow_start_event.timestamp
        summary_data["duration_seconds"] = duration.total_seconds()
    elif summary_data["duration_seconds"] is None and flow_start_event and summary_data["status"] != "INCOMPLETE":
        last_event_ts = events[-1].timestamp
        duration = last_event_ts - flow_start_event.timestamp
        summary_data["duration_seconds"] = duration.total_seconds()

    # Refine overall status if not explicitly set by FLOW_END and no other errors indicated failure
    if summary_data["status"] == "INCOMPLETE":
        if flow_start_event and not flow_end_event:
            # Check if there's a PAUSED event
            paused_event = next((e for e in reversed(events) if e.event_type == MetricEventType.FLOW_PAUSED), None)
            if paused_event:
                summary_data["status"] = FlowPauseStatus.PAUSED_FOR_REVIEW.value # Example, map to actual pause status if available
            else:
                summary_data["status"] = "RUNNING_OR_CRASHED"
        elif flow_end_event is None and summary_data["errors_reported"]:
            summary_data["status"] = StageStatus.COMPLETED_FAILURE.value # If errors but no clean FLOW_END
        elif flow_end_event is None: # No end, no errors, but also not explicitly running
             summary_data["status"] = "UNKNOWN_TERMINATION"


    click.echo(f"Summary for Run ID: {summary_data['run_id']}")
    click.echo(f"  Flow ID: {summary_data['flow_id'] or 'N/A'}")
    click.echo(f"  Overall Status: {summary_data['status']}")
    click.echo(f"  Start Time: {summary_data['start_time'] or 'N/A'}")
    click.echo(f"  End Time: {summary_data['end_time'] or 'N/A'}")
    click.echo(f"  Duration: {summary_data['duration_seconds']:.2f}s" if summary_data['duration_seconds'] is not None else "Duration: N/A")
    click.echo(f"  Total Master Stages Encountered: {summary_data['total_master_stages_encountered']}")
    click.echo(f"  Master Stages Succeeded: {summary_data['master_stages_completed_success']}")
    click.echo(f"  Master Stages Failed: {summary_data['master_stages_completed_failure']}")
    if summary_data["errors_reported"]:
        click.echo("  Reported Errors/Failures:")
        for err_item in summary_data["errors_reported"]:
            click.echo(f"    - {str(err_item)[:200]}{'...' if len(str(err_item)) > 200 else ''}")
    else:
        click.echo("  Reported Errors/Failures: None")

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
    default=None, 
    help="Directory to save the HTML report (defaults to project_dir/.chungoid/reports/)."
)
@click.option("--limit", type=int, default=1000, show_default=True, help="Max events to include in the report data.")
@click.pass_context
def metrics_report(
    ctx: click.Context,
    project_dir: Path,
    run_id: Optional[str],
    output_dir: Optional[Path],
    limit: int
) -> None:
    """Generate an HTML metrics report by calling an external script."""
    logger = logging.getLogger("chungoid.cli.metrics.report")
    project_path = project_dir.expanduser().resolve()

    try:
        cli_dir = Path(__file__).resolve().parent # chungoid/
        scripts_dir = cli_dir.parent.parent / "scripts" # src/scripts/ -> WRONG if cli.py is in src/chungoid
        # Correct path if cli.py is in chungoid-core/src/chungoid/
        # and scripts are in chungoid-core/scripts/
        correct_scripts_dir = cli_dir.parent.parent / "scripts" # This assumes src/chungoid -> src -> scripts
        
        # More robust: find root of the 'chungoid-core' package structure if possible
        # This is still a bit fragile. Best would be if script path was configurable or installed.
        # Assuming `chungoid-core/scripts/generate_metrics_report.py` relative to `chungoid-core/src/chungoid/cli.py`
        # Path(__file__).resolve() -> /path/to/chungoid-core/src/chungoid/cli.py
        # .parent -> /path/to/chungoid-core/src/chungoid
        # .parent -> /path/to/chungoid-core/src
        # .parent -> /path/to/chungoid-core
        # then /scripts
        package_root_dir = Path(__file__).resolve().parent.parent.parent
        script_path = package_root_dir / "scripts" / "generate_metrics_report.py"

        if not script_path.exists():
             # Fallback for environments where the script might be elsewhere relative to execution
             logger.warning(f"Script not found at primary path {script_path}, trying alternative relative to project.")
             script_path = project_path / "scripts" / "generate_metrics_report.py" # Common for dev setup
             if not script_path.exists():
                click.echo(f"Error: Report generation script not found at {script_path} or primary path. Searched relative to CLI and project.", err=True)
                sys.exit(1)
        
    except Exception as e_script_path:
        logger.error(f"Error determining report script path: {e_script_path}")
        click.echo(f"Error: Could not determine path to report generation script: {e_script_path}", err=True)
        sys.exit(1)


    cmd = [sys.executable, str(script_path), "--project-dir", str(project_path)]
    if run_id:
        cmd.extend(["--run-id", run_id])
    if output_dir:
        cmd.extend(["--output-dir", str(output_dir.resolve())]) # Resolve output_dir
    cmd.extend(["--limit", str(limit)])

    try:
        logger.info(f"Executing report generation script: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
        if result.stdout:
            click.echo(result.stdout)
        if result.stderr: # Always print stderr if present
            click.echo(f"Report script stderr:\n{result.stderr}", err=True)
        
        if result.returncode != 0:
            click.echo(f"Error: Report generation script failed with exit code {result.returncode}.", err=True)
            # sys.exit(result.returncode) # Optional: exit CLI if script fails

    except FileNotFoundError:
        click.echo(f"Error: Python interpreter '{sys.executable}' not found or script path '{script_path}' incorrect.", err=True)
        sys.exit(1)
    except Exception as e: # General catch-all for other subprocess or runtime errors
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
        # Pass project_dir if available from a higher-level group context, or default
        # For a standalone command, it might need its own --project-dir option or assume cwd
        current_project_dir_str = ctx.obj.get("project_dir", ".") # Example if project_dir was passed higher up
        
        # More robust: try to determine project_dir from where CLI is run if not in ctx.obj
        # This assumes show-config might be run from within a project.
        # If it needs to be flexible, it should take its own --project-dir.
        # For now, let's assume get_config can find it if run from project root.
        # Or, we can make it take an optional project_dir argument.
        
        # Simplest for now: get_config without args tries to find config relative to CWD or uses default.
        config = get_config() # No argument, relies on get_config's internal logic or default.

        if not config or not config.get("project_root"): # Check if essential key is missing
            click.secho("Error: Project configuration could not be loaded or is incomplete.", fg="red")
            click.secho("Ensure you are in a Chungoid project directory, or that config is valid.", fg="red")
            ctx.exit(1)
            
        config_file_location = config.get('config_file_loaded', 'Default values (no file loaded)')
        click.secho(f"Current Project Configuration (from {config_file_location}):", fg="cyan", bold=True)
        
        # Print key config values, checking for their existence for robustness
        click.echo(f"  project_root: {config.get('project_root', 'N/A')}")
        click.echo(f"  dot_chungoid_path: {config.get('dot_chungoid_path', 'N/A')}")
        click.echo(f"  state_manager_db_path: {config.get('state_manager_db_path', 'N/A')}") # Was state_manager_path
        click.echo(f"  master_flows_dir: {config.get('master_flows_dir', 'N/A')}")
        click.echo(f"  host_system_info: {config.get('host_system_info', 'N/A')}")
        click.echo(f"  log_level: {config.get('log_level', 'N/A')}")
        
        # Display chromadb config if present
        chromadb_config = config.get("chromadb")
        if chromadb_config and isinstance(chromadb_config, dict):
            click.echo("  ChromaDB Configuration:")
            click.echo(f"    Mode: {chromadb_config.get('mode', 'N/A')}")
            click.echo(f"    Host: {chromadb_config.get('host', 'N/A')}")
            click.echo(f"    Port: {chromadb_config.get('port', 'N/A')}")
            click.echo(f"    Persist Path: {chromadb_config.get('persist_path', 'N/A')}")
        else:
            click.echo("  ChromaDB Configuration: Not specified or invalid format")

    except Exception as e:
        click.secho(f"An unexpected error occurred while retrieving configuration: {e}", fg="red")
        logger.error(f"Error in show_config command: {e}", exc_info=True) # Keep logger for internal trace
        ctx.exit(1)

# Register the command under 'utils' group (only once)
# Ensure this is done only once, typically after the function definition
if 'show-config' not in [c.name for c in utils.commands.values()]:
    utils.add_command(show_config)

@utils.command(name="show-modules")
@click.pass_context
def show_modules(ctx: click.Context):
    """(Dev utility) Show loaded Chungoid modules and their paths."""
    # ... (rest of the function)

# Ensure this file can be run as a script for Click CLI discovery
# if __name__ == "__main__":
#    cli() # This makes it runnable but is not needed for Click entry point

# BUG_WORKAROUND: code_to_integrate was a literal context path.
utils.add_command(show_config)