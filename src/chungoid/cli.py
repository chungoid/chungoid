from __future__ import annotations
import json
from chungoid.utils.config_loader import get_config, ConfigError, load_config
import click
import uuid # ADDED FOR PROJECT ID GENERATION

from dotenv import load_dotenv # ADDED
load_dotenv() # ADDED: Load .env file at the start

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
from typing import Any, Optional, Dict, List, cast, Union, Type

import click
import rich.traceback
from rich.logging import RichHandler

import chungoid
from chungoid.constants import (DEFAULT_MASTER_FLOWS_DIR, DEFAULT_SERVER_STAGES_DIR, MIN_PYTHON_VERSION,
                              PROJECT_CHUNGOID_DIR, STATE_FILE_NAME)
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus, HumanReviewDecision, OnFailureAction
from chungoid.core_utils import get_project_root_or_raise, init_project_structure
from chungoid.schemas.master_flow import MasterExecutionPlan
from chungoid.schemas.metrics import MetricEventType
from chungoid.schemas.flows import PausedRunDetails
from chungoid.runtime.agents.system_master_planner_reviewer_agent import get_agent_card_static as get_reviewer_card
from chungoid.runtime.orchestrator import AsyncOrchestrator
from chungoid.schemas.metrics import MetricEvent
from chungoid.utils.agent_registry import AgentRegistry
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.llm_provider import MockLLMProvider, OpenAILLMProvider, LLMManager # Keep MockLLMProvider for --use-mock-llm-provider flag

# For Agent Cards (used in agent_registry.add())
from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card

from chungoid.runtime.agents.system_master_planner_agent import get_agent_card_static as get_master_planner_agent_card
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent

# For Agent Classes (used in the fallback_agents_map and direct instantiation)
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent

# Real Agent Imports (ensure all necessary are here)
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1 as CodeGeneratorAgent
from chungoid.runtime.agents.core_test_generator_agent import CoreTestGeneratorAgent_v1 as TestGeneratorAgent
from chungoid.runtime.agents.smart_code_integration_agent import SmartCodeIntegrationAgent_v1
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1
from chungoid.runtime.agents.system_test_runner_agent import SystemTestRunnerAgent_v1 as SystemTestRunnerAgent
from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1
from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

import subprocess
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
from chungoid.utils.agent_resolver import RegistryAgentProvider, AgentCallable, AgentFallbackItem # MODIFIED
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

# New imports for MasterPlannerInput
from chungoid.schemas.agent_master_planner import MasterPlannerInput # <<< ADD THIS IMPORT

# Import for CoreCodeGeneratorAgent
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1 as CodeGeneratorAgent
# Import for CoreTestGeneratorAgent
from chungoid.runtime.agents.core_test_generator_agent import CoreTestGeneratorAgent_v1 as TestGeneratorAgent
# Import for CodeIntegrationAgent - UPDATED
from chungoid.runtime.agents.smart_code_integration_agent import SmartCodeIntegrationAgent_v1
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1

# New imports for MockTestGenerationAgentV1
# from chungoid.runtime.agents.mocks.mock_test_generation_agent import MockTestGenerationAgentV1, get_agent_card_static as get_mock_test_generation_agent_v1_card

# Import the new system_test_runner_agent
# from chungoid.runtime.agents import system_test_runner_agent # ADDED # OLD IMPORT
from chungoid.runtime.agents.system_test_runner_agent import SystemTestRunnerAgent_v1 as SystemTestRunnerAgent # NEW IMPORT
from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1 # ADDED IMPORT

# Ensure AgentID type is available if used for keys, though strings are fine for dict keys.
from chungoid.schemas.common import AgentID # CORRECTED IMPORT
# from chungoid.runtime.agents.base import AgentBase # For type hinting if needed # REMOVED
from chungoid.runtime.agents.agent_base import BaseAgent # For type hinting # CORRECTED IMPORT

# --- ADDED IMPORTS FOR MOCK SETUP AGENT ---
# from chungoid.runtime.agents.mocks.testing_mock_agents import (
#     MockSetupAgentV1,
#     MockFailPointAgentV1,
#     get_mock_agent_fallback_map, # ADDED THIS IMPORT
#     MockSystemInterventionAgent, # ADDED THIS IMPORT
#     # MockNoOpAgent # REMOVED THIS IMPORT - Will import directly
# )
# --- END ADDED IMPORTS ---

# ADDED: Import MockNoOpAgent directly from its new file
# from chungoid.runtime.agents.mocks.mock_noop_agent import MockNoOpAgent

# ADDED: Imports for production system agents and dependencies
from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1
from chungoid.agents.autonomous_engine import get_autonomous_engine_agent_fallback_map

# For dependencies of RegistryAgentProvider and agents
from chungoid.utils.llm_provider import LLMProvider # Already imported but ensure it's available
# from chungoid.utils.prompt_manager import PromptManager # Already imported but ensure it's available

# ADDED: Import ProjectChromaManagerAgent_v1 at the top for the map
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

# ADDED: Import for NoOpAgent_v1
from chungoid.runtime.agents.system_agents.noop_agent import NoOpAgent_v1 

# --- Production System Agents Fallback Map ---
# This map defines the primary fallback for system agents.
# It uses the actual agent classes.
PRODUCTION_SYSTEM_AGENTS_MAP: Dict[AgentID, Union[Type[BaseAgent], BaseAgent]] = {
    MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
    MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent,
    CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # Alias for CoreCodeGeneratorAgent_v1
    TestGeneratorAgent.AGENT_ID: TestGeneratorAgent,   # Alias for CoreTestGeneratorAgent_v1
    SmartCodeIntegrationAgent_v1.AGENT_ID: SmartCodeIntegrationAgent_v1,
    "SmartCodeGeneratorAgent_v1": SmartCodeIntegrationAgent_v1, # ADDED ALIAS
    SystemTestRunnerAgent.AGENT_ID: SystemTestRunnerAgent, # This is SystemTestRunnerAgent_v1
    SystemFileSystemAgent_v1.AGENT_ID: SystemFileSystemAgent_v1, # Ensure this uses the class name
    SystemRequirementsGatheringAgent_v1.AGENT_ID: SystemRequirementsGatheringAgent_v1,
    ArchitectAgent_v1.AGENT_ID: ArchitectAgent_v1,
    NoOpAgent_v1.AGENT_ID: NoOpAgent_v1, # ADDED NoOpAgent_v1
    # ProjectChromaManagerAgent_v1 is typically instantiated directly in CLI commands
    # and added to the fallback map as an instance, not as a class here.
    # AutonomousEngineAgent_v1 is handled by get_autonomous_engine_agent_fallback_map
}
# --- End Production System Agents Fallback Map ---

# --- Autonomous Engine Agents Fallback Map ---
def get_autonomous_engine_agent_fallback_map() -> Dict[AgentID, Union[Type[BaseAgent], BaseAgent]]:
    """Returns a dictionary of core autonomous engine agents for fallback."""
    # This map should only contain system-critical, non-mock agents
    # that are part of the autonomous engine's core capabilities.
    return {
        # AutonomousEngineAgent_v1.AGENT_ID: AutonomousEngineAgent_v1, # COMMENTED OUT
        # Other autonomous engine agents can be added here as classes.
        # ProjectChromaManagerAgent_v1 is often instantiated specifically with project context
        # in the CLI command logic and added to the final_fallback_map there.
    }
# --- End Autonomous Engine Agents Fallback Map ---

# Assuming StatusFileError might be a custom exception, if not defined elsewhere, it might need to be.
# For now, let's assume it's imported or defined if critical. If it's from a known module, add import.
# Example: from chungoid.utils.exceptions import StatusFileError (if it exists there)
# If it was a typo and meant something else, that would need correction.
# For now, proceeding as if it will be resolved by existing imports or is not critical path for this edit.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] # Corrected this line


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

    # MODIFIED: Define server_stages_dir_path_str_flow consistently and early
    try:
        cli_file_path_stages = Path(__file__).resolve()
        script_dir_stages = cli_file_path_stages.parent.resolve() # .../src/chungoid
        core_root_dir_stages = script_dir_stages.parent.parent    # .../chungoid-core
        server_stages_dir_path_str_flow = str(core_root_dir_stages / "server_prompts" / "stages")

        if project_config.get("server_stages_dir_override"):
            server_stages_dir_path_str_flow = project_config["server_stages_dir_override"]
        elif not Path(server_stages_dir_path_str_flow).is_dir():
            logger.warning(f"Flow Run Default server_stages_dir {server_stages_dir_path_str_flow} not found. Trying project-relative.")
            server_stages_dir_path_str_flow = str(project_path / "server_prompts" / "stages")
            if not Path(server_stages_dir_path_str_flow).is_dir():
                logger.warning(f"Flow Run Project-relative server_stages_dir also not found. Using constant DEFAULT_SERVER_STAGES_DIR: {DEFAULT_SERVER_STAGES_DIR}")
                server_stages_dir_path_str_flow = str(DEFAULT_SERVER_STAGES_DIR)
    except Exception as e_stages_dir_early_flow:
        logger.error(f"Flow Run: Error determining server_stages_dir_path_str early: {e_stages_dir_early_flow}. Defaulting.")
        server_stages_dir_path_str_flow = str(DEFAULT_SERVER_STAGES_DIR)
    logger.info(f"Flow Run: Determined server_stages_dir for StateManager/PCMA context: {server_stages_dir_path_str_flow}")

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
    # agent_registry.add(get_mock_system_intervention_agent_card(), overwrite=True) # REMOVE
    # agent_registry.add(get_mock_code_generator_agent_card(), overwrite=True) # REMOVE
    # agent_registry.add(get_mock_test_generator_agent_card(), overwrite=True) # REMOVE
    # agent_registry.add(get_mock_system_requirements_gathering_agent_card(), overwrite=True) # REMOVE
    agent_registry.add(CodeGeneratorAgent.get_agent_card_static(), overwrite=True)
    agent_registry.add(TestGeneratorAgent.get_agent_card_static(), overwrite=True)
    agent_registry.add(SmartCodeIntegrationAgent_v1.get_agent_card_static(), overwrite=True)
    agent_registry.add(SystemTestRunnerAgent.get_agent_card_static(), overwrite=True)

    # For simplicity in RegistryAgentProvider, we provide one merged map. Let's ensure system agents are there.
    
    # Explicitly add core system agents to the fallback map if not already covered.
    # These are agents that provide core functionality and might have specific mock behaviors for MVPs.
    # core_system_agents = { # This local variable is replaced by production_system_agents_map
    #     MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
    #     MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent,
    #     CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # MVP uses its mocked output
    #     TestGeneratorAgent.AGENT_ID: TestGeneratorAgent, # MVP uses its mocked output
    #     SmartCodeIntegrationAgent_v1.AGENT_ID: SmartCodeIntegrationAgent_v1, # Handles actual file edits
    #     SystemTestRunnerAgent.AGENT_ID: SystemTestRunnerAgent # Use the AGENT_ID and invoke_async function for functional agents
    #     # Add other essential system agents here if their local Python class should be directly invokable via fallback
    # }
    
    # Start with mock agents from testing_mock_agents.py
    # final_fallback_map: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map() # MODIFIED LOGIC
    # Add/override with core system agents
    # final_fallback_map.update(core_system_agents) # MODIFIED LOGIC
    # Add/override with autonomous engine agents
    # final_fallback_map.update(get_autonomous_engine_agent_fallback_map()) # MODIFIED LOGIC

    # MODIFIED: Construct final_fallback_map with new precedence
    final_fallback_map: Dict[AgentID, AgentFallbackItem] = PRODUCTION_SYSTEM_AGENTS_MAP.copy()
    final_fallback_map.update(get_autonomous_engine_agent_fallback_map())
    
    # MODIFIED: Ensure all necessary dependencies are prepared for RegistryAgentProvider
    # LLMProvider setup
    # For flow_run, we need a robust way to get an LLM provider if non-mock agents requiring it are used.
    # This setup mirrors the logic in the `build` command for consistency.
    llm_provider_instance_for_flow_run: Optional[LLMProvider] = None
    # Assuming project_config is loaded. If not, this logic needs to be adjusted or llm_provider remains None.
    if project_config.get("llm_config", {}).get("use_mock_llm_provider", False): # Check a hypothetical config flag or CLI option
        logger.info("Flow Run: Using MockLLMProvider based on config/flag.")
        llm_provider_instance_for_flow_run = MockLLMProvider()
    else:
        openai_api_key_flow_run = os.getenv("OPENAI_API_KEY")
        if openai_api_key_flow_run:
            default_openai_model_flow_run = project_config.get("llm_config", {}).get("default_openai_model", "gpt-3.5-turbo")
            logger.info(f"Flow Run: Initializing OpenAILLMProvider with default model: {default_openai_model_flow_run}")
            llm_provider_instance_for_flow_run = OpenAILLMProvider(api_key=openai_api_key_flow_run, default_model=default_openai_model_flow_run)
        else:
            logger.warning("Flow Run: OPENAI_API_KEY not set. LLM-dependent agents might fail if not mocked.")
            # llm_provider_instance_for_flow_run remains None

    # PromptManager setup
    # Use the server_stages_dir_path_str_flow which is now defined early
    prompt_manager_base_dir = Path(server_stages_dir_path_str_flow).parent if Path(server_stages_dir_path_str_flow).name == "stages" else Path(server_stages_dir_path_str_flow)
    prompt_manager_instance = PromptManager(prompt_directory_paths=[prompt_manager_base_dir])
    logger.info(f"Flow Run: PromptManager initialized with directory: {prompt_manager_base_dir}")

    # ProjectChromaManager setup
    project_id_for_pcma = project_config.get('project_id')
    if not project_id_for_pcma:
        # Use server_stages_dir_path_str_flow for StateManager context
        temp_sm_for_pcma_flow_run = StateManager(target_directory=project_path, server_stages_dir=server_stages_dir_path_str_flow)
        try:
            project_id_for_pcma = temp_sm_for_pcma_flow_run.get_project_state().project_id
        except Exception: 
             logger.warning("Flow Run: Project ID for PCMA could not be retrieved from config or state. PCMA might not be available.")

    project_chroma_manager_instance_for_flow_run: Optional[ProjectChromaManagerAgent_v1] = None
    if project_id_for_pcma:
        try:
            project_chroma_manager_instance_for_flow_run = ProjectChromaManagerAgent_v1(
                project_root_workspace_path=str(project_path),
                project_id=project_id_for_pcma
            )
            logger.info(f"Flow Run: Instantiated ProjectChromaManagerAgent_v1 with project_id: {project_id_for_pcma}")
        except Exception as e_pcma_flow:
            logger.error(f"Flow Run: Failed to instantiate ProjectChromaManagerAgent_v1: {e_pcma_flow}", exc_info=True)
    
    agent_provider = RegistryAgentProvider(
        registry=agent_registry,
        fallback=final_fallback_map,
        llm_provider=llm_provider_instance_for_flow_run,
        prompt_manager=prompt_manager_instance,
        project_chroma_manager=project_chroma_manager_instance_for_flow_run
    )
    
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
    
    # --- Initialize Project State --- 
    try:
        # Generate a project ID if one isn't already in the config or determined
        # For a new build, we usually generate one.
        new_project_id = project_config.get('project_id') or f"proj_{uuid.uuid4().hex[:12]}"
        
        initialized_project_state = state_manager.initialize_project(
            project_id=new_project_id, # ADDED project_id
            project_name=project_path.name, 
            initial_user_goal_summary=goal
        )
        current_project_id = initialized_project_state.project_id # Use the ID from the initialized state
        logger.info(f"Project initialized/loaded with ID: {current_project_id} and state file written/verified.")
        
        # Ensure config object has project_id and project_root_path for AsyncOrchestrator
        project_config['project_id'] = current_project_id
        project_config['project_root_path'] = str(project_path) # project_root is already a Path
        project_config['project_root'] = str(project_path) # Also ensure 'project_root' key for other potential uses

    except Exception as e_init_proj:
        logger.error(f"Failed to initialize project state: {e_init_proj}", exc_info=True)
        click.echo(f"Error initializing project state: {e_init_proj}", err=True)
        sys.exit(1)
    # --- End Initialize Project State ---

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

        # --- ADDED: Setup for AgentProvider dependencies in flow_resume --- 
        # Determine server_stages_dir_path_str_flow for resume context
        try:
            cli_file_path_stages_resume = Path(__file__).resolve()
            script_dir_stages_resume = cli_file_path_stages_resume.parent.resolve()
            core_root_dir_stages_resume = script_dir_stages_resume.parent.parent
            server_stages_dir_path_str_resume_flow = str(core_root_dir_stages_resume / "server_prompts" / "stages")

            if resumed_project_config.get("server_stages_dir_override"):
                server_stages_dir_path_str_resume_flow = resumed_project_config["server_stages_dir_override"]
            elif not Path(server_stages_dir_path_str_resume_flow).is_dir():
                server_stages_dir_path_str_resume_flow = str(project_path / "server_prompts" / "stages")
                if not Path(server_stages_dir_path_str_resume_flow).is_dir():
                    server_stages_dir_path_str_resume_flow = str(DEFAULT_SERVER_STAGES_DIR)
        except Exception as e_stages_dir_resume:
            logger.error(f"Flow Resume: Error determining server_stages_dir_path_str: {e_stages_dir_resume}. Defaulting.")
            server_stages_dir_path_str_resume_flow = str(DEFAULT_SERVER_STAGES_DIR)
        logger.info(f"Flow Resume: Determined server_stages_dir for PrompManager/PCMA context: {server_stages_dir_path_str_resume_flow}")

        # LLMProvider setup for resume
        llm_provider_instance_for_resume: Optional[LLMProvider] = None
        llm_manager_for_resume: Optional[LLMManager] = None # To manage the lifecycle of the provider
        if resumed_project_config.get("llm_config", {}).get("use_mock_llm_provider", False):
            logger.info("Flow Resume: Using MockLLMProvider based on config/flag.")
            llm_provider_instance_for_resume = MockLLMProvider()
        else:
            openai_api_key_resume = os.getenv("OPENAI_API_KEY")
            if openai_api_key_resume:
                default_openai_model_resume = resumed_project_config.get("llm_config", {}).get("default_openai_model", "gpt-3.5-turbo")
                logger.info(f"Flow Resume: Initializing OpenAILLMProvider with default model: {default_openai_model_resume}")
                llm_provider_instance_for_resume = OpenAILLMProvider(api_key=openai_api_key_resume, default_model=default_openai_model_resume)
            else:
                logger.warning("Flow Resume: OPENAI_API_KEY not set. LLM-dependent agents might fail if not mocked.")
        
        if llm_provider_instance_for_resume:
            # PromptManager setup for resume (needed for LLMManager and potentially agents)
            prompt_manager_base_dir_resume = Path(server_stages_dir_path_str_resume_flow).parent if Path(server_stages_dir_path_str_resume_flow).name == "stages" else Path(server_stages_dir_path_str_resume_flow)
            prompt_manager_instance_for_resume = PromptManager(prompt_directory_paths=[prompt_manager_base_dir_resume])
            logger.info(f"Flow Resume: PromptManager initialized with directory: {prompt_manager_base_dir_resume}")
            llm_manager_for_resume = LLMManager(llm_provider_instance=llm_provider_instance_for_resume, prompt_manager=prompt_manager_instance_for_resume)
        else:
            prompt_manager_instance_for_resume = None # No LLM provider, so no LLMManager, and PromptManager might not be strictly needed by provider
            logger.info("Flow Resume: No LLM provider, so LLMManager not created. PromptManager also may not be initialized if not otherwise needed.")

        # ProjectChromaManager setup for resume
        project_id_for_pcma_resume = resumed_project_config.get('project_id')
        if not project_id_for_pcma_resume:
            temp_sm_for_pcma_resume = StateManager(target_directory=project_path, server_stages_dir=server_stages_dir_path_str_resume_flow)
            try:
                project_id_for_pcma_resume = temp_sm_for_pcma_resume.get_project_state().project_id
            except Exception: 
                 logger.warning("Flow Resume: Project ID for PCMA could not be retrieved from config or state. PCMA might not be available.")

        project_chroma_manager_instance_for_resume: Optional[ProjectChromaManagerAgent_v1] = None
        if project_id_for_pcma_resume:
            try:
                project_chroma_manager_instance_for_resume = ProjectChromaManagerAgent_v1(
                    project_root_workspace_path=str(project_path),
                    project_id=project_id_for_pcma_resume
                )
                logger.info(f"Flow Resume: Instantiated ProjectChromaManagerAgent_v1 with project_id: {project_id_for_pcma_resume}")
            except Exception as e_pcma_resume:
                logger.error(f"Flow Resume: Failed to instantiate ProjectChromaManagerAgent_v1: {e_pcma_resume}", exc_info=True)
        # --- END ADDED: Setup for AgentProvider dependencies --- 

        registry_project_root = Path(resumed_project_config["project_root_dir"])
        registry_chroma_mode = resumed_project_config.get("chromadb", {}).get("mode", "persistent")

        agent_registry = AgentRegistry(project_root=registry_project_root, chroma_mode=registry_chroma_mode)
        # Ensure agents are registered as in flow_run, or that registry is persistent/shared
        agent_registry.add(get_master_planner_reviewer_agent_card(), overwrite=True)
        agent_registry.add(get_master_planner_agent_card(), overwrite=True)
        # ... (add other necessary agents as in flow_run)
        # agent_registry.add(get_mock_system_intervention_agent_card(), overwrite=True)
        # agent_registry.add(get_mock_code_generator_agent_card(), overwrite=True)
        # agent_registry.add(get_mock_test_generation_agent_v1_card(), overwrite=True)
        # agent_registry.add(get_mock_system_requirements_gathering_agent_card(), overwrite=True)
        agent_registry.add(CodeGeneratorAgent.get_agent_card_static(), overwrite=True)
        agent_registry.add(TestGeneratorAgent.get_agent_card_static(), overwrite=True)
        agent_registry.add(SmartCodeIntegrationAgent_v1.get_agent_card_static(), overwrite=True)
        # ADD SystemTestRunnerAgent registration here for consistency if it's expected in flow_resume's context
        agent_registry.add(SystemTestRunnerAgent.get_agent_card_static(), overwrite=True)


        # fallback_agents_map_resume: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map() # REMOVE
        # Add/override with core system agents
        # core_system_agents_resume = { # REMOVE
        #     MasterPlannerAgent.AGENT_ID: MasterPlannerAgent, # REMOVE
        #     MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent, # REMOVE
        #     # Add other essential system agents here if needed # REMOVE
        # } # REMOVE
        # Merge, ensuring core system agents take precedence if IDs overlap (though unlikely for mocks)
        # Or, decide on a clear priority. For now, let mock map be primary, then add system agents.
        # A more robust way would be to have separate maps and the resolver check them in order.
        # For simplicity in RegistryAgentProvider, we provide one merged map. Let's ensure system agents are there.
        
        # Start with mock agents
        # final_fallback_map_resume: Dict[AgentID, AgentCallable] = get_mock_agent_fallback_map() # REMOVE
        # Add/override with core system agents
        # final_fallback_map_resume.update(core_system_agents_resume) # REMOVE

        # RECONSTRUCT final_fallback_map_resume similar to flow_run
        final_fallback_map_resume: Dict[AgentID, AgentFallbackItem] = PRODUCTION_SYSTEM_AGENTS_MAP.copy()
        final_fallback_map_resume.update(get_autonomous_engine_agent_fallback_map())
        # If flow_resume needs an instantiated ProjectChromaManagerAgent, it should be created and added here.
        # For now, assuming it's not needed or handled by the global maps for resume.
        # Consider if specific instances like a ProjectChromaManager for THIS run are needed.
        # If so, they'd be instantiated and added to final_fallback_map_resume here.
        
        agent_provider = RegistryAgentProvider(
            registry=agent_registry, 
            fallback=final_fallback_map_resume,
            # Pass llm_provider, prompt_manager, project_chroma_manager if RegistryAgentProvider might instantiate agents needing them
            # This requires setting up llm_provider_instance, prompt_manager_instance, etc., similar to flow_run
            # For brevity, assuming flow_resume might not always need to re-instantiate complex agents
            # requiring these, or that the global ones are sufficient. This might need revisiting.
            llm_provider=llm_provider_instance_for_resume, # Use the one defined in flow_resume context
            prompt_manager=prompt_manager_instance_for_resume, # Use the one defined
            project_chroma_manager=project_chroma_manager_instance_for_resume # Use the one defined
            )

        # The main try/except/finally for do_resume actions
        try:
            cli_file_path = Path(__file__).resolve()
            server_stages_dir_path_str_resume = str(cli_file_path.parent.parent.parent / "server_prompts" / "stages")
            if not Path(server_stages_dir_path_str_resume).is_dir():
                server_stages_dir_path_str_resume = str(project_path / "server_prompts" / "stages")
                if not Path(server_stages_dir_path_str_resume).is_dir():
                    server_stages_dir_path_str_resume = resumed_project_config.get("server_stages_dir_fallback", DEFAULT_SERVER_STAGES_DIR)
            else:
                click.echo(f"Flow run '{run_id}' processed with action '{action}'. Check logs and project status for outcome.")

        except Exception as e:
            # This catches errors from the block above (orchestrator call, state_manager, etc.)
            logger.error(f"An unexpected error occurred during active flow_resume operations: {e}", exc_info=True)
            click.echo(f"An unexpected error occurred during resume: {e}", err=True)
            # sys.exit(1) # Let the outer try/except handle sys.exit
            raise # Re-raise to be caught by the outer try/except that calls asyncio.run()
        finally:
            # Ensure LLM client (if any) is closed
            if llm_manager_for_resume and llm_manager_for_resume._llm_provider is not None: # Check llm_manager_for_resume
                 logger.info(f"Flow Resume: Attempting to close LLM provider client of type: {type(llm_manager_for_resume._llm_provider).__name__}")
                 await llm_manager_for_resume.close_client()

    # Outer try/except that calls asyncio.run()
    try:
        asyncio.run(do_resume())
        # Successful completion, exit code 0 is implicit
    except Exception:
         # Errors are logged within do_resume's try/except block and re-raised.
         # This ensures a non-zero exit code if do_resume itself had an unhandled exception or re-raised one.
         # click.echo("Flow resume failed. See logs for details.", err=True) # Error message already echoed in do_resume
         sys.exit(1) # Ensure non-zero exit code on any exception from do_resume


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

# --- NEW Project Group ---
@cli.group()
@click.pass_context
def project(ctx: click.Context) -> None: # noqa: D401
    """Commands for managing Chungoid autonomous projects and their lifecycles."""
    pass

@project.command(name="review")
@click.option(
    "--project-dir",
    "project_dir_opt",
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    show_default=True,
    help="Project directory containing the \'.chungoid\' subdirectory."
)
@click.option("--cycle-id", type=str, required=True, help="The ID of the cycle being reviewed.")
@click.option("--reviewer-id", type=str, required=True, help="Identifier for the human reviewer (e.g., username, email).")
@click.option("--comments", type=str, required=True, help="Overall comments and observations from the reviewer.")
@click.option(
    "--decision",
    type=click.Choice([d.value for d in HumanReviewDecision], case_sensitive=False),
    required=True,
    help="The decision for the next step after this review."
)
@click.option("--next-objective", type=str, default=None, help="Objective for the next cycle, if applicable (e.g., for REFINEMENT or MODIFIED_GOAL decisions).")
@click.option("--linked-feedback-doc-id", type=str, default=None, help="Optional ChromaDB document ID for more detailed feedback.")
@click.pass_context
def project_review(
    ctx: click.Context,
    project_dir_opt: Path,
    cycle_id: str,
    reviewer_id: str,
    comments: str,
    decision: str,
    next_objective: Optional[str],
    linked_feedback_doc_id: Optional[str],
):
    """Submits a human review for a completed autonomous cycle."""
    logger.info(f"Attempting to submit project review for cycle: {cycle_id} in project: {project_dir_opt.resolve()}")
    
    try:
        project_root = get_project_root_or_raise(project_dir_opt)
        # Ensure server_stages_dir is valid, even if not directly used by this command path,
        # StateManager init requires it.
        # Consider making server_stages_dir optional in StateManager or providing a default/dummy.
        # For now, use a potentially existing default or raise if not found.
        config = get_config(project_root)
        server_stages_dir_path = project_root / config.get("server_stages_dir", DEFAULT_SERVER_STAGES_DIR)
        if not server_stages_dir_path.is_dir():
             logger.warning(f"Server stages directory {server_stages_dir_path} not found. StateManager might have issues if it needs it.")
             # If StateManager strictly needs it, this command might fail here.
             # For review submission, StateManager primarily needs project_status.json.

        state_manager = StateManager(target_directory=str(project_root), server_stages_dir=str(server_stages_dir_path))

        review_record = HumanReviewRecord(
            reviewer_id=reviewer_id,
            review_timestamp_utc=datetime.now(timezone.utc),
            cycle_id_reviewed=cycle_id,
            comments_and_observations=comments,
            decision_for_next_step=HumanReviewDecision(decision), # Convert string back to enum
            next_cycle_objective_override=next_objective,
            linked_detailed_feedback_doc_id=linked_feedback_doc_id
            # review_id is auto-generated by Pydantic default_factory
        )

        state_manager.record_human_review(review_record)

        click.secho(f"Successfully recorded review for cycle {cycle_id}.", fg="green")
        click.echo(f"Project status updated. Decision: {decision}")
        if next_objective:
            click.echo(f"Next cycle objective override: {next_objective}")
        
        # Advise on next steps
        current_project_state = state_manager.get_project_state()
        click.echo(f"Current overall project status: {current_project_state.overall_status.value}")
        if current_project_state.overall_status == "AWAITING_NEXT_CYCLE_START":
            click.echo(f"To start the next cycle, you might use: chungoid flow run --project-dir \\\"{project_root}\\\" --goal \\\"<your_goal_or_use_override>\\\"") # Adjust command as needed
        elif current_project_state.overall_status == "PROJECT_PAUSED_BY_USER":
             click.echo("Project is now paused.")
        # Add more advice based on other statuses

    except FileNotFoundError:
        click.secho(f"Error: Project directory or .chungoid structure not found in {project_dir_opt.resolve()}", fg="red")
        sys.exit(1)
    except ConfigError as e:
        click.secho(f"Configuration error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red")
        logger.error("Unexpected error in project review CLI command:", exc_info=True)
        sys.exit(1)

# Import for ProjectChromaManagerAgent_v1, needed by the 'build' command
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
# CORRECTED IMPORT for HumanReviewRecord
from chungoid.schemas.project_status_schema import HumanReviewRecord 

# NEW BUILD COMMAND
@cli.command("build", help="Build a project from a goal file.")
@click.option("--goal-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help="Path to the file containing the user goal.")
@click.option("--project-dir", "project_dir_opt", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=".", help="Target project directory. Defaults to current directory. Will be created if it doesn't exist.")
@click.option("--run-id", "run_id_override_opt", type=str, default=None, help="Specify a custom run ID for this execution.")
@click.option("--initial-context", type=str, default=None, help="JSON string containing initial context variables for the build.")
@click.option("--tags", type=str, default=None, help="Comma-separated tags for this build (e.g., 'dev,release').")
@click.option("--use-mock-llm-provider/--no-use-mock-llm-provider", default=False, help="Use mock LLM provider instead of a real one. Defaults to False (uses real LLM).") # Keep this flag
@click.pass_context
def build_from_goal_file(ctx: click.Context, goal_file: Path, project_dir_opt: Path, run_id_override_opt: Optional[str], initial_context: Optional[str], tags: Optional[str], use_mock_llm_provider: bool):
    """Initiates a project build from a user goal specified in a file."""
    logger.info(f"Starting build from goal file: {goal_file} for project directory: {project_dir_opt} with use_mock_llm_provider={use_mock_llm_provider}")
    log_level = ctx.obj.get("log_level", "INFO") # Get log_level from context

    # Ensure project directory exists
    project_dir_opt.mkdir(parents=True, exist_ok=True)
    logger.info(f"Ensured project directory exists: {project_dir_opt.resolve()}")

    try:
        user_goal = goal_file.read_text().strip()
        if not user_goal:
            logger.error("Goal file is empty.")
            click.echo("Error: Goal file is empty.", err=True)
            raise click.Abort()
        logger.info(f"User goal loaded: '{user_goal[:100]}...'")
    except Exception as e:
        logger.error(f"Error reading goal file {goal_file}: {e}")
        click.echo(f"Error: Could not read goal file {goal_file}. Reason: {e}", err=True)
        raise click.Abort()

    # Convert project_dir_opt to absolute path for consistency
    abs_project_dir = project_dir_opt.resolve()
    logger.info(f"Absolute project directory: {abs_project_dir}")

    # Initial context parsing
    parsed_initial_context = {}
    if initial_context:
        try:
            parsed_initial_context = py_json.loads(initial_context)
            if not isinstance(parsed_initial_context, dict):
                raise ValueError("Initial context must be a JSON object (dict).")
            logger.info(f"Parsed initial context: {parsed_initial_context}")
        except py_json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --initial-context: {e}")
            click.echo(f"Error: Invalid JSON provided for --initial-context. Details: {e}", err=True)
            raise click.Abort()
        except ValueError as e: 
            logger.error(f"Error with initial context structure: {e}")
            click.echo(f"Error: {e}", err=True) 
            raise click.Abort()

    # Add MCP root workspace path to the initial context
    # cli.py is in chungoid-core/src/chungoid/cli.py
    # MCP root is four levels up from cli.py's directory
    mcp_root_path = Path(__file__).parent.parent.parent.parent.resolve()
    parsed_initial_context['mcp_root_workspace_path'] = str(mcp_root_path)
    logger.info(f"Added 'mcp_root_workspace_path': {str(mcp_root_path)} to initial context.")

    # Prepare run_id
    current_run_id = run_id_override_opt if run_id_override_opt else str(uuid.uuid4())
    parsed_initial_context["_run_id"] = current_run_id 
    logger.info(f"Build Run ID: {current_run_id}")

    # Merge tags into parsed_initial_context
    if tags:
        parsed_initial_context["_run_tags"] = [tag.strip() for tag in tags.split(',')]
        logger.info(f"Added tags to initial context: {parsed_initial_context.get('_run_tags')}")

    async def do_build():
        nonlocal user_goal
        # nonlocal parsed_initial_context # No longer needed as it's passed correctly or reconstructed

        try:
            # Configuration loading
            logger.info(f"Attempting to load configuration for project: {abs_project_dir}")
            
            project_specific_config_path = abs_project_dir / PROJECT_CHUNGOID_DIR / "project_config.yaml"
            if project_specific_config_path.is_file(): # Check if it's a file
                logger.info(f"Found project-specific config file at: {project_specific_config_path}")
                try:
                    config = load_config(str(project_specific_config_path))
                    logger.info(f"Successfully loaded project-specific configuration. Project ID from config: {config.get('project_id', 'Not set')}")
                except ConfigError as e: # Catch specific ConfigError from load_config
                    logger.warning(f"ConfigError loading {project_specific_config_path}: {e}. Falling back to default config.")
                    config = get_config() # Use default config from load_config() without path
                    logger.info("Using default configuration due to ConfigError in project-specific file.")
            else:
                logger.info(f"Project-specific config file not found at {project_specific_config_path}. Attempting to load default/global config.")
                try:
                    config = get_config() # load_config() without path, uses its internal default logic
                    logger.info("Successfully loaded default/global configuration.")
                    # If using default, ensure project_root related paths are set based on abs_project_dir
                    config['project_root_dir'] = str(abs_project_dir) # Store the actual project root
                    config['project_root_path'] = str(abs_project_dir) # ADDED for AsyncOrchestrator
                    config['project_root'] = str(abs_project_dir)      # ADDED for consistency/other uses
                    config['dot_chungoid_path'] = str(abs_project_dir / PROJECT_CHUNGOID_DIR)
                except ConfigError as e:
                    logger.error(f"Critical ConfigError during default config load: {e}. Cannot proceed.")
                    click.echo(f"Error: Critical configuration error: {e}", err=True)
                    raise ValueError(f"Critical configuration error: {e}") # Re-raise as ValueError to be caught by outer
            
            # Ensure 'project_id' is in the config or generate one
            # Also ensure project_root_path is set before StateManager might need it indirectly,
            # or before Orchestrator needs it.
            if 'project_root_path' not in config or not config['project_root_path']:
                 config['project_root_path'] = str(abs_project_dir)
                 config['project_root'] = str(abs_project_dir) # Keep consistent
                 logger.info(f"Ensured project_root_path is set in config: {config['project_root_path']}")
            
            if "project_id" not in config or not config["project_id"]:
                # Try to load from state manager first if it exists
                # script_dir, core_root_dir, server_prompts_dir are defined below, 
                # ensure they are defined before this block or pass a sensible default to StateManager here.
                # For now, this instantiation was moved down after server_prompts_dir is defined.
                # temp_state_manager_for_pid = StateManager(target_directory=abs_project_dir)
                # existing_project_id = temp_state_manager_for_pid.get_project_id_from_status()
                # This block is being moved after server_prompts_dir is defined.
                pass # Placeholder, actual logic moved down.
            
            # current_project_id = config["project_id"] # This is also moved down

            # Prompt Manager Setup
            # Determine prompt directories relative to the script or a known structure
            # This assumes cli.py is at <some_root>/chungoid-core/src/chungoid/cli.py
            # And server_prompts are at <some_root>/chungoid-core/server_prompts/
            script_dir = Path(__file__).parent.resolve() # chungoid-core/src/chungoid/
            core_root_dir = script_dir.parent.parent # chungoid-core/
            server_prompts_dir = core_root_dir / "server_prompts"
            
            prompt_manager = PromptManager(prompt_directory_paths=[server_prompts_dir])
            logger.info(f"PromptManager initialized with directory: {server_prompts_dir}")

            # State Manager must be initialized to potentially load an existing project_id
            # from its state file before we decide if we need to generate a new one.
            state_manager = StateManager(
                target_directory=abs_project_dir,
                server_stages_dir=server_prompts_dir # Use the defined server_prompts_dir
            )
            logger.info(f"StateManager initialized for project directory: {abs_project_dir} (pre-project_id check)")

            # Now that server_prompts_dir is defined, handle project_id loading/generation
            if "project_id" not in config or not config["project_id"]:
                try:
                    # Attempt to get project_id from the loaded state within StateManager
                    # _project_state should be populated by StateManager's __init__
                    # via _load_or_initialize_project_state
                    if hasattr(state_manager, '_project_state') and state_manager._project_state and state_manager._project_state.project_id:
                        existing_project_id = state_manager._project_state.project_id
                        config["project_id"] = existing_project_id
                        logger.info(f"Using existing project_id from StateManager's loaded state: {existing_project_id}")
                    else:
                        # This case means StateManager initialized a new (default/placeholder) state
                        # or the existing file was truly empty/corrupt regarding project_id.
                        # So, we generate a new one and it will be set in the config.
                        # The subsequent call to state_manager.initialize_project will then use this new ID
                        # and if it was a fresh default state, it will get updated, or it might
                        # still raise an error if a *different* valid ID was already there, which initialize_project handles.
                        new_project_id = str(uuid.uuid4())
                        config["project_id"] = new_project_id
                        logger.info(f"Generated new project_id: {new_project_id} (as it was not in config or initial StateManager state) and added to config.")
                except Exception as e_sm_init_for_id:
                    logger.warning(f"Could not reliably get existing project_id from StateManager: {e_sm_init_for_id}. Generating new project_id.")
                    new_project_id = str(uuid.uuid4())
                    config["project_id"] = new_project_id
                    logger.info(f"Generated new project_id due to StateManager access issue: {new_project_id} and added to config.")
            
            current_project_id = config.get("project_id") 
            if not current_project_id: 
                logger.error("Critical error: project_id is still not set in config after attempted load/generation. Aborting.")
                raise ValueError("project_id could not be determined or generated.")
            logger.info(f"Confirmed current_project_id to be used: {current_project_id}")


            # LLM Provider Setup
            llm_provider_instance: Union[OpenAILLMProvider, MockLLMProvider]
            if use_mock_llm_provider:
                logger.info("Using MockLLMProvider as requested by flag.")
                llm_provider_instance = MockLLMProvider()
            else:
                logger.info("Attempting to use OpenAILLMProvider.")
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.error("OPENAI_API_KEY environment variable not set. Cannot use OpenAILLMProvider.")
                    click.echo("Error: OPENAI_API_KEY environment variable is required when not using --use-mock-llm-provider.", err=True)
            # Instead of click.Abort(), raise a specific error that can be caught by the outer try/except
                    raise ValueError("OPENAI_API_KEY not set for OpenAILLMProvider")
                
                # You might want to fetch a default_model from config or use a hardcoded one
                default_openai_model = config.get("llm_config", {}).get("default_openai_model", "gpt-3.5-turbo")
                logger.info(f"Initializing OpenAILLMProvider with default model: {default_openai_model}")
                llm_provider_instance = OpenAILLMProvider(api_key=openai_api_key, default_model=default_openai_model)

            llm_manager = LLMManager(
                llm_provider_instance=llm_provider_instance, 
                prompt_manager=prompt_manager
            )
            logger.info(f"LLMManager initialized with {type(llm_provider_instance).__name__}.")

            # State Manager Setup (already initialized above to get project_id, ensure it's the same instance)
            # Re-initializing would be problematic. We just ensure the config has the right ID now.
            # The state_manager instance created before the project_id logic block is the one we use.
            # project_internal_dir = abs_project_dir / PROJECT_CHUNGOID_DIR # Already defined
            # state_manager = StateManager( # DO NOT RE-INITIALIZE HERE
            #     target_directory=abs_project_dir, 
            #     server_stages_dir=server_prompts_dir 
            # )
            # logger.info(f"StateManager initialized for project directory: {abs_project_dir}") # Already logged
            
            # Initialize project state and get/confirm project_id
            # This will create .chungoid/chungoid_status.json if it doesn't exist
            # And ensure the project_id in the config matches or is set from state.
            logger.info(f"Calling state_manager.initialize_project with project_id: {current_project_id}, project_name: {abs_project_dir.name}, goal: {user_goal[:50]}...") # Log goal snippet
            # initialize_project returns a ProjectStateV2 object; extract the authoritative project_id from it
            initialized_project_state = state_manager.initialize_project(
                project_id=current_project_id,
                project_name=abs_project_dir.name,
                initial_user_goal_summary=user_goal
            )

            authoritative_project_id: str = initialized_project_state.project_id

            if authoritative_project_id != current_project_id:
                logger.warning(
                    f"Project ID mismatch or update: Config had {current_project_id}, StateManager initialized with {authoritative_project_id}. Using {authoritative_project_id}."
                )
                config["project_id"] = authoritative_project_id  # Update config with the authoritative ID
                current_project_id = authoritative_project_id


            # Agent Registry Setup
            agent_registry = AgentRegistry(project_root=abs_project_dir) # MODIFIED to use abs_project_dir
            agent_registry.add(core_stage_executor_card, overwrite=True)
            # Remove mock agent card registrations
            # agent_registry.add_agent_card(get_mock_system_intervention_agent_card()) # REMOVE
            # agent_registry.add_agent_card(get_mock_code_generator_agent_card()) # REMOVE
            # agent_registry.add_agent_card(get_mock_test_generation_agent_v1_card()) # REMOVE
            agent_registry = AgentRegistry(project_root=abs_project_dir) # MODIFIED: Added project_root
            # Register essential system agents (example, adapt as needed)
            agent_registry.add(core_stage_executor_card, overwrite=True) # MODIFIED: Added overwrite=True
            # agent_registry.add_agent_card(get_master_planner_agent_card()) # Removed for now, to be added via fallback or other mechanism
            # agent_registry.add_agent_card(get_master_planner_reviewer_agent_card())

            # Add mock agents for testing if needed (or make this conditional)
            # agent_registry.add_agent_card(get_mock_system_intervention_agent_card())
            # agent_registry.add_agent_card(get_mock_code_generator_agent_card())
            # agent_registry.add_agent_card(get_mock_test_generation_agent_card())
            # agent_registry.add_agent_card(get_mock_system_requirements_gathering_agent_card())

            # Fallback agents map - these are classes, not cards.
            # RegistryAgentProvider will instantiate them if they are not in the AgentRegistry (cards).
            
            # Project Chroma Manager - needed for many core flows if not using mocks for it
            project_chroma_manager = ProjectChromaManagerAgent_v1(
                project_root_workspace_path=str(abs_project_dir), # MODIFIED: Renamed project_root_path and ensured it's a string
                project_id=current_project_id # ADDED: Pass the project_id
                # llm_provider=llm_provider_instance, # If it needs LLM
                # prompt_manager=prompt_manager    # If it needs prompts
            )
            # No card needed if we are directly instantiating and passing,
            # but if plan references by ID, it MUST be in fallback or registry.
            
            core_system_agent_classes = {
                # "SystemMasterPlannerAgent_v1": MasterPlannerAgent, # Provided directly by Orchestrator if not in registry/fallback
                # "SystemMasterPlannerReviewerAgent_v1": MasterPlannerReviewerAgent,
                SystemTestRunnerAgent.AGENT_ID: SystemTestRunnerAgent,
                ProjectChromaManagerAgent_v1.AGENT_ID: project_chroma_manager, # Pass the instance
            }

            # Determine final fallback map
            final_fallback_map: Dict[AgentID, Union[Type[BaseAgent], BaseAgent]] = {}
            
            # Option 1: Always use mock fallback map if `use_mock_fallback_agents` is true (add this flag if needed)
            # For build, let's be more explicit or rely on a slim set of core agents.
            # if config.get("developer_settings", {}).get("use_mock_fallback_agents", False): # Example flag
            #    logger.info("Using mock agent fallback map from testing_mock_agents.")
            #    final_fallback_map.update(get_mock_agent_fallback_map()) # This returns a map of ID to CLASS

            # Option 2: Selective fallback for core agents
            # These are agent CLASSES that the provider will instantiate if needed
            # The orchestrator primarily needs the MasterPlannerAgent. Others are plan-dependent.
            final_fallback_map.update({
                MasterPlannerAgent.AGENT_ID: MasterPlannerAgent, # ADDED: Ensure MasterPlannerAgent is in fallback
                ArchitectAgent_v1.AGENT_ID: ArchitectAgent_v1,
                CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # Original CoreCodeGeneratorAgent_v1
                TestGeneratorAgent.AGENT_ID: TestGeneratorAgent, # Original CoreTestGeneratorAgent_v1
                SmartCodeIntegrationAgent_v1.AGENT_ID: SmartCodeIntegrationAgent_v1,
                "FileOperationAgent_v1": SystemFileSystemAgent_v1, # ADDED: Map FileOperationAgent_v1 to SystemFileSystemAgent_v1
                SystemRequirementsGatheringAgent_v1.AGENT_ID: SystemRequirementsGatheringAgent_v1, # Ensure this is here
                # "NoOpAgent_v1": MockNoOpAgent,  # <<< REMOVE THIS LINE
                # "SystemInterventionAgent_v1": MockSystemInterventionAgent, # <<< REMOVE THIS LINE
            })
            final_fallback_map.update(core_system_agent_classes)

            # ADD HumanInputAgent_v1 to the fallback map explicitly as the mock planner uses it.
            # This ensures it's available even if use_mock_fallback_agents is false.
            # The mock plan specifically requests "HumanInputAgent_v1".
            # human_input_agent_id_from_plan = "HumanInputAgent_v1" # AGENT_ID of MockHumanInputAgent
            # if human_input_agent_id_from_plan not in final_fallback_map:
            #     final_fallback_map[human_input_agent_id_from_plan] = MockSystemInterventionAgent # The class # CORRECTED
            #     logger.info(f"Explicitly added '{human_input_agent_id_from_plan}' (maps to MockSystemInterventionAgent class) to fallback map.")

            # --- DIAGNOSTIC LOGGING for final_fallback_map in do_build ---
            logger.info(f"FINAL_FALLBACK_MAP_KEYS (do_build): {list(final_fallback_map.keys())}")
            if "NoOpAgent_v1" in final_fallback_map:
                logger.info("DIAGNOSTIC (do_build): 'NoOpAgent_v1' IS PRESENT in final_fallback_map keys.")
                logger.info(f"DIAGNOSTIC (do_build): Value for 'NoOpAgent_v1' is {final_fallback_map['NoOpAgent_v1']}")
            else: 
                logger.error("DIAGNOSTIC (do_build): 'NoOpAgent_v1' IS MISSING from final_fallback_map keys.")
            # --- END DIAGNOSTIC LOGGING ---
            
            agent_provider = RegistryAgentProvider( # Align this block with the `if/else` above
                registry=agent_registry,
                fallback=final_fallback_map, 
                llm_provider=llm_provider_instance,
                prompt_manager=prompt_manager,
                project_chroma_manager=project_chroma_manager,
            )
            logger.info("RegistryAgentProvider initialized.") # Align this with agent_provider

            # Orchestrator Setup
            # The orchestrator will now use the goal string to generate a plan via the LLMManager
            
            # --- Metrics Store Setup --- ADDED THIS BLOCK
            metrics_store_root = abs_project_dir # Use the absolute project directory path
            metrics_store = MetricsStore(project_root=metrics_store_root)
            logger.info(f"MetricsStore initialized for project root: {metrics_store_root}")
            # --- End Metrics Store Setup ---

            # Now, initialize the orchestrator with the loaded/defaulted config
            # and other necessary components.
            raw_on_failure_action = config.get("orchestrator", {}).get("default_on_failure_action")
            try:
                default_on_failure_action_enum = OnFailureAction(raw_on_failure_action) if raw_on_failure_action else None
            except ValueError:
                logger.error(f"Invalid default_on_failure_action value '{raw_on_failure_action}' in config. Falling back to INVOKE_REVIEWER.")
                default_on_failure_action_enum = OnFailureAction.INVOKE_REVIEWER

            orchestrator = AsyncOrchestrator(
                config=config, # Pass the dict
                agent_provider=agent_provider,
                state_manager=state_manager,
                metrics_store=metrics_store,
                # llm_manager=llm_manager, # REMOVED - Not an expected argument
                master_planner_reviewer_agent_id=config.get("orchestrator", {}).get("master_planner_reviewer_agent_id"),
                default_on_failure_action=default_on_failure_action_enum
            )

            # Generate a unique run ID (already done as current_run_id from outer scope)
            logger.info(f"Build Run ID: {current_run_id}")


            # Run the orchestrator with the user goal
            # The orchestrator's `run` method should handle plan generation if goal_str is provided
            logger.info(f"Executing orchestrator with user goal: {user_goal[:100]}...")
            
            # final_context: Dict[str, Any] = await orchestrator.run( # OLD, incorrect type
            final_status, final_shared_context, final_error_details = await orchestrator.run(
                goal_str=user_goal, 
                initial_context=parsed_initial_context, 
                run_id_override=current_run_id 
            )

            # Extract results from the final context
            # This part needs careful consideration based on what `run` actually returns
            # and what 'final_context' is expected to hold for summarization.
            # Assuming final_shared_context holds the relevant data for now.
            
            # Log the final status and any error details
            logger.info(f"Orchestrator finished with status: {final_status}")
            if final_error_details:
                # logger.error(f"Orchestrator final error details: {final_error_details.model_dump_json(indent=2)}") # OLD
                try:
                    error_dict = final_error_details.to_dict() # Use to_dict()
                    logger.error(f"Orchestrator final error details: {py_json.dumps(error_dict, indent=2)}")
                except AttributeError:
                    # Fallback if to_dict() is also missing for some reason, or final_error_details is not what we expect
                    logger.error(f"Orchestrator final error details (raw): {final_error_details}")
            
            output_summary = "Build process completed."
            if final_shared_context and final_shared_context.data:
                output_summary += f" Final context keys: {list(final_shared_context.data.keys())}"
                # You might want to serialize part of final_shared_context.data if it's useful
                # For example, if there's a specific output key you expect:
                # final_result_data = final_shared_context.data.get("final_output_data", "No specific output found.")
                # logger.info(f"Final result data: {final_result_data}")
            else:
                output_summary += " No final shared context data available."
            
            print(f"\n{output_summary}")
            # Example of accessing specific fields if needed, adapt as per actual SharedContext structure
            # final_status_val = final_context.get("_orchestrator_final_status", "ERROR_STATUS_NOT_FOUND") # OLD
            final_status_val = final_status.value # Accessing the value of the Enum

            if final_status in [StageStatus.COMPLETED_SUCCESS, StageStatus.COMPLETED_WITH_WARNINGS]:
                logger.info("Build process completed successfully.")
                click.echo("Build process completed successfully.")
                if final_shared_context and final_shared_context.data:
                    click.echo(f"Final context keys: {list(final_shared_context.data.keys())}")
                else:
                    click.echo("No final shared context data available.")
            elif final_status == StageStatus.COMPLETED_FAILURE:
                logger.error("Build process failed.")
                click.echo("Build process failed.")
                if final_error_details:
                    click.echo(f"Final error details: {final_error_details}")
                else:
                    click.echo("No final error details available.")
            else:
                logger.warning("Build process completed with warnings.")
                click.echo("Build process completed with warnings.")
                if final_shared_context and final_shared_context.data:
                    click.echo(f"Final context keys: {list(final_shared_context.data.keys())}")
                else:
                    click.echo("No final shared context data available.")
                if final_error_details:
                    click.echo(f"Final error details: {final_error_details}")
                else:
                    click.echo("No final error details available.")

        except Exception as e:
            logger.error(f"An error occurred during the build process: {e}", exc_info=True)
            # rich.traceback.Console().print_exception() # Already logged with exc_info
            click.echo(f"An unexpected error occurred: {e}", err=True)
            # No sys.exit(1) here
            raise # Re-raise
        finally:
            # Ensure LLM client (if any) is closed, e.g., if OpenAILLMProvider was used
            if 'llm_manager' in locals() and llm_manager._llm_provider is not None:
                 logger.info(f"Attempting to close LLM provider client of type: {type(llm_manager._llm_provider).__name__}")
                 await llm_manager.close_client() # LLMManager now has a close_client method

    try:
        asyncio.run(do_build())
        # Successful completion, exit code 0 is implicit
    except Exception:
         # Errors are logged within do_build. We re-raise them to allow a non-zero exit code.
         # click.echo("Build failed. See logs for details.", err=True) # Already echoed
         sys.exit(1) # Ensure non-zero exit code on any exception from do_build

# Ensure the main CLI entry point is correct
if __name__ == "__main__":
    cli()