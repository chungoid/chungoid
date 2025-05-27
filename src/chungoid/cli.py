from __future__ import annotations
import json
from chungoid.utils.config_manager import ConfigurationManager, ConfigurationError
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
import yaml
from typing import Any, Optional, Dict, List, cast, Union, Type

import click
import rich.traceback
from rich.logging import RichHandler

import chungoid
from chungoid.utils.logger_setup import setup_logging # Ensure this import is present
from chungoid.constants import (DEFAULT_MASTER_FLOWS_DIR, DEFAULT_SERVER_STAGES_DIR, MIN_PYTHON_VERSION,
                              PROJECT_CHUNGOID_DIR, STATE_FILE_NAME) # REMOVED DEFAULT_SERVER_PROMPTS_DIR
from chungoid.schemas.common_enums import FlowPauseStatus, StageStatus, HumanReviewDecision, OnFailureAction
from chungoid.core_utils import get_project_root_or_raise, init_project_structure
from chungoid.schemas.master_flow import MasterExecutionPlan
from chungoid.schemas.metrics import MetricEventType
from chungoid.schemas.flows import PausedRunDetails
# Legacy agents moved to agents-old - using stubs for now
# from chungoid.runtime.agents.system_master_planner_reviewer_agent import get_agent_card_static as get_reviewer_card
from chungoid.runtime.unified_orchestrator import UnifiedOrchestrator  # Phase-3 UAEI migration
from chungoid.schemas.metrics import MetricEvent
from chungoid.utils.agent_registry import AgentRegistry
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.llm_provider import MockLLMProvider, LLMManager # Keep MockLLMProvider for --use-mock-llm-provider flag

# Legacy agent imports replaced with Phase-3 UAEI system
# All legacy runtime.agents imports have been migrated to agents-old
# Using the new autonomous_engine agents and UnifiedOrchestrator

# Core Phase-3 agents that exist
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1

# Legacy agents are temporarily disabled during Phase-3 migration
# These will be re-enabled as they are migrated to UnifiedAgent pattern
# REMOVED: ProjectChromaManagerAgent_v1 import - replaced with MCP tools
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

import subprocess
import json as py_json # Avoid conflict with click.json

# Diagnostic code has been removed for cleaner user experience

# Original application imports
from chungoid.utils.state_manager import StateManager # StatusFileError was used here but not defined, assume from StateManager or custom exceptions
from chungoid.utils.config_manager import ConfigurationManager, ConfigurationError # Use new system
from chungoid.utils.logger_setup import setup_logging
from chungoid.schemas.user_goal_schemas import UserGoalRequest # <<< ADD THIS IMPORT

# Imports needed for new 'flow resume' command
import asyncio
import json as py_json # Alias to avoid conflict with click option
from chungoid.runtime.unified_orchestrator import UnifiedOrchestrator  # Phase-3 UAEI migration
from chungoid.schemas.master_flow import MasterExecutionPlan # <<< Import MasterExecutionPlan
# REMOVED: Legacy agent resolver imports - Phase 3 UAEI migration eliminates RegistryAgentProvider patterns
# from chungoid.utils.flow_registry import FlowRegistry # No longer used directly for master plans
from chungoid.utils.master_flow_registry import MasterFlowRegistry
from chungoid.utils.agent_registry import AgentRegistry # Import AgentRegistry
# Legacy imports commented out during Phase-3 migration
# from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card, core_stage_executor_agent
from chungoid.schemas.flows import PausedRunDetails # Ensure this is imported
from chungoid.utils.config_manager import get_config # For default config
# <<< Import patch and AsyncMock >>>
from unittest.mock import patch, AsyncMock
# import chungoid.server_prompts as server_prompts_pkg # REMOVED IMPORT

# New imports for metrics CLI
from chungoid.utils.metrics_store import MetricsStore
from chungoid.schemas.metrics import MetricEventType
from datetime import datetime, timezone # For summary display

# Legacy agent imports commented out during Phase-3 migration
# from chungoid.runtime.agents.system_master_planner_reviewer_agent import MasterPlannerReviewerAgent, get_agent_card_static as get_master_planner_reviewer_agent_card
# from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent, get_agent_card_static as get_master_planner_agent_card

# New imports for MasterPlannerInput
from chungoid.schemas.agent_master_planner import MasterPlannerInput # <<< ADD THIS IMPORT

# Legacy agent imports commented out during Phase-3 migration
# from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1 as CodeGeneratorAgent
# from chungoid.runtime.agents.core_test_generator_agent import CoreTestGeneratorAgent_v1 as TestGeneratorAgent  # COMMENTED OUT - Module doesn't exist
# from chungoid.runtime.agents.smart_code_integration_agent import SmartCodeIntegrationAgent_v1
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1

# New imports for MockTestGenerationAgentV1
# from chungoid.runtime.agents.mocks.mock_test_generation_agent import MockTestGenerationAgentV1, get_agent_card_static as get_mock_test_generation_agent_v1_card

# Legacy agent imports commented out during Phase-3 migration
# from chungoid.runtime.agents import system_test_runner_agent # ADDED # OLD IMPORT
# from chungoid.runtime.agents.system_test_runner_agent import SystemTestRunnerAgent_v1 as SystemTestRunnerAgent # NEW IMPORT - Module doesn't exist
# from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1 # ADDED IMPORT

# Ensure AgentID type is available if used for keys, though strings are fine for dict keys.
from chungoid.schemas.common import AgentID # CORRECTED IMPORT
# from chungoid.runtime.agents.base import AgentBase # For type hinting if needed # REMOVED

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

# Legacy agent imports commented out during Phase-3 migration
# from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1
# REMOVED: from chungoid.agents.autonomous_engine import get_autonomous_engine_agent_fallback_map
# Legacy fallback map import removed - using registry-first architecture

# For dependencies of agents and UnifiedAgentResolver
from chungoid.utils.llm_provider import LLMProvider # Already imported but ensure it's available
# from chungoid.utils.prompt_manager import PromptManager # Already imported but ensure it's available

# ADDED: Import ProjectChromaManagerAgent_v1 at the top for the map
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

# Legacy agent imports commented out during Phase-3 migration
# from chungoid.runtime.agents.system_agents.noop_agent import NoOpAgent_v1 

# ADDED: Import get_registry_agent_provider function
from chungoid.agents import get_registry_agent_provider

# --- Production System Agents Fallback Map ---
# This map defines the primary fallback for system agents.
# It uses the actual agent classes.
# REMOVED: PRODUCTION_SYSTEM_AGENTS_MAP - replaced with registry-first architecture
# All agents are now auto-registered via @register_agent decorators
# PRODUCTION_SYSTEM_AGENTS_MAP: Dict[AgentID, Union[Type[ProtocolAwareAgent], ProtocolAwareAgent]] = {
#     MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
#     MasterPlannerReviewerAgent.AGENT_ID: MasterPlannerReviewerAgent,
#     CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # Alias for CoreCodeGeneratorAgent_v1
#     SmartCodeIntegrationAgent_v1.AGENT_ID: SmartCodeIntegrationAgent_v1,
#     "SmartCodeGeneratorAgent_v1": SmartCodeIntegrationAgent_v1, # ADDED ALIAS
#     # SystemTestRunnerAgent.AGENT_ID: SystemTestRunnerAgent, # COMMENTED OUT - Module doesn't exist
#     SystemFileSystemAgent_v1.AGENT_ID: SystemFileSystemAgent_v1, # Ensure this uses the class name
#     SystemRequirementsGatheringAgent_v1.AGENT_ID: SystemRequirementsGatheringAgent_v1,
#     ArchitectAgent_v1.AGENT_ID: ArchitectAgent_v1,
#     NoOpAgent_v1.AGENT_ID: NoOpAgent_v1, # ADDED NoOpAgent_v1
#     # ProjectChromaManagerAgent_v1 is typically instantiated directly in CLI commands
#     # and added to the fallback map as an instance, not as a class here.
#     # AutonomousEngineAgent_v1 is handled by get_autonomous_engine_agent_fallback_map
# }
# --- End Production System Agents Fallback Map ---

# REMOVED: Legacy autonomous engine fallback map - replaced with registry-first architecture
# --- Autonomous Engine Agents Fallback Map ---
# def get_autonomous_engine_agent_fallback_map() -> Dict[AgentID, Union[Type[ProtocolAwareAgent], ProtocolAwareAgent]]:
#     """Returns a dictionary of core autonomous engine agents for fallback."""
#     # This map should only contain system-critical, non-mock agents
#     # that are part of the autonomous engine's core capabilities.
#     return {
#         # AutonomousEngineAgent_v1.AGENT_ID: AutonomousEngineAgent_v1, # COMMENTED OUT
#         # Other autonomous engine agents can be added here as classes.
#         # ProjectChromaManagerAgent_v1 is often instantiated specifically with project context
#         # in the CLI command logic and added to the final_fallback_map there.
#     }
# --- End Autonomous Engine Agents Fallback Map ---

# Assuming StatusFileError might be a custom exception, if not defined elsewhere, it might need to be.
# For now, let's assume it's imported or defined if critical. If it's from a known module, add import.
# Example: from chungoid.utils.exceptions import StatusFileError (if it exists there)
# If it was a typo and meant something else, that would need correction.
# For now, proceeding as if it will be resolved by existing imports or is not critical path for this edit.

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NEW HELPER for default server prompts base directory
# ---------------------------------------------------------------------------
def _get_default_server_prompts_base_dir() -> Path:
    """
    Determines the default base directory for server prompts.
    This is typically <chungoid_core_root>/server_prompts/
    """
    try:
        # cli.py is in <core_root>/src/chungoid/cli.py
        # server_prompts is at <core_root>/server_prompts/
        cli_file_path = Path(__file__).resolve()
        core_root_dir = cli_file_path.parent.parent.parent # Gets to <core_root>
        default_path = core_root_dir / "server_prompts"
        if default_path.is_dir():
            logger.debug(f"Determined default server_prompts base dir: {default_path}")
            return default_path
        else:
            logger.warning(f"Default server_prompts base dir not found at {default_path}. Attempting fallback via constants.")
    except Exception as e:
        logger.warning(f"Error determining default server_prompts base dir from script path: {e}. Attempting fallback via constants.")

    # Fallback to constant if available and seems correct
    # Ensure DEFAULT_SERVER_STAGES_DIR is imported from chungoid.constants
    if DEFAULT_SERVER_STAGES_DIR:
        stages_path = Path(DEFAULT_SERVER_STAGES_DIR)
        # Try to derive from DEFAULT_SERVER_STAGES_DIR (e.g., if it's '.../server_prompts/stages')
        if stages_path.is_absolute() and stages_path.name == "stages" and stages_path.parent.is_dir():
            logger.info(f"Derived server_prompts base dir from DEFAULT_SERVER_STAGES_DIR: {stages_path.parent}")
            return stages_path.parent
        elif not stages_path.is_absolute(): # If it's a relative path like "server_stages"
            # This case is less ideal as we don't have a clear root to resolve it against here.
            # The primary method using __file__ should ideally work.
            # If we reach here with a relative DEFAULT_SERVER_STAGES_DIR, it implies a less standard setup.
            # We might assume it's relative to a conceptual 'server_prompts' dir, so its parent would be that.
            # However, this is speculative. Let's prioritize the __file__ based method.
            # If primary method fails, and DEFAULT_SERVER_STAGES_DIR is just "server_stages",
            # then Path("server_stages").parent is just ".", which isn't helpful.
            # The initial __file__ based logic should be the most reliable.
            # The original logic for DEFAULT_SERVER_PROMPTS_DIR was a direct path.
            # Let's refine the fallback: if DEFAULT_SERVER_STAGES_DIR is like "server_prompts/stages", its parent is "server_prompts".
            # This requires a bit more convention on what DEFAULT_SERVER_STAGES_DIR might be.
            # For now, the absolute path check is the most robust derivative.
            pass # Avoid making unsafe assumptions with relative DEFAULT_SERVER_STAGES_DIR here.

    # Absolute last resort, though this is unlikely to be correct if others failed.
    final_fallback = Path("server_prompts").resolve()
    logger.error(f"Could not reliably determine default server_prompts base directory. Falling back to relative path: {final_fallback}. This may be incorrect.")
    return final_fallback

# ---------------------------------------------------------------------------
# NEW: LLM Configuration Helper
# ---------------------------------------------------------------------------
def _get_llm_config(
    cli_params: Dict[str, Any], # Parameters from the specific click command context
    project_config_llm_settings: Optional[Dict[str, Any]] = None
    # REMOVED use_mock_override parameter
) -> Dict[str, Any]:
    """
    Constructs the LLM configuration dictionary based on CLI parameters,
    environment variables, and project configuration.
    """
    llm_cfg: Dict[str, Any] = {}

    # REMOVED Mock Override logic block that used use_mock_override
    
    # Check if mock is specified via CHUNGOID_LLM_PROVIDER env var (updated name)
    env_provider = os.getenv("CHUNGOID_LLM_PROVIDER", "").lower()
    if env_provider == "mock":
        logger.info("LLM Config: Using MockLLMProvider due to CHUNGOID_LLM_PROVIDER=mock environment variable.")
        return {
            "provider": "mock",
            "mock_llm_responses": project_config_llm_settings.get("mock_llm_responses", {}) if project_config_llm_settings else {}
        }
    
    # Check if mock is specified in project_config_llm_settings (and not overridden by env var to something else)
    config_provider = (project_config_llm_settings.get("provider", "").lower() if project_config_llm_settings else "")
    if not env_provider and config_provider == "mock":
        logger.info("LLM Config: Using MockLLMProvider due to project configuration.")
        return {
            "provider": "mock",
            "mock_llm_responses": project_config_llm_settings.get("mock_llm_responses", {}) if project_config_llm_settings else {}
        }

    # If not mock, proceed to configure LiteLLM or other providers
    llm_cfg["provider"] = env_provider or config_provider or "openai"

    llm_cfg["default_model"] = (
        cli_params.get("llm_model") or # Assumes --llm-model CLI option exists
        os.getenv("CHUNGOID_LLM_DEFAULT_MODEL") or 
        (project_config_llm_settings.get("default_model") if project_config_llm_settings else None) or 
        "gpt-4o-mini-2024-07-18" # Modern, cost-effective fallback model
    )

    explicit_api_key = (
        cli_params.get("llm_api_key") or # Assumes --llm-api-key CLI option
        os.getenv("CHUNGOID_LLM_API_KEY") or 
        (project_config_llm_settings.get("api_key") if project_config_llm_settings else None)
    )
    if explicit_api_key:
        # Handle SecretStr objects from configuration
        from pydantic import SecretStr
        if isinstance(explicit_api_key, SecretStr):
            llm_cfg["api_key"] = explicit_api_key.get_secret_value()
        else:
            llm_cfg["api_key"] = explicit_api_key

    base_url = (
        cli_params.get("llm_base_url") or # Assumes --llm-base-url CLI option
        os.getenv("CHUNGOID_LLM_BASE_URL") or 
        os.getenv("OLLAMA_BASE_URL") or  # Common for Ollama users
        (project_config_llm_settings.get("base_url") if project_config_llm_settings else None)
    )
    if base_url:
        llm_cfg["base_url"] = base_url
    
    # For provider_env_vars, this would typically come from project_config or be hardcoded
    # if there are specific env vars LiteLLM needs set programmatically.
    if project_config_llm_settings and "provider_env_vars" in project_config_llm_settings:
        llm_cfg["provider_env_vars"] = project_config_llm_settings["provider_env_vars"]

    logger.info(f"LLM Config generated: Provider=\'{llm_cfg['provider']}\', Model=\'{llm_cfg['default_model']}\', APIKey={'present' if 'api_key' in llm_cfg else 'not set'}, BaseURL={'present' if 'base_url' in llm_cfg else 'not set'}")
    return llm_cfg

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_LOG_LEVELS = ["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"] # Corrected this line


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
    """Chungoid-core unified command-line interface."""
    # MODIFIED: Use new setup_logging from chungoid.utils
    setup_logging(level=log_level) # Pass the string directly

    # Store log_level and other shared context if needed.
    ctx.obj = {"log_level": log_level}
    
    # Attempt to load project-specific config if in a project context
    try:
        # Find project root relative to current working directory
        # This might not always be correct if CLI is run from outside a project
        # For commands like \'init\', project_dir might not exist yet.
        # For commands like \'run\', \'status\', \'build\', it\'s more relevant.
        # We should ideally load config based on the explicit project_dir option of each command.
        
        # Placeholder for project_config, to be loaded by individual commands
        # based on their specific project_dir context.
        ctx.obj["project_config"] = {} 
        # config = get_config() # OLD: This was too generic.
        # ctx.obj[\"project_config\"] = config.dict() if config else {}
    except ConfigurationError:
        # logger.warning(\"Could not load project-specific config at CLI entry. Some defaults may apply.\")
        # Let commands handle specific config loading.
        pass
    except Exception: # Catch other potential errors like project not found
        # logger.warning(\"Could not determine project root or load config at CLI entry.\")
        pass


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
@click.option(
    "--llm-provider",
    "llm_provider_cli",
    type=str,
    default=None,
    help="Override LLM provider type for this run."
)
@click.option(
    "--llm-model",
    "llm_model_cli",
    type=str,
    default=None,
    help="Override LLM model ID for this run."
)
@click.option(
    "--llm-api-key",
    "llm_api_key_cli",
    type=str,
    default=None,
    help="Override LLM API key for this run."
)
@click.option(
    "--llm-base-url",
    "llm_base_url_cli",
    type=str,
    default=None,
    help="Override LLM base URL for this run."
)
@click.pass_context
def flow_run(ctx: click.Context, 
             master_flow_id_opt: Optional[str], 
             flow_yaml_opt: Optional[Path], 
             goal: Optional[str], 
             project_dir_opt: Path, 
             initial_context: Optional[str],
             run_id_override_opt: Optional[str],
             tags: Optional[str],
             llm_provider_cli: Optional[str],
             llm_model_cli: Optional[str],
             llm_api_key_cli: Optional[str],
             llm_base_url_cli: Optional[str]
             # REMOVED use_mock_llm_flag from signature
             ) -> None:
    logger = logging.getLogger("chungoid.cli.flow_run")
    project_path = project_dir_opt.resolve()
    logger.info(f"Flow Run: Project directory set to {project_path}")

    if not project_path.exists() or not (project_path / PROJECT_CHUNGOID_DIR).exists():
        click.echo(f"Error: Project directory {project_path} or its .chungoid subdirectory does not exist.", err=True)
        raise click.Abort()

    # Load project-specific config using ConfigurationManager
    try:
        config_manager = ConfigurationManager()
        config_manager.set_project_root(project_path)
        system_config = config_manager.get_config()
        project_config = system_config.model_dump()  # Convert to dict for backward compatibility
        
        # Handle SecretStr conversion for LLM configuration
        if "llm" in project_config and "api_key" in project_config["llm"] and project_config["llm"]["api_key"] is not None:
            from pydantic import SecretStr
            if isinstance(project_config["llm"]["api_key"], SecretStr):
                project_config["llm"]["api_key"] = project_config["llm"]["api_key"].get_secret_value()
        
        logger.info(f"Loaded project config for flow run using ConfigurationManager from {project_path}")
    except ConfigurationError as e:
        logger.warning(f"Configuration error during flow run: {e}. Using default settings.")
        project_config = {}
    except Exception as e:
        logger.warning(f"Unexpected error loading project config: {e}. Using default settings.")
        project_config = {}
        
    ctx.obj["project_config"] = project_config # Store in context

    # Determine server_prompts_base_dir and server_stages_dir for StateManager
    server_prompts_base_dir_str = str(project_config.get("server_prompts_dir") or _get_default_server_prompts_base_dir())
    server_stages_dir_for_sm_str = str(Path(server_prompts_base_dir_str) / "stages")
    logger.info(f"Flow Run: Server prompts base directory: {server_prompts_base_dir_str}")
    logger.info(f"Flow Run: Server stages directory for StateManager: {server_stages_dir_for_sm_str}")

    if goal and (master_flow_id_opt or flow_yaml_opt):
        click.echo("Error: --goal cannot be used with --master-flow-id or --flow-yaml.", err=True)
        raise click.Abort()
    if not goal and not master_flow_id_opt and not flow_yaml_opt:
        click.echo("Error: Must provide --goal, or --master-flow-id, or --flow-yaml.", err=True)
        raise click.Abort()

    # Setup StateManager first as it might hold project_id
    state_manager = StateManager(target_directory=project_path, server_stages_dir=server_stages_dir_for_sm_str)
    run_id = run_id_override_opt or str(uuid.uuid4())
    
    # Determine project_id (crucial for many components including PCMA)
    project_id = state_manager.get_project_id() # Tries status file
    if not project_id:
        project_id = project_config.get("project_id") # Tries loaded config
        if not project_id:
            # This case should be rare if `chungoid init` or `chungoid build` was run, as they establish project_id.
            # If running a flow on a raw directory, a project_id might need to be generated or passed.
            logger.warning("Flow Run: project_id not found in state or config. A new one will not be generated for safety during flow run. Operations requiring project_id may fail.")
            # For robust operation, project_id should exist. Consider erroring if strictly needed.
            # For now, allow to proceed, but some agents (like PCMA) might fail if they require it for init.
            # project_id = str(uuid.uuid4()) # Avoid generating new ID during a run if not present

    logger.info(f"Flow Run: Using Run ID: {run_id}, Project ID: {project_id or 'Not Set'}")

    final_initial_context_dict = {}
    if initial_context:
        try:
            final_initial_context_dict = py_json.loads(initial_context)
        except py_json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in --initial-context: {e}")
            click.echo(f"Error: Invalid JSON provided for --initial-context. Details: {e}", err=True)
            raise click.Abort()
    
    tags_list = [t.strip() for t in tags.split(",")] if tags else []

    # REPLACED: Legacy fallback map approach with registry-first architecture
    # final_fallback_map: Dict[AgentID, Union[Type[ProtocolAwareAgent], ProtocolAwareAgent, AgentFallbackItem]] = dict(PRODUCTION_SYSTEM_AGENTS_MAP)
    # final_fallback_map.update(get_autonomous_engine_agent_fallback_map())
    
    llm_manager_for_flow_run: Optional[LLMManager] = None
    try:
        llm_project_settings = project_config.get("llm", {})
        
        flow_run_cli_params = {
            "llm_provider": llm_provider_cli,
            "llm_model": llm_model_cli,
            "llm_api_key": llm_api_key_cli,
            "llm_base_url": llm_base_url_cli
            # Note: use_mock_llm_provider_flag is passed directly to _get_llm_config use_mock_override
        }

        current_llm_config = _get_llm_config(
                cli_params=flow_run_cli_params,
                project_config_llm_settings=llm_project_settings
                # REMOVED use_mock_override=use_mock_llm_flag
            )

        prompt_manager_base_dir_for_pm = Path(server_prompts_base_dir_str) # Use the corrected base for PromptManager
        prompt_manager_instance = PromptManager(prompt_directory_paths=[prompt_manager_base_dir_for_pm])
        logger.info(f"Flow Run: PromptManager initialized with directory: {prompt_manager_base_dir_for_pm}")

        llm_manager_for_flow_run = LLMManager(llm_config=current_llm_config, prompt_manager=prompt_manager_instance)

    except Exception as e_llm_init:
        logger.error(f"Flow Run: Failed to initialize LLMManager: {e_llm_init}", exc_info=True)
        pass # Allow orchestrator creation; flows not needing LLM might still work.

    # REPLACED: Legacy agent registry with registry-first architecture
    # Agent Registry and Provider
    # agent_registry = AgentRegistry(project_root=project_path, chroma_mode="persistent")
    # # Register essential agents (can be expanded)
    # agent_registry.add_agent_card(core_stage_executor_card())
    # agent_registry.add_agent_card(get_master_planner_agent_card())
    # agent_registry.add_agent_card(get_master_planner_reviewer_agent_card())

    # agent_provider = RegistryAgentProvider(
    #     registry=agent_registry,
    #     fallback=final_fallback_map,  # RENAMED from fallback_agents_map
    #     llm_provider=llm_manager_for_flow_run, # RENAMED from llm_manager
    #     prompt_manager=prompt_manager_instance 
    # )
    
    # ADDED: Registry-first agent provider (NO fallback maps)
    agent_provider = get_registry_agent_provider(
        llm_provider=llm_manager_for_flow_run,
        prompt_manager=prompt_manager_instance
    )
    logger.info("Flow Run: Registry-first AgentProvider initialized (no fallback maps).")

    # Shared context for this run
    # Crucial: project_id MUST be correctly determined and passed if agents like PCMA rely on it in their __init__ via shared_context.
    current_shared_context = {
        "project_id": project_id, 
        "run_id": run_id,
        "project_root_path": str(project_path),
        "llm_manager": llm_manager_for_flow_run,
        "prompt_manager": prompt_manager_instance,
        "agent_provider": agent_provider,
        "state_manager": state_manager,
    }

    # PHASE 3 UAEI: Use UnifiedOrchestrator with UnifiedAgentResolver
    orchestrator = UnifiedOrchestrator(
        config=project_config,
        state_manager=state_manager, 
        agent_resolver=agent_provider,  # Phase 3: agent_provider is now UnifiedAgentResolver
        metrics_store=MetricsStore(project_root=project_path),
        llm_provider=llm_manager_for_flow_run.actual_provider if llm_manager_for_flow_run else None  # Add LLM provider for goal analysis
        # REMOVED: Legacy AsyncOrchestrator parameters eliminated in Phase 3
    )

    async def do_run():
        # ... (rest of do_run logic remains largely the same) ...
        # It will use the orchestrator initialized above.
        nonlocal goal, master_flow_id_opt, flow_yaml_opt, final_initial_context_dict, run_id, tags_list

        master_plan_loaded: Optional[MasterExecutionPlan] = None
        if flow_yaml_opt:
            logger.info(f"Loading master flow from YAML: {flow_yaml_opt}")
            master_plan_loaded = MasterExecutionPlan.load_from_yaml(flow_yaml_opt)
            # If master_flow_id_opt is also given, it might be used as the canonical ID for this loaded plan
            if master_flow_id_opt:
                master_plan_loaded.id = master_flow_id_opt
                logger.info(f"Using provided master_flow_id '{master_flow_id_opt}' for plan loaded from YAML.")
        elif master_flow_id_opt:
            logger.info(f"Attempting to load master flow by ID: {master_flow_id_opt}")
            # This assumes MasterFlowRegistry can load it based on ID from a known location (e.g., .chungoid/master_flows)
            # MasterFlowRegistry needs to be initialized with the project path
            master_flow_reg = MasterFlowRegistry(project_dir=project_path)
            try:
                master_plan_loaded = master_flow_reg.get_master_flow(master_flow_id_opt)
                if not master_plan_loaded:
                    logger.error(f"Master flow with ID '{master_flow_id_opt}' not found by MasterFlowRegistry.")
                    click.echo(f"Error: Master flow with ID '{master_flow_id_opt}' not found.", err=True)
                    return # Abort async task
            except Exception as e_load_mf:
                logger.error(f"Error loading master flow '{master_flow_id_opt}': {e_load_mf}", exc_info=True)
                click.echo(f"Error loading master flow '{master_flow_id_opt}': {e_load_mf}", err=True)
                return
        
        if master_plan_loaded:
            logger.info(f"Executing loaded/retrieved master plan: {master_plan_loaded.id}")
            # Ensure initial_context from CLI is merged with any existing context in the plan
            if master_plan_loaded.initial_context:
                merged_context = {**master_plan_loaded.initial_context, **final_initial_context_dict}
            else:
                merged_context = final_initial_context_dict
            master_plan_loaded.initial_context = merged_context
            
            await orchestrator.execute_master_plan_async(master_plan=master_plan_loaded, run_id_override=run_id, tags_override=tags_list)
        elif goal:
            logger.info(f"Generating and executing new master plan for goal: {goal}")
            planner_input = MasterPlannerInput(
                user_goal=goal,
                project_id=project_id or "UNKNOWN_PROJECT", # Ensure a string, even if unknown
                run_id=run_id,
                initial_context=final_initial_context_dict,
                tags=tags_list
            )
            await orchestrator.execute_master_planner_goal_async(master_planner_input=planner_input)
        else:
            # This case should have been caught by initial CLI param checks
            logger.error("No execution path determined in do_run (no plan, no goal).")
            click.echo("Internal Error: No execution path determined.", err=True)
            return

        logger.info(f"Flow run '{run_id}' completed processing through orchestrator.")

    try:
        asyncio.run(do_run())
    except Exception as e:
        logger.critical(f"Unhandled error during flow run '{run_id}': {e}", exc_info=True)
        click.echo(f"Critical error during flow execution: {e}", err=True)
        sys.exit(1)


@flow.command(name="resume")
@click.argument("run_id", type=str)
@click.option(
    "--project-dir",
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
@click.option(
    "--llm-provider",
    "llm_provider_cli_resume",
    type=str,
    default=None,
    help="Override LLM provider type for this resume operation."
)
@click.option(
    "--llm-model",
    "llm_model_cli_resume",
    type=str,
    default=None,
    help="Override LLM model ID for this resume operation."
)
@click.option(
    "--llm-api-key",
    "llm_api_key_cli_resume",
    type=str,
    default=None,
    help="Override LLM API key for this resume operation."
)
@click.option(
    "--llm-base-url",
    "llm_base_url_cli_resume",
    type=str,
    default=None,
    help="Override LLM base URL for this resume operation."
)
@click.pass_context
def flow_resume(ctx: click.Context, run_id: str, project_dir_opt: Path, action: str, inputs: Optional[str], target_stage: Optional[str],
                  llm_provider_cli_resume: Optional[str],
                  llm_model_cli_resume: Optional[str],
                  llm_api_key_cli_resume: Optional[str],
                  llm_base_url_cli_resume: Optional[str]
                  # REMOVED use_mock_llm_flag_resume from signature
) -> None:
    logger = logging.getLogger("chungoid.cli.flow.resume")
    project_path = project_dir_opt.resolve()
    logger.info(f"Flow Resume: Project directory set to {project_path} for Run ID: {run_id}")

    # Load project-specific config using ConfigurationManager
    try:
        config_manager = ConfigurationManager()
        config_manager.set_project_root(project_path)
        system_config = config_manager.get_config()
        resumed_project_config = system_config.model_dump()  # Convert to dict for backward compatibility
        
        # Handle SecretStr conversion for LLM configuration
        if "llm" in resumed_project_config and "api_key" in resumed_project_config["llm"] and resumed_project_config["llm"]["api_key"] is not None:
            from pydantic import SecretStr
            if isinstance(resumed_project_config["llm"]["api_key"], SecretStr):
                resumed_project_config["llm"]["api_key"] = resumed_project_config["llm"]["api_key"].get_secret_value()
        
        logger.info(f"Loaded project config for flow resume using ConfigurationManager from {project_path}")
    except ConfigurationError as e:
        logger.warning(f"Configuration error during flow resume: {e}. Using default settings.")
        resumed_project_config = {}
    except Exception as e:
        logger.warning(f"Unexpected error loading project config for resume: {e}. Using default settings.")
        resumed_project_config = {}
        
    ctx.obj["project_config"] = resumed_project_config # Store in context

    # Determine server_prompts_base_dir and server_stages_dir for StateManager
    server_prompts_base_dir_str_resume = str(resumed_project_config.get("server_prompts_dir") or _get_default_server_prompts_base_dir())
    server_stages_dir_for_sm_str_resume = str(Path(server_prompts_base_dir_str_resume) / "stages")
    logger.info(f"Flow Resume: Server prompts base directory: {server_prompts_base_dir_str_resume}")
    logger.info(f"Flow Resume: Server stages directory for StateManager: {server_stages_dir_for_sm_str_resume}")

    async def do_resume():
        nonlocal run_id, project_path, action, inputs, target_stage # Ensure these are from outer scope
        
        resumed_sm = StateManager(target_directory=project_path, server_stages_dir=server_stages_dir_for_sm_str_resume)
        
        # Project ID for resume context
        resumed_project_id = resumed_sm.get_project_id() or resumed_project_config.get("project_id")
        if not resumed_project_id:
             logger.warning(f"Flow Resume: project_id not found for run {run_id}. Some operations might be affected.")

        llm_manager_for_resume: Optional[LLMManager] = None
        try:
            resumed_llm_project_settings = resumed_project_config.get("llm", {})
            
            resume_cli_params = {
                "llm_provider": llm_provider_cli_resume,
                "llm_model": llm_model_cli_resume,
                "llm_api_key": llm_api_key_cli_resume,
                "llm_base_url": llm_base_url_cli_resume
            }
            
            current_llm_config_resume = _get_llm_config(
                cli_params=resume_cli_params, 
                project_config_llm_settings=resumed_llm_project_settings 
                # REMOVED use_mock_override=use_mock_llm_flag_resume
            )

            prompt_manager_base_dir_for_pm_resume = Path(server_prompts_base_dir_str_resume) # Use the base for PromptManager
            prompt_manager_instance_for_resume = PromptManager(prompt_directory_paths=[prompt_manager_base_dir_for_pm_resume])
            logger.info(f"Flow Resume: PromptManager initialized with directory: {prompt_manager_base_dir_for_pm_resume}")
            
            llm_manager_for_resume = LLMManager(llm_config=current_llm_config_resume, prompt_manager=prompt_manager_instance_for_resume)
            logger.info(f"Flow Resume: LLMManager initialized with provider: {current_llm_config_resume.get('provider')}")
            
        except Exception as e_llm_resume_init:
            logger.error(f"Flow Resume: Failed to initialize LLMManager: {e_llm_resume_init}", exc_info=True)
            pass

        # REPLACED: Legacy fallback map approach with registry-first architecture
        # Fallback map for agents (consistency with other commands)
        # resume_fallback_map: Dict[AgentID, Union[Type[ProtocolAwareAgent], ProtocolAwareAgent, AgentFallbackItem]] = dict(PRODUCTION_SYSTEM_AGENTS_MAP)
        # resume_fallback_map.update(get_autonomous_engine_agent_fallback_map())

        # agent_registry_resume = AgentRegistry(project_root=project_path, chroma_mode="persistent")
        # agent_registry_resume.add_agent_card(core_stage_executor_card())
        # agent_registry_resume.add_agent_card(get_master_planner_agent_card())
        # agent_registry_resume.add_agent_card(get_master_planner_reviewer_agent_card())

        # agent_provider_resume = RegistryAgentProvider(
        #     registry=agent_registry_resume,
        #     fallback=resume_fallback_map,
        #     llm_provider=llm_manager_for_resume, # RENAMED from llm_manager
        #     prompt_manager=prompt_manager_instance_for_resume
        # )
        
        # ADDED: Registry-first agent provider (NO fallback maps)
        agent_provider_resume = get_registry_agent_provider(
            llm_provider=llm_manager_for_resume,
            prompt_manager=prompt_manager_instance_for_resume
        )
        logger.info("Flow Resume: Registry-first AgentProvider initialized (no fallback maps).")
        
        # Shared context for resume operation
        resume_shared_context = {
            "project_id": resumed_project_id,
            "run_id": run_id, # The run_id being resumed
            "project_root_path": str(project_path),
            "llm_manager": llm_manager_for_resume,
            "prompt_manager": prompt_manager_instance_for_resume,
            "agent_provider": agent_provider_resume,
            "state_manager": resumed_sm,
        }

        # PHASE 1 MIGRATION: Replace AsyncOrchestrator with UnifiedOrchestrator
        # PHASE 3 UAEI: Use UnifiedOrchestrator with UnifiedAgentResolver
        resumed_orchestrator = UnifiedOrchestrator(
            config=resumed_project_config,
            state_manager=resumed_sm, 
            agent_resolver=agent_provider_resume,  # Phase 3: agent_provider is now UnifiedAgentResolver
            metrics_store=MetricsStore(project_root=project_path),
            llm_provider=llm_manager_for_resume.actual_provider if llm_manager_for_resume else None  # Add LLM provider for goal analysis
            # REMOVED: Legacy AsyncOrchestrator parameters eliminated in Phase 3
        )

        inputs_dict: Optional[Dict[str, Any]] = None
        if inputs:
            try:
                inputs_dict = py_json.loads(inputs)
            except py_json.JSONDecodeError as e:
                logger.error(f"Invalid JSON for --inputs: {e}")
                click.echo(f"Error: Invalid JSON provided for --inputs: {e}", err=True)
                return # Abort async task

        await resumed_orchestrator.resume_flow_async(
            run_id_to_resume=run_id, 
            action=action, 
            new_inputs=inputs_dict, 
            target_stage_id_for_branch=target_stage
        )
        logger.info(f"Flow resume for run '{run_id}' completed processing through orchestrator.")

    try:
        asyncio.run(do_resume())
    except Exception as e:
        logger.critical(f"Unhandled error during flow resume '{run_id}': {e}", exc_info=True)
        click.echo(f"Critical error during flow resume: {e}", err=True)
        sys.exit(1)

# Register the command under 'utils' group (only once)
# Ensure this is done only once, typically after the function definition
# if 'show-config' not in [c.name for c in utils.commands.values()]:  # REMOVE THIS LINE
#     utils.add_command(show_config)                                  # REMOVE THIS LINE

@utils.command(name="show-modules")
@click.pass_context
def show_modules(ctx: click.Context):
    """(Dev utility) Show loaded Chungoid modules and their paths."""
    # ... (rest of the function)

# if __name__ == "__main__":
#    cli() # This makes it runnable but is not needed for Click entry point

# BUG_WORKAROUND: code_to_integrate was a literal context path.
# utils.add_command(show_config) # REMOVE THIS LINE

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
    except ConfigurationError as e:
        click.secho(f"Configuration error: {e}", fg="red")
        sys.exit(1)
    except Exception as e:
        click.secho(f"An unexpected error occurred: {e}", fg="red")
        logger.error("Unexpected error in project review CLI command:", exc_info=True)
        sys.exit(1)

# Import for ProjectChromaManagerAgent_v1, needed by the 'build' command
# from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
# CORRECTED IMPORT for HumanReviewRecord
from chungoid.schemas.project_status_schema import HumanReviewRecord 

# NEW BUILD COMMAND
@cli.command("discuss", help="Interactive requirements gathering to create comprehensive project specifications.")
@click.option("--goal-file", type=click.Path(file_okay=True, dir_okay=False, path_type=Path), default="./goal.txt", help="Path to the goal file to read/write. Will be created if it doesn't exist.")
@click.option("--project-dir", "project_dir_opt", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=".", help="Project directory to analyze. Defaults to current directory.")
@click.option("--model", type=str, default=None, help="Override the default LLM model for requirements analysis.")
@click.pass_context
def discuss_requirements(ctx: click.Context, goal_file: Path, project_dir_opt: Path, model: Optional[str]):
    """Interactive requirements gathering agent.
    
    This command runs an interactive conversation to gather comprehensive
    project requirements and generates a structured goal file that can be
    used with 'chungoid build' for any type of software project.
    
    Examples:
        chungoid discuss --goal-file ./my_project_goal.txt --project-dir ./my_project
        chungoid discuss  # Uses default goal.txt in current directory
    """
    
    # Set up logging
    setup_logging(ctx.obj.get("log_level", "INFO"))
    
    # Resolve paths
    goal_file_path = goal_file.resolve()
    project_dir_path = project_dir_opt.resolve()
    
    click.echo(f"🤖 Starting interactive requirements gathering...")
    click.echo(f"📁 Project directory: {project_dir_path}")
    click.echo(f"📄 Goal file: {goal_file_path}")
    click.echo()
    
    async def do_discuss():
        try:
            # Initialize configuration
            config_manager = ConfigurationManager()
            config_manager.set_project_root(project_dir_path)
            system_config = config_manager.get_config()
            config = system_config.model_dump()
            
            # Initialize prompt manager first
            script_dir = Path(__file__).parent.resolve()
            core_root_dir = script_dir.parent.parent
            server_prompts_dir = core_root_dir / "server_prompts"
            prompt_manager = PromptManager(prompt_directory_paths=[server_prompts_dir])
            
            # Set up LLM configuration - handle SecretStr conversion
            llm_settings = config.get("llm", {})
            # Convert SecretStr objects to regular strings for LiteLLM compatibility
            if "api_key" in llm_settings and llm_settings["api_key"] is not None:
                from pydantic import SecretStr
                if isinstance(llm_settings["api_key"], SecretStr):
                    llm_settings["api_key"] = llm_settings["api_key"].get_secret_value()
            
            llm_config = _get_llm_config(
                cli_params={"model": model} if model else {},
                project_config_llm_settings=llm_settings
            )
            
            # Initialize LLM provider
            llm_manager = LLMManager(llm_config=llm_config, prompt_manager=prompt_manager)
            llm_provider = llm_manager.actual_provider
            
            # Create InteractiveRequirementsAgent
            from chungoid.agents.interactive_requirements_agent import InteractiveRequirementsAgent
            
            agent = InteractiveRequirementsAgent(
                llm_provider=llm_provider,
                prompt_manager=prompt_manager
            )
            
            # Create execution context
            from chungoid.schemas.unified_execution_schemas import ExecutionContext, ExecutionConfig, StageInfo
            
            context = ExecutionContext(
                inputs={
                    "goal_file_path": str(goal_file_path),
                    "project_dir": str(project_dir_path)
                },
                shared_context={},
                stage_info=StageInfo(stage_id="interactive_requirements"),
                execution_config=ExecutionConfig(
                    max_iterations=1,  # Single conversation session
                    quality_threshold=0.9
                )
            )
            
            # Execute interactive requirements gathering
            click.echo("Starting conversation...")
            click.echo("=" * 60)
            
            result = await agent._execute_iteration(context, 0)
            
            click.echo("=" * 60)
            
            if result.quality_score > 0.8:
                click.echo("✅ Successfully generated enhanced project specification!")
                click.echo(f"📄 Enhanced goal file written to: {goal_file_path}")
                click.echo()
                click.echo("You can now run 'chungoid build' to start building your project:")
                click.echo(f"  chungoid build --goal-file {goal_file_path} --project-dir {project_dir_path}")
            else:
                click.echo("❌ Requirements gathering encountered issues:")
                if "error" in result.output:
                    click.echo(f"Error: {result.output['error']}")
                else:
                    click.echo("Please try running the command again.")
                    
        except Exception as e:
            logger.error(f"Error in interactive requirements gathering: {e}")
            click.echo(f"❌ Error: {e}")
            sys.exit(1)
    
    # Run the async function
    asyncio.run(do_discuss())


@cli.command("build", help="Build a project from a goal file.")
@click.option("--goal-file", type=click.Path(exists=True, file_okay=True, dir_okay=False, readable=True, path_type=Path), required=True, help="Path to the file containing the user goal.")
@click.option("--project-dir", "project_dir_opt", type=click.Path(file_okay=False, dir_okay=True, path_type=Path), default=".", help="Target project directory. Defaults to current directory. Will be created if it doesn't exist.")
@click.option("--run-id", "run_id_override_opt", type=str, default=None, help="Specify a custom run ID for this execution.")
@click.option("--initial-context", type=str, default=None, help="JSON string containing initial context variables for the build.")
@click.option("--tags", type=str, default=None, help="Comma-separated tags for this build (e.g., 'dev,release').")
@click.option("--model", type=str, default=None, help="Override the default LLM model for this build (e.g., 'gpt-4o', 'gpt-4o-mini-2024-07-18').")
@click.pass_context
def build_from_goal_file(ctx: click.Context, goal_file: Path, project_dir_opt: Path, run_id_override_opt: Optional[str], initial_context: Optional[str], tags: Optional[str], model: Optional[str]):
    """Initiates a project build from a user goal specified in a file."""
    logger = logging.getLogger("chungoid.cli.build")
    abs_project_dir = project_dir_opt.resolve()
    logger.info(f"Starting build from goal file: {goal_file} for project directory: {project_dir_opt}")
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
            click.echo(f"Error: Invalid JSON in --initial-context: {e}", err=True)
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
            # Configuration loading using new ConfigurationManager system
            logger.info(f"Attempting to load configuration for project: {abs_project_dir}")
            
            # Initialize the new configuration manager with project context
            config_manager = ConfigurationManager()
            config_manager.set_project_root(abs_project_dir)
            
            # Get configuration - this will automatically load project-specific config
            system_config = config_manager.get_config()
            logger.info(f"Successfully loaded project configuration using ConfigurationManager")
            
            # Debug logging for configuration
            logger.info(f"DEBUG: LLM config from new system: provider={system_config.llm.provider}, model={system_config.llm.default_model}")
            
        except ConfigurationError as e:
            logger.error(f"Configuration error: {e}")
            click.echo(f"Configuration error: {e}", err=True)
            return
        except Exception as e:
            logger.error(f"Unexpected error loading configuration: {e}")
            click.echo(f"Error loading configuration: {e}", err=True)
            return

        # Convert to dictionary format for compatibility with existing code
        config = system_config.model_dump()
        config['project_root_path'] = str(abs_project_dir)
        config['project_root'] = str(abs_project_dir)
        config['project_id'] = config.get('project_id') or abs_project_dir.name
        
        logger.info(f"Configuration loaded with LLM settings: {system_config.llm.model_dump()}")
        
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

        # --- MODIFIED LLM SETUP FOR BUILD COMMAND START ---
        llm_manager: Optional[LLMManager] = None
        try:
            # Parameters for _get_llm_config specific to the build command
            # For build, there are no explicit --llm-model, --llm-api-key etc. flags directly on `chungoid build`
            # So, these will typically be None, relying on env vars or project_config.
            # The --use-mock-llm-provider flag IS available on `build`.
            build_command_llm_cli_params = {
                "llm_provider": None, # No specific CLI flag for provider on 'build'
                "llm_model": model,    # Use the --model CLI parameter
                "llm_api_key": None,  # No specific CLI flag for api key on 'build'
                "llm_base_url": None  # No specific CLI flag for base url on 'build'
            }
            
            # Project config LLM settings come from the new SystemConfiguration structure
            project_config_llm_settings = {
                "provider": system_config.llm.provider,  # Changed from provider_type
                "default_model": system_config.llm.default_model,
                "api_key": system_config.llm.api_key.get_secret_value() if system_config.llm.api_key else None,
                "base_url": system_config.llm.api_base_url,  # Changed from base_url to api_base_url
                "max_tokens": system_config.llm.max_tokens_per_request,  # Changed from max_tokens
                "temperature": 0.1,  # Default for now
                "timeout": system_config.llm.timeout,
                "max_retries": system_config.llm.max_retries
            }
            
            logger.info(f"DEBUG: Extracted LLM settings: {project_config_llm_settings}")
            
            effective_llm_config = _get_llm_config(
                cli_params=build_command_llm_cli_params,
                project_config_llm_settings=project_config_llm_settings
            )
            
            logger.info(f"DEBUG: Final effective LLM config: {effective_llm_config}")
            
            llm_manager = LLMManager(
                llm_config=effective_llm_config, 
                prompt_manager=prompt_manager # Re-use the already initialized prompt_manager
            )
            logger.info(f"LLMManager initialized for build command with provider: {effective_llm_config.get('provider')}")

        except Exception as e_llm_build_init:
            logger.error(f"Build Command: Failed to initialize LLMManager: {e_llm_build_init}", exc_info=True)
            # Depending on strictness, you might want to raise an error or allow proceeding if LLM isn't critical
            # For a build process, LLM is likely critical for plan generation.
            click.echo(f"Error: Failed to initialize LLM services for build: {e_llm_build_init}", err=True)
            raise ValueError(f"LLM Initialization failed for build: {e_llm_build_init}")
        # --- MODIFIED LLM SETUP FOR BUILD COMMAND END ---
        
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
        agent_registry = AgentRegistry(project_root=abs_project_dir)
        # PHASE 1 MIGRATION: Removed core_stage_executor_card - replaced with UnifiedOrchestrator registry pattern
        # agent_registry.add(core_stage_executor_card, overwrite=True)

        # REMOVED: Project Chroma Manager - replaced with MCP tools
        # project_chroma_manager = ProjectChromaManagerAgent_v1(
        #     project_root_workspace_path=str(abs_project_dir),
        #     project_id=current_project_id
        # )
        
        core_system_agent_classes = {
            # SystemTestRunnerAgent.AGENT_ID: SystemTestRunnerAgent,  # COMMENTED OUT - Module doesn't exist
            # REMOVED: ProjectChromaManagerAgent_v1.AGENT_ID: project_chroma_manager, # Pass the instance
        }

        # REPLACED: Legacy fallback map approach with registry-first architecture
        # Determine final fallback map
        # final_fallback_map: Dict[AgentID, Union[Type[ProtocolAwareAgent], ProtocolAwareAgent, AgentFallbackItem]] = {}
        
        # Option 2: Selective fallback for core agents
        # These are agent CLASSES that the provider will instantiate if needed
        # The orchestrator primarily needs the MasterPlannerAgent. Others are plan-dependent.
        # final_fallback_map.update({
        #     MasterPlannerAgent.AGENT_ID: MasterPlannerAgent,
        #     ArchitectAgent_v1.AGENT_ID: ArchitectAgent_v1,
        #     CodeGeneratorAgent.AGENT_ID: CodeGeneratorAgent, # Original CoreCodeGeneratorAgent_v1
        #     SmartCodeIntegrationAgent_v1.AGENT_ID: SmartCodeIntegrationAgent_v1,
        #     "FileOperationAgent_v1": SystemFileSystemAgent_v1,
        #     SystemRequirementsGatheringAgent_v1.AGENT_ID: SystemRequirementsGatheringAgent_v1,
        # })
        # final_fallback_map.update(core_system_agent_classes)
        
        # agent_provider = RegistryAgentProvider(
        #     registry=agent_registry,
        #     fallback=final_fallback_map,
        #     llm_provider=llm_manager,
        #     prompt_manager=prompt_manager,
        #     # REMOVED: project_chroma_manager=project_chroma_manager
        # )
        
        # ADDED: Registry-first agent provider (NO fallback maps)
        agent_provider = get_registry_agent_provider(
            llm_provider=llm_manager.actual_provider,  # Extract LLMProvider from LLMManager
            prompt_manager=prompt_manager
        )
        logger.info("Build: Registry-first AgentProvider initialized (no fallback maps).")

        # Metrics Store Setup
        metrics_store_root = abs_project_dir
        metrics_store = MetricsStore(project_root=metrics_store_root)
        logger.info(f"MetricsStore initialized for project root: {metrics_store_root}")

        # Now, initialize the orchestrator with the loaded/defaulted config
        # and other necessary components.
        raw_on_failure_action = config.get("orchestrator", {}).get("default_on_failure_action")
        try:
            default_on_failure_action_enum = OnFailureAction(raw_on_failure_action) if raw_on_failure_action else None
        except ValueError:
            logger.error(f"Invalid default_on_failure_action value '{raw_on_failure_action}' in config. Falling back to INVOKE_REVIEWER.")
            default_on_failure_action_enum = OnFailureAction.INVOKE_REVIEWER

        # Create the shared context for this build run
        build_shared_context_data = {
            "project_id": current_project_id,
            "run_id": current_run_id,
            "project_root_path": str(abs_project_dir),
            "llm_manager": llm_manager,
            "prompt_manager": prompt_manager,
            "agent_provider": agent_provider,
            "state_manager": state_manager,
            "metrics_store": metrics_store,
        }
        if parsed_initial_context: # Merge CLI initial context
            build_shared_context_data.update(parsed_initial_context)

        # PHASE 3 UAEI: Use UnifiedOrchestrator with UnifiedAgentResolver
        orchestrator = UnifiedOrchestrator(
            config=config,
            state_manager=state_manager,
            agent_resolver=agent_provider,  # Phase 3: agent_provider is now UnifiedAgentResolver
            metrics_store=metrics_store,
            llm_provider=llm_manager.actual_provider if llm_manager else None  # Add LLM provider for goal analysis
            # REMOVED: Legacy AsyncOrchestrator parameters eliminated in Phase 3
        )

        # Generate a unique run ID (already done as current_run_id from outer scope)
        logger.info(f"Build Run ID: {current_run_id}")

        # Run the orchestrator with the user goal
        # The orchestrator's `run` method should handle plan generation if goal_str is provided
        logger.info(f"Executing orchestrator with user goal: {user_goal[:100]}...")
        
        try:
            # final_context: Dict[str, Any] = await orchestrator.run( # OLD, incorrect type
            final_status, final_shared_context, final_error_details = await orchestrator.run(
                goal_str=user_goal, 
                initial_context=build_shared_context_data,
                run_id_override=current_run_id 
            )

            # Log the final status and any error details
            logger.info(f"Orchestrator finished with status: {final_status}")
            if final_error_details:
                try:
                    error_dict = final_error_details.to_dict()
                    logger.error(f"Orchestrator final error details: {py_json.dumps(error_dict, indent=2)}")
                except AttributeError:
                    logger.error(f"Orchestrator final error details (raw): {final_error_details}")
            
            output_summary = "Build process completed."
            if final_shared_context:
                output_summary += f" Final context keys: {list(final_shared_context.keys())}"
            else:
                output_summary += " No final shared context data available."
            
            print(f"\n{output_summary}")
            final_status_val = final_status.value

            if final_status in [StageStatus.COMPLETED_SUCCESS, StageStatus.COMPLETED_WITH_WARNINGS]:
                logger.info("Build process completed successfully.")
                click.echo("Build process completed successfully.")
                if final_shared_context:
                    click.echo(f"Final context keys: {list(final_shared_context.keys())}")
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
                if final_shared_context:
                    click.echo(f"Final context keys: {list(final_shared_context.keys())}")
                else:
                    click.echo("No final shared context data available.")
                if final_error_details:
                    click.echo(f"Final error details: {final_error_details}")
                else:
                    click.echo("No final error details available.")

        except Exception as e:
            logger.error(f"An error occurred during the build process: {e}", exc_info=True)
            click.echo(f"An unexpected error occurred: {e}", err=True)
            raise # Re-raise
        finally:
            # Ensure LLM client (if any) is closed
            if llm_manager is not None and hasattr(llm_manager, 'close_client'):
                logger.info(f"Attempting to close LLM provider client via LLMManager in build command.")
                await llm_manager.close_client()

    try:
        asyncio.run(do_build())
        # Successful completion, exit code 0 is implicit
    except Exception as e:
         # Show the actual error instead of just "Build failed"
         logger.error(f"Build command failed with exception: {e}", exc_info=True)
         click.echo(f"Build failed with error: {e}", err=True)
         click.echo("See logs for full details.", err=True)
         sys.exit(1) # Ensure non-zero exit code on any exception from do_build

@utils.command(name="show-config")
@click.option(
    "--project-dir",
    "project_dir_opt",
    type=click.Path(file_okay=False, dir_okay=True, exists=False, path_type=Path), # Allow non-existent for inspection
    default=".",
    show_default=True,
    help="Project directory to load config from (default: current directory)."
)
@click.option("--raw", is_flag=True, help="Show raw config dictionary without interpolation or defaults.")
@click.pass_context
def show_config(ctx: click.Context, project_dir_opt: Path, raw: bool):
    """Displays the current Chungoid configuration (loaded or default)."""
    logger = logging.getLogger("chungoid.cli.utils.show_config")
    abs_project_dir = project_dir_opt.resolve()
    click.echo(f"--- Chungoid Configuration Utility ---")

    config_to_display: Optional[Dict[str, Any]] = None
    config_source_info = "Defaults"

    try:
        # Use the new configuration system
        config_manager = ConfigurationManager()
        if abs_project_dir != Path.cwd():
            config_manager.set_project_root(abs_project_dir)
        
        if raw:
            # For raw mode, try to load the actual config file if it exists
            project_config_file = abs_project_dir / PROJECT_CHUNGOID_DIR / "config.yaml"
            if project_config_file.exists():
                with open(project_config_file, 'r') as f:
                    config_to_display = yaml.safe_load(f) or {}
                config_source_info = f"Raw content of {project_config_file}"
            else:
                # Show raw defaults from the Pydantic model
                system_config = config_manager.get_config()
                config_to_display = system_config.model_dump()
                config_source_info = "Default configuration (Pydantic model defaults)"
        else:
            # Show effective merged configuration
            system_config = config_manager.get_config()
            config_to_display = system_config.model_dump()
            config_source_info = "Effective configuration (merged from all sources)"
        
        click.echo(f"Configuration Source: {config_source_info}")
        if config_to_display:
            click.echo(py_json.dumps(config_to_display, indent=2, default=str))
        else:
            click.secho("No configuration loaded or available.", fg="red")

    except Exception as e:
        click.secho(f"An unexpected error occurred while retrieving configuration: {e}", fg="red")
        logger.error(f"Error in show_config: {e}", exc_info=True)

# Add helper functions for backward compatibility
    except Exception as e:
        raise ConfigurationError(f"Failed to load config from {config_path}: {e}")


# Ensure the main CLI entry point is correct
if __name__ == "__main__":
    cli()

@cli.group()
def config():
    """Configuration management commands."""
    pass

@config.command()
@click.option('--format', 'output_format', type=click.Choice(['yaml', 'json', 'table']), default='table', help='Output format')
def show(output_format):
    """Show current configuration."""
    try:
        from chungoid.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        if output_format == 'yaml':
            import yaml
            click.echo(yaml.dump(config.dict(), default_flow_style=False))
        elif output_format == 'json':
            import json
            click.echo(json.dumps(config.dict(), indent=2))
        else:  # table format
            click.echo("Current Configuration:")
            click.echo(f"  LLM Provider: {config.llm.provider}")
            click.echo(f"  Default Model: {config.llm.default_model}")
            click.echo(f"  Fallback Model: {config.llm.fallback_model}")
            click.echo(f"  Max Tokens: {config.llm.max_tokens_per_request}")
            click.echo(f"  Cost Tracking: {config.llm.enable_cost_tracking}")
            if config.llm.monthly_budget_limit:
                click.echo(f"  Monthly Budget: ${config.llm.monthly_budget_limit}")
            
            api_key = config_manager.get_secret('llm.api_key')
            click.echo(f"  API Key: {'Configured' if api_key else 'Not configured'}")
            
    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        sys.exit(1)

@config.command()
@click.argument('model_name')
@click.option('--global', 'is_global', is_flag=True, help='Set globally instead of project-specific')
@click.option('--budget', type=float, help='Set monthly budget limit')
def set_model(model_name, is_global, budget):
    """Set the default LLM model."""
    try:
        from chungoid.utils.config_manager import get_config_manager
        from pathlib import Path
        
        # Validate model name
        valid_models = [
            "gpt-4o-mini-2024-07-18",
            "gpt-4o", 
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ]
        
        if model_name not in valid_models:
            click.echo(f"Warning: '{model_name}' is not a recognized model name.")
            click.echo(f"Valid models: {', '.join(valid_models)}")
            if not click.confirm("Continue anyway?"):
                return
        
        config_manager = get_config_manager()
        
        # Prepare updates
        updates = {
            "llm": {
                "default_model": model_name
            }
        }
        
        if budget:
            updates["llm"]["monthly_budget_limit"] = budget
        
        # Set project root if not global
        if not is_global:
            config_manager.set_project_root(Path.cwd())
        
        # Update configuration
        config_manager.update_configuration(updates, persist=True)
        
        scope = "globally" if is_global else "for this project"
        click.echo(f"✓ Set default model to '{model_name}' {scope}")
        
        if budget:
            click.echo(f"✓ Set monthly budget limit to ${budget}")
            
        # Show cost warning for expensive models
        if model_name in ["gpt-4o", "gpt-4-turbo"] and not budget:
            click.echo("⚠️  Warning: This is a more expensive model. Consider setting a budget limit with --budget")
            
    except Exception as e:
        click.echo(f"Error setting model: {e}", err=True)
        sys.exit(1)

@config.command()
def validate():
    """Validate current configuration."""
    try:
        from chungoid.utils.config_manager import get_config_manager
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        click.echo("✓ Configuration is valid")
        
        # Check for common issues
        api_key = config_manager.get_secret('llm.api_key')
        if not api_key:
            click.echo("⚠️  Warning: No API key configured. Set OPENAI_API_KEY environment variable.")
        
        if not config.llm.enable_cost_tracking:
            click.echo("⚠️  Warning: Cost tracking is disabled. Enable it to monitor usage.")
            
        if not config.llm.monthly_budget_limit:
            click.echo("⚠️  Warning: No monthly budget limit set. Consider setting one to prevent unexpected costs.")
            
    except Exception as e:
        click.echo(f"✗ Configuration validation failed: {e}", err=True)
        sys.exit(1)

@config.command()
@click.option('--model', help='Test with specific model')
def test(model):
    """Test LLM configuration with a simple request."""
    try:
        from chungoid.utils.config_manager import get_config_manager
        from chungoid.utils.llm_provider import LLMManager
        from chungoid.utils.prompt_manager import PromptManager
        import asyncio
        
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # Check API key
        api_key = config_manager.get_secret('llm.api_key')
        if not api_key:
            click.echo("✗ No API key configured. Set OPENAI_API_KEY environment variable.", err=True)
            sys.exit(1)
        
        click.echo("Testing LLM configuration...")
        
        # Create minimal LLM manager
        llm_config = {
            "provider": config.llm.provider,
            "default_model": model or config.llm.default_model,
            "api_key": api_key,
            "timeout": config.llm.timeout,
            "max_retries": config.llm.max_retries
        }
        
        # Create minimal prompt manager (empty for test)
        prompt_manager = PromptManager(prompt_directory_paths=[str(Path(__file__).parent.parent.parent / "server_prompts")])
        llm_manager = LLMManager(llm_config=llm_config, prompt_manager=prompt_manager)
        
        async def test_llm():
            response = await llm_manager.actual_provider.generate(
                prompt="Say 'Configuration test successful' and nothing else.",
                max_tokens=50,
                temperature=0.1
            )
            return response.strip()
        
        result = asyncio.run(test_llm())
        
        click.echo(f"✓ Test successful!")
        click.echo(f"Model: {model or config.llm.default_model}")
        click.echo(f"Response: {result}")
        
    except Exception as e:
        click.echo(f"✗ Test failed: {e}", err=True)
        sys.exit(1)