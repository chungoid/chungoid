from __future__ import annotations

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
import json
import logging
import sys
import os
from typing import Any, Optional, Dict

import click

from chungoid.engine import ChungoidEngine
from chungoid.utils.state_manager import StateManager, StatusFileError

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

# Imports needed for new 'flow resume' command
import asyncio
import json as py_json # Alias to avoid conflict with click option
from chungoid.runtime.orchestrator import AsyncOrchestrator #, ExecutionPlan no longer primary
from chungoid.schemas.master_flow import MasterExecutionPlan # <<< Import MasterExecutionPlan
from chungoid.utils.agent_resolver import RegistryAgentProvider # Example provider
# from chungoid.utils.flow_registry import FlowRegistry # No longer used directly for master plans
from chungoid.utils.master_flow_registry import MasterFlowRegistry # <<< Import MasterFlowRegistry
from chungoid.utils.agent_registry import AgentRegistry # Import AgentRegistry
from chungoid.runtime.agents.core_stage_executor import core_stage_executor_card, core_stage_executor_agent # <<< For registration
from chungoid.schemas.flows import PausedRunDetails # Ensure this is imported

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
    """Chungoid command-line interface."""
    _configure_logging(log_level.upper())
    # Expose common values to sub-commands via click context obj
    ctx.ensure_object(dict)
    ctx.obj["log_level"] = log_level.upper()


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

    if project_path.exists() and any(project_path.iterdir()):  # directory not empty
        click.confirm(
            f"Directory {project_path} is not empty. Continue and create .chungoid/ anyway?",
            abort=True,
        )
    project_path.mkdir(parents=True, exist_ok=True)

    chungoid_dir = project_path / ".chungoid"
    chungoid_dir.mkdir(exist_ok=True)

    status_file = chungoid_dir / "project_status.json"
    if not status_file.exists():
        status_file.write_text(json.dumps({"runs": []}, indent=2))
        logger.debug("Created fresh status file at %s", status_file)

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
# Sub-command group: flow (manage flows)
# ---------------------------------------------------------------------------

@cli.group()
@click.pass_context
def flow(ctx: click.Context) -> None:
    """Manage and interact with execution flows."""
    pass


@flow.command(name="run")
@click.argument("master_flow_id", type=str)
@click.option(
    "--project-dir",
    "project_dir_opt", # Use a different var name to avoid conflict with default context
    type=click.Path(file_okay=False, dir_okay=True, exists=True, path_type=Path),
    default=".",
    help="Project directory containing the 'master_flows' subdirectory (default: current directory)."
)
@click.option(
    "--initial-context",
    type=str,
    default=None,
    help="JSON string containing initial context variables to pass to the flow."
)
@click.pass_context
def flow_run(ctx: click.Context, master_flow_id: str, project_dir_opt: Path, initial_context: Optional[str]) -> None:
    """Run a Master Flow definition from the beginning."""
    logger = logging.getLogger("chungoid.cli.flow.run")
    project_root = project_dir_opt.expanduser().resolve()
    logger.info(f"Attempting to run master_flow_id='{master_flow_id}' in project {project_root}")

    # --- Parse Initial Context --- 
    start_context: Dict[str, Any] = {'outputs': {}} # Ensure outputs dict exists
    if initial_context:
        try:
            parsed_initial_context = py_json.loads(initial_context)
            if not isinstance(parsed_initial_context, dict):
                raise ValueError("Initial context must be a JSON object (dictionary).")
            start_context.update(parsed_initial_context)
            logger.debug(f"Using initial context: {start_context}")
        except (py_json.JSONDecodeError, ValueError) as e:
            click.echo(f"Error: Invalid JSON string provided for --initial-context: {e}", err=True)
            sys.exit(1)

    # --- Async runner --- 
    async def do_run():
        # --- Find server_stages_dir (for StateManager) ---
        # (Reusing the same logic as in flow_resume)
        core_package_dir = Path(__file__).parent.resolve()
        server_stages_dir = core_package_dir / "server_prompts" / "stages"
        if not server_stages_dir.is_dir():
            logger.warning(f"Default server_stages_dir not found at {server_stages_dir}. Trying alternative.")
            try:
                import chungoid
                chungoid_pkg_root = Path(list(chungoid.__path__)[0]).resolve()
                server_stages_dir = chungoid_pkg_root / "server_prompts" / "stages"
                if not server_stages_dir.is_dir():
                    raise FileNotFoundError(f"Fallback server_stages_dir not found: {server_stages_dir}")
                logger.info(f"Using alternative server_stages_dir: {server_stages_dir}")
            except Exception as e_ssd_fallback:
                logger.error(f"Failed to find server_stages_dir via fallback: {e_ssd_fallback}")
                click.echo(f"Error: Critical - could not locate server_prompts/stages directory. {e_ssd_fallback}. Cannot proceed.", err=True)
                sys.exit(1)

        # --- State Manager ---
        state_manager = StateManager(
            target_directory=str(project_root),
            server_stages_dir=str(server_stages_dir),
            use_locking=False # Start with locking disabled for CLI run?
        )
        # Initialize a new run ID for this execution
        run_id = state_manager.get_or_create_current_run_id(new_run=True)
        logger.info(f"Initialized new run_id: {run_id}")

        # --- Master Flow Registry & Load MasterExecutionPlan ---
        master_flows_project_dir = project_root / "master_flows"
        master_flow_registry = MasterFlowRegistry(master_flows_dir=master_flows_project_dir)
        if master_flow_registry.get_scan_errors():
            # Log warnings, but maybe allow proceeding if the requested flow is found
            logger.warning(f"Errors during Master Flow scan in {master_flows_project_dir}:")
            for err in master_flow_registry.get_scan_errors(): logger.warning(f"  - {err}")

        master_plan: Optional[MasterExecutionPlan] = master_flow_registry.get(master_flow_id)
        if not master_plan:
            click.echo(f"Error: Master Flow definition with ID '{master_flow_id}' not found in '{master_flows_project_dir}'.", err=True)
            available_ids = master_flow_registry.list_ids()
            if available_ids:
                click.echo(f"Available flow IDs: {", ".join(available_ids)}")
            else:
                click.echo("No master flows found in the specified directory.")
            sys.exit(1)
        logger.info(f"Loaded MasterExecutionPlan '{master_plan.id}' for execution.")

        # --- Agent Registry (and register core agents) ---
        agent_registry = AgentRegistry()
        try:
            agent_registry.add(core_stage_executor_card, core_stage_executor_agent)
            logger.info(f"Registered built-in agent: {core_stage_executor_card.agent_id}")
        except Exception as reg_err:
            logger.exception(f"Fatal: Failed to register core agent '{core_stage_executor_card.agent_id}': {reg_err}")
            click.echo(f"Error: Failed to register core executor agent. Cannot proceed.", err=True)
            sys.exit(1)
        
        agent_provider = RegistryAgentProvider(registry=agent_registry)

        # --- Orchestrator --- 
        orchestrator = AsyncOrchestrator(
            pipeline_def=master_plan, # Pass the loaded MasterExecutionPlan
            config={}, # Provide config if needed
            agent_provider=agent_provider,
            state_manager=state_manager
        )

        # --- Run the Flow --- 
        click.echo(f"Running Master Flow '{master_plan.id}' (Run ID: {run_id})...")
        try:
            final_context_or_error = await orchestrator.run(
                plan=master_plan,
                context=start_context
            )
        except Exception as e:
            logger.exception(f"An unexpected error occurred during orchestrator.run: {e}")
            click.echo(f"Error during flow execution: {e}", err=True)
            sys.exit(1)

        # --- Handle Result --- 
        if isinstance(final_context_or_error, dict) and final_context_or_error.get("error"):
            # This case might not happen if orchestrator.run raises exceptions or pauses
            click.echo(f"Flow execution finished with error: {final_context_or_error['error']}", err=True)
            sys.exit(1)
        elif isinstance(final_context_or_error, dict): 
            # Check if it paused - relies on pause logic adding info to context or state manager status
            # We need a reliable way to check if the run ended in a paused state.
            # Let's check the StateManager for a paused file for this run_id.
            if state_manager.load_paused_flow_state(str(run_id)):
                 click.echo(f"Flow execution PAUSED. Run ID: {run_id}. Use 'chungoid flow resume {run_id} ...' to continue.")
            else:
                click.echo(f"Flow execution finished successfully for run_id '{run_id}'.")
                # Optionally print final context summary
                # click.echo("Final context keys:")
                # click.echo(list(final_context_or_error.keys()))
        else:
            click.echo(f"Flow execution finished with unexpected result type: {type(final_context_or_error)}", err=True)
            sys.exit(1)

    # Run the async part
    try:
        asyncio.run(do_run())
    except Exception as e:
        logger.error(f"High-level error during flow run execution: {e}")
        click.echo(f"An unexpected error occurred: {e}", err=True)
        sys.exit(1)


@flow.command(name="resume")
@click.argument("run_id", type=str)
@click.option(
    "--action",
    required=True,
    type=click.Choice(["retry", "retry_with_inputs", "skip_stage", "force_branch", "abort"], case_sensitive=False),
    help="The action to perform for resuming the flow."
)
@click.option(
    "--inputs",
    type=str,
    default=None,
    help="JSON string containing inputs to merge into context (for 'retry_with_inputs' action)."
)
@click.option(
    "--target-stage",
    type=str,
    default=None,
    help="The stage ID to jump to (for 'force_branch' action)."
)
@click.pass_context
def flow_resume(ctx: click.Context, run_id: str, action: str, inputs: Optional[str], target_stage: Optional[str]) -> None:
    """Resume a paused flow with a specific action."""
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
        core_package_dir = Path(__file__).parent.resolve()
        server_stages_dir = core_package_dir / "server_prompts" / "stages"
        if not server_stages_dir.is_dir():
            # Fallback if not found relative to cli.py (e.g., in a weird install state)
            logger.warning(f"Default server_stages_dir not found at {server_stages_dir}. Trying alternative based on 'chungoid' package path.")
            try:
                import chungoid
                # Assuming chungoid package is installed and has a __path__
                # This might be fragile depending on how chungoid is packaged/installed.
                chungoid_pkg_root = Path(list(chungoid.__path__)[0]).resolve()
                server_stages_dir = chungoid_pkg_root / "server_prompts" / "stages"
                if not server_stages_dir.is_dir():
                    click.echo(f"Error: Critical - could not locate server_prompts/stages directory. Looked at {core_package_dir / 'server_prompts' / 'stages'} and {server_stages_dir}. Cannot proceed.", err=True)
                    sys.exit(1)
                logger.info(f"Using alternative server_stages_dir: {server_stages_dir}")
            except Exception as e_ssd_fallback:
                logger.error(f"Failed to find server_stages_dir via fallback: {e_ssd_fallback}")
                click.echo(f"Error: Critical - could not locate server_prompts/stages directory. Fallback failed. {e_ssd_fallback}. Cannot proceed.", err=True)
                sys.exit(1)

        # --- State Manager (must be initialized first to load PausedRunDetails) ---
        state_manager = StateManager(
            target_directory=str(project_root),
            server_stages_dir=str(server_stages_dir),
            use_locking=False
        )

        # --- Load PausedRunDetails --- 
        paused_details: Optional[PausedRunDetails] = state_manager.load_paused_flow_state(run_id)
        if not paused_details:
            click.echo(f"Error: No paused run details found for run_id '{run_id}' in project '{project_root}'.", err=True)
            sys.exit(1)
        logger.info(f"Loaded PausedRunDetails for flow_id '{paused_details.flow_id}', paused at '{paused_details.paused_at_stage_id}'")

        # --- Master Flow Registry & Load MasterExecutionPlan ---
        # Assuming master flows are in a 'master_flows' subdirectory of the project
        master_flows_project_dir = project_root / "master_flows"
        master_flow_registry = MasterFlowRegistry(master_flows_dir=master_flows_project_dir)
        
        if master_flow_registry.get_scan_errors():
            logger.warning(f"Errors encountered while scanning for Master Flows in {master_flows_project_dir}:")
            for err in master_flow_registry.get_scan_errors():
                logger.warning(f"  - {err}")
                # click.echo(f"Warning (Master Flow Scan): {err}", err=True) # Optional: echo to user

        master_plan: Optional[MasterExecutionPlan] = master_flow_registry.get(paused_details.flow_id)
        if not master_plan:
            click.echo(f"Error: Master Flow definition with ID '{paused_details.flow_id}' not found in '{master_flows_project_dir}'. Cannot resume.", err=True)
            sys.exit(1)
        logger.info(f"Successfully loaded MasterExecutionPlan '{master_plan.id}' (Name: {master_plan.name or 'N/A'}) for resume.")

        # --- Agent Registry (and register core agents) ---
        agent_registry = AgentRegistry()
        try:
            agent_registry.add(core_stage_executor_card, core_stage_executor_agent)
            logger.info(f"Registered built-in agent: {core_stage_executor_card.agent_id}")
        except Exception as reg_err:
            logger.exception(f"Fatal: Failed to register core agent '{core_stage_executor_card.agent_id}': {reg_err}")
            click.echo(f"Error: Failed to register core executor agent. Cannot proceed.", err=True)
            sys.exit(1)
        
        agent_provider = RegistryAgentProvider(registry=agent_registry)

        # --- Orchestrator --- 
        orchestrator = AsyncOrchestrator(
            pipeline_def=master_plan, # Pass the loaded MasterExecutionPlan
            config={}, # Provide config if needed
            agent_provider=agent_provider,
            state_manager=state_manager
        )

        # --- Perform Resume Action --- 
        click.echo(f"Flow resumption initiated for run_id '{run_id}' with action '{action}'...")
        try:
            # Call the orchestrator's resume_flow method
            # The orchestrator's resume_flow already knows about master_plan via its pipeline_def
            final_context_or_error = await orchestrator.resume_flow(
                run_id=run_id,
                action=action,
                action_data=action_data
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
# Entrypoint for `python -m chungoid.cli` (optional)
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    cli() 