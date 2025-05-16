from __future__ import annotations

import asyncio
import logging
import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, Optional
import os

from chungoid.schemas.agent_system_test_runner import SystemTestRunnerAgentInput, SystemTestRunnerAgentOutput

logger = logging.getLogger(__name__)

AGENT_ID = "system_test_runner_agent"

async def invoke_async(
    inputs: Dict[str, Any],
    full_context: Optional[Dict[str, Any]] = None,
) -> SystemTestRunnerAgentOutput:
    try:
        parsed_inputs = SystemTestRunnerAgentInput(**inputs)
    except Exception as e:
        logger.error(f"Input validation failed for {AGENT_ID}: {e}")
        return SystemTestRunnerAgentOutput(
            exit_code=-1,
            summary=f"Input validation error: {e}",
            status="FAILURE_INPUT_VALIDATION",
        )

    command = ["pytest"]
    if parsed_inputs.pytest_options:
        command.extend(shlex.split(parsed_inputs.pytest_options))
    command.append(parsed_inputs.test_target_path)

    # Determine CWD
    cwd_path: Optional[Path] = None
    if parsed_inputs.project_root_path:
        cwd_path = Path(parsed_inputs.project_root_path)
        if not cwd_path.is_dir():
            logger.warning(f"Provided project_root_path '{cwd_path}' is not a valid directory. Running pytest from default CWD.")
            cwd_path = None
    elif full_context and isinstance(full_context.get('project_config'), dict) and full_context['project_config'].get('project_root'):
        # Try to get from project_config in full_context if available
        try:
            cwd_path = Path(full_context['project_config']['project_root'])
            if not cwd_path.is_dir():
                logger.warning(f"project_config.project_root '{cwd_path}' is not a valid directory. Running pytest from default CWD.")
                cwd_path = None
            else:
                logger.info(f"Using project_root '{cwd_path}' from full_context.project_config as CWD for pytest.")
        except Exception as e:
            logger.warning(f"Error accessing project_root from full_context.project_config: {e}. Running pytest from default CWD.")
            cwd_path = None
    else:
        logger.info("No project_root_path provided and not found in context.project_config.project_root. Running pytest from default CWD.")

    # Prepare environment for subprocess
    env = os.environ.copy()
    if cwd_path:
        src_path = (Path(cwd_path) / "src").resolve()
        if 'PYTHONPATH' in env:
            env['PYTHONPATH'] = f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        else:
            env['PYTHONPATH'] = str(src_path)
        logger.info(f"Modified PYTHONPATH for subprocess: {env['PYTHONPATH']}")
    else:
        logger.info("CWD for pytest is not set, PYTHONPATH not modified for src.")

    try:
        logger.info(f"Running pytest command: {' '.join(command)} in CWD: {cwd_path if cwd_path else 'default'}")
        process = subprocess.run(command, capture_output=True, text=True, cwd=cwd_path, env=env, timeout=300) # Added timeout
        stdout = process.stdout
        stderr = process.stderr
        exit_code = process.returncode
        
        # Extract summary (simple version, might need refinement for complex pytest outputs)
        summary_line = "No summary found"
        if stdout: # Check stdout first for summary
            lines = stdout.strip().split('\n')
            for line in reversed(lines):
                if "===" in line and ("passed" in line or "failed" in line or "error" in line or "skipped" in line):
                    summary_line = line.strip().lstrip("=").rstrip("=") # Trim ===
                    summary_line = summary_line.strip() # Trim extra spaces
                    break
        if summary_line == "No summary found" and stderr: # Fallback to stderr if not in stdout
             lines = stderr.strip().split('\n')
             for line in reversed(lines):
                if "===" in line and ("passed" in line or "failed" in line or "error" in line or "skipped" in line):
                    summary_line = line.strip().lstrip("=").rstrip("=")
                    summary_line = summary_line.strip()
                    break

        logger.info(f"Pytest completed with exit code {exit_code}. Summary: {summary_line}")
        
        if exit_code != 0:
            if stdout:
                logger.info(f"Pytest stdout (exit code {exit_code}):\n{stdout}")
            if stderr:
                logger.info(f"Pytest stderr (exit code {exit_code}):\n{stderr}")
        else: # Successful run
            if stdout:
                logger.debug(f"Pytest stdout:\n{stdout}")
            if stderr: # Log stderr even on success if it contains anything (e.g., warnings)
                logger.debug(f"Pytest stderr (exit code 0):\n{stderr}")


        return SystemTestRunnerAgentOutput(
            exit_code=exit_code,
            summary=summary_line,
            stdout=stdout,
            stderr=stderr,
            status="SUCCESS" if exit_code == 0 else "FAILURE_TESTS_FAILED",
        )

    except FileNotFoundError:
        logger.error("pytest command not found. Ensure pytest is installed and in PATH.")
        return SystemTestRunnerAgentOutput(
            exit_code=-2,
            summary="pytest command not found.",
            status="FAILURE_PYTEST_NOT_FOUND",
        )
    except Exception as e:
        logger.error(f"Error running pytest: {e}", exc_info=True)
        return SystemTestRunnerAgentOutput(
            exit_code=-3,
            summary=f"Error during pytest execution: {e}",
            status="FAILURE_EXECUTION_ERROR",
            stderr=str(e)
        ) 