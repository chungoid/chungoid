from __future__ import annotations

import asyncio
import logging
import subprocess
import shlex
from pathlib import Path
from typing import Dict, Any, Optional, ClassVar
import os
import uuid # Added for task_id default

from chungoid.schemas.agent_system_test_runner import SystemTestRunnerAgentInput, SystemTestRunnerAgentOutput
from chungoid.runtime.agents.agent_base import BaseAgent # Added
from chungoid.utils.agent_registry import AgentCard # Added
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility # Added
from chungoid.schemas.orchestration import SharedContext # Was chungoid.schemas.shared_context

logger = logging.getLogger(__name__) # Keep module-level logger for now, class will also have one

# AGENT_ID = "system_test_runner_agent" # Will be a class attribute

class SystemTestRunnerAgent_v1(BaseAgent[SystemTestRunnerAgentInput, SystemTestRunnerAgentOutput]):
    AGENT_ID: ClassVar[str] = "SystemTestRunnerAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Test Runner Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Runs pytest tests and reports results."
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.TEST_EXECUTION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    def __init__(self, system_context: Optional[Dict[str, Any]] = None, **kwargs):
        if system_context is not None:
            super().__init__(system_context=system_context, **kwargs)
        else:
            super().__init__(**kwargs) # Pass other kwargs like llm_provider, etc.
        logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) v{self.AGENT_VERSION} initialized.")

    async def invoke_async(
        self,
        inputs: SystemTestRunnerAgentInput,
        full_context: Optional[SharedContext] = None,
        project_root: Optional[Path] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any
    ) -> SystemTestRunnerAgentOutput:
        # Parse the input dictionary into the SystemTestRunnerAgentInput model
        if isinstance(inputs, dict):
            try:
                task_input = SystemTestRunnerAgentInput(**inputs)
            except Exception as e:
                logger.error(f"Failed to parse inputs for SystemTestRunnerAgent_v1: {e}", exc_info=True)
                return SystemTestRunnerAgentOutput(
                    exit_code=-4, # Arbitrary new exit code for parsing failure
                    summary=f"Input parsing error: {e}",
                    status="FAILURE_INPUT_PARSING_ERROR",
                    stderr=str(e)
                )
        else: # inputs is already SystemTestRunnerAgentInput
            task_input = inputs
        
        logger.debug(f"{self.AGENT_ID} invoked with inputs: {task_input}")

        command = ["pytest"]
        if task_input.pytest_options:
            command.extend(shlex.split(task_input.pytest_options))
        command.append(task_input.test_target_path)

        # Determine CWD
        cwd_path: Optional[Path] = None
        if project_root:
            cwd_path = Path(project_root)
            if not cwd_path.is_dir():
                logger.warning(f"Provided project_root '{cwd_path}' is not a valid directory. Running pytest from default CWD.")
                cwd_path = None
        elif full_context and hasattr(full_context, 'project_root_path') and full_context.project_root_path:
            try:
                cwd_path = Path(full_context.project_root_path)
                if not cwd_path.is_dir():
                    logger.warning(f"full_context.project_root_path '{cwd_path}' is not a valid directory. Running pytest from default CWD.")
                    cwd_path = None
                else:
                    logger.info(f"Using project_root_path '{cwd_path}' from full_context as CWD for pytest.")
            except Exception as e:
                logger.warning(f"Error accessing project_root_path from full_context: {e}. Running pytest from default CWD.")
                cwd_path = None
        elif full_context and hasattr(full_context, 'global_project_settings') and isinstance(full_context.global_project_settings, dict):
            project_root_from_global = full_context.global_project_settings.get('project_root')
            if project_root_from_global:
                try:
                    cwd_path = Path(project_root_from_global)
                    if not cwd_path.is_dir():
                        logger.warning(f"full_context.global_project_settings['project_root'] '{cwd_path}' is not a valid directory. Running pytest from default CWD.")
                        cwd_path = None
                    else:
                        logger.info(f"Using project_root '{cwd_path}' from full_context.global_project_settings as CWD for pytest.")
                except Exception as e:
                    logger.warning(f"Error accessing project_root from full_context.global_project_settings: {e}. Running pytest from default CWD.")
                    cwd_path = None
            else:
                logger.info("No 'project_root' key in full_context.global_project_settings. Running pytest from default CWD.")
        else:
            logger.info("No project_root_path provided in task_input or found in full_context. Running pytest from default CWD.")

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
            process = subprocess.run(command, capture_output=True, text=True, cwd=cwd_path, env=env, timeout=300)
            stdout = process.stdout
            stderr = process.stderr
            exit_code = process.returncode
            
            summary_line = "No summary found"
            if stdout:
                lines = stdout.strip().split('\n')
                for line in reversed(lines):
                    if "===" in line and ("passed" in line or "failed" in line or "error" in line or "skipped" in line):
                        summary_line = line.strip().lstrip("=").rstrip("=")
                        summary_line = summary_line.strip()
                        break
            if summary_line == "No summary found" and stderr:
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
            else:
                if stdout:
                    logger.debug(f"Pytest stdout:\n{stdout}")
                if stderr:
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

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        return AgentCard(
            agent_id=SystemTestRunnerAgent_v1.AGENT_ID,
            name=SystemTestRunnerAgent_v1.AGENT_NAME,
            description=SystemTestRunnerAgent_v1.AGENT_DESCRIPTION,
            version=SystemTestRunnerAgent_v1.AGENT_VERSION,
            input_schema=SystemTestRunnerAgentInput.model_json_schema(),
            output_schema=SystemTestRunnerAgentOutput.model_json_schema(),
            categories=[
                SystemTestRunnerAgent_v1.CATEGORY.value,
                AgentCategory.AUTONOMOUS_PROJECT_ENGINE.value
            ],
            visibility=SystemTestRunnerAgent_v1.VISIBILITY.value,
            capability_profile={
                "supported_frameworks": ["pytest"],
                "report_format": "structured_json_output_model",
                "execution_environment": "local_subprocess"
            },
            metadata={
                "callable_fn_path": f"{SystemTestRunnerAgent_v1.__module__}.{SystemTestRunnerAgent_v1.__name__}"
            }
        ) 