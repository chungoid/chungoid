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

        # Handle command override vs default pytest
        if task_input.test_command_override:
            # Use override command
            command = list(task_input.test_command_override)  # Copy to avoid modifying original
            if task_input.test_command_args:
                command.extend(task_input.test_command_args)
            logger.info(f"Using command override: {' '.join(command)}")
        else:
            # Default pytest mode
            if not task_input.test_target_path:
                logger.error("test_target_path is required when not using test_command_override")
                return SystemTestRunnerAgentOutput(
                    exit_code=-5,
                    summary="test_target_path is required when not using test_command_override",
                    status="FAILURE_INPUT_VALIDATION_ERROR",
                    stderr="test_target_path is required when not using test_command_override"
                )
            
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
        elif full_context and isinstance(full_context.data, dict) and full_context.data.get('project_root_path'): # Check full_context.data
            try:
                raw_path_from_context = full_context.data.get('project_root_path') # CORRECTED ACCESS
                self._logger.info(f"DEBUG: full_context.data.get('project_root_path') is '{raw_path_from_context}' (type: {type(raw_path_from_context)})")
                
                candidate_path = Path(raw_path_from_context)
                self._logger.info(f"DEBUG: Path(full_context.data.get('project_root_path')) is '{candidate_path}' (type: {type(candidate_path)})")
                
                is_dir_check = candidate_path.is_dir()
                self._logger.info(f"DEBUG: candidate_path.is_dir() result: {is_dir_check}")

                if is_dir_check:
                    cwd_path = candidate_path.resolve() # Resolve to absolute path
                    logger.info(f"Using project_root_path '{cwd_path}' from full_context as CWD for pytest.")
                else:
                    logger.warning(f"full_context.data.get('project_root_path') '{candidate_path}' is not a valid directory. Running pytest from default CWD.")
                    cwd_path = None
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
        
        # Get current PYTHONPATH or an empty string if it's not set
        current_python_path = env.get('PYTHONPATH', '')
        additional_paths = []

        # Add the project's src directory if cwd_path is set
        if cwd_path:
            project_src_path = (cwd_path / "src").resolve()
            if project_src_path.is_dir():
                additional_paths.append(str(project_src_path))
                logger.info(f"Adding project-specific src to PYTHONPATH: {project_src_path}")
            else:
                logger.warning(f"Project-specific src directory not found: {project_src_path}")

        # Add the workspace's src directory
        workspace_root_path_str = None
        if full_context and hasattr(full_context, 'data') and isinstance(full_context.data, dict):
            workspace_root_path_str = full_context.data.get('mcp_root_workspace_path')

        if workspace_root_path_str:
            # It seems the original log showed chungoid-core/src. Let's ensure both core and project src are handled if distinct.
            # For now, let's assume 'mcp_root_workspace_path' is the absolute root, and 'src' directly under it is one to add.
            # And if chungoid-core is separate, that might need specific handling or be part of a broader strategy.
            # The immediate fix is for fresh_test/src.
            
            # Add MCP workspace's main src (e.g., /home/flip/Desktop/chungoid-mcp/src)
            mcp_workspace_src_path = (Path(workspace_root_path_str) / "src").resolve()
            if mcp_workspace_src_path.is_dir() and str(mcp_workspace_src_path) not in additional_paths:
                 additional_paths.append(str(mcp_workspace_src_path))
                 logger.info(f"Adding MCP workspace src to PYTHONPATH: {mcp_workspace_src_path}")
            
            # Add chungoid-core/src as it contains schema definitions etc.
            chungoid_core_src_path = (Path(workspace_root_path_str) / "chungoid-core" / "src").resolve()
            if chungoid_core_src_path.is_dir() and str(chungoid_core_src_path) not in additional_paths:
                additional_paths.append(str(chungoid_core_src_path))
                logger.info(f"Adding chungoid-core src to PYTHONPATH: {chungoid_core_src_path}")
            else:
                logger.warning(f"chungoid-core src directory not found ({chungoid_core_src_path}) or already added to PYTHONPATH.")

        # Construct the new PYTHONPATH
        # Prepend additional paths, then add the original PYTHONPATH
        if additional_paths:
            new_python_path_parts = additional_paths
            if current_python_path:
                new_python_path_parts.append(current_python_path)
            env['PYTHONPATH'] = os.pathsep.join(new_python_path_parts)
            logger.info(f"PYTHONPATH for subprocess set to: {env['PYTHONPATH']}")
        elif not current_python_path: # No additional paths and no existing PYTHONPATH
             logger.info("PYTHONPATH not set and no project/workspace src paths found to add.")
        else: # No additional paths, but existing PYTHONPATH is kept
            logger.info(f"Using existing PYTHONPATH: {current_python_path}")

        try:
            command_name = "pytest" if not task_input.test_command_override else task_input.test_command_override[0]
            logger.info(f"Running {command_name} command: {' '.join(command)} in CWD: {cwd_path if cwd_path else 'default'}")
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

            logger.info(f"{command_name} completed with exit code {exit_code}. Summary: {summary_line}")
            
            if exit_code != 0:
                if stdout:
                    logger.info(f"{command_name} stdout (exit code {exit_code}):\n{stdout}")
                if stderr:
                    logger.info(f"{command_name} stderr (exit code {exit_code}):\n{stderr}")
            else:
                if stdout:
                    logger.debug(f"{command_name} stdout:\n{stdout}")
                if stderr:
                    logger.debug(f"{command_name} stderr (exit code 0):\n{stderr}")

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