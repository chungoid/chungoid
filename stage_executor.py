"""Handles the execution of individual project stages."""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

# Assuming these utilities are in the 'utils' directory
# Adjust imports based on actual final project structure
try:
    from utils.state_manager import StateManager
    from utils.prompt_manager import PromptManager, PromptLoadError

    # Removed import of non-existent ToolAdapter class
    # Tool execution functions like execute_tool can be imported if needed
    # from utils.tool_adapters import ToolAdapter
    # Import specific security functions and errors instead of non-existent class
    from application_security import (
        run_safe_subprocess,
        validate_input,
        sanitize_filepath,
        SecurityError,
        InputValidationError,
        CommandInjectionError,
    )

    # DecisionFramework will be implemented in Stage 5.5
    # from utils.decision_framework import DecisionFramework
except ImportError as e:
    logging.error(f"Failed to import utility modules: {e}. Ensure utils directory is accessible.")

    # Define dummy classes/functions if imports fail to allow basic structure loading
    class StateManager:
        pass

    class PromptManager:
        pass

    class PromptLoadError(Exception):
        pass

    # Removed dummy ToolAdapter class
    # class ToolAdapter:
    #     pass

    # Add dummy security functions/errors if import fails
    def run_safe_subprocess(*args, **kwargs):
        logging.warning("Using dummy run_safe_subprocess")
        return 0, "dummy stdout", "dummy stderr"

    def validate_input(*args, **kwargs):
        logging.warning("Using dummy validate_input")
        return True

    def sanitize_filepath(filepath):
        logging.warning("Using dummy sanitize_filepath")
        return filepath

    class SecurityError(Exception):
        pass

    class InputValidationError(SecurityError):
        pass

    class CommandInjectionError(SecurityError):
        pass

    # class DecisionFramework: pass


# Placeholder for DecisionFramework until Stage 5.5
class DecisionFramework:
    def __init__(self, *args, **kwargs):
        logging.warning("Using placeholder DecisionFramework.")

    def decide(self, *args, **kwargs):
        logging.warning("Placeholder DecisionFramework decide called.")
        return {"decision": "placeholder_decision"}


class StageExecutionError(Exception):
    """Custom exception for errors during stage execution."""

    def __init__(self, message: str, details: Optional[Any] = None):
        super().__init__(message)
        self.details = details


class StageExecutor:
    """Orchestrates the execution of a single project stage.

    Reads prompts, interacts with tools (via adapters or direct calls),
    makes decisions (via framework), and manages state updates.
    """

    # Define expected stage sequence (should ideally match the master plan)
    _STAGE_SEQUENCE = [0, 0.5, 1, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5]

    def __init__(
        self,
        state_manager: StateManager,
        decision_logic_config: Dict[str, Any],
        project_root_dir: Union[str, Path],
        reflections_summary: str,
        server_stages_dir: str,
        common_template_path: str,
    ):
        """Initializes the StageExecutor.

        Args:
            state_manager: Instance of StateManager.
            decision_logic_config: Configuration for the DecisionFramework.
            project_root_dir: The root directory of the target project.
            reflections_summary: Summary of past reflections to inject into the prompt.
            server_stages_dir: Absolute path to the server's stage template directory.
            common_template_path: Absolute path to the server's common template file.
        """
        self.state_manager = state_manager
        self.project_root_dir = Path(project_root_dir).resolve()
        self.reflections_summary = reflections_summary  # Store reflections summary
        self.logger = logging.getLogger(__name__)

        try:
            # Initialize PromptManager internally using the NEW server-specific path
            self.prompt_manager = PromptManager(
                server_stages_dir=server_stages_dir,  # Pass server stages dir
                common_template_path=common_template_path,  # <<< Pass common template path
            )
        except (PromptLoadError, ValueError) as e:
            self.logger.critical("Failed to initialize PromptManager: %s", e)
            raise RuntimeError(f"StageExecutor could not initialize PromptManager: {e}") from e

        # Initialize DecisionFramework internally
        self.decision_framework = DecisionFramework(config=decision_logic_config)
        self.last_result: Optional[Dict[str, Any]] = None
        self.logger.info("StageExecutor initialized.")

    async def execute_next_stage_async(self) -> Dict[str, Any]:
        """Determines the next stage and executes its logic asynchronously.

        Returns:
            A dictionary indicating the status and result of the execution attempt.
            Includes keys like 'status', 'message', 'stage_executed', 'result'.
        """
        self.logger.info("Attempting to execute next stage...")
        next_stage = None  # Define outside try block
        try:
            next_stage = self.state_manager.get_next_stage()

            if next_stage is None:
                # Handle project completion or failure state
                return self._handle_no_next_stage()

            self.logger.info("Executing Stage %s...", next_stage)

            # --- Load and Render Prompt using YAML Manager --- MODIFIED LOGIC ---
            try:
                # --- REMOVED Table of Contents Generation ---
                # toc_summary = ""
                # try:
                #     toc_summary = self.prompt_manager.get_toc_summary()
                #     if toc_summary:
                #         toc_summary += "\\n\\n---\\n\\n" # Add separators
                #     self.logger.debug("Generated TOC summary.")
                # except AttributeError:
                #     self.logger.warning("PromptManager does not have 'get_toc_summary'. Skipping TOC.")
                # except Exception as toc_err:
                #     self.logger.warning(f"Error generating TOC summary: {toc_err}. Skipping TOC.")
                # --- End Generate Table of Contents ---

                context_for_render = {"reflections_summary": self.reflections_summary}
                # Potentially add more dynamic context here if needed

                stage_specific_prompt = self.prompt_manager.get_rendered_prompt(
                    stage_number=next_stage, context_data=context_for_render
                )

                # REMOVED Prepending TOC
                full_prompt = stage_specific_prompt

                self.logger.info("Prompt for stage %s rendered successfully.", next_stage)
                self.logger.debug(
                    "Full prompt for stage %s prepared (%d chars).",
                    next_stage,
                    len(full_prompt),
                )

            except PromptLoadError as e:
                self.logger.error("Failed to load/render prompt for stage %s: %s", next_stage, e)
                # Return error immediately if prompt loading fails
                return {
                    "status": "ERROR",
                    "message": f"Prompt load/render error: {e}",
                }
            # --- End Prompt Loading/Rendering ---

            # --- Execute Stage Specific Logic ---
            # Pass the fully rendered prompt to the logic execution
            execution_result = await self._execute_stage_logic_async(
                next_stage,
                full_prompt,  # Pass rendered prompt here
            )

            # --- Format Success Response ---
            response = {
                "status": "SUCCESS",
                "stage_executed": next_stage,
                "result": execution_result,
            }
            self.last_result = response
            self.logger.info("Stage %s execution attempt completed: SUCCESS", next_stage)
            return response

        except StageExecutionError as e:  # Catch errors during actual stage execution logic
            self.logger.exception("Error during execution of stage %s: %s", next_stage, e)
            self.last_result = {
                "status": "STAGE_ERROR",
                "stage": next_stage,
                "message": f"Execution error: {e}",
                "details": e.details,
            }
            return self.last_result
        except Exception as e:  # Catch unexpected errors (e.g., in state manager)
            stage_str = next_stage if next_stage is not None else "unknown"
            self.logger.exception(
                "Unexpected internal error during stage %s execution: %s", stage_str, e
            )
            self.last_result = {
                "status": "INTERNAL_ERROR",
                "message": f"Unexpected error: {e}",
            }
            return self.last_result

    def _handle_no_next_stage(self) -> Dict[str, Any]:
        """Handles the case where state_manager.get_next_stage() returns None."""
        last_info = self.state_manager.get_last_status()
        if last_info:
            last_stage_num = last_info.get("stage")
            last_stage_status = last_info.get("status")
            # Check if the last completed stage is the final one in the sequence
            if last_stage_status in ["DONE", "PASS"] and last_stage_num == self._STAGE_SEQUENCE[-1]:
                msg = "Project already completed."
                self.logger.info(msg)
                self.last_result = {"status": "COMPLETE", "message": msg}
            elif last_stage_status == "FAIL":
                msg = f"Project failed at stage {last_stage_num}. Cannot proceed."
                self.logger.warning(msg)
                self.last_result = {
                    "status": "BLOCKED",
                    "message": msg,
                    "failed_stage": last_stage_num,
                }
            else:
                # Should not happen if sequence is correct and status is valid
                msg = f"Cannot determine next stage after stage {last_stage_num} (Status: {last_stage_status})."
                self.logger.error(msg)
                self.last_result = {"status": "ERROR", "message": msg}
        else:
            # Status file might be empty or unreadable
            msg = "Cannot determine next stage. Status file might be empty or invalid."
            self.logger.error(msg)
            self.last_result = {"status": "ERROR", "message": msg}
        return self.last_result

    async def _execute_stage_logic_async(
        self, stage_number: Union[int, float], full_rendered_prompt: str
    ) -> Dict[str, Any]:
        """Executes the specific logic for the given stage number.

        Args:
            stage_number: The stage number (e.g., 0.5, 1, 2).
            full_rendered_prompt: The fully rendered prompt string for this stage.

        Returns:
            A dictionary containing the results of the stage execution,
            including output and artifacts generated.

        Raises:
            StageExecutionError: If specific stage logic fails.
            NotImplementedError: If the logic for the stage number is not implemented.
        """
        self.logger.info(f"Executing specific logic for Stage {stage_number}...")
        project_root = Path(self.project_root_dir)  # Ensure it's a Path object
        dev_docs_dir = project_root / "dev-docs"
        artifacts_generated = []
        execution_output = f"Stage {stage_number} started.\n"
        stage_status_to_set = "DONE"  # Default success status
        stage_fail_reason = None  # Default no fail reason
        result = None  # Initialize result to None before the try block

        # Ensure dev-docs exists as many stages use it
        dev_docs_dir.mkdir(parents=True, exist_ok=True)

        # --- Stage-Specific Logic ---

        try:
            if stage_number == 0:
                # Stage 0: Goal Refinement - Request Generation from Agent (Aligned with generic tool)
                self.logger.info("Stage 0: Requesting goal refinement generation from agent.")
                goal_file_rel = "goal.txt"  # Relative path for consistency
                target_file_path = (
                    project_root / goal_file_rel
                )  # Keep absolute path for potential internal use

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": "Agent needs to generate the refined goal using the provided prompt.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": [goal_file_rel],  # Use expected_artifacts list
                    "target_file": str(target_file_path),  # Keep target_file for context if useful
                    "next_tool_hint": "submit_stage_artifacts",  # Use the generic tool hint
                    "stage_number_for_tool": stage_number,  # Add stage number for the tool
                    "artifacts": [],
                }

            elif stage_number == 0.5:
                # Stage 0.5: Research-Advisor - Request Generation from Agent
                self.logger.info(
                    "Stage 0.5: Requesting research/library list generation from agent."
                )
                libraries_file_rel = "dev-docs/planning/stage0.5_libraries.txt"
                target_file_path = project_root / libraries_file_rel

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate research results for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": [libraries_file_rel],  # List expected relative paths
                    "target_file": str(
                        target_file_path
                    ),  # Keep single target for simplicity? Revisit.
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 1.0:
                # Stage 1.0: Blueprint-Architect - Request Generation from Agent
                self.logger.info("Stage 1.0: Requesting blueprint generation from agent.")
                blueprint_file_rel = "dev-docs/architecture/blueprint.txt"
                target_file_path = project_root / blueprint_file_rel

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate the project blueprint for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": [blueprint_file_rel],
                    "target_file": str(target_file_path),
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 2.0:
                # Stage 2.0: Blueprint-Validator - Request Generation from Agent
                self.logger.info("Stage 2.0: Requesting blueprint validation report from agent.")
                report_file_rel = "dev-docs/architecture/validation_report.json"
                # Instruct agent to determine PASS/FAIL status and potentially include patched blueprint
                target_file_path = project_root / report_file_rel

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate the validation report (including PASS/FAIL status) for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": [report_file_rel],  # Agent might add blueprint if patched
                    "target_file": str(target_file_path),
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "requires_status": True,  # Indicate agent should provide status (PASS/FAIL)
                    "artifacts": [],
                }

            elif stage_number == 2.5:
                # Stage 2.5: Development-Coordinator-Designer - Request Generation
                self.logger.info("Stage 2.5: Requesting coordination design from agent.")
                coord_file_rel = "dev-docs/architecture/component_coordination.json"
                readme_file_rel = "dev-docs/architecture/architecture_readme.md"
                expected_artifacts_list = [coord_file_rel, readme_file_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate coordination artifacts for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,  # No single primary target
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 3.0:
                # Stage 3.0: Stage-Planner - Request Generation
                self.logger.info("Stage 3.0: Requesting stage plan generation from agent.")
                index_file_rel = "dev-docs/planning/stages.txt"
                plans_dir_rel = "dev-docs/planning/stage_plans/"  # Indicate directory expectation?
                # Agent needs to generate index + individual plan files inside the dir
                expected_artifacts_list = [index_file_rel, plans_dir_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate the master stage plan and individual stage plans for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 4.0:
                # Stage 4.0: Implementation-Designer - Request Generation
                self.logger.info(
                    "Stage 4.0: Requesting implementation notes generation from agent."
                )
                notes_dir_rel = "dev-docs/planning/implementation_notes/"
                # Agent generates notes files inside this dir
                expected_artifacts_list = [notes_dir_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate implementation notes for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 4.5:
                # Stage 4.5: Tool-Integration-Designer - Request Generation
                self.logger.info("Stage 4.5: Requesting tool integration design from agent.")
                schema_file_rel = "dev-docs/tools/tools_schema.json"
                readme_file_rel = "dev-docs/tools/tools_readme.md"
                adapters_file_rel = "tool_adapters.py"  # In project root
                expected_artifacts_list = [schema_file_rel, readme_file_rel, adapters_file_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate tool integration artifacts for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 5.0:
                # Stage 5.0: Code-Implementer - Request Generation
                self.logger.info("Stage 5.0: Requesting code implementation from agent.")
                src_dir_rel = "src/"  # Directory target
                # Agent generates code files inside this dir
                expected_artifacts_list = [src_dir_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate code implementation for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 5.5:
                # Stage 5.5: Decision-Logic-Framework-Implementer - Request Generation
                self.logger.info("Stage 5.5: Requesting decision logic implementation from agent.")
                framework_file_rel = "decision_framework.py"
                test_file_rel = "decision_test.py"
                readme_file_rel = "dev-docs/decision_logic/decision_readme.md"
                expected_artifacts_list = [framework_file_rel, test_file_rel, readme_file_rel]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate decision logic artifacts for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 6.0:
                # Stage 6.0: QA-Runner - Request Generation
                self.logger.info("Stage 6.0: Requesting QA report generation from agent.")
                report_file_rel = "dev-docs/qa_and_security/qa_report.json"
                target_file_path = project_root / report_file_rel

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate the QA report (including PASS/FAIL status) for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": [report_file_rel],
                    "target_file": str(target_file_path),
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "requires_status": True,  # Indicate agent should provide status (PASS/FAIL)
                    "artifacts": [],
                }

            elif stage_number == 6.5:
                # Stage 6.5: Application-Security-Validator - Request Generation
                self.logger.info("Stage 6.5: Requesting security validation artifacts from agent.")
                app_sec_file_rel = "application_security.py"
                sec_test_file_rel = "security_tests.py"
                readme_file_rel = "dev-docs/qa_and_security/security_readme.md"
                report_file_rel = "dev-docs/qa_and_security/security_report.json"
                expected_artifacts_list = [
                    app_sec_file_rel,
                    sec_test_file_rel,
                    readme_file_rel,
                    report_file_rel,
                ]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate security artifacts and report (including DONE/FAIL status) for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "requires_status": True,  # Indicate agent should provide status (DONE/FAIL)
                    "artifacts": [],
                }

            elif stage_number == 7.0:
                # Stage 7.0: Release-Manager - Request Generation
                self.logger.info("Stage 7.0: Requesting release artifacts from agent.")
                notes_file_rel = "dev-docs/release/release_notes.md"
                readme_file_rel = "README.md"
                docs_dir_rel = "dev-docs/release/docs/"
                setup_file_rel = "pyproject.toml"  # Assuming this, adjust if needed
                expected_artifacts_list = [
                    notes_file_rel,
                    readme_file_rel,
                    docs_dir_rel,
                    setup_file_rel,
                ]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate release artifacts for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            elif stage_number == 7.5:
                # Stage 7.5: Application-Performance-Evaluator - Request Generation
                self.logger.info(
                    "Stage 7.5: Requesting performance evaluation artifacts from agent."
                )
                eval_py_file_rel = "performance_evaluation.py"
                run_py_file_rel = "run_evaluation.py"
                report_file_rel = "evaluation_report.md"
                benchmarks_dir_rel = (
                    "evaluation_benchmarks/"  # Agent needs to create benchmark.dat inside
                )
                expected_artifacts_list = [
                    eval_py_file_rel,
                    run_py_file_rel,
                    report_file_rel,
                    benchmarks_dir_rel,
                ]

                return {
                    "status": "GENERATION_REQUIRED",
                    "message": f"Agent needs to generate performance evaluation artifacts for stage {stage_number}.",
                    "prompt": full_rendered_prompt,
                    "expected_artifacts": expected_artifacts_list,
                    "target_file": None,
                    "next_tool_hint": "submit_stage_artifacts",
                    "stage_number_for_tool": stage_number,
                    "artifacts": [],
                }

            else:
                # If the stage number doesn't match any implemented logic
                raise NotImplementedError(f"Stage {stage_number} logic is not implemented.")

            # Assign the final result dictionary if the try block completes successfully
            result = {
                "status": stage_status_to_set,  # Use the status determined by the logic
                "message": stage_fail_reason if stage_status_to_set == "FAIL" else execution_output,
                "artifacts": artifacts_generated,
            }

        except Exception as stage_err:
            # If any error occurs during the stage logic, mark as FAIL
            self.logger.exception(f"Error during stage {stage_number} execution logic: {stage_err}")
            # Capture the fail reason for the finally block
            stage_fail_reason = f"Stage execution error: {stage_err}"
            # Raise the specific error to be caught by the outer handler,
            # including the artifacts gathered so far.
            raise StageExecutionError(
                stage_fail_reason, details={"artifacts_before_error": artifacts_generated}
            ) from stage_err

        finally:
            # Always request a status update, regardless of success or failure
            self.logger.debug("Entered finally block for stage %s", stage_number)

            # Determine status, artifacts, and reason based on whether `result` was assigned
            if result is not None:
                # Try block completed successfully (or failed late but assigned result)
                self.logger.debug("Finally: 'result' variable was assigned.")
                current_status = result.get("status", "FAIL")  # Default to FAIL if key missing
                artifacts = result.get("artifacts", [])
                reason = result.get("message") if current_status == "FAIL" else None
            else:
                # Try block failed before `result` was assigned
                self.logger.debug("Finally: 'result' variable was None (likely early error).")
                current_status = "FAIL"
                artifacts = artifacts_generated  # Use artifacts gathered before the error
                # Use the fail reason captured in the except block if available, else generic message
                reason = (
                    stage_fail_reason
                    if "stage_fail_reason" in locals()
                    else "Unknown error occurred before stage completion"
                )

            self.logger.debug("Finally: Determined current_status: %s", current_status)
            self.logger.debug("Finally: Determined artifacts: %s", artifacts)
            self.logger.debug("Finally: Determined reason: %s", reason)

            # Ensure artifacts is a list of strings
            if not isinstance(artifacts, list) or not all(isinstance(a, str) for a in artifacts):
                self.logger.warning(
                    "Stage %s returned artifacts in non-standard format: %s. Coercing to list.",
                    stage_number,
                    artifacts,
                )
                # Coerce carefully, handle potential non-list artifacts_generated from error case
                if artifacts:
                    artifacts = list(
                        map(str, artifacts if isinstance(artifacts, list) else [artifacts])
                    )
                else:
                    artifacts = []
                self.logger.debug("Finally: Coerced artifacts: %s", artifacts)

            self.logger.info(
                "Requesting status update for stage %s to %s", stage_number, current_status
            )
            # Add a debug log right before the call
            self.logger.debug("Calling self.request_status_update...")
            update_success = self.request_status_update(
                confirmed_stage=stage_number,
                confirmed_status=current_status,
                artifacts=artifacts,
                reason=reason,
            )
            self.logger.debug("self.request_status_update returned: %s", update_success)
            if not update_success:
                self.logger.error(
                    "StateManager failed to update status for stage %s.", stage_number
                )
                # Optionally modify the final result if status update fails
                # Check result exists before modifying
                if result and result.get("status") != "error":  # Don't overwrite existing errors
                    result["status"] = "warning"
                    result["message"] = (
                        result.get("message", "")
                        + " (Warning: Failed to update project status file)"
                    )
                    self.logger.debug("Finally: Modified result due to update failure: %s", result)

        # Return the original result (potentially modified if update failed)
        # Check result exists before returning
        if result is None:
            # This should technically not be reachable if StageExecutionError is always raised
            # but as a fallback, return a generic error structure.
            self.logger.error("Exiting _execute_stage_logic_async but 'result' is still None.")
            return {
                "status": "FAIL",
                "message": stage_fail_reason
                if "stage_fail_reason" in locals()
                else "Unknown error, result not generated",
                "artifacts": artifacts_generated,
            }

        self.logger.debug("Exiting _execute_stage_logic_async, returning: %s", result)
        return result

    def request_status_update(
        self,
        confirmed_stage: Union[int, float],
        confirmed_status: str,
        artifacts: list[str],
        reason: Optional[str] = None,
    ) -> bool:
        """Public method to request a status update from the StateManager.

        This should typically be called by the main server loop after receiving user
        confirmation or post-execution analysis.

        Args:
            confirmed_stage: The stage number that was executed.
            confirmed_status: The final status ('DONE', 'FAIL', 'PASS').
            artifacts: List of artifacts generated by the stage.
            reason: Optional reason for failure.

        Returns:
            Boolean result from StateManager.update_status.
        """
        self.logger.info(
            "Requesting status update for stage %s to %s with %d artifacts.",
            confirmed_stage,
            confirmed_status,
            len(artifacts),
        )
        if not self.state_manager:
            self.logger.error("StateManager not available to update status.")
            return False
        try:
            success = self.state_manager.update_status(
                stage=confirmed_stage,
                status=confirmed_status,
                artifacts=artifacts,
                reason=reason,
            )
            if success:
                self.logger.info(
                    "StateManager successfully updated status for stage %s.",
                    confirmed_stage,
                )
            else:
                self.logger.error(
                    "StateManager failed to update status for stage %s.",
                    confirmed_stage,
                )
            return success
        except Exception as e:
            self.logger.exception(
                "Error calling StateManager update for stage %s: %s", confirmed_stage, e
            )
            return False

    def _run_verification_command(self, command: str):
        # This method is mentioned in the original code but not implemented in the new version
        # It's left unchanged as it's not clear if it's still needed or if it should be implemented
        pass


# Example of how it might be used (requires async context)
# import asyncio
# if __name__ == '__main__':
#     async def run_test():
#         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
#         logger = logging.getLogger(__name__)
#         logger.info("Setting up mock components for StageExecutor test...")
#
#         # Simplified setup for example
#         test_status_file = './temp_executor_test_status.json'
#         # ... (cleanup and setup as before) ...
#         mock_state_manager = StateManager(test_status_file)
#         mock_decision_framework = DecisionFramework(config={})
#         mock_prompt_template_dir = './templates' # Ensure exists
#         mock_stage_prompt_dir = './temp_stages'   # Ensure exists with files
#
#         executor = StageExecutor(
#             state_manager=mock_state_manager,
#             prompt_template_dir=mock_prompt_template_dir,
#             stage_prompt_dir=mock_stage_prompt_dir,
#             decision_logic_config={}
#         )
#
#         # Execute first stage
#         result = await executor.execute_next_stage_async()
#         print(f"\nExecution Result 1: {result}")
#
#         # Manually update status (simulating external confirmation)
#         if result.get('status') == 'SUCCESS':
#             update_success = executor.request_status_update(
#                 confirmed_stage=result['stage_executed'],
#                 confirmed_status='DONE',
#                 artifacts=result['result']['artifacts_generated']
#             )
#             print(f"Status update success: {update_success}")
#
#         # Execute next stage
#         result2 = await executor.execute_next_stage_async()
#         print(f"\nExecution Result 2: {result2}")
#
#         # ... cleanup ...
#
#     asyncio.run(run_test())
