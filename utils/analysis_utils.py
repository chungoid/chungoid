"""Utilities for running static analysis tools like Ruff and Bandit."""

import logging
import json
import subprocess
from typing import List, Dict, Any, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Define standard report structure
AnalysisIssue = Dict[
    str, Any
]  # e.g., {"code": "F841", "filename": "...", "line_number": ..., "message": "...", "severity": "LOW|MEDIUM|HIGH"(Bandit)}
AnalysisReport = Dict[
    str, Dict[str, Any]
]  # e.g., {"ruff": {"status": "PASS"|"FAIL", "issues": [...]}, "bandit": {"status": "PASS"|"FAIL", "issues": [...]}}


def _run_analysis_tool(
    tool_name: str, command_args: List[str], target_path_str: str
) -> Tuple[bool, List[AnalysisIssue]]:
    """Helper to run an analysis tool as a subprocess and parse JSON output."""
    command = command_args + [target_path_str]
    logger.info(f"Running {tool_name} analysis: {' '.join(command)}")
    issues: List[AnalysisIssue] = []
    success = False  # Assume failure initially
    try:
        # Using subprocess.run for simplicity; consider Popen for more control
        # Capture stdout and stderr, run in UTF-8
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,  # Don't throw exception on non-zero exit code, handle it manually
        )

        # Log stderr for debugging potential tool errors
        if process.stderr:
            logger.warning(f"{tool_name} stderr: {process.stderr.strip()}")

        if process.returncode == 0:
            # Tool executed successfully, potentially with findings
            logger.info(f"{tool_name} completed successfully (exit code 0). Parsing output...")
            try:
                # Ruff and Bandit (with -f json) output JSON to stdout
                parsed_output = json.loads(process.stdout)

                # --- Adapt parsing based on tool's JSON structure ---
                if tool_name == "ruff" and isinstance(parsed_output, list):
                    issues = parsed_output  # Ruff's JSON format might be a list of issues directly
                    # Assuming Ruff success means no errors found OR exit code was 0
                    success = not bool(issues)  # Pass if no issues
                    logger.info(f"Ruff found {len(issues)} issues.")

                elif tool_name == "bandit" and isinstance(parsed_output, dict):
                    issues = parsed_output.get("results", [])
                    # Bandit exit code 0 means execution success, not necessarily no findings
                    # Define success based on severity or just completion?
                    # For now, assume PASS if executed, FAIL only on tool error.
                    # More sophisticated logic could filter by severity/confidence.
                    success = True  # Bandit PASS means it ran okay
                    logger.info(f"Bandit found {len(issues)} issues.")
                else:
                    logger.error(
                        f"Unexpected JSON structure from {tool_name}. Output: {process.stdout[:500]}..."
                    )
                    # Still treat as execution success if exit code was 0
                    success = True
            except json.JSONDecodeError:
                logger.error(
                    f"Failed to decode JSON output from {tool_name}. Output: {process.stdout[:500]}..."
                )
                # Consider this a failure even if exit code was 0
                success = False
        else:
            # Tool failed to execute properly
            logger.error(f"{tool_name} execution failed with exit code {process.returncode}.")
            success = False

    except FileNotFoundError:
        logger.error(
            f"{tool_name} command not found. Is it installed and in PATH? Command: {command[0]}"
        )
        success = False
    except Exception as e:
        logger.exception(f"Unexpected error running {tool_name}: {e}")
        success = False

    # Normalize issue structure if needed (e.g., add severity for Ruff)
    for issue in issues:
        if tool_name == "ruff":
            issue["severity"] = "UNKNOWN"  # Or map Ruff codes to severity
        elif tool_name == "bandit":
            # Bandit's severity might already be present, keep it
            pass

    logger.info(f"{tool_name} analysis result: Success={success}, Issues={len(issues)}")
    return success, issues


def run_ruff_analysis(target_path: Path) -> Tuple[bool, List[AnalysisIssue]]:
    """Runs Ruff on a file/directory and returns PASS/FAIL and issues."""
    # Command: ruff check path/to/target --output-format=json --exit-zero
    # Use --exit-zero so exit code 0 doesn't mean "no issues found", just "ran successfully"
    command = ["ruff", "check", "--output-format=json", "--exit-zero"]
    return _run_analysis_tool("ruff", command, str(target_path))


def run_bandit_analysis(target_path: Path) -> Tuple[bool, List[AnalysisIssue]]:
    """Runs Bandit on a file/directory and returns PASS/FAIL and issues."""
    # Command: bandit -r path/to/target -f json -q
    # -r for recursive, -f json for format, -q for quiet (less verbose logging)
    command = ["bandit", "-r", "-f", "json", "-q"]
    # Note: Bandit's exit code often indicates if issues *were found*, not just execution success.
    # The helper function needs to handle this based on the exit code check.
    # Re-evaluating: Let helper handle non-zero as failure for now. Can refine later.
    return _run_analysis_tool("bandit", command, str(target_path))


def generate_analysis_report(
    ruff_success: bool,
    ruff_issues: List[AnalysisIssue],
    bandit_success: bool,
    bandit_issues: List[AnalysisIssue],
) -> AnalysisReport:
    """Combines issues into a standard report format."""
    # Determine overall status - FAIL if any tool failed execution or found critical issues (if we define critical)
    # Simple: FAIL if either execution failed
    overall_status = "PASS" if ruff_success and bandit_success else "FAIL"
    logger.info(
        f"Generating analysis report. Ruff Success: {ruff_success}, Bandit Success: {bandit_success}. Overall Status: {overall_status}"
    )

    report = {
        "ruff": {"status": "PASS" if ruff_success else "FAIL", "issues": ruff_issues},
        "bandit": {"status": "PASS" if bandit_success else "FAIL", "issues": bandit_issues},
        "overall_status": overall_status,
    }
    return report
