import pytest
import subprocess
import json
from unittest.mock import patch, MagicMock
from pathlib import Path

from chungoid.utils.analysis_utils import (
    _run_analysis_tool,
    run_ruff_analysis,
    run_bandit_analysis,
    generate_analysis_report,
    AnalysisIssue
)

# Sample valid outputs for mocking subprocess
VALID_RUFF_OUTPUT_NO_ISSUES = "[]"
VALID_RUFF_OUTPUT_WITH_ISSUES = json.dumps([
    {"code": "F841", "filename": "file.py", "line_number": 10, "message": "Unused var"}
])
VALID_BANDIT_OUTPUT_NO_ISSUES = json.dumps({"results": []})
VALID_BANDIT_OUTPUT_WITH_ISSUES = json.dumps({
    "results": [
        {"code": "B101", "filename": "file.py", "line_number": 20, "issue_severity": "LOW", "issue_text": "Assert test"}
    ]
})

@patch('subprocess.run')
def test_run_analysis_tool_ruff_success_no_issues(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = VALID_RUFF_OUTPUT_NO_ISSUES
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("ruff", ["ruff", "check"], "target/path")
    assert success is True
    assert len(issues) == 0
    mock_subproc_run.assert_called_once_with(["ruff", "check", "target/path"], capture_output=True, text=True, encoding='utf-8', check=False)

@patch('subprocess.run')
def test_run_analysis_tool_ruff_success_with_issues(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = VALID_RUFF_OUTPUT_WITH_ISSUES
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("ruff", ["ruff", "check"], "target/path")
    assert success is False # Ruff success means no issues found if exit code is 0 (after --exit-zero)
    assert len(issues) == 1
    assert issues[0]["code"] == "F841"
    assert issues[0]["severity"] == "UNKNOWN" # Check normalization

@patch('subprocess.run')
def test_run_analysis_tool_bandit_success_no_issues(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = VALID_BANDIT_OUTPUT_NO_ISSUES
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("bandit", ["bandit", "-r"], "target/path")
    assert success is True
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_bandit_success_with_issues(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = VALID_BANDIT_OUTPUT_WITH_ISSUES
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("bandit", ["bandit", "-r"], "target/path")
    assert success is True # Bandit success means it ran okay, regardless of issues
    assert len(issues) == 1
    assert issues[0]["code"] == "B101"

@patch('subprocess.run')
def test_run_analysis_tool_json_decode_error(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = "this is not json"
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("anytool", ["cmd"], "target/path")
    assert success is False
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_unexpected_json_structure(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = json.dumps({"unexpected_key": "unexpected_value"}) # Not list for ruff, not results for bandit
    mock_process.stderr = ""
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("ruff", ["cmd"], "target/path") # Test with ruff
    assert success is True # Still True if exit code 0
    assert len(issues) == 0

    success, issues = _run_analysis_tool("bandit", ["cmd"], "target/path") # Test with bandit
    assert success is True # Still True if exit code 0
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_execution_failed_non_zero_exit(mock_subproc_run):
    mock_process = MagicMock()
    mock_process.returncode = 1 # Non-zero exit code
    mock_process.stdout = ""
    mock_process.stderr = "Tool crashed"
    mock_subproc_run.return_value = mock_process

    success, issues = _run_analysis_tool("anytool", ["cmd"], "target/path")
    assert success is False
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_file_not_found(mock_subproc_run):
    mock_subproc_run.side_effect = FileNotFoundError("command not found")

    success, issues = _run_analysis_tool("anytool", ["nonexistent_cmd"], "target/path")
    assert success is False
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_unexpected_exception(mock_subproc_run):
    mock_subproc_run.side_effect = Exception("Something broke badly")

    success, issues = _run_analysis_tool("anytool", ["cmd"], "target/path")
    assert success is False
    assert len(issues) == 0

@patch('subprocess.run')
def test_run_analysis_tool_stderr_logged(mock_subproc_run, caplog):
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.stdout = VALID_RUFF_OUTPUT_NO_ISSUES
    mock_process.stderr = "Some warning from the tool"
    mock_subproc_run.return_value = mock_process

    with caplog.at_level("WARNING"):
        _run_analysis_tool("ruff", ["ruff"], "target/path")
    assert "ruff stderr: Some warning from the tool" in caplog.text

@patch('chungoid.utils.analysis_utils._run_analysis_tool')
def test_run_ruff_analysis(mock_run_helper):
    dummy_path = Path("/fake/path")
    mock_run_helper.return_value = (True, []) # Simulate success
    
    run_ruff_analysis(dummy_path)
    mock_run_helper.assert_called_once_with(
        "ruff",
        ["ruff", "check", "--output-format=json", "--exit-zero"],
        str(dummy_path)
    )

@patch('chungoid.utils.analysis_utils._run_analysis_tool')
def test_run_bandit_analysis(mock_run_helper):
    dummy_path = Path("/fake/path")
    mock_run_helper.return_value = (True, []) # Simulate success
    
    run_bandit_analysis(dummy_path)
    mock_run_helper.assert_called_once_with(
        "bandit",
        ["bandit", "-r", "-f", "json", "-q"],
        str(dummy_path)
    )

def test_generate_analysis_report():
    ruff_issues_sample: List[AnalysisIssue] = [{"code": "R001", "message": "Ruff issue"}]
    bandit_issues_sample: List[AnalysisIssue] = [{"code": "B001", "message": "Bandit issue"}]

    # All pass
    report1 = generate_analysis_report(True, [], True, [])
    assert report1["ruff"]["status"] == "PASS"
    assert report1["bandit"]["status"] == "PASS"
    assert report1["overall_status"] == "PASS"
    assert len(report1["ruff"]["issues"]) == 0

    # Ruff fails execution, Bandit passes
    report2 = generate_analysis_report(False, ruff_issues_sample, True, bandit_issues_sample)
    assert report2["ruff"]["status"] == "FAIL"
    assert report2["bandit"]["status"] == "PASS"
    assert report2["overall_status"] == "FAIL"
    assert report2["ruff"]["issues"] == ruff_issues_sample
    assert report2["bandit"]["issues"] == bandit_issues_sample

    # Ruff passes, Bandit fails execution
    report3 = generate_analysis_report(True, ruff_issues_sample, False, bandit_issues_sample)
    assert report3["ruff"]["status"] == "PASS"
    assert report3["bandit"]["status"] == "FAIL"
    assert report3["overall_status"] == "FAIL"

    # Both fail execution
    report4 = generate_analysis_report(False, [], False, [])
    assert report4["ruff"]["status"] == "FAIL"
    assert report4["bandit"]["status"] == "FAIL"
    assert report4["overall_status"] == "FAIL"

    # Pass with issues
    report5 = generate_analysis_report(True, ruff_issues_sample, True, bandit_issues_sample)
    assert report5["ruff"]["status"] == "PASS"
    assert report5["ruff"]["issues"] == ruff_issues_sample
    assert report5["bandit"]["status"] == "PASS"
    assert report5["bandit"]["issues"] == bandit_issues_sample
    assert report5["overall_status"] == "PASS" 