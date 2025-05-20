from __future__ import annotations

import uuid
from typing import List, Optional, Literal

from pydantic import BaseModel, Field
from chungoid.schemas.code_debugging_agent_schemas import FailedTestReport # Re-use this for test failures

class CodeIntegrationTaskInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this code integration task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    cycle_id: str = Field(..., description="The ID of the current refinement cycle.")
    target_file_path: str = Field(..., description="Path to the code file to be modified.")
    code_changes: str = Field(..., description="The proposed code changes (e.g., diff or full snippet).")
    solution_type: Literal["CODE_PATCH", "MODIFIED_SNIPPET"] = Field(..., description="Type of the proposed solution, indicating how code_changes should be interpreted.")

class CodeIntegrationTaskOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    status: Literal["SUCCESS_APPLIED", "FAILURE_PATCH_CONFLICT", "FAILURE_OTHER_INTEGRATION_ERROR"] = Field(..., description="Status of the code integration attempt.")
    message: str = Field(..., description="A message detailing the outcome.")
    integrated_file_path: Optional[str] = Field(None, description="Path to the file after changes were applied. Usually same as input target_file_path.")
    error_details: Optional[str] = Field(None, description="Specific error details if integration failed.")

class TestRunnerTaskInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this test execution task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    cycle_id: str = Field(..., description="The ID of the current refinement cycle.")
    code_module_file_path: str = Field(..., description="The path to the code module that was modified and needs re-testing.")
    specific_tests_to_run: Optional[List[str]] = Field(None, description="List of specific test names to run. If None, relevant tests or all tests for the module might be run.")
    run_all_tests_for_module: bool = Field(default=True, description="Flag to indicate if all tests related to the module should be run.")

class TestRunnerTaskOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    status: Literal["SUCCESS_ALL_PASSED", "FAILURE_TESTS_FAILED", "ERROR_RUNNER_FAILED"] = Field(..., description="Status of the test execution.")
    message: str = Field(..., description="A message detailing the outcome.")
    failed_test_reports: Optional[List[FailedTestReport]] = Field(None, description="List of structured failure reports if any tests failed. Re-uses FailedTestReport schema.")
    passed_tests_count: int = Field(0, description="Number of tests that passed.")
    failed_tests_count: int = Field(0, description="Number of tests that failed.")
    total_tests_run: int = Field(0, description="Total number of tests executed.")
    error_details: Optional[str] = Field(None, description="Specific error details if the test runner itself failed.") 