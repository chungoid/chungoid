from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SystemTestRunnerAgentInput(BaseModel):
    test_target_path: str = Field(description="Path to the test file or directory to run.")
    pytest_options: Optional[str] = Field(default=None, description="Additional options to pass to pytest.")
    project_root_path: Optional[str] = Field(default=None, description="The root directory of the project, used as CWD for pytest.")


class SystemTestRunnerAgentOutput(BaseModel):
    exit_code: int = Field(description="Exit code from the pytest command.")
    summary: str = Field(description="A summary of the test run (e.g., '1 passed, 1 failed').")
    stdout: Optional[str] = Field(default=None, description="Standard output from the pytest command.")
    stderr: Optional[str] = Field(default=None, description="Standard error from the pytest command.")
    status: str = Field(default="UNKNOWN", description="Overall status of the test execution (e.g., SUCCESS, FAILURE).") 