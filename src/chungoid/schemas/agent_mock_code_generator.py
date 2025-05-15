from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class MockCodeGeneratorAgentInput(BaseModel):
    target_file_path: str = Field(..., description="The target file path where code would be generated.")
    code_specification_prompt: Optional[str] = Field(None, description="Prompt or specification for code generation (mock will ignore detailed content).")
    relevant_context_keys: Optional[List[str]] = Field(None, description="Keys from context relevant to code generation.")
    # Can include a field for the actual specification if passed directly
    specification: Optional[Dict[str, Any]] = Field(None, description="The code specification data.")

class MockCodeGeneratorAgentOutput(BaseModel):
    code_changes_applied: bool = Field(True, description="Indicates if mock code changes were 'applied'.")
    generated_artifact_path: Optional[str] = Field(None, description="Mock path of the 'generated' or 'modified' artifact.")
    notes: str = Field(default="MockCodeGeneratorAgent simulated code generation.", description="Notes from the agent execution.") 