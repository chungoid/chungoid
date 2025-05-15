from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class MockTestGeneratorAgentInput(BaseModel):
    target_file_path_to_test: str = Field(..., description="The path to the source file that would be tested.")
    code_specification_prompt: Optional[str] = Field(None, description="Prompt or specification for test generation (mock will ignore detailed content).")
    relevant_context_keys: Optional[List[str]] = Field(None, description="Keys from context relevant to test generation.")
    # Can include a field for the actual specification if passed directly
    specification: Optional[Dict[str, Any]] = Field(None, description="The code/feature specification data for test generation.")

class MockTestGeneratorAgentOutput(BaseModel):
    test_file_generated_path: str = Field(..., description="Mock path of the 'generated' test file.")
    tests_generated_count: int = Field(default=5, description="Number of mock tests 'generated'.")
    notes: str = Field(default="MockTestGeneratorAgent simulated test generation.", description="Notes from the agent execution.") 