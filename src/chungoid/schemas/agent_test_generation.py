from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class TestGenerationInput(BaseModel):
    """Input model for the Test Generation Agent."""
    command_code: str = Field(description="The actual code of the command to be tested.")
    command_spec: str = Field(description="The specification or requirements document for the command.")
    testing_framework_hint: Optional[str] = Field(None, description="Hint for the testing framework to be used (e.g., 'pytest').")
    test_file_structure_hint: Optional[str] = Field(None, description="Hint for the desired test file structure or location.")
    output_test_file_path: Optional[str] = Field("dummy_project/generated_tests/test_mock_generated.py", description="Suggested path to write the generated test file.")

class TestGenerationOutput(BaseModel):
    """Output model for the Test Generation Agent."""
    status: str = Field(default="SUCCESS", description="Status of the test generation process (e.g., 'SUCCESS', 'FAILURE').")
    message: str = Field(default="Mock tests generated successfully.", description="A message detailing the outcome of the test generation.")
    generated_tests: Optional[str] = Field(default="# Mock test content\n# def test_mock():\n# assert True", description="The string content of the generated tests.")
    generated_test_filepath: Optional[str] = Field(None, description="The absolute or relative path to the file where tests were written.")
    tests_generated_count: int = Field(default=1, description="Number of distinct test cases or test functions generated.") 