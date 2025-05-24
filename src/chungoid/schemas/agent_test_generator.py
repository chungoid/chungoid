from __future__ import annotations

from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

class TestGeneratorAgentInput(BaseModel):
    code_to_test: str = Field(description="The actual source code string that needs to be tested.")
    file_path_of_code: str = Field(description="The relative path of the file containing the code_to_test. Used for context by the LLM.")
    target_test_file_path: str = Field(description="The intended relative path where the generated test file should eventually be written.")
    test_framework_preference: Optional[str] = Field("pytest", description="Preferred testing framework, e.g., 'pytest', 'unittest'.")
    related_files_context: Optional[Dict[str, str]] = Field(None, description="Content of other relevant files as context for the LLM, e.g., {'models.py': '<content of models.py>'}.")
    programming_language: str = Field(..., description="The programming language of the code to be tested and the tests to be generated (e.g., 'python', 'javascript', 'typescript', 'java', 'csharp'). This field is required and must be explicitly provided.")
    project_root_path: Optional[str] = Field(None, description="Absolute path to the project root, for LLM context if needed.")
    
    # Added for richer context from PCMA
    project_id: Optional[str] = Field(None, description="Project ID for context, used by agent for PCMA calls.")
    task_id: Optional[str] = Field(None, description="Task ID for context, used by agent for logging/tracing.")
    relevant_loprd_requirements_ids: Optional[list[str]] = Field(None, description="List of LOPRD requirement IDs (e.g., FRs, ACs) relevant to the code being tested.")
    relevant_blueprint_section_ids: Optional[list[str]] = Field(None, description="List of Blueprint section IDs relevant to the code's design.")

class TestGeneratorAgentOutput(BaseModel):
    target_test_file_path: str = Field(..., description="The local file path where the generated tests should be (or were) written.")
    generated_test_code: Optional[str] = Field(None, description="The actual generated test code as a string.")
    status: Literal["SUCCESS_GENERATED", "FAILURE_LLM_GENERATION", "FAILURE_INPUT_ERROR", "FAILURE_INPUT_RETRIEVAL", "FAILURE_INTERNAL"] = Field(..., description="Status of the test generation process.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")
    llm_full_response: Optional[str] = Field(None, description="The full, raw response from the LLM, if available, for debugging.")
    # Placeholder for detailed metrics if the LLM provides them
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.") 
    # NEW fields for PCMA storage
    generated_test_artifact_id: Optional[str] = Field(None, description="The document ID of the stored generated test artifact in ChromaDB.")
    stored_in_collection: Optional[str] = Field(None, description="The ChromaDB collection where the test artifact was stored.") 