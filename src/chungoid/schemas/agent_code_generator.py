from __future__ import annotations

import uuid # Added for default task_id
from typing import Optional, Dict, Any, List, Literal # Added List and Literal
from pydantic import BaseModel, Field

from chungoid.schemas.common import ConfidenceScore # Assuming common schema

class SmartCodeGeneratorAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this code generation task.")
    project_id: str = Field(..., description="Project ID for ProjectChromaManagerAgent interactions.")
    
    task_description: str = Field(..., description="Core description of the code to be generated or task to be performed. Used as primary spec if code_specification_doc_id is absent.")

    code_specification_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the detailed code specification document (e.g., from a plan stage). Content expected to be Markdown or structured text.")
    target_file_path: str = Field(..., description="Intended relative path of the file to be created or modified.")
    programming_language: str = Field(..., description="Target programming language (e.g., 'python', 'javascript', 'typescript', 'java', 'csharp'). This field is required and must be explicitly provided.")

    existing_code_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the current content of target_file_path if modifying. From live_codebase_collection.")
    blueprint_context_doc_id: Optional[str] = Field(None, description="ChromaDB ID of relevant Project Blueprint section(s). From planning_artifacts collection.")
    loprd_requirements_doc_ids: Optional[List[str]] = Field(None, description="List of ChromaDB IDs for relevant LOPRD requirements. From planning_artifacts collection.")
    
    agent_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    additional_instructions: Optional[str] = Field(None, description="Additional free-text instructions or constraints.")

class SmartCodeGeneratorAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    target_file_path: str = Field(..., description="The intended relative path (mirrors input).")
    status: Literal["SUCCESS", "FAILURE_LLM_GENERATION", "FAILURE_CONTEXT_RETRIEVAL", "FAILURE_OUTPUT_STORAGE", "FAILURE_INPUT_VALIDATION"] = Field(..., description="Status of the code generation attempt.")
    
    generated_code_artifact_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the stored generated code string (e.g., in live_codebase_collection or generated_code_pending_integration collection).")
    stored_in_collection: Optional[str] = Field(None, description="The ChromaDB collection where the generated code artifact was stored.")
    generated_code_string: Optional[str] = Field(None, description="The generated code string (can be omitted if very large and doc_id is provided).")
    
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the generated code's correctness and contextual adherence.")
    llm_full_response: Optional[str] = Field(None, description="Raw LLM response for debugging, if applicable.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="LLM token usage, etc.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")

# --- Old Schemas (to be deprecated/removed later if SmartCodeGeneratorAgent fully replaces CoreCodeGeneratorAgent functionality) --- #
class CodeGeneratorAgentInput(BaseModel):
    task_description: str = Field(description="Detailed description of the code to be generated or modified. e.g., 'Implement a FastAPI endpoint /foo that returns {\"bar\": \"baz\"}'")
    target_file_path: str = Field(description="The intended relative path of the file to be created or modified within the project.")
    code_to_modify: Optional[str] = Field(None, description="The existing relevant code snippet if the task is a modification.")
    related_files_context: Optional[Dict[str, str]] = Field(None, description="Content of other relevant files as context for the LLM, e.g., {'utils.py': '<content of utils.py>'}.")
    programming_language: str = Field(..., description="The programming language of the code to be generated (e.g., 'python', 'javascript', 'typescript', 'java', 'csharp'). This field is required and must be explicitly provided.")
    project_root_path: Optional[str] = Field(None, description="Absolute path to the project root, if available and needed for context by the agent/LLM (though agent should primarily use relative paths for output).")

class CodeGeneratorAgentOutput(BaseModel):
    generated_code_string: Optional[str] = Field(None, description="The code string generated by the LLM.")
    target_file_path: str = Field(description="The intended relative path where the generated code should be written (same as input target_file_path).")
    status: str = Field(description="Status of the code generation attempt, e.g., 'SUCCESS', 'FAILURE_LLM_GENERATION', 'FAILURE_INPUT_VALIDATION'.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")
    llm_full_response: Optional[str] = Field(None, description="The full raw response from the LLM, if available, for debugging.")
    llm_confidence: Optional[float] = Field(None, description="LLM's confidence in the generation, if available.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="LLM usage metadata, e.g., token counts.") 