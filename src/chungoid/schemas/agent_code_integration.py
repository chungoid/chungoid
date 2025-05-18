from __future__ import annotations

import uuid
from typing import Optional, Dict, Any, Literal
from pydantic import BaseModel, Field

from chungoid.schemas.common import ConfidenceScore

# Schemas for SmartCodeIntegrationAgent_v1 (enhancements of CoreCodeIntegrationAgentV1)
class SmartCodeIntegrationInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this integration task.")
    project_id: str = Field(..., description="Project ID for ProjectChromaManagerAgent interactions.")
    
    generated_code_artifact_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the artifact containing the code string to integrate (e.g., from SmartCodeGeneratorAgentOutput).")
    code_to_integrate_directly: Optional[str] = Field(None, description="Direct code string to integrate. If provided and generated_code_artifact_doc_id is also present, behavior might be to prioritize one or combine, TBD.")
    
    target_file_path: str = Field(..., description="Absolute or project-relative path to the file to be modified or created.")
    edit_action: Literal["APPEND", "CREATE_OR_APPEND", "ADD_TO_CLICK_GROUP", "ADD_PYTHON_IMPORTS", "REPLACE_FILE_CONTENT"] = Field(..., description="Specifies the type of edit action.")
    
    existing_target_file_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the current version of target_file_path in live_codebase_collection, for context.")
    
    integration_point_hint: Optional[str] = Field(None, description="Hint for ADD_TO_CLICK_GROUP or future AST modifications.")
    click_command_name: Optional[str] = Field(None, description="For ADD_TO_CLICK_GROUP: name of the command function.")
    imports_to_add: Optional[str] = Field(None, description="For ADD_PYTHON_IMPORTS: newline-separated import statements.")
    backup_original: bool = Field(True, description="If true, creates a backup of the original file before modification.")
    
    agent_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

class SmartCodeIntegrationOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    status: Literal["SUCCESS", "FAILURE", "FAILURE_CHROMA_RETRIEVAL", "FAILURE_CHROMA_UPDATE"] = Field(..., description="Status of the integration operation.")
    message: str = Field(..., description="A message detailing the outcome.")
    modified_file_path: Optional[str] = Field(None, description="Path to the file that was modified or created.")
    backup_file_path: Optional[str] = Field(None, description="Path to the backup file if one was made.")
    
    updated_live_codebase_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the updated/newly_created file artifact in live_codebase_collection.")
    md5_hash_of_integrated_file: Optional[str] = Field(None, description="MD5 hash of the file content after integration, for verification.")
    
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the successful and correct integration.")
    error_message: Optional[str] = Field(None)


# --- Old Schemas (to be deprecated/removed later) --- #
class CodeIntegrationInput(BaseModel):
    code_to_integrate: str = Field(description="The code snippet (e.g., function, class, imports) to be integrated.")
    target_file_path: str = Field(description="Absolute or project-relative path to the file to be modified or created.")
    # Available actions: APPEND, CREATE_OR_APPEND, ADD_TO_CLICK_GROUP, ADD_PYTHON_IMPORTS
    edit_action: str = Field(description="Specifies the type of edit action to perform.")
    integration_point_hint: Optional[str] = Field(None, description="Hint for integration. For ADD_TO_CLICK_GROUP, this is the Click group variable name (e.g., 'utils_group'). Not strictly used by V1 insertion logic but useful for code generators.")
    click_command_name: Optional[str] = Field(None, description="Required if edit_action is ADD_TO_CLICK_GROUP. The name of the command function defined in code_to_integrate. Not strictly used by V1 insertion logic.")
    imports_to_add: Optional[str] = Field(None, description="For ADD_PYTHON_IMPORTS: newline-separated import statements.") # Added to match agent logic more closely
    backup_original: bool = Field(True, description="If true, creates a backup of the original file (e.g., file.py.bak) before modification.")

class CodeIntegrationOutput(BaseModel):
    status: str = Field(description="Status of the integration operation, e.g., 'SUCCESS', 'FAILURE'.")
    message: str = Field(description="A message detailing the outcome, e.g., 'Integration successful' or error details.")
    modified_file_path: Optional[str] = Field(None, description="Path to the file that was modified or created.")
    backup_file_path: Optional[str] = Field(None, description="Path to the backup file if one was made.")

    class Config:
        extra = 'forbid' 