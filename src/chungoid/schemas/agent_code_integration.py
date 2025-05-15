from typing import Literal, Optional, List
from pydantic import BaseModel, Field

class CodeIntegrationInput(BaseModel):
    code_to_integrate: str = Field(description="The code snippet (e.g., function, class, imports) to be integrated.")
    target_file_path: str = Field(description="Absolute or project-relative path to the file to be modified or created.")
    edit_action: Literal[
        "APPEND", 
        "CREATE_OR_APPEND", 
        "ADD_TO_CLICK_GROUP", 
        "ADD_PYTHON_IMPORTS"
    ] = Field(description="Specifies the type of edit action to perform.")
    
    integration_point_hint: Optional[str] = Field(default=None, description="Hint for integration. For ADD_TO_CLICK_GROUP, this is the Click group variable name (e.g., 'utils_group'). For ADD_PYTHON_IMPORTS, can be a marker or ignored for simple top-level add.")
    click_command_name: Optional[str] = Field(default=None, description="Required if edit_action is ADD_TO_CLICK_GROUP. The name of the command function defined in code_to_integrate.")
    backup_original: bool = Field(default=True, description="If true, creates a backup of the original file before modification.")

    class Config:
        extra = 'forbid'

class CodeIntegrationOutput(BaseModel):
    status: Literal["SUCCESS", "FAILURE"] = Field(description="Status of the integration operation.")
    message: str = Field(description="A message detailing the outcome, e.g., 'Integration successful' or error details.")
    modified_file_path: Optional[str] = Field(default=None, description="Path to the file that was modified or created.")
    backup_file_path: Optional[str] = Field(default=None, description="Path to the backup file if one was made.")

    class Config:
        extra = 'forbid' 