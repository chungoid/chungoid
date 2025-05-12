from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class AgentErrorDetails(BaseModel):
    """Standardized structure for agents to report errors."""
    error_type: str = Field(..., description="The class name of the exception raised (e.g., 'ValueError').")
    message: str = Field(..., description="The error message.")
    agent_id: Optional[str] = Field(None, description="The ID of the agent that raised the error.")
    stage_id: Optional[str] = Field(None, description="The ID of the stage where the error occurred, if known by the agent.")
    traceback: Optional[str] = Field(None, description="Optional string representation of the error traceback.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary for additional structured error details.")