from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict, Any, List

class AgentErrorDetails(Exception):
    """Standardized structure for agents to report errors."""
    error_type: str = Field(..., description="The class name of the exception raised (e.g., 'ValueError').")
    message: str = Field(..., description="The error message.")
    agent_id: Optional[str] = Field(None, description="The ID of the agent that raised the error.")
    stage_id: Optional[str] = Field(None, description="The ID of the stage where the error occurred, if known by the agent.")
    traceback: Optional[str] = Field(None, description="Optional string representation of the error traceback.")
    details: Optional[Dict[str, Any]] = Field(None, description="Optional dictionary for additional structured error details.")
    resolved_inputs_at_failure: Optional[Dict[str, Any]] = Field(None, description="The resolved inputs for the stage at the time of failure.")
    can_retry: bool = False
    can_escalate: bool = True
    output_payload_if_proceeding: Optional[Any] = Field(None, description="Intended output if flow proceeds past this error.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(
        self,
        message: str,
        error_type: str,
        agent_id: Optional[str] = None,
        stage_id: Optional[str] = None,
        traceback: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        resolved_inputs_at_failure: Optional[Dict[str, Any]] = None,
        can_retry: bool = False,
        can_escalate: bool = True,
        output_payload_if_proceeding: Optional[Any] = None,
        *args: Any
    ) -> None:
        super().__init__(message, *args)
        self.message = message # Exception stores the first arg as message, but we set it explicitly too for consistency
        self.error_type = error_type
        self.agent_id = agent_id
        self.stage_id = stage_id
        self.traceback = traceback
        self.details = details
        self.resolved_inputs_at_failure = resolved_inputs_at_failure
        self.can_retry = can_retry
        self.can_escalate = can_escalate
        self.output_payload_if_proceeding = output_payload_if_proceeding

    def to_dict(self) -> Dict[str, Any]:
        """Converts the AgentErrorDetails instance to a dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "agent_id": self.agent_id,
            "stage_id": self.stage_id,
            "traceback": self.traceback,
            "details": self.details,
            "resolved_inputs_at_failure": self.resolved_inputs_at_failure,
            "can_retry": self.can_retry,
            "can_escalate": self.can_escalate,
            "output_payload_if_proceeding": self.output_payload_if_proceeding
        }

class OrchestratorError(Exception):
    """Custom exception for errors originating within the orchestrator itself."""
    stage_name: Optional[str]
    details: Optional[Dict[str, Any]]

    def __init__(
        self,
        message: str,
        stage_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        *args: Any 
    ) -> None:
        super().__init__(message, *args)
        self.message = message # For consistency, as super() stores it in args[0]
        self.stage_name = stage_name
        self.details = details