from pydantic import BaseModel
from chungoid.schemas.errors import AgentErrorDetails
from typing import TypeVar, Generic, Optional, Dict, Any

InputSchema = TypeVar('InputSchema')
OutputSchema = TypeVar('OutputSchema')

class BaseAgent(BaseModel, Generic[InputSchema, OutputSchema]):
    """Base class for all Chungoid agents."""
    
    _config_internal: Optional[Dict[str, Any]] = None
    _system_context_internal: Optional[Dict[str, Any]] = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        # Store config and system_context if they are provided in data
        # These are passed by subclasses in their super().__init__(config=..., system_context=...)
        # Pydantic will put them in `data` if they are not model fields.
        self._config_internal = data.get('config')
        self._system_context_internal = data.get('system_context')

# Re-export AgentErrorDetails as AgentError to satisfy existing imports
AgentError = AgentErrorDetails

class AgentInputError(AgentError):
    """Custom exception for errors related to agent input validation or processing."""
    def __init__(self, message: str, agent_id: str = "Unknown", details: Optional[Dict[str, Any]] = None):
        super().__init__(message=message, agent_id=agent_id, error_type="AgentInputError", details=details)
        # Note: AgentErrorDetails (which AgentError is an alias for) takes these named args.


__all__ = ["BaseAgent", "InputSchema", "OutputSchema", "AgentError", "AgentInputError"]

# Example of a more distinct AgentError if needed later:
# class AgentError(Exception):
#     """Base exception for agent-related errors."""
#     def __init__(self, message: str, agent_id: str = "Unknown", stage_id: str = "Unknown"):
#         super().__init__(message)
#         self.agent_id = agent_id
#         self.stage_id = stage_id
#         self.message = message

#     def __str__(self):
#         return f"AgentError in agent '{self.agent_id}' (Stage: '{self.stage_id}'): {self.message}" 