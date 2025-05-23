from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, TYPE_CHECKING

from pydantic import BaseModel, Field, ConfigDict
from chungoid.schemas.errors import AgentErrorDetails

# Added imports for LLMProvider and PromptManager
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager

# MODIFIED: Wrap ProjectChromaManagerAgent_v1 import in TYPE_CHECKING
if TYPE_CHECKING:
    from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

InputSchema = TypeVar('InputSchema', bound=BaseModel)
OutputSchema = TypeVar('OutputSchema', bound=BaseModel)

# MODIFIED: Inheritance order changed
class BaseAgent(BaseModel, Generic[InputSchema, OutputSchema], ABC):
    """Base class for all Chungoid agents."""
    
    # Declared llm_provider and prompt_manager as fields
    llm_provider: Optional[LLMProvider] = Field(None, description="LLM provider instance for AI capabilities.")
    prompt_manager: Optional[PromptManager] = Field(None, description="Prompt manager for loading and rendering prompts.")
    project_chroma_manager: Optional['ProjectChromaManagerAgent_v1'] = Field(None, description="Agent for managing project-specific ChromaDB collections.")
    
    # System context can hold things like logger instances, shared configuration, etc.
    # Passed down from the environment where the agent is run.
    system_context: Dict[str, Any] = Field(default_factory=dict, description="System-level context dictionary.")

    # Agent-specific configuration, can be overridden by subclasses or at instantiation.
    config: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Agent-specific configuration dictionary.")

    # Added model_config to allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any):
        """Initializes the agent with configuration and system context."""
        # Pydantic's BaseModel.__init__ will handle llm_provider and prompt_manager
        # and project_chroma_manager if they are in 'data'
        super().__init__(**data)

        # Store config and system_context if provided, for internal use by agent subclasses
        # These are not Pydantic fields themselves but ways to pass arbitrary dicts.
        self.config = data.get('config')
        self.system_context = data.get('system_context')

    @abstractmethod
    async def invoke_async(self, task_input: InputSchema, full_context: Optional[Dict[str, Any]] = None) -> OutputSchema:
        """Asynchronously processes the input and returns the output. This is the primary method for agent execution."""
        raise NotImplementedError("Subclasses must implement invoke_async")

    def get_name(self) -> str:
        """Returns the name of the agent. Subclasses might override AGENT_NAME."""
        if hasattr(self, 'AGENT_NAME') and isinstance(self.AGENT_NAME, str):
            return self.AGENT_NAME
        return self.__class__.__name__

    def get_id(self) -> str:
        """Returns the unique ID of the agent. Subclasses must override AGENT_ID."""
        if hasattr(self, 'AGENT_ID') and isinstance(self.AGENT_ID, str):
            return self.AGENT_ID
        raise NotImplementedError(f"Agent class {self.__class__.__name__} must define an AGENT_ID class variable.")

    # Optional: Method to update agent's internal configuration if needed post-instantiation
    def update_config(self, new_config: Dict[str, Any]):
        if self.config is None:
            self.config = {}
        self.config.update(new_config)

    # Optional: Method to get a specific config value
    def get_config_value(self, key: str, default: Any = None) -> Any:
        if self.config is None:
            return default
        return self.config.get(key, default)

    # Optional: Helper to get logger from system_context or create a default one
    @property
    def logger(self):
        if hasattr(self, '_logger_instance') and self._logger_instance:
            return self._logger_instance
        
        _logger = self.system_context.get('logger')
        if not _logger:
            import logging
            _logger = logging.getLogger(self.get_name())
        self._logger_instance = _logger # Cache it
        return _logger

    @classmethod
    def get_agent_card_static(cls) -> Any: # Changed to Any to avoid circular import with AgentCard
        """Subclasses should override this to return their AgentCard."""
        raise NotImplementedError("Subclasses must implement get_agent_card_static to provide an AgentCard.")


class AgentExecutionError(Exception):
    """Custom exception for errors during agent execution."""
    def __init__(self, message: str, details: Optional[AgentErrorDetails] = None):
        super().__init__(message)
        self.details = details if details else AgentErrorDetails(error_type=self.__class__.__name__, error_message=message)

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

# BaseAgent.model_rebuild() # COMMENTED OUT 