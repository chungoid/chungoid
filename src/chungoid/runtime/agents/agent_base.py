from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, Optional, TypeVar, TYPE_CHECKING
import sys
import logging

from pydantic import BaseModel, Field, ConfigDict
from chungoid.schemas.errors import AgentErrorDetails

# Added imports for LLMProvider and PromptManager
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager

InputSchema = TypeVar('InputSchema', bound=BaseModel)
OutputSchema = TypeVar('OutputSchema', bound=BaseModel)

# Define a default log format string if not already available globally
DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"

# MODIFIED: Inheritance order changed
class BaseAgent(BaseModel, Generic[InputSchema, OutputSchema], ABC):
    """Base class for all Chungoid agents."""
    
    # Declared llm_provider and prompt_manager as fields
    llm_provider: Optional[LLMProvider] = Field(None, description="LLM provider instance for AI capabilities.")
    prompt_manager: Optional[PromptManager] = Field(None, description="Prompt manager for loading and rendering prompts.")
    
    # System context can hold things like logger instances, shared configuration, etc.
    # Passed down from the environment where the agent is run.
    system_context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="System-level context dictionary.")

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
    def logger(self) -> logging.Logger:
        # Check if we already have a cached logger instance
        if hasattr(self, '_logger_instance') and self._logger_instance is not None:
            return self._logger_instance

        _logger: Optional[logging.Logger] = None
        
        # Attempt to get logger from system_context first
        if self.system_context and isinstance(self.system_context, dict): # Handle None case
            _logger = self.system_context.get('logger')
            if _logger and not isinstance(_logger, logging.Logger):
                # Log a warning if what we got isn't a Logger instance
                # Use a temporary basic logger for this warning to avoid recursion if self.logger is called here
                temp_logger_for_warning = logging.getLogger(f"agent_base.{self.__class__.__name__}")
                temp_logger_for_warning.warning(
                    f"Expected 'logger' in system_context to be a logging.Logger instance, got {type(_logger)}. Falling back."
                )
                _logger = None # Reset to trigger fallback

        # Fallback if logger not found in system_context or was invalid
        if _logger is None:
            # Create a robust fallback logger, independent of potentially uninitialized agent state
            fallback_logger_name = f"agent.{self.__class__.__name__}" # Includes module and class name
            _logger = logging.getLogger(fallback_logger_name)
            
            # Ensure the fallback logger has at least one handler to output messages
            # This prevents messages from being lost if no other configuration is applied.
            if not _logger.hasHandlers():
                try:
                    # Attempt to add a basic StreamHandler if none exist.
                    # This configuration should be minimal and not rely on self.config or other agent state.
                    stream_handler = logging.StreamHandler(sys.stdout) # Default to stdout
                    formatter = logging.Formatter(
                        DEFAULT_LOG_FORMAT # Use a globally defined default format if available
                    )
                    stream_handler.setFormatter(formatter)
                    _logger.addHandler(stream_handler)
                    _logger.setLevel(logging.INFO) # Default level for fallback logger
                    _logger.propagate = False # Avoid duplicate messages if root logger is also configured
                except Exception as e_fallback_log_setup:
                    # If even basic handler setup fails, print to stderr and use a completely basic logger.
                    print(f"CRITICAL: Failed to set up fallback logger handler for {fallback_logger_name}: {e_fallback_log_setup}", file=sys.stderr)
                    # As a last resort, the getLogger call itself provides a logger, even if unconfigured.
                    pass # _logger from getLogger(name) is already set

        # Cache the resolved or created logger instance
        self._logger_instance = _logger
        return self._logger_instance

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