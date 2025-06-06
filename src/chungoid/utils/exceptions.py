"""Custom exceptions for the Chungoid MCP Core system."""

class ChungoidError(Exception):
    """Base class for custom exceptions in Chungoid Core."""
    pass

class StageExecutionError(ChungoidError):
    """Indicates an error during the execution of a stage."""
    pass

class ToolExecutionError(ChungoidError):
    """Indicates an error during the execution of a tool handler."""
    def __init__(self, message, status_code=500):
        super().__init__(message)
        self.status_code = status_code

class PromptRenderError(ChungoidError):
    """Indicates an error rendering a Jinja2 prompt template."""
    pass

class ChromaOperationError(ChungoidError):
    """Custom exception for errors during ChromaDB operations."""
    pass

# Note: ChromaOperationError is also defined in chroma_utils.py for backward compatibility

__all__ = [
    "ChungoidError",
    "StageExecutionError",
    "ToolExecutionError",
    "PromptRenderError",
    "ChromaOperationError",
] 