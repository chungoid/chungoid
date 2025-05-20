from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field
from typing import NewType, Optional, Literal, List, Dict, Any, Union

# Type alias for dot-notation context paths
InputOutputContextPathStr = NewType('InputOutputContextPathStr', str)
AgentID = NewType('AgentID', str)

class ArbitraryModel(BaseModel):
    """A Pydantic model that allows arbitrary extra fields."""
    model_config = ConfigDict(extra='allow')

class ConfidenceScore(BaseModel):
    """Represents a confidence score, typically from an LLM or analytical process."""
    value: float = Field(..., ge=0.0, le=1.0, description="Numerical confidence score (0.0 to 1.0).")
    level: Optional[Literal["Low", "Medium", "High"]] = Field(None, description="Qualitative confidence level.")
    explanation: Optional[str] = Field(None, description="Brief explanation for the confidence score.")
    method: Optional[str] = Field(None, description="Method used to determine the confidence (e.g., LLM_SELF_ASSESSMENT, HEURISTIC, STATISTICAL).")
    # Potentially add source: Optional[str] = None if needed to track who/what generated the score.

class LLMResponseErrorDetails(BaseModel):
    pass

__all__ = [
    "InputOutputContextPathStr",
    "AgentID",
    "ArbitraryModel",
    "ConfidenceScore"
] 