from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from typing import NewType

# Type alias for dot-notation context paths
InputOutputContextPathStr = NewType('InputOutputContextPathStr', str)

class ArbitraryModel(BaseModel):
    """A Pydantic model that allows arbitrary extra fields."""
    model_config = ConfigDict(extra='allow')

__all__ = [
    "InputOutputContextPathStr",
    "ArbitraryModel"
] 