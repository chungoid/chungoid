"""Package that ships JSON/YAML schema files used at runtime.

This file exists solely so that `setuptools` treats the *schemas* directory as
package data, ensuring the files are included in wheels and editable installs.
"""

# chungoid.schemas
"""Pydantic models for Chungoid data structures."""

from .common_enums import StageStatus
from .errors import AgentErrorDetails

# TODO: Add other models as needed

__all__ = [
    "StageStatus",
    "AgentErrorDetails",
]