"""
DEPRECATED: FlowExecutor - Legacy Flow Execution

⚠️ DEPRECATED: This module is deprecated as part of Phase 3 UAEI migration.
FlowExecutor uses legacy agent resolution patterns that have been eliminated.

Use UnifiedOrchestrator with UnifiedAgentResolver instead.

This file will be removed after Phase 3 migration is complete.
"""

import warnings
from pathlib import Path

warnings.warn(
    "FlowExecutor is deprecated. Use UnifiedOrchestrator with UnifiedAgentResolver instead.",
    DeprecationWarning,
    stacklevel=2
)

# REMOVED: All FlowExecutor implementation to prevent legacy pattern usage
# The functionality has been superseded by:
# - UnifiedOrchestrator for orchestration
# - UnifiedAgentResolver for agent resolution  
# - ExecutionContext for unified data models

class FlowExecutor:
    """DEPRECATED: Use UnifiedOrchestrator instead."""
    
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "FlowExecutor is deprecated and has been replaced by UnifiedOrchestrator. "
            "Please use the new UAEI architecture instead of legacy flow execution patterns."
        ) 