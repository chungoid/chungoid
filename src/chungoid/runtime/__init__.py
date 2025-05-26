"""Runtime package with unified orchestrator (Phase 3 UAEI Complete).

All legacy interfaces have been eliminated per the enhanced_cycle.md blueprint.
Only UnifiedOrchestrator exists - no backward compatibility.
"""

from __future__ import annotations

from .unified_orchestrator import UnifiedOrchestrator

__all__ = ["UnifiedOrchestrator"] 