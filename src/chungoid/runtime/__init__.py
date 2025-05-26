"""Runtime package with compatibility aliases (Phase-1 UAEI).

The legacy `AsyncOrchestrator` symbol now points to the new
`UnifiedOrchestrator` class so external callers do not break during the
migration.  Importers will receive a *DeprecationWarning* at import time.

Remove this alias in Phase-2 once all callers have been updated.
"""

from __future__ import annotations

import warnings

from .unified_orchestrator import UnifiedOrchestrator

warnings.warn(
    "`AsyncOrchestrator` is deprecated – import `UnifiedOrchestrator` from 'chungoid.runtime.unified_orchestrator' instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Compat alias --------------------------------------------------------------
AsyncOrchestrator = UnifiedOrchestrator  # noqa: N816 – keep legacy camel case

__all__ = ["UnifiedOrchestrator", "AsyncOrchestrator"] 