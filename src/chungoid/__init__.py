from importlib import import_module

# Lazy import to avoid heavy deps on package import

def __getattr__(name):
    """Lazy-import top-level symbols to keep import footprint light.

    Currently supports:
    • ``ChungoidEngine`` – defers heavy engine import until requested.
    • Arbitrary first-level sub-modules (e.g. ``chungoid.utils``) so that
      ``import chungoid.utils`` works without wiring every sub-package
      explicitly.  This avoids CI failures when test suites import
      sub-modules directly from the package root.
    """

    # Special-case for the engine class to avoid circular import costs.
    if name == "ChungoidEngine":
        engine_module = import_module("chungoid.engine")
        return engine_module.ChungoidEngine

    # Attempt dynamic sub-module import (e.g. chungoid.utils).
    try:
        return import_module(f"chungoid.{name}")
    except ModuleNotFoundError:
        raise AttributeError(name) from None

__all__ = [
    "ChungoidEngine",
    # Allow wild-card exports for known sub-modules that tests expect.
    "utils",
]
