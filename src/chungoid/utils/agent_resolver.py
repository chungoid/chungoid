from __future__ import annotations

"""Adapters that resolve agent identifiers to callables.

Two use-cases:
• **DictAgentProvider** – thin wrapper around an in-memory mapping used by
  existing tests/demo scripts.
• **RegistryAgentProvider** – consults `AgentRegistry` (Chroma-backed)
  and optionally falls back to a supplied mapping.

Both implement the `AgentProvider` protocol expected by the refactored
`FlowExecutor`.
"""

from typing import Callable, Dict, Protocol, runtime_checkable, Optional

StageDict = Dict[str, object]
AgentCallable = Callable[[StageDict], Dict[str, object]]


@runtime_checkable
class AgentProvider(Protocol):
    """Minimal interface for resolving agent identifiers."""

    def get(self, identifier: str) -> AgentCallable:  # noqa: D401 – simple protocol
        """Return a callable that executes the given agent."""
        ...


class DictAgentProvider:
    """AgentProvider backed by a plain dict (legacy behaviour)."""

    def __init__(self, mapping: Dict[str, AgentCallable]):
        self._mapping = mapping

    def get(self, identifier: str) -> AgentCallable:  # noqa: D401 – impl of protocol
        try:
            return self._mapping[identifier]
        except KeyError as exc:  # pragma: no cover
            raise KeyError(f"Unknown agent '{identifier}'") from exc


class RegistryAgentProvider:
    """Resolve agents via AgentRegistry; fallback to optional mapping.

    The returned callable is **currently a stub** that simply invokes the
    MCP tool specified by `AgentCard.tool_names[0]` when present, or raises
    *NotImplementedError* if no tool mapping exists.  The interface is
    expected to evolve once the MCP client supports full dynamic dispatch.
    """

    def __init__(
        self,
        registry: "AgentRegistry",
        fallback: Optional[Dict[str, AgentCallable]] = None,
    ) -> None:
        from .agent_registry import AgentRegistry  # local import to avoid circular

        if not isinstance(registry, AgentRegistry):  # noqa: E501 – defensive
            raise TypeError("registry must be an AgentRegistry instance")
        self._registry = registry
        self._fallback: Dict[str, AgentCallable] = fallback or {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, identifier: str) -> AgentCallable:  # noqa: D401 – impl of protocol
        card = self._registry.get(identifier)
        if card is None:
            # graceful fallback to mapping if present
            if identifier in self._fallback:
                return self._fallback[identifier]
            raise KeyError(f"Agent '{identifier}' not found in registry")

        # Simple placeholder implementation: echo stage + agent_id
        def _stub_callable(stage: StageDict) -> Dict[str, object]:  # type: ignore[override]
            """Placeholder callable until dynamic MCP invocation is wired up."""
            return {"agent_id": identifier, "stage_inputs": stage.get("inputs", {})}

        return _stub_callable


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 