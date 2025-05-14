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

from typing import Callable, Dict, Protocol, runtime_checkable, Optional, List, Any
from .agent_registry import AgentCard

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
        from .agent_registry import AgentRegistry as ConcreteAgentRegistry

        if not isinstance(registry, ConcreteAgentRegistry):  # noqa: E501 – defensive
            raise TypeError("registry must be an AgentRegistry instance")
        self._registry = registry
        self._fallback: Dict[str, AgentCallable] = fallback or {}
        self._cache: Dict[str, AgentCallable] = {}

        # Lazy MCP client import to keep meta-layer optional in pure-core tests
        try:
            import sys
            from pathlib import Path
            proj_root = Path(__file__).resolve().parents[3]
            dev_scripts = proj_root / "dev" / "scripts"
            if dev_scripts.exists():
                sys.path.append(str(dev_scripts))
            from core_mcp_client import CoreMCPClient  # type: ignore

            self._CoreMCPClient = CoreMCPClient  # stash
        except Exception:
            # No dev scripts or import fails – disable MCP dispatch gracefully
            self._CoreMCPClient = None  # type: ignore

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, identifier: str) -> AgentCallable:  # noqa: D401 – impl of protocol
        # Fast path: return cached callable
        if identifier in self._cache:
            return self._cache[identifier]

        # If the identifier is directly mapped in the fallback, use that callable.
        if identifier in self._fallback:
            callable_from_fallback = self._fallback[identifier]
            self._cache[identifier] = callable_from_fallback # Cache it
            return callable_from_fallback # Return the (potentially async) callable

        # If not in fallback, check registry card
        card = self._registry.get(identifier)
        if card is None:
            # Not in fallback and no card found in registry.
            raise KeyError(f"Agent '{identifier}' not found in registry or direct fallback.")

        # Card exists. Try to use MCP tool dispatch if possible.
        if self._CoreMCPClient and card.tool_names:
            tool_name = card.tool_names[0]

            async def _async_invoke(stage: StageDict) -> Dict[str, object]:  # noqa: D401
                async with self._CoreMCPClient("http://localhost:9000", api_key="dev-key") as mcp:
                    return await mcp.invoke_tool(tool_name, **stage.get("inputs", {}))

            # Synchronous wrapper so FlowExecutor remains sync
            def _sync_callable(stage: StageDict):  # type: ignore[override]
                import asyncio

                return asyncio.run(_async_invoke(stage))

            self._cache[identifier] = _sync_callable
            return _sync_callable

        # Card exists, not in fallback, and no MCP tool/client.
        # Return the stub as the last resort.
        def _stub(stage: StageDict) -> Dict[str, object]:  # type: ignore[override]
            return {"agent_id": identifier, "stage_inputs": stage.get("inputs", {}), "message": "Agent is a stub (card found, no tool/fallback)."}

        self._cache[identifier] = _stub
        return _stub

    def search_agents(self, query_text: str, n_results: int = 3, where_filter: Optional[Dict[str, Any]] = None) -> List[AgentCard]:
        """Performs semantic search for agents via the underlying AgentRegistry."""
        if not hasattr(self._registry, 'search_agents'):
            # Log an error or raise an exception if the registry doesn't support search
            # For now, print and return empty list
            # Consider adding logging if a logger is available on self
            print("ERROR: AgentRegistry instance in RegistryAgentProvider does not have 'search_agents' method.")
            return []
        try:
            return self._registry.search_agents(query_text, n_results=n_results, where_filter=where_filter)
        except Exception as e:
            # Log error appropriately
            print(f"ERROR: Call to AgentRegistry.search_agents failed: {e}")
            return []


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 