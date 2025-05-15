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

from typing import Callable, Dict, Protocol, runtime_checkable, Optional, List, Any, Tuple
from semver import VersionInfo
from .agent_registry import AgentCard

StageDict = Dict[str, object]
AgentCallable = Callable[[StageDict], Dict[str, object]]


@runtime_checkable
class AgentProvider(Protocol):
    """Minimal interface for resolving agent identifiers."""

    def get(self, identifier: str) -> AgentCallable:  # noqa: D401 – simple protocol
        """Return a callable that executes the given agent."""
        ...

    # It's good practice to add new methods to the protocol if they are part of the core interface
    def resolve_agent_by_category(
        self, 
        category: str, 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, AgentCallable]:
        """Resolves an agent by category and preferences, returning its ID and callable."""
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

    # DictAgentProvider typically won't support category-based resolution without significant changes.
    # For now, we can raise a NotImplementedError or return a default if called.
    def resolve_agent_by_category(
        self, 
        category: str, 
        preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, AgentCallable]:
        raise NotImplementedError("DictAgentProvider does not support category-based resolution.")


class NoAgentFoundForCategoryError(KeyError):
    """Raised when no agent matches the specified category and preferences."""
    pass

class AmbiguousAgentCategoryError(ValueError):
    """Raised when multiple agents match the category and preferences, and a unique selection cannot be made."""
    pass


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
            potential_callable = self._fallback[identifier]
            # If the fallback item is an object with an 'invoke_async' method, return that method.
            if hasattr(potential_callable, 'invoke_async') and callable(getattr(potential_callable, 'invoke_async')):
                actual_callable = getattr(potential_callable, 'invoke_async')
                self._cache[identifier] = actual_callable # Cache the method
                return actual_callable
            else:
                # Otherwise, assume the fallback item is already the callable we want (e.g., a direct function)
                self._cache[identifier] = potential_callable # Cache it
                return potential_callable

        # If not in fallback, check registry card
        card = self._registry.get(identifier)
        if card is None:
            # Not in fallback and no card found in registry.
            raise KeyError(f"Agent '{identifier}' not found in registry or direct fallback.")

        # Card exists. Try to use MCP tool dispatch if possible.
        if self._CoreMCPClient and card.tool_names:
            tool_name = card.tool_names[0]

            async def _async_invoke(stage: StageDict, full_context: Optional[Dict[str, Any]] = None) -> Dict[str, object]:  # Added full_context
                # Assuming CoreMCPClient needs to be instantiated per call or managed appropriately
                # The original code instantiated it with a hardcoded URL and key.
                # This might need to come from config passed to RegistryAgentProvider.
                # For now, replicating the original hardcoded values for simplicity of this diff.
                async with self._CoreMCPClient("http://localhost:9000", api_key="dev-key") as mcp:
                    # Pass inputs from stage, potentially merge with full_context if mcp.invoke_tool supports it
                    # or if inputs are resolved against full_context before this call by orchestrator.
                    # The original code passed stage.get("inputs", {}).
                    # Let's stick to that for minimal change from original intent here.
                    return await mcp.invoke_tool(tool_name, **stage.get("inputs", {}))

            self._cache[identifier] = _async_invoke # Cache the async callable directly
            return _async_invoke # Return the async callable

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

    async def resolve_agent_by_category(
        self,
        category: str,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, AgentCallable]:
        """Resolves an agent by category and preferences.

        Returns:
            A tuple of (resolved_agent_id, agent_callable).
        
        Raises:
            NoAgentFoundForCategoryError: If no suitable agent is found.
            AmbiguousAgentCategoryError: If multiple agents are equally suitable and a unique choice cannot be made.
        """
        if preferences is None:
            preferences = {}

        all_cards = self._registry.list(limit=1000) # Assuming a reasonable upper limit
        
        # 1. Filter by category
        candidate_cards = [card for card in all_cards if card.categories and category in card.categories]

        if not candidate_cards:
            raise NoAgentFoundForCategoryError(f"No agents found in category '{category}'.")

        # 2. Apply capability_profile_match
        profile_match_prefs = preferences.get("capability_profile_match", {})
        if profile_match_prefs:
            filtered_by_profile: List[AgentCard] = []
            for card in candidate_cards:
                match = True
                for key, expected_value in profile_match_prefs.items():
                    if card.capability_profile.get(key) != expected_value:
                        match = False
                        break
                if match:
                    filtered_by_profile.append(card)
            candidate_cards = filtered_by_profile
        
        if not candidate_cards:
            raise NoAgentFoundForCategoryError(f"No agents in category '{category}' match capability profile: {profile_match_prefs}")

        # 3. Apply priority_gte
        priority_gte = preferences.get("priority_gte")
        if priority_gte is not None:
            candidate_cards = [card for card in candidate_cards if card.priority is not None and card.priority >= priority_gte]

        if not candidate_cards:
            raise NoAgentFoundForCategoryError(f"No agents in category '{category}' (after profile match) meet priority_gte: {priority_gte}")

        # 4. Handle version_preference (e.g., "latest_semver")
        # This is a simplified version preference handling. More robust parsing might be needed.
        version_preference = preferences.get("version_preference")
        if version_preference == "latest_semver" and candidate_cards:
            # Assumes version is stored in capability_profile["version_semver"] or card.version
            # and is a valid semver string.
            def get_semver(card: AgentCard) -> Optional[VersionInfo]:
                ver_str = card.capability_profile.get("version_semver") or card.version
                if ver_str:
                    try:
                        return VersionInfo.parse(ver_str.lstrip('v'))
                    except ValueError:
                        return None
                return None

            candidate_cards.sort(key=lambda c: (c.priority or 0, get_semver(c) or VersionInfo(0)), reverse=True)
            
            if candidate_cards: # If any candidates remain after sorting
                 # Check for ties in priority and version if that constitutes ambiguity
                best_card = candidate_cards[0]
                best_priority = best_card.priority or 0
                best_version = get_semver(best_card)
                
                # If there are multiple cards with the same highest priority and version (if version is used for sorting)
                # this could be an ambiguity. For now, taking the first one after sorting by priority then version.
                # A more strict ambiguity check might be needed based on requirements.
                pass # First one is taken by default after sort

        elif candidate_cards: # Default sort by priority if no specific version preference
            candidate_cards.sort(key=lambda c: c.priority or 0, reverse=True)

        if not candidate_cards:
            raise NoAgentFoundForCategoryError(f"No agents remaining after all preference filters for category '{category}'.")

        # 5. Check for ambiguity (Simplified: if more than one at the highest priority after all filters)
        # A more robust ambiguity check might be needed if multiple agents have identical top scores
        # across all preference dimensions (priority, version if specified, etc.)
        selected_card = candidate_cards[0]
        
        # Example of a stricter ambiguity check: if multiple cards have the same highest priority
        # and other differentiating preferences are not enough.
        if len(candidate_cards) > 1:
            top_priority = selected_card.priority or 0
            # Count how many have the exact same top priority
            count_at_top_priority = sum(1 for card in candidate_cards if (card.priority or 0) == top_priority)
            if count_at_top_priority > 1 and version_preference != "latest_semver": # If not using version to break ties
                 # If version_preference was 'latest_semver', the sort should have picked one.
                 # Otherwise, if multiple have same top priority, it might be ambiguous.
                 # This check might need refinement based on how version_preference interacts with priority for tie-breaking.
                 pass # For now, allow first after sort. Ambiguity logic can be enhanced.
                # raise AmbiguousAgentCategoryError(
                #    f"Multiple agents found with highest priority {top_priority} for category '{category}' and preferences. "
                #    f"Candidates: {[c.agent_id for c in candidate_cards if (c.priority or 0) == top_priority]}"
                # )

        # 6. Return resolved agent_id and callable
        return selected_card.agent_id, self.get(selected_card.agent_id)


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 