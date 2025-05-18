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

from typing import Callable, Dict, Protocol, runtime_checkable, Optional, List, Any, Tuple, Union
from semver import VersionInfo
from .agent_registry import AgentCard
import logging
import asyncio
import functools
import importlib
import sys
import inspect

# Define logger at the module level
logger = logging.getLogger(__name__)
# --- TEMPORARY DEBUGGING CODE HAS BEEN REMOVED ---

# For type hinting AgentCallable
# AgentCallable = Callable[..., Any]  # Simple version
# More precise: Callable[Concatenate[InputSchema, P], OutputSchema] or similar with Pydantic models
# For now, stick to Any to avoid over-complicating until fully typed Agent execution is ready.
AgentCallable = Callable[..., Any]
AgentFallbackItem = Union[AgentCallable, type] # Can be a direct callable or a class to instantiate

# --- DIAGNOSTIC CODE AT THE TOP OF agent_resolver.py ---
# print("--- DIAGNOSING chungoid.utils.agent_resolver (Top of agent_resolver.py) ---")
# print(f"Python Executable: {sys.executable}")
# print(f"Initial sys.path: {sys.path}")
# print(f"os.getcwd(): {os.getcwd()}")
# print(f"__file__ (agent_resolver.py): {__file__}")

# # Try to see where 'chungoid' itself is found from
# try:
#     print("Relevant sys.path entries for 'chungoid' (in agent_resolver.py):")
#     for p in sys.path:
#         if 'chungoid' in p.lower() or 'site-packages' in p.lower() or p == os.getcwd() or '.local/pipx/venvs' in p.lower():
#             print(f"  - {p}")

#     import chungoid
#     print(f"Found chungoid (in agent_resolver.py): {chungoid.__file__ if hasattr(chungoid, '__file__') else 'Namespace package'}")
#     if hasattr(chungoid, '__path__'):
#         print(f"chungoid.__path__ (in agent_resolver.py): {chungoid.__path__}")
#         for p_item_chungoid in chungoid.__path__:
#             print(f"  Contents of chungoid path item {p_item_chungoid}: {os.listdir(p_item_chungoid) if os.path.exists(p_item_chungoid) and os.path.isdir(p_item_chungoid) else 'Not a dir or does not exist'}")
#             utils_dir_path_ar = Path(p_item_chungoid) / 'utils'
#             print(f"    Looking for {utils_dir_path_ar} (from agent_resolver.py): Exists? {utils_dir_path_ar.exists()}, IsDir? {utils_dir_path_ar.is_dir()}")
#             if utils_dir_path_ar.is_dir():
#                  print(f"    Contents of {utils_dir_path_ar} (from agent_resolver.py): {os.listdir(utils_dir_path_ar)}")
# except ModuleNotFoundError as e_chungoid_diag_ar:
#     print(f"DIAGNOSTIC (Agent Resolver): Failed to import top-level 'chungoid' in agent_resolver.py: {e_chungoid_diag_ar}")
# except Exception as e_diag_general_ar:
#     print(f"DIAGNOSTIC (Agent Resolver): General error during diagnostic imports in agent_resolver.py: {e_diag_general_ar}")

# print("--- END DIAGNOSTIC (Top of agent_resolver.py) ---")
# --- END DIAGNOSTIC CODE ---

StageDict = Dict[str, object]


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

    # ------------------------------------------------------------------
    # New helper for orchestrator compatibility
    # ------------------------------------------------------------------
    def get_by_category(
        self,
        category: "AgentCategory | str",
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return the *agent_id* resolved for *category*.

        This is a thin wrapper around :py:meth:`resolve_agent_by_category` so that
        legacy callers (e.g., `AsyncOrchestrator`) can continue using the shorter
        *get_by_category* helper that returns only the identifier.  If resolution
        fails an exception bubbles up unchanged so the caller can handle pause /
        retry logic.
        """
        try:
            resolved_agent_id, _callable = self.resolve_agent_by_category(
                category=str(category), preferences=preferences
            )
            return resolved_agent_id
        except Exception:
            # Propagate – orchestrator has dedicated handling for these errors
            raise


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
        logger.debug(f"RegistryAgentProvider.get() called for identifier: '{identifier}'")
        logger.debug(f"RegistryAgentProvider: Current fallback map keys: {list(self._fallback.keys())}")
        logger.debug(f"RegistryAgentProvider: Checking if '{identifier}' is in fallback map...")

        # Fast path: return cached callable
        # ---- START TEMPORARY MODIFICATION FOR DEBUGGING ----
        # if identifier == "mock_alternative_agent_v1": # Add other problematic mocks if needed
        #     logger.warning(f"RegistryAgentProvider: DEBUG - SKIPPING CACHE for '{identifier}' to force re-resolution.")
        # el
        if identifier in self._cache: # Reverted to original cache check
        # ---- END TEMPORARY MODIFICATION FOR DEBUGGING ----
            logger.debug(f"RegistryAgentProvider: Found '{identifier}' in cache. Returning cached item.")
            return self._cache[identifier]
        logger.debug(f"RegistryAgentProvider: '{identifier}' not in cache.")

        # If the identifier is directly mapped in the fallback, use that callable.
        if identifier in self._fallback:
            logger.info(f"RegistryAgentProvider: Identifier '{identifier}' FOUND in fallback map.")
            potential_item = self._fallback[identifier]
            
            # DETAILED LOGGING for the item itself
            logger.debug(f"RegistryAgentProvider: Fallback item for '{identifier}' IS: {potential_item}")
            logger.debug(f"RegistryAgentProvider: Fallback item type for '{identifier}' IS: {type(potential_item)}")

            is_class_via_isinstance = isinstance(potential_item, type)
            is_class_via_inspect = inspect.isclass(potential_item)
            is_item_callable = callable(potential_item)

            logger.debug(
                f"RegistryAgentProvider: Checks for fallback item '{identifier}':\n"
                f"  - isinstance(item, type)                  = {is_class_via_isinstance}\n"
                f"  - inspect.isclass(item)                   = {is_class_via_inspect}\n"
                f"  - callable(item)                          = {is_item_callable}\n"
                f"  - hasattr(item, '__call__')             = {hasattr(potential_item, '__call__')}\n"
                f"  - inspect.iscoroutinefunction(__call__) = {inspect.iscoroutinefunction(getattr(potential_item, '__call__', None))}\n"
                f"  - hasattr(item, 'invoke_async')         = {hasattr(potential_item, 'invoke_async')}\n"
                f"  - inspect.iscoroutinefunction(invoke_async) = {inspect.iscoroutinefunction(getattr(potential_item, 'invoke_async', None))}"
            )

            if is_class_via_isinstance or is_class_via_inspect:
                logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' identified as a class (isinstance: {is_class_via_isinstance}, inspect: {is_class_via_inspect}). Attempting instantiation.")
                try:
                    # Removed attempt to get agent_config_for_fallback via self._get_agent_config()
                    logger.debug(f"RegistryAgentProvider: Instantiating fallback class '{identifier}' directly (no specific config from resolver).")
                    agent_instance = potential_item() # Directly instantiate
                    
                    logger.info(f"RegistryAgentProvider: Successfully instantiated fallback class for '{identifier}' to: {agent_instance} (type: {type(agent_instance)})")

                    # Now check if the INSTANCE is awaitable or has an awaitable invoke_async
                    if hasattr(agent_instance, '__call__') and callable(agent_instance.__call__) and inspect.iscoroutinefunction(agent_instance.__call__):
                        logger.info(f"RegistryAgentProvider: Instantiated agent '{identifier}' has awaitable __call__. Returning its __call__ method.")
                        self._cache[identifier] = agent_instance.__call__ # Return the bound async method
                        return agent_instance.__call__                   # Return the bound async method
                    elif hasattr(agent_instance, 'invoke_async') and callable(agent_instance.invoke_async) and inspect.iscoroutinefunction(agent_instance.invoke_async):
                        logger.info(f"RegistryAgentProvider: Instantiated agent '{identifier}' has awaitable invoke_async. Returning method.")
                        # Corrected: Instance methods are already bound.
                        self._cache[identifier] = agent_instance.invoke_async
                        return agent_instance.invoke_async
                    elif callable(agent_instance): # If it's some other callable (sync class with __call__ or sync function)
                        logger.info(f"RegistryAgentProvider: Instantiated agent '{identifier}' is a sync callable. Wrapping with to_thread.")
                        wrapped_sync_callable = functools.partial(self._sync_to_async_wrapper, agent_instance)
                        self._cache[identifier] = wrapped_sync_callable
                        return wrapped_sync_callable
                    else:
                        logger.warning(f"RegistryAgentProvider: Instantiated agent '{identifier}' is not directly callable nor has a recognized invoke method. Agent instance: {agent_instance}")
                        # Fall through to card-based if instance isn't suitable
                except Exception as e:
                    logger.error(f"RegistryAgentProvider: Error instantiating or preparing fallback class '{identifier}': {e}", exc_info=True)
            
            # If it wasn't a class, check if the potential_item (already an instance) is directly awaitable or has awaitable invoke_async
            elif hasattr(potential_item, '__call__') and callable(potential_item.__call__) and inspect.iscoroutinefunction(potential_item.__call__):
                logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' (instance) has awaitable __call__. Returning its __call__ method.")
                self._cache[identifier] = potential_item.__call__ # Return the bound async method
                return potential_item.__call__                   # Return the bound async method
            elif hasattr(potential_item, 'invoke_async') and callable(potential_item.invoke_async) and inspect.iscoroutinefunction(potential_item.invoke_async):
                logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' (instance) has awaitable invoke_async. Returning method.")
                # Bind the method to the instance before caching/returning
                # This was functools.partial(potential_item.invoke_async, potential_item) which is incorrect for instance methods
                # Should just be potential_item.invoke_async as it's already bound if it's an instance method
                self._cache[identifier] = potential_item.invoke_async
                return potential_item.invoke_async
            elif inspect.iscoroutinefunction(potential_item): # Check if potential_item itself is an async function
                logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' is an async function. Returning it directly.")
                self._cache[identifier] = potential_item
                return potential_item
            elif callable(potential_item): # Sync instance with __call__, or sync function
                logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' is a sync callable. Wrapping with to_thread.")
                # TODO: Implement or remove _sync_to_async_wrapper. For now, this path will error if hit.
                # For the current MVP, async functions should be caught by the preceding elif.
                wrapped_sync_callable = functools.partial(self._sync_to_async_wrapper, potential_item)
                self._cache[identifier] = wrapped_sync_callable
                return wrapped_sync_callable
            else:
                # ADDING DETAILED LOGS HERE specifically for the instance that's about to fall through
                if not (is_class_via_isinstance or is_class_via_inspect): # Ensure it's an instance
                    logger.warning(f"RegistryAgentProvider: DETAILED CHECK FOR FALLBACK INSTANCE '{identifier}':")
                    has_call_attr = hasattr(potential_item, '__call__')
                    logger.warning(f"  - hasattr(item, '__call__'): {has_call_attr}")
                    if has_call_attr:
                        call_method = getattr(potential_item, '__call__')
                        is_call_method_callable = callable(call_method)
                        is_call_method_coro_func = inspect.iscoroutinefunction(call_method)
                        logger.warning(f"    - callable(item.__call__): {is_call_method_callable}")
                        logger.warning(f"    - inspect.iscoroutinefunction(item.__call__): {is_call_method_coro_func}")
                    
                    has_invoke_async_attr = hasattr(potential_item, 'invoke_async')
                    logger.warning(f"  - hasattr(item, 'invoke_async'): {has_invoke_async_attr}")
                    if has_invoke_async_attr:
                        invoke_method = getattr(potential_item, 'invoke_async')
                        is_invoke_method_callable = callable(invoke_method)
                        is_invoke_method_coro_func = inspect.iscoroutinefunction(invoke_method)
                        logger.warning(f"    - callable(item.invoke_async): {is_invoke_method_callable}")
                        logger.warning(f"    - inspect.iscoroutinefunction(item.invoke_async): {is_invoke_method_coro_func}")
                    
                    is_item_itself_callable = callable(potential_item)
                    logger.warning(f"  - callable(item): {is_item_itself_callable}")

                logger.warning(f"RegistryAgentProvider: Fallback item '{identifier}' was not identified as a class and is not directly callable or has a recognized invoke method. Actual item: {potential_item}. Falling through to card-based resolution.")
                # Fall through to card-based if item isn't suitable

        logger.debug(f"RegistryAgentProvider: Identifier '{identifier}' NOT returned from fallback map. Proceeding to card-based resolution.")

        # If not in fallback or fallback processing failed/skipped, check registry card
        logger.debug(f"RegistryAgentProvider: Proceeding to registry lookup for '{identifier}'.")
        card = self._registry.get(identifier)
        if card is None:
            # Not in fallback and no card found in registry.
            raise KeyError(f"Agent '{identifier}' not found in registry or direct fallback.")

        # Card exists. Try to use MCP tool dispatch if possible.
        if self._CoreMCPClient and card.tool_names:
            tool_name = card.tool_names[0]
            logger.debug(f"RegistryAgentProvider: Card for '{identifier}' has MCP tool: {tool_name}. Preparing MCP invoke.")

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

        # NEW: If not an MCP tool, check if it's a known local Python mock agent
        try:
            # Lazy import to avoid circular dependencies and keep it contained
            from chungoid.runtime.agents.mocks.testing_mock_agents import ALL_MOCK_TESTING_AGENTS
            known_mock_classes = {agent_cls.AGENT_ID: agent_cls for agent_cls in ALL_MOCK_TESTING_AGENTS}

            if identifier in known_mock_classes:
                agent_cls = known_mock_classes[identifier]
                logger.debug(f"Agent ID '{identifier}' matches a known local mock agent class. Instantiating.")
                agent_instance = agent_cls() # Instantiate the agent class
                actual_callable = getattr(agent_instance, 'invoke_async') # Get the bound invoke_async method
                logger.debug(f"RegistryAgentProvider: For '{identifier}', got actual_callable of type: {type(actual_callable)}, iscoroutinefunction: {inspect.iscoroutinefunction(actual_callable)}")
                self._cache[identifier] = actual_callable # Cache the actual callable
                return actual_callable # Return the actual callable
            else:
                # This else means identifier was not in known_mock_classes
                logger.warning(f"RegistryAgentProvider: Agent ID '{identifier}' was not found in known_mock_classes. Keys available: {list(known_mock_classes.keys())}")
                _resolution_issue_reason = f"Identifier '{identifier}' not in known_mock_classes."
        except ImportError as ie: # Capture specific exception
            # This means Line A failed.
            logger.warning(f"Could not import ALL_MOCK_TESTING_AGENTS. Local mock agent resolution will be skipped. Error: {ie}")
            _resolution_issue_reason = f"ImportError for ALL_MOCK_TESTING_AGENTS: {ie}"
        except Exception as e:
            # This means Line B, C, D, E, F, G, or H failed with something other than ImportError from Line A.
            logger.error(f"Error during local mock agent resolution for '{identifier}': {e}", exc_info=True)
            _resolution_issue_reason = f"Exception during local mock agent resolution for '{identifier}': {e}"
            # Fall through to stub if dynamic instantiation fails

        # Card exists, not in fallback, not an MCP tool, and not resolved as a known local mock.
        # Return the stub as the last resort.
        logger.warning(f"RegistryAgentProvider: Agent '{identifier}' resolved via card but no MCP tool and not a recognized local Python agent. Reason for no local resolution: {_resolution_issue_reason if '_resolution_issue_reason' in locals() else 'Unknown (fell through fallback and local mock checks)'}. Returning synchronous stub.")
        def _stub(stage: StageDict, full_context: Optional[Dict[str, Any]] = None) -> Dict[str, object]:  # type: ignore[override]
            logger.debug(f"RegistryAgentProvider: Executing STUB for '{identifier}'.")
            return {"agent_id": identifier, "stage_inputs": stage.get("inputs", {}), "message": "Agent is a stub (card found, no tool/fallback)."}

        self._cache[identifier] = _stub
        logger.debug(f"RegistryAgentProvider: Caching and returning STUB for '{identifier}'.")
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
        logger.info(f"Attempting to resolve agent for category: '{category}' with preferences: {preferences}")

        if preferences is None:
            preferences = {}

        all_cards: List[AgentCard] = self._registry.list() 
        logger.info(f"Found {len(all_cards)} cards in registry. Checking for category '{category}'.")

        candidate_cards_after_category_filter: List[AgentCard] = []
        for card in all_cards:
            is_match = False 
            if card.categories: 
                if category in card.categories:
                    is_match = True
            
            if is_match:
                candidate_cards_after_category_filter.append(card)

        if not candidate_cards_after_category_filter:
            logger.warning(f"No candidate cards after category filter for '{category}'. all_cards count: {len(all_cards)}") 
            raise NoAgentFoundForCategoryError(f"No agents found in category '{category}'.")

        # 2. Apply capability_profile_match
        profile_match_prefs = preferences.get("capability_profile_match", {})
        if profile_match_prefs:
            filtered_by_profile: List[AgentCard] = []
            for card in candidate_cards_after_category_filter:
                match = True
                if not card.capability_profile: # Agent has no capability profile, so cannot match if prefs exist
                    match = False
                else:
                    for pref_key, pref_expected_value in profile_match_prefs.items():
                        agent_capability_value = None
                        # Map preference keys to actual capability_profile keys
                        if pref_key == "language":
                            agent_capability_value = card.capability_profile.get("language_support")
                        elif pref_key == "framework":
                            agent_capability_value = card.capability_profile.get("target_frameworks")
                        # Add other specific mappings here if needed
                        # else: # Fallback to direct key match if no specific mapping
                        #     agent_capability_value = card.capability_profile.get(pref_key)

                        # If after mapping (or direct access if no mapping applied), the key isn't in agent's profile
                        if agent_capability_value is None and not (pref_key == "language" or pref_key == "framework"):
                             # If it wasn't a mapped key, try direct access as a fallback
                             agent_capability_value = card.capability_profile.get(pref_key)
                        
                        if agent_capability_value is None: # Capability not present in agent card after mapping or direct lookup
                            match = False
                            break
                        
                        # Refined comparison logic
                        if isinstance(pref_expected_value, list): # Preference is a list (e.g., for edit_action_support)
                            if not isinstance(agent_capability_value, list):
                                match = False # Agent's capability must also be a list to compare all items
                                break
                            # Check if all items in pref_expected_value are in agent_capability_value
                            if not all(item in agent_capability_value for item in pref_expected_value):
                                match = False
                                break
                        elif isinstance(agent_capability_value, list): # Agent capability is a list, preference is single value
                            if pref_expected_value not in agent_capability_value:
                                match = False
                                break
                        # Direct comparison if both are single values (or other non-list types)
                        elif agent_capability_value != pref_expected_value:
                            match = False
                            break
                if match:
                    filtered_by_profile.append(card)
            candidate_cards_after_category_filter = filtered_by_profile
        
        if not candidate_cards_after_category_filter:
            raise NoAgentFoundForCategoryError(f"No agents in category '{category}' match capability profile: {profile_match_prefs}")

        # 3. Apply priority_gte
        priority_gte = preferences.get("priority_gte")
        if priority_gte is not None:
            candidate_cards_after_category_filter = [card for card in candidate_cards_after_category_filter if card.priority is not None and card.priority >= priority_gte]

        if not candidate_cards_after_category_filter:
            raise NoAgentFoundForCategoryError(f"No agents in category '{category}' (after profile match) meet priority_gte: {priority_gte}")

        # 4. Handle version_preference (e.g., "latest_semver")
        # This is a simplified version preference handling. More robust parsing might be needed.
        version_preference = preferences.get("version_preference")
        if version_preference == "latest_semver" and candidate_cards_after_category_filter:
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

            candidate_cards_after_category_filter.sort(key=lambda c: (c.priority or 0, get_semver(c) or VersionInfo(0)), reverse=True)
            
            if candidate_cards_after_category_filter: # If any candidates remain after sorting
                 # Check for ties in priority and version if that constitutes ambiguity
                best_card = candidate_cards_after_category_filter[0]
                best_priority = best_card.priority or 0
                best_version = get_semver(best_card)
                
                # If there are multiple cards with the same highest priority and version (if version is used for sorting)
                # this could be an ambiguity. For now, taking the first one after sorting by priority then version.
                # A more strict ambiguity check might be needed based on requirements.
                pass # First one is taken by default after sort

        elif candidate_cards_after_category_filter: # Default sort by priority if no specific version preference
            candidate_cards_after_category_filter.sort(key=lambda c: c.priority or 0, reverse=True)

        if not candidate_cards_after_category_filter:
            raise NoAgentFoundForCategoryError(f"No agents remaining after all preference filters for category '{category}'.")

        # 5. Check for ambiguity (Simplified: if more than one at the highest priority after all filters)
        # A more robust ambiguity check might be needed if multiple agents have identical top scores
        # across all preference dimensions (priority, version if specified, etc.)
        selected_card = candidate_cards_after_category_filter[0]
        
        # Example of a stricter ambiguity check: if multiple cards have the same highest priority
        # and other differentiating preferences are not enough.
        if len(candidate_cards_after_category_filter) > 1:
            top_priority = selected_card.priority or 0
            # Count how many have the exact same top priority
            count_at_top_priority = sum(1 for card in candidate_cards_after_category_filter if (card.priority or 0) == top_priority)
            if count_at_top_priority > 1 and version_preference != "latest_semver": # If not using version to break ties
                 # If version_preference was 'latest_semver', the sort should have picked one.
                 # Otherwise, if multiple have same top priority, it might be ambiguous.
                 # This check might need refinement based on how version_preference interacts with priority for tie-breaking.
                 pass # For now, allow first after sort. Ambiguity logic can be enhanced.
                # raise AmbiguousAgentCategoryError(
                #    f"Multiple agents found with highest priority {top_priority} for category '{category}' and preferences. "
                #    f"Candidates: {[c.agent_id for c in candidate_cards_after_category_filter if (c.priority or 0) == top_priority]}"
                # )

        # 6. Return resolved agent_id and callable
        return selected_card.agent_id, self.get(selected_card.agent_id)

    # ------------------------------------------------------------------
    # Sync helper for compatibility with orchestrator
    # ------------------------------------------------------------------
    def get_by_category(
        self,
        category: "AgentCategory | str",
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Return just the *agent_id* for the resolved category.

        The orchestrator historically expected this synchronous helper that
        returns only the identifier.  Internally we leverage the richer
        *resolve_agent_by_category* implementation to reuse the same logic.
        """
        resolved_agent_id, _callable = self.resolve_agent_by_category(
            str(category), preferences=preferences
        )
        return resolved_agent_id


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 