from __future__ import annotations

"""Adapters that resolve agent identifiers to callables.

Two use-cases:
• **DictAgentProvider** – thin wrapper around an in-memory mapping used by
  existing tests/demo scripts.
• **RegistryAgentProvider** – consults `InMemoryAgentRegistry` (registry-first)
  and optionally falls back to a supplied mapping.

Both implement the `AgentProvider` protocol expected by the refactored
`FlowExecutor`.
"""

from typing import Callable, Dict, Protocol, runtime_checkable, Optional, List, Any, Tuple, Union, Type, cast, Coroutine
from pathlib import Path
from semver import VersionInfo
from .agent_registry import AgentCard, AgentRegistry
from chungoid.schemas.common import AgentCallable
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
import logging
import asyncio
import functools
import importlib
import sys
import inspect
import traceback
import re

# ADDED: Import for registry-first architecture
from chungoid.registry.in_memory_agent_registry import InMemoryAgentRegistry

# ADDED: Import LLMManager for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from chungoid.utils.llm_provider import LLMManager

# REMOVED: All individual agent imports - registry-first architecture handles this
# The registry will dynamically import and instantiate agents as needed

# Define logger at the module level
logger = logging.getLogger(__name__)

# For type hinting AgentCallable
AgentFallbackItem = Union[AgentCallable, type] # Can be a direct callable or a class to instantiate

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
        category: str,
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


class NoAgentFoundError(Exception):
    """Raised when an agent cannot be resolved or instantiated for various reasons."""
    pass

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
        registry: "InMemoryAgentRegistry",
        fallback: Optional[Dict[str, AgentCallable]] = None,
        # MODIFIED: Accept either LLMManager or LLMProvider for backward compatibility
        llm_provider: Optional[Union["LLMManager", LLMProvider]] = None,
        prompt_manager: Optional[PromptManager] = None,
    ) -> None:
        # MODIFIED: Import the correct registry types for registry-first architecture
        from chungoid.registry.in_memory_agent_registry import InMemoryAgentRegistry
        from chungoid.utils.llm_provider import LLMManager
        from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1

        # FIXED: Accept InMemoryAgentRegistry for registry-first architecture
        if not isinstance(registry, InMemoryAgentRegistry):
            logger.error(f"RegistryAgentProvider __init__: PASSED REGISTRY TYPE IS {type(registry)}. EXPECTED InMemoryAgentRegistry.") # DIAGNOSTIC
            raise TypeError("registry must be an InMemoryAgentRegistry instance")
        self._registry = registry
        logger.info(f"RegistryAgentProvider __init__: self._registry SET to {type(self._registry)}, id: {id(self._registry)}") # DIAGNOSTIC

        self._fallback: Dict[str, AgentCallable] = fallback or {}
        self._cache: Dict[str, AgentCallable] = {}
        
        # MODIFIED: Handle both LLMManager and LLMProvider types
        if isinstance(llm_provider, LLMManager):
            # Extract the actual LLMProvider from LLMManager
            self._llm_provider = llm_provider.actual_provider
            logger.info(f"RegistryAgentProvider: Received LLMManager, extracted underlying {type(self._llm_provider).__name__}")
        else:
            # Direct LLMProvider or None
            self._llm_provider = llm_provider
            if llm_provider:
                logger.info(f"RegistryAgentProvider: Received direct LLMProvider: {type(self._llm_provider).__name__}")
        
        self._prompt_manager = prompt_manager
        # ADDED: Store shared_context if provided, or initialize an empty one
        # This is a slight change in approach: self._shared_context will be set by the orchestrator
        # via a new method or during get() call, rather than __init__
        self._orchestrator_shared_context: Optional[Dict[str, Any]] = None 


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
    def get(self, identifier: str, category: Optional[str] = None, shared_context: Optional[Dict[str, Any]] = None) -> Callable[..., Coroutine[Any, Any, Any]]:
        """
        Retrieves an agent's callable (invoke_async method) by its ID or category.
        Pure registry-first architecture - no hardcoded agent references.
        """
        # Stash the shared_context for use during agent instantiation in helper methods
        self._orchestrator_shared_context = shared_context
        
        logger.debug(f"RegistryAgentProvider GET: identifier='{identifier}', category='{category}', has_shared_context={shared_context is not None}")
        logger.info(f"RegistryAgentProvider GET: hasattr(self, '_registry'): {hasattr(self, '_registry')}")
        if hasattr(self, '_registry'):
            logger.info(f"RegistryAgentProvider GET: self._registry type: {type(self._registry)}, id: {id(self._registry)}")

        try:
            # Check if the agent is in the fallback map first (for backward compatibility)
            if identifier in self._fallback:
                potential_item = self._fallback[identifier]
                logger.info(f"RegistryAgentProvider: Identifier '{identifier}' FOUND in fallback map.")
                logger.info(f"RegistryAgentProvider: Fallback item type: {type(potential_item)}, name: {getattr(potential_item, '__name__', 'N/A')}")

                if inspect.isclass(potential_item) and issubclass(potential_item, ProtocolAwareAgent):
                    logger.info(f"RegistryAgentProvider: Fallback item for '{identifier}' is a ProtocolAwareAgent class. Instantiating...")
                    agent_instance = self._instantiate_agent_class(potential_item, identifier)
                    if agent_instance and hasattr(agent_instance, 'invoke_async'):
                        return agent_instance.invoke_async
                elif isinstance(potential_item, ProtocolAwareAgent):
                    logger.info(f"RegistryAgentProvider: Fallback item for '{identifier}' is already an instantiated ProtocolAwareAgent.")
                    if hasattr(potential_item, 'invoke_async'):
                        return potential_item.invoke_async
                elif callable(potential_item):
                    logger.info(f"RegistryAgentProvider: Fallback item for '{identifier}' is a direct callable.")
                    if asyncio.iscoroutinefunction(potential_item):
                        return potential_item
                    else:
                        logger.warning(f"RegistryAgentProvider: Fallback callable for '{identifier}' is not async. May cause issues.")
                        return potential_item

            # Registry-first approach: Query the registry for the agent
            logger.info(f"RegistryAgentProvider: Identifier '{identifier}' not found in fallback. Querying registry...")
            
            if not self._registry:
                raise NoAgentFoundError(f"Registry not available and agent '{identifier}' not in fallback map.")
            
            # Get agent from registry
            agent_class = self._registry.get_agent(identifier)
            if agent_class:
                logger.info(f"RegistryAgentProvider: Found agent '{identifier}' in registry: {agent_class.__name__}")
                
                # Instantiate the agent class
                agent_instance = self._instantiate_agent_class(agent_class, identifier)
                if agent_instance:
                    logger.info(f"RegistryAgentProvider: Successfully instantiated agent '{identifier}' from registry.")
                    # Return protocol adapter instead of invoke_async
                    from .protocol_adapter import ProtocolExecutionAdapter
                    return ProtocolExecutionAdapter(agent_instance)
                else:
                    raise NoAgentFoundError(f"Failed to instantiate agent '{identifier}' from registry.")
            else:
                raise NoAgentFoundError(f"Agent '{identifier}' not found in registry or fallback map.")

        except NoAgentFoundError:
            raise
        except Exception as exc:
            logger.error(f"RegistryAgentProvider: Unexpected error resolving/instantiating agent '{identifier}': {exc}", exc_info=True)
            raise NoAgentFoundError(f"Unexpected error resolving/instantiating agent '{identifier}': {exc}") from exc

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
        """Resolves an agent by category from the AgentRegistry, applying preferences."""
        logger.debug(f"Attempting to resolve agent by category: {category} with preferences: {preferences}")

        if not self._registry:
            logger.error("AgentRegistry ('_registry') not initialized in RegistryAgentProvider.")
            raise NoAgentFoundForCategoryError(f"AgentRegistry not available to search category '{category}'.")

        # Defer actual AgentRegistry import and usage to avoid circular deps if possible
        # from .agent_registry import AgentRegistry # Already imported at top as ConcreteAgentRegistry
        # if not isinstance(self._registry, AgentRegistry):
        #     logger.error(f"_registry is not an AgentRegistry instance, it is {type(self._registry)}")
        #     raise NoAgentFoundForCategoryError(f"_registry is not an AgentRegistry instance for category '{category}'.")

        try:
            # TODO: Implement preference-based selection logic here
            # For now, just get all agents in the category and pick the first one if any
            # This mimics the simplified get_by_category logic for now
            # agent_cards = self._registry.get_agents_by_category(category) # CORRECTED
            # Using the new method that handles the Chroma query and AgentCard creation
            agent_cards = await self._registry.async_get_agents_by_category(category) # CORRECTED, and now async

            if not agent_cards:
                logger.warning(f"No agents found in category '{category}'.")
                raise NoAgentFoundForCategoryError(f"No agents found in category '{category}'.")

            # 2. Apply capability_profile_match
            profile_match_prefs = preferences.get("capability_profile_match", {})
            if profile_match_prefs:
                filtered_by_profile: List[AgentCard] = []
                for card in agent_cards:
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
                agent_cards = filtered_by_profile
            
            if not agent_cards:
                raise NoAgentFoundForCategoryError(f"No agents in category '{category}' match capability profile: {profile_match_prefs}")

            # 3. Apply priority_gte
            priority_gte = preferences.get("priority_gte")
            if priority_gte is not None:
                agent_cards = [card for card in agent_cards if card.priority is not None and card.priority >= priority_gte]

            if not agent_cards:
                raise NoAgentFoundForCategoryError(f"No agents in category '{category}' (after profile match) meet priority_gte: {priority_gte}")

            # 4. Handle version_preference (e.g., "latest_semver")
            # This is a simplified version preference handling. More robust parsing might be needed.
            version_preference = preferences.get("version_preference")
            if version_preference == "latest_semver" and agent_cards:
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

                agent_cards.sort(key=lambda c: (c.priority or 0, get_semver(c) or VersionInfo(0)), reverse=True)
                
                if agent_cards: # If any candidates remain after sorting
                     # Check for ties in priority and version if that constitutes ambiguity
                    best_card = agent_cards[0]
                    best_priority = best_card.priority or 0
                    best_version = get_semver(best_card)
                    
                    # If there are multiple cards with the same highest priority and version (if version is used for sorting)
                    # this could be an ambiguity. For now, taking the first one after sorting by priority then version.
                    # A more strict ambiguity check might be needed based on requirements.
                    pass # First one is taken by default after sort

            elif agent_cards: # Default sort by priority if no specific version preference
                agent_cards.sort(key=lambda c: c.priority or 0, reverse=True)

            if not agent_cards:
                raise NoAgentFoundForCategoryError(f"No agents remaining after all preference filters for category '{category}'.")

            # 5. Check for ambiguity (Simplified: if more than one at the highest priority after all filters)
            # A more robust ambiguity check might be needed if multiple agents have identical top scores
            # across all preference dimensions (priority, version if specified, etc.)
            selected_card = agent_cards[0]
            
            # Example of a stricter ambiguity check: if multiple cards have the same highest priority
            # and other differentiating preferences are not enough.
            if len(agent_cards) > 1:
                top_priority = selected_card.priority or 0
                # Count how many have the exact same top priority
                count_at_top_priority = sum(1 for card in agent_cards if (card.priority or 0) == top_priority)
                if count_at_top_priority > 1 and version_preference != "latest_semver": # If not using version to break ties
                     # If version_preference was 'latest_semver', the sort should have picked one.
                     # Otherwise, if multiple have same top priority, it might be ambiguous.
                     # This check might need refinement based on how version_preference interacts with priority for tie-breaking.
                     pass # For now, allow first after sort. Ambiguity logic can be enhanced.
                # raise AmbiguousAgentCategoryError(
                #    f"Multiple agents found with highest priority {top_priority} for category '{category}' and preferences. "
                #    f"Candidates: {[c.agent_id for c in agent_cards if (c.priority or 0) == top_priority]}"
                # )

            # 6. Return resolved agent_id and callable
            return selected_card.agent_id, self.get(selected_card.agent_id)

        except NoAgentFoundError:
            logger.error(f"Agent '{category}' not found by RegistryAgentProvider.")
            raise
        except AmbiguousAgentCategoryError:
            logger.error(f"Agent category '{category}' is ambiguous.")
            raise
        except Exception as e:
            logger.error(f"RegistryAgentProvider: Unexpected error resolving agent '{category}': {e}")
            logger.debug(traceback.format_exc())
            raise NoAgentFoundForCategoryError(f"Unexpected error resolving agent '{category}': {e}")

    # ------------------------------------------------------------------
    # Sync helper for compatibility with orchestrator
    # ------------------------------------------------------------------
    def get_by_category(
        self,
        category: str,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Resolves an agent ID by category, potentially using preferences.

        Returns the agent ID or None if not found/ambiguous without more logic.
        This is a simplified version; a real implementation would use preferences.
        """
        logger.debug(f"RegistryAgentProvider: get_by_category: '{category}', preferences: {preferences}")
        if not self._registry:
            logger.warning("RegistryAgentProvider: AgentRegistry ('_registry') not initialized.")
            return None

        # agent_cards = self._registry.get_agents_by_category(category) # CORRECTED
        # This method should probably be async if it involves DB queries, but keeping sync for now
        # to match existing signature. If async_get_agents_by_category is preferred, this needs refactor.
        # For simplicity, let's assume there's a sync version or adapt.
        # Given the file structure, assuming agent_registry has a sync method or this part is illustrative.
        # Reverting to a conceptual sync placeholder call, as `async_get_agents_by_category` would require this method to be async.
        try:
            # This is a conceptual placeholder for how it *might* work synchronously
            # In reality, self._registry.get_agents_by_category would be called.
            # The actual `AgentRegistry.get_agents_by_category` in `agent_registry.py` appears to be synchronous.
            agent_cards = self._registry.get_agents_by_category(category) # CORRECTED
        except Exception as e:
            logger.error(f"Error calling self._registry.get_agents_by_category for category '{category}': {e}")
            return None

        if not agent_cards:
            logger.warning(f"No agents found in category '{category}'.")
            return None

        # 2. Apply capability_profile_match
        profile_match_prefs = preferences.get("capability_profile_match", {})
        if profile_match_prefs:
            filtered_by_profile: List[AgentCard] = []
            for card in agent_cards:
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
            agent_cards = filtered_by_profile
        
        if not agent_cards:
            logger.warning(f"No agents found in category '{category}' after capability profile match.")
            return None

        # 3. Apply priority_gte
        priority_gte = preferences.get("priority_gte")
        if priority_gte is not None:
            agent_cards = [card for card in agent_cards if card.priority is not None and card.priority >= priority_gte]

        if not agent_cards:
            logger.warning(f"No agents found in category '{category}' after priority_gte filter.")
            return None

        # 4. Handle version_preference (e.g., "latest_semver")
        # This is a simplified version preference handling. More robust parsing might be needed.
        version_preference = preferences.get("version_preference")
        if version_preference == "latest_semver" and agent_cards:
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

            agent_cards.sort(key=lambda c: (c.priority or 0, get_semver(c) or VersionInfo(0)), reverse=True)
            
            if agent_cards: # If any candidates remain after sorting
                 # Check for ties in priority and version if that constitutes ambiguity
                best_card = agent_cards[0]
                best_priority = best_card.priority or 0
                best_version = get_semver(best_card)
                
                # If there are multiple cards with the same highest priority and version (if version is used for sorting)
                # this could be an ambiguity. For now, taking the first one after sorting by priority then version.
                # A more strict ambiguity check might be needed based on requirements.
                pass # First one is taken by default after sort

        elif agent_cards: # Default sort by priority if no specific version preference
            agent_cards.sort(key=lambda c: c.priority or 0, reverse=True)

        if not agent_cards:
            logger.warning(f"No agents remaining after all preference filters for category '{category}'.")
            return None

        # 5. Check for ambiguity (Simplified: if more than one at the highest priority after all filters)
        # A more robust ambiguity check might be needed if multiple agents have identical top scores
        # across all preference dimensions (priority, version if specified, etc.)
        selected_card = agent_cards[0]
        
        # Example of a stricter ambiguity check: if multiple cards have the same highest priority
        # and other differentiating preferences are not enough.
        if len(agent_cards) > 1:
            top_priority = selected_card.priority or 0
            # Count how many have the exact same top priority
            count_at_top_priority = sum(1 for card in agent_cards if (card.priority or 0) == top_priority)
            if count_at_top_priority > 1 and version_preference != "latest_semver": # If not using version to break ties
                 # If version_preference was 'latest_semver', the sort should have picked one.
                 # Otherwise, if multiple have same top priority, it might be ambiguous.
                 # This check might need refinement based on how version_preference interacts with priority for tie-breaking.
                 pass # For now, allow first after sort. Ambiguity logic can be enhanced.
                # raise AmbiguousAgentCategoryError(
                #    f"Multiple agents found with highest priority {top_priority} for category '{category}' and preferences. "
                #    f"Candidates: {[c.agent_id for c in agent_cards if (c.priority or 0) == top_priority]}"
                # )

        # 6. Return resolved agent_id and callable
        return selected_card.agent_id

    def get_agent_callable(self, agent_id: str, shared_context: Optional[Dict[str, Any]] = None) -> AgentCallable:
        """
        Resolves an agent ID to its `invoke_async` callable.
        Handles instantiation of ProtocolAwareAgent subclasses and provides necessary context.
        MODIFIED: Accepts shared_context and passes it for agent instantiation.
        """
        logger.debug(f"Attempting to get callable for agent_id: {agent_id}")
        
        # Stash/update shared_context for this call sequence
        if shared_context:
            self._orchestrator_shared_context = shared_context

        if agent_id in self._cache:
            logger.debug(f"Cache hit for agent_id: {agent_id}")
            # TODO: Ensure cached instances are also updated with fresh shared_context if necessary,
            # or that system_context is mutable and updated. For now, assumes direct callable is fine.
            # If the cached item is an instance method, it will use its original system_context.
            # This might be an issue if shared_context changes between calls for the same agent_id.
            # For now, we are returning the invoke_async method of a NEW instance if it's a ProtocolAwareAgent type.
            # So, the cache here is more for direct callables than for ProtocolAwareAgent classes.
            # Let's refine caching for ProtocolAwareAgent types.
            
            # If the cached item is a ProtocolAwareAgent class, we need to instantiate it.
            cached_item = self._cache[agent_id]
            if inspect.isclass(cached_item) and issubclass(cached_item, ProtocolAwareAgent):
                logger.debug(f"Cache hit for {agent_id} is a ProtocolAwareAgent class. Instantiating with current shared_context.")
                agent_instance = self._instantiate_agent_class(cached_item, agent_id)
                if agent_instance:
                    return getattr(agent_instance, "invoke_async")
                else:
                    # Fall through if instantiation failed, should not happen if class is valid
                    logger.error(f"Failed to re-instantiate cached ProtocolAwareAgent class {agent_id}")
            elif callable(cached_item): # It's a direct callable (function or method of an already instantiated object)
                return cached_item
            # else: fall through to standard resolution

        # Try fallback map first (often contains system agents or mocks)
        if agent_id in self._fallback:
            potential_item = self._fallback[agent_id]
            logger.debug(f"Found '{agent_id}' in fallback map. Type: {type(potential_item)}")

            if inspect.isclass(potential_item) and issubclass(potential_item, ProtocolAwareAgent):
                agent_instance = self._instantiate_agent_class(potential_item, agent_id)
                if agent_instance:
                    # Cache the class itself for future re-instantiation
                    # self._cache[agent_id] = potential_item # Caching class
                    # Or cache the method of the new instance?
                    # For now, let's not cache agent instances or their methods to ensure fresh context
                    return getattr(agent_instance, "invoke_async")
            elif callable(potential_item): # Direct callable
                self._cache[agent_id] = potential_item
                return potential_item
            else:
                logger.warning(f"Item for '{agent_id}' in fallback map is not a callable or ProtocolAwareAgent class: {type(potential_item)}")

        # If not in fallback or fallback item was not suitable, try AgentRegistry (Chroma)
        # This part of the logic would involve querying self._registry
        # For now, let's assume if it's not in fallback, we raise an error if it's a known core agent
        # that should have been in fallback. This mimics current limited scope.
        
        # --- SIMULATED REGISTRY LOOKUP / ADVANCED INSTANTIATION ---
        # In a full implementation, this would query self._registry:
        # agent_card = self._registry.get_agent_card(agent_id)
        # if agent_card and agent_card.fully_qualified_class_name:
        #     try:
        #         module_path, class_name = agent_card.fully_qualified_class_name.rsplit('.', 1)
        #         module = importlib.import_module(module_path)
        #         agent_class = getattr(module, class_name)
        #             agent_instance = self._instantiate_agent_class(agent_class, agent_id, agent_card=agent_card)
        #             if agent_instance:
        #                 return getattr(agent_instance, "invoke_async")
        #     except Exception as e:
        #         logger.error(f"Error dynamically importing or instantiating agent {agent_id} from card: {e}")
        #         raise NoAgentFoundError(f"Agent {agent_id} found in registry but failed to load: {e}") from e

        # If we reach here, the agent_id was not resolved by any means
        raise NoAgentFoundError(f"Agent '{agent_id}' could not be resolved to a callable or instantiable class.")

    def _instantiate_agent_class(
        self, 
        agent_class: Type[ProtocolAwareAgent], 
        agent_id: str,
        agent_card: Optional[AgentCard] = None
    ) -> Optional[ProtocolAwareAgent]:
        """
        Helper to instantiate a ProtocolAwareAgent subclass, injecting necessary dependencies.
        It uses self._orchestrator_shared_context which should be set before calling this.
        """
        logger.debug(f"Instantiating agent class {agent_class.__name__} for agent_id '{agent_id}'")
        
        # Prepare system_context for the agent
        agent_system_context = {}
        if self._orchestrator_shared_context:
            # Pass specific items from orchestrator's shared_context to agent's system_context
            if 'logger' in self._orchestrator_shared_context:
                agent_system_context['logger'] = self._orchestrator_shared_context['logger']
            if 'llm_provider' in self._orchestrator_shared_context: # Agents might need this directly
                 agent_system_context['llm_provider'] = self._orchestrator_shared_context['llm_provider']
            if 'prompt_manager' in self._orchestrator_shared_context: # Agents might need this directly
                 agent_system_context['prompt_manager'] = self._orchestrator_shared_context['prompt_manager']
            # Add other necessary shared components if agents expect them in system_context
        else:
            logger.warning(f"Orchestrator shared_context not available while instantiating {agent_id}. Agent's system_context will be minimal.")
            # Fallback: provide a default logger if none from shared_context
            agent_system_context['logger'] = logging.getLogger(f"agent.{agent_class.__name__}")


        # Prepare agent_init_params (e.g., from agent_card or defaults)
        agent_init_params: Dict[str, Any] = {}
        
        # Example: If agent_card has specific init_params defined
        # if agent_card and agent_card.agent_specific_config:
        #     agent_init_params.update(agent_card.agent_specific_config.get("init_params", {}))

        # Add dependencies if the agent's __init__ expects them and they are available in the provider
        # This requires inspecting the agent_class.__init__ signature
        init_signature = inspect.signature(agent_class.__init__)
        constructor_params = init_signature.parameters

        # Explicitly pass provider-held dependencies if agent expects them
        if 'llm_provider' in constructor_params and self._llm_provider:
            agent_init_params['llm_provider'] = self._llm_provider
            logger.debug(f"Added self._llm_provider to init_params for {agent_class.__name__}")
        # ADDED: Support for legacy agents that use llm_client instead of llm_provider
        if 'llm_client' in constructor_params and self._llm_provider:
            agent_init_params['llm_client'] = self._llm_provider
            logger.debug(f"Added self._llm_provider as llm_client to init_params for {agent_class.__name__}")
        if 'prompt_manager' in constructor_params and self._prompt_manager:
            agent_init_params['prompt_manager'] = self._prompt_manager
            logger.debug(f"Added self._prompt_manager to init_params for {agent_class.__name__}")

        # MODIFIED: Correct injection of ProjectChromaManagerAgent
        # Check if the agent's constructor expects 'pcma_agent' or 'project_chroma_manager'
        pcma_param_name_in_agent_constructor = None
        if 'pcma_agent' in constructor_params:
            pcma_param_name_in_agent_constructor = 'pcma_agent'
        elif 'project_chroma_manager' in constructor_params: # Fallback for alternative naming
            pcma_param_name_in_agent_constructor = 'project_chroma_manager'

        
        # Crucially, pass the prepared system_context
        agent_init_params['system_context'] = agent_system_context
        
        # Add agent_id if the constructor expects it (some ProtocolAwareAgent might)
        if 'agent_id' in constructor_params:
            agent_init_params['agent_id'] = agent_id # Pass the resolved agent_id

        try:
            logger.debug(f"Final init params for {agent_class.__name__}: {list(agent_init_params.keys())}") # Log keys to avoid large values
            if 'system_context' in agent_init_params:
                 logger.debug(f"  system_context for {agent_class.__name__} will contain: {list(agent_init_params['system_context'].keys())}")

            # Filter params to only those accepted by __init__ to avoid unexpected keyword arg errors
            valid_params = {k: v for k, v in agent_init_params.items() if k in constructor_params or \
                            any(p.kind == inspect.Parameter.VAR_KEYWORD for p in constructor_params.values())} # MODIFIED: Correct **kwargs check

            # If 'self' is a param, remove it (it's for the method itself)
            if 'self' in valid_params: del valid_params['self']

            # ADDED: Diagnostic logging for system_context in valid_params
            sc_in_vp = valid_params.get('system_context')
            if sc_in_vp is not None:
                logger.info(f"DIAGNOSTIC ({agent_class.__name__}): system_context IS in valid_params. Keys: {list(sc_in_vp.keys())}")
            else:
                logger.warning(f"DIAGNOSTIC ({agent_class.__name__}): system_context IS NOT in valid_params or is None.")
            logger.info(f"DIAGNOSTIC ({agent_class.__name__}): All keys in valid_params before instantiation: {list(valid_params.keys())}")

            agent_instance = agent_class(**valid_params)
            logger.info(f"Successfully instantiated {agent_class.__name__} for agent_id '{agent_id}'")
            
            # Store the instance's invoke_async method in cache, associated with agent_id
            # This ensures that subsequent calls to get() for this agent_id will use this instance
            # if the shared_context hasn't changed in a way that requires re-instantiation.
            # For now, let's NOT cache instances here to always get fresh context from orchestrator.
            # Caching strategy needs refinement.
            # self._cache[agent_id] = getattr(agent_instance, "invoke_async")
            
            return agent_instance
        except Exception as e:
            logger.error(f"Error instantiating agent class {agent_class.__name__} for agent_id '{agent_id}': {e}")
            logger.error(traceback.format_exc())
            return None

    def get_raw_agent_instance(self, agent_id: str, shared_context: Optional[Dict[str, Any]] = None) -> Optional[ProtocolAwareAgent]:
        """
        Retrieves or creates a raw instance of a ProtocolAwareAgent.
        MODIFIED: Accepts shared_context and passes it for agent instantiation.
        """
        logger.debug(f"Attempting to get raw instance for agent_id: {agent_id}")

        # Stash/update shared_context for this call sequence
        if shared_context:
            self._orchestrator_shared_context = shared_context

        # Check fallback (common for system agents that are classes)
        if agent_id in self._fallback:
            potential_item = self._fallback[agent_id]
            if inspect.isclass(potential_item) and issubclass(potential_item, ProtocolAwareAgent):
                return self._instantiate_agent_class(potential_item, agent_id)
            elif isinstance(potential_item, ProtocolAwareAgent): # Already an instance
                # TODO: Update system_context if necessary? For now, return as-is.
                return potential_item
            else:
                logger.warning(f"Fallback item for {agent_id} is not a ProtocolAwareAgent class or instance: {type(potential_item)}")
        
        # If not in fallback, try AgentRegistry (Chroma)
        # This part of the logic would involve querying self._registry
        # agent_card = self._registry.get_agent_card(agent_id)
        # if agent_card and agent_card.fully_qualified_class_name:
        #     try:
        #         module_path, class_name = agent_card.fully_qualified_class_name.rsplit('.', 1)
        #         module = importlib.import_module(module_path)
        #         agent_class = getattr(module, class_name)
        #             return self._instantiate_agent_class(agent_class, agent_id, agent_card=agent_card)
        #     except Exception as e:
        #         logger.error(f"Error dynamically importing or instantiating agent {agent_id} from card for raw instance: {e}")
        #         # Fall through or raise, depending on desired strictness

        logger.warning(f"Could not get raw agent instance for agent_id: {agent_id}. Not found in fallback or registry (registry lookup not fully implemented here).")
        return None
            
    # --- METHOD TO BE ADDED ---
    def set_orchestrator_shared_context(self, shared_context: Dict[str, Any]):
        """Allows the orchestrator to set its shared context on the provider instance."""
        self._orchestrator_shared_context = shared_context
        logger.debug(f"RegistryAgentProvider shared_context has been set/updated by orchestrator. Keys: {list(shared_context.keys())}")


    def get_agent_card(self, agent_id: str) -> Optional[AgentCard]:
        # This method is not provided in the original code or the new implementation
        # It's unclear what this method is supposed to do, so it's left unchanged
        # If you need to implement this method, you'll need to add the appropriate logic here
        # This is a placeholder and should be replaced with the actual implementation
        # In a real scenario, this would be:
        # return self._registry.get_agent_card(agent_id)
        logger.warning(f"get_agent_card for '{agent_id}' called, but not fully implemented in RegistryAgentProvider beyond registry passthrough.")
        if hasattr(self, '_registry') and self._registry:
            try:
                return self._registry.get_agent_card(agent_id=agent_id)
            except Exception as e:
                logger.error(f"Error calling self._registry.get_agent_card for {agent_id}: {e}")
                return None
        return None


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 

# REMOVED: All duplicate imports and model_rebuild calls
# Registry-first architecture handles agent registration automatically via decorators
# No need for manual model rebuilding or imports in the resolver 