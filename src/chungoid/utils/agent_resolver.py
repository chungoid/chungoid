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

from typing import Callable, Dict, Protocol, runtime_checkable, Optional, List, Any, Tuple, Union, Type, cast
from pathlib import Path
from semver import VersionInfo
from .agent_registry import AgentCard, AgentRegistry
from chungoid.schemas.common import AgentCallable
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.runtime.agents.agent_base import BaseAgent
import logging
import asyncio
import functools
import importlib
import sys
import inspect
import traceback

# ADDED: Moved import to module level
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent # MOVED TO MODULE LEVEL
from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1 # ADDED IMPORT
from chungoid.runtime.agents.system_test_runner_agent import SystemTestRunnerAgent_v1 # ADDED FOR SYSTEM TEST RUNNER
# ADDED: Import ArchitectAgent_v1
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1

# ADDED: Import ProjectChromaManagerAgent_v1
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

# REMOVED: Incorrect import for SmartCodeGeneratorAgent_v1
# from chungoid.runtime.agents.smart_code_generator_agent import SmartCodeGeneratorAgent_v1

# Define logger at the module level
logger = logging.getLogger(__name__)
# --- TEMPORARY DEBUGGING CODE HAS BEEN REMOVED ---

# For type hinting AgentCallable
# AgentCallable = Callable[..., Any]  # Simple version
# More precise: Callable[Concatenate[InputSchema, P], OutputSchema] or similar with Pydantic models
# For now, stick to Any to avoid over-complicating until fully typed Agent execution is ready.
# AgentCallable = Callable[..., Any] # REMOVED
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
        registry: "AgentRegistry",
        fallback: Optional[Dict[str, AgentCallable]] = None,
        # MODIFIED: Add dependencies for agent instantiation
        llm_provider: Optional[LLMProvider] = None,
        prompt_manager: Optional[PromptManager] = None,
        project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = None
    ) -> None:
        from .agent_registry import AgentRegistry as ConcreteAgentRegistry
        # MODIFIED: Import dependencies
        # from chungoid.utils.llm_provider import LLMProvider # MOVED UP
        # from chungoid.utils.prompt_manager import PromptManager # MOVED UP
        from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
        # from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent # REMOVED FROM HERE
        # ADDED: Import for SystemFileSystemAgent_v1 to check its type
        from chungoid.runtime.agents.system_file_system_agent import SystemFileSystemAgent_v1

        if not isinstance(registry, ConcreteAgentRegistry):
            logger.error(f"RegistryAgentProvider __init__: PASSED REGISTRY TYPE IS {type(registry)}. EXPECTED ConcreteAgentRegistry.") # DIAGNOSTIC
            raise TypeError("registry must be an AgentRegistry instance")
        self._registry = registry
        logger.info(f"RegistryAgentProvider __init__: self._registry SET to {type(self._registry)}, id: {id(self._registry)}") # DIAGNOSTIC

        self._fallback: Dict[str, AgentCallable] = fallback or {}
        self._cache: Dict[str, AgentCallable] = {}
        
        # MODIFIED: Store dependencies
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._project_chroma_manager = project_chroma_manager


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
    def get(self, identifier: str, category: Optional[str] = None) -> Callable[..., Coroutine[Any, Any, Any]]:
        """
        Retrieves an agent's callable (invoke_async method) by its ID or category.

        Order of resolution:
        1. Direct ID match in fallback map.
        2. Direct ID match in AgentRegistry (ChromaDB) - (Not fully shown in this snippet, assumes leads to error or different path for now if not in fallback).
        3. Category match in AgentRegistry (ChromaDB) - (Not fully shown in this snippet, assumes leads to error or different path for now if not in fallback).
        """
        logger.debug(f"RegistryAgentProvider GET: identifier='{identifier}', category='{category}'") # Changed log level
        logger.info(f"RegistryAgentProvider GET: hasattr(self, '_registry'): {hasattr(self, '_registry')}") # DIAGNOSTIC
        if hasattr(self, '_registry'):
            logger.info(f"RegistryAgentProvider GET: self._registry type: {type(self._registry)}, id: {id(self._registry)}") # DIAGNOSTIC

        agent_instance: Optional[BaseAgent] = None
        potential_item: Optional[Union[Type[BaseAgent], BaseAgent]] = None

        try:
            # Check if the agent is in the fallback map first
            if identifier in self._fallback:
                potential_item = self._fallback[identifier]
                logger.info(f"RegistryAgentProvider: Identifier '{identifier}' FOUND in fallback map.")
                logger.info(f"RegistryAgentProvider: Fallback item type: {type(potential_item)}, name: {getattr(potential_item, '__name__', 'N/A')}")
                logger.info(f"RegistryAgentProvider: BaseAgent module in resolver: {BaseAgent.__module__}, id: {id(BaseAgent)}")
                if hasattr(potential_item, '__bases__') and potential_item.__bases__:
                    logger.info(f"RegistryAgentProvider: Fallback item's BaseAgent module: {potential_item.__bases__[0].__module__}, id: {id(potential_item.__bases__[0])}")

                # --- Begin specific agent instantiation logic ---
                if isinstance(potential_item, type) and issubclass(potential_item, BaseAgent):
                    logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' is a BaseAgent subclass. Attempting instantiation.")
                    # Special handling for agents requiring dependencies like LLMProvider, PromptManager, or ProjectChromaManager
                    
                    # MODIFIED: Use self._llm_provider, self._prompt_manager, self._project_chroma_manager
                    if potential_item.__name__ == "CoreCodeGeneratorAgent_v1" or potential_item.__name__ == "SmartCodeGeneratorAgent_v1":
                        logger.info(f"RegistryAgentProvider: Special instantiation for {potential_item.__name__} with llm_provider, prompt_manager, and pcma_agent (was project_chroma_manager).")
                        agent_instance = potential_item(
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager,
                            pcma_agent=self._project_chroma_manager
                        )
                    elif potential_item.__name__ == "MasterPlannerAgent":
                        logger.info(f"RegistryAgentProvider: Special instantiation for MasterPlannerAgent with llm_provider and prompt_manager.")
                        agent_instance = potential_item(
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager,
                            project_chroma_manager=self._project_chroma_manager # Also needs PCMA
                        )
                    elif potential_item.__name__ == "SystemFileSystemAgent_v1":
                        logger.info(f"RegistryAgentProvider: Special instantiation for SystemFileSystemAgent_v1 with pcma_agent, llm_provider, and prompt_manager.")
                        agent_instance = potential_item(
                            pcma_agent=self._project_chroma_manager,
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager
                        )
                    elif potential_item.__name__ == "SystemTestRunnerAgent_v1":
                        logger.info(f"RegistryAgentProvider: Special instantiation for SystemTestRunnerAgent_v1 with llm_provider, prompt_manager, and project_chroma_manager.")
                        agent_instance = potential_item(
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager,
                            project_chroma_manager=self._project_chroma_manager
                        )
                    # ADDED: Ensure ArchitectAgent_v1 also gets its dependencies
                    elif potential_item.__name__ == "ArchitectAgent_v1":
                        logger.info(f"RegistryAgentProvider: Special instantiation for ArchitectAgent_v1 with llm_provider, prompt_manager, and project_chroma_manager.")
                        agent_instance = potential_item(
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager,
                            project_chroma_manager=self._project_chroma_manager
                        )
                    elif potential_item.__name__ == "SystemRequirementsGatheringAgent_v1":
                        logger.info(f"RegistryAgentProvider: Special instantiation for SystemRequirementsGatheringAgent_v1 with llm_provider, prompt_manager, and project_chroma_manager.")
                        agent_instance = potential_item(
                            llm_provider=self._llm_provider,
                            prompt_manager=self._prompt_manager,
                            project_chroma_manager=self._project_chroma_manager
                        )
                    else:
                        logger.info(f"RegistryAgentProvider: Standard instantiation for fallback agent {potential_item.__name__}.")
                        try:
                            agent_instance = potential_item() # Standard instantiation
                        except TypeError as e_type:
                            logger.error(f"TypeError during standard instantiation of {potential_item.__name__}: {e_type}. This might mean it needs specific dependencies not handled here.", exc_info=True)
                            raise NoAgentFoundError(f"Failed to instantiate agent '{identifier}' ({potential_item.__name__}) due to TypeError: {e_type}. Check agent constructor and resolver logic.") from e_type

                elif inspect.isfunction(potential_item) or inspect.ismethod(potential_item):
                    logger.info(f"RegistryAgentProvider: Fallback item '{identifier}' is a function or method. Wrapping in a callable.")
                    agent_instance = BaseAgent(identifier, potential_item)
                else:
                    raise NoAgentFoundError(f"Fallback item for '{identifier}' is not a valid BaseAgent class or instance.")
                # --- End specific agent instantiation logic ---

            # If not in fallback, try AgentRegistry by ID (Simplified for this example)
            # In a real scenario, this would involve querying ChromaDB / self._registry
            # and then potentially instantiating based on the AgentCard, including pcma_agent if needed.
            elif self._registry and not category:
                logger.warning(f"RegistryAgentProvider: ID-based registry lookup for '{identifier}' not fully implemented in this example. Assuming not found for now if not in fallback.")
                # Placeholder: agent_card = self._registry.get_agent_by_id(identifier)
                # if agent_card: ... instantiate based on card ...
                pass

            # If category is provided, try AgentRegistry by category (Simplified for this example)
            elif self._registry and category:
                logger.warning(f"RegistryAgentProvider: Category-based registry lookup for '{category}' not fully implemented in this example. Assuming not found for now if not in fallback.")
                # Placeholder: agent_cards = self._registry.get_agents_by_category(category)
                # if agent_cards: ... select and instantiate ...
                pass
            
            if agent_instance:
                logger.info(f"RegistryAgentProvider: Returning 'invoke_async' method for instantiated agent '{identifier or category}'.")
                # Cache the invoke_async method
                self._cache[identifier] = agent_instance.invoke_async
                return agent_instance.invoke_async
            else:
                logger.error(f"RegistryAgentProvider: Agent not found via fallback for identifier '{identifier}'. Registry logic simplified in this example.")
                raise NoAgentFoundError(f"Agent '{identifier or category}' not found in fallback map or registry (registry logic simplified in this example).")

        except NoAgentFoundError:
            logger.error(f"Agent '{identifier or category}' not found by RegistryAgentProvider.")
            raise
        except AmbiguousAgentCategoryError:
            logger.error(f"Agent category '{category}' is ambiguous.")
            raise
        except Exception as e:
            logger.error(f"RegistryAgentProvider: Unexpected error resolving agent '{identifier or category}': {e}")
            logger.debug(traceback.format_exc()) # Log full traceback for unexpected errors
            raise NoAgentFoundError(f"Unexpected error resolving agent '{identifier or category}': {e}")

        except NoAgentFoundForCategoryError:
            raise
        except AmbiguousAgentCategoryError:
            raise
        except Exception as e:
            logger.error(f"RegistryAgentProvider: Unexpected error resolving agent '{identifier or category}': {e}")
            logger.debug(traceback.format_exc())
            raise NoAgentFoundForCategoryError(f"Unexpected error resolving agent '{identifier or category}': {e}")

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


# Convenient alias used by FlowExecutor refactor
LegacyProvider = DictAgentProvider 

# After all class definitions that might have forward references to AgentProvider

# Resolve forward references for Pydantic models that use \'AgentProvider\'
# This is necessary because of the TYPE_CHECKING import of AgentProvider
# in agent files, which Pydantic needs to resolve before model instantiation.
SystemRequirementsGatheringAgent_v1.model_rebuild()
ArchitectAgent_v1.model_rebuild() # ArchitectAgent_v1 also uses Optional['AgentProvider']

# Add other agents here if they also have `agent_provider: Optional['AgentProvider']`
# and are imported in this file or are part of the typical resolution path.
# e.g.:
MasterPlannerAgent.model_rebuild() # SystemMasterPlannerAgent_v1
# CoreCodeGeneratorAgent_v1.model_rebuild() # REMOVED: Not a Pydantic model (this one was missed)
SystemTestRunnerAgent_v1.model_rebuild()

# It's also good practice to rebuild any other models that might have unresolved
# forward references due to TYPE_CHECKING blocks, if they are defined or imported
# globally in this module. 

# ADDED: Call model_rebuild for BaseAgent and other relevant Pydantic models
# This is crucial for resolving forward references used in type hints, especially
# when TYPE_CHECKING is used to break circular dependencies.
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.runtime.agents.core_code_generator_agent import CoreCodeGeneratorAgent_v1
from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent
from chungoid.runtime.agents.system_test_runner_agent import SystemTestRunnerAgent_v1
from chungoid.agents.autonomous_engine.architect_agent import ArchitectAgent_v1
from chungoid.runtime.agents.system_requirements_gathering_agent import SystemRequirementsGatheringAgent_v1

BaseAgent.model_rebuild()
# CoreCodeGeneratorAgent_v1.model_rebuild() # REMOVED: Not a Pydantic model
MasterPlannerAgent.model_rebuild()
SystemTestRunnerAgent_v1.model_rebuild()
ArchitectAgent_v1.model_rebuild()
SystemRequirementsGatheringAgent_v1.model_rebuild() 