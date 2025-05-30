"""
Cache Invalidation Handlers

Implements cache invalidation handlers for all major discovery and caching
systems in the chungoid framework. These handlers integrate with the
ProjectStateCoordinator to ensure consistent state across all systems.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import asyncio

from .project_state_coordinator import CacheInvalidationHandler, FilesystemEvent, FilesystemEventType

logger = logging.getLogger(__name__)


class ProjectTypeDetectionCacheHandler(CacheInvalidationHandler):
    """Cache invalidation handler for ProjectTypeDetectionService."""
    
    def __init__(self):
        self._cached_results: Dict[Path, Any] = {}
        self._cache_timestamps: Dict[Path, float] = {}
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate project type detection cache."""
        project_path = event.project_path
        
        # Clear cached detection results for this project
        if project_path in self._cached_results:
            del self._cached_results[project_path]
            logger.debug(f"Cleared project type detection cache for {project_path}")
        
        if project_path in self._cache_timestamps:
            del self._cache_timestamps[project_path]
        
        # If this was a significant file change, force re-detection
        if self._is_significant_for_project_type(event):
            logger.info(f"Significant file change detected: {event.path} - project type will be re-detected")
    
    def get_cache_identifier(self) -> str:
        return "project_type_detection"
    
    def _is_significant_for_project_type(self, event: FilesystemEvent) -> bool:
        """Check if the filesystem event is significant for project type detection."""
        if event.event_type not in [FilesystemEventType.FILE_CREATED, FilesystemEventType.FILE_MODIFIED]:
            return False
        
        # Config files and package files are significant
        significant_patterns = [
            "package.json", "requirements.txt", "pyproject.toml", "Cargo.toml",
            "go.mod", "pom.xml", "build.gradle", "composer.json", 
            "Dockerfile", "tsconfig.json", "webpack.config", "vite.config",
            ".env", "Makefile", "makefile"
        ]
        
        filename = event.path.name.lower()
        return any(pattern in filename for pattern in significant_patterns)


class FilesystemScanCacheHandler(CacheInvalidationHandler):
    """Cache invalidation handler for filesystem scan operations."""
    
    def __init__(self):
        self._directory_scan_cache: Dict[Path, Any] = {}
        self._file_list_cache: Dict[Path, Any] = {}
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate filesystem scan caches."""
        project_path = event.project_path
        affected_dir = event.path.parent if event.path.is_file() else event.path
        
        # Clear directory scan cache
        cache_keys_to_remove = []
        for cached_path in self._directory_scan_cache.keys():
            if self._is_path_affected(cached_path, affected_dir, project_path):
                cache_keys_to_remove.append(cached_path)
        
        for key in cache_keys_to_remove:
            del self._directory_scan_cache[key]
            logger.debug(f"Cleared directory scan cache for {key}")
        
        # Clear file list cache
        cache_keys_to_remove = []
        for cached_path in self._file_list_cache.keys():
            if self._is_path_affected(cached_path, affected_dir, project_path):
                cache_keys_to_remove.append(cached_path)
        
        for key in cache_keys_to_remove:
            del self._file_list_cache[key]
            logger.debug(f"Cleared file list cache for {key}")
    
    def get_cache_identifier(self) -> str:
        return "filesystem_scan"
    
    def _is_path_affected(self, cached_path: Path, affected_dir: Path, project_path: Path) -> bool:
        """Check if a cached path is affected by the change."""
        try:
            # If cached path is within the affected directory or project, invalidate
            return (
                cached_path == affected_dir or
                cached_path == project_path or
                affected_dir in cached_path.parents or
                cached_path in affected_dir.parents
            )
        except (ValueError, OSError):
            return True  # Err on side of caution


class UniversalDiscoveryCacheHandler(CacheInvalidationHandler):
    """Cache invalidation handler for UnifiedAgent universal discovery."""
    
    def __init__(self):
        self._discovery_cache: Dict[str, Any] = {}
        self._pattern_cache: Dict[str, Any] = {}
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate universal discovery caches."""
        project_path = event.project_path
        
        # Clear all discovery cache entries for this project
        cache_keys_to_remove = []
        for cache_key in self._discovery_cache.keys():
            if str(project_path) in cache_key:
                cache_keys_to_remove.append(cache_key)
        
        for key in cache_keys_to_remove:
            del self._discovery_cache[key]
            logger.debug(f"Cleared universal discovery cache entry: {key}")
        
        # Clear pattern matching cache
        pattern_keys_to_remove = []
        for cache_key in self._pattern_cache.keys():
            if str(project_path) in cache_key:
                pattern_keys_to_remove.append(cache_key)
        
        for key in pattern_keys_to_remove:
            del self._pattern_cache[key]
            logger.debug(f"Cleared pattern cache entry: {key}")
    
    def get_cache_identifier(self) -> str:
        return "universal_discovery"


class StateManagerCacheHandler(CacheInvalidationHandler):
    """Cache invalidation handler for StateManager persistence layer."""
    
    def __init__(self):
        self._state_cache: Dict[Path, Any] = {}
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate state manager caches."""
        project_path = event.project_path
        
        # Clear cached state for this project
        if project_path in self._state_cache:
            del self._state_cache[project_path]
            logger.debug(f"Cleared state manager cache for {project_path}")
        
        # If .chungoid directory was affected, definitely clear
        if ".chungoid" in str(event.path):
            logger.info(f"State directory affected: {event.path} - forcing state refresh")
    
    def get_cache_identifier(self) -> str:
        return "state_manager"


class AgentDiscoveryCacheHandler(CacheInvalidationHandler):
    """Cache invalidation handler for agent-level discovery caches."""
    
    def __init__(self):
        self._agent_caches: Dict[str, Dict[str, Any]] = {}
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate agent discovery caches."""
        project_path = event.project_path
        
        # Clear all agent caches that might be affected
        for agent_id, cache in self._agent_caches.items():
            cache_keys_to_remove = []
            for cache_key in cache.keys():
                if str(project_path) in cache_key:
                    cache_keys_to_remove.append(cache_key)
            
            for key in cache_keys_to_remove:
                del cache[key]
                logger.debug(f"Cleared {agent_id} discovery cache entry: {key}")
    
    def get_cache_identifier(self) -> str:
        return "agent_discovery"
    
    def register_agent_cache(self, agent_id: str, cache: Dict[str, Any]) -> None:
        """Register an agent's cache for invalidation."""
        self._agent_caches[agent_id] = cache


# Factory function to create and register all standard handlers
def register_standard_cache_handlers():
    """Register all standard cache invalidation handlers with the coordinator."""
    from .project_state_coordinator import get_project_state_coordinator
    
    coordinator = get_project_state_coordinator()
    
    handlers = [
        ProjectTypeDetectionCacheHandler(),
        FilesystemScanCacheHandler(),
        UniversalDiscoveryCacheHandler(),
        StateManagerCacheHandler(),
        AgentDiscoveryCacheHandler(),
    ]
    
    for handler in handlers:
        coordinator.register_cache_handler(handler)
    
    logger.info(f"Registered {len(handlers)} standard cache invalidation handlers")
    return handlers 