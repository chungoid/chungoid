"""
Simple synchronous cache invalidation that works with existing architecture.

NO async/await, NO event loops, NO complexity - just direct cache clearing.
"""

import logging
from typing import Dict, Any, Set, Optional, Callable
from pathlib import Path
import threading
import weakref

logger = logging.getLogger(__name__)


class SimpleCacheInvalidator:
    """Simple, synchronous cache invalidation system."""
    
    def __init__(self):
        self._cache_clearers: Set[Callable[[str], None]] = set()
        self._lock = threading.Lock()
    
    def register_cache_clearer(self, clearer_fn: Callable[[str], None]):
        """Register a function that clears a specific cache."""
        with self._lock:
            self._cache_clearers.add(clearer_fn)
    
    def invalidate_all_caches(self, project_path: str, reason: str = "file_operation"):
        """Clear all registered caches."""
        logger.info(f"🧹 CACHE INVALIDATION: {reason} in {project_path}")
        
        cleared_count = 0
        with self._lock:
            for clearer in self._cache_clearers.copy():
                try:
                    clearer(project_path)
                    cleared_count += 1
                except Exception as e:
                    logger.warning(f"Cache clearer failed: {e}")
        
        logger.info(f"✅ CLEARED {cleared_count} caches for {project_path}")


# Global instance
_invalidator = SimpleCacheInvalidator()

def get_cache_invalidator() -> SimpleCacheInvalidator:
    return _invalidator

def invalidate_project_caches(project_path: str, reason: str = "file_operation"):
    """Convenience function to invalidate all caches."""
    _invalidator.invalidate_all_caches(project_path, reason)

def register_cache_clearer(clearer_fn: Callable[[str], None]):
    """Convenience function to register cache clearer."""
    _invalidator.register_cache_clearer(clearer_fn)

def auto_invalidate_on_file_operation(func):
    """Decorator to automatically invalidate caches after file operations."""
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        
        # Try to extract project_path from various argument patterns
        project_path = None
        
        # Check kwargs first
        for key in ['project_path', 'target_directory', 'directory_path', 'base_path']:
            if key in kwargs:
                project_path = str(kwargs[key])
                break
        
        # Check positional args
        if not project_path and args:
            for arg in args:
                if isinstance(arg, (str, Path)) and ('/' in str(arg) or '\\' in str(arg)):
                    project_path = str(arg)
                    break
        
        if project_path:
            invalidate_project_caches(project_path, f"auto_invalidation_after_{func.__name__}")
        
        return result
    return wrapper 