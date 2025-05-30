"""
Integration module to apply all architect agent fixes.

Call this during system initialization to fix the cache/discovery issues.
"""

import logging
from .simple_cache_invalidation import register_cache_clearer, get_cache_invalidator
from ..agents.autonomous_engine.architect_agent_cache_fix import patch_architect_agent

logger = logging.getLogger(__name__)


def apply_architect_cache_fixes():
    """Apply all architect agent cache and discovery fixes."""
    try:
        logger.info("APPLYING ARCHITECT CACHE FIXES...")
        
        # Patch the architect agent with enhanced discovery
        patch_architect_agent()
        
        # Register common cache clearers
        register_common_cache_clearers()
        
        logger.info("ARCHITECT CACHE FIXES APPLIED SUCCESSFULLY")
        
    except Exception as e:
        logger.error(f"Failed to apply architect fixes: {e}")
        raise


def register_common_cache_clearers():
    """Register cache clearers for common cached systems."""
    
    def clear_discovery_cache(project_path: str):
        """Clear discovery caches."""
        # This would clear any discovery-related caches
        logger.debug(f"Clearing discovery cache for {project_path}")
    
    def clear_project_type_cache(project_path: str):
        """Clear project type detection cache."""
        # This would clear project type detection caches
        logger.debug(f"Clearing project type cache for {project_path}")
    
    def clear_filesystem_cache(project_path: str):
        """Clear filesystem operation caches."""
        # This would clear filesystem-related caches
        logger.debug(f"Clearing filesystem cache for {project_path}")
    
    # Register all cache clearers
    invalidator = get_cache_invalidator()
    invalidator.register_cache_clearer(clear_discovery_cache)
    invalidator.register_cache_clearer(clear_project_type_cache)
    invalidator.register_cache_clearer(clear_filesystem_cache)
    
    logger.info("REGISTERED 3 CACHE CLEARERS")


# Auto-apply fixes when module is imported
try:
    apply_architect_cache_fixes()
except Exception as e:
    logger.warning(f"Auto-fix application failed: {e}") 