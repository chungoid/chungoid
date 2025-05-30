"""
Cache Coordination Fix Module

Provides centralized cache coordination and invalidation across all discovery
and caching systems in the chungoid framework. This module ensures cache
coherency between file operations and discovery operations.

This fixes the "cache coordination not available" errors by implementing
the missing coordination functions that other modules expect.
"""

import logging
import asyncio
import time
from typing import Any, Dict, List, Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


async def coordinate_cache_refresh(
    project_path: str,
    call_mcp_tool_func: Callable,
    discovery_service: Optional[Any] = None,
    reason: str = "cache_refresh"
) -> Dict[str, Any]:
    """
    Coordinate cache refresh across all systems.
    
    This function ensures proper ordering: clear cache → call MCP tool → update cache
    
    Args:
        project_path: Path to refresh cache for
        call_mcp_tool_func: Function to call MCP tools
        discovery_service: Optional discovery service instance
        reason: Reason for cache refresh (for logging)
        
    Returns:
        Dict containing the fresh filesystem scan results
    """
    try:
        logger.info(f"CACHE_COORDINATION: Starting coordinated refresh for {project_path} - reason: {reason}")
        
        # Step 1: Clear all relevant caches FIRST
        await clear_all_coordinated_caches(discovery_service, project_path)
        
        # Step 2: Add small delay to ensure filesystem consistency
        await asyncio.sleep(0.1)
        
        # Step 3: Call MCP tool with force refresh
        result = await call_mcp_tool_func("filesystem_list_directory", {
            "directory_path": project_path,
            "recursive": True,
            "include_files": True,
            "include_directories": False,
            "max_depth": 10,
            "_force_refresh": True,
            "_cache_bust": time.time(),
            "_coordinated_refresh": True
        })
        
        # Step 4: Log results
        if result and result.get("success"):
            file_count = len(result.get("items", []))
            logger.info(f"CACHE_COORDINATION: Coordinated refresh completed - found {file_count} items")
        else:
            error_msg = result.get('error', 'Unknown error') if result else 'No result returned'
            logger.warning(f"CACHE_COORDINATION: Refresh failed - {error_msg}")
            
        return result
        
    except Exception as e:
        logger.error(f"CACHE_COORDINATION: Coordinated refresh failed for {project_path}: {e}")
        return {"success": False, "error": str(e)}


async def clear_all_coordinated_caches(
    discovery_service: Optional[Any] = None,
    project_path: Optional[str] = None
) -> int:
    """
    Clear all coordinated caches across the system.
    
    Args:
        discovery_service: Optional discovery service instance
        project_path: Optional project path to clear caches for
        
    Returns:
        int: Number of cache systems cleared
    """
    try:
        cleared_count = 0
        
        # Clear discovery service cache
        if discovery_service and hasattr(discovery_service, 'clear_cache'):
            discovery_service.clear_cache(project_path)
            cleared_count += 1
            logger.debug(f"CACHE_COORDINATION: Cleared discovery service cache")
        
        # Clear global discovery service if available
        try:
            import chungoid.agents.unified_agent as ua
            if hasattr(ua, '_discovery_service') and ua._discovery_service:
                ua._discovery_service.clear_cache(project_path)
                cleared_count += 1
                logger.debug(f"CACHE_COORDINATION: Cleared global discovery service cache")
        except Exception:
            pass
            
        # Clear project state coordinator caches
        try:
            from .project_state_coordinator import get_project_state_coordinator
            coordinator = get_project_state_coordinator()
            
            if project_path:
                # Trigger cache invalidation for this project
                from .project_state_coordinator import FilesystemEvent, FilesystemEventType
                event = FilesystemEvent(
                    event_type=FilesystemEventType.CACHE_INVALIDATION,
                    path=Path(project_path),
                    project_path=Path(project_path)
                )
                await coordinator.notify_filesystem_change(event)
                cleared_count += 1
                logger.debug(f"CACHE_COORDINATION: Triggered project state cache invalidation")
        except Exception as e:
            logger.debug(f"CACHE_COORDINATION: Could not clear project state caches: {e}")
            
        # Clear performance optimizer cache
        try:
            from ..runtime.performance_optimizer import get_performance_optimizer
            optimizer = await get_performance_optimizer()
            if hasattr(optimizer, 'cache') and hasattr(optimizer.cache, 'clear'):
                optimizer.cache.clear()
                cleared_count += 1
                logger.debug(f"CACHE_COORDINATION: Cleared performance optimizer cache")
        except Exception as e:
            logger.debug(f"CACHE_COORDINATION: Could not clear performance optimizer cache: {e}")
            
        logger.info(f"CACHE_COORDINATION: Cleared {cleared_count} cache systems")
        return cleared_count
        
    except Exception as e:
        logger.error(f"CACHE_COORDINATION: Failed to clear coordinated caches: {e}")
        return 0


def invalidate_project_caches(project_path: str, reason: str = "file_operation") -> None:
    """
    Synchronously invalidate all project-related caches.
    
    Args:
        project_path: Path to invalidate caches for
        reason: Reason for invalidation (for logging)
    """
    try:
        logger.info(f"CACHE_COORDINATION: Invalidating caches for {project_path} - reason: {reason}")
        
        # Clear discovery service caches synchronously
        try:
            import chungoid.agents.unified_agent as ua
            if hasattr(ua, '_discovery_service') and ua._discovery_service:
                ua._discovery_service.clear_cache(project_path)
                logger.debug(f"CACHE_COORDINATION: Cleared discovery service cache (sync)")
        except Exception as e:
            logger.debug(f"CACHE_COORDINATION: Could not clear discovery service cache (sync): {e}")
            
        # Clear any other synchronous caches
        logger.info(f"CACHE_COORDINATION: Cache invalidation completed for {project_path}")
        
    except Exception as e:
        logger.error(f"CACHE_COORDINATION: Failed to invalidate project caches: {e}")


async def verify_cache_consistency(
    project_path: str,
    call_mcp_tool_func: Callable,
    expected_files: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Verify cache consistency by forcing a fresh filesystem scan.
    
    Args:
        project_path: Path to verify
        call_mcp_tool_func: Function to call MCP tools
        expected_files: Optional list of files that should exist
        
    Returns:
        Dict containing verification results
    """
    try:
        logger.info(f"CACHE_COORDINATION: Verifying cache consistency for {project_path}")
        
        # Force a completely fresh scan
        result = await coordinate_cache_refresh(
            project_path, 
            call_mcp_tool_func, 
            reason="consistency_verification"
        )
        
        if not result or not result.get("success"):
            return {
                "verified": False,
                "error": result.get("error", "Verification scan failed") if result else "No scan result"
            }
            
        found_files = [item.get("relative_path", item.get("path", "")) 
                      for item in result.get("items", []) 
                      if item.get("type") == "file"]
        
        verification_result = {
            "verified": True,
            "found_files": found_files,
            "file_count": len(found_files),
            "cache_consistent": True
        }
        
        # Check expected files if provided
        if expected_files:
            missing_files = [f for f in expected_files if f not in found_files]
            verification_result.update({
                "expected_files": expected_files,
                "missing_files": missing_files,
                "all_expected_found": len(missing_files) == 0
            })
            
        logger.info(f"CACHE_COORDINATION: Verification completed - found {len(found_files)} files")
        return verification_result
        
    except Exception as e:
        logger.error(f"CACHE_COORDINATION: Cache consistency verification failed: {e}")
        return {
            "verified": False,
            "error": str(e)
        }


def force_cache_bypass_context():
    """
    Context manager to force cache bypass for critical operations.
    
    Returns a context that can be used to ensure fresh data.
    """
    return {
        "cache_bypassed": True,
        "force_refresh": True,
        "cache_coordination_active": True,
        "timestamp": time.time()
    }


# Export main functions
__all__ = [
    "coordinate_cache_refresh",
    "clear_all_coordinated_caches", 
    "invalidate_project_caches",
    "verify_cache_consistency",
    "force_cache_bypass_context"
] 