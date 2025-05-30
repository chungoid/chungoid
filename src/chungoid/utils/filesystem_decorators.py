"""
Filesystem Operation Decorators

Provides decorators that automatically emit filesystem events for cache
invalidation. This ensures that all file operations trigger appropriate
cache invalidation regardless of where they're called from.
"""

import logging
import asyncio
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import inspect

from .project_state_coordinator import (
    get_project_state_coordinator, 
    FilesystemEventType,
    emit_file_created,
    emit_directory_created,
    emit_batch_operation
)

logger = logging.getLogger(__name__)


def auto_invalidate_on_file_operation(
    project_path_param: str = "project_path",
    file_path_param: str = "file_path",
    operation_type: FilesystemEventType = FilesystemEventType.FILE_CREATED
):
    """
    Decorator that automatically emits filesystem events for cache invalidation.
    
    Args:
        project_path_param: Parameter name containing project path
        file_path_param: Parameter name containing file path
        operation_type: Type of filesystem operation
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                # Execute the original function
                result = await func(*args, **kwargs)
                
                # Extract paths from function parameters
                try:
                    project_path = _extract_path_param(
                        func, args, kwargs, project_path_param
                    )
                    file_path = _extract_path_param(
                        func, args, kwargs, file_path_param
                    )
                    
                    if project_path and file_path:
                        # Emit filesystem event for cache invalidation
                        coordinator = get_project_state_coordinator()
                        from .project_state_coordinator import FilesystemEvent
                        
                        event = FilesystemEvent(
                            event_type=operation_type,
                            path=Path(file_path),
                            project_path=Path(project_path),
                            metadata={"function": func.__name__, "result": str(result)[:200]}
                        )
                        
                        await coordinator.emit_filesystem_event(event)
                        logger.debug(f"Auto-invalidated caches for {operation_type} at {file_path}")
                
                except Exception as e:
                    logger.warning(f"Cache invalidation failed for {func.__name__}: {e}")
                
                return result
            
            return async_wrapper
        
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                # Execute the original function
                result = func(*args, **kwargs)
                
                # Extract paths and emit event asynchronously
                try:
                    project_path = _extract_path_param(
                        func, args, kwargs, project_path_param
                    )
                    file_path = _extract_path_param(
                        func, args, kwargs, file_path_param
                    )
                    
                    if project_path and file_path:
                        # Schedule async emission
                        asyncio.create_task(_emit_sync_operation_event(
                            operation_type, file_path, project_path, func.__name__, result
                        ))
                
                except Exception as e:
                    logger.warning(f"Cache invalidation failed for {func.__name__}: {e}")
                
                return result
            
            return sync_wrapper
    
    return decorator


async def _emit_sync_operation_event(
    operation_type: FilesystemEventType,
    file_path: str,
    project_path: str, 
    function_name: str,
    result: Any
):
    """Emit filesystem event from sync function."""
    try:
        coordinator = get_project_state_coordinator()
        from .project_state_coordinator import FilesystemEvent
        
        event = FilesystemEvent(
            event_type=operation_type,
            path=Path(file_path),
            project_path=Path(project_path),
            metadata={"function": function_name, "result": str(result)[:200]}
        )
        
        await coordinator.emit_filesystem_event(event)
        logger.debug(f"Auto-invalidated caches for {operation_type} at {file_path}")
        
    except Exception as e:
        logger.warning(f"Async cache invalidation failed: {e}")


def _extract_path_param(
    func: Callable, 
    args: tuple, 
    kwargs: dict, 
    param_name: str
) -> Optional[str]:
    """Extract a path parameter from function arguments."""
    # First check kwargs
    if param_name in kwargs:
        return str(kwargs[param_name])
    
    # Then check positional args
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    
    if param_name in param_names:
        param_index = param_names.index(param_name)
        if param_index < len(args):
            return str(args[param_index])
    
    # Also check common alternative parameter names
    alt_names = {
        "project_path": ["project_dir", "project_directory", "target_directory"],
        "file_path": ["filepath", "filename", "target_file", "file_name"]
    }
    
    if param_name in alt_names:
        for alt_name in alt_names[param_name]:
            if alt_name in kwargs:
                return str(kwargs[alt_name])
            
            if alt_name in param_names:
                alt_index = param_names.index(alt_name)
                if alt_index < len(args):
                    return str(args[alt_index])
    
    return None


def batch_file_operation(project_path_param: str = "project_path"):
    """
    Decorator for batch file operations that emits a single batch event.
    
    Args:
        project_path_param: Parameter name containing project path
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                result = await func(*args, **kwargs)
                
                try:
                    project_path = _extract_path_param(
                        func, args, kwargs, project_path_param
                    )
                    
                    if project_path:
                        # Extract file operations from result if possible
                        operations = _extract_operations_from_result(result)
                        
                        if operations:
                            await emit_batch_operation(Path(project_path), operations)
                            logger.debug(f"Emitted batch operation event for {len(operations)} operations")
                
                except Exception as e:
                    logger.warning(f"Batch cache invalidation failed for {func.__name__}: {e}")
                
                return result
            
            return async_wrapper
        
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                result = func(*args, **kwargs)
                
                try:
                    project_path = _extract_path_param(
                        func, args, kwargs, project_path_param
                    )
                    
                    if project_path:
                        operations = _extract_operations_from_result(result)
                        
                        if operations:
                            asyncio.create_task(
                                emit_batch_operation(Path(project_path), operations)
                            )
                
                except Exception as e:
                    logger.warning(f"Batch cache invalidation failed for {func.__name__}: {e}")
                
                return result
            
            return sync_wrapper
    
    return decorator


def _extract_operations_from_result(result: Any) -> List[Dict[str, Any]]:
    """Extract operation information from function result."""
    operations = []
    
    if isinstance(result, dict):
        # Look for common result patterns
        if "files_created" in result:
            files = result["files_created"]
            if isinstance(files, (list, tuple)):
                for file_info in files:
                    operations.append({
                        "type": "file_created",
                        "path": str(file_info) if isinstance(file_info, (str, Path)) else str(file_info.get("path", ""))
                    })
        
        if "directories_created" in result:
            dirs = result["directories_created"]
            if isinstance(dirs, (list, tuple)):
                for dir_info in dirs:
                    operations.append({
                        "type": "directory_created", 
                        "path": str(dir_info) if isinstance(dir_info, (str, Path)) else str(dir_info.get("path", ""))
                    })
        
        # Check for success indicators with file lists
        if result.get("success") and "files" in result:
            files = result["files"]
            if isinstance(files, (list, tuple)):
                for file_path in files:
                    operations.append({
                        "type": "file_operation",
                        "path": str(file_path)
                    })
    
    return operations


# Convenience decorators for common operations
file_created = lambda **kwargs: auto_invalidate_on_file_operation(
    operation_type=FilesystemEventType.FILE_CREATED, **kwargs
)

file_modified = lambda **kwargs: auto_invalidate_on_file_operation(
    operation_type=FilesystemEventType.FILE_MODIFIED, **kwargs
)

directory_created = lambda **kwargs: auto_invalidate_on_file_operation(
    operation_type=FilesystemEventType.DIRECTORY_CREATED, **kwargs
)


# Context manager for manual event emission
class FilesystemEventContext:
    """Context manager for manual filesystem event emission."""
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self.operations: List[Dict[str, Any]] = []
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.operations:
            # Only emit if no exception occurred
            await emit_batch_operation(self.project_path, self.operations)
    
    def record_file_created(self, file_path: Path, metadata: Dict[str, Any] = None):
        """Record a file creation operation."""
        self.operations.append({
            "type": "file_created",
            "path": str(file_path),
            "metadata": metadata or {}
        })
    
    def record_directory_created(self, dir_path: Path, metadata: Dict[str, Any] = None):
        """Record a directory creation operation."""
        self.operations.append({
            "type": "directory_created", 
            "path": str(dir_path),
            "metadata": metadata or {}
        }) 