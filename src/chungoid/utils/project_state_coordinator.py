"""
Universal Project State Coordination System

Ensures all project discovery and caching systems stay synchronized 
across file operations, regardless of project type or technology stack.

This solves the "existing files[]" empty / "project type unknown" problem
by providing centralized cache invalidation and state coordination.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Protocol
from datetime import datetime
import asyncio
import weakref
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class FilesystemEventType(Enum):
    """Types of filesystem events that affect project state."""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    FILE_DELETED = "file_deleted"
    DIRECTORY_CREATED = "directory_created"
    DIRECTORY_DELETED = "directory_deleted"
    BATCH_OPERATION = "batch_operation"


@dataclass
class FilesystemEvent:
    """Represents a filesystem change event."""
    event_type: FilesystemEventType
    path: Path
    project_path: Path
    timestamp: datetime = field(default_factory=lambda: datetime.utcnow())
    metadata: Dict[str, Any] = field(default_factory=dict)


class CacheInvalidationHandler(Protocol):
    """Protocol for cache invalidation handlers."""
    
    async def invalidate_cache(self, event: FilesystemEvent) -> None:
        """Invalidate cache based on filesystem event."""
        ...
    
    def get_cache_identifier(self) -> str:
        """Get unique identifier for this cache system."""
        ...


class ProjectStateCoordinator:
    """
    Central coordinator for project state synchronization.
    
    Ensures all discovery and caching systems stay consistent
    when filesystem changes occur.
    """
    
    def __init__(self):
        self._handlers: Dict[str, CacheInvalidationHandler] = {}
        self._project_locks: Dict[Path, asyncio.Lock] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._processing_task: Optional[asyncio.Task] = None
        self._active = False
        self._lock = threading.Lock()
    
    def register_cache_handler(self, handler: CacheInvalidationHandler) -> None:
        """Register a cache invalidation handler."""
        cache_id = handler.get_cache_identifier()
        with self._lock:
            self._handlers[cache_id] = handler
        logger.info(f"Registered cache handler: {cache_id}")
    
    def unregister_cache_handler(self, cache_id: str) -> None:
        """Unregister a cache invalidation handler."""
        with self._lock:
            self._handlers.pop(cache_id, None)
        logger.info(f"Unregistered cache handler: {cache_id}")
    
    async def start(self) -> None:
        """Start the event processing system."""
        if self._active:
            return
        
        self._active = True
        self._processing_task = asyncio.create_task(self._process_events())
        logger.info("ProjectStateCoordinator started")
    
    async def stop(self) -> None:
        """Stop the event processing system."""
        if not self._active:
            return
        
        self._active = False
        if self._processing_task:
            self._processing_task.cancel()
            try:
                await self._processing_task
            except asyncio.CancelledError:
                pass
        logger.info("ProjectStateCoordinator stopped")
    
    async def emit_filesystem_event(self, event: FilesystemEvent) -> None:
        """Emit a filesystem event for processing."""
        if not self._active:
            await self.start()
        
        await self._event_queue.put(event)
        logger.debug(f"Emitted event: {event.event_type} for {event.path}")
    
    async def _process_events(self) -> None:
        """Process filesystem events and trigger cache invalidation."""
        while self._active:
            try:
                # Wait for events with timeout to allow clean shutdown
                event = await asyncio.wait_for(
                    self._event_queue.get(), 
                    timeout=1.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing filesystem event: {e}", exc_info=True)
    
    async def _handle_event(self, event: FilesystemEvent) -> None:
        """Handle a single filesystem event."""
        project_path = event.project_path
        
        # Get or create project lock
        if project_path not in self._project_locks:
            self._project_locks[project_path] = asyncio.Lock()
        
        project_lock = self._project_locks[project_path]
        
        async with project_lock:
            # Notify all registered cache handlers
            handlers = list(self._handlers.values())
            
            if handlers:
                logger.info(
                    f"Invalidating {len(handlers)} cache systems for "
                    f"{event.event_type} at {event.path}"
                )
                
                # Run invalidation handlers concurrently
                tasks = [
                    handler.invalidate_cache(event) 
                    for handler in handlers
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any failures
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        handler_id = handlers[i].get_cache_identifier()
                        logger.error(
                            f"Cache invalidation failed for {handler_id}: {result}",
                            exc_info=True
                        )
    
    @asynccontextmanager
    async def atomic_filesystem_operation(self, project_path: Path):
        """
        Context manager for atomic filesystem operations.
        
        Ensures cache invalidation happens after successful operations.
        """
        if project_path not in self._project_locks:
            self._project_locks[project_path] = asyncio.Lock()
        
        project_lock = self._project_locks[project_path]
        
        async with project_lock:
            # Track operations during the context
            operations = []
            
            # Provide operation tracker to context
            class OperationTracker:
                def record_operation(self, event_type: FilesystemEventType, path: Path, metadata: Dict[str, Any] = None):
                    operations.append(FilesystemEvent(
                        event_type=event_type,
                        path=path,
                        project_path=project_path,
                        metadata=metadata or {}
                    ))
            
            tracker = OperationTracker()
            
            try:
                yield tracker
                
                # If we get here, operations succeeded
                # Emit all recorded events
                for event in operations:
                    await self.emit_filesystem_event(event)
                
            except Exception as e:
                logger.warning(
                    f"Filesystem operation failed, skipping cache invalidation: {e}"
                )
                raise


# Global coordinator instance
_coordinator: Optional[ProjectStateCoordinator] = None


def get_project_state_coordinator() -> ProjectStateCoordinator:
    """Get the global project state coordinator instance."""
    global _coordinator
    if _coordinator is None:
        _coordinator = ProjectStateCoordinator()
    return _coordinator


async def emit_file_created(file_path: Path, project_path: Path, metadata: Dict[str, Any] = None) -> None:
    """Convenience function to emit file created event."""
    coordinator = get_project_state_coordinator()
    event = FilesystemEvent(
        event_type=FilesystemEventType.FILE_CREATED,
        path=file_path,
        project_path=project_path,
        metadata=metadata or {}
    )
    await coordinator.emit_filesystem_event(event)


async def emit_directory_created(dir_path: Path, project_path: Path, metadata: Dict[str, Any] = None) -> None:
    """Convenience function to emit directory created event."""
    coordinator = get_project_state_coordinator()
    event = FilesystemEvent(
        event_type=FilesystemEventType.DIRECTORY_CREATED,
        path=dir_path,
        project_path=project_path,
        metadata=metadata or {}
    )
    await coordinator.emit_filesystem_event(event)


async def emit_batch_operation(project_path: Path, operations: List[Dict[str, Any]]) -> None:
    """Convenience function to emit batch operation event."""
    coordinator = get_project_state_coordinator()
    event = FilesystemEvent(
        event_type=FilesystemEventType.BATCH_OPERATION,
        path=project_path,
        project_path=project_path,
        metadata={"operations": operations}
    )
    await coordinator.emit_filesystem_event(event) 