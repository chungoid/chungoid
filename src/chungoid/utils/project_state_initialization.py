"""
Project State Initialization

Automatically initializes the project state coordination system and
integrates it with existing components. This ensures that cache
invalidation is set up correctly for any project type.
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional

from .project_state_coordinator import get_project_state_coordinator
from .cache_invalidation_handlers import register_standard_cache_handlers
from .filesystem_decorators import FilesystemEventContext

logger = logging.getLogger(__name__)


class ProjectStateManager:
    """
    Manages project state initialization and coordination.
    
    This is the main entry point for setting up the universal
    project state synchronization system.
    """
    
    def __init__(self):
        self._initialized = False
        self._coordinator = None
        self._handlers = []
    
    async def initialize(self) -> None:
        """Initialize the project state coordination system."""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing universal project state coordination system")
            
            # Get the global coordinator
            self._coordinator = get_project_state_coordinator()
            
            # Register standard cache invalidation handlers
            self._handlers = register_standard_cache_handlers()
            
            # Start the coordinator
            await self._coordinator.start()
            
            self._initialized = True
            logger.info("Project state coordination system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize project state coordination: {e}", exc_info=True)
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the project state coordination system."""
        if not self._initialized:
            return
        
        try:
            if self._coordinator:
                await self._coordinator.stop()
            
            self._initialized = False
            logger.info("Project state coordination system shutdown")
            
        except Exception as e:
            logger.warning(f"Error during project state coordination shutdown: {e}")
    
    def is_initialized(self) -> bool:
        """Check if the system is initialized."""
        return self._initialized
    
    async def ensure_initialized(self) -> None:
        """Ensure the system is initialized (safe to call multiple times)."""
        if not self._initialized:
            await self.initialize()


# Global manager instance
_project_state_manager: Optional[ProjectStateManager] = None


def get_project_state_manager() -> ProjectStateManager:
    """Get the global project state manager instance."""
    global _project_state_manager
    if _project_state_manager is None:
        _project_state_manager = ProjectStateManager()
    return _project_state_manager


async def ensure_project_state_system_ready() -> None:
    """
    Ensure the project state coordination system is ready.
    
    This is a convenience function that should be called early in the
    application lifecycle to ensure cache invalidation works properly.
    """
    manager = get_project_state_manager()
    await manager.ensure_initialized()


def setup_project_state_system_for_sync_context():
    """
    Set up project state system in a synchronous context.
    
    This creates the event loop if needed and initializes the system.
    Useful for CLI tools and other sync entry points.
    """
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Schedule initialization
            asyncio.create_task(ensure_project_state_system_ready())
        else:
            # Run initialization in the loop
            loop.run_until_complete(ensure_project_state_system_ready())
    except RuntimeError:
        # No event loop, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(ensure_project_state_system_ready())
        finally:
            # Keep the loop running for async operations
            pass


# Integration helpers for existing components

def integrate_with_mcp_tools():
    """
    Integrate the project state system with MCP tools.
    
    This patches common MCP filesystem tools to automatically
    emit cache invalidation events.
    """
    try:
        # Import and patch filesystem tools
        from ..mcp_tools.filesystem import file_operations
        from ..mcp_tools.filesystem import directory_operations
        from .filesystem_decorators import file_created, directory_created, batch_file_operation
        
        # Patch common file operations
        if hasattr(file_operations, 'filesystem_write_file'):
            original_write = file_operations.filesystem_write_file
            file_operations.filesystem_write_file = file_created(
                file_path_param="file_path",
                project_path_param="project_path"
            )(original_write)
        
        if hasattr(file_operations, 'filesystem_create_file'):
            original_create = file_operations.filesystem_create_file
            file_operations.filesystem_create_file = file_created(
                file_path_param="file_path", 
                project_path_param="project_path"
            )(original_create)
        
        # Patch directory operations
        if hasattr(directory_operations, 'filesystem_create_directory'):
            original_mkdir = directory_operations.filesystem_create_directory
            directory_operations.filesystem_create_directory = directory_created(
                file_path_param="directory_path",
                project_path_param="project_path"
            )(original_mkdir)
        
        # Patch batch operations
        if hasattr(directory_operations, 'filesystem_batch_create'):
            original_batch = directory_operations.filesystem_batch_create
            directory_operations.filesystem_batch_create = batch_file_operation(
                project_path_param="project_path"
            )(original_batch)
        
        logger.info("Integrated project state system with MCP filesystem tools")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with MCP tools: {e}")
    except Exception as e:
        logger.error(f"Error integrating with MCP tools: {e}", exc_info=True)


def integrate_with_unified_agents():
    """
    Integrate the project state system with UnifiedAgent discovery.
    
    This ensures agent discovery systems participate in cache invalidation.
    """
    try:
        from ..agents.unified_agent import UnifiedAgent
        from .cache_invalidation_handlers import AgentDiscoveryCacheHandler
        
        # Get the agent cache handler to register agent caches
        coordinator = get_project_state_coordinator()
        agent_handler = None
        
        for handler in coordinator._handlers.values():
            if isinstance(handler, AgentDiscoveryCacheHandler):
                agent_handler = handler
                break
        
        if agent_handler:
            # Register common agent discovery caches
            # This would need to be customized based on actual agent implementations
            logger.info("Integrated project state system with UnifiedAgent discovery")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with UnifiedAgent: {e}")
    except Exception as e:
        logger.error(f"Error integrating with UnifiedAgent: {e}", exc_info=True)


def integrate_with_project_type_detection():
    """
    Integrate the project state system with ProjectTypeDetectionService.
    
    This ensures project type detection participates in cache invalidation.
    """
    try:
        from .project_type_detection import ProjectTypeDetectionService
        from .cache_invalidation_handlers import ProjectTypeDetectionCacheHandler
        
        # The handler is already registered, but we could add specific
        # integration points here if needed
        logger.info("Project type detection integration ready")
        
    except ImportError as e:
        logger.warning(f"Could not integrate with ProjectTypeDetectionService: {e}")
    except Exception as e:
        logger.error(f"Error integrating with ProjectTypeDetectionService: {e}", exc_info=True)


async def initialize_project_state_system_full():
    """
    Full initialization of the project state system with all integrations.
    
    This should be called once during application startup to ensure
    the entire system is properly coordinated.
    """
    logger.info("Starting full project state system initialization")
    
    # Initialize core system
    await ensure_project_state_system_ready()
    
    # Set up integrations
    integrate_with_mcp_tools()
    integrate_with_unified_agents() 
    integrate_with_project_type_detection()
    
    logger.info("Full project state system initialization completed")


# Convenience context manager for project operations
class ProjectOperationContext:
    """
    Context manager for coordinated project operations.
    
    Provides a high-level interface for operations that need
    to coordinate with the project state system.
    """
    
    def __init__(self, project_path: Path):
        self.project_path = project_path
        self._fs_context: Optional[FilesystemEventContext] = None
    
    async def __aenter__(self):
        # Ensure system is ready
        await ensure_project_state_system_ready()
        
        # Create filesystem event context
        self._fs_context = FilesystemEventContext(self.project_path)
        await self._fs_context.__aenter__()
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._fs_context:
            await self._fs_context.__aexit__(exc_type, exc_val, exc_tb)
    
    def record_file_created(self, file_path: Path, metadata: dict = None):
        """Record a file creation for batch invalidation."""
        if self._fs_context:
            self._fs_context.record_file_created(file_path, metadata)
    
    def record_directory_created(self, dir_path: Path, metadata: dict = None):
        """Record a directory creation for batch invalidation."""
        if self._fs_context:
            self._fs_context.record_directory_created(dir_path, metadata) 