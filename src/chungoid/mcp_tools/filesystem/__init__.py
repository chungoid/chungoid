"""
File System MCP Tools

Project-aware file system tools that provide intelligent and safe file operations
for autonomous agents with enhanced functionality over standard file operations.

These tools handle:
- Smart file reading with encoding detection and validation
- Safe file writing with backup and conflict resolution
- Project-aware directory traversal and scanning
- Efficient bulk file operations with atomic semantics
- Protected deletion with safety checks and recovery options

Features:
- Project context awareness with automatic path resolution
- Integration with execution state persistence for rollback capabilities
- Advanced error handling and validation
- Support for various file types and encodings
- Atomic operations and transaction-like semantics
"""

from .file_operations import (
    filesystem_read_file,
    filesystem_write_file,
    filesystem_copy_file,
    filesystem_move_file,
    filesystem_safe_delete,
    filesystem_delete_file,
    filesystem_get_file_info,
    filesystem_search_files,
)

from .directory_operations import (
    filesystem_list_directory,
    filesystem_create_directory,
    filesystem_delete_directory,
    filesystem_project_scan,
    filesystem_sync_directories,
)

from .batch_operations import (
    filesystem_batch_operations,
    filesystem_backup_restore,
    filesystem_template_expansion,
)

__all__ = [
    # File Operations Tools
    "filesystem_read_file",
    "filesystem_write_file", 
    "filesystem_copy_file",
    "filesystem_move_file",
    "filesystem_safe_delete",
    "filesystem_delete_file",
    "filesystem_get_file_info",
    "filesystem_search_files",
    
    # Directory Operations Tools
    "filesystem_list_directory",
    "filesystem_create_directory",
    "filesystem_delete_directory",
    "filesystem_project_scan",
    "filesystem_sync_directories",
    
    # Batch & Advanced Operations Tools
    "filesystem_batch_operations",
    "filesystem_backup_restore", 
    "filesystem_template_expansion",
] 