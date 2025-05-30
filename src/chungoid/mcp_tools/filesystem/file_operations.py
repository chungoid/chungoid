"""
File System Operations Tools

Core file operations with intelligent encoding detection, safe writing with backups,
and project-aware path resolution for autonomous agents.

These tools provide:
- Smart file reading with automatic encoding detection
- Safe file writing with backup creation and atomic operations
- File copying and moving with conflict resolution
- Protected deletion with recovery options
- Integration with project context and state persistence
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import hashlib
import chardet
from datetime import datetime

from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService

logger = logging.getLogger(__name__)


def _safe_relative_path(path: Path, root: Path) -> str:
    """
    Safely compute relative path without throwing ValueError.
    
    Args:
        path: The path to make relative
        root: The root path to make it relative to
        
    Returns:
        str: Relative path if possible, otherwise absolute path
    """
    try:
        return str(path.relative_to(root))
    except ValueError:
        # Path is not relative to root, return absolute path
        return str(path)


def _ensure_project_context(project_path: Optional[str] = None, project_id: Optional[str] = None) -> Path:
    """
    Ensures project context is set for file operations.
    
    Args:
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Path: The resolved project path
    """
    if project_path:
        return Path(project_path).resolve()
    elif project_id:
        # Use current working directory if project_id is provided without path
        return Path.cwd().resolve()
    else:
        return Path.cwd().resolve()


def _detect_encoding(file_path: Path) -> str:
    """
    Detect file encoding using chardet.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Detected encoding
    """
    try:
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            encoding = result.get('encoding', 'utf-8')
            confidence = result.get('confidence', 0.0)
            
            # Use utf-8 as fallback for low confidence detections
            if confidence < 0.5:
                encoding = 'utf-8'
                
            return encoding
    except Exception as e:
        logger.warning(f"Failed to detect encoding for {file_path}: {e}")
        return 'utf-8'


def _create_backup(file_path: Path, backup_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Create a backup of the file before modification.
    
    Args:
        file_path: Path to the file to backup
        backup_dir: Optional directory for backups
        
    Returns:
        Path: Path to the backup file, or None if backup failed
    """
    try:
        if not file_path.exists():
            return None
            
        if backup_dir is None:
            # Find the project root by walking up from file_path
            current_path = file_path.parent
            project_root = None
            
            # Look for project indicators (.chungoid, .git, etc.) or stop at filesystem root
            while current_path != current_path.parent:
                if any((current_path / indicator).exists() for indicator in ['.chungoid', '.git', 'goal.txt']):
                    project_root = current_path
                    break
                current_path = current_path.parent
            
            # If no project root found, use the file's parent directory
            if project_root is None:
                project_root = file_path.parent
                
            # Always use project root for backups - centralized location
            backup_dir = project_root / '.chungoid_backups'
            
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped backup filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.backup_{timestamp}"
        backup_path = backup_dir / backup_name
        
        shutil.copy2(file_path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to create backup for {file_path}: {e}")
        return None


async def filesystem_read_file(
    file_path: str,
    encoding: Optional[str] = None,
    detect_encoding: bool = True,
    max_size_mb: int = 50,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Smart file reading with encoding detection and validation.
    
    Provides intelligent file reading with automatic encoding detection,
    size validation, and comprehensive error handling.
    
    Args:
        file_path: Path to the file to read
        encoding: Specific encoding to use (optional)
        detect_encoding: Whether to auto-detect encoding
        max_size_mb: Maximum file size in MB
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing:
        - success: bool - Read operation success
        - content: str - File content
        - encoding: str - Used encoding
        - size_bytes: int - File size in bytes
        - metadata: Dict - File metadata
        - error: str - Error message if failed
    """
    try:
        # Ensure project context
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve file path relative to project
        if not os.path.isabs(file_path):
            resolved_path = project_root / file_path
        else:
            resolved_path = Path(file_path)
            
        resolved_path = resolved_path.resolve()
        
        # Validate file exists
        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {resolved_path}",
                "file_path": str(resolved_path)
            }
            
        # Check if it's a file
        if not resolved_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {resolved_path}",
                "file_path": str(resolved_path)
            }
            
        # Check file size
        file_size = resolved_path.stat().st_size
        max_size_bytes = max_size_mb * 1024 * 1024
        
        if file_size > max_size_bytes:
            return {
                "success": False,
                "error": f"File too large: {file_size} bytes (max: {max_size_bytes} bytes)",
                "file_path": str(resolved_path),
                "size_bytes": file_size
            }
        
        # Detect encoding if not specified
        if encoding is None and detect_encoding:
            encoding = _detect_encoding(resolved_path)
        elif encoding is None:
            encoding = 'utf-8'
            
        # Read file content
        with open(resolved_path, 'r', encoding=encoding, errors='replace') as f:
            content = f.read()
            
        # Gather file metadata
        stat = resolved_path.stat()
        metadata = {
            "size_bytes": file_size,
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_executable": os.access(resolved_path, os.X_OK),
            "file_extension": resolved_path.suffix,
            "mime_type": _guess_mime_type(resolved_path)
        }
        
        result = {
            "success": True,
            "content": content,
            "encoding": encoding,
            "size_bytes": file_size,
            "file_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "metadata": metadata,
            "project_path": str(project_root),
            "project_id": project_id
        }
        
        logger.info(f"Successfully read file: {resolved_path} ({file_size} bytes, {encoding})")
        return result
        
    except UnicodeDecodeError as e:
        return {
            "success": False,
            "error": f"Encoding error reading file: {str(e)}",
            "file_path": file_path,
            "encoding": encoding
        }
    except PermissionError as e:
        return {
            "success": False,
            "error": f"Permission denied reading file: {str(e)}",
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Failed to read file {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


async def filesystem_write_file(
    file_path: str,
    content: str,
    encoding: str = 'utf-8',
    create_backup: bool = True,
    atomic_write: bool = True,
    create_directories: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Safe file writing with backup and atomic operations.
    
    Provides safe file writing with automatic backup creation,
    atomic operations, and directory creation.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        encoding: Encoding to use for writing
        create_backup: Whether to create backup before writing
        atomic_write: Whether to use atomic write (write to temp then move)
        create_directories: Whether to create parent directories
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing:
        - success: bool - Write operation success
        - file_path: str - Written file path
        - size_bytes: int - Written content size
        - backup_path: str - Path to backup file (if created)
        - created_directories: List[str] - Created directory paths
        - error: str - Error message if failed
    """
    try:
        # Ensure project context
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve file path
        if not os.path.isabs(file_path):
            resolved_path = project_root / file_path
        else:
            resolved_path = Path(file_path)
            
        resolved_path = resolved_path.resolve()
        
        # Create parent directories if needed
        created_directories = []
        if create_directories and not resolved_path.parent.exists():
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            created_directories.append(str(resolved_path.parent))
            logger.info(f"Created directories: {resolved_path.parent}")
        
        # Create backup if file exists and backup is requested
        backup_path = None
        if create_backup and resolved_path.exists():
            backup_path = _create_backup(resolved_path)
            
        # Write content
        content_bytes = len(content.encode(encoding))
        
        if atomic_write:
            # Atomic write: write to temporary file then move
            temp_dir = resolved_path.parent
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding=encoding,
                dir=temp_dir,
                delete=False,
                prefix=f'.tmp_{resolved_path.name}_'
            ) as temp_file:
                temp_file.write(content)
                temp_path = Path(temp_file.name)
                
            # Atomic move
            shutil.move(str(temp_path), str(resolved_path))
        else:
            # Direct write
            with open(resolved_path, 'w', encoding=encoding) as f:
                f.write(content)
        
        result = {
            "success": True,
            "file_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "size_bytes": content_bytes,
            "encoding": encoding,
            "backup_path": str(backup_path) if backup_path else None,
            "created_directories": created_directories,
            "atomic_write": atomic_write,
            "project_path": str(project_root),
            "project_id": project_id
        }
        
        logger.info(f"Successfully wrote file: {resolved_path} ({content_bytes} bytes)")
        return result
        
    except PermissionError as e:
        return {
            "success": False,
            "error": f"Permission denied writing file: {str(e)}",
            "file_path": file_path
        }
    except OSError as e:
        return {
            "success": False,
            "error": f"OS error writing file: {str(e)}",
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Failed to write file {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


async def filesystem_copy_file(
    source_path: str,
    destination_path: str,
    overwrite: bool = False,
    preserve_metadata: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Copy file with conflict resolution and metadata preservation.
    
    Args:
        source_path: Source file path
        destination_path: Destination file path
        overwrite: Whether to overwrite existing files
        preserve_metadata: Whether to preserve file metadata
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing copy operation results
    """
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve paths
        if not os.path.isabs(source_path):
            source = project_root / source_path
        else:
            source = Path(source_path)
            
        if not os.path.isabs(destination_path):
            destination = project_root / destination_path
        else:
            destination = Path(destination_path)
            
        source = source.resolve()
        destination = destination.resolve()
        
        # Validate source exists
        if not source.exists():
            return {
                "success": False,
                "error": f"Source file does not exist: {source}",
                "source_path": str(source)
            }
            
        # Check destination conflicts
        if destination.exists() and not overwrite:
            return {
                "success": False,
                "error": f"Destination file exists and overwrite=False: {destination}",
                "destination_path": str(destination)
            }
        
        # Create destination directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        if preserve_metadata:
            shutil.copy2(source, destination)
        else:
            shutil.copy(source, destination)
            
        # Get file sizes
        source_size = source.stat().st_size
        dest_size = destination.stat().st_size
        
        result = {
            "success": True,
            "source_path": str(source),
            "destination_path": str(destination),
            "source_relative": _safe_relative_path(source, project_root),
            "destination_relative": _safe_relative_path(destination, project_root),
            "size_bytes": source_size,
            "preserved_metadata": preserve_metadata,
            "overwrite": overwrite and destination.exists(),
            "project_path": str(project_root)
        }
        
        logger.info(f"Successfully copied file: {source} -> {destination}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to copy file {source_path} -> {destination_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_path": source_path,
            "destination_path": destination_path
        }


async def filesystem_move_file(
    source_path: str,
    destination_path: str,
    overwrite: bool = False,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Move file with conflict resolution.
    
    Args:
        source_path: Source file path
        destination_path: Destination file path  
        overwrite: Whether to overwrite existing files
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing move operation results
    """
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve paths
        if not os.path.isabs(source_path):
            source = project_root / source_path
        else:
            source = Path(source_path)
            
        if not os.path.isabs(destination_path):
            destination = project_root / destination_path  
        else:
            destination = Path(destination_path)
            
        source = source.resolve()
        destination = destination.resolve()
        
        # Validate source exists
        if not source.exists():
            return {
                "success": False,
                "error": f"Source file does not exist: {source}",
                "source_path": str(source)
            }
            
        # Check destination conflicts
        if destination.exists() and not overwrite:
            return {
                "success": False,
                "error": f"Destination file exists and overwrite=False: {destination}",
                "destination_path": str(destination)
            }
        
        # Get source size before move
        source_size = source.stat().st_size
        
        # Create destination directory if needed
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Move file
        shutil.move(source, destination)
        
        result = {
            "success": True,
            "source_path": str(source),
            "destination_path": str(destination),
            "source_relative": _safe_relative_path(source, project_root),
            "destination_relative": _safe_relative_path(destination, project_root),
            "size_bytes": source_size,
            "overwrite": overwrite and destination.exists(),
            "project_path": str(project_root)
        }
        
        logger.info(f"Successfully moved file: {source} -> {destination}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to move file {source_path} -> {destination_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_path": source_path,
            "destination_path": destination_path
        }


async def filesystem_safe_delete(
    file_path: str,
    create_backup: bool = True,
    confirm: bool = False,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Protected file deletion with backup and confirmation.
    
    Args:
        file_path: Path to the file to delete
        create_backup: Whether to create backup before deletion
        confirm: Confirmation flag to prevent accidental deletion
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing deletion operation results
    """
    try:
        if not confirm:
            return {
                "success": False,
                "error": "Deletion requires explicit confirmation (confirm=True)",
                "file_path": file_path
            }
            
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve path
        if not os.path.isabs(file_path):
            resolved_path = project_root / file_path
        else:
            resolved_path = Path(file_path)
            
        resolved_path = resolved_path.resolve()
        
        # Validate file exists
        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {resolved_path}",
                "file_path": str(resolved_path)
            }
            
        # Get file info before deletion
        file_size = resolved_path.stat().st_size
        
        # Create backup if requested
        backup_path = None
        if create_backup:
            backup_path = _create_backup(resolved_path)
            
        # Delete file
        resolved_path.unlink()
        
        result = {
            "success": True,
            "deleted_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "size_bytes": file_size,
            "backup_path": str(backup_path) if backup_path else None,
            "backup_created": backup_path is not None,
            "project_path": str(project_root)
        }
        
        logger.info(f"Successfully deleted file: {resolved_path}")
        return result
        
    except PermissionError as e:
        return {
            "success": False,
            "error": f"Permission denied deleting file: {str(e)}",
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Failed to delete file {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


# BIG-BANG FIX: Add missing filesystem_delete_file function
async def filesystem_delete_file(
    file_path: str,
    confirm: bool = True,
    create_backup: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete a file permanently (alias for filesystem_safe_delete with confirm=True by default).
    
    Args:
        file_path: Path to the file to delete
        confirm: Confirmation flag to prevent accidental deletion (defaults to True)
        create_backup: Whether to create backup before deletion
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing deletion operation results
    """
    return await filesystem_safe_delete(
        file_path=file_path,
        create_backup=create_backup,
        confirm=confirm,
        project_path=project_path,
        project_id=project_id
    )


# BIG-BANG FIX: Add missing filesystem_get_file_info function
async def filesystem_get_file_info(
    file_path: str,
    include_content_stats: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get detailed file information and metadata.
    
    Args:
        file_path: Path to the file to analyze
        include_content_stats: Whether to include content analysis (line count, etc.)
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing detailed file information
    """
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve path
        if not os.path.isabs(file_path):
            resolved_path = project_root / file_path
        else:
            resolved_path = Path(file_path)
            
        resolved_path = resolved_path.resolve()
        
        # Validate file exists
        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {resolved_path}",
                "file_path": str(resolved_path)
            }
            
        if not resolved_path.is_file():
            return {
                "success": False,
                "error": f"Path is not a file: {resolved_path}",
                "file_path": str(resolved_path)
            }
        
        # Get file stats
        stat = resolved_path.stat()
        
        file_info = {
            "success": True,
            "file_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "name": resolved_path.name,
            "extension": resolved_path.suffix,
            "size_bytes": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "accessed_time": datetime.fromtimestamp(stat.st_atime).isoformat(),
            "permissions": oct(stat.st_mode)[-3:],
            "is_executable": os.access(resolved_path, os.X_OK),
            "is_readable": os.access(resolved_path, os.R_OK),
            "is_writable": os.access(resolved_path, os.W_OK),
            "mime_type": _guess_mime_type(resolved_path),
            "project_path": str(project_root)
        }
        
        # Add content stats if requested
        if include_content_stats and stat.st_size > 0:
            try:
                # Detect encoding first
                encoding = _detect_encoding(resolved_path)
                
                # Read content for analysis
                with open(resolved_path, 'r', encoding=encoding, errors='replace') as f:
                    content = f.read()
                    
                file_info.update({
                    "encoding": encoding,
                    "line_count": len(content.splitlines()),
                    "character_count": len(content),
                    "word_count": len(content.split()),
                    "is_binary": False
                })
                
            except (UnicodeDecodeError, PermissionError):
                file_info.update({
                    "encoding": "binary",
                    "is_binary": True,
                    "line_count": 0,
                    "character_count": 0,
                    "word_count": 0
                })
        
        logger.info(f"Retrieved file info: {resolved_path}")
        return file_info
        
    except Exception as e:
        logger.error(f"Failed to get file info {file_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path
        }


# BIG-BANG FIX: Add missing filesystem_search_files function
async def filesystem_search_files(
    directory_path: str,
    pattern: str = "*",
    search_type: str = "name",  # "name", "content", "both"
    case_sensitive: bool = False,
    max_results: int = 100,
    file_extensions: Optional[List[str]] = None,
    recursive: bool = True,
    max_depth: int = 10,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for files based on name patterns or content.
    
    Args:
        directory_path: Directory to search in
        pattern: Search pattern (glob for name, regex for content)
        search_type: Type of search - "name", "content", or "both"
        case_sensitive: Whether search should be case sensitive
        max_results: Maximum number of results to return
        file_extensions: Filter by file extensions
        recursive: Whether to search recursively
        max_depth: Maximum recursion depth
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing search results
    """
    import fnmatch
    import re
    
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve directory path
        if not os.path.isabs(directory_path):
            resolved_path = project_root / directory_path
        else:
            resolved_path = Path(directory_path)
            
        resolved_path = resolved_path.resolve()
        
        # Validate directory exists
        if not resolved_path.exists():
            return {
                "success": False,
                "error": f"Directory does not exist: {resolved_path}",
                "directory_path": str(resolved_path)
            }
            
        if not resolved_path.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {resolved_path}",
                "directory_path": str(resolved_path)
            }
        
        matches = []
        total_searched = 0
        
        # Prepare search pattern for content search
        if search_type in ["content", "both"]:
            try:
                content_regex = re.compile(pattern, 0 if case_sensitive else re.IGNORECASE)
            except re.error as e:
                return {
                    "success": False,
                    "error": f"Invalid regex pattern: {pattern} - {e}",
                    "pattern": pattern
                }
        
        def _search_directory(current_path: Path, current_depth: int = 0):
            nonlocal total_searched, matches
            
            if current_depth > max_depth or len(matches) >= max_results:
                return
                
            try:
                for item in current_path.iterdir():
                    if len(matches) >= max_results:
                        break
                        
                    # Skip common ignore patterns
                    if item.name.startswith('.') and item.name not in ['.', '..']:
                        continue
                        
                    if item.is_file():
                        total_searched += 1
                        
                        # Filter by file extension
                        if file_extensions and item.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                            continue
                        
                        match_found = False
                        match_info = {
                            "file_path": str(item),
                            "relative_path": _safe_relative_path(item, project_root),
                            "name": item.name,
                            "size_bytes": item.stat().st_size,
                            "modified_time": datetime.fromtimestamp(item.stat().st_mtime).isoformat(),
                            "match_type": [],
                            "content_matches": []
                        }
                        
                        # Name-based search
                        if search_type in ["name", "both"]:
                            if case_sensitive:
                                name_match = fnmatch.fnmatch(item.name, pattern)
                            else:
                                name_match = fnmatch.fnmatch(item.name.lower(), pattern.lower())
                                
                            if name_match:
                                match_found = True
                                match_info["match_type"].append("name")
                        
                        # Content-based search
                        if search_type in ["content", "both"] and item.stat().st_size > 0:
                            try:
                                # Only search text files
                                if _guess_mime_type(item).startswith('text/'):
                                    encoding = _detect_encoding(item)
                                    with open(item, 'r', encoding=encoding, errors='replace') as f:
                                        content = f.read()
                                        
                                    content_matches = list(content_regex.finditer(content))
                                    if content_matches:
                                        match_found = True
                                        match_info["match_type"].append("content")
                                        
                                        # Add match details
                                        for match in content_matches[:5]:  # Limit to first 5 matches
                                            lines = content[:match.start()].count('\n') + 1
                                            match_info["content_matches"].append({
                                                "line": lines,
                                                "text": match.group(),
                                                "start": match.start(),
                                                "end": match.end()
                                            })
                                            
                            except (UnicodeDecodeError, PermissionError, OSError):
                                # Skip binary or inaccessible files
                                pass
                        
                        if match_found:
                            matches.append(match_info)
                            
                    elif item.is_dir() and recursive and current_depth < max_depth:
                        _search_directory(item, current_depth + 1)
                        
            except PermissionError:
                # Skip directories we can't access
                pass
        
        # Start search
        _search_directory(resolved_path)
        
        result = {
            "success": True,
            "matches": matches,
            "total_matches": len(matches),
            "total_searched": total_searched,
            "pattern": pattern,
            "search_type": search_type,
            "directory_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "case_sensitive": case_sensitive,
            "recursive": recursive,
            "max_results": max_results,
            "truncated": len(matches) >= max_results,
            "project_path": str(project_root)
        }
        
        logger.info(f"Search completed: {len(matches)} matches in {total_searched} files")
        return result
        
    except Exception as e:
        logger.error(f"Failed to search files in {directory_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path,
            "pattern": pattern
        }


def _guess_mime_type(file_path: Path) -> str:
    """
    Guess MIME type based on file extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Guessed MIME type
    """
    import mimetypes
    
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


# Setup logging
logger = logging.getLogger(__name__) 