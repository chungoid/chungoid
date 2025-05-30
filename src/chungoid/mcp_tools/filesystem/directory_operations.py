"""
Directory Operations Tools

Project-aware directory operations with intelligent scanning, listing,
and synchronization capabilities for autonomous agents.

These tools provide:
- Smart directory listing with filtering and metadata
- Project-aware scanning with type detection
- Directory creation with proper permissions
- Directory synchronization and comparison
- Integration with project context and patterns
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Set
import json
import fnmatch
from datetime import datetime

from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.project_type_detection import ProjectTypeDetectionService

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
    Ensures project context is set for directory operations.
    
    Args:
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Path: The resolved project path
    """
    if project_path:
        return Path(project_path).resolve()
    elif project_id:
        return Path.cwd().resolve()
    else:
        return Path.cwd().resolve()


def _get_default_ignore_patterns() -> List[str]:
    """
    Get default ignore patterns for common development files.
    
    Returns:
        List[str]: Default ignore patterns
    """
    return [
        # Version control
        '.git', '.svn', '.hg', '.bzr',
        # Python
        '__pycache__', '*.pyc', '*.pyo', '*.pyd', '.Python',
        'env', 'venv', '.venv', 'ENV', 'env.bak', 'venv.bak',
        '.pytest_cache', '.coverage', 'htmlcov',
        # Node.js
        'node_modules', 'npm-debug.log*', 'yarn-debug.log*', 'yarn-error.log*',
        '.npm', '.yarn-integrity',
        # Build outputs
        'build', 'dist', 'target', 'bin', 'obj',
        # IDE and editors
        '.vscode', '.idea', '*.swp', '*.swo', '*~',
        # OS files
        '.DS_Store', 'Thumbs.db', 'desktop.ini',
        # Temporary files
        '*.tmp', '*.temp', '.tmp', '.temp',
        # Logs
        '*.log', 'logs',
        # Backups
        '.chungoid_backups', '*.backup', '*.bak'
    ]


def _should_ignore_path(path: Path, ignore_patterns: List[str]) -> bool:
    """
    Check if a path should be ignored based on patterns.
    
    Args:
        path: Path to check
        ignore_patterns: List of ignore patterns
        
    Returns:
        bool: True if path should be ignored
    """
    path_str = str(path)
    name = path.name
    
    for pattern in ignore_patterns:
        # Check if pattern matches the full path or just the name
        if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(name, pattern):
            return True
            
        # Check if any parent directory matches the pattern
        for parent in path.parents:
            if fnmatch.fnmatch(parent.name, pattern):
                return True
                
    return False


async def filesystem_list_directory(
    directory_path: str,
    recursive: bool = False,
    include_files: bool = True,
    include_directories: bool = True,
    max_depth: int = 10,
    ignore_patterns: Optional[List[str]] = None,
    file_extensions: Optional[List[str]] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    List directory contents with filtering and metadata.
    
    Provides intelligent directory listing with support for filtering,
    recursion limits, and detailed file metadata.
    
    Args:
        directory_path: Path to the directory to list
        recursive: Whether to list recursively
        include_files: Whether to include files in results
        include_directories: Whether to include directories in results
        max_depth: Maximum recursion depth
        ignore_patterns: Patterns to ignore (defaults to common dev files)
        file_extensions: Filter by file extensions (e.g., ['.py', '.js'])
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing:
        - success: bool - Operation success
        - items: List - Directory items with metadata
        - total_files: int - Total file count
        - total_directories: int - Total directory count
        - total_size_bytes: int - Total size of all files
        - directory_path: str - Listed directory path
    """
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
        
        # Use default ignore patterns if none provided
        if ignore_patterns is None:
            ignore_patterns = _get_default_ignore_patterns()
            
        items = []
        total_files = 0
        total_directories = 0
        total_size_bytes = 0
        
        def _scan_directory(current_path: Path, current_depth: int = 0):
            nonlocal total_files, total_directories, total_size_bytes
            
            if current_depth > max_depth:
                return
                
            try:
                for item in current_path.iterdir():
                    # Check if should be ignored
                    if _should_ignore_path(item, ignore_patterns):
                        continue
                        
                    # Get item metadata
                    try:
                        stat = item.stat()
                        is_file = item.is_file()
                        is_dir = item.is_dir()
                        
                        # FIRST: Handle recursion (before filtering) to ensure subdirectories are scanned
                        if recursive and is_dir and current_depth < max_depth:
                            _scan_directory(item, current_depth + 1)
                        
                        # SECOND: Apply filtering for inclusion in results
                        if is_file and not include_files:
                            continue
                        if is_dir and not include_directories:
                            continue
                            
                        # Filter by file extension
                        if is_file and file_extensions:
                            if item.suffix.lower() not in [ext.lower() for ext in file_extensions]:
                                continue
                        
                        # Calculate relative path safely
                        relative_path = _safe_relative_path(item, project_root)
                        
                        item_info = {
                            "name": item.name,
                            "path": str(item),
                            "relative_path": relative_path,
                            "type": "file" if is_file else "directory",
                            "size_bytes": stat.st_size if is_file else 0,
                            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                            "permissions": oct(stat.st_mode)[-3:],
                            "depth": current_depth,
                        }
                        
                        if is_file:
                            total_files += 1
                            total_size_bytes += stat.st_size
                            item_info.update({
                                "extension": item.suffix,
                                "mime_type": _guess_mime_type(item)
                            })
                        else:
                            total_directories += 1
                            
                        items.append(item_info)
                            
                    except (PermissionError, OSError) as e:
                        logger.warning(f"Cannot access {item}: {e}")
                        continue
                        
            except PermissionError as e:
                logger.warning(f"Cannot access directory {current_path}: {e}")
                
        # Start scanning
        _scan_directory(resolved_path)
        
        # Sort items by type then name
        items.sort(key=lambda x: (x["type"], x["name"]))
        
        # Calculate relative path safely for the main directory
        directory_relative_path = _safe_relative_path(resolved_path, project_root)
        
        result = {
            "success": True,
            "items": items,
            "total_files": total_files,
            "total_directories": total_directories,
            "total_size_bytes": total_size_bytes,
            "directory_path": str(resolved_path),
            "relative_path": directory_relative_path,
            "recursive": recursive,
            "max_depth": max_depth,
            "project_path": str(project_root)
        }
        
        logger.info(f"Listed directory: {resolved_path} ({total_files} files, {total_directories} dirs)")
        return result
        
    except Exception as e:
        logger.error(f"Failed to list directory {directory_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path
        }


async def filesystem_create_directory(
    directory_path: str,
    create_parents: bool = True,
    permissions: Optional[str] = None,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create directory with parent creation and permission setting.
    
    Args:
        directory_path: Path to the directory to create
        create_parents: Whether to create parent directories
        permissions: Octal permissions string (e.g., "755")
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing creation operation results
    """
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Resolve directory path
        if not os.path.isabs(directory_path):
            resolved_path = project_root / directory_path
        else:
            resolved_path = Path(directory_path)
            
        resolved_path = resolved_path.resolve()
        
        # Check if directory already exists
        if resolved_path.exists():
            if resolved_path.is_dir():
                return {
                    "success": True,
                    "directory_path": str(resolved_path),
                    "relative_path": _safe_relative_path(resolved_path, project_root),
                    "already_exists": True,
                    "project_path": str(project_root)
                }
            else:
                return {
                    "success": False,
                    "error": f"Path exists but is not a directory: {resolved_path}",
                    "directory_path": str(resolved_path)
                }
        
        # Create directory
        resolved_path.mkdir(parents=create_parents, exist_ok=False)
        
        # Set permissions if specified
        if permissions:
            try:
                mode = int(permissions, 8)
                resolved_path.chmod(mode)
            except ValueError:
                logger.warning(f"Invalid permissions format: {permissions}")
        
        result = {
            "success": True,
            "directory_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "already_exists": False,
            "created_parents": create_parents,
            "permissions": permissions,
            "project_path": str(project_root)
        }
        
        logger.info(f"Successfully created directory: {resolved_path}")
        return result
        
    except FileExistsError:
        return {
            "success": True,
            "directory_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "already_exists": True,
            "project_path": str(project_root)
        }
    except PermissionError as e:
        return {
            "success": False,
            "error": f"Permission denied creating directory: {str(e)}",
            "directory_path": directory_path
        }
    except Exception as e:
        logger.error(f"Failed to create directory {directory_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path
        }


async def filesystem_delete_directory(
    directory_path: str,
    recursive: bool = False,
    confirm: bool = False,
    create_backup: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Delete directory with safety checks and backup options.
    
    Args:
        directory_path: Path to the directory to delete
        recursive: Whether to delete recursively (required for non-empty directories)
        confirm: Confirmation flag to prevent accidental deletion
        create_backup: Whether to create backup before deletion
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
                "directory_path": directory_path
            }
            
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
        
        # Check if directory is empty
        try:
            items = list(resolved_path.iterdir())
            is_empty = len(items) == 0
        except PermissionError:
            return {
                "success": False,
                "error": f"Permission denied accessing directory: {resolved_path}",
                "directory_path": str(resolved_path)
            }
        
        if not is_empty and not recursive:
            return {
                "success": False,
                "error": f"Directory is not empty and recursive=False: {resolved_path}",
                "directory_path": str(resolved_path),
                "item_count": len(items)
            }
        
        # Create backup if requested
        backup_path = None
        if create_backup and not is_empty:
            try:
                import shutil
                import tempfile
                from datetime import datetime
                
                # Create backup in .chungoid_backups directory
                backup_dir = project_root / ".chungoid_backups"
                backup_dir.mkdir(exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"{resolved_path.name}_backup_{timestamp}"
                backup_path = backup_dir / backup_name
                
                shutil.copytree(resolved_path, backup_path)
                logger.info(f"Created backup: {backup_path}")
                
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")
        
        # Get stats before deletion
        total_files = 0
        total_size = 0
        if not is_empty:
            for item in resolved_path.rglob("*"):
                if item.is_file():
                    try:
                        total_files += 1
                        total_size += item.stat().st_size
                    except (PermissionError, OSError):
                        pass
        
        # Delete directory
        if is_empty:
            resolved_path.rmdir()
        else:
            import shutil
            shutil.rmtree(resolved_path)
        
        result = {
            "success": True,
            "deleted_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "was_empty": is_empty,
            "recursive": recursive,
            "total_files_deleted": total_files,
            "total_size_deleted": total_size,
            "backup_path": str(backup_path) if backup_path else None,
            "backup_created": backup_path is not None,
            "project_path": str(project_root)
        }
        
        logger.info(f"Successfully deleted directory: {resolved_path}")
        return result
        
    except PermissionError as e:
        return {
            "success": False,
            "error": f"Permission denied deleting directory: {str(e)}",
            "directory_path": directory_path
        }
    except Exception as e:
        logger.error(f"Failed to delete directory {directory_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path
        }


async def filesystem_project_scan(
    scan_path: Optional[str] = None,
    detect_project_type: bool = True,
    analyze_structure: bool = True,
    include_stats: bool = True,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Project-aware scanning with type detection and structure analysis.
    
    Provides comprehensive project scanning with intelligent type detection,
    structure analysis, and development-focused insights.
    
    Args:
        scan_path: Path to scan (defaults to project root)
        detect_project_type: Whether to detect project type
        analyze_structure: Whether to analyze project structure
        include_stats: Whether to include detailed statistics
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing:
        - success: bool - Scan operation success
        - project_type: Dict - Detected project type information
        - structure: Dict - Project structure analysis
        - statistics: Dict - File and directory statistics
        - key_files: List - Important project files
        - recommendations: List - Structure recommendations
    """
    try:
        project_root = _ensure_project_context(project_path, project_id)
        
        # Use project root if no scan path specified
        if scan_path is None:
            scan_path = str(project_root)
            
        # Resolve scan path
        if not os.path.isabs(scan_path):
            resolved_path = project_root / scan_path
        else:
            resolved_path = Path(scan_path)
            
        resolved_path = resolved_path.resolve()
        
        if not resolved_path.exists() or not resolved_path.is_dir():
            return {
                "success": False,
                "error": f"Scan path does not exist or is not a directory: {resolved_path}",
                "scan_path": str(resolved_path)
            }
        
        result = {
            "success": True,
            "scan_path": str(resolved_path),
            "relative_path": _safe_relative_path(resolved_path, project_root),
            "project_path": str(project_root)
        }
        
        # Project type detection
        if detect_project_type:
            try:
                detection_service = ProjectTypeDetectionService()
                detection_result = detection_service.detect_project_type(resolved_path)
                
                # CRITICAL FIX: Use correct ProjectTypeDetectionResult schema attributes
                result["project_type"] = {
                    "primary_language": detection_result.primary_language,
                    "language_confidence": detection_result.language_confidence,
                    "frameworks": [
                        {
                            "name": fw.name,
                            "category": fw.category,
                            "confidence": fw.confidence,
                            "evidence": fw.evidence
                        } for fw in detection_result.frameworks
                    ],
                    "build_tools": [
                        {
                            "name": bt.name,
                            "category": bt.category,
                            "confidence": bt.confidence,
                            "evidence": bt.evidence
                        } for bt in detection_result.build_tools
                    ],
                    "testing_frameworks": [
                        {
                            "name": tf.name,
                            "category": tf.category,
                            "confidence": tf.confidence,
                            "evidence": tf.evidence
                        } for tf in detection_result.testing_frameworks
                    ],
                    "deployment_tools": [
                        {
                            "name": dt.name,
                            "category": dt.category,
                            "confidence": dt.confidence,
                            "evidence": dt.evidence
                        } for dt in detection_result.deployment_tools
                    ],
                    "project_structure_type": detection_result.project_structure_type,
                    "config_files": detection_result.config_files,
                    "overall_confidence": detection_result.overall_confidence,
                    "detection_metadata": detection_result.detection_metadata
                }
            except Exception as e:
                logger.warning(f"Project type detection failed: {e}")
                result["project_type"] = {"error": str(e)}
        
        # Structure analysis
        if analyze_structure:
            structure = await _analyze_project_structure(resolved_path)
            result["structure"] = structure
            
        # Statistics
        if include_stats:
            stats = await _calculate_project_statistics(resolved_path)
            result["statistics"] = stats
            
        # Key files identification
        key_files = _identify_key_files(resolved_path)
        result["key_files"] = key_files
        
        # Recommendations
        recommendations = _generate_structure_recommendations(result)
        result["recommendations"] = recommendations
        
        logger.info(f"Project scan completed: {resolved_path}")
        return result
        
    except Exception as e:
        logger.error(f"Failed to scan project {scan_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "scan_path": scan_path
        }


async def filesystem_sync_directories(
    source_path: str,
    destination_path: str,
    sync_mode: str = "update",  # "update", "mirror", "merge"
    dry_run: bool = False,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Synchronize directories with different modes.
    
    Args:
        source_path: Source directory path
        destination_path: Destination directory path
        sync_mode: Synchronization mode ("update", "mirror", "merge")
        dry_run: Whether to perform a dry run without actual changes
        project_path: Optional project directory path
        project_id: Optional project identifier
        
    Returns:
        Dict containing synchronization results
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
        if not source.exists() or not source.is_dir():
            return {
                "success": False,
                "error": f"Source directory does not exist: {source}",
                "source_path": str(source)
            }
        
        operations = []
        
        # Different sync modes
        if sync_mode == "update":
            # Copy newer files from source to destination
            operations = await _plan_update_sync(source, destination)
        elif sync_mode == "mirror":
            # Make destination identical to source
            operations = await _plan_mirror_sync(source, destination)
        elif sync_mode == "merge":
            # Merge both directories, keeping newer files
            operations = await _plan_merge_sync(source, destination)
        else:
            return {
                "success": False,
                "error": f"Invalid sync mode: {sync_mode}. Use 'update', 'mirror', or 'merge'",
                "sync_mode": sync_mode
            }
        
        # Execute operations if not dry run
        executed_operations = []
        if not dry_run:
            for op in operations:
                try:
                    if op["action"] == "copy":
                        shutil.copy2(op["source"], op["destination"])
                    elif op["action"] == "delete":
                        if Path(op["path"]).is_file():
                            Path(op["path"]).unlink()
                        else:
                            shutil.rmtree(op["path"])
                    executed_operations.append(op)
                except Exception as e:
                    op["error"] = str(e)
                    logger.error(f"Sync operation failed: {op}: {e}")
        
        result = {
            "success": True,
            "source_path": str(source),
            "destination_path": str(destination),
            "sync_mode": sync_mode,
            "dry_run": dry_run,
            "planned_operations": len(operations),
            "executed_operations": len(executed_operations),
            "operations": operations if dry_run else executed_operations,
            "project_path": str(project_root)
        }
        
        logger.info(f"Directory sync completed: {source} -> {destination} ({sync_mode})")
        return result
        
    except Exception as e:
        logger.error(f"Failed to sync directories {source_path} -> {destination_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "source_path": source_path,
            "destination_path": destination_path
        }


# Helper functions for project scanning and analysis

async def _analyze_project_structure(project_path: Path) -> Dict[str, Any]:
    """Analyze project directory structure."""
    structure = {
        "source_directories": [],
        "config_files": [],
        "documentation": [],
        "tests": [],
        "build_artifacts": [],
        "dependency_files": []
    }
    
    common_patterns = {
        "source_directories": ["src", "lib", "app", "source", "code"],
        "config_files": ["config", "conf", "settings", "cfg"],
        "documentation": ["docs", "doc", "documentation", "README*", "*.md"],
        "tests": ["test", "tests", "spec", "specs", "__tests__"],
        "build_artifacts": ["build", "dist", "target", "bin", "out"],
        "dependency_files": ["requirements.txt", "package.json", "Pipfile", "pyproject.toml"]
    }
    
    try:
        for item in project_path.rglob("*"):
            if item.is_dir():
                name = item.name.lower()
                for category, patterns in common_patterns.items():
                    if category != "dependency_files":  # Skip files for directory check
                        for pattern in patterns:
                            if fnmatch.fnmatch(name, pattern.lower()):
                                structure[category].append(str(item.relative_to(project_path)))
                                break
            else:
                name = item.name
                # Check for dependency files
                for pattern in common_patterns["dependency_files"]:
                    if fnmatch.fnmatch(name, pattern):
                        structure["dependency_files"].append(str(item.relative_to(project_path)))
    except Exception as e:
        logger.warning(f"Structure analysis error: {e}")
    
    return structure


async def _calculate_project_statistics(project_path: Path) -> Dict[str, Any]:
    """Calculate project statistics."""
    stats = {
        "total_files": 0,
        "total_directories": 0,
        "total_size_bytes": 0,
        "file_types": {},
        "language_distribution": {},
        "largest_files": []
    }
    
    file_sizes = []
    ignore_patterns = _get_default_ignore_patterns()
    
    try:
        for item in project_path.rglob("*"):
            if _should_ignore_path(item, ignore_patterns):
                continue
                
            if item.is_file():
                try:
                    size = item.stat().st_size
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += size
                    
                    # Track file types
                    ext = item.suffix.lower()
                    if ext:
                        stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
                    # Track for largest files
                    file_sizes.append((str(item.relative_to(project_path)), size))
                    
                except (PermissionError, OSError):
                    continue
            elif item.is_dir():
                stats["total_directories"] += 1
    
        # Get largest files
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        stats["largest_files"] = file_sizes[:10]  # Top 10 largest files
        
    except Exception as e:
        logger.warning(f"Statistics calculation error: {e}")
    
    return stats


def _identify_key_files(project_path: Path) -> List[Dict[str, Any]]:
    """Identify key project files."""
    key_files = []
    
    important_files = [
        ("README.md", "documentation"),
        ("README.rst", "documentation"), 
        ("README.txt", "documentation"),
        ("requirements.txt", "dependencies"),
        ("package.json", "dependencies"),
        ("Pipfile", "dependencies"),
        ("pyproject.toml", "build_config"),
        ("setup.py", "build_config"),
        ("Dockerfile", "deployment"),
        ("docker-compose.yml", "deployment"),
        (".gitignore", "version_control"),
        ("Makefile", "build_system"),
        ("tox.ini", "testing"),
        ("pytest.ini", "testing")
    ]
    
    for filename, category in important_files:
        file_path = project_path / filename
        if file_path.exists():
            try:
                stat = file_path.stat()
                key_files.append({
                    "name": filename,
                    "category": category,
                    "path": str(file_path.relative_to(project_path)),
                    "size_bytes": stat.st_size,
                    "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
            except (PermissionError, OSError):
                continue
    
    return key_files


def _generate_structure_recommendations(scan_result: Dict[str, Any]) -> List[str]:
    """Generate structure improvement recommendations."""
    recommendations = []
    
    # Check for common missing files
    key_files = scan_result.get("key_files", [])
    key_file_names = [f["name"] for f in key_files]
    
    if "README.md" not in key_file_names and "README.rst" not in key_file_names:
        recommendations.append("Consider adding a README file for project documentation")
    
    if "requirements.txt" not in key_file_names and "package.json" not in key_file_names:
        recommendations.append("Consider adding a dependency management file")
    
    if ".gitignore" not in key_file_names:
        recommendations.append("Consider adding a .gitignore file for version control")
    
    # Check structure
    structure = scan_result.get("structure", {})
    if not structure.get("tests"):
        recommendations.append("Consider adding a tests directory for unit tests")
    
    if not structure.get("documentation"):
        recommendations.append("Consider adding a docs directory for detailed documentation")
    
    return recommendations


async def _plan_update_sync(source: Path, destination: Path) -> List[Dict[str, Any]]:
    """Plan update synchronization operations."""
    operations = []
    
    for item in source.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source)
            dest_file = destination / rel_path
            
            should_copy = False
            if not dest_file.exists():
                should_copy = True
            else:
                # Compare modification times
                source_mtime = item.stat().st_mtime
                dest_mtime = dest_file.stat().st_mtime
                if source_mtime > dest_mtime:
                    should_copy = True
            
            if should_copy:
                operations.append({
                    "action": "copy",
                    "source": str(item),
                    "destination": str(dest_file),
                    "relative_path": str(rel_path)
                })
    
    return operations


async def _plan_mirror_sync(source: Path, destination: Path) -> List[Dict[str, Any]]:
    """Plan mirror synchronization operations."""
    operations = []
    
    # First, copy/update files from source
    update_ops = await _plan_update_sync(source, destination)
    operations.extend(update_ops)
    
    # Then, remove files in destination that don't exist in source
    if destination.exists():
        for item in destination.rglob("*"):
            if item.is_file():
                rel_path = item.relative_to(destination)
                source_file = source / rel_path
                
                if not source_file.exists():
                    operations.append({
                        "action": "delete",
                        "path": str(item),
                        "relative_path": str(rel_path)
                    })
    
    return operations


async def _plan_merge_sync(source: Path, destination: Path) -> List[Dict[str, Any]]:
    """Plan merge synchronization operations."""
    operations = []
    
    # Copy newer files from source to destination
    for item in source.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(source)
            dest_file = destination / rel_path
            
            should_copy = False
            if not dest_file.exists():
                should_copy = True
            else:
                # Compare modification times and copy newer
                source_mtime = item.stat().st_mtime
                dest_mtime = dest_file.stat().st_mtime
                if source_mtime > dest_mtime:
                    should_copy = True
            
            if should_copy:
                operations.append({
                    "action": "copy",
                    "source": str(item),
                    "destination": str(dest_file),
                    "relative_path": str(rel_path)
                })
    
    return operations


def _guess_mime_type(file_path: Path) -> str:
    """Guess MIME type based on file extension."""
    import mimetypes
    mime_type, _ = mimetypes.guess_type(str(file_path))
    return mime_type or 'application/octet-stream'


# Setup logging
logger = logging.getLogger(__name__) 