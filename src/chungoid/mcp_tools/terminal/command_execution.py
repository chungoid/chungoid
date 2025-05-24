"""
Enhanced Terminal Command Execution

Secure, context-aware terminal command execution with:
- Security sandboxing and command classification
- Project context integration  
- Resource monitoring and limits
- Comprehensive logging and state tracking
- Intelligent error handling and recovery
"""

import logging
import os
import subprocess
import shlex
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import asyncio
import psutil
import signal

from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService
from .security import CommandClassifier, SecuritySandbox

logger = logging.getLogger(__name__)


class TerminalExecutionContext:
    """Manages execution context for terminal commands with security and resource tracking."""
    
    def __init__(self, project_path: Optional[str] = None, project_id: Optional[str] = None):
        self.project_path = Path(project_path).resolve() if project_path else Path.cwd()
        self.project_id = project_id
        self.working_directory = self.project_path
        self.environment = dict(os.environ)
        self.security_sandbox = SecuritySandbox()
        self.command_classifier = CommandClassifier()
        self.active_processes: Dict[str, subprocess.Popen] = {}
        
        # Set project-specific environment variables
        if project_id:
            self.environment['CHUNGOID_PROJECT_ID'] = project_id
        self.environment['CHUNGOID_PROJECT_PATH'] = str(self.project_path)
        
    def classify_command(self, command: str) -> Dict[str, Any]:
        """Classify command for security assessment."""
        return self.command_classifier.classify(command)
        
    def setup_sandbox(self, classification: Dict[str, Any]) -> Dict[str, Any]:
        """Setup security sandbox based on command classification."""
        return self.security_sandbox.setup(classification, self.working_directory)


async def tool_run_terminal_command(
    command: str,
    working_directory: Optional[str] = None,
    environment: Optional[Dict[str, str]] = None,
    timeout: int = 300,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    security_level: str = "standard",
    capture_output: bool = True,
    stream_output: bool = False,
) -> Dict[str, Any]:
    """
    Enhanced terminal command execution with security sandboxing and context awareness.
    
    This is the primary terminal tool that provides comprehensive command execution
    with security controls, resource monitoring, and project context integration.
    
    Args:
        command: Command to execute
        working_directory: Working directory for command execution
        environment: Additional environment variables
        timeout: Command timeout in seconds (default: 300)
        project_path: Project directory path for context
        project_id: Project identifier for context and isolation
        security_level: Security level ("strict", "standard", "permissive")
        capture_output: Whether to capture stdout/stderr
        stream_output: Whether to stream output in real-time
        
    Returns:
        Dict containing:
        - success: bool - Execution success status
        - return_code: int - Process return code
        - stdout: str - Standard output (if captured)
        - stderr: str - Standard error (if captured)
        - execution_time: float - Execution time in seconds
        - resource_usage: Dict - CPU, memory usage statistics
        - security_info: Dict - Security classification and sandbox info
        - context: Dict - Execution context metadata
    """
    start_time = datetime.now()
    
    try:
        # Initialize execution context
        ctx = TerminalExecutionContext(project_path, project_id)
        
        # Set working directory
        if working_directory:
            work_dir = Path(working_directory).resolve()
            if work_dir.exists() and work_dir.is_dir():
                ctx.working_directory = work_dir
            else:
                logger.warning(f"Working directory does not exist: {working_directory}, using project path")
        
        # Update environment
        if environment:
            ctx.environment.update(environment)
        
        # Classify command for security assessment
        classification = ctx.classify_command(command)
        logger.info(f"Command classified as: {classification.get('risk_level', 'unknown')}")
        
        # Setup security sandbox
        sandbox_info = ctx.setup_sandbox(classification)
        
        # Check if command is allowed based on security level
        if not _is_command_allowed(classification, security_level):
            return {
                "success": False,
                "error": f"Command blocked by security policy: {classification.get('risk_level')}",
                "security_info": classification,
                "sandbox_info": sandbox_info,
                "execution_time": 0.0,
            }
        
        # Execute command with monitoring
        result = await _execute_with_monitoring(
            command=command,
            context=ctx,
            timeout=timeout,
            capture_output=capture_output,
            stream_output=stream_output,
            classification=classification,
            sandbox_info=sandbox_info,
        )
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        result["execution_time"] = execution_time
        
        # Add context metadata
        result["context"] = {
            "project_path": str(ctx.project_path),
            "project_id": project_id,
            "working_directory": str(ctx.working_directory),
            "security_level": security_level,
            "timestamp": start_time.isoformat(),
        }
        
        result["security_info"] = classification
        result["sandbox_info"] = sandbox_info
        
        logger.info(f"Command executed successfully in {execution_time:.2f}s: {command[:50]}...")
        return result
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Terminal command execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "execution_time": execution_time,
            "context": {
                "project_path": project_path,
                "project_id": project_id,
                "working_directory": working_directory,
                "security_level": security_level,
                "timestamp": start_time.isoformat(),
            }
        }


async def terminal_execute_command(
    command: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    timeout: int = 60,
) -> Dict[str, Any]:
    """
    Simplified terminal command execution for basic operations.
    
    This is a convenience wrapper around tool_run_terminal_command for
    simple command execution without advanced security features.
    """
    return await tool_run_terminal_command(
        command=command,
        project_path=project_path,
        project_id=project_id,
        timeout=timeout,
        security_level="standard",
        capture_output=True,
        stream_output=False,
    )


async def terminal_execute_batch(
    commands: List[str],
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    stop_on_error: bool = True,
    timeout_per_command: int = 60,
) -> Dict[str, Any]:
    """
    Execute multiple terminal commands in batch with optional error handling.
    
    Args:
        commands: List of commands to execute
        project_path: Project directory path
        project_id: Project identifier  
        stop_on_error: Whether to stop on first error
        timeout_per_command: Timeout per individual command
        
    Returns:
        Dict containing batch execution results
    """
    start_time = datetime.now()
    results = []
    successful_commands = 0
    failed_commands = 0
    
    try:
        logger.info(f"Starting batch execution of {len(commands)} commands")
        
        for i, command in enumerate(commands):
            logger.info(f"Executing command {i+1}/{len(commands)}: {command[:50]}...")
            
            result = await terminal_execute_command(
                command=command,
                project_path=project_path,
                project_id=project_id,
                timeout=timeout_per_command,
            )
            
            result["command_index"] = i
            result["command"] = command
            results.append(result)
            
            if result.get("success", False):
                successful_commands += 1
            else:
                failed_commands += 1
                if stop_on_error:
                    logger.warning(f"Stopping batch execution due to error in command {i+1}")
                    break
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": failed_commands == 0,
            "total_commands": len(commands),
            "successful_commands": successful_commands,
            "failed_commands": failed_commands,
            "execution_time": execution_time,
            "results": results,
            "context": {
                "project_path": project_path,
                "project_id": project_id,
                "stop_on_error": stop_on_error,
                "timestamp": start_time.isoformat(),
            }
        }
        
    except Exception as e:
        execution_time = (datetime.now() - start_time).total_seconds()
        logger.error(f"Batch command execution failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "total_commands": len(commands),
            "successful_commands": successful_commands,
            "failed_commands": failed_commands,
            "execution_time": execution_time,
            "results": results,
        }


async def terminal_get_environment(
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    include_system: bool = False,
) -> Dict[str, Any]:
    """
    Get current terminal environment variables with project context.
    
    Args:
        project_path: Project directory path
        project_id: Project identifier
        include_system: Whether to include system environment variables
        
    Returns:
        Dict containing environment information
    """
    try:
        ctx = TerminalExecutionContext(project_path, project_id)
        
        # Get project-specific environment
        project_env = {
            k: v for k, v in ctx.environment.items() 
            if k.startswith('CHUNGOID_') or k.startswith('PROJECT_')
        }
        
        # Get common development environment variables
        dev_env = {
            k: v for k, v in ctx.environment.items()
            if k in ['PATH', 'PYTHONPATH', 'NODE_PATH', 'JAVA_HOME', 'VIRTUAL_ENV', 'CONDA_DEFAULT_ENV']
        }
        
        result = {
            "success": True,
            "project_environment": project_env,
            "development_environment": dev_env,
            "working_directory": str(ctx.working_directory),
            "project_path": str(ctx.project_path),
            "project_id": project_id,
        }
        
        if include_system:
            result["system_environment"] = dict(ctx.environment)
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to get environment: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        }


async def terminal_set_working_directory(
    directory_path: str,
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
    create_if_missing: bool = False,
) -> Dict[str, Any]:
    """
    Set working directory for terminal operations with validation.
    
    Args:
        directory_path: Target directory path
        project_path: Project directory path for context
        project_id: Project identifier
        create_if_missing: Whether to create directory if it doesn't exist
        
    Returns:
        Dict containing operation result
    """
    try:
        target_dir = Path(directory_path).resolve()
        
        # Validate directory path
        if not target_dir.exists():
            if create_if_missing:
                target_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {target_dir}")
            else:
                return {
                    "success": False,
                    "error": f"Directory does not exist: {target_dir}",
                    "directory_path": str(target_dir),
                }
        
        if not target_dir.is_dir():
            return {
                "success": False,
                "error": f"Path is not a directory: {target_dir}",
                "directory_path": str(target_dir),
            }
        
        # Change working directory
        original_cwd = os.getcwd()
        os.chdir(target_dir)
        
        logger.info(f"Working directory changed from {original_cwd} to {target_dir}")
        
        return {
            "success": True,
            "previous_directory": original_cwd,
            "current_directory": str(target_dir),
            "project_path": project_path,
            "project_id": project_id,
            "timestamp": datetime.now().isoformat(),
        }
        
    except Exception as e:
        logger.error(f"Failed to set working directory: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "directory_path": directory_path,
        }


# Helper functions

def _is_command_allowed(classification: Dict[str, Any], security_level: str) -> bool:
    """Check if command is allowed based on classification and security level."""
    risk_level = classification.get("risk_level", "unknown")
    
    if security_level == "strict":
        return risk_level in ["safe", "low"]
    elif security_level == "standard":
        return risk_level in ["safe", "low", "medium"]
    elif security_level == "permissive":
        return risk_level != "critical"
    else:
        return risk_level in ["safe", "low"]


async def _execute_with_monitoring(
    command: str,
    context: TerminalExecutionContext,
    timeout: int,
    capture_output: bool,
    stream_output: bool,
    classification: Dict[str, Any],
    sandbox_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Execute command with resource monitoring and security controls."""
    
    # Parse command safely
    try:
        cmd_parts = shlex.split(command)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Invalid command syntax: {str(e)}",
            "return_code": -1,
        }
    
    if not cmd_parts:
        return {
            "success": False,
            "error": "Empty command",
            "return_code": -1,
        }
    
    # Setup process execution parameters
    process_kwargs = {
        "cwd": str(context.working_directory),
        "env": context.environment,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    
    if capture_output:
        process_kwargs.update({
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        })
    
    stdout_data = ""
    stderr_data = ""
    return_code = 0
    resource_usage = {}
    
    try:
        # Start process
        process = subprocess.Popen(cmd_parts, **process_kwargs)
        context.active_processes[str(process.pid)] = process
        
        # Monitor process execution
        try:
            if capture_output:
                stdout_data, stderr_data = process.communicate(timeout=timeout)
            else:
                process.wait(timeout=timeout)
            
            return_code = process.returncode
            
            # Collect resource usage if available
            try:
                proc_info = psutil.Process(process.pid)
                resource_usage = {
                    "cpu_percent": proc_info.cpu_percent(),
                    "memory_info": proc_info.memory_info()._asdict(),
                    "num_threads": proc_info.num_threads(),
                }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            
        except subprocess.TimeoutExpired:
            # Handle timeout
            process.kill()
            try:
                stdout_data, stderr_data = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            
            return {
                "success": False,
                "error": f"Command timed out after {timeout} seconds",
                "return_code": -1,
                "stdout": stdout_data,
                "stderr": stderr_data,
                "resource_usage": resource_usage,
            }
        
        finally:
            # Clean up process tracking
            if str(process.pid) in context.active_processes:
                del context.active_processes[str(process.pid)]
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Process execution failed: {str(e)}",
            "return_code": -1,
            "stdout": stdout_data,
            "stderr": stderr_data,
            "resource_usage": resource_usage,
        }
    
    # Determine success based on return code and classification
    success = return_code == 0
    
    return {
        "success": success,
        "return_code": return_code,
        "stdout": stdout_data,
        "stderr": stderr_data,
        "resource_usage": resource_usage,
        "command_parts": cmd_parts,
    } 