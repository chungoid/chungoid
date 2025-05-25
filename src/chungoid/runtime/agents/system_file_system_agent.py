"""
Agent for performing file system operations.
"""
import asyncio
import os
import shutil
from pathlib import Path
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, Dict, List, Optional, Union, ClassVar, Tuple
import logging
import traceback # For detailed error logging

from pydantic import BaseModel, Field

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
from chungoid.runtime.agents.agent_base import BaseAgent # MODIFIED: Changed AgentBase to BaseAgent
from chungoid.utils.agent_registry import AgentCard, AgentToolSpec # MODIFIED: Changed import path
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
# from chungoid.utils.security import is_safe_path # For path safety checks # REMOVED
from chungoid.schemas.errors import AgentErrorDetails # For structured errors
from chungoid.schemas.orchestration import SharedContext # ADDED
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.registry import register_system_agent

logger = logging.getLogger(__name__)

# Registry-first architecture import
@register_system_agent(capabilities=["file_management", "directory_operations", "file_operations"])
class SystemFileSystemAgent_v1(BaseAgent):
    """
    An agent that provides tools for interacting with the file system.
    """

    AGENT_ID: ClassVar[str] = "SystemFileSystemAgent_v1"
    AGENT_NAME: ClassVar[str] = "System File System Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Performs file system operations like creating/modifying directories and files."
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.FILE_MANAGEMENT
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    # --- Pydantic Schemas for Agent Tools ---

    class _BaseFileSystemInput(BaseModel):
        """Base model for inputs that take a single path."""
        path: str = Field(description="Path relative to the project root.")

    class CreateDirectoryInput(_BaseFileSystemInput):
        """Input for creating a directory."""
        pass # No additional fields needed beyond path

    class CreateFileInput(_BaseFileSystemInput):
        """Input for creating a file."""
        content: str = Field(default="", description="Initial content for the file.")
        overwrite: bool = Field(default=False, description="Whether to overwrite the file if it already exists.")

    class WriteToFileInput(_BaseFileSystemInput):
        """Input for writing content to a file."""
        content: str = Field(description="Content to write to the file.")
        append: bool = Field(default=False, description="Whether to append to the file or overwrite it.")

    class ReadFileInput(_BaseFileSystemInput):
        """Input for reading a file's content."""
        pass

    class DeletePathInput(_BaseFileSystemInput): # Generic for file or directory
        """Input for deleting a file or directory."""
        recursive: bool = Field(default=False, description="Required for deleting non-empty directories. Ignored for files.")

    class PathExistsInput(_BaseFileSystemInput):
        """Input for checking if a path exists."""
        pass
        
    class ListDirectoryInput(_BaseFileSystemInput):
        """Input for listing directory contents."""
        pass

    class MoveCopyPathInput(BaseModel):
        """Input for moving or copying a path."""
        src_path: str = Field(description="Source path relative to project root.")
        dest_path: str = Field(description="Destination path relative to project root.")
        overwrite: bool = Field(default=False, description="Whether to overwrite the destination if it exists.")

    class FileSystemOutput(BaseModel):
        """Standard output for file system operations."""
        success: bool
        path: Optional[str] = Field(default=None, description="The absolute path affected by the operation, if applicable.")
        message: Optional[str] = Field(default=None, description="A message describing the outcome.")
        error: Optional[str] = Field(default=None, description="Error message if the operation failed.")
        content: Optional[Union[str, List[str]]] = Field(default=None, description="Content read from a file or list of directory items.")
        exists: Optional[bool] = Field(default=None, description="Result of a path_exists check.")

    _logger: Any # Declare _logger as an instance variable, not a Pydantic field

    def __init__(self, **data: Any):
        super().__init__(**data)
        # BaseAgent's __init__ will store 'system_context' in self.system_context if passed via **data
        # Initialize _logger from self.system_context
        current_logger = None
        if self.system_context and "logger" in self.system_context:
            current_logger = self.system_context.get("logger")
        
        self._logger = current_logger if current_logger else logger

        self._logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) v{self.AGENT_VERSION} initialized.")

    def _resolve_and_sandbox_path(self, relative_path_arg: Union[str, Path], project_root_override_from_invoke_async: Optional[Union[str,Path]] = None) -> Tuple[Path, Optional[str]]: # RENAMED project_root_override to be specific
        # Determine the effective project root
        base_path_arg: Union[str, Path, None]
        if project_root_override_from_invoke_async is not None: # MODIFIED: Parameter name change
            base_path_arg = project_root_override_from_invoke_async # MODIFIED: Parameter name change
            self._logger.info(f"RESOLVE_PATH_DEBUG: Using project_root_override_from_invoke_async: {base_path_arg} (type: {type(base_path_arg)})")
        elif hasattr(self, 'project_root') and self.project_root: # This might be from BaseAgent if set
            base_path_arg = self.project_root
            self._logger.info(f"RESOLVE_PATH_DEBUG: Using self.project_root: {base_path_arg} (type: {type(base_path_arg)})")
        else: # Fallback: Try to get from system_context if available (should be less common)
            base_path_arg = self.system_context.get("project_root_path") if self.system_context else None
            if base_path_arg:
                self._logger.info(f"RESOLVE_PATH_DEBUG: Using project_root_path from system_context: {base_path_arg} (type: {type(base_path_arg)})")
            else:
                self._logger.error("RESOLVE_PATH_DEBUG: effective_project_root could not be determined (None or empty).")
                return Path(), "Project root not defined for file operation."

        # Ensure base_path_arg (effective_project_root) is a Path object
        effective_project_root: Path
        if isinstance(base_path_arg, str):
            self._logger.info(f"RESOLVE_PATH_DEBUG: base_path_arg (effective_project_root) was string '{base_path_arg}'. Converting to Path.")
            try:
                effective_project_root = Path(base_path_arg)
            except Exception as e_conv_base:
                self._logger.error(f"RESOLVE_PATH_DEBUG: Error converting base_path_arg '{base_path_arg}' to Path: {e_conv_base}")
                return Path(), f"Project root '{base_path_arg}' could not be converted to a Path object."
        elif isinstance(base_path_arg, Path):
            effective_project_root = base_path_arg
        else: # Should not happen if logic above is correct, but as a safeguard
            self._logger.error(f"RESOLVE_PATH_DEBUG: base_path_arg (effective_project_root) is of unexpected type {type(base_path_arg)}. Value: '{base_path_arg}'")
            return Path(), f"Project root is of an invalid type: {type(base_path_arg)}."
        
        self._logger.info(f"RESOLVE_PATH_DEBUG: effective_project_root FINAL type {type(effective_project_root)}, value '{effective_project_root}'.")

        # Ensure relative_path_arg is also Path for the operation
        path_to_join: Path
        if isinstance(relative_path_arg, str):
            self._logger.info(f"RESOLVE_PATH_DEBUG: relative_path_arg was string '{relative_path_arg}'. Converting to Path.")
            try:
                path_to_join = Path(relative_path_arg)
            except Exception as e_conv_rel:
                self._logger.error(f"RESOLVE_PATH_DEBUG: Error converting relative_path_arg '{relative_path_arg}' to Path: {e_conv_rel}")
                return Path(), f"Relative path '{relative_path_arg}' could not be converted to a Path object."
        elif isinstance(relative_path_arg, Path):
            path_to_join = relative_path_arg
        else:
            self._logger.error(f"RESOLVE_PATH_DEBUG: Invalid type for relative_path_arg: {type(relative_path_arg)}")
            return Path(), f"Invalid type for relative_path: {type(relative_path_arg)}"
        
        self._logger.info(f"RESOLVE_PATH_DEBUG: path_to_join FINAL type {type(path_to_join)}, value '{path_to_join}'.")
        
        # Check for path traversal attempts in relative_path
        if ".." in path_to_join.parts:
            self._logger.error(f"RESOLVE_PATH_DEBUG: path_to_join '{path_to_join}' contains '..', which is not allowed for a relative_path.")
            return Path(), f"Relative path '{relative_path_arg}' cannot contain '..'."

        if path_to_join.is_absolute():
            self._logger.error(f"RESOLVE_PATH_DEBUG: path_to_join '{path_to_join}' is absolute, which is not allowed for a relative_path.")
            return Path(), f"Relative path '{relative_path_arg}' cannot be absolute."

        self._logger.info(f"RESOLVE_PATH_DEBUG: PRE-JOIN: effective_project_root='{effective_project_root}' (type: {type(effective_project_root)}), path_to_join='{path_to_join}' (type: {type(path_to_join)})")

        # At this point, effective_project_root and path_to_join are guaranteed to be Path objects by the logic above.
        absolute_path = (effective_project_root / path_to_join).resolve()
        self._logger.info(f"RESOLVE_PATH_DEBUG: absolute_path AFTER JOIN & RESOLVE: '{absolute_path}' (type: {type(absolute_path)})")

        # Security check: Ensure the resolved path is within the project root
        # Use effective_project_root.resolve() for comparison
        resolved_root_for_check = effective_project_root.resolve()
        if resolved_root_for_check not in absolute_path.parents and absolute_path != resolved_root_for_check:
            self._logger.error(f"RESOLVE_PATH_DEBUG: absolute_path '{absolute_path}' is outside the project root '{resolved_root_for_check}'.")
            return Path(), f"Path '{absolute_path}' is outside the project root '{resolved_root_for_check}'."

        return absolute_path, None

    # ADDED: Protocol-aware execution method (hybrid approach)
    async def execute_with_protocols(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using appropriate protocol with fallback to traditional method.
        Follows AI agent best practices for hybrid execution.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                self._logger.warning("Protocol execution failed, falling back to traditional method")
                raise ProtocolExecutionError("Pure protocol execution failed")
                
        except Exception as e:
            self._logger.warning(f"Protocol execution error: {e}, falling back to traditional method")
            raise ProtocolExecutionError("Pure protocol execution failed")

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }

    async def invoke_async(
        self,
        tool_name: Optional[str] = None,
        tool_input: Optional[Dict[str, Any]] = None,
        project_root: Optional[Path] = None,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        full_context: Optional[SharedContext] = None,
        inputs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke a file system operation.
        Adapts to different invocation patterns from the orchestrator.
        """
        actual_tool_name: Optional[str] = tool_name
        actual_tool_input: Optional[Dict[str, Any]] = tool_input
        actual_project_root: Optional[Path] = project_root

        if inputs is not None:
            if not actual_tool_name and "tool_name" in inputs:
                actual_tool_name = inputs["tool_name"]
            if actual_tool_input is None and "tool_input" in inputs:
                actual_tool_input = inputs["tool_input"]

        # MODIFIED: More robust retrieval of project_root from full_context.data
        if not actual_project_root and full_context and hasattr(full_context, 'data') and isinstance(full_context.data, dict):
            project_root_from_context_str = full_context.data.get('mcp_root_workspace_path') or full_context.data.get('project_root_path')
            if project_root_from_context_str:
                try:
                    actual_project_root = Path(project_root_from_context_str)
                    self._logger.debug(f"Retrieved project_root from full_context.data ('mcp_root_workspace_path' or 'project_root_path'): {actual_project_root}")
                except (TypeError, ValueError) as e: # Added ValueError for robustness
                    self._logger.warning(f"Could not convert project_root ('{project_root_from_context_str}') from full_context.data to Path: {e}")
            else:
                self._logger.debug("Neither 'mcp_root_workspace_path' nor 'project_root_path' found in full_context.data.")
        elif not actual_project_root:
             self._logger.debug("actual_project_root not set and full_context or full_context.data is not available/suitable for project_root retrieval.")

        if not actual_tool_name:
            return {"success": False, "error": f"'tool_name' is required but was not provided directly or in 'inputs' for {self.AGENT_NAME}."}
        if actual_tool_input is None:
            actual_tool_input = {}
            self._logger.debug(f"tool_input was None for tool '{actual_tool_name}', defaulting to {{}}.")
        elif not isinstance(actual_tool_input, dict):
            return {"success": False, "error": f"'tool_input' must be a dictionary for {self.AGENT_NAME}, got {type(actual_tool_input)}."}

        if not actual_project_root:
            return {"success": False, "error": f"'project_root' is required but was not provided and could not be retrieved from context for {self.AGENT_NAME}."}

        if not hasattr(self, actual_tool_name):
            return {"success": False, "error": f"Tool '{actual_tool_name}' not found on {self.AGENT_NAME}."}

        method = getattr(self, actual_tool_name)
        
        try:
            # If method is an async function, await it directly.
            # The method itself should handle blocking I/O appropriately (e.g., using to_thread internally if needed for actual file ops)
            if asyncio.iscoroutinefunction(method):
                if "project_root" in method.__code__.co_varnames:
                    # Assuming tool methods like create_directory are defined as async
                    # and might take project_root
                    result = await method(**actual_tool_input, project_root=actual_project_root)
                else:
                    result = await method(**actual_tool_input)
            else: # If it's a synchronous method (should ideally not be the case for tool methods)
                self._logger.warning(f"Tool method {actual_tool_name} is synchronous. Running in thread.")
                if "project_root" in method.__code__.co_varnames:
                    result = await asyncio.to_thread(method, **actual_tool_input, project_root=actual_project_root)
                else:
                    result = await asyncio.to_thread(method, **actual_tool_input)
            
            # Ensure the result is a dictionary, as expected by the type hint and orchestrator
            if isinstance(result, BaseModel):
                return result.model_dump() # Standard Pydantic model output
            elif isinstance(result, dict):
                return result # Already a dict
            else:
                # This case should ideally not be reached if tool methods return FileSystemOutput or a dict.
                self._logger.warning(f"Tool method {actual_tool_name} returned non-dict/BaseModel type: {type(result)}. Value: {str(result)[:200]}. Wrapping in a generic error dict.")
                return {"success": False, "error": f"Tool {actual_tool_name} returned unexpected type: {type(result)}", "details": str(result)[:500]}

        except Exception as e:
            self._logger.error(f"Error executing tool {actual_tool_name} for agent {self.AGENT_ID}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "details": f"Exception type: {type(e).__name__}"}

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """
        Get the AgentCard for this agent.
        """
        tools = [
            AgentToolSpec(
                name="create_directory",
                description="Creates a directory (and any parent directories if they don't exist). Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.CreateDirectoryInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="create_file",
                description="Creates a file with optional content. Fails if overwrite is False and file exists. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.CreateFileInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="write_to_file",
                description="Writes content to a file, with an option to append or overwrite. Creates the file if it doesn't exist. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.WriteToFileInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="read_file",
                description="Reads the content of a file. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.ReadFileInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="delete_file",
                description="Deletes a file. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.DeletePathInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="delete_directory",
                description="Deletes a directory. If recursive is False, the directory must be empty. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.DeletePathInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="path_exists",
                description="Checks if a path (file or directory) exists. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.PathExistsInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="list_directory_contents",
                description="Lists the contents (files and subdirectories) of a directory. Input path is relative to project root.",
                input_schema=SystemFileSystemAgent_v1.ListDirectoryInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="move_path",
                description="Moves a file or directory. Overwrites destination if overwrite is True. Input paths are relative to project root.",
                input_schema=SystemFileSystemAgent_v1.MoveCopyPathInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="copy_path",
                description="Copies a file or directory. Overwrites destination if overwrite is True. Input paths are relative to project root.",
                input_schema=SystemFileSystemAgent_v1.MoveCopyPathInput.model_json_schema(),
                output_schema=SystemFileSystemAgent_v1.FileSystemOutput.model_json_schema(),
            ),
        ]
        return AgentCard(
            agent_id=SystemFileSystemAgent_v1.AGENT_ID,
            name=SystemFileSystemAgent_v1.AGENT_NAME,
            description=SystemFileSystemAgent_v1.AGENT_DESCRIPTION,
            version=SystemFileSystemAgent_v1.AGENT_VERSION,
            tools=tools,
            categories=[SystemFileSystemAgent_v1.CATEGORY.value, AgentCategory.AUTONOMOUS_PROJECT_ENGINE.value],
            visibility=SystemFileSystemAgent_v1.VISIBILITY.value,
            metadata={
                "author": "Chungoid Systems",
                "tags": ["filesystem", "file management", "io"],
                "callable_fn_path": f"{SystemFileSystemAgent_v1.__module__}.{SystemFileSystemAgent_v1.__name__}"
            }
        )

    # --------------------------------------------------------------------------
    # File System Operation Methods
    # --------------------------------------------------------------------------

    async def create_directory(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Creates a directory (and any parent directories if they don't exist)."""
        self._logger.info(f"CREATE_DIRECTORY_DEBUG: Received project_root type: {type(project_root)}, value: {project_root}")
        if not project_root:
            return self.FileSystemOutput(success=False, error="Project root not provided.").model_dump()

        try:
            absolute_path, error = self._resolve_and_sandbox_path(path, project_root)
            if error:
                return self.FileSystemOutput(success=False, path=path, error=error).model_dump()

            self._logger.info(f"CREATE_DIR_PRE_MKDIR: Type of absolute_path: {type(absolute_path)}, Value: {absolute_path}")

            if absolute_path.exists() and not absolute_path.is_dir():
                return self.FileSystemOutput(
                    success=False, 
                    path=str(absolute_path), 
                    error=f"Path exists but is not a directory."
                ).model_dump()

            absolute_path.mkdir(parents=True, exist_ok=True)
            self._logger.info(f"Directory created: {absolute_path}")
            return self.FileSystemOutput(
                success=True, 
                path=str(absolute_path), 
                message="Directory created or already exists."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(
                success=False, 
                error=f"Failed to create directory: {str(e)}", 
                path=path
            ).model_dump()

    async def create_file(self, path: str, project_root: Path, content: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """Creates a file with optional content. Fails if overwrite is False and file exists."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()

            if sandboxed_path.is_dir():
                return self.FileSystemOutput(
                    success=False,
                    path=str(sandboxed_path),
                    error="Path exists and is a directory, not a file."
                ).model_dump()

            if sandboxed_path.exists() and not overwrite:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error="File exists and overwrite is set to False."
                ).model_dump()
            
            # Ensure parent directory exists
            parent_dir = sandboxed_path.parent
            if not parent_dir.exists():
                os.makedirs(parent_dir, exist_ok=True)
            elif not parent_dir.is_dir():
                return self.FileSystemOutput(
                    success=False,
                    path=str(parent_dir),
                    error=f"Cannot create file. Parent path '{parent_dir}' exists but is not a directory."
                ).model_dump()

            with open(sandboxed_path, 'w' if overwrite else 'x') as f:
                f.write(content)
            
            return self.FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message="File created successfully."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except FileExistsError: # Specifically for 'x' mode when file exists and overwrite is False
             return self.FileSystemOutput(
                success=False, 
                path=path, # Use original path in error message
                error="File exists and overwrite is set to False (FileExistsError)."
            ).model_dump()
        except Exception as e:
            return self.FileSystemOutput(
                success=False, 
                error=f"Failed to create file: {str(e)}", 
                path=path
            ).model_dump()

    async def write_to_file(self, path: str, project_root: Path, content: str, append: bool = False) -> Dict[str, Any]:
        """Writes content to a file, with an option to append or overwrite.

        If the file does not exist, it will be created.
        If the file exists and append is False, it will be overwritten.
        If the file exists and append is True, content will be added to the end.
        """
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()

            if sandboxed_path.is_dir():
                return self.FileSystemOutput(
                    success=False,
                    path=str(sandboxed_path),
                    error="Path exists and is a directory, cannot write content to it."
                ).model_dump()

            # Ensure parent directory exists
            parent_dir = sandboxed_path.parent
            if not parent_dir.exists():
                os.makedirs(parent_dir, exist_ok=True)
            elif not parent_dir.is_dir():
                return self.FileSystemOutput(
                    success=False,
                    path=str(parent_dir),
                    error=f"Cannot write to file. Parent path '{parent_dir}' exists but is not a directory."
                ).model_dump()

            mode = 'a' if append else 'w'
            with open(sandboxed_path, mode) as f:
                f.write(content)
            
            action = "appended to" if append else "written to"
            return self.FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message=f"Content successfully {action} file."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(
                success=False, 
                error=f"Failed to write to file: {str(e)}", 
                path=path
            ).model_dump()

    async def read_file(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Reads the content of a file."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()
            
            if sandboxed_path.is_dir():
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error="Path is a directory, not a file."
                ).model_dump()

            with open(sandboxed_path, 'r') as f:
                content = f.read()
            
            return self.FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message="File read successfully.",
                content=content
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(
                success=False, 
                error=f"Failed to read file: {str(e)}", 
                path=path
            ).model_dump()

    async def delete_file(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Deletes a file."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()

            if not sandboxed_path.exists():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="File not found.").model_dump()
            if sandboxed_path.is_dir():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is a directory, not a file. Use delete_directory.").model_dump()

            os.remove(sandboxed_path)
            return self.FileSystemOutput(success=True, path=str(sandboxed_path), message="File deleted successfully.").model_dump()
        except ValueError as ve:
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(success=False, error=f"Failed to delete file: {str(e)}", path=path).model_dump()

    async def delete_directory(self, path: str, project_root: Path, recursive: bool = False) -> Dict[str, Any]:
        """Deletes a directory. If recursive is False, the directory must be empty."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()

            if not sandboxed_path.exists():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory not found.").model_dump()
            if not sandboxed_path.is_dir():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is a file, not a directory. Use delete_file.").model_dump()

            if recursive:
                shutil.rmtree(sandboxed_path)
                message = "Directory and its contents deleted successfully."
            else:
                if any(sandboxed_path.iterdir()): # Check if directory is not empty
                    return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory is not empty and recursive is False.").model_dump()
                os.rmdir(sandboxed_path)
                message = "Empty directory deleted successfully."
            
            return self.FileSystemOutput(success=True, path=str(sandboxed_path), message=message).model_dump()
        except ValueError as ve:
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(success=False, error=f"Failed to delete directory: {str(e)}", path=path).model_dump()

    async def path_exists(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Checks if a path (file or directory) exists."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)
            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()
            exists = sandboxed_path.exists()
            return self.FileSystemOutput(success=True, path=str(sandboxed_path), exists=exists, message=f"Path checked.").model_dump()
        except ValueError as ve: # Path traversal or outside project
            return self.FileSystemOutput(success=False, error=str(ve), path=path, exists=False).model_dump()
        except Exception as e: # Other unexpected errors
            return self.FileSystemOutput(success=False, error=f"Error checking path existence: {str(e)}", path=path, exists=False).model_dump()

    async def list_directory_contents(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Lists the contents (files and subdirectories) of a directory."""
        try:
            sandboxed_path, error = self._resolve_and_sandbox_path(path, project_root)

            if error:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=error
                ).model_dump()

            if not sandboxed_path.exists():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory not found.").model_dump()
            if not sandboxed_path.is_dir():
                return self.FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is not a directory.").model_dump()

            contents = [item.name for item in os.listdir(sandboxed_path)]
            return self.FileSystemOutput(success=True, path=str(sandboxed_path), content=contents, message="Directory contents listed successfully.").model_dump()
        except ValueError as ve:
            return self.FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return self.FileSystemOutput(success=False, error=f"Failed to list directory contents: {str(e)}", path=path).model_dump()

    async def move_path(self, src_path: str, dest_path: str, project_root: Path, overwrite: bool = False) -> Dict[str, Any]:
        """Moves a file or directory. Overwrites destination if overwrite is True."""
        try:
            sandboxed_src, error_src = self._resolve_and_sandbox_path(src_path, project_root)
            sandboxed_dest, error_dest = self._resolve_and_sandbox_path(dest_path, project_root)

            if error_src or error_dest:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_src) if error_src else str(sandboxed_dest), 
                    error=error_src or error_dest
                ).model_dump()

            if not sandboxed_src.exists():
                return self.FileSystemOutput(success=False, path=str(sandboxed_src), error="Source path not found.").model_dump()

            if sandboxed_dest.exists() and not overwrite:
                return self.FileSystemOutput(success=False, path=str(sandboxed_dest), error="Destination path exists and overwrite is False.").model_dump()
            
            if sandboxed_dest.exists() and overwrite:
                if sandboxed_dest.is_dir() and not sandboxed_src.is_dir(): # Cannot overwrite dir with file
                     return self.FileSystemOutput(success=False, path=str(sandboxed_dest), error="Cannot overwrite a directory with a file.").model_dump()
                elif sandboxed_dest.is_file() and sandboxed_src.is_dir(): # Cannot overwrite file with dir using simple move
                     return self.FileSystemOutput(success=False, path=str(sandboxed_dest), error="Cannot overwrite a file with a directory using simple move. Delete destination first or use a different destination.").model_dump()
                elif sandboxed_dest.is_dir():
                    shutil.rmtree(sandboxed_dest) # remove existing dir if src is also dir
                else:
                    os.remove(sandboxed_dest) # remove existing file

            shutil.move(str(sandboxed_src), str(sandboxed_dest))
            return self.FileSystemOutput(success=True, path=str(sandboxed_dest), message=f"Path moved successfully from '{sandboxed_src}' to '{sandboxed_dest}'.").model_dump()
        except ValueError as ve:
            return self.FileSystemOutput(success=False, error=str(ve)).model_dump()
        except Exception as e:
            return self.FileSystemOutput(success=False, error=f"Failed to move path: {str(e)}").model_dump()

    async def copy_path(self, src_path: str, dest_path: str, project_root: Path, overwrite: bool = False) -> Dict[str, Any]:
        """Copies a file or directory. Overwrites destination if overwrite is True."""
        try:
            sandboxed_src, error_src = self._resolve_and_sandbox_path(src_path, project_root)
            sandboxed_dest, error_dest = self._resolve_and_sandbox_path(dest_path, project_root)

            if error_src or error_dest:
                return self.FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_src) if error_src else str(sandboxed_dest), 
                    error=error_src or error_dest
                ).model_dump()

            if not sandboxed_src.exists():
                return self.FileSystemOutput(success=False, path=str(sandboxed_src), error="Source path not found.").model_dump()

            if sandboxed_dest.exists() and not overwrite:
                return self.FileSystemOutput(success=False, path=str(sandboxed_dest), error="Destination path exists and overwrite is False.").model_dump()

            if sandboxed_dest.exists() and overwrite:
                if sandboxed_dest.is_dir():
                    shutil.rmtree(sandboxed_dest)
                else:
                    os.remove(sandboxed_dest)
            
            if sandboxed_src.is_dir():
                shutil.copytree(sandboxed_src, sandboxed_dest)
            else: # It's a file
                shutil.copy2(sandboxed_src, sandboxed_dest) # copy2 preserves metadata
            
            return self.FileSystemOutput(success=True, path=str(sandboxed_dest), message=f"Path copied successfully from '{sandboxed_src}' to '{sandboxed_dest}'.").model_dump()
        except ValueError as ve:
            return self.FileSystemOutput(success=False, error=str(ve)).model_dump()
        except Exception as e:
            return self.FileSystemOutput(success=False, error=f"Failed to copy path: {str(e)}").model_dump()

# To be able to run this agent directly for testing (optional)
if __name__ == "__main__":
    async def main():
        agent = SystemFileSystemAgent_v1()
        
        # Example usage (will not work until methods and schemas are defined)
        # test_project_dir = Path("./temp_test_project_fs_agent")
        # test_project_dir.mkdir(exist_ok=True)
        #
        # print("Agent Card:", agent.get_agent_card_static().model_dump_json(indent=2))
        #
        # # Test create_directory
        # result_create_dir = await agent.invoke_async(
        #     tool_name="create_directory",
        #     tool_input=SystemFileSystemAgent_v1.CreateDirectoryInput(path="my_new_folder_from_invoke").model_dump(),
        #     project_root=test_project_dir
        # )
        # print("\nCreate Directory (via invoke_async) Result:", result_create_dir)

        # # Test sandbox violation
        # result_sandbox_violation = await agent.invoke_async(
        #    tool_name="create_directory",
        #    tool_input=SystemFileSystemAgent_v1.CreateDirectoryInput(path="../outside_project_folder").model_dump(),
        #    project_root=test_project_dir
        # )
        # print("\nCreate Directory (sandbox violation) Result:", result_sandbox_violation)


        # # Direct method call for easier debugging during development
        # direct_result = await agent.create_directory(path="my_new_folder_direct", project_root=test_project_dir)
        # print("\nCreate Directory (direct call) Result:", direct_result)
        #
        # shutil.rmtree(test_project_dir, ignore_errors=True)
        pass

    asyncio.run(main()) 