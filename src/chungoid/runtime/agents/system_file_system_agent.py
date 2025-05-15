"""
Agent for performing file system operations.
"""
import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from chungoid.runtime.agents.agent_base import AgentBase
from chungoid.schemas.agent_registry import AgentCard, AgentToolSpec
from chungoid.utils.async_utils import run_in_executor_wrapper


class FileSystemAgent(AgentBase):
    """
    An agent that provides tools for interacting with the file system.
    """

    AGENT_ID = "chungoid.system.file_system"
    AGENT_NAME = "File System Agent"
    AGENT_DESCRIPTION = "Performs file system operations like creating/modifying directories and files."
    AGENT_VERSION = "0.1.0"

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


    def _resolve_and_sandbox_path(self, relative_path: str, project_root: Path) -> Path:
        """
        Resolves a relative path against the project root and ensures it's within the project.
        Raises ValueError if path is outside project scope or invalid.
        """
        if ".." in Path(relative_path).parts:
            raise ValueError("Path traversal detected (contains '..'). Path must be relative and within project.")

        absolute_path = (project_root / relative_path).resolve()
        
        # Ensure the resolved path is still within the project_root directory
        if project_root.resolve() not in absolute_path.parents and absolute_path != project_root.resolve():
            raise ValueError(f"Path '{absolute_path}' is outside the project root '{project_root.resolve()}'.")
        return absolute_path

    async def invoke_async(
        self,
        tool_name: str,
        tool_input: Dict[str, Any],
        project_root: Path,
        run_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Invoke a file system operation.
        """
        # Resolve the actual file path against the project_root
        # This is a crucial security and sandboxing measure.
        # All paths provided in tool_input should be relative to the project_root.

        # Path sandboxing and resolution will be handled within each tool method
        # as they know which arguments are paths.

        if not hasattr(self, tool_name):
            return {"success": False, "error": f"Tool '{tool_name}' not found on {self.AGENT_NAME}."}

        method = getattr(self, tool_name)
        
        # Most file system operations are blocking, so run them in an executor
        try:
            # Pass project_root to methods that need it for sandboxing/validation
            if "project_root" in method.__code__.co_varnames: # Check if method expects project_root
                 result = await run_in_executor_wrapper(method, **tool_input, project_root=project_root)
            else:
                 result = await run_in_executor_wrapper(method, **tool_input)
            return result
        except Exception as e:
            # Log the exception details for debugging
            # self.logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return {"success": False, "error": str(e), "details": f"Exception type: {type(e).__name__}"}

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        """
        Get the AgentCard for this agent.
        """
        tools = [
            AgentToolSpec(
                name="create_directory",
                description="Creates a directory (and any parent directories if they don't exist). Input path is relative to project root.",
                input_schema=cls.CreateDirectoryInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="create_file",
                description="Creates a file with optional content. Fails if overwrite is False and file exists. Input path is relative to project root.",
                input_schema=cls.CreateFileInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="write_to_file",
                description="Writes content to a file, with an option to append or overwrite. Creates the file if it doesn't exist. Input path is relative to project root.",
                input_schema=cls.WriteToFileInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="read_file",
                description="Reads the content of a file. Input path is relative to project root.",
                input_schema=cls.ReadFileInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="delete_file",
                description="Deletes a file. Input path is relative to project root.",
                input_schema=cls.DeletePathInput.model_json_schema(), # Uses generic DeletePathInput
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="delete_directory",
                description="Deletes a directory. If recursive is False, the directory must be empty. Input path is relative to project root.",
                input_schema=cls.DeletePathInput.model_json_schema(), # Uses generic DeletePathInput
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="path_exists",
                description="Checks if a path (file or directory) exists. Input path is relative to project root.",
                input_schema=cls.PathExistsInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="list_directory_contents",
                description="Lists the contents (files and subdirectories) of a directory. Input path is relative to project root.",
                input_schema=cls.ListDirectoryInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="move_path",
                description="Moves a file or directory. Overwrites destination if overwrite is True. Input paths are relative to project root.",
                input_schema=cls.MoveCopyPathInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
            AgentToolSpec(
                name="copy_path",
                description="Copies a file or directory. Overwrites destination if overwrite is True. Input paths are relative to project root.",
                input_schema=cls.MoveCopyPathInput.model_json_schema(),
                output_schema=cls.FileSystemOutput.model_json_schema(),
            ),
        ]
        return AgentCard(
            id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            description=cls.AGENT_DESCRIPTION,
            version=cls.AGENT_VERSION,
            tools=tools,
            category="System",
            visibility="Public",
            author="Chungoid Systems",
            # icon_url="URL to an icon for this agent", # Optional
            # tags=["filesystem", "file management", "io"], # Optional
        )

    # --------------------------------------------------------------------------
    # File System Operation Methods
    # --------------------------------------------------------------------------

    async def create_directory(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Creates a directory (and any parent directories if they don't exist)."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)
            
            if sandboxed_path.exists() and not sandboxed_path.is_dir():
                return FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error=f"Path exists but is not a directory."
                ).model_dump()

            os.makedirs(sandboxed_path, exist_ok=True)
            return FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message="Directory created or already exists."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(
                success=False, 
                error=f"Failed to create directory: {str(e)}", 
                path=path
            ).model_dump()

    async def create_file(self, path: str, project_root: Path, content: str = "", overwrite: bool = False) -> Dict[str, Any]:
        """Creates a file with optional content. Fails if overwrite is False and file exists."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if sandboxed_path.is_dir():
                return FileSystemOutput(
                    success=False,
                    path=str(sandboxed_path),
                    error="Path exists and is a directory, not a file."
                ).model_dump()

            if sandboxed_path.exists() and not overwrite:
                return FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error="File exists and overwrite is set to False."
                ).model_dump()
            
            # Ensure parent directory exists
            parent_dir = sandboxed_path.parent
            if not parent_dir.exists():
                os.makedirs(parent_dir, exist_ok=True)
            elif not parent_dir.is_dir():
                return FileSystemOutput(
                    success=False,
                    path=str(parent_dir),
                    error=f"Cannot create file. Parent path '{parent_dir}' exists but is not a directory."
                ).model_dump()

            with open(sandboxed_path, 'w' if overwrite else 'x') as f:
                f.write(content)
            
            return FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message="File created successfully."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except FileExistsError: # Specifically for 'x' mode when file exists and overwrite is False
             return FileSystemOutput(
                success=False, 
                path=path, # Use original path in error message
                error="File exists and overwrite is set to False (FileExistsError)."
            ).model_dump()
        except Exception as e:
            return FileSystemOutput(
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
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if sandboxed_path.is_dir():
                return FileSystemOutput(
                    success=False,
                    path=str(sandboxed_path),
                    error="Path exists and is a directory, cannot write content to it."
                ).model_dump()

            # Ensure parent directory exists
            parent_dir = sandboxed_path.parent
            if not parent_dir.exists():
                os.makedirs(parent_dir, exist_ok=True)
            elif not parent_dir.is_dir():
                return FileSystemOutput(
                    success=False,
                    path=str(parent_dir),
                    error=f"Cannot write to file. Parent path '{parent_dir}' exists but is not a directory."
                ).model_dump()

            mode = 'a' if append else 'w'
            with open(sandboxed_path, mode) as f:
                f.write(content)
            
            action = "appended to" if append else "written to"
            return FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message=f"Content successfully {action} file."
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(
                success=False, 
                error=f"Failed to write to file: {str(e)}", 
                path=path
            ).model_dump()

    async def read_file(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Reads the content of a file."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if not sandboxed_path.exists():
                return FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error="File not found."
                ).model_dump()
            
            if sandboxed_path.is_dir():
                return FileSystemOutput(
                    success=False, 
                    path=str(sandboxed_path), 
                    error="Path is a directory, not a file."
                ).model_dump()

            with open(sandboxed_path, 'r') as f:
                content = f.read()
            
            return FileSystemOutput(
                success=True, 
                path=str(sandboxed_path), 
                message="File read successfully.",
                content=content
            ).model_dump()
        except ValueError as ve: # Catch sandbox violation
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(
                success=False, 
                error=f"Failed to read file: {str(e)}", 
                path=path
            ).model_dump()

    async def delete_file(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Deletes a file."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if not sandboxed_path.exists():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="File not found.").model_dump()
            if sandboxed_path.is_dir():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is a directory, not a file. Use delete_directory.").model_dump()

            os.remove(sandboxed_path)
            return FileSystemOutput(success=True, path=str(sandboxed_path), message="File deleted successfully.").model_dump()
        except ValueError as ve:
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(success=False, error=f"Failed to delete file: {str(e)}", path=path).model_dump()

    async def delete_directory(self, path: str, project_root: Path, recursive: bool = False) -> Dict[str, Any]:
        """Deletes a directory. If recursive is False, the directory must be empty."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if not sandboxed_path.exists():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory not found.").model_dump()
            if not sandboxed_path.is_dir():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is a file, not a directory. Use delete_file.").model_dump()

            if recursive:
                shutil.rmtree(sandboxed_path)
                message = "Directory and its contents deleted successfully."
            else:
                if any(sandboxed_path.iterdir()): # Check if directory is not empty
                    return FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory is not empty and recursive is False.").model_dump()
                os.rmdir(sandboxed_path)
                message = "Empty directory deleted successfully."
            
            return FileSystemOutput(success=True, path=str(sandboxed_path), message=message).model_dump()
        except ValueError as ve:
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(success=False, error=f"Failed to delete directory: {str(e)}", path=path).model_dump()

    async def path_exists(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Checks if a path (file or directory) exists."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)
            exists = sandboxed_path.exists()
            return FileSystemOutput(success=True, path=str(sandboxed_path), exists=exists, message=f"Path checked.").model_dump()
        except ValueError as ve: # Path traversal or outside project
            return FileSystemOutput(success=False, error=str(ve), path=path, exists=False).model_dump()
        except Exception as e: # Other unexpected errors
            return FileSystemOutput(success=False, error=f"Error checking path existence: {str(e)}", path=path, exists=False).model_dump()

    async def list_directory_contents(self, path: str, project_root: Path) -> Dict[str, Any]:
        """Lists the contents (files and subdirectories) of a directory."""
        try:
            sandboxed_path = self._resolve_and_sandbox_path(path, project_root)

            if not sandboxed_path.exists():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="Directory not found.").model_dump()
            if not sandboxed_path.is_dir():
                return FileSystemOutput(success=False, path=str(sandboxed_path), error="Path is not a directory.").model_dump()

            contents = [item.name for item in os.listdir(sandboxed_path)]
            return FileSystemOutput(success=True, path=str(sandboxed_path), content=contents, message="Directory contents listed successfully.").model_dump()
        except ValueError as ve:
            return FileSystemOutput(success=False, error=str(ve), path=path).model_dump()
        except Exception as e:
            return FileSystemOutput(success=False, error=f"Failed to list directory contents: {str(e)}", path=path).model_dump()

    async def move_path(self, src_path: str, dest_path: str, project_root: Path, overwrite: bool = False) -> Dict[str, Any]:
        """Moves a file or directory. Overwrites destination if overwrite is True."""
        try:
            sandboxed_src = self._resolve_and_sandbox_path(src_path, project_root)
            sandboxed_dest = self._resolve_and_sandbox_path(dest_path, project_root)

            if not sandboxed_src.exists():
                return FileSystemOutput(success=False, path=str(sandboxed_src), error="Source path not found.").model_dump()

            if sandboxed_dest.exists() and not overwrite:
                return FileSystemOutput(success=False, path=str(sandboxed_dest), error="Destination path exists and overwrite is False.").model_dump()
            
            if sandboxed_dest.exists() and overwrite:
                if sandboxed_dest.is_dir() and not sandboxed_src.is_dir(): # Cannot overwrite dir with file
                     return FileSystemOutput(success=False, path=str(sandboxed_dest), error="Cannot overwrite a directory with a file.").model_dump()
                elif sandboxed_dest.is_file() and sandboxed_src.is_dir(): # Cannot overwrite file with dir using simple move
                     return FileSystemOutput(success=False, path=str(sandboxed_dest), error="Cannot overwrite a file with a directory using simple move. Delete destination first or use a different destination.").model_dump()
                elif sandboxed_dest.is_dir():
                    shutil.rmtree(sandboxed_dest) # remove existing dir if src is also dir
                else:
                    os.remove(sandboxed_dest) # remove existing file

            shutil.move(str(sandboxed_src), str(sandboxed_dest))
            return FileSystemOutput(success=True, path=str(sandboxed_dest), message=f"Path moved successfully from '{sandboxed_src}' to '{sandboxed_dest}'.").model_dump()
        except ValueError as ve:
            return FileSystemOutput(success=False, error=str(ve)).model_dump() # path will be in ve
        except Exception as e:
            return FileSystemOutput(success=False, error=f"Failed to move path: {str(e)}").model_dump()

    async def copy_path(self, src_path: str, dest_path: str, project_root: Path, overwrite: bool = False) -> Dict[str, Any]:
        """Copies a file or directory. Overwrites destination if overwrite is True."""
        try:
            sandboxed_src = self._resolve_and_sandbox_path(src_path, project_root)
            sandboxed_dest = self._resolve_and_sandbox_path(dest_path, project_root)

            if not sandboxed_src.exists():
                return FileSystemOutput(success=False, path=str(sandboxed_src), error="Source path not found.").model_dump()

            if sandboxed_dest.exists() and not overwrite:
                return FileSystemOutput(success=False, path=str(sandboxed_dest), error="Destination path exists and overwrite is False.").model_dump()

            if sandboxed_dest.exists() and overwrite:
                if sandboxed_dest.is_dir():
                    shutil.rmtree(sandboxed_dest)
                else:
                    os.remove(sandboxed_dest)
            
            if sandboxed_src.is_dir():
                shutil.copytree(sandboxed_src, sandboxed_dest)
            else: # It's a file
                shutil.copy2(sandboxed_src, sandboxed_dest) # copy2 preserves metadata
            
            return FileSystemOutput(success=True, path=str(sandboxed_dest), message=f"Path copied successfully from '{sandboxed_src}' to '{sandboxed_dest}'.").model_dump()
        except ValueError as ve:
            return FileSystemOutput(success=False, error=str(ve)).model_dump()
        except Exception as e:
            return FileSystemOutput(success=False, error=f"Failed to copy path: {str(e)}").model_dump()

    # --------------------------------------------------------------------------

# To be able to run this agent directly for testing (optional)
if __name__ == "__main__":
    async def main():
        agent = FileSystemAgent()
        
        # Example usage (will not work until methods and schemas are defined)
        # test_project_dir = Path("./temp_test_project_fs_agent")
        # test_project_dir.mkdir(exist_ok=True)
        #
        # print("Agent Card:", agent.get_agent_card().model_dump_json(indent=2))
        #
        # # Test create_directory
        # result_create_dir = await agent.invoke_async(
        #     tool_name="create_directory",
        #     tool_input=CreateDirectoryInput(path="my_new_folder_from_invoke").model_dump(),
        #     project_root=test_project_dir
        # )
        # print("\nCreate Directory (via invoke_async) Result:", result_create_dir)

        # # Test sandbox violation
        # result_sandbox_violation = await agent.invoke_async(
        #    tool_name="create_directory",
        #    tool_input=CreateDirectoryInput(path="../outside_project_folder").model_dump(),
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