import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

from chungoid.schemas.agent_code_integration import CodeIntegrationInput, CodeIntegrationOutput
from chungoid.schemas.agent_registry import AgentCard, AgentCategory, AgentConfigSchema, AgentInputSchema, AgentOutputSchema # Assuming these are standard
from chungoid.utils.logger_setup import logger # Assuming a common logger

# Placeholder for a more formal BaseAgent if it exists in the project
# For now, a simple class structure will suffice.

class CoreCodeIntegrationAgentV1:
    """    Core Code Integration Agent (Version 1).

    Provides functionalities to integrate and modify Python code in files.
    Supports actions such as appending code, creating or appending to files,
    adding Python import statements, and adding Click command definitions (V1: by appending).
    Includes options for backing up original files before modification.
    """

    AGENT_ID = "core.code_integration_agent_v1"
    AGENT_FRIENDLY_NAME = "Core Code Integration Agent V1"
    AGENT_DESCRIPTION = "Integrates code snippets into Python files, supporting appending, Click command integration, and import addition."
    AGENT_CATEGORY = AgentCategory.CODE_EDITING # Assuming AgentCategory.CODE_EDITING exists

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"CoreCodeIntegrationAgentV1 initialized with config: {self.config}")

    async def invoke_async(self, inputs: CodeIntegrationInput) -> CodeIntegrationOutput:
        logger.info(f"CoreCodeIntegrationAgentV1 invoked with action: {inputs.edit_action} for target: {inputs.target_file_path}")
        
        target_path = Path(inputs.target_file_path)
        backup_file_path_str: Optional[str] = None

        try:
            # Ensure parent directory exists if we are creating a file
            if inputs.edit_action in ["CREATE_OR_APPEND", "ADD_TO_CLICK_GROUP", "ADD_PYTHON_IMPORTS"] and not target_path.parent.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created parent directory: {target_path.parent}")

            # Handle backup
            if inputs.backup_original and target_path.exists() and target_path.is_file():
                backup_file_path = target_path.with_suffix(target_path.suffix + ".bak")
                shutil.copy2(target_path, backup_file_path)
                backup_file_path_str = str(backup_file_path)
                logger.info(f"Backed up original file to: {backup_file_path_str}")

            if inputs.edit_action == "APPEND":
                if not target_path.exists() or not target_path.is_file():
                    logger.error(f"Action 'APPEND' failed: Target file {target_path} does not exist or is not a file.")
                    return CodeIntegrationOutput(
                        status="FAILURE",
                        message=f"Target file {target_path} does not exist or is not a file for APPEND action.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )
                
                content_to_add = inputs.code_to_integrate
                # Ensure a newline before appending if the file is not empty and doesn't end with one
                if target_path.stat().st_size > 0:
                    with open(target_path, 'r', encoding='utf-8') as f_read:
                        if not f_read.read().endswith('\n'):
                            content_to_add = "\n" + content_to_add
                
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(content_to_add)
                logger.info(f"Appended code to {target_path}")
                return CodeIntegrationOutput(
                    status="SUCCESS", 
                    message=f"Action 'APPEND' completed successfully for {target_path}", 
                    modified_file_path=str(target_path),
                    backup_file_path=backup_file_path_str
                )
            elif inputs.edit_action == "CREATE_OR_APPEND":
                content_to_add = inputs.code_to_integrate
                if target_path.exists() and target_path.is_file():
                    # Ensure a newline before appending if the file is not empty and doesn't end with one
                    if target_path.stat().st_size > 0:
                        with open(target_path, 'r', encoding='utf-8') as f_read:
                            if not f_read.read().endswith('\n'):
                                content_to_add = "\n" + content_to_add
                    with open(target_path, "a", encoding="utf-8") as f:
                        f.write(content_to_add)
                    logger.info(f"Appended code to existing file {target_path}")
                else:
                    # If it's not a file (e.g. a directory) or doesn't exist, create/overwrite
                    if target_path.is_dir():
                         logger.error(f"Action 'CREATE_OR_APPEND' failed: Target path {target_path} is a directory.")
                         return CodeIntegrationOutput(
                            status="FAILURE",
                            message=f"Target path {target_path} is a directory, cannot write file.",
                            modified_file_path=str(target_path),
                            backup_file_path=backup_file_path_str
                        )
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(inputs.code_to_integrate) # Write original content for new file
                    logger.info(f"Created new file and wrote code to {target_path}")
                
                return CodeIntegrationOutput(
                    status="SUCCESS", 
                    message=f"Action 'CREATE_OR_APPEND' completed successfully for {target_path}", 
                    modified_file_path=str(target_path),
                    backup_file_path=backup_file_path_str
                )
            elif inputs.edit_action == "ADD_TO_CLICK_GROUP":
                # V1: Assumes inputs.code_to_integrate is a full, decorated Click command.
                # integration_point_hint and click_command_name are for the generator of the code,
                # this agent simply appends the provided code block.
                if not target_path.exists() or not target_path.is_file():
                    logger.error(f"Action 'ADD_TO_CLICK_GROUP' failed: Target file {target_path} does not exist or is not a file.")
                    return CodeIntegrationOutput(
                        status="FAILURE",
                        message=f"Target file {target_path} does not exist or is not a file for ADD_TO_CLICK_GROUP action.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )

                code_block_to_add = inputs.code_to_integrate
                
                # Ensure separation with a couple of newlines if file is not empty
                prepended_newlines = "\n\n" # Start with two newlines

                if target_path.stat().st_size > 0:
                    with open(target_path, 'r', encoding='utf-8') as f_read:
                        content = f_read.read()
                        if not content.endswith('\n'): # Needs at least one to separate
                            prepended_newlines = "\n" + prepended_newlines
                        elif content.endswith('\n\n'): # Already has two or more
                            prepended_newlines = ""
                        elif content.endswith('\n'): # Has one, need one more
                            prepended_newlines = "\n"
                else: # Empty file
                    prepended_newlines = "" # No leading newlines needed for an empty file

                final_code_to_add = prepended_newlines + code_block_to_add
                # Ensure the block itself ends with a newline if it doesn't already
                if not final_code_to_add.endswith('\n'):
                    final_code_to_add += '\n'

                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(final_code_to_add)
                
                logger.info(f"Appended Click command to {target_path}. Input hint: '{inputs.integration_point_hint}', cmd: '{inputs.click_command_name}'. V1 appends block.")
                return CodeIntegrationOutput(
                    status="SUCCESS",
                    message=f"Action 'ADD_TO_CLICK_GROUP' (V1: append) completed for {target_path}",
                    modified_file_path=str(target_path),
                    backup_file_path=backup_file_path_str
                )
            elif inputs.edit_action == "ADD_PYTHON_IMPORTS":
                if not target_path.exists() or not target_path.is_file():
                    logger.error(f"Action 'ADD_PYTHON_IMPORTS' failed: Target file {target_path} does not exist or is not a file.")
                    return CodeIntegrationOutput(
                        status="FAILURE",
                        message=f"Target file {target_path} does not exist or is not a file for ADD_PYTHON_IMPORTS action.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )

                with open(target_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()

                new_imports_str = inputs.code_to_integrate.strip()
                new_imports_list = [imp.strip() for imp in new_imports_str.split('\n') if imp.strip()]
                
                existing_imports_condensed = set()
                for line in lines:
                    line_stripped = line.strip()
                    if line_stripped.startswith("import ") or line_stripped.startswith("from "):
                        existing_imports_condensed.add(" ".join(line_stripped.split()))


                imports_to_add = []
                for new_import in new_imports_list:
                    condensed_new_import = " ".join(new_import.split())
                    if condensed_new_import not in existing_imports_condensed:
                        imports_to_add.append(new_import + "\n") # Ensure newline

                if not imports_to_add:
                    logger.info(f"No new imports to add to {target_path} as they already exist or none were provided.")
                    return CodeIntegrationOutput(
                        status="SUCCESS",
                        message=f"No new imports needed for {target_path}.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )

                last_import_line_index = -1
                for i, line in enumerate(lines):
                    if line.strip().startswith("import ") or line.strip().startswith("from "):
                        last_import_line_index = i
                
                # Determine if a blank line is needed after imports
                blank_line_needed = False
                if last_import_line_index != -1 and (last_import_line_index + 1) < len(lines):
                     # Check if the line after the last import is not empty and not another import
                    next_line_strip = lines[last_import_line_index + 1].strip()
                    if next_line_strip and not next_line_strip.startswith("import ") and not next_line_strip.startswith("from "):
                         blank_line_needed = True


                if last_import_line_index == -1: # No imports found, add at the beginning
                    # If file is not empty, add a newline after the new imports
                    if lines and lines[0].strip():
                        imports_to_add.append("\n")
                    lines = imports_to_add + lines
                else: # Imports found, add after the last one
                    # If a blank line is needed, add it before inserting new imports
                    # or if the new imports themselves should be followed by one if they are the last thing.
                    # Simplified: just add the imports. A blank line might already be there or added by formatter.
                    # For now, just ensure the imports themselves end with a newline.
                    # The last import in imports_to_add already has \n.
                    # If blank_line_needed is True, it implies we should insert a blank line before the next code block
                    # IF the new imports are inserted. Let's refine:
                    
                    # Insert imports
                    lines = lines[:last_import_line_index + 1] + imports_to_add + lines[last_import_line_index + 1:]

                    # Check if a blank line is needed between the newly added imports and subsequent code.
                    # The new imports are now at lines[last_import_line_index + 1] up to lines[last_import_line_index + len(imports_to_add)]
                    new_last_import_idx = last_import_line_index + len(imports_to_add)
                    if (new_last_import_idx + 1) < len(lines):
                        next_line_strip = lines[new_last_import_idx + 1].strip()
                        # If the line immediately after our new imports is not empty and not a comment (simple check)
                        if next_line_strip and not next_line_strip.startswith("#"):
                             # And if the last of our new imports didn't naturally create a blank line (e.g. user put \n\n)
                             if lines[new_last_import_idx].strip(): # our last import has content
                                lines.insert(new_last_import_idx + 1, "\n")


                with open(target_path, "w", encoding="utf-8") as f:
                    f.writelines(lines)

                logger.info(f"Added Python imports to {target_path}")
                return CodeIntegrationOutput(
                    status="SUCCESS",
                    message=f"Action 'ADD_PYTHON_IMPORTS' completed successfully for {target_path}",
                    modified_file_path=str(target_path),
                    backup_file_path=backup_file_path_str
                )
            else:
                return CodeIntegrationOutput(
                    status="FAILURE", 
                    message=f"Unknown edit_action: {inputs.edit_action}",
                    modified_file_path=str(target_path)
                )
            
            # If an action were implemented successfully:
            # return CodeIntegrationOutput(
            #     status="SUCCESS", 
            #     message=f"Action '{inputs.edit_action}' completed successfully for {target_path}", 
            #     modified_file_path=str(target_path),
            #     backup_file_path=backup_file_path_str
            # )

        except NotImplementedError as e:
            logger.warning(f"Action {inputs.edit_action} for {target_path} is not implemented: {e}")
            return CodeIntegrationOutput(status="FAILURE", message=str(e), modified_file_path=str(target_path), backup_file_path=backup_file_path_str)
        except Exception as e:
            logger.error(f"Error during code integration action '{inputs.edit_action}' on '{target_path}': {e}", exc_info=True)
            return CodeIntegrationOutput(status="FAILURE", message=str(e), modified_file_path=str(target_path), backup_file_path=backup_file_path_str)

    @classmethod
    def get_agent_card(cls) -> AgentCard:
        return AgentCard(
            agent_id=cls.AGENT_ID,
            friendly_name=cls.AGENT_FRIENDLY_NAME,
            description=cls.AGENT_DESCRIPTION,
            category=cls.AGENT_CATEGORY,
            config_schema=AgentConfigSchema(is_configurable=False).model_dump(), # Example, adjust as needed
            input_schema=AgentInputSchema(schema_type="pydantic", schema_definition=CodeIntegrationInput.model_json_schema()).model_dump(),
            output_schema=AgentOutputSchema(schema_type="pydantic", schema_definition=CodeIntegrationOutput.model_json_schema()).model_dump(),
            version="0.1.0"
        )

# Example of how it might be registered (actual registration mechanism might differ)
# from chungoid.utils.agent_registry import AgentRegistry
# AgentRegistry.register_agent_class(CoreCodeIntegrationAgentV1) 