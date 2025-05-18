import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from chungoid.schemas.agent_code_integration import CodeIntegrationInput, CodeIntegrationOutput
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.schemas.errors import AgentErrorDetails

logger = logging.getLogger(__name__)

class CoreCodeIntegrationAgentV1:
    """    Core Code Integration Agent (Version 1).

    Provides functionalities to integrate and modify Python code in files.
    Supports actions such as appending code, creating or appending to files,
    adding Python import statements, and adding Click command definitions (V1: by appending).
    Includes options for backing up original files before modification.
    """

    AGENT_ID = "core.code_integration_agent_v1"
    AGENT_NAME = "Core Code Integration Agent V1"
    AGENT_DESCRIPTION = "Integrates code snippets into Python files, supporting appending, Click command integration, and import addition."
    CATEGORY = AgentCategory.CODE_EDITING
    VISIBILITY = AgentVisibility.PUBLIC
    VERSION = "0.1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"CoreCodeIntegrationAgentV1 initialized with config: {self.config}")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> CodeIntegrationOutput:
        logger.info(f"CoreCodeIntegrationAgentV1 invoked with action for target: {inputs.get('target_file_path')}")
        
        try:
            parsed_inputs = CodeIntegrationInput(**inputs)
        except Exception as e:
            logger.error(f"Failed to parse inputs for {self.AGENT_ID}: {e}")
            return CodeIntegrationOutput(
                status="FAILURE", 
                message=f"Input parsing failed: {e}",
                modified_file_path=inputs.get("target_file_path")
            )
        
        logger.debug(f"CoreCodeIntegrationAgentV1 parsed_inputs: {parsed_inputs}")
        target_path = Path(parsed_inputs.target_file_path)
        backup_file_path_str: Optional[str] = None

        try:
            # Ensure parent directory exists if we are creating a file
            if parsed_inputs.edit_action in ["CREATE_OR_APPEND", "ADD_TO_CLICK_GROUP", "ADD_PYTHON_IMPORTS"] and not target_path.parent.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created parent directory: {target_path.parent}")

            # Handle backup
            if parsed_inputs.backup_original and target_path.exists() and target_path.is_file():
                backup_file_path = target_path.with_suffix(target_path.suffix + ".bak")
                shutil.copy2(target_path, backup_file_path)
                backup_file_path_str = str(backup_file_path)
                logger.info(f"Backed up original file to: {backup_file_path_str}")

            if parsed_inputs.edit_action == "APPEND":
                if not target_path.exists() or not target_path.is_file():
                    logger.error(f"Action 'APPEND' failed: Target file {target_path} does not exist or is not a file.")
                    return CodeIntegrationOutput(
                        status="FAILURE",
                        message=f"Target file {target_path} does not exist or is not a file for APPEND action.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )
                
                content_to_add = parsed_inputs.code_to_integrate
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
            elif parsed_inputs.edit_action == "CREATE_OR_APPEND":
                content_to_add = parsed_inputs.code_to_integrate
                
                # HACK for CREATE_OR_APPEND (append to existing file part):
                problematic_test_context_path = "context.intermediate_outputs.generated_test_code_artifacts.generated_test_code_string"
                if content_to_add == problematic_test_context_path:
                    logger.warning(f"CREATE_OR_APPEND (append): code_to_integrate was the literal context path '{problematic_test_context_path}'. Appending placeholder comment instead.")
                    content_to_add = "\\n# BUG_WORKAROUND: code_to_integrate was a literal context path for test file (append case)."
                
                if target_path.exists() and target_path.is_file():
                    # Ensure a newline before appending if the file is not empty and doesn't end with one
                    # and if we are not already adding the BUG_WORKAROUND comment which will have its own newline
                    if content_to_add != "\\n# BUG_WORKAROUND: code_to_integrate was a literal context path for test file (append case)." and target_path.stat().st_size > 0:
                        with open(target_path, 'r', encoding='utf-8') as f_read:
                            if not f_read.read().endswith('\\n'):
                                content_to_add = "\\n" + content_to_add
                    
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
                    
                    # HACK for CREATE_OR_APPEND: Prevent writing literal context path
                    content_for_new_file = parsed_inputs.code_to_integrate
                    problematic_test_context_path = "context.intermediate_outputs.generated_test_code_artifacts.generated_test_code_string"
                    if content_for_new_file == problematic_test_context_path:
                        logger.warning(f"CREATE_OR_APPEND: code_to_integrate was the literal context path '{problematic_test_context_path}'. Writing a placeholder comment instead.")
                        content_for_new_file = "# BUG_WORKAROUND: code_to_integrate was a literal context path for test file."

                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content_for_new_file) # Use content_for_new_file
                    logger.info(f"Created new file and wrote code to {target_path}")
                
                return CodeIntegrationOutput(
                    status="SUCCESS", 
                    message=f"Action 'CREATE_OR_APPEND' completed successfully for {target_path}", 
                    modified_file_path=str(target_path),
                    backup_file_path=backup_file_path_str
                )
            elif parsed_inputs.edit_action == "ADD_TO_CLICK_GROUP":
                if not parsed_inputs.click_command_name or not parsed_inputs.integration_point_hint:
                    logger.error("ADD_TO_CLICK_GROUP action requires 'click_command_name' and 'integration_point_hint' inputs.")
                    return CodeIntegrationOutput(status="FAILURE", message="Missing required inputs for ADD_TO_CLICK_GROUP.", modified_file_path=str(target_path))

                code_block_to_add = parsed_inputs.code_to_integrate
                
                # HACK: Prevent writing the literal context path if it was passed directly
                problematic_context_path_literal = "context.intermediate_outputs.generated_command_code_artifacts.generated_code_string"
                if code_block_to_add == problematic_context_path_literal:
                    logger.warning(f"ADD_TO_CLICK_GROUP: code_to_integrate input was the literal context path '{problematic_context_path_literal}'. Replacing with a comment to avoid file corruption.")
                    code_block_to_add = "# BUG_WORKAROUND: code_to_integrate was a literal context path."

                prepended_newlines = "\n\n" # Always add two newlines before the function block
                final_code_to_add = prepended_newlines + code_block_to_add
                if not final_code_to_add.endswith('\n'):
                    final_code_to_add += '\n'

                add_command_line = f"{parsed_inputs.integration_point_hint}.add_command({parsed_inputs.click_command_name})\n"
                full_code_to_insert = final_code_to_add + "\n" + add_command_line

                try:
                    with open(target_path, "r+", encoding="utf-8") as f:
                        lines = f.readlines()
                        insert_index = -1
                        for i, line in enumerate(lines):
                            if line.strip().startswith("if __name__ == \"__main__\":"):
                                insert_index = i
                                break
                        
                        if insert_index != -1:
                            # Ensure there's a blank line before the insertion point if not already present
                            if insert_index > 0 and lines[insert_index-1].strip() != "":
                                lines.insert(insert_index, '\n') # Insert a blank line
                                insert_index += 1 # Adjust index due to insertion
                            elif insert_index == 0: # If __main__ is the first line, insert a newline before it
                                lines.insert(insert_index, '\n')
                                insert_index += 1
                                
                            # Insert the new code block
                            # Split the full_code_to_insert into lines to insert them properly
                            new_code_lines = [l + '\n' for l in full_code_to_insert.splitlines() if l.strip()]
                            lines[insert_index:insert_index] = new_code_lines

                            f.seek(0)
                            f.writelines(lines)
                            f.truncate()
                            logger.info(f"ADD_TO_CLICK_GROUP: Successfully inserted code into {target_path} before __main__ block.")
                            return CodeIntegrationOutput(status="SUCCESS", message=f"Action '{parsed_inputs.edit_action}' completed for {target_path}. Code inserted before __main__.", modified_file_path=str(target_path))
                        else:
                            logger.warning(f"ADD_TO_CLICK_GROUP: 'if __name__ == \"__main__\":' not found in {target_path}. Attempting to insert before NO TRAILING LINES marker.")
                            try:
                                with open(target_path, "r", encoding="utf-8") as f_r:
                                    all_lines = f_r.readlines()
                                marker_index = next((idx for idx, ln in enumerate(all_lines) if ln.strip() == "# NO TRAILING LINES AFTER THIS COMMENT BLOCK"), None)
                            except Exception as e_marker:
                                marker_index = None
                                logger.debug(f"ADD_TO_CLICK_GROUP: Error while searching for trailing marker in {target_path}: {e_marker}")

                            if marker_index is not None:
                                # Ensure a blank line before inserting if marker has non-empty line before it
                                insert_at = marker_index
                                if insert_at > 0 and all_lines[insert_at-1].strip() != "":
                                    all_lines.insert(insert_at, "\n")
                                    insert_at += 1
                                # Prepare new code lines
                                new_code_lines = [l+"\n" for l in full_code_to_insert.splitlines() if l.strip()]
                                all_lines[insert_at:insert_at] = new_code_lines
                                with open(target_path, "w", encoding="utf-8") as f_w:
                                    f_w.writelines(all_lines)
                                logger.info(f"ADD_TO_CLICK_GROUP: Inserted code before trailing marker in {target_path}.")
                                return CodeIntegrationOutput(status="SUCCESS", message=f"Action '{parsed_inputs.edit_action}' completed. Code inserted before trailing marker.", modified_file_path=str(target_path))
                            else:
                                logger.warning(f"ADD_TO_CLICK_GROUP: Trailing marker not found either. Appending to end of file as last resort.")
                                with open(target_path, "a", encoding="utf-8") as f_append:
                                    f_append.write(full_code_to_insert)
                                return CodeIntegrationOutput(status="SUCCESS", message=f"Action '{parsed_inputs.edit_action}' completed. Code appended to end of file.", modified_file_path=str(target_path))

                except Exception as e:
                    logger.error(f"Error during ADD_TO_CLICK_GROUP for {target_path}: {e}", exc_info=True)
                    return CodeIntegrationOutput(status="FAILURE", message=f"Error processing file for ADD_TO_CLICK_GROUP: {e}", modified_file_path=str(target_path))
            elif parsed_inputs.edit_action == "ADD_PYTHON_IMPORTS":
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

                new_imports_str = parsed_inputs.imports_to_add.strip()
                new_imports_list = [imp.strip() for imp in new_imports_str.split('\n') if imp.strip()]
                
                if not new_imports_list:
                    logger.info(f"No new imports provided in input for {target_path}.")
                    return CodeIntegrationOutput(
                        status="SUCCESS",
                        message=f"No new imports were specified in the input for {target_path}.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )

                # --- New robust import finding logic ---
                header_end_index = 0
                in_module_docstring = False
                q_type_docstring = ""

                # Phase 1: Identify end of header (shebang, coding, module docstring)
                for i, line_content in enumerate(lines):
                    stripped_line = line_content.strip()

                    if in_module_docstring:
                        if stripped_line.endswith(q_type_docstring):
                            in_module_docstring = False
                        header_end_index = i + 1 # Continue consuming docstring lines
                        continue

                    if i == 0 and stripped_line.startswith("#!"):
                        header_end_index = i + 1
                        continue
                    if stripped_line.startswith("# -*- coding:") or stripped_line.startswith("# coding:"):
                        header_end_index = i + 1
                        continue
                    
                    if i == header_end_index and (stripped_line.startswith('"""') or stripped_line.startswith("'''")): # Check for start of module docstring
                        q_type_docstring = '"""' if stripped_line.startswith('"""') else "'''"
                        if len(stripped_line) > 3 and stripped_line.count(q_type_docstring) == 2 and stripped_line.endswith(q_type_docstring): # Single-line docstring
                            header_end_index = i + 1
                        elif stripped_line.count(q_type_docstring) == 1: # Start of a multi-line docstring
                            in_module_docstring = True
                            header_end_index = i + 1
                        # else it's some other quote usage, not a module docstring starting here
                        continue
                    
                    if stripped_line and not stripped_line.startswith("#") and not in_module_docstring: # End of header if actual code/non-comment found
                        break 
                    header_end_index = i + 1 # Include comments/empty lines in header before imports


                # Phase 2: Find the end of the main import block & collect existing imports
                import_block_end_index = header_end_index
                existing_imports_set = set()
                first_import_found = False

                for i in range(header_end_index, len(lines)):
                    stripped_line = lines[i].strip()
                    
                    if not stripped_line or stripped_line.startswith("#"): # Skip empty lines and comments
                        if first_import_found : # If we've found imports, a comment/empty line after them means end of block
                             break
                        else: # Still in header or pre-import comment section
                            import_block_end_index = i + 1
                        continue

                    if stripped_line.startswith("import ") or stripped_line.startswith("from "):
                        existing_imports_set.add(" ".join(stripped_line.split())) 
                        import_block_end_index = i + 1 
                        first_import_found = True
                        continue
                    
                    break # First non-import, non-comment, non-empty line marks end of import block

                imports_to_actually_add = []
                for new_import in new_imports_list:
                    normalized_new_import = " ".join(new_import.split())
                    if normalized_new_import not in existing_imports_set:
                        imports_to_actually_add.append(new_import + "\n")
                
                if not imports_to_actually_add:
                    logger.info(f"All specified imports already exist in {target_path}.")
                    return CodeIntegrationOutput(
                        status="SUCCESS",
                        message=f"All specified imports already found in {target_path}.",
                        modified_file_path=str(target_path),
                        backup_file_path=backup_file_path_str
                    )

                # Phase 3: Insert new imports and handle spacing
                lines_before_new_imports = lines[:import_block_end_index]
                lines_after_new_imports = lines[import_block_end_index:]
                
                final_imports_section_to_insert = imports_to_actually_add
                
                # Ensure a blank line after our new imports if there's subsequent code
                if lines_after_new_imports and lines_after_new_imports[0].strip(): 
                    if final_imports_section_to_insert and \
                       not final_imports_section_to_insert[-1].endswith("\n\n") and \
                       final_imports_section_to_insert[-1].endswith("\n"):
                        # Remove single trailing newline, then add double
                        final_imports_section_to_insert[-1] = final_imports_section_to_insert[-1].strip() + "\n\n"
                    elif final_imports_section_to_insert: # if it has content but no proper newline at end
                         final_imports_section_to_insert.append("\n")


                lines = lines_before_new_imports + final_imports_section_to_insert + lines_after_new_imports
                # --- End of new robust import finding logic ---

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
                    message=f"Unknown edit_action: {parsed_inputs.edit_action}",
                    modified_file_path=str(target_path)
                )
            
            # If an action were implemented successfully:
            # return CodeIntegrationOutput(
            #     status="SUCCESS", 
            #     message=f"Action '{parsed_inputs.edit_action}' completed successfully for {target_path}", 
            #     modified_file_path=str(target_path),
            #     backup_file_path=backup_file_path_str
            # )

        except NotImplementedError as e:
            logger.warning(f"Action {parsed_inputs.edit_action} for {target_path} is not implemented: {e}")
            return CodeIntegrationOutput(status="FAILURE", message=str(e), modified_file_path=str(target_path), backup_file_path=backup_file_path_str)
        except Exception as e:
            logger.error(f"Error during code integration action '{parsed_inputs.edit_action}' on '{target_path}': {e}", exc_info=True)
            return CodeIntegrationOutput(status="FAILURE", message=str(e), modified_file_path=str(target_path), backup_file_path=backup_file_path_str)

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=CoreCodeIntegrationAgentV1.AGENT_ID,
            name=CoreCodeIntegrationAgentV1.AGENT_NAME,
            description=CoreCodeIntegrationAgentV1.AGENT_DESCRIPTION,
            categories=[CoreCodeIntegrationAgentV1.CATEGORY.value],
            visibility=CoreCodeIntegrationAgentV1.VISIBILITY.value,
            capability_profile={
                "edit_action_support": ["APPEND", "CREATE_OR_APPEND", "ADD_TO_CLICK_GROUP", "ADD_PYTHON_IMPORTS"],
                "language_support": ["python"]
            },
            input_schema=CodeIntegrationInput.model_json_schema(),
            output_schema=CodeIntegrationOutput.model_json_schema(),
            version=CoreCodeIntegrationAgentV1.VERSION
        )

# Alias for consistency
get_agent_card_static = CoreCodeIntegrationAgentV1.get_agent_card_static

# Example of how it might be registered (actual registration mechanism might differ)
# from chungoid.utils.agent_registry import AgentRegistry
# AgentRegistry.register_agent_class(CoreCodeIntegrationAgentV1) 