import asyncio
import os
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Literal
import logging
import hashlib
import uuid
import tempfile
import datetime

from chungoid.schemas.agent_code_integration import SmartCodeIntegrationInput, SmartCodeIntegrationOutput
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, LIVE_CODEBASE_COLLECTION

logger = logging.getLogger(__name__)

class SmartCodeIntegrationAgent_v1:
    """    Smart Code Integration Agent (Version 1).

    Fetches code from ChromaDB (or direct input), integrates it into files using various edit actions,
    and updates the live codebase representation in ChromaDB.
    """

    AGENT_ID = "SmartCodeIntegrationAgent_v1"
    AGENT_NAME = "Smart Code Integration Agent V1"
    AGENT_DESCRIPTION = "Integrates code (sourced from ChromaDB or direct input) into files and updates live codebase in ChromaDB."
    CATEGORY = AgentCategory.CODE_EDITING
    VISIBILITY = AgentVisibility.PUBLIC
    VERSION = "0.2.1"

    _pcma_agent: ProjectChromaManagerAgent_v1

    def __init__(self,
                 pcma_agent: ProjectChromaManagerAgent_v1,
                 config: Optional[Dict[str, Any]] = None,
                 system_context: Optional[Dict[str, Any]] = None
                ):
        if not pcma_agent:
            raise ValueError("ProjectChromaManagerAgent_v1 is required for SmartCodeIntegrationAgent_v1")
        self._pcma_agent = pcma_agent
        self.config = config if config else {}
        self.system_context = system_context or {}
        self._logger_instance = self.system_context.get("logger", logger)
        self._logger_instance.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> SmartCodeIntegrationOutput:
        task_id_from_input = inputs.get("task_id", str(uuid.uuid4()))
        try:
            parsed_inputs = SmartCodeIntegrationInput(**inputs)
        except Exception as e:
            self._logger_instance.error(f"Failed to parse inputs for {self.AGENT_ID}: {e}")
            return SmartCodeIntegrationOutput(
                task_id=task_id_from_input,
                status="FAILURE", 
                message=f"Input parsing failed: {e}",
                modified_file_path=inputs.get("target_file_path"),
                error_message=f"Input parsing failed: {e}"
            )
        
        self._logger_instance.info(f"{self.AGENT_NAME} invoked with action \'{parsed_inputs.edit_action}\' for target: {parsed_inputs.target_file_path} in project {parsed_inputs.project_id}")
        self._logger_instance.debug(f"{self.AGENT_NAME} parsed_inputs: {parsed_inputs}")

        code_to_integrate_str: Optional[str] = None
        if parsed_inputs.generated_code_artifact_doc_id:
            self._logger_instance.info(f"Fetching code from ChromaDB artifact ID: {parsed_inputs.generated_code_artifact_doc_id}")
            try:
                code_doc = await self._pcma_agent.get_document_by_id(
                    doc_id=parsed_inputs.generated_code_artifact_doc_id,
                    project_id=parsed_inputs.project_id 
                )
                if code_doc and code_doc.document_content:
                    code_to_integrate_str = code_doc.document_content
                    self._logger_instance.info(f"Successfully fetched code from doc_id: {parsed_inputs.generated_code_artifact_doc_id}")
                else:
                    raise ValueError(f"Document not found or content is empty for doc_id: {parsed_inputs.generated_code_artifact_doc_id}")
            except Exception as e_fetch:
                err_msg = f"Failed to fetch code from ChromaDB artifact {parsed_inputs.generated_code_artifact_doc_id}: {e_fetch}"
                self._logger_instance.error(err_msg, exc_info=True)
                return SmartCodeIntegrationOutput(task_id=parsed_inputs.task_id, status="FAILURE_CHROMA_FETCH", message=err_msg, modified_file_path=parsed_inputs.target_file_path, error_message=err_msg)
        elif parsed_inputs.code_to_integrate_directly:
            self._logger_instance.info("Using directly provided code string for integration.")
            code_to_integrate_str = parsed_inputs.code_to_integrate_directly
        else:
            err_msg = "Neither 'generated_code_artifact_doc_id' nor 'code_to_integrate_directly' was provided."
            self._logger_instance.error(err_msg)
            return SmartCodeIntegrationOutput(task_id=parsed_inputs.task_id, status="FAILURE_INPUT", message=err_msg, modified_file_path=parsed_inputs.target_file_path, error_message=err_msg)

        if code_to_integrate_str is None:
            err_msg = "Code to integrate could not be determined."
            self._logger_instance.error(err_msg)
            return SmartCodeIntegrationOutput(task_id=parsed_inputs.task_id, status="FAILURE_INPUT", message=err_msg, modified_file_path=parsed_inputs.target_file_path, error_message=err_msg)

        target_path = Path(parsed_inputs.target_file_path)
        backup_file_path_str: Optional[str] = None
        action_status: Literal["SUCCESS", "FAILURE"] = "FAILURE"
        action_message: str = "Action not completed."

        try:
            if parsed_inputs.edit_action in ["CREATE_OR_APPEND", "REPLACE_FILE_CONTENT"] and not target_path.parent.exists():
                target_path.parent.mkdir(parents=True, exist_ok=True)
                self._logger_instance.info(f"Created parent directory: {target_path.parent}")

            if parsed_inputs.backup_original and target_path.exists() and target_path.is_file():
                backup_file_path = target_path.with_suffix(target_path.suffix + f".bak_{uuid.uuid4().hex[:8]}")
                shutil.copy2(target_path, backup_file_path)
                backup_file_path_str = str(backup_file_path)
                self._logger_instance.info(f"Backed up original file to: {backup_file_path_str}")

            if parsed_inputs.edit_action == "REPLACE_FILE_CONTENT":
                if target_path.is_dir():
                    action_message = f"Target path {target_path} is a directory, cannot replace content."
                    raise IsADirectoryError(action_message)
                with open(target_path, "w", encoding="utf-8") as f:
                    f.write(code_to_integrate_str)
                self._logger_instance.info(f"Replaced content of file {target_path}")
                action_status, action_message = "SUCCESS", f"Action 'REPLACE_FILE_CONTENT' completed for {target_path}"
            
            elif parsed_inputs.edit_action == "APPEND":
                if not target_path.exists() or not target_path.is_file():
                    action_message = f"Target file {target_path} does not exist or is not a file for APPEND action."
                    raise FileNotFoundError(action_message)
                
                content_with_newline = code_to_integrate_str
                if target_path.stat().st_size > 0:
                    with open(target_path, 'r', encoding='utf-8') as f_read:
                        if not f_read.read().endswith('\n'):
                            content_with_newline = "\n" + code_to_integrate_str
                else:
                    if not content_with_newline.endswith('\n'):
                         content_with_newline += '\n'
                
                with open(target_path, "a", encoding="utf-8") as f:
                    f.write(content_with_newline)
                self._logger_instance.info(f"Appended code to {target_path}")
                action_status, action_message = "SUCCESS", f"Action 'APPEND' completed for {target_path}"

            elif parsed_inputs.edit_action == "CREATE_OR_APPEND":
                if target_path.is_dir():
                    action_message = f"Target path {target_path} is a directory, cannot write file."
                    raise IsADirectoryError(action_message)

                content_to_add_for_action = code_to_integrate_str
                if target_path.exists() and target_path.is_file():
                    if target_path.stat().st_size > 0:
                        with open(target_path, 'r', encoding='utf-8') as f_read:
                            if not f_read.read().endswith('\n'):
                                content_to_add_for_action = "\n" + code_to_integrate_str
                    with open(target_path, "a", encoding="utf-8") as f:
                        f.write(content_to_add_for_action)
                    self._logger_instance.info(f"Appended code to existing file {target_path}")
                else:
                    if not content_to_add_for_action.endswith('\n'):
                        content_to_add_for_action += '\n'
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(content_to_add_for_action)
                    self._logger_instance.info(f"Created new file and wrote code to {target_path}")
                action_status, action_message = "SUCCESS", f"Action 'CREATE_OR_APPEND' completed for {target_path}"
            
            elif parsed_inputs.edit_action == "ADD_PYTHON_IMPORTS":
                if not target_path.exists() or not target_path.is_file():
                    raise FileNotFoundError(f"Target file {target_path} does not exist for ADD_PYTHON_IMPORTS.")
                
                imports_str_from_input = parsed_inputs.imports_to_add or code_to_integrate_str
                if not imports_str_from_input:
                    action_status, action_message = "SUCCESS", f"No imports specified to add for {target_path} via dedicated field or primary code input."
                else:
                    if not target_path.exists(): 
                        with open(target_path, "w", encoding="utf-8") as f: f.write(imports_str_from_input + "\n\n# Rest of the code...")
                    else: 
                        with open(target_path, "a", encoding="utf-8") as f: f.write("\n" + imports_str_from_input)
                    self._logger_instance.info(f"(Mocked) Added Python imports to {target_path}")
                    action_status, action_message = "SUCCESS", f"Action 'ADD_PYTHON_IMPORTS' (mocked impl) completed for {target_path}"
            
            elif parsed_inputs.edit_action == "ADD_TO_CLICK_GROUP":
                with open(target_path, "a", encoding="utf-8") as f: f.write("\n\n" + code_to_integrate_str)
                self._logger_instance.info(f"(Mocked) Added Click group to {target_path}")
                action_status, action_message = "SUCCESS", f"Action 'ADD_TO_CLICK_GROUP' (mocked impl) completed for {target_path}"
            else:
                action_message = f"Unknown or not-yet-fully-adapted edit_action: {parsed_inputs.edit_action}"
                self._logger_instance.warning(action_message)
        
        except Exception as e:
            action_status = "FAILURE"
            action_message = str(e)
            self._logger_instance.error(f"Error during file system integration action '{parsed_inputs.edit_action}' on '{target_path}': {e}", exc_info=True)

        updated_doc_id: Optional[str] = None
        md5_hash: Optional[str] = None
        chroma_update_status: Literal["SUCCESS", "FAILURE_CHROMA_UPDATE"] = "FAILURE_CHROMA_UPDATE"
        chroma_update_message: str = "ChromaDB update skipped due to file operation failure or not applicable."

        if action_status == "SUCCESS":
            try:
                with open(target_path, 'r', encoding='utf-8') as f_read_final:
                    final_content = f_read_final.read()
                md5_hash = hashlib.md5(final_content.encode('utf-8')).hexdigest()
                
                doc_metadata = {
                    "source_agent": self.AGENT_ID,
                    "task_id": parsed_inputs.task_id,
                    "original_path": parsed_inputs.target_file_path,
                    "md5_hash": md5_hash,
                    "edit_action": parsed_inputs.edit_action,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }
                if parsed_inputs.generated_code_artifact_doc_id:
                    doc_metadata["derived_from_doc_id"] = parsed_inputs.generated_code_artifact_doc_id

                updated_doc_id = await self._pcma_agent.store_document_content(
                    project_id=parsed_inputs.project_id,
                    collection_name=LIVE_CODEBASE_COLLECTION,
                    document_content=final_content,
                    document_relative_path=parsed_inputs.target_file_path,
                    metadata=doc_metadata
                )

                chroma_update_status = "SUCCESS"
                chroma_update_message = f"Successfully updated {LIVE_CODEBASE_COLLECTION} for {target_path} (Doc ID: {updated_doc_id})."
                self._logger_instance.info(chroma_update_message)

            except Exception as e_chroma:
                chroma_update_message = f"Error during ChromaDB update for {target_path} into {LIVE_CODEBASE_COLLECTION}: {e_chroma}"
                self._logger_instance.error(chroma_update_message, exc_info=True)
        
        final_status = chroma_update_status if action_status == "SUCCESS" else action_status
        final_message = f"File Op: {action_message} Chroma Op: {chroma_update_message}"

        confidence = None
        if final_status == "SUCCESS":
            confidence = ConfidenceScore(value=0.8, level="High", method="RuleBasedSuccess", reasoning="All operations (file and Chroma update) completed.")
        elif action_status == "SUCCESS" and chroma_update_status == "FAILURE_CHROMA_UPDATE":
            confidence = ConfidenceScore(value=0.4, level="Low", method="PartialSuccess", reasoning="File operation succeeded but ChromaDB update failed.")
        else:
            confidence = ConfidenceScore(value=0.1, level="Low", method="OperationFailure", reasoning=f"Operation failed: {action_message}")

        return SmartCodeIntegrationOutput(
            task_id=parsed_inputs.task_id,
            status=final_status,
            message=final_message,
            modified_file_path=str(target_path) if target_path.exists() else None,
            backup_file_path=backup_file_path_str,
            updated_live_codebase_doc_id=updated_doc_id,
            md5_hash_of_integrated_file=md5_hash,
            confidence_score=confidence,
            error_message=action_message if final_status != "SUCCESS" else None
        )

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=SmartCodeIntegrationAgent_v1.AGENT_ID,
            name=SmartCodeIntegrationAgent_v1.AGENT_NAME,
            description=SmartCodeIntegrationAgent_v1.AGENT_DESCRIPTION,
            categories=[cat.value for cat in [SmartCodeIntegrationAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=SmartCodeIntegrationAgent_v1.VISIBILITY.value,
            capability_profile={
                "edit_action_support": ["APPEND", "CREATE_OR_APPEND", "ADD_TO_CLICK_GROUP", "ADD_PYTHON_IMPORTS", "REPLACE_FILE_CONTENT"],
                "language_support": ["python"],
                "pcma_collections_used": [
                    LIVE_CODEBASE_COLLECTION, 
                ]
            },
            input_schema=SmartCodeIntegrationInput.model_json_schema(),
            output_schema=SmartCodeIntegrationOutput.model_json_schema(),
            version=SmartCodeIntegrationAgent_v1.VERSION,
            metadata={
                "callable_fn_path": f"{SmartCodeIntegrationAgent_v1.__module__}.{SmartCodeIntegrationAgent_v1.__name__}"
            }
        )

async def main_test_integration():
    logging.basicConfig(level=logging.DEBUG)
    temp_dir = tempfile.mkdtemp()
    project_root = Path(temp_dir)
    mock_project_id = "test_smart_integration_proj_001"

    agent = SmartCodeIntegrationAgent_v1(
        pcma_agent=ProjectChromaManagerAgent_v1(project_root=project_root, project_id=mock_project_id),
        config={"project_root_dir_for_pcma_init": str(project_root)}
    )

    test_file_1 = project_root / "new_code.py"
    mock_code_artifact_id_1 = "gen_code_doc_abc123"

    inputs_1 = {
        "task_id": "task_create_new",
        "project_id": mock_project_id,
        "generated_code_artifact_doc_id": mock_code_artifact_id_1,
        "target_file_path": str(test_file_1),
        "edit_action": "REPLACE_FILE_CONTENT",
        "backup_original": False
    }
    output_1 = await agent.invoke_async(inputs_1)
    logger.info(f"Test 1 Output: {output_1.model_dump_json(indent=2)}")
    assert output_1.status == "SUCCESS"
    assert test_file_1.exists()
    assert "Hello from ChromaDB artifact!" in test_file_1.read_text()
    assert output_1.updated_live_codebase_doc_id is not None
    assert output_1.md5_hash_of_integrated_file is not None

    test_file_2 = project_root / "existing_script.py"
    test_file_2.write_text("def main():\n    pass\n")
    inputs_2 = {
        "task_id": "task_append_direct",
        "project_id": mock_project_id,
        "code_to_integrate_directly": "# Appended directly\nprint(\"Appended!\")",
        "target_file_path": str(test_file_2),
        "edit_action": "APPEND",
        "backup_original": True
    }
    output_2 = await agent.invoke_async(inputs_2)
    logger.info(f"Test 2 Output: {output_2.model_dump_json(indent=2)}")
    assert output_2.status == "SUCCESS"
    assert "Appended!" in test_file_2.read_text()
    assert output_2.backup_file_path is not None
    assert Path(output_2.backup_file_path).exists()
    assert output_2.updated_live_codebase_doc_id is not None

    shutil.rmtree(temp_dir)
    logger.info("SmartCodeIntegrationAgent tests completed and temp dir removed.")

if __name__ == "__main__":
    asyncio.run(main_test_integration()) 