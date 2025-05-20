from __future__ import annotations

import logging
import uuid
import json
import asyncio
import datetime
from typing import Any, Dict, Optional, List, Literal, ClassVar, get_args

from pydantic import BaseModel, Field, ValidationError

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)

CDA_PROMPT_NAME = "code_debugging_agent_v1_prompt"

# --- Input and Output Schemas based on Design Document --- #

class FailedTestReport(BaseModel):
    test_name: str
    error_message: str
    stack_trace: str
    expected_behavior_summary: Optional[str] = None

class PreviousDebuggingAttempt(BaseModel):
    attempted_fix_summary: str
    outcome: str # e.g., 'tests_still_failed', 'new_errors_introduced'

class DebuggingTaskInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this debugging task.")
    project_id: str = Field(..., description="Identifier for the current project.")
    faulty_code_path: str = Field(..., description="Path to the code file needing debugging.")
    faulty_code_snippet: Optional[str] = Field(None, description="(Optional) The specific code snippet if already localized.")
    failed_test_reports: List[FailedTestReport] = Field(..., description="List of structured test failure objects.")
    relevant_loprd_requirements_ids: List[str] = Field(..., description="List of LOPRD requirement IDs relevant to the faulty code.")
    relevant_blueprint_section_ids: Optional[List[str]] = Field(None, description="List of Blueprint section IDs relevant to the code's design.")
    previous_debugging_attempts: Optional[List[PreviousDebuggingAttempt]] = Field(None, description="(Optional) List of previous fixes attempted for this issue.")
    max_iterations_for_this_call: Optional[int] = Field(None, description="(Optional) A limit set by ARCA for this specific debugging invocation's internal reasoning.")

class DebuggingTaskOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    proposed_solution_type: Literal["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"]
    proposed_code_changes: Optional[str] = Field(None, description="The actual patch (e.g., diff format) or the full modified code snippet. Null if no fix identified.")
    explanation_of_fix: Optional[str] = Field(None, description="LLM-generated explanation of the diagnosed bug and the proposed fix. Null if no fix identified.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Likelihood the proposed fix resolves the issue.")
    areas_of_uncertainty: Optional[List[str]] = Field(None, description="(Optional) Any parts of the code, problem, or context the agent is unsure about.")
    suggestions_for_ARCA: Optional[str] = Field(None, description="(Optional) E.g., 'Consider broader refactoring...'")
    status: Literal["SUCCESS_FIX_PROPOSED", "FAILURE_NO_FIX_IDENTIFIED", "FAILURE_NEEDS_CLARIFICATION", "ERROR_INTERNAL", "FAILURE_LLM", "FAILURE_LLM_OUTPUT_PARSING", "FAILURE_PROMPT_RENDERING"]
    message: str = Field(..., description="A message detailing the outcome.")
    error_message: Optional[str] = Field(None, description="Error message if status indicates failure.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging for analysis.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")


class CodeDebuggingAgent_v1(BaseAgent[DebuggingTaskInput, DebuggingTaskOutput]):
    AGENT_ID: ClassVar[str] = "CodeDebuggingAgent_v1"
    AGENT_NAME: ClassVar[str] = "Code Debugging Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes faulty code with test failures and proposes fixes."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = CDA_PROMPT_NAME
    VERSION: ClassVar[str] = "0.1.0"
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.CODE_REMEDIATION 
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL 

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger

    def __init__(
        self, 
        llm_provider: LLMProvider, 
        prompt_manager: PromptManager, 
        system_context: Optional[Dict[str, Any]] = None,
        **kwargs 
    ):
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            **kwargs
        )
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        if system_context and "logger" in system_context:
            self._logger = system_context["logger"]
        else:
            self._logger = logging.getLogger(f"{__name__}.{self.AGENT_ID}")

        if not self._llm_provider:
            self._logger.error("LLMProvider not provided during initialization.")
            raise ValueError("LLMProvider is required for CodeDebuggingAgent_v1.")
        if not self._prompt_manager:
            self._logger.error("PromptManager not provided during initialization.")
            raise ValueError("PromptManager is required for CodeDebuggingAgent_v1.")
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized.")

    async def invoke_async(
        self,
        task_input: DebuggingTaskInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> DebuggingTaskOutput:
        llm_provider = self._llm_provider
        logger_instance = self._logger
        
        if full_context:
            if "llm_provider" in full_context and full_context["llm_provider"] != self._llm_provider:
                llm_provider = full_context["llm_provider"]
                logger_instance.info("Using LLMProvider from full_context.")
            if "system_context" in full_context and "logger" in full_context["system_context"] and \
               full_context["system_context"]["logger"] != self._logger:
                logger_instance = full_context["system_context"]["logger"]
                logger_instance.info("Using Logger from full_context.")
        
        logger_instance.info(f"{self.AGENT_ID} invoked for task {task_input.task_id} in project {task_input.project_id} for file {task_input.faulty_code_path}.")
        llm_response_str = None 

        try:
            failed_reports_list = [report.model_dump() for report in task_input.failed_test_reports]
            previous_attempts_list = [attempt.model_dump() for attempt in task_input.previous_debugging_attempts] if task_input.previous_debugging_attempts else []

            prompt_render_data = {
                "faulty_code_path": task_input.faulty_code_path,
                "faulty_code_snippet": task_input.faulty_code_snippet or "", 
                "failed_test_reports_str": json.dumps(failed_reports_list, indent=2),
                "relevant_loprd_requirements_ids_str": ", ".join(task_input.relevant_loprd_requirements_ids),
                "relevant_blueprint_section_ids_str": ", ".join(task_input.relevant_blueprint_section_ids) if task_input.relevant_blueprint_section_ids else "",
                "previous_debugging_attempts_str": json.dumps(previous_attempts_list, indent=2) if previous_attempts_list else "",
                "max_iterations_for_this_call": task_input.max_iterations_for_this_call
            }

            logger_instance.debug(f"Prompt render data prepared.")
            
            llm_response_str = await llm_provider.generate_text_async_with_prompt_manager(
                prompt_name=self.PROMPT_TEMPLATE_NAME,
                prompt_version=self.VERSION, 
                prompt_render_data=prompt_render_data,
                prompt_sub_path="autonomous_engine", 
                temperature=0.3, 
                model_id=None, 
            )

            if not llm_response_str or not isinstance(llm_response_str, str) or not llm_response_str.strip():
                 raise ValueError("LLM returned empty or non-string response where a JSON string was expected.")

            llm_output_data = json.loads(llm_response_str)
            logger_instance.info("Successfully received and parsed debugging proposal from LLM.")

            confidence_score_val = None
            if "confidence_score_obj" in llm_output_data and llm_output_data["confidence_score_obj"] is not None:
                try:
                    confidence_score_val = ConfidenceScore(**llm_output_data["confidence_score_obj"])
                except ValidationError as ve_conf:
                    logger_instance.warning(f"LLM provided confidence_score_obj that failed Pydantic validation: {ve_conf}. Confidence will be None.", exc_info=True)
            
            required_llm_fields = ["proposed_solution_type"]
            for field in required_llm_fields:
                if field not in llm_output_data:
                    raise ValueError(f"LLM JSON output missing required field: {field}. Got: {llm_response_str[:500]}")

            output_status_str = "SUCCESS_FIX_PROPOSED"
            if llm_output_data["proposed_solution_type"] == "NO_FIX_IDENTIFIED":
                output_status_str = "FAILURE_NO_FIX_IDENTIFIED"
            elif llm_output_data["proposed_solution_type"] == "NEEDS_MORE_CONTEXT":
                output_status_str = "FAILURE_NEEDS_CLARIFICATION"
            
            valid_statuses = get_args(DebuggingTaskOutput.__annotations__['status'])
            if output_status_str not in valid_statuses: # type: ignore
                 logger_instance.error(f"Derived status '{output_status_str}' is not a valid status literal. Defaulting to ERROR_INTERNAL.")
                 output_status_str = "ERROR_INTERNAL"

            return DebuggingTaskOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                proposed_solution_type=llm_output_data["proposed_solution_type"],
                proposed_code_changes=llm_output_data.get("proposed_code_changes"),
                explanation_of_fix=llm_output_data.get("explanation_of_fix"),
                confidence_score=confidence_score_val,
                areas_of_uncertainty=llm_output_data.get("areas_of_uncertainty"),
                suggestions_for_ARCA=llm_output_data.get("suggestions_for_ARCA"),
                status=output_status_str, # type: ignore
                message=f"Debugging analysis completed. Solution type: {llm_output_data['proposed_solution_type']}",
                llm_full_response=llm_response_str,
            )

        except PromptRenderError as e_prompt:
            logger_instance.error(f"Prompt rendering failed for {self.PROMPT_TEMPLATE_NAME}: {e_prompt}", exc_info=True)
            return DebuggingTaskOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_PROMPT_RENDERING", message=str(e_prompt), error_message=str(e_prompt), llm_full_response=llm_response_str)
        except json.JSONDecodeError as e_json:
            logger_instance.error(f"Failed to decode LLM JSON response: {e_json}. Response: {llm_response_str[:500] if llm_response_str else 'N/A'}...", exc_info=True)
            return DebuggingTaskOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM_OUTPUT_PARSING", message=f"LLM response not valid JSON: {e_json}", error_message=str(e_json), llm_full_response=llm_response_str)
        except ValidationError as e_val: 
            logger_instance.error(f"LLM output failed Pydantic validation during parsing: {e_val}. Response: {llm_response_str[:500] if llm_response_str else 'N/A'}...", exc_info=True)
            return DebuggingTaskOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM_OUTPUT_PARSING", message=f"LLM output did not match expected structure: {e_val}", error_message=str(e_val), llm_full_response=llm_response_str)
        except ValueError as e_val_custom:
             logger_instance.error(f"Error processing LLM output: {e_val_custom}. Response: {llm_response_str[:500] if llm_response_str else 'N/A'}...", exc_info=True)
             return DebuggingTaskOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM_OUTPUT_PARSING", message=str(e_val_custom), error_message=str(e_val_custom), llm_full_response=llm_response_str)
        except Exception as e_gen: 
            logger_instance.error(f"General error during CodeDebuggingAgent execution: {e_gen}", exc_info=True)
            return DebuggingTaskOutput(task_id=task_input.task_id, project_id=task_input.project_id, status="FAILURE_LLM", message=str(e_gen), error_message=str(e_gen), llm_full_response=llm_response_str)

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = DebuggingTaskInput.model_json_schema()
        output_schema = DebuggingTaskOutput.model_json_schema()
        
        llm_direct_output_schema = {
            "type": "object",
            "properties": {
                "proposed_solution_type": {"type": "string", "enum": ["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"]},
                "proposed_code_changes": {"type": ["string", "null"]},
                "explanation_of_fix": {"type": ["string", "null"]},
                "confidence_score_obj": { 
                    "type": "object",
                    "properties": {
                        "value": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        "level": {"type": ["string", "null"], "enum": ["Low", "Medium", "High", None]},
                        "explanation": {"type": ["string", "null"]},
                        "method": {"type": ["string", "null"]}
                    },
                    "required": ["value"]
                },
                "areas_of_uncertainty": {"type": ["array", "null"], "items": {"type": "string"}},
                "suggestions_for_ARCA": {"type": ["string", "null"]}
            },
            "required": ["proposed_solution_type", "confidence_score_obj"]
        }

        return AgentCard(
            agent_id=CodeDebuggingAgent_v1.AGENT_ID,
            name=CodeDebuggingAgent_v1.AGENT_NAME,
            description=CodeDebuggingAgent_v1.DESCRIPTION,
            version=CodeDebuggingAgent_v1.VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            llm_direct_output_schema=llm_direct_output_schema,
            categories=[CodeDebuggingAgent_v1.CATEGORY.value],
            visibility=CodeDebuggingAgent_v1.VISIBILITY.value,
            capability_profile={
                "analyzes_code_and_tests": True,
                "proposes_code_fixes": True,
                "diagnoses_bugs": True
            },
            metadata={
                 "callable_fn_path": f"{CodeDebuggingAgent_v1.__module__}.{CodeDebuggingAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[DebuggingTaskInput]:
        return DebuggingTaskInput

    def get_output_schema(self) -> Type[DebuggingTaskOutput]:
        return DebuggingTaskOutput 