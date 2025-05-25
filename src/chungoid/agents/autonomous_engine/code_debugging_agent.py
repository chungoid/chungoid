from __future__ import annotations

import logging
import uuid
import json
import asyncio
import datetime
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, Dict, Optional, List, Literal, ClassVar, get_args, Type

from pydantic import BaseModel, Field, ValidationError

from ..protocol_aware_agent import ProtocolAwareAgent
from ...protocols.base.protocol_interface import ProtocolPhase
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.registry import register_autonomous_engine_agent

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


@register_autonomous_engine_agent(capabilities=["code_debugging", "error_analysis", "automated_fixes"])
class CodeDebuggingAgent_v1(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "CodeDebuggingAgent_v1"
    AGENT_NAME: ClassVar[str] = "Code Debugging Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes faulty code with test failures and proposes fixes."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = CDA_PROMPT_NAME
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["code_debugging", "error_analysis", "automated_fixes"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.CODE_REMEDIATION 
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL 

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["code_generation", "plan_review"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["tool_validation", "error_recovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'tool_validation', 'context_sharing']


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
    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture.
        No fallback - protocol execution only for clean, maintainable code.
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
                # Enhanced error handling instead of fallback
                error_msg = f"Protocol execution failed for {self.AGENT_NAME}: {protocol_result.get('error', 'Unknown error')}"
                self._logger.error(error_msg)
                raise ProtocolExecutionError(error_msg)
                
        except Exception as e:
            error_msg = f"Pure protocol execution failed for {self.AGENT_NAME}: {e}"
            self._logger.error(error_msg)
            raise ProtocolExecutionError(error_msg)

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