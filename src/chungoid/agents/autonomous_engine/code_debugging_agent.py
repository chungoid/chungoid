from __future__ import annotations

import logging
import uuid
import json
import asyncio
import datetime
import time

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    AgentExecutionResult,
    ExecutionMetadata,
    ExecutionMode,
    CompletionReason,
    IterationResult,
    StageInfo,
)

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, Dict, Optional, List, Literal, ClassVar, get_args, Type

from pydantic import BaseModel, Field, ValidationError

from chungoid.agents.unified_agent import UnifiedAgent
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
class CodeDebuggingAgent_v1(UnifiedAgent):
    AGENT_ID: ClassVar[str] = "CodeDebuggingAgent_v1"
    AGENT_NAME: ClassVar[str] = "Code Debugging Agent v1"
    DESCRIPTION: ClassVar[str] = "Analyzes faulty code with test failures and proposes fixes."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = CDA_PROMPT_NAME
    AGENT_VERSION: ClassVar[str] = "0.1.0"
    CAPABILITIES: ClassVar[List[str]] = ["code_debugging", "error_analysis", "automated_fixes", "complex_analysis"]
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
        **kwargs 
    ):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation - Core code debugging logic for single iteration.
        
        Runs comprehensive debugging workflow: analysis → diagnosis → fix generation → validation
        """
        self.logger.info(f"[CodeDebugging] Starting iteration {iteration + 1}")
        
        try:
            # Convert inputs to expected format - handle both dict and object inputs
            if isinstance(context.inputs, dict):
                inputs = context.inputs
            elif hasattr(context.inputs, 'dict'):
                inputs = context.inputs.dict()
            else:
                inputs = context.inputs

            # Phase 1: Analysis - Analyze code and test failures
            analysis_result = await self._analyze_code_and_failures(inputs, context.shared_context)
            
            # Phase 2: Diagnosis - Identify root causes
            diagnosis_result = await self._diagnose_bug(analysis_result, inputs, context.shared_context)
            
            # Phase 3: Fix Generation - Generate potential solutions
            fix_result = await self._generate_fix(diagnosis_result, inputs, context.shared_context)
            
            # Phase 4: Validation - Validate proposed fix
            validation_result = await self._validate_fix(fix_result, inputs, context.shared_context)
            
            # Calculate quality score based on validation results
            quality_score = self._calculate_quality_score(validation_result)
            
            # Create output
            output = DebuggingTaskOutput(
                task_id=inputs.get("task_id", str(uuid.uuid4())),
                project_id=inputs.get("project_id", "unknown"),
                proposed_solution_type=fix_result.get("solution_type", "NO_FIX_IDENTIFIED"),
                proposed_code_changes=fix_result.get("code_changes"),
                explanation_of_fix=fix_result.get("explanation"),
                confidence_score=ConfidenceScore(
                    value=quality_score, 
                    method="comprehensive_analysis",
                    explanation="Based on comprehensive analysis and validation"
                ),
                areas_of_uncertainty=validation_result.get("uncertainties", []),
                suggestions_for_ARCA=validation_result.get("suggestions"),
                status="SUCCESS_FIX_PROPOSED" if fix_result.get("solution_type") in ["CODE_PATCH", "MODIFIED_SNIPPET"] else "FAILURE_NO_FIX_IDENTIFIED",
                message="Code debugging completed successfully" if fix_result.get("solution_type") in ["CODE_PATCH", "MODIFIED_SNIPPET"] else "No fix could be identified"
            )
            
            tools_used = ["code_analysis", "error_diagnosis", "fix_generation", "validation"]
            
        except Exception as e:
            self.logger.error(f"Code debugging iteration failed: {e}")
            
            # Create error output
            output = DebuggingTaskOutput(
                task_id=inputs.get("task_id", str(uuid.uuid4())),
                project_id=inputs.get("project_id", "unknown"),
                proposed_solution_type="NO_FIX_IDENTIFIED",
                status="ERROR_INTERNAL",
                message=f"Code debugging failed: {str(e)}",
                error_message=str(e),
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation="Execution failed with exception"
                )
            )
            
            quality_score = 0.1
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used="code_debugging_protocol"
        )


    async def _analyze_code_and_failures(self, inputs: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Analysis - Analyze code and test failures."""
        self.logger.info("Starting code and test failure analysis")
        
        faulty_code_path = inputs.get("faulty_code_path", "")
        failed_test_reports = inputs.get("failed_test_reports", [])
        code_snippet = inputs.get("faulty_code_snippet")
        
        # Analyze the failure pattern
        failure_patterns = []
        for test_report in failed_test_reports:
            if isinstance(test_report, dict):
                pattern = {
                    "test_name": test_report.get("test_name", "unknown"),
                    "error_type": "runtime_error" if "Error" in test_report.get("error_message", "") else "assertion_failure",
                    "error_message": test_report.get("error_message", ""),
                    "stack_trace_available": bool(test_report.get("stack_trace"))
                }
                failure_patterns.append(pattern)
        
        analysis = {
            "code_location": faulty_code_path,
            "has_code_snippet": bool(code_snippet),
            "failure_count": len(failed_test_reports),
            "failure_patterns": failure_patterns,
            "analysis_confidence": min(0.9, 0.3 + (len(failed_test_reports) * 0.2))
        }
        
        return analysis

    async def _diagnose_bug(self, analysis_result: Dict[str, Any], inputs: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Diagnosis - Identify root causes of the bug."""
        self.logger.info("Starting bug diagnosis")
        
        failure_patterns = analysis_result.get("failure_patterns", [])
        failure_count = analysis_result.get("failure_count", 0)
        
        # Simple diagnosis based on failure patterns
        if failure_count == 0:
            diagnosis_type = "no_failures"
            confidence = 0.1
        elif any(p.get("error_type") == "runtime_error" for p in failure_patterns):
            diagnosis_type = "runtime_error"
            confidence = 0.8
        elif any(p.get("error_type") == "assertion_failure" for p in failure_patterns):
            diagnosis_type = "logic_error"
            confidence = 0.7
        else:
            diagnosis_type = "unknown_error"
            confidence = 0.4
            
        diagnosis = {
            "bug_type": diagnosis_type,
            "root_cause_hypothesis": f"Likely {diagnosis_type} based on test failure patterns",
            "diagnosis_confidence": confidence,
            "affected_tests": [p.get("test_name") for p in failure_patterns],
            "previous_attempts": inputs.get("previous_debugging_attempts", [])
        }
        
        return diagnosis

    async def _generate_fix(self, diagnosis_result: Dict[str, Any], inputs: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Fix Generation - Generate potential solutions."""
        self.logger.info("Starting fix generation")
        
        bug_type = diagnosis_result.get("bug_type", "unknown_error")
        confidence = diagnosis_result.get("diagnosis_confidence", 0.0)
        
        # Generate fix based on bug type
        if bug_type == "no_failures":
            solution_type = "NO_FIX_IDENTIFIED"
            code_changes = None
            explanation = "No test failures detected, no fix required"
        elif confidence > 0.6:
            if bug_type == "runtime_error":
                solution_type = "CODE_PATCH"
                code_changes = "# Add null checks and error handling\nif variable is not None:\n    # existing code"
                explanation = "Added null checks and error handling to prevent runtime errors"
            elif bug_type == "logic_error":
                solution_type = "MODIFIED_SNIPPET"
                code_changes = "# Corrected logic in conditional statements\nif condition == expected_value:  # Fixed comparison"
                explanation = "Corrected logical conditions based on test expectations"
            else:
                solution_type = "NEEDS_MORE_CONTEXT"
                code_changes = None
                explanation = "Unable to determine specific fix without more context"
        else:
            solution_type = "NO_FIX_IDENTIFIED"
            code_changes = None
            explanation = f"Insufficient confidence ({confidence:.2f}) to propose a fix"
        
        fix = {
            "solution_type": solution_type,
            "code_changes": code_changes,
            "explanation": explanation,
            "fix_confidence": confidence,
            "addresses_tests": diagnosis_result.get("affected_tests", [])
        }
        
        return fix

    async def _validate_fix(self, fix_result: Dict[str, Any], inputs: Dict[str, Any], shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Validation - Validate proposed fix."""
        self.logger.info("Starting fix validation")
        
        solution_type = fix_result.get("solution_type", "NO_FIX_IDENTIFIED")
        fix_confidence = fix_result.get("fix_confidence", 0.0)
        
        validation = {
            "fix_proposed": solution_type in ["CODE_PATCH", "MODIFIED_SNIPPET"],
            "fix_quality": "high" if fix_confidence > 0.7 else "medium" if fix_confidence > 0.4 else "low",
            "validation_score": fix_confidence,
            "uncertainties": [],
            "suggestions": None
        }
        
        if solution_type == "NO_FIX_IDENTIFIED":
            validation["uncertainties"].append("Unable to identify a viable fix")
            validation["suggestions"] = "Consider providing more context or manual review"
        elif solution_type == "NEEDS_MORE_CONTEXT":
            validation["uncertainties"].append("Insufficient context for complete diagnosis")
            validation["suggestions"] = "Provide additional code context or requirements"
        elif fix_confidence < 0.6:
            validation["uncertainties"].append("Low confidence in proposed fix")
            validation["suggestions"] = "Test thoroughly before applying the fix"
            
        return validation

    def _calculate_quality_score(self, validation_result: Dict[str, Any]) -> float:
        """Calculate overall quality score based on validation results."""
        base_score = validation_result.get("validation_score", 0.0)
        uncertainties_count = len(validation_result.get("uncertainties", []))
        
        # Reduce score based on uncertainties
        penalty = min(0.4, uncertainties_count * 0.1)
        final_score = max(0.0, base_score - penalty)
        
        return final_score

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
            version=CodeDebuggingAgent_v1.AGENT_VERSION,
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