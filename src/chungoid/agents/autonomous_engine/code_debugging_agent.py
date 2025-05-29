"""
CodeDebuggingAgent_v1: Clean, unified LLM-powered code debugging and testing.

This agent provides comprehensive code debugging and testing by:
1. Using unified discovery to understand project structure and codebase
2. Using YAML prompt template with rich discovery data
3. Letting the LLM make intelligent debugging decisions with maximum intelligence

No legacy patterns, no hardcoded phases, no complex tool orchestration.
Pure unified approach for maximum agentic debugging intelligence.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type, Literal

from pydantic import BaseModel, Field, model_validator

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.registry import register_autonomous_engine_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
)

logger = logging.getLogger(__name__)


class FailedTestReport(BaseModel):
    """Structured test failure information."""
    test_name: str
    error_message: str
    stack_trace: str
    expected_behavior_summary: Optional[str] = None


class PreviousDebuggingAttempt(BaseModel):
    """Previous debugging attempt information."""
    attempted_fix_summary: str
    outcome: str  # e.g., 'tests_still_failed', 'new_errors_introduced'


class DebuggingAgentInput(BaseModel):
    """Clean input schema focused on core debugging needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # Core requirements
    user_goal: str = Field(..., description="What the user wants debugged or fixed")
    project_path: str = Field(default=".", description="Project directory path")
    
    # Debugging context
    faulty_code_path: Optional[str] = Field(None, description="Path to the code file needing debugging")
    faulty_code_snippet: Optional[str] = Field(None, description="Specific code snippet if already localized")
    failed_test_reports: Optional[List[FailedTestReport]] = Field(None, description="List of test failure objects")
    relevant_loprd_requirements_ids: Optional[List[str]] = Field(None, description="LOPRD requirement IDs relevant to code")
    relevant_blueprint_section_ids: Optional[List[str]] = Field(None, description="Blueprint section IDs relevant to design")
    previous_debugging_attempts: Optional[List[PreviousDebuggingAttempt]] = Field(None, description="Previous fix attempts")
    max_iterations_for_this_call: Optional[int] = Field(None, description="Limit for debugging iterations")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications")
    intelligent_context: bool = Field(default=False, description="Whether intelligent specifications provided")
    
    @model_validator(mode='after')
    def validate_requirements(self) -> 'DebuggingAgentInput':
        """Ensure we have minimum requirements for debugging."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for code debugging")
        return self


class DebuggingAgentOutput(BaseModel):
    """Clean output schema focused on debugging deliverables."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core debugging deliverables
    proposed_solution_type: Literal["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"]
    proposed_code_changes: Optional[str] = Field(None, description="Actual patch or modified code snippet")
    explanation_of_fix: Optional[str] = Field(None, description="Explanation of diagnosed bug and proposed fix")
    
    # Quality insights
    debugging_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of debugging process")
    test_results: Dict[str, Any] = Field(default_factory=dict, description="Generated test results and coverage")
    areas_of_uncertainty: Optional[List[str]] = Field(None, description="Areas the agent is unsure about")
    suggestions_for_ARCA: Optional[str] = Field(None, description="Suggestions for broader improvements")
    
    # Metadata
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in debugging solution")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["code_debugging", "error_analysis", "automated_fixes"])
class CodeDebuggingAgent_v1(UnifiedAgent):
    """
    Clean, unified code debugging agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence for debugging.
    No legacy patterns, no hardcoded phases, no complex tool orchestration.
    """
    
    AGENT_ID: ClassVar[str] = "CodeDebuggingAgent_v1"
    AGENT_NAME: ClassVar[str] = "Code Debugging Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered code debugging and testing"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "code_debugging_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "3.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["code_debugging", "error_analysis", "automated_fixes"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.CODE_REMEDIATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL
    INPUT_SCHEMA: ClassVar[Type[DebuggingAgentInput]] = DebuggingAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[DebuggingAgentOutput]] = DebuggingAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_debugging"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified debugging")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Clean execution: Unified discovery + YAML template + LLM debugging intelligence.
        Single iteration, maximum intelligence, no hardcoded phases.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"Debugging code: {task_input.user_goal}")

            # Generate debugging solution using unified approach
            debugging_result = await self._generate_debugging_solution(task_input)
            
            # Create clean output
            output = DebuggingAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                proposed_solution_type=debugging_result["proposed_solution_type"],
                proposed_code_changes=debugging_result["proposed_code_changes"],
                explanation_of_fix=debugging_result["explanation_of_fix"],
                debugging_analysis=debugging_result["debugging_analysis"],
                test_results=debugging_result["test_results"],
                areas_of_uncertainty=debugging_result["areas_of_uncertainty"],
                suggestions_for_ARCA=debugging_result["suggestions_for_ARCA"],
                confidence_score=debugging_result["confidence_score"],
                message=f"Generated debugging solution for: {task_input.user_goal}"
            )
            
            return IterationResult(
                output=output,
                quality_score=debugging_result["confidence_score"].value,
                tools_used=["unified_discovery", "yaml_template", "llm_debugging"],
                protocol_used="unified_debugging"
            )
            
        except Exception as e:
            self.logger.error(f"Code debugging failed: {e}")
            
            # Clean error handling
            error_output = DebuggingAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                proposed_solution_type="NO_FIX_IDENTIFIED",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Debugging failed"
                ),
                message="Code debugging failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="unified_debugging"
            )

    def _parse_inputs(self, inputs: Any) -> DebuggingAgentInput:
        """Parse inputs cleanly into DebuggingAgentInput."""
        if isinstance(inputs, DebuggingAgentInput):
            return inputs
        elif isinstance(inputs, dict):
            return DebuggingAgentInput(**inputs)
        elif hasattr(inputs, 'dict'):
            return DebuggingAgentInput(**inputs.dict())
        else:
            raise ValueError(f"Invalid input type: {type(inputs)}")

    async def _generate_debugging_solution(self, task_input: DebuggingAgentInput) -> Dict[str, Any]:
        """
        Generate debugging solution using unified discovery + YAML template.
        Pure unified approach - no hardcoded phases or debugging logic.
        """
        try:
            # Get YAML template (no fallbacks)
            prompt_template = self.prompt_manager.get_prompt_definition(
                "code_debugging_agent_v1_prompt",
                "2.0.0",
                sub_path="autonomous_engine"
            )
            
            # Unified discovery for intelligent debugging context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "artifacts", "code_analysis", "testing"]
            )
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            # Build template variables for maximum LLM debugging intelligence
            template_vars = {
                # Original template variables (maintaining compatibility)
                "task_id": task_input.task_id,
                "project_id": task_input.project_id,
                "faulty_code_path": task_input.faulty_code_path or "discovered_from_codebase",
                "faulty_code_snippet": task_input.faulty_code_snippet,
                "failed_test_reports": [report.dict() for report in (task_input.failed_test_reports or [])],
                "relevant_loprd_requirements_ids": task_input.relevant_loprd_requirements_ids or [],
                "relevant_blueprint_section_ids": task_input.relevant_blueprint_section_ids or [],
                "previous_debugging_attempts": [attempt.dict() for attempt in (task_input.previous_debugging_attempts or [])],
                "max_iterations_for_this_call": task_input.max_iterations_for_this_call,
                "project_specifications": task_input.project_specifications or {},
                "intelligent_context": task_input.intelligent_context,
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                
                # Enhanced unified discovery variables for maximum intelligence
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Additional context for intelligent debugging
                "codebase_structure": discovery_results.get("structure", {}),
                "testing_framework": discovery_results.get("testing", {}),
                "dependencies": discovery_results.get("dependencies", {}),
                "patterns": discovery_results.get("patterns", {})
            }
            
            # Render template
            formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            # Get system prompt if available
            system_prompt = getattr(prompt_template, 'system_prompt', None)
            
            # Call LLM with maximum debugging intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse LLM response (expecting JSON from template)
            try:
                result = json.loads(response)
                
                # Transform template output to our clean schema
                return {
                    "proposed_solution_type": result.get("proposed_solution_type", "NO_FIX_IDENTIFIED"),
                    "proposed_code_changes": result.get("proposed_code_changes"),
                    "explanation_of_fix": result.get("explanation_of_fix"),
                    "debugging_analysis": {
                        "issues_identified": self._extract_issues_from_result(result, discovery_results),
                        "root_causes": self._extract_root_causes(result, discovery_results),
                        "fix_strategy": result.get("proposed_solution_type", "NO_FIX_IDENTIFIED"),
                        "testing_strategy": discovery_results.get("testing", {})
                    },
                    "test_results": {
                        "test_coverage": discovery_results.get("testing", {}).get("coverage", "unknown"),
                        "test_framework": discovery_results.get("testing", {}).get("framework", "detected"),
                        "generated_tests": "included_in_code_changes" if result.get("proposed_code_changes") else "none"
                    },
                    "areas_of_uncertainty": result.get("areas_of_uncertainty", []),
                    "suggestions_for_ARCA": result.get("suggestions_for_ARCA"),
                    "confidence_score": ConfidenceScore(
                        value=result.get("confidence_score_obj", {}).get("value", 0.7),
                        method=result.get("confidence_score_obj", {}).get("method", "llm_debugging_assessment"),
                        explanation=result.get("confidence_score_obj", {}).get("explanation", "Debugging solution generated with unified discovery")
                    )
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM response not valid JSON: {e}")
                raise ValueError(f"LLM response parsing failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Debugging solution generation failed: {e}")
            raise

    def _extract_issues_from_result(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> List[str]:
        """Extract identified issues from LLM result and discovery."""
        issues = []
        
        # Extract from explanation
        explanation = llm_result.get("explanation_of_fix", "")
        if "error" in explanation.lower():
            issues.append("Error identified in code logic")
        if "test" in explanation.lower():
            issues.append("Test-related issues found")
        
        # Add discovery-based issues
        if discovery_results.get("issues"):
            issues.extend(discovery_results["issues"])
        
        return issues

    def _extract_root_causes(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> List[str]:
        """Extract root causes from analysis."""
        causes = []
        
        solution_type = llm_result.get("proposed_solution_type", "")
        if solution_type == "CODE_PATCH":
            causes.append("Code logic error requiring patch")
        elif solution_type == "MODIFIED_SNIPPET":
            causes.append("Code structure requiring modification")
        
        # Add discovery-based causes
        patterns = discovery_results.get("patterns", {})
        if patterns.get("complexity") == "high":
            causes.append("High code complexity contributing to issues")
        
        return causes

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = DebuggingAgentInput.model_json_schema()
        output_schema = DebuggingAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=CodeDebuggingAgent_v1.AGENT_ID,
            name=CodeDebuggingAgent_v1.AGENT_NAME,
            description=CodeDebuggingAgent_v1.AGENT_DESCRIPTION,
            version=CodeDebuggingAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[CodeDebuggingAgent_v1.CATEGORY.value],
            visibility=CodeDebuggingAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_debugging": True,
                "clean_debugging": True,
                "no_hardcoded_logic": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{CodeDebuggingAgent_v1.__module__}.{CodeDebuggingAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[DebuggingAgentInput]:
        return DebuggingAgentInput

    def get_output_schema(self) -> Type[DebuggingAgentOutput]:
        return DebuggingAgentOutput 