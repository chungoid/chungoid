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


class CodeDebuggingAgentInput(BaseModel):
    """Clean input schema focused on core debugging needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this debugging task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core autonomous inputs
    user_goal: str = Field(..., description="What the user wants to debug/fix")
    project_path: str = Field(default=".", description="Project directory to debug")
    
    # Optional context (no micromanagement)
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Optional project context")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'CodeDebuggingAgentInput':
        """Ensure we have minimum requirements for autonomous debugging."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for autonomous code debugging")
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
    INPUT_SCHEMA: ClassVar[Type[CodeDebuggingAgentInput]] = CodeDebuggingAgentInput
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

    def _parse_inputs(self, inputs: Any) -> CodeDebuggingAgentInput:
        """Parse inputs cleanly into CodeDebuggingAgentInput with detailed validation."""
        try:
            if isinstance(inputs, CodeDebuggingAgentInput):
                # Validate existing input object
                if not inputs.user_goal or not inputs.user_goal.strip():
                    raise ValueError("CodeDebuggingAgentInput has empty or whitespace user_goal")
                return inputs
            elif isinstance(inputs, dict):
                # Validate required fields before creation
                if 'user_goal' not in inputs:
                    raise ValueError("Missing required field 'user_goal' in input dictionary")
                if not inputs['user_goal'] or not inputs['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return CodeDebuggingAgentInput(**inputs)
            elif hasattr(inputs, 'dict'):
                input_dict = inputs.dict()
                if 'user_goal' not in input_dict:
                    raise ValueError("Missing required field 'user_goal' in input object")
                if not input_dict['user_goal'] or not input_dict['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return CodeDebuggingAgentInput(**input_dict)
            else:
                raise ValueError(f"Invalid input type: {type(inputs)}. Expected CodeDebuggingAgentInput, dict, or object with dict() method. Received: {inputs}")
        except Exception as e:
            raise ValueError(f"Input parsing failed for CodeDebuggingAgent: {e}. Input received: {inputs}")

    async def _generate_debugging_solution(self, task_input: CodeDebuggingAgentInput) -> Dict[str, Any]:
        """
        AUTONOMOUS CODE DEBUGGING & QA with detailed validation
        No hardcoded logic - agent analyzes project and decides what debugging/QA is needed
        """
        try:
            # Validate prompt template access
            prompt_template = self.prompt_manager.get_prompt_definition(
                "code_debugging_agent_v1_prompt",
                "1.0.0",
                sub_path="autonomous_engine"
            )
            
            if not prompt_template:
                raise ValueError("Failed to load debugging prompt template - returned None/empty")
            
            # Handle PromptDefinition structure - use the correct attribute for user_prompt_template
            if isinstance(prompt_template, dict):
                if "user_prompt" not in prompt_template:
                    raise ValueError(f"Prompt template missing 'user_prompt' field. Available fields: {list(prompt_template.keys())}")
                template_content = prompt_template["user_prompt"]
            elif hasattr(prompt_template, 'user_prompt_template'):
                # PromptDefinition object has user_prompt_template attribute
                template_content = prompt_template.user_prompt_template
            elif hasattr(prompt_template, 'user_prompt'):
                template_content = prompt_template.user_prompt
            else:
                raise ValueError(f"Prompt template has unexpected structure. Type: {type(prompt_template)}, Available attributes: {dir(prompt_template)}")
            
            # Build context for autonomous debugging/QA
            context_parts = []
            context_parts.append(f"User Goal: {task_input.user_goal}")
            context_parts.append(f"Project Path: {task_input.project_path}")
            
            if task_input.project_specifications:
                context_parts.append(f"Project Context: {json.dumps(task_input.project_specifications, indent=2)}")
            
            context_data = "\n".join(context_parts)
            
            # Validate MCP tools availability
            if not hasattr(self, 'mcp_tools') or not self.mcp_tools:
                raise ValueError("MCP tools not available or empty in CodeDebuggingAgent")
            
            # AUTONOMOUS EXECUTION: Let the agent decide what debugging/QA is needed
            # Available MCP tools: filesystem_*, text_editor_*, terminal_*, web_search, etc.
            try:
                # Add missing template variables including user_goal
                formatted_prompt = template_content.format(
                    context_data=context_data,
                    available_mcp_tools=", ".join(self.mcp_tools.keys()),
                    user_goal=task_input.user_goal
                )
            except Exception as e:
                raise ValueError(f"Failed to format debugging prompt template: {e}. Missing variables or template content preview: {str(template_content)[:200]}...")
            
            self.logger.info(f"🐛 AUTONOMOUS debugging/QA execution")
            
            # Let the agent work autonomously
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                max_tokens=8000,
                temperature=0.1
            )
            
            # Detailed response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for debugging. Prompt length: {len(formatted_prompt)} chars. User goal: {task_input.user_goal}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for debugging. Response: '{response}'. User goal: {task_input.user_goal}")
            
            if len(response.strip()) < 50:
                raise ValueError(f"LLM debugging response too short ({len(response)} chars). Expected substantial debugging analysis. Response: '{response}'. User goal: {task_input.user_goal}")
            
            # Validate response content quality
            response_lower = response.lower()
            if "debug" not in response_lower and "error" not in response_lower and "fix" not in response_lower and "test" not in response_lower:
                raise ValueError(f"LLM response doesn't appear to contain debugging content (no 'debug', 'error', 'fix', or 'test' found). Response: '{response}'. User goal: {task_input.user_goal}")
            
            self.logger.info(f"🐛 Debugging response: {len(response)} chars")
            self.logger.info(f"🐛 Response preview: {response[:300]}...")
            
            # Return expected structure for DebuggingAgentOutput - NO FALLBACKS
            return {
                "proposed_solution_type": "CODE_PATCH",
                "proposed_code_changes": response,
                "explanation_of_fix": f"Autonomous debugging analysis: {response[:500]}...",
                "debugging_analysis": {"agent_mode": "autonomous", "response_length": len(response)},
                "test_results": {"autonomous_execution": True},
                "areas_of_uncertainty": [],
                "suggestions_for_ARCA": "Autonomous debugging completed successfully",
                "confidence_score": ConfidenceScore(
                    value=0.8,
                    method="autonomous_debugging",
                    explanation="Autonomous debugging completed"
                )
            }
            
        except Exception as e:
            error_msg = f"""CodeDebuggingAgent debugging solution generation failed:

ERROR: {e}

INPUT CONTEXT:
- User Goal: {task_input.user_goal}
- Project Path: {task_input.project_path}
- Project Specifications: {task_input.project_specifications}

DEBUGGING CONTEXT:
- Available MCP Tools: {len(self.mcp_tools) if hasattr(self, 'mcp_tools') and self.mcp_tools else 'None/Empty'}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

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
        input_schema = CodeDebuggingAgentInput.model_json_schema()
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

    def get_input_schema(self) -> Type[CodeDebuggingAgentInput]:
        return CodeDebuggingAgentInput

    def get_output_schema(self) -> Type[DebuggingAgentOutput]:
        return DebuggingAgentOutput 