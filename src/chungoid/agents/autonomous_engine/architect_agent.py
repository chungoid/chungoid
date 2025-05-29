"""
ArchitectAgent_v1: Clean, unified LLM-powered blueprint generation.

This agent generates architecture blueprints by:
1. Using unified discovery to understand project context
2. Using YAML prompt template with rich discovery data  
3. Letting the LLM create comprehensive blueprints with maximum intelligence

No legacy patterns, no redundant discovery, no hardcoded logic, no fallbacks.
Pure unified approach for maximum agentic intelligence.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type

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


class ArchitectAgentInput(BaseModel):
    """Clean input schema focused on core architectural needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # Core requirements
    user_goal: str = Field(..., description="What the user wants to build")
    project_path: str = Field(default=".", description="Project directory path")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications")
    intelligent_context: bool = Field(default=False, description="Whether intelligent specifications provided")
    
    @model_validator(mode='after')
    def validate_requirements(self) -> 'ArchitectAgentInput':
        """Ensure we have minimum requirements for architecture generation."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for architecture generation")
        return self


class ArchitectAgentOutput(BaseModel):
    """Clean output schema focused on architectural deliverables."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core deliverables
    blueprint_content: str = Field(..., description="Generated architecture blueprint in Markdown")
    architectural_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Key architectural decisions with rationale")
    technology_recommendations: Dict[str, Any] = Field(default_factory=dict, description="Recommended technology stack")
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="Identified risks and mitigations")
    
    # Metadata
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in the blueprint")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["architecture_design", "system_planning", "blueprint_generation"])
class ArchitectAgent_v1(UnifiedAgent):
    """
    Clean, unified architecture blueprint generation agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence.
    No legacy patterns, no fallbacks, no hardcoded logic.
    """
    
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Enhanced Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered architecture blueprint generation"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "4.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["architecture_design", "system_planning", "blueprint_generation"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ArchitectAgentInput]] = ArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[ArchitectAgentOutput]] = ArchitectAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_blueprint_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified approach")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Clean execution: Unified discovery + YAML template + LLM intelligence.
        Single iteration, maximum intelligence, no legacy patterns.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"Generating architecture blueprint: {task_input.user_goal}")

            # Generate blueprint using unified approach
            blueprint_result = await self._generate_architecture_blueprint(task_input)
            
            # Create clean output
            output = ArchitectAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                blueprint_content=blueprint_result["blueprint_content"],
                architectural_decisions=blueprint_result["architectural_decisions"],
                technology_recommendations=blueprint_result["technology_recommendations"],
                risk_assessments=blueprint_result["risk_assessments"],
                confidence_score=blueprint_result["confidence_score"],
                message=f"Generated comprehensive architecture blueprint for: {task_input.user_goal}"
            )
            
            return IterationResult(
                output=output,
                quality_score=blueprint_result["confidence_score"].value,
                tools_used=["unified_discovery", "yaml_template", "llm_intelligence"],
                protocol_used="unified_blueprint_generation"
            )
            
        except Exception as e:
            self.logger.error(f"Architecture blueprint generation failed: {e}")
            
            # Clean error handling
            error_output = ArchitectAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                blueprint_content="",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Architecture generation failed"
                ),
                message="Architecture blueprint generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="unified_blueprint_generation"
            )

    def _parse_inputs(self, inputs: Any) -> ArchitectAgentInput:
        """Parse inputs cleanly into ArchitectAgentInput."""
        if isinstance(inputs, ArchitectAgentInput):
            return inputs
        elif isinstance(inputs, dict):
            return ArchitectAgentInput(**inputs)
        elif hasattr(inputs, 'dict'):
            return ArchitectAgentInput(**inputs.dict())
        else:
            raise ValueError(f"Invalid input type: {type(inputs)}")

    async def _generate_architecture_blueprint(self, task_input: ArchitectAgentInput) -> Dict[str, Any]:
        """
        Generate architecture blueprint using unified discovery + YAML template.
        Pure unified approach - no legacy patterns.
        """
        try:
            # Get YAML template (no fallbacks)
            prompt_template = self.prompt_manager.get_prompt_definition(
                "architect_agent_v1_prompt",
                "0.2.0",
                sub_path="autonomous_engine"
            )
            
            # Unified discovery for intelligent context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "architecture"]
            )
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            # Build template variables for maximum LLM intelligence
            template_vars = {
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "project_id": task_input.project_id,
                "project_context": f"User Goal: {task_input.user_goal}",
                
                # Rich discovery data for intelligent decisions
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Intelligent context
                "intelligent_context": task_input.intelligent_context,
                "project_specifications": task_input.project_specifications or {}
            }
            
            # Render template
            formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            # Get system prompt if available
            system_prompt = getattr(prompt_template, 'system_prompt', None)
            
            # Call LLM with maximum intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse LLM response (expecting JSON from template)
            try:
                result = json.loads(response)
                
                return {
                    "blueprint_content": result.get("blueprint_markdown_content", ""),
                    "architectural_decisions": result.get("architectural_decisions", []),
                    "technology_recommendations": result.get("technology_recommendations", {}),
                    "risk_assessments": result.get("risk_assessments", []),
                    "confidence_score": ConfidenceScore(
                        value=result.get("confidence_score", {}).get("value", 0.8),
                        method=result.get("confidence_score", {}).get("method", "llm_self_assessment"),
                        explanation=result.get("confidence_score", {}).get("explanation", "Architecture blueprint generated successfully")
                    )
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM response not valid JSON: {e}")
                raise ValueError(f"LLM response parsing failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Blueprint generation failed: {e}")
            raise

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = ArchitectAgentInput.model_json_schema()
        output_schema = ArchitectAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.AGENT_DESCRIPTION,
            version=ArchitectAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[ArchitectAgent_v1.CATEGORY.value],
            visibility=ArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_intelligence": True,
                "clean_architecture": True,
                "no_fallbacks": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{ArchitectAgent_v1.__module__}.{ArchitectAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[ArchitectAgentInput]:
        return ArchitectAgentInput

    def get_output_schema(self) -> Type[ArchitectAgentOutput]:
        return ArchitectAgentOutput
