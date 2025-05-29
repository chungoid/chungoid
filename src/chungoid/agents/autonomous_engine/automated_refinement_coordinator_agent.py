"""
AutomatedRefinementCoordinatorAgent_v1: Clean, unified LLM-powered coordination.

This agent coordinates project refinement by:
1. Using unified discovery to understand project state and progress
2. Using YAML prompt template with rich discovery data
3. Letting the LLM make intelligent coordination decisions with maximum intelligence

No legacy patterns, no hardcoded decision logic, no complex phases.
Pure unified approach for maximum agentic coordination intelligence.
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


class CoordinatorAgentInput(BaseModel):
    """Clean input schema focused on core coordination needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # Core requirements
    user_goal: str = Field(..., description="What the user wants to achieve")
    project_path: str = Field(default=".", description="Project directory path")
    
    # Coordination context
    current_cycle_id: Optional[str] = Field(None, description="Current refinement cycle identifier")
    coordination_focus: Optional[str] = Field(None, description="Specific area to focus coordination on")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications")
    intelligent_context: bool = Field(default=False, description="Whether intelligent specifications provided")
    
    @model_validator(mode='after')
    def validate_requirements(self) -> 'CoordinatorAgentInput':
        """Ensure we have minimum requirements for coordination."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for coordination")
        return self


class CoordinatorAgentOutput(BaseModel):
    """Clean output schema focused on coordination decisions."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core coordination deliverables
    coordination_decision: str = Field(..., description="Primary coordination decision")
    coordination_reasoning: str = Field(..., description="LLM reasoning for the coordination decision")
    next_actions: List[Dict[str, Any]] = Field(default_factory=list, description="Recommended next actions")
    project_status_assessment: Dict[str, Any] = Field(default_factory=dict, description="Assessment of overall project status")
    
    # Quality and refinement insights
    quality_gates_status: Dict[str, Any] = Field(default_factory=dict, description="Status of quality gates")
    refinement_recommendations: List[Dict[str, Any]] = Field(default_factory=list, description="Specific refinement recommendations")
    
    # Metadata
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in coordination decision")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["autonomous_coordination", "quality_gates", "refinement_orchestration"])
class AutomatedRefinementCoordinatorAgent_v1(UnifiedAgent):
    """
    Clean, unified project coordination agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence for coordination.
    No legacy patterns, no hardcoded logic, no complex phases.
    """
    
    AGENT_ID: ClassVar[str] = "AutomatedRefinementCoordinatorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Automated Refinement Coordinator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered project coordination and refinement orchestration"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "automated_refinement_coordinator_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "2.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["autonomous_coordination", "quality_gates", "refinement_orchestration"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.AUTONOMOUS_COORDINATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL
    INPUT_SCHEMA: ClassVar[Type[CoordinatorAgentInput]] = CoordinatorAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[CoordinatorAgentOutput]] = CoordinatorAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_coordination"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified coordination")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Clean execution: Unified discovery + YAML template + LLM coordination intelligence.
        Single iteration, maximum intelligence, no hardcoded phases.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"Coordinating refinement: {task_input.user_goal}")

            # Generate coordination decision using unified approach
            coordination_result = await self._generate_coordination_decision(task_input)
            
            # Create clean output
            output = CoordinatorAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                coordination_decision=coordination_result["coordination_decision"],
                coordination_reasoning=coordination_result["coordination_reasoning"],
                next_actions=coordination_result["next_actions"],
                project_status_assessment=coordination_result["project_status_assessment"],
                quality_gates_status=coordination_result["quality_gates_status"],
                refinement_recommendations=coordination_result["refinement_recommendations"],
                confidence_score=coordination_result["confidence_score"],
                message=f"Generated coordination decision for: {task_input.user_goal}"
            )
            
            return IterationResult(
                output=output,
                quality_score=coordination_result["confidence_score"].value,
                tools_used=["unified_discovery", "yaml_template", "llm_coordination"],
                protocol_used="unified_coordination"
            )
            
        except Exception as e:
            self.logger.error(f"Coordination decision generation failed: {e}")
            
            # Clean error handling
            error_output = CoordinatorAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                coordination_decision="ERROR",
                coordination_reasoning="Coordination failed",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Coordination generation failed"
                ),
                message="Coordination decision generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="unified_coordination"
            )

    def _parse_inputs(self, inputs: Any) -> CoordinatorAgentInput:
        """Parse inputs cleanly into CoordinatorAgentInput."""
        if isinstance(inputs, CoordinatorAgentInput):
            return inputs
        elif isinstance(inputs, dict):
            return CoordinatorAgentInput(**inputs)
        elif hasattr(inputs, 'dict'):
            return CoordinatorAgentInput(**inputs.dict())
        else:
            raise ValueError(f"Invalid input type: {type(inputs)}")

    async def _generate_coordination_decision(self, task_input: CoordinatorAgentInput) -> Dict[str, Any]:
        """
        Generate coordination decision using unified discovery + YAML template.
        Pure unified approach - no hardcoded phases or decision logic.
        """
        try:
            # Get YAML template (no fallbacks)
            prompt_template = self.prompt_manager.get_prompt_definition(
                "automated_refinement_coordinator_agent_v1_prompt",
                "0.2.0",
                sub_path="autonomous_engine"
            )
            
            # Unified discovery for intelligent coordination context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "artifacts", "progress"]
            )
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            # Build project state from discovery
            project_state = await self._build_project_state_from_discovery(
                task_input, discovery_results, technology_context
            )
            
            # Build template variables for maximum LLM coordination intelligence
            template_vars = {
                "project_id": task_input.project_id,
                "current_cycle_id": task_input.current_cycle_id or "initial_cycle",
                
                # Template expects these specific variables
                "project_state_v2_json": json.dumps(project_state, indent=2),
                "recent_agent_outputs_json": json.dumps(discovery_results.get("agent_outputs", []), indent=2),
                "key_project_artifact_ids_json": json.dumps(discovery_results.get("artifacts", {}), indent=2),
                
                # Rich discovery data for intelligent decisions
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Coordination context
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "coordination_focus": task_input.coordination_focus or "overall_progress",
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
            
            # Call LLM with maximum coordination intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=3000
            )
            
            # Parse LLM response (expecting JSON from template)
            try:
                result = json.loads(response)
                
                # Transform template output to our clean schema
                return {
                    "coordination_decision": result.get("decision_outcome", "PROCEED_TO_NEXT_STAGE"),
                    "coordination_reasoning": result.get("decision_rationale", ""),
                    "next_actions": self._extract_next_actions(result),
                    "project_status_assessment": {
                        "current_status": project_state.get("overall_status", "active"),
                        "recommended_status": result.get("next_overall_project_status", "active_development"),
                        "confidence": result.get("decision_confidence", {})
                    },
                    "quality_gates_status": discovery_results.get("quality_assessment", {}),
                    "refinement_recommendations": result.get("feedback_for_refinement_agents", []),
                    "confidence_score": ConfidenceScore(
                        value=result.get("decision_confidence", {}).get("value", 0.8),
                        method=result.get("decision_confidence", {}).get("level", "llm_coordination"),
                        explanation=result.get("decision_confidence", {}).get("reasoning", "Coordination decision generated successfully")
                    )
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM response not valid JSON: {e}")
                raise ValueError(f"LLM response parsing failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Coordination decision generation failed: {e}")
            raise

    async def _build_project_state_from_discovery(
        self, 
        task_input: CoordinatorAgentInput, 
        discovery_results: Dict[str, Any],
        technology_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build project state from unified discovery results."""
        
        # Extract key information from discovery
        structure = discovery_results.get("structure", {})
        patterns = discovery_results.get("patterns", {})
        requirements = discovery_results.get("requirements", {})
        artifacts = discovery_results.get("artifacts", {})
        
        # Build comprehensive project state
        project_state = {
            "project_id": task_input.project_id,
            "user_goal": task_input.user_goal,
            "project_path": task_input.project_path,
            "current_cycle": task_input.current_cycle_id or "initial",
            "overall_status": "active_development",  # Can be derived from discovery
            
            # Discovery-based state
            "structure_analysis": structure,
            "technology_stack": technology_context,
            "identified_patterns": patterns,
            "requirements_analysis": requirements,
            "available_artifacts": artifacts,
            
            # Quality assessment from discovery
            "quality_metrics": discovery_results.get("quality_assessment", {}),
            "completion_status": discovery_results.get("progress", {}),
            "risk_factors": discovery_results.get("risks", []),
            
            # Coordination metadata
            "coordination_focus": task_input.coordination_focus,
            "intelligent_context": task_input.intelligent_context,
            "last_updated": discovery_results.get("timestamp", "")
        }
        
        return project_state

    def _extract_next_actions(self, llm_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract and format next actions from LLM decision result."""
        actions = []
        
        # Extract from feedback_for_refinement_agents
        refinement_feedback = llm_result.get("feedback_for_refinement_agents", [])
        for feedback in refinement_feedback:
            actions.append({
                "action_type": "refinement",
                "target_agent": feedback.get("target_agent_id", ""),
                "target_artifact": feedback.get("target_artifact_id_to_refine", ""),
                "directives": feedback.get("refinement_directives", ""),
                "priority": "high"
            })
        
        # Extract from decision outcome
        decision = llm_result.get("decision_outcome", "")
        if decision == "REQUEST_HUMAN_REVIEW":
            actions.append({
                "action_type": "human_review",
                "description": llm_result.get("issues_for_human_review_summary", ""),
                "priority": "urgent"
            })
        elif decision == "PROCEED_TO_NEXT_STAGE":
            actions.append({
                "action_type": "proceed",
                "description": "Continue to next development stage",
                "priority": "normal"
            })
        elif decision == "MARK_CYCLE_COMPLETE_SUCCESS":
            actions.append({
                "action_type": "complete",
                "description": "Mark cycle as successfully completed",
                "priority": "normal"
            })
        
        return actions

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = CoordinatorAgentInput.model_json_schema()
        output_schema = CoordinatorAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=AutomatedRefinementCoordinatorAgent_v1.AGENT_ID,
            name=AutomatedRefinementCoordinatorAgent_v1.AGENT_NAME,
            description=AutomatedRefinementCoordinatorAgent_v1.AGENT_DESCRIPTION,
            version=AutomatedRefinementCoordinatorAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[AutomatedRefinementCoordinatorAgent_v1.CATEGORY.value],
            visibility=AutomatedRefinementCoordinatorAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_coordination": True,
                "clean_coordination": True,
                "no_hardcoded_logic": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{AutomatedRefinementCoordinatorAgent_v1.__module__}.{AutomatedRefinementCoordinatorAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[CoordinatorAgentInput]:
        return CoordinatorAgentInput

    def get_output_schema(self) -> Type[CoordinatorAgentOutput]:
        return CoordinatorAgentOutput 