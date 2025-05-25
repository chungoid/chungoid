"""
Plan Review Protocol

Protocol for comprehensive plan analysis and optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus, ProtocolTemplate

logger = logging.getLogger(__name__)


class PlanReviewProtocol(ProtocolInterface):
    """Protocol for comprehensive plan analysis and optimization."""
    
    def __init__(self):
        super().__init__()
        
    @property
    def name(self) -> str:
        """Protocol name."""
        return "Plan Review Protocol"
    
    @property
    def description(self) -> str:
        """Protocol description."""
        return "Comprehensive plan analysis and optimization"
    
    @property
    def total_estimated_time(self) -> float:
        """Total estimated time in hours."""
        return 4.0  # Sum of all phase time estimates
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        """Initialize all protocol phases."""
        return [
            ProtocolPhase(
                name="analysis",
                description="Analyze execution plan",
                time_box_hours=1.0,
                required_outputs=["plan_analysis", "risk_assessment"],
                validation_criteria=[
                    "plan_analysis.coverage >= 0.95",
                    "risk_assessment.level != 'unacceptable'"
                ],
                tools_required=["plan_analyzer", "risk_assessor"]
            ),
            ProtocolPhase(
                name="optimization",
                description="Optimize plan for efficiency",
                time_box_hours=1.5,
                required_outputs=["optimized_plan", "improvement_recommendations"],
                validation_criteria=[
                    "optimized_plan.efficiency > original_plan.efficiency",
                    "improvement_recommendations.count >= 3"
                ],
                tools_required=["plan_optimizer", "efficiency_analyzer"],
                dependencies=["analysis"]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate optimized plan",
                time_box_hours=1.0,
                required_outputs=["validated_plan", "approval_status"],
                validation_criteria=[
                    "validated_plan.feasible == true",
                    "approval_status == 'approved'"
                ],
                tools_required=["plan_validator", "feasibility_checker"],
                dependencies=["optimization"]
            ),
            ProtocolPhase(
                name="documentation",
                description="Document review findings",
                time_box_hours=0.5,
                required_outputs=["review_report", "execution_recommendations"],
                validation_criteria=[
                    "review_report EXISTS",
                    "execution_recommendations IS_NOT_EMPTY"
                ],
                tools_required=["documentation_generator", "report_formatter"],
                dependencies=["validation"]
            )
        ]
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates."""
        return {
            "analysis_prompt": ProtocolTemplate(
                name="analysis_prompt",
                description="Template for plan analysis",
                template_content="""
                Analyze the following execution plan:
                
                Execution Plan: [execution_plan]
                Success Criteria: [success_criteria]
                Constraints: [constraints]
                
                Please analyze:
                1. Plan completeness and coverage
                2. Potential risks and mitigation strategies
                3. Resource requirements and availability
                4. Timeline feasibility and dependencies
                """,
                variables=["execution_plan", "success_criteria", "constraints"]
            ),
            "optimization_prompt": ProtocolTemplate(
                name="optimization_prompt",
                description="Template for plan optimization",
                template_content="""
                Optimize the plan for better efficiency:
                
                Plan Analysis: [plan_analysis]
                Optimization Criteria: [optimization_criteria]
                
                Please optimize:
                1. Execution sequence and parallelization
                2. Resource allocation and utilization
                3. Risk reduction strategies
                4. Performance improvements
                """,
                variables=["plan_analysis", "optimization_criteria"]
            ),
            "validation_prompt": ProtocolTemplate(
                name="validation_prompt",
                description="Template for plan validation",
                template_content="""
                Validate the optimized plan:
                
                Optimized Plan: [optimized_plan]
                Validation Criteria: [validation_criteria]
                
                Please validate:
                1. Technical feasibility
                2. Resource constraints compliance
                3. Timeline achievability
                4. Risk acceptability
                """,
                variables=["optimized_plan", "validation_criteria"]
            ),
            "documentation_prompt": ProtocolTemplate(
                name="documentation_prompt",
                description="Template for review documentation",
                template_content="""
                Document the review findings:
                
                Validated Plan: [validated_plan]
                Review Findings: [review_findings]
                
                Please document:
                1. Comprehensive review report
                2. Execution recommendations
                3. Risk mitigation strategies
                4. Success metrics and monitoring
                """,
                variables=["validated_plan", "review_findings"]
            )
        }
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        if phase.name == "analysis":
            return True  # First phase is always ready
        elif phase.name == "optimization":
            analysis_phase = next((p for p in self.phases if p.name == "analysis"), None)
            return analysis_phase and analysis_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "validation":
            optimization_phase = next((p for p in self.phases if p.name == "optimization"), None)
            return optimization_phase and optimization_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "documentation":
            validation_phase = next((p for p in self.phases if p.name == "validation"), None)
            return validation_phase and validation_phase.status == PhaseStatus.COMPLETED
        
        return False
    
    def get_current_phase(self) -> Optional[ProtocolPhase]:
        """Get the current active phase."""
        for phase in self.phases:
            if phase.status in [PhaseStatus.IN_PROGRESS, PhaseStatus.REQUIRES_RETRY]:
                return phase
        
        # Return next phase that's ready
        for phase in self.phases:
            if phase.status == PhaseStatus.PENDING and self.is_phase_ready(phase):
                return phase
        
        return None
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get protocol execution progress summary."""
        completed_phases = [p for p in self.phases if p.status == PhaseStatus.COMPLETED]
        failed_phases = [p for p in self.phases if p.status == PhaseStatus.FAILED]
        
        return {
            "total_phases": len(self.phases),
            "completed_phases": len(completed_phases),
            "failed_phases": len(failed_phases),
            "progress_percentage": (len(completed_phases) / len(self.phases)) * 100,
            "current_phase": self.get_current_phase().name if self.get_current_phase() else None,
            "status": "completed" if len(completed_phases) == len(self.phases) else "in_progress"
        }
    
    def get_template(self, template_name: str, **variables) -> str:
        """Get protocol template with variable substitution."""
        templates = self.initialize_templates()
        template = templates.get(template_name, None)
        if template:
            content = template.template_content
            # Simple variable substitution using [variable] format
            for var_name, var_value in variables.items():
                placeholder = f"[{var_name}]"
                content = content.replace(placeholder, str(var_value))
            return content
        else:
            raise ValueError(f"Template '{template_name}' not found") 