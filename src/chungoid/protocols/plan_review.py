"""
Plan Review Protocol

Protocol for comprehensive plan analysis and optimization.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus

logger = logging.getLogger(__name__)


class PlanReviewProtocol(ProtocolInterface):
    """Protocol for comprehensive plan analysis and optimization."""
    
    def __init__(self):
        super().__init__(
            protocol_id="plan_review",
            name="Plan Review Protocol",
            description="Comprehensive plan analysis and optimization"
        )
        
        # Define protocol phases
        self.phases = [
            ProtocolPhase(
                name="analysis",
                description="Analyze execution plan",
                required_inputs=["execution_plan", "success_criteria", "constraints"],
                required_outputs=["plan_analysis", "risk_assessment"],
                validation_criteria=[
                    "plan_analysis.coverage >= 0.95",
                    "risk_assessment.level != 'unacceptable'"
                ]
            ),
            ProtocolPhase(
                name="optimization",
                description="Optimize plan for efficiency",
                required_inputs=["plan_analysis", "optimization_criteria"],
                required_outputs=["optimized_plan", "improvement_recommendations"],
                validation_criteria=[
                    "optimized_plan.efficiency > original_plan.efficiency",
                    "improvement_recommendations.count >= 3"
                ]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate optimized plan",
                required_inputs=["optimized_plan", "validation_criteria"],
                required_outputs=["validated_plan", "approval_status"],
                validation_criteria=[
                    "validated_plan.feasible == true",
                    "approval_status == 'approved'"
                ]
            ),
            ProtocolPhase(
                name="documentation",
                description="Document review findings",
                required_inputs=["validated_plan", "review_findings"],
                required_outputs=["review_report", "execution_recommendations"],
                validation_criteria=[
                    "review_report EXISTS",
                    "execution_recommendations IS_NOT_EMPTY"
                ]
            )
        ]
    
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
        templates = {
            "analysis_prompt": """
            Analyze the following execution plan:
            
            Execution Plan: {execution_plan}
            Success Criteria: {success_criteria}
            Constraints: {constraints}
            
            Please analyze:
            1. Plan completeness and coverage
            2. Potential risks and mitigation strategies
            3. Resource requirements and availability
            4. Timeline feasibility and dependencies
            """,
            "optimization_prompt": """
            Optimize the plan for better efficiency:
            
            Plan Analysis: {plan_analysis}
            Optimization Criteria: {optimization_criteria}
            
            Please optimize:
            1. Execution sequence and parallelization
            2. Resource allocation and utilization
            3. Risk reduction strategies
            4. Performance improvements
            """,
            "validation_prompt": """
            Validate the optimized plan:
            
            Optimized Plan: {optimized_plan}
            Validation Criteria: {validation_criteria}
            
            Please validate:
            1. Technical feasibility
            2. Resource constraints compliance
            3. Timeline achievability
            4. Risk acceptability
            """,
            "documentation_prompt": """
            Document the review findings:
            
            Validated Plan: {validated_plan}
            Review Findings: {review_findings}
            
            Please document:
            1. Comprehensive review report
            2. Execution recommendations
            3. Risk mitigation strategies
            4. Success metrics and monitoring
            """
        }
        
        template = templates.get(template_name, "")
        return template.format(**variables) 