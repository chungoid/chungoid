"""
Stakeholder Analysis Protocol

Protocol for comprehensive stakeholder identification and analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus

logger = logging.getLogger(__name__)


class StakeholderAnalysisProtocol(ProtocolInterface):
    """Protocol for comprehensive stakeholder analysis."""
    
    def __init__(self):
        super().__init__(
            protocol_id="stakeholder_analysis",
            name="Stakeholder Analysis Protocol",
            description="Comprehensive stakeholder identification and analysis"
        )
        
        # Define protocol phases
        self.phases = [
            ProtocolPhase(
                name="identification",
                description="Identify key stakeholders",
                required_inputs=["project_context", "requirements"],
                required_outputs=["stakeholder_list", "stakeholder_categories"],
                validation_criteria=[
                    "stakeholder_list.count >= 3",
                    "stakeholder_categories IS_NOT_EMPTY"
                ]
            ),
            ProtocolPhase(
                name="analysis",
                description="Analyze stakeholder needs and influence",
                required_inputs=["stakeholder_list", "project_goals"],
                required_outputs=["stakeholder_needs", "influence_matrix"],
                validation_criteria=[
                    "stakeholder_needs IS_NOT_EMPTY",
                    "influence_matrix.analyzed == true"
                ]
            ),
            ProtocolPhase(
                name="prioritization",
                description="Prioritize stakeholder requirements",
                required_inputs=["stakeholder_needs", "business_constraints"],
                required_outputs=["prioritized_requirements", "stakeholder_map"],
                validation_criteria=[
                    "prioritized_requirements.count > 0",
                    "stakeholder_map EXISTS"
                ]
            ),
            ProtocolPhase(
                name="engagement_planning",
                description="Plan stakeholder engagement strategy",
                required_inputs=["stakeholder_map", "project_timeline"],
                required_outputs=["engagement_plan", "communication_strategy"],
                validation_criteria=[
                    "engagement_plan EXISTS",
                    "communication_strategy.defined == true"
                ]
            )
        ]
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        if phase.name == "identification":
            return True  # First phase is always ready
        elif phase.name == "analysis":
            identification_phase = next((p for p in self.phases if p.name == "identification"), None)
            return identification_phase and identification_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "prioritization":
            analysis_phase = next((p for p in self.phases if p.name == "analysis"), None)
            return analysis_phase and analysis_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "engagement_planning":
            prioritization_phase = next((p for p in self.phases if p.name == "prioritization"), None)
            return prioritization_phase and prioritization_phase.status == PhaseStatus.COMPLETED
        
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
            "identification_prompt": """
            Identify stakeholders for the following project:
            
            Project Context: {project_context}
            Requirements: {requirements}
            
            Please identify:
            1. Primary stakeholders (direct impact)
            2. Secondary stakeholders (indirect impact)
            3. Key decision makers
            4. Subject matter experts
            """,
            "analysis_prompt": """
            Analyze stakeholder needs and influence:
            
            Stakeholders: {stakeholder_list}
            Project Goals: {project_goals}
            
            Please analyze:
            1. Individual stakeholder needs and expectations
            2. Influence level and decision-making power
            3. Potential conflicts or alignment
            4. Communication preferences
            """,
            "prioritization_prompt": """
            Prioritize stakeholder requirements:
            
            Stakeholder Needs: {stakeholder_needs}
            Business Constraints: {business_constraints}
            
            Please prioritize:
            1. Must-have requirements
            2. Should-have requirements
            3. Could-have requirements
            4. Won't-have requirements (this iteration)
            """,
            "engagement_prompt": """
            Plan stakeholder engagement strategy:
            
            Stakeholder Map: {stakeholder_map}
            Project Timeline: {project_timeline}
            
            Please plan:
            1. Engagement frequency and methods
            2. Communication channels and formats
            3. Feedback collection mechanisms
            4. Conflict resolution procedures
            """
        }
        
        template = templates.get(template_name, "")
        return template.format(**variables) 