"""
Requirements Analysis Protocol

Comprehensive protocol for requirements gathering and analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus

logger = logging.getLogger(__name__)


class RequirementsAnalysisProtocol(ProtocolInterface):
    """Protocol for comprehensive requirements analysis."""
    
    def __init__(self):
        super().__init__(
            protocol_id="requirements_analysis",
            name="Requirements Analysis Protocol",
            description="Comprehensive requirements gathering and analysis"
        )
        
        # Define protocol phases
        self.phases = [
            ProtocolPhase(
                name="discovery",
                description="Discover and gather requirements",
                required_inputs=["user_goal", "context"],
                required_outputs=["raw_requirements", "stakeholder_list"],
                validation_criteria=[
                    "raw_requirements EXISTS",
                    "stakeholder_list.count >= 1"
                ]
            ),
            ProtocolPhase(
                name="analysis",
                description="Analyze and structure requirements",
                required_inputs=["raw_requirements", "domain_knowledge"],
                required_outputs=["structured_requirements", "dependencies"],
                validation_criteria=[
                    "structured_requirements IS_NOT_EMPTY",
                    "dependencies.identified == true"
                ]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate requirements",
                required_inputs=["structured_requirements", "constraints"],
                required_outputs=["validated_requirements", "risk_assessment"],
                validation_criteria=[
                    "validated_requirements.status == 'approved'",
                    "risk_assessment.level != 'critical'"
                ]
            ),
            ProtocolPhase(
                name="documentation",
                description="Document requirements",
                required_inputs=["validated_requirements"],
                required_outputs=["requirements_document", "acceptance_criteria"],
                validation_criteria=[
                    "requirements_document EXISTS",
                    "acceptance_criteria.count >= 3"
                ]
            )
        ]
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        # Basic dependency checking
        if phase.name == "discovery":
            return True  # First phase is always ready
        elif phase.name == "analysis":
            # Requires discovery to be completed
            discovery_phase = next((p for p in self.phases if p.name == "discovery"), None)
            return discovery_phase and discovery_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "validation":
            # Requires analysis to be completed
            analysis_phase = next((p for p in self.phases if p.name == "analysis"), None)
            return analysis_phase and analysis_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "documentation":
            # Requires validation to be completed
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
            "discovery_prompt": """
            Analyze the following user goal and discover requirements:
            
            Goal: {user_goal}
            Context: {context}
            
            Please identify:
            1. Raw requirements from the goal
            2. Key stakeholders involved
            3. Initial constraints and assumptions
            """,
            "analysis_prompt": """
            Structure and analyze the following raw requirements:
            
            Raw Requirements: {raw_requirements}
            Domain Knowledge: {domain_knowledge}
            
            Please provide:
            1. Structured functional requirements
            2. Non-functional requirements
            3. Dependencies and relationships
            """,
            "validation_prompt": """
            Validate the following structured requirements:
            
            Requirements: {structured_requirements}
            Constraints: {constraints}
            
            Please assess:
            1. Completeness and consistency
            2. Feasibility and risks
            3. Approval recommendation
            """,
            "documentation_prompt": """
            Create comprehensive documentation for:
            
            Validated Requirements: {validated_requirements}
            
            Please generate:
            1. Formal requirements document
            2. Acceptance criteria
            3. Implementation guidelines
            """
        }
        
        template = templates.get(template_name, "")
        return template.format(**variables) 