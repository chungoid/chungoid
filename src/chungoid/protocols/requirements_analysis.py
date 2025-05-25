"""
Requirements Analysis Protocol

Comprehensive protocol for requirements gathering and analysis.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, ProtocolTemplate, PhaseStatus

logger = logging.getLogger(__name__)


class RequirementsAnalysisProtocol(ProtocolInterface):
    """Protocol for comprehensive requirements analysis."""
    
    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        """Protocol name."""
        return "requirements_analysis"
    
    @property
    def description(self) -> str:
        """Protocol description."""
        return "Comprehensive requirements gathering and analysis"
    
    @property
    def total_estimated_time(self) -> float:
        """Total estimated time in hours."""
        return 4.0  # Sum of all phase time estimates: 1.0 + 1.5 + 1.0 + 0.5
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        """Initialize all protocol phases."""
        return [
            ProtocolPhase(
                name="discovery",
                description="Discover and gather requirements",
                time_box_hours=1.0,
                required_outputs=["raw_requirements", "stakeholder_list"],
                validation_criteria=[
                    "raw_requirements EXISTS",
                    "stakeholder_list.count >= 1"
                ],
                tools_required=["codebase_search", "filesystem_read_file", "content_generate"],
                dependencies=[]
            ),
            ProtocolPhase(
                name="analysis",
                description="Analyze and structure requirements",
                time_box_hours=1.5,
                required_outputs=["structured_requirements", "dependencies"],
                validation_criteria=[
                    "structured_requirements IS_NOT_EMPTY",
                    "dependencies.identified == true"
                ],
                tools_required=["content_generate", "content_validate"],
                dependencies=["discovery"]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate requirements",
                time_box_hours=1.0,
                required_outputs=["validated_requirements", "risk_assessment"],
                validation_criteria=[
                    "validated_requirements.status == 'approved'",
                    "risk_assessment.level != 'critical'"
                ],
                tools_required=["content_validate", "content_generate"],
                dependencies=["analysis"]
            ),
            ProtocolPhase(
                name="documentation",
                description="Document requirements",
                time_box_hours=0.5,
                required_outputs=["requirements_document", "acceptance_criteria"],
                validation_criteria=[
                    "requirements_document EXISTS",
                    "acceptance_criteria.count >= 3"
                ],
                tools_required=["filesystem_write_file", "content_generate"],
                dependencies=["validation"]
            )
        ]
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates."""
        return {
            "discovery_prompt": ProtocolTemplate(
                name="discovery_prompt",
                description="Template for requirements discovery",
                template_content="""
                Analyze the following user goal and discover requirements:
                
                Goal: [user_goal]
                Context: [context]
                
                Please identify:
                1. Raw requirements from the goal
                2. Key stakeholders involved
                3. Initial constraints and assumptions
                """,
                variables=["user_goal", "context"]
            ),
            "analysis_prompt": ProtocolTemplate(
                name="analysis_prompt",
                description="Template for requirements analysis",
                template_content="""
                Structure and analyze the following raw requirements:
                
                Raw Requirements: [raw_requirements]
                Domain Knowledge: [domain_knowledge]
                
                Please provide:
                1. Structured functional requirements
                2. Non-functional requirements
                3. Dependencies and relationships
                """,
                variables=["raw_requirements", "domain_knowledge"]
            ),
            "validation_prompt": ProtocolTemplate(
                name="validation_prompt",
                description="Template for requirements validation",
                template_content="""
                Validate the following structured requirements:
                
                Requirements: [structured_requirements]
                Constraints: [constraints]
                
                Please assess:
                1. Completeness and consistency
                2. Feasibility and risks
                3. Approval recommendation
                """,
                variables=["structured_requirements", "constraints"]
            ),
            "documentation_prompt": ProtocolTemplate(
                name="documentation_prompt",
                description="Template for requirements documentation",
                template_content="""
                Create comprehensive documentation for:
                
                Validated Requirements: [validated_requirements]
                
                Please generate:
                1. Formal requirements document
                2. Acceptance criteria
                3. Implementation guidelines
                """,
                variables=["validated_requirements"]
            )
        }
    
    # Legacy compatibility methods - can be removed once all code uses base class methods
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        # Use base class implementation which checks dependencies
        return super().is_phase_ready(phase)
    
    def get_current_phase(self) -> Optional[ProtocolPhase]:
        """Get the current active phase."""
        # Use base class implementation
        return super().get_current_phase()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get protocol execution progress summary."""
        # Use base class implementation
        return super().get_progress_summary()
    
    def get_template(self, template_name: str, **variables) -> str:
        """Get protocol template with variable substitution."""
        # Use base class implementation
        return super().get_template(template_name, **variables) 