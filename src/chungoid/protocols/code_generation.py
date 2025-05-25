"""
Code Generation Protocol

Protocol for autonomous code generation and implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus

logger = logging.getLogger(__name__)


class CodeGenerationProtocol(ProtocolInterface):
    """Protocol for autonomous code generation and implementation."""
    
    def __init__(self):
        super().__init__(
            protocol_id="code_generation",
            name="Code Generation Protocol",
            description="Autonomous code generation and implementation"
        )
        
        # Define protocol phases
        self.phases = [
            ProtocolPhase(
                name="planning",
                description="Plan implementation approach",
                required_inputs=["specifications", "architecture_context"],
                required_outputs=["implementation_plan", "file_structure"],
                validation_criteria=[
                    "implementation_plan EXISTS",
                    "file_structure.files.count > 0"
                ]
            ),
            ProtocolPhase(
                name="generation",
                description="Generate source code",
                required_inputs=["implementation_plan", "coding_standards"],
                required_outputs=["source_code", "test_files"],
                validation_criteria=[
                    "source_code IS_NOT_EMPTY",
                    "test_files.count >= 1"
                ]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate generated code",
                required_inputs=["source_code", "quality_metrics"],
                required_outputs=["validated_code", "quality_report"],
                validation_criteria=[
                    "quality_report.score >= 0.8",
                    "validated_code.syntax_errors == 0"
                ]
            ),
            ProtocolPhase(
                name="integration",
                description="Integrate with existing codebase",
                required_inputs=["validated_code", "existing_codebase"],
                required_outputs=["integrated_code", "integration_report"],
                validation_criteria=[
                    "integration_report.conflicts == 0",
                    "integrated_code.tests_pass == true"
                ]
            )
        ]
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        if phase.name == "planning":
            return True  # First phase is always ready
        elif phase.name == "generation":
            planning_phase = next((p for p in self.phases if p.name == "planning"), None)
            return planning_phase and planning_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "validation":
            generation_phase = next((p for p in self.phases if p.name == "generation"), None)
            return generation_phase and generation_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "integration":
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
            "planning_prompt": """
            Plan implementation for the following specifications:
            
            Specifications: {specifications}
            Architecture Context: {architecture_context}
            
            Please create:
            1. Detailed implementation plan
            2. File structure and organization
            3. Module dependencies
            4. Testing strategy
            """,
            "generation_prompt": """
            Generate code based on the implementation plan:
            
            Implementation Plan: {implementation_plan}
            Coding Standards: {coding_standards}
            
            Please generate:
            1. Clean, well-documented source code
            2. Comprehensive unit tests
            3. Integration tests where applicable
            4. Documentation and comments
            """,
            "validation_prompt": """
            Validate the generated code:
            
            Source Code: {source_code}
            Quality Metrics: {quality_metrics}
            
            Please validate:
            1. Syntax and semantic correctness
            2. Code quality and maintainability
            3. Test coverage and effectiveness
            4. Performance considerations
            """,
            "integration_prompt": """
            Integrate code with existing codebase:
            
            Validated Code: {validated_code}
            Existing Codebase: {existing_codebase}
            
            Please ensure:
            1. No naming conflicts
            2. Compatible interfaces
            3. Proper dependency management
            4. Successful test execution
            """
        }
        
        template = templates.get(template_name, "")
        return template.format(**variables) 