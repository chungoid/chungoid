"""
Code Generation Protocol

Protocol for autonomous code generation and implementation.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus, ProtocolTemplate

logger = logging.getLogger(__name__)


class CodeGenerationProtocol(ProtocolInterface):
    """Protocol for autonomous code generation and implementation."""
    
    def __init__(self):
        super().__init__()
        
    @property
    def name(self) -> str:
        """Protocol name."""
        return "Code Generation Protocol"
    
    @property
    def description(self) -> str:
        """Protocol description."""
        return "Autonomous code generation and implementation protocol"
    
    @property
    def total_estimated_time(self) -> int:
        """Total estimated time in minutes."""
        return 45  # Estimated time for code generation workflow
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        """Initialize protocol phases."""
        return [
            ProtocolPhase(
                name="planning",
                description="Plan implementation approach",
                time_box_hours=0.5,
                required_outputs=["implementation_plan", "file_structure"],
                validation_criteria=[
                    "implementation_plan EXISTS",
                    "file_structure.files.count > 0"
                ],
                tools_required=["llm_provider", "file_system"]
            ),
            ProtocolPhase(
                name="generation",
                description="Generate source code",
                time_box_hours=1.0,
                required_outputs=["source_code", "test_files"],
                validation_criteria=[
                    "source_code IS_NOT_EMPTY",
                    "test_files.count >= 1"
                ],
                tools_required=["llm_provider", "file_system", "code_validator"]
            ),
            ProtocolPhase(
                name="validation",
                description="Validate generated code",
                time_box_hours=0.5,
                required_outputs=["validated_code", "quality_report"],
                validation_criteria=[
                    "quality_report.score >= 0.8",
                    "validated_code.syntax_errors == 0"
                ],
                tools_required=["code_validator", "test_runner"]
            ),
            ProtocolPhase(
                name="integration",
                description="Integrate with existing codebase",
                time_box_hours=0.5,
                required_outputs=["integrated_code", "integration_report"],
                validation_criteria=[
                    "integration_report.conflicts == 0",
                    "integrated_code.tests_pass == true"
                ],
                tools_required=["file_system", "test_runner", "version_control"]
            )
        ]
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates."""
        return {
            "planning_prompt": ProtocolTemplate(
                name="planning_prompt",
                description="Template for planning phase",
                template_content="""
                Plan implementation for the following specifications:
                
                Specifications: [specifications]
                Architecture Context: [architecture_context]
                
                Please create:
                1. Detailed implementation plan
                2. File structure and organization
                3. Module dependencies
                4. Testing strategy
                """,
                variables=["specifications", "architecture_context"]
            ),
            "generation_prompt": ProtocolTemplate(
                name="generation_prompt",
                description="Template for code generation phase",
                template_content="""
                Generate code based on the implementation plan:
                
                Implementation Plan: [implementation_plan]
                Coding Standards: [coding_standards]
                
                Please generate:
                1. Clean, well-documented source code
                2. Comprehensive unit tests
                3. Integration tests where applicable
                4. Documentation and comments
                """,
                variables=["implementation_plan", "coding_standards"]
            ),
            "validation_prompt": ProtocolTemplate(
                name="validation_prompt",
                description="Template for validation phase",
                template_content="""
                Validate the generated code:
                
                Source Code: [source_code]
                Quality Metrics: [quality_metrics]
                
                Please validate:
                1. Syntax and semantic correctness
                2. Code quality and maintainability
                3. Test coverage and effectiveness
                4. Performance considerations
                """,
                variables=["source_code", "quality_metrics"]
            ),
            "integration_prompt": ProtocolTemplate(
                name="integration_prompt",
                description="Template for integration phase",
                template_content="""
                Integrate code with existing codebase:
                
                Validated Code: [validated_code]
                Existing Codebase: [existing_codebase]
                
                Please ensure:
                1. No naming conflicts
                2. Compatible interfaces
                3. Proper dependency management
                4. Successful test execution
                """,
                variables=["validated_code", "existing_codebase"]
            )
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
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.initialize_phases()
    
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
            if phase.status == PhaseStatus.NOT_STARTED and self.is_phase_ready(phase):
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
            "progress_percentage": (len(completed_phases) / len(self.phases)) * 100 if self.phases else 0,
            "current_phase": self.get_current_phase().name if self.get_current_phase() else None,
            "status": "completed" if len(completed_phases) == len(self.phases) else "in_progress"
        } 