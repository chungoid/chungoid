"""
File Management Protocol

Protocol for autonomous file system operations.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .base.protocol_interface import ProtocolInterface, ProtocolPhase, PhaseStatus

logger = logging.getLogger(__name__)


class FileManagementProtocol(ProtocolInterface):
    """Protocol for autonomous file system operations."""
    
    def __init__(self):
        super().__init__(
            protocol_id="file_management",
            name="File Management Protocol",
            description="Autonomous file system operations"
        )
        
        # Define protocol phases
        self.phases = [
            ProtocolPhase(
                name="discovery",
                description="Discover and inventory files",
                required_inputs=["target_directory", "file_patterns"],
                required_outputs=["file_inventory", "directory_structure"],
                validation_criteria=[
                    "file_inventory EXISTS",
                    "directory_structure.scanned == true"
                ]
            ),
            ProtocolPhase(
                name="planning",
                description="Plan file operations",
                required_inputs=["file_inventory", "operation_requirements"],
                required_outputs=["operation_plan", "backup_strategy"],
                validation_criteria=[
                    "operation_plan.steps.count > 0",
                    "backup_strategy EXISTS"
                ]
            ),
            ProtocolPhase(
                name="execution",
                description="Execute file operations",
                required_inputs=["operation_plan", "safety_checks"],
                required_outputs=["operation_results", "file_changes"],
                validation_criteria=[
                    "operation_results.success == true",
                    "file_changes.logged == true"
                ]
            ),
            ProtocolPhase(
                name="verification",
                description="Verify operation results",
                required_inputs=["operation_results", "expected_outcomes"],
                required_outputs=["verification_report", "rollback_plan"],
                validation_criteria=[
                    "verification_report.verified == true",
                    "rollback_plan EXISTS"
                ]
            )
        ]
    
    def get_phases(self) -> List[ProtocolPhase]:
        """Get all protocol phases."""
        return self.phases
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute."""
        if phase.name == "discovery":
            return True  # First phase is always ready
        elif phase.name == "planning":
            discovery_phase = next((p for p in self.phases if p.name == "discovery"), None)
            return discovery_phase and discovery_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "execution":
            planning_phase = next((p for p in self.phases if p.name == "planning"), None)
            return planning_phase and planning_phase.status == PhaseStatus.COMPLETED
        elif phase.name == "verification":
            execution_phase = next((p for p in self.phases if p.name == "execution"), None)
            return execution_phase and execution_phase.status == PhaseStatus.COMPLETED
        
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
            Discover and inventory files in the target location:
            
            Target Directory: {target_directory}
            File Patterns: {file_patterns}
            
            Please identify:
            1. Existing files and their properties
            2. Directory structure and organization
            3. File permissions and ownership
            4. Potential conflicts or issues
            """,
            "planning_prompt": """
            Plan file operations based on inventory:
            
            File Inventory: {file_inventory}
            Operation Requirements: {operation_requirements}
            
            Please plan:
            1. Sequence of file operations
            2. Backup and safety measures
            3. Error handling procedures
            4. Rollback strategies
            """,
            "execution_prompt": """
            Execute planned file operations:
            
            Operation Plan: {operation_plan}
            Safety Checks: {safety_checks}
            
            Please execute:
            1. File creation, modification, or deletion
            2. Directory structure changes
            3. Permission and ownership updates
            4. Logging of all changes
            """,
            "verification_prompt": """
            Verify operation results:
            
            Operation Results: {operation_results}
            Expected Outcomes: {expected_outcomes}
            
            Please verify:
            1. All operations completed successfully
            2. Files are in expected state
            3. No unintended side effects
            4. System integrity maintained
            """
        }
        
        template = templates.get(template_name, "")
        return template.format(**variables) 