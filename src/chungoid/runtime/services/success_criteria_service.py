"""
Service for evaluating success criteria for orchestrator stages.
UNIFIED SCHEMA VERSION - NO BACKWARDS COMPATIBILITY
"""
import logging
from typing import List, Dict, Any, Tuple

from chungoid.schemas.master_flow import MasterStageSpec
from chungoid.schemas.orchestration import SharedContext

logger = logging.getLogger(__name__)

# UNIFIED SUCCESS CRITERIA FIELDS - SINGLE SOURCE OF TRUTH
UNIFIED_SUCCESS_CRITERIA_FIELDS: Dict[str, type] = {
    # Analysis & Requirements
    "requirements_extracted": bool,
    "requirements_documented": bool, 
    "stakeholders_identified": bool,
    
    # Architecture & Design  
    "architecture_documented": bool,
    "components_defined": bool,
    "design_validated": bool,
    
    # Environment & Setup
    "environment_bootstrapped": bool,
    "dependencies_installed": bool,
    "environment_verified": bool,
    
    # Code Generation
    "code_generated": bool,
    "code_files_created": bool,
    "code_validated": bool,
    
    # Testing & Quality
    "tests_pass": bool,
    "tests_passed": bool,
    "quality_threshold_met": bool,
    "no_critical_vulnerabilities": bool,
    
    # Deployment & Documentation
    "deployment_successful": bool,
    "documentation_complete": bool,
    
    # Universal Requirements (ALL AGENTS)
    "phase_completed": bool,
    "validation_passed": bool,
    "integration_successful": bool,
    
    # Agent-Specific Fields
    "refinement_analysis_completed": bool,
    "improvement_recommendations_generated": bool,
    "coordination_successful": bool,
    "risk_assessment_completed": bool,
    "risk_mitigation_strategies_identified": bool,
    "risk_level_determined": bool,
    "requirements_traced": bool,
    "traceability_matrix_created": bool,
    "coverage_analysis_completed": bool,
    "debugging_analysis_completed": bool,
    "issues_identified": bool,
    "fixes_recommended": bool,
    "blueprint_reviewed": bool,
    "review_feedback_generated": bool,
    "approval_status_determined": bool,
    "documentation_generated": bool,
    "documentation_quality_verified": bool,
    "dependencies_analyzed": bool,
    "dependency_conflicts_resolved": bool,
    "dependency_tree_optimized": bool,
    "master_plan_generated": bool,
    "execution_strategy_defined": bool,
    "resource_allocation_completed": bool,
    "plan_review_completed": bool,
    "review_recommendations_generated": bool,
    "escalation_decision_made": bool,
    "code_integration_completed": bool,
    "integration_conflicts_resolved": bool,
    "codebase_updated": bool,
    "file_operations_completed": bool,
    "file_system_state_verified": bool,
    "backup_strategy_executed": bool,
    "requirements_gathered": bool,
    "requirements_analyzed": bool,
    "requirements_validated": bool,
    "intervention_request_processed": bool,
    "user_response_captured": bool,
    "intervention_status_updated": bool,
    "operation_acknowledged": bool,
    "passthrough_data_preserved": bool,
    "no_op_completed": bool
}

class UnifiedSuccessCriteriaService:
    """
    Simplified service with direct field lookup only.
    NO backwards compatibility. NO complex path resolution. ONE schema only.
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger

    async def check_criteria_for_stage(
        self,
        stage_name: str,
        stage_spec: MasterStageSpec,
        stage_outputs: Dict[str, Any], 
        shared_context_for_stage: SharedContext 
    ) -> Tuple[bool, List[str]]:
        """
        Check success criteria using direct field lookup only.
        
        Args:
            stage_name: Name of the stage being evaluated
            stage_spec: Stage specification containing success criteria
            stage_outputs: Direct outputs from the stage (must be Dict)
            shared_context_for_stage: Shared context (not used in unified approach)
            
        Returns:
            Tuple of (all_passed: bool, failed_criteria: List[str])
        """
        if not stage_spec.success_criteria:
            self.logger.info(f"No success criteria defined for stage '{stage_name}'. Defaulting to success.")
            return True, []

        # Ensure stage_outputs is a dictionary for direct field lookup
        if not isinstance(stage_outputs, dict):
            self.logger.error(
                f"Stage '{stage_name}': stage_outputs must be a dictionary for unified success criteria evaluation. "
                f"Got {type(stage_outputs)}. Agent must return Dict[str, Any] with standardized success criteria fields."
            )
            return False, ["INVALID_OUTPUT_FORMAT"]

        failed_criteria = []
        
        self.logger.info(
            f"Checking {len(stage_spec.success_criteria)} success criteria for stage '{stage_name}' using direct field lookup."
        )
        
        for criterion_str in stage_spec.success_criteria:
            if not await self._evaluate_single_criterion(criterion_str, stage_name, stage_outputs):
                self.logger.warning(f"Stage '{stage_name}' failed success criterion: {criterion_str}")
                failed_criteria.append(criterion_str)
        
        all_passed = len(failed_criteria) == 0
        
        if not all_passed:
            self.logger.warning(
                f"Stage '{stage_name}' failed {len(failed_criteria)} success criteria. Failed: {failed_criteria}"
            )
        else:
            self.logger.info(f"All success criteria passed for stage '{stage_name}'.")
            
        return all_passed, failed_criteria

    async def _evaluate_single_criterion(
        self, 
        criterion_str: str, 
        stage_name: str,
        stage_outputs: Dict[str, Any]
    ) -> bool:
        """
        Direct field lookup - no complex path resolution.
        
        Args:
            criterion_str: The success criteria field name
            stage_name: Name of the stage (for logging)
            stage_outputs: Stage outputs dictionary
            
        Returns:
            True if criterion is met, False otherwise
        """
        self.logger.debug(f"Evaluating criterion: '{criterion_str}' for stage '{stage_name}'")
        
        # Validate criterion is in unified schema
        if criterion_str not in UNIFIED_SUCCESS_CRITERIA_FIELDS:
            self.logger.warning(
                f"Stage '{stage_name}': Criterion '{criterion_str}' is not in UNIFIED_SUCCESS_CRITERIA_FIELDS. "
                f"This indicates the agent or stage specification needs to be updated to use standardized field names."
            )
            # For now, still try to evaluate it
        
        # Direct field lookup only
        field_value = stage_outputs.get(criterion_str, False)
        
        # Success criteria must be explicitly True
        result = field_value is True
        
        self.logger.debug(
            f"Stage '{stage_name}': Criterion '{criterion_str}' = {field_value} -> {result}"
        )
        
        if not result:
            self.logger.info(
                f"Stage '{stage_name}': Criterion '{criterion_str}' failed. "
                f"Expected: True, Got: {field_value} (type: {type(field_value)})"
            )
        
        return result

    def validate_agent_output_format(self, agent_name: str, stage_outputs: Any) -> List[str]:
        """
        Validate that agent output follows the unified schema format.
        
        Args:
            agent_name: Name of the agent
            stage_outputs: Outputs from the agent
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not isinstance(stage_outputs, dict):
            errors.append(f"Agent '{agent_name}' must return Dict[str, Any], got {type(stage_outputs)}")
            return errors
        
        # Check for universal requirements
        universal_fields = ["phase_completed", "validation_passed"]
        for field in universal_fields:
            if field not in stage_outputs:
                errors.append(f"Agent '{agent_name}' missing universal field: '{field}'")
            elif stage_outputs[field] is not True:
                errors.append(f"Agent '{agent_name}' field '{field}' must be True, got {stage_outputs[field]}")
        
        # Check for unknown fields (not in unified schema)
        for field_name, field_value in stage_outputs.items():
            if field_name not in UNIFIED_SUCCESS_CRITERIA_FIELDS and not field_name.startswith("_"):
                # Allow data fields that start with underscore or are common data fields
                if field_name not in ["generated_code", "approach", "success", "error", "result", "data", "output"]:
                    self.logger.warning(
                        f"Agent '{agent_name}' returned unknown field '{field_name}'. "
                        f"Consider adding to UNIFIED_SUCCESS_CRITERIA_FIELDS if this is a success criteria field."
                    )
        
        return errors

# Backwards compatibility alias (will be removed in future version)
SuccessCriteriaService = UnifiedSuccessCriteriaService 