"""
Input Validation Service

This service provides generic input validation and default value injection
for agents to prevent input resolution bugs and improve robustness.
"""

import logging
from typing import Dict, Any, Optional, Type, List, Union
from pydantic import BaseModel, ValidationError

from chungoid.agents.unified_agent import UnifiedAgent
# Legacy imports commented out during Phase-3 migration
# from chungoid.runtime.agents.system_requirements_gathering_agent import (
#     SystemRequirementsGatheringAgent_v1,
#     SystemRequirementsGatheringInput
# )


class InputValidationResult(BaseModel):
    """Result of input validation and injection."""
    
    is_valid: bool
    final_inputs: Dict[str, Any]
    validation_errors: List[str] = []
    injected_fields: Dict[str, Any] = {}
    warnings: List[str] = []


class InputValidationService:
    """
    Service for validating agent inputs and injecting default values.
    
    This service centralizes the logic for:
    1. Validating that agent inputs meet their schema requirements
    2. Injecting default values when required fields are missing
    3. Providing clear error messages and warnings
    """
    
    # Agent input requirements and injection rules
    # Legacy rules commented out during Phase-3 migration
    AGENT_INPUT_RULES = {
        # "SystemRequirementsGatheringAgent_v1": {
        #     "input_schema": SystemRequirementsGatheringInput,
        #     "required_fields": ["user_goal"],
        #     "injection_rules": {
        #         "user_goal": "initial_goal_str"  # Inject from orchestrator.initial_goal_str
        #     },
        #     "optional_fields": ["project_context_summary"]
        # },
        # "system_requirements_gathering_agent": {  # Alternative ID
        #     "input_schema": SystemRequirementsGatheringInput,
        #     "required_fields": ["user_goal"],
        #     "injection_rules": {
        #         "user_goal": "initial_goal_str"
        #     },
        #     "optional_fields": ["project_context_summary"]
        # }
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_and_inject_inputs(
        self,
        agent_id: str,
        agent_instance: Optional[ProtocolAwareAgent],
        resolved_inputs: Dict[str, Any],
        injection_context: Optional[Dict[str, Any]] = None,
        run_id: str = "unknown"
    ) -> InputValidationResult:
        """
        Validate agent inputs and inject defaults if needed.
        
        Args:
            agent_id: The agent identifier
            agent_instance: The agent instance for type checking
            resolved_inputs: The inputs resolved by ContextResolutionService
            injection_context: Context for injecting default values (e.g., {"initial_goal_str": "..."})
            run_id: Run ID for logging
            
        Returns:
            InputValidationResult with validation status and final inputs
        """
        self.logger.debug(f"Run {run_id}: Validating inputs for agent '{agent_id}'")
        
        # Get agent rules
        agent_rules = self._get_agent_rules(agent_id, agent_instance)
        if not agent_rules:
            # No specific rules for this agent - return inputs as-is
            return InputValidationResult(
                is_valid=True,
                final_inputs=resolved_inputs,
                warnings=[f"No validation rules defined for agent '{agent_id}'"]
            )
        
        result = InputValidationResult(is_valid=True, final_inputs=resolved_inputs.copy())
        input_schema = agent_rules.get("input_schema")
        injection_rules = agent_rules.get("injection_rules", {})
        injection_context = injection_context or {}
        
        # FIRST: Check for injection needs (before validation)
        for field_name, injection_source in injection_rules.items():
            field_value = result.final_inputs.get(field_name)
            
            # Check if field is missing, None, or contains placeholder values
            should_inject = (
                field_name not in result.final_inputs or 
                field_value is None or
                (isinstance(field_value, str) and field_value.startswith("TODO_REPLACE_WITH_ACTUAL_"))
            )
            
            self.logger.debug(f"Run {run_id}: Checking injection for field '{field_name}': value='{field_value}', should_inject={should_inject}")
            
            if should_inject:
                # Try to inject from context
                injected_value = injection_context.get(injection_source)
                if injected_value is not None:
                    self.logger.info(f"Run {run_id}: Injecting '{injection_source}' as '{field_name}' for agent '{agent_id}'")
                    result.final_inputs[field_name] = injected_value
                    result.injected_fields[field_name] = injected_value
                else:
                    self.logger.warning(f"Run {run_id}: Cannot inject '{field_name}' for agent '{agent_id}' - '{injection_source}' not available in context")
        
        # SECOND: Validate with final inputs (after injection)
        if input_schema:
            try:
                input_schema(**result.final_inputs)
                result.is_valid = True
                result.validation_errors = []
                self.logger.debug(f"Run {run_id}: Agent '{agent_id}' inputs are valid after injection")
            except ValidationError as ve:
                self.logger.warning(f"Run {run_id}: Agent '{agent_id}' input validation failed: {ve}")
                result.validation_errors = [str(ve)]
                result.is_valid = False
        
        return result
    
    def _get_agent_rules(self, agent_id: str, agent_instance: Optional[ProtocolAwareAgent]) -> Optional[Dict[str, Any]]:
        """Get validation rules for an agent."""
        # First try by agent_id
        if agent_id in self.AGENT_INPUT_RULES:
            return self.AGENT_INPUT_RULES[agent_id]
        
        # Legacy instance type checking commented out during Phase-3 migration
        if agent_instance:
            # if isinstance(agent_instance, SystemRequirementsGatheringAgent_v1):
            #     return self.AGENT_INPUT_RULES["SystemRequirementsGatheringAgent_v1"]
            pass
        
        return None
    
    def get_required_inputs_for_agent(self, agent_id: str) -> List[str]:
        """Get the list of required inputs for an agent."""
        rules = self.AGENT_INPUT_RULES.get(agent_id)
        if rules:
            return rules.get("required_fields", [])
        return []
    
    def check_flow_stage_inputs(
        self,
        stage_name: str,
        agent_id: str,
        stage_inputs: Dict[str, Any]
    ) -> List[str]:
        """
        Check if a flow stage definition has all required inputs.
        
        Args:
            stage_name: Name of the stage
            agent_id: Agent ID for the stage
            stage_inputs: The inputs dict from the stage definition
            
        Returns:
            List of validation error messages
        """
        errors = []
        required_inputs = self.get_required_inputs_for_agent(agent_id)
        
        for required_input in required_inputs:
            if required_input not in stage_inputs:
                errors.append(
                    f"Stage '{stage_name}' using agent '{agent_id}' is missing required input '{required_input}'. "
                    f"Add it to the stage inputs or ensure it's available via context resolution."
                )
        
        return errors
    
    def suggest_fixes_for_missing_inputs(
        self,
        agent_id: str,
        missing_inputs: List[str]
    ) -> List[str]:
        """Generate fix suggestions for missing inputs."""
        suggestions = []
        
        if agent_id in ["SystemRequirementsGatheringAgent_v1", "system_requirements_gathering_agent"]:
            if "user_goal" in missing_inputs:
                suggestions.append(
                    "Fix for SystemRequirementsGatheringAgent_v1 missing user_goal:\n"
                    "  Add to stage inputs:\n"
                    "    inputs:\n"
                    "      user_goal: \"Your project goal description here\"\n"
                    "      # OR use context path:\n"
                    "      # user_goal: \"{context.data.initial_goal}\"\n"
                    "      # OR let orchestrator inject from initial_goal_str"
                )
        
        return suggestions 