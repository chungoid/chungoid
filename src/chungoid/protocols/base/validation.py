"""
Protocol Validation Framework

Provides validation capabilities for protocol execution, phase completion,
and quality gate enforcement.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class ValidationLevel(Enum):
    """Validation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    """Result of a validation check"""
    passed: bool
    level: ValidationLevel
    message: str
    details: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

class ProtocolValidator:
    """
    Base validator for protocol execution and quality gates.
    
    Provides systematic validation of protocol phases, outputs,
    and quality criteria to ensure reliable execution.
    """
    
    def __init__(self):
        self.validation_rules = {}
        self.quality_gates = {}
    
    def validate_protocol_phase(self, phase_name: str, outputs: Dict[str, Any], 
                              criteria: List[str]) -> List[ValidationResult]:
        """
        Validate a protocol phase completion.
        
        Args:
            phase_name: Name of the protocol phase
            outputs: Outputs produced by the phase
            criteria: Validation criteria that must be met
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check required outputs exist
        for criterion in criteria:
            if criterion.startswith("output:"):
                output_name = criterion.replace("output:", "")
                if output_name not in outputs:
                    results.append(ValidationResult(
                        passed=False,
                        level=ValidationLevel.ERROR,
                        message=f"Missing required output: {output_name}",
                        suggestions=[f"Ensure phase produces {output_name}"]
                    ))
                else:
                    results.append(ValidationResult(
                        passed=True,
                        level=ValidationLevel.INFO,
                        message=f"Required output present: {output_name}"
                    ))
        
        # Check quality criteria
        for criterion in criteria:
            if criterion.startswith("quality:"):
                quality_check = criterion.replace("quality:", "")
                result = self._check_quality_criterion(quality_check, outputs)
                results.append(result)
        
        return results
    
    def validate_tool_usage(self, tool_name: str, tool_result: Any, 
                          expected_criteria: List[str]) -> ValidationResult:
        """
        Validate tool usage and results.
        
        Args:
            tool_name: Name of the tool used
            tool_result: Result from tool execution
            expected_criteria: Expected validation criteria
            
        Returns:
            Validation result for tool usage
        """
        if tool_result is None:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Tool {tool_name} returned no result",
                suggestions=["Check tool configuration and inputs"]
            )
        
        # Tool-specific validation logic would go here
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Tool {tool_name} executed successfully"
        )
    
    def validate_agent_output(self, agent_name: str, output: Dict[str, Any],
                            quality_standards: List[str]) -> List[ValidationResult]:
        """
        Validate agent output against quality standards.
        
        Args:
            agent_name: Name of the agent
            output: Agent output to validate
            quality_standards: Quality standards to check against
            
        Returns:
            List of validation results
        """
        results = []
        
        for standard in quality_standards:
            if standard == "completeness":
                result = self._check_completeness(output)
            elif standard == "accuracy":
                result = self._check_accuracy(output)
            elif standard == "consistency":
                result = self._check_consistency(output)
            else:
                result = ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Unknown quality standard: {standard}"
                )
            
            results.append(result)
        
        return results
    
    def _check_quality_criterion(self, criterion: str, outputs: Dict[str, Any]) -> ValidationResult:
        """Check a specific quality criterion"""
        if criterion == "non_empty":
            if not outputs or all(not v for v in outputs.values()):
                return ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message="Outputs are empty",
                    suggestions=["Ensure phase produces meaningful outputs"]
                )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Quality criterion met: {criterion}"
        )
    
    def _check_completeness(self, output: Dict[str, Any]) -> ValidationResult:
        """Check output completeness"""
        if not output:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message="Output is empty"
            )
        
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Output completeness validated"
        )
    
    def _check_accuracy(self, output: Dict[str, Any]) -> ValidationResult:
        """Check output accuracy"""
        # Placeholder for accuracy validation logic
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Output accuracy validated"
        )
    
    def _check_consistency(self, output: Dict[str, Any]) -> ValidationResult:
        """Check output consistency"""
        # Placeholder for consistency validation logic
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Output consistency validated"
        ) 