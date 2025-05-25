"""
Success Criteria Schema

Operator-based success criteria format for consistent evaluation.
"""

from typing import Dict, Any, List, Literal, Union, Optional
from pydantic import BaseModel, Field
from enum import Enum


class SuccessCriteriaOperator(str, Enum):
    """Supported operators for success criteria evaluation."""
    EXISTS = "exists"
    COUNT = "count"
    EQUALS = "equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    IS_NOT_EMPTY = "is_not_empty"
    IS_EMPTY = "is_empty"
    MATCHES_PATTERN = "matches_pattern"


class SuccessCriterion(BaseModel):
    """Individual success criterion with operator-based evaluation."""
    name: str = Field(..., description="Unique name for this criterion")
    operator: SuccessCriteriaOperator = Field(..., description="Evaluation operator")
    value: Union[bool, int, float, str, List[str]] = Field(..., description="Expected value for comparison")
    description: str = Field(..., description="Human-readable description of the criterion")
    field_path: Optional[str] = Field(None, description="Dot-notation path to the field being evaluated")
    weight: float = Field(default=1.0, ge=0.0, le=1.0, description="Weight of this criterion (0.0-1.0)")
    required: bool = Field(default=True, description="Whether this criterion is required for success")


class SuccessCriteriaSet(BaseModel):
    """Collection of success criteria for evaluation."""
    criteria: List[SuccessCriterion] = Field(..., description="List of success criteria")
    evaluation_mode: Literal["all", "any", "weighted"] = Field(
        default="all", description="How to combine criteria results"
    )
    minimum_score: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Minimum score required for success (weighted mode)"
    )
    description: Optional[str] = Field(None, description="Description of this criteria set")


class SuccessCriteriaEvaluationResult(BaseModel):
    """Result of evaluating success criteria."""
    overall_success: bool = Field(..., description="Whether all criteria were met")
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall success score")
    individual_results: Dict[str, bool] = Field(..., description="Results for each criterion")
    failed_criteria: List[str] = Field(default_factory=list, description="Names of failed criteria")
    evaluation_details: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed evaluation information"
    )
    timestamp: str = Field(..., description="When the evaluation was performed")


# Predefined success criteria templates for common scenarios
STANDARD_SUCCESS_CRITERIA = {
    "requirements_documented": SuccessCriterion(
        name="requirements_documented",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Requirements document must exist",
        field_path="requirements_document"
    ),
    
    "requirements_extracted": SuccessCriterion(
        name="requirements_extracted",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Requirements must be extracted from user goal",
        field_path="goal_analysis"
    ),
    
    "stakeholder_needs_identified": SuccessCriterion(
        name="stakeholder_needs_identified",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Stakeholder needs must be identified",
        field_path="goal_analysis.key_stakeholders"
    ),
    
    "stakeholders_identified": SuccessCriterion(
        name="stakeholders_identified",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Stakeholders must be identified",
        field_path="stakeholder_needs_identified"
    ),
    
    "code_generated": SuccessCriterion(
        name="code_generated",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Code must be generated successfully",
        field_path="generated_code"
    ),
    
    "tests_pass": SuccessCriterion(
        name="tests_pass",
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description="Tests must pass successfully",
        field_path="test_results"
    ),
    
    "code_files_created": SuccessCriterion(
        name="code_files_created",
        operator=SuccessCriteriaOperator.GREATER_THAN,
        value=0,
        description="At least one code file must be created",
        field_path="generated_files.count"
    ),
    
    "tests_passed": SuccessCriterion(
        name="tests_passed",
        operator=SuccessCriteriaOperator.EQUALS,
        value=True,
        description="All tests must pass",
        field_path="test_results.success"
    ),
    
    "quality_threshold_met": SuccessCriterion(
        name="quality_threshold_met",
        operator=SuccessCriteriaOperator.GREATER_EQUAL,
        value=0.8,
        description="Code quality score must be at least 0.8",
        field_path="quality_metrics.score"
    ),
    
    "no_critical_vulnerabilities": SuccessCriterion(
        name="no_critical_vulnerabilities",
        operator=SuccessCriteriaOperator.EQUALS,
        value=0,
        description="No critical security vulnerabilities allowed",
        field_path="vulnerabilities.critical.count"
    ),
    
    "deployment_successful": SuccessCriterion(
        name="deployment_successful",
        operator=SuccessCriteriaOperator.EQUALS,
        value="success",
        description="Deployment must be successful",
        field_path="deployment_status"
    ),
    
    "documentation_complete": SuccessCriterion(
        name="documentation_complete",
        operator=SuccessCriteriaOperator.GREATER_EQUAL,
        value=0.9,
        description="Documentation coverage must be at least 90%",
        field_path="documentation_coverage"
    )
}


def create_success_criteria_from_legacy(legacy_criteria: List[str]) -> SuccessCriteriaSet:
    """Convert legacy simple string criteria to operator-based format."""
    criteria = []
    
    for legacy_criterion in legacy_criteria:
        # Try to map to standard criteria first
        if legacy_criterion in STANDARD_SUCCESS_CRITERIA:
            criteria.append(STANDARD_SUCCESS_CRITERIA[legacy_criterion])
        else:
            # Create a default EXISTS criterion for unknown legacy criteria
            criteria.append(SuccessCriterion(
                name=legacy_criterion,
                operator=SuccessCriteriaOperator.EXISTS,
                value=True,
                description=f"Legacy criterion: {legacy_criterion}",
                field_path=legacy_criterion
            ))
    
    return SuccessCriteriaSet(
        criteria=criteria,
        evaluation_mode="all",
        description="Converted from legacy criteria format"
    )


def parse_success_criterion_string(criterion_string: str) -> SuccessCriterion:
    """Parse a string-based success criterion into structured format.
    
    Examples:
        "requirements_document EXISTS" -> SuccessCriterion with EXISTS operator
        "stakeholder_list.count >= 1" -> SuccessCriterion with GREATER_EQUAL operator
        "test_results.success == true" -> SuccessCriterion with EQUALS operator
    """
    # Simple parsing logic - can be enhanced for more complex expressions
    parts = criterion_string.strip().split()
    
    if len(parts) >= 2:
        field_path = parts[0]
        operator_str = parts[1].lower()
        value_str = " ".join(parts[2:]) if len(parts) > 2 else "true"
        
        # Map operator strings to enum values
        operator_mapping = {
            "exists": SuccessCriteriaOperator.EXISTS,
            "==": SuccessCriteriaOperator.EQUALS,
            "!=": SuccessCriteriaOperator.NOT_CONTAINS,
            ">": SuccessCriteriaOperator.GREATER_THAN,
            "<": SuccessCriteriaOperator.LESS_THAN,
            ">=": SuccessCriteriaOperator.GREATER_EQUAL,
            "<=": SuccessCriteriaOperator.LESS_EQUAL,
            "contains": SuccessCriteriaOperator.CONTAINS,
            "is_not_empty": SuccessCriteriaOperator.IS_NOT_EMPTY,
            "is_empty": SuccessCriteriaOperator.IS_EMPTY
        }
        
        operator = operator_mapping.get(operator_str, SuccessCriteriaOperator.EXISTS)
        
        # Parse value
        value: Union[bool, int, float, str]
        if value_str.lower() in ["true", "false"]:
            value = value_str.lower() == "true"
        elif value_str.isdigit():
            value = int(value_str)
        elif "." in value_str and value_str.replace(".", "").isdigit():
            value = float(value_str)
        else:
            value = value_str.strip("'\"")
        
        return SuccessCriterion(
            name=field_path.replace(".", "_"),
            operator=operator,
            value=value,
            description=f"Parsed from: {criterion_string}",
            field_path=field_path
        )
    
    # Fallback for unparseable strings
    return SuccessCriterion(
        name=criterion_string.replace(" ", "_").lower(),
        operator=SuccessCriteriaOperator.EXISTS,
        value=True,
        description=f"Fallback parsing for: {criterion_string}",
        field_path=criterion_string
    ) 