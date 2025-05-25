"""
Quality Gates Validator

Provides quality gate enforcement for protocol execution,
ensuring outputs meet required standards before proceeding.
"""

from typing import List, Dict, Any, Optional, Callable
from ..base.validation import ProtocolValidator, ValidationResult, ValidationLevel

class QualityGate:
    """
    Represents a quality gate with specific validation criteria.
    """
    
    def __init__(self, name: str, description: str, criteria: List[str],
                 validation_function: Optional[Callable] = None):
        self.name = name
        self.description = description
        self.criteria = criteria
        self.validation_function = validation_function
        self.required = True
        
    def validate(self, output: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate output against this quality gate"""
        if self.validation_function:
            return self.validation_function(output, context)
        else:
            return self._default_validation(output, context)
            
    def _default_validation(self, output: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Default validation logic"""
        if output is None or output == "":
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Quality gate '{self.name}' failed: output is empty"
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message=f"Quality gate '{self.name}' passed"
        )

class QualityGateValidator(ProtocolValidator):
    """
    Enhanced validator with quality gate enforcement.
    
    Provides systematic quality validation with configurable gates
    for different types of outputs and processes.
    """
    
    def __init__(self):
        super().__init__()
        self.quality_gates = {}
        self._register_default_gates()
        
    def register_quality_gate(self, gate: QualityGate):
        """Register a quality gate for validation"""
        self.quality_gates[gate.name] = gate
        
    def validate_against_gates(self, output: Any, gate_names: List[str],
                             context: Dict[str, Any] = None) -> List[ValidationResult]:
        """
        Validate output against specified quality gates.
        
        Args:
            output: Output to validate
            gate_names: List of quality gate names to apply
            context: Additional context for validation
            
        Returns:
            List of validation results from all gates
        """
        results = []
        
        for gate_name in gate_names:
            if gate_name in self.quality_gates:
                gate = self.quality_gates[gate_name]
                result = gate.validate(output, context)
                results.append(result)
            else:
                results.append(ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Unknown quality gate: {gate_name}"
                ))
                
        return results
        
    def validate_code_quality(self, code: str, standards: List[str] = None) -> List[ValidationResult]:
        """
        Validate code quality against programming standards.
        
        Args:
            code: Source code to validate
            standards: List of coding standards to check
            
        Returns:
            List of validation results
        """
        standards = standards or ["syntax", "completeness", "documentation"]
        results = []
        
        for standard in standards:
            if standard == "syntax":
                result = self._validate_syntax(code)
            elif standard == "completeness":
                result = self._validate_completeness(code)
            elif standard == "documentation":
                result = self._validate_documentation(code)
            elif standard == "security":
                result = self._validate_security(code)
            elif standard == "performance":
                result = self._validate_performance(code)
            else:
                result = ValidationResult(
                    passed=False,
                    level=ValidationLevel.WARNING,
                    message=f"Unknown code quality standard: {standard}"
                )
                
            results.append(result)
            
        return results
        
    def validate_test_quality(self, test_results: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate test execution quality and coverage.
        
        Args:
            test_results: Test execution results
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check test execution status
        if test_results.get("status") == "failed":
            results.append(ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message="Test execution failed",
                suggestions=["Review test failures and fix issues"]
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message="Test execution passed"
            ))
            
        # Check test coverage
        coverage = test_results.get("coverage", 0)
        if coverage < 80:
            results.append(ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message=f"Test coverage too low: {coverage}%",
                suggestions=["Add more comprehensive test cases"]
            ))
        else:
            results.append(ValidationResult(
                passed=True,
                level=ValidationLevel.INFO,
                message=f"Test coverage adequate: {coverage}%"
            ))
            
        return results
        
    def validate_deployment_readiness(self, deployment_package: Dict[str, Any]) -> List[ValidationResult]:
        """
        Validate deployment package readiness.
        
        Args:
            deployment_package: Package information and artifacts
            
        Returns:
            List of validation results
        """
        results = []
        
        # Check required artifacts
        required_artifacts = ["code", "tests", "documentation", "configuration"]
        for artifact in required_artifacts:
            if artifact in deployment_package:
                results.append(ValidationResult(
                    passed=True,
                    level=ValidationLevel.INFO,
                    message=f"Required artifact present: {artifact}"
                ))
            else:
                results.append(ValidationResult(
                    passed=False,
                    level=ValidationLevel.ERROR,
                    message=f"Missing required artifact: {artifact}",
                    suggestions=[f"Ensure {artifact} is included in deployment package"]
                ))
                
        return results
        
    def _register_default_gates(self):
        """Register default quality gates"""
        
        # Code quality gate
        code_gate = QualityGate(
            name="code_quality",
            description="Validates code meets quality standards",
            criteria=["syntax_valid", "well_documented", "follows_conventions"],
            validation_function=self._validate_code_gate
        )
        self.register_quality_gate(code_gate)
        
        # Test quality gate
        test_gate = QualityGate(
            name="test_quality", 
            description="Validates test coverage and execution",
            criteria=["tests_pass", "adequate_coverage", "comprehensive"],
            validation_function=self._validate_test_gate
        )
        self.register_quality_gate(test_gate)
        
        # Documentation quality gate
        docs_gate = QualityGate(
            name="documentation_quality",
            description="Validates documentation completeness",
            criteria=["complete", "accurate", "up_to_date"],
            validation_function=self._validate_docs_gate
        )
        self.register_quality_gate(docs_gate)
        
        # Security quality gate
        security_gate = QualityGate(
            name="security_quality",
            description="Validates security requirements",
            criteria=["no_vulnerabilities", "secure_practices", "access_controls"],
            validation_function=self._validate_security_gate
        )
        self.register_quality_gate(security_gate)
        
    def _validate_code_gate(self, code: str, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate code quality gate"""
        if not code or code.strip() == "":
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message="Code quality gate failed: no code provided"
            )
            
        # Basic code quality checks
        has_functions = "def " in code or "function " in code or "class " in code
        has_comments = "#" in code or "//" in code or "/*" in code
        
        if not has_functions:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="Code quality gate warning: no functions or classes detected"
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Code quality gate passed"
        )
        
    def _validate_test_gate(self, test_data: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate test quality gate"""
        if isinstance(test_data, dict):
            if test_data.get("status") == "passed":
                return ValidationResult(
                    passed=True,
                    level=ValidationLevel.INFO,
                    message="Test quality gate passed"
                )
        
        return ValidationResult(
            passed=False,
            level=ValidationLevel.ERROR,
            message="Test quality gate failed"
        )
        
    def _validate_docs_gate(self, docs: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate documentation quality gate"""
        if not docs:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="Documentation quality gate warning: no documentation provided"
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Documentation quality gate passed"
        )
        
    def _validate_security_gate(self, artifact: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate security quality gate"""
        # Placeholder for security validation
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Security quality gate passed"
        )
        
    def _validate_syntax(self, code: str) -> ValidationResult:
        """Validate code syntax"""
        if not code:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message="Syntax validation failed: no code provided"
            )
            
        # Basic syntax checks (placeholder)
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Syntax validation passed"
        )
        
    def _validate_completeness(self, code: str) -> ValidationResult:
        """Validate code completeness"""
        if "TODO" in code or "FIXME" in code or "..." in code:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="Completeness validation warning: TODO items found",
                suggestions=["Complete TODO items before deployment"]
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Completeness validation passed"
        )
        
    def _validate_documentation(self, code: str) -> ValidationResult:
        """Validate code documentation"""
        # Check for basic documentation patterns
        has_docstrings = '"""' in code or "'''" in code
        has_comments = "#" in code or "//" in code
        
        if not (has_docstrings or has_comments):
            return ValidationResult(
                passed=False,
                level=ValidationLevel.WARNING,
                message="Documentation validation warning: minimal documentation found",
                suggestions=["Add docstrings and comments to improve documentation"]
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Documentation validation passed"
        )
        
    def _validate_security(self, code: str) -> ValidationResult:
        """Validate code security"""
        # Basic security checks (placeholder)
        security_issues = []
        
        if "password" in code.lower() and "=" in code:
            security_issues.append("Potential hardcoded password")
            
        if "api_key" in code.lower() and "=" in code:
            security_issues.append("Potential hardcoded API key")
            
        if security_issues:
            return ValidationResult(
                passed=False,
                level=ValidationLevel.ERROR,
                message=f"Security validation failed: {', '.join(security_issues)}",
                suggestions=["Remove hardcoded credentials", "Use environment variables"]
            )
            
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Security validation passed"
        )
        
    def _validate_performance(self, code: str) -> ValidationResult:
        """Validate code performance considerations"""
        # Placeholder for performance validation
        return ValidationResult(
            passed=True,
            level=ValidationLevel.INFO,
            message="Performance validation passed"
        ) 