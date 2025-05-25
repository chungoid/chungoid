"""
Security Validation Module for Enhanced Agent System

This module implements security validation and vulnerability assessment for the
enhanced autonomous agent orchestration system, following Systematic Implementation
Protocol Phase 4 requirements.

Key Features:
- Agent capability validation and sandboxing
- Input sanitization and validation
- Resource access control
- Performance-based security monitoring
- Vulnerability scanning for agent interactions
"""

import asyncio
import logging
import re
import hashlib
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import os

logger = logging.getLogger(__name__)


class SecurityLevel(Enum):
    """Security levels for agent operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VulnerabilityType(Enum):
    """Types of security vulnerabilities."""
    INJECTION = "injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_EXPOSURE = "data_exposure"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    MALICIOUS_INPUT = "malicious_input"


@dataclass
class SecurityViolation:
    """Represents a security violation or vulnerability."""
    
    violation_type: VulnerabilityType
    severity: SecurityLevel
    description: str
    agent_id: Optional[str] = None
    task_type: Optional[str] = None
    input_data: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    remediation: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/reporting."""
        return {
            'type': self.violation_type.value,
            'severity': self.severity.value,
            'description': self.description,
            'agent_id': self.agent_id,
            'task_type': self.task_type,
            'timestamp': self.timestamp,
            'remediation': self.remediation
        }


@dataclass
class SecurityAssessment:
    """Results of a security assessment."""
    
    passed: bool
    violations: List[SecurityViolation] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    assessment_time: float = field(default_factory=time.time)
    
    def get_severity_counts(self) -> Dict[str, int]:
        """Get count of violations by severity."""
        counts = {level.value: 0 for level in SecurityLevel}
        for violation in self.violations:
            counts[violation.severity.value] += 1
        return counts
    
    def has_critical_violations(self) -> bool:
        """Check if assessment has critical violations."""
        return any(v.severity == SecurityLevel.CRITICAL for v in self.violations)


class InputSanitizer:
    """
    Sanitizes and validates inputs for agent execution.
    
    Implements input validation to prevent injection attacks and
    malicious input processing.
    """
    
    def __init__(self):
        # Patterns for detecting potentially malicious input
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',                # JavaScript URLs
            r'data:text/html',            # Data URLs
            r'eval\s*\(',                 # Code evaluation
            r'exec\s*\(',                 # Code execution
            r'import\s+os',               # OS module import
            r'__import__',                # Dynamic imports
            r'subprocess\.',              # Subprocess calls
            r'system\s*\(',               # System calls
            r'\.\./',                     # Path traversal
            r'file://',                   # File protocol
            r'ftp://',                    # FTP protocol
        ]
        
        # Compile patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.dangerous_patterns]
        
        # Maximum input sizes
        self.max_input_size = 1024 * 1024  # 1MB
        self.max_string_length = 10000
    
    def sanitize_input(self, input_data: Any) -> Tuple[Any, List[SecurityViolation]]:
        """
        Sanitize input data and return violations found.
        
        Args:
            input_data: Input data to sanitize
            
        Returns:
            Tuple of (sanitized_data, violations)
        """
        violations = []
        
        if isinstance(input_data, str):
            sanitized, string_violations = self._sanitize_string(input_data)
            violations.extend(string_violations)
            return sanitized, violations
        
        elif isinstance(input_data, dict):
            sanitized = {}
            for key, value in input_data.items():
                # Sanitize key
                clean_key, key_violations = self.sanitize_input(key)
                violations.extend(key_violations)
                
                # Sanitize value
                clean_value, value_violations = self.sanitize_input(value)
                violations.extend(value_violations)
                
                sanitized[clean_key] = clean_value
            
            return sanitized, violations
        
        elif isinstance(input_data, list):
            sanitized = []
            for item in input_data:
                clean_item, item_violations = self.sanitize_input(item)
                violations.extend(item_violations)
                sanitized.append(clean_item)
            
            return sanitized, violations
        
        else:
            # For other types, just return as-is
            return input_data, violations
    
    def _sanitize_string(self, text: str) -> Tuple[str, List[SecurityViolation]]:
        """Sanitize a string input."""
        violations = []
        
        # Check string length
        if len(text) > self.max_string_length:
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                severity=SecurityLevel.MEDIUM,
                description=f"Input string too long: {len(text)} > {self.max_string_length}",
                input_data=text[:100] + "..." if len(text) > 100 else text,
                remediation="Truncate input to maximum allowed length"
            ))
            text = text[:self.max_string_length]
        
        # Check for dangerous patterns
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                violations.append(SecurityViolation(
                    violation_type=VulnerabilityType.INJECTION,
                    severity=SecurityLevel.HIGH,
                    description=f"Potentially malicious pattern detected: {pattern.pattern}",
                    input_data=text[:100] + "..." if len(text) > 100 else text,
                    remediation="Remove or escape malicious patterns"
                ))
        
        # Basic HTML escaping for safety
        sanitized = (text
                    .replace('&', '&amp;')
                    .replace('<', '&lt;')
                    .replace('>', '&gt;')
                    .replace('"', '&quot;')
                    .replace("'", '&#x27;'))
        
        return sanitized, violations


class AgentCapabilityValidator:
    """
    Validates agent capabilities and enforces security constraints.
    
    Ensures agents only access resources and capabilities they're
    authorized to use.
    """
    
    def __init__(self):
        # Define capability security levels
        self.capability_security_levels = {
            # Low risk capabilities
            'requirements_analysis': SecurityLevel.LOW,
            'documentation': SecurityLevel.LOW,
            'stakeholder_analysis': SecurityLevel.LOW,
            
            # Medium risk capabilities
            'architecture_design': SecurityLevel.MEDIUM,
            'system_planning': SecurityLevel.MEDIUM,
            'code_generation': SecurityLevel.MEDIUM,
            'test_generation': SecurityLevel.MEDIUM,
            
            # High risk capabilities
            'environment_setup': SecurityLevel.HIGH,
            'dependency_management': SecurityLevel.HIGH,
            'file_operations': SecurityLevel.HIGH,
            'code_debugging': SecurityLevel.HIGH,
            
            # Critical capabilities
            'system_administration': SecurityLevel.CRITICAL,
            'network_access': SecurityLevel.CRITICAL,
            'database_access': SecurityLevel.CRITICAL,
        }
        
        # Define restricted capability combinations
        self.restricted_combinations = [
            {'file_operations', 'network_access'},  # File + network access
            {'system_administration', 'code_generation'},  # Admin + code gen
            {'database_access', 'code_debugging'},  # DB + debugging
        ]
    
    def validate_agent_capabilities(self, agent_id: str, capabilities: List[str]) -> SecurityAssessment:
        """
        Validate an agent's capabilities for security compliance.
        
        Args:
            agent_id: ID of the agent to validate
            capabilities: List of capabilities the agent has
            
        Returns:
            SecurityAssessment with validation results
        """
        violations = []
        warnings = []
        recommendations = []
        
        # Check individual capability security levels
        high_risk_capabilities = []
        critical_capabilities = []
        
        for capability in capabilities:
            security_level = self.capability_security_levels.get(capability, SecurityLevel.MEDIUM)
            
            if security_level == SecurityLevel.HIGH:
                high_risk_capabilities.append(capability)
            elif security_level == SecurityLevel.CRITICAL:
                critical_capabilities.append(capability)
        
        # Flag critical capabilities
        if critical_capabilities:
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                severity=SecurityLevel.CRITICAL,
                description=f"Agent has critical capabilities: {critical_capabilities}",
                agent_id=agent_id,
                remediation="Review and restrict critical capabilities"
            ))
        
        # Warn about high-risk capabilities
        if high_risk_capabilities:
            warnings.append(f"Agent {agent_id} has high-risk capabilities: {high_risk_capabilities}")
            recommendations.append("Consider implementing additional monitoring for high-risk operations")
        
        # Check for restricted capability combinations
        capability_set = set(capabilities)
        for restricted_combo in self.restricted_combinations:
            if restricted_combo.issubset(capability_set):
                violations.append(SecurityViolation(
                    violation_type=VulnerabilityType.PRIVILEGE_ESCALATION,
                    severity=SecurityLevel.HIGH,
                    description=f"Agent has restricted capability combination: {restricted_combo}",
                    agent_id=agent_id,
                    remediation="Separate capabilities into different agents"
                ))
        
        # Check for excessive capabilities
        if len(capabilities) > 10:
            warnings.append(f"Agent {agent_id} has many capabilities ({len(capabilities)})")
            recommendations.append("Consider splitting agent into more specialized agents")
        
        return SecurityAssessment(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            recommendations=recommendations
        )


class ResourceAccessController:
    """
    Controls and monitors resource access for agents.
    
    Implements access control policies and monitors resource usage
    to prevent abuse and unauthorized access.
    """
    
    def __init__(self):
        # Define allowed file paths for different security levels
        self.allowed_paths = {
            SecurityLevel.LOW: [
                '/tmp/chungoid/safe/',
                '/var/log/chungoid/',
            ],
            SecurityLevel.MEDIUM: [
                '/tmp/chungoid/',
                '/var/log/chungoid/',
                '/opt/chungoid/data/',
            ],
            SecurityLevel.HIGH: [
                '/tmp/',
                '/var/log/',
                '/opt/chungoid/',
                '/home/*/chungoid/',
            ],
            SecurityLevel.CRITICAL: [
                '/',  # Full access (use with extreme caution)
            ]
        }
        
        # Resource usage limits
        self.resource_limits = {
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'max_files_per_operation': 1000,
            'max_memory_mb': 512,
            'max_execution_time': 300,  # 5 minutes
        }
        
        # Track resource usage
        self.resource_usage = {}
    
    def validate_file_access(self, agent_id: str, file_path: str, 
                           security_level: SecurityLevel) -> SecurityAssessment:
        """
        Validate if an agent can access a specific file path.
        
        Args:
            agent_id: ID of the agent requesting access
            file_path: Path to the file
            security_level: Security level of the agent
            
        Returns:
            SecurityAssessment with validation results
        """
        violations = []
        warnings = []
        
        # Normalize path
        normalized_path = os.path.normpath(file_path)
        
        # Check for path traversal attempts
        if '..' in normalized_path or normalized_path.startswith('/'):
            if not any(normalized_path.startswith(allowed) 
                      for allowed in self.allowed_paths.get(security_level, [])):
                violations.append(SecurityViolation(
                    violation_type=VulnerabilityType.UNAUTHORIZED_ACCESS,
                    severity=SecurityLevel.HIGH,
                    description=f"Unauthorized file access attempt: {file_path}",
                    agent_id=agent_id,
                    remediation="Restrict file access to allowed directories"
                ))
        
        # Check if path is in allowed directories
        allowed_dirs = self.allowed_paths.get(security_level, [])
        if not any(normalized_path.startswith(allowed) for allowed in allowed_dirs):
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.UNAUTHORIZED_ACCESS,
                severity=SecurityLevel.MEDIUM,
                description=f"File access outside allowed directories: {file_path}",
                agent_id=agent_id,
                remediation="Access only files in allowed directories"
            ))
        
        # Check file size if it exists
        if os.path.exists(normalized_path):
            file_size = os.path.getsize(normalized_path)
            if file_size > self.resource_limits['max_file_size']:
                violations.append(SecurityViolation(
                    violation_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                    severity=SecurityLevel.MEDIUM,
                    description=f"File too large: {file_size} bytes",
                    agent_id=agent_id,
                    remediation="Use smaller files or implement streaming"
                ))
        
        return SecurityAssessment(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )
    
    def track_resource_usage(self, agent_id: str, resource_type: str, amount: float):
        """Track resource usage for an agent."""
        if agent_id not in self.resource_usage:
            self.resource_usage[agent_id] = {}
        
        if resource_type not in self.resource_usage[agent_id]:
            self.resource_usage[agent_id][resource_type] = 0
        
        self.resource_usage[agent_id][resource_type] += amount
    
    def check_resource_limits(self, agent_id: str) -> SecurityAssessment:
        """Check if an agent has exceeded resource limits."""
        violations = []
        warnings = []
        
        usage = self.resource_usage.get(agent_id, {})
        
        # Check memory usage
        memory_usage = usage.get('memory_mb', 0)
        if memory_usage > self.resource_limits['max_memory_mb']:
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                severity=SecurityLevel.HIGH,
                description=f"Memory usage exceeded: {memory_usage}MB > {self.resource_limits['max_memory_mb']}MB",
                agent_id=agent_id,
                remediation="Optimize memory usage or increase limits"
            ))
        
        # Check execution time
        execution_time = usage.get('execution_time', 0)
        if execution_time > self.resource_limits['max_execution_time']:
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.RESOURCE_EXHAUSTION,
                severity=SecurityLevel.MEDIUM,
                description=f"Execution time exceeded: {execution_time}s > {self.resource_limits['max_execution_time']}s",
                agent_id=agent_id,
                remediation="Optimize execution or implement timeouts"
            ))
        
        return SecurityAssessment(
            passed=len(violations) == 0,
            violations=violations,
            warnings=warnings
        )


class SecurityValidator:
    """
    Main security validation coordinator for the enhanced agent system.
    
    Integrates all security validation components and provides comprehensive
    security assessment capabilities.
    """
    
    def __init__(self):
        self.input_sanitizer = InputSanitizer()
        self.capability_validator = AgentCapabilityValidator()
        self.access_controller = ResourceAccessController()
        
        # Security configuration
        self.security_config = {
            'strict_mode': True,
            'log_violations': True,
            'block_critical_violations': True,
            'monitoring_enabled': True
        }
        
        # Violation history for analysis
        self.violation_history = []
        
        logger.info("Security Validator initialized")
    
    async def validate_agent_execution(self, agent_id: str, task_type: str, 
                                     input_data: Any, capabilities: List[str]) -> SecurityAssessment:
        """
        Perform comprehensive security validation for agent execution.
        
        Args:
            agent_id: ID of the agent to execute
            task_type: Type of task being performed
            input_data: Input data for the task
            capabilities: Agent capabilities
            
        Returns:
            SecurityAssessment with comprehensive validation results
        """
        all_violations = []
        all_warnings = []
        all_recommendations = []
        
        # 1. Validate input data
        sanitized_input, input_violations = self.input_sanitizer.sanitize_input(input_data)
        all_violations.extend(input_violations)
        
        # 2. Validate agent capabilities
        capability_assessment = self.capability_validator.validate_agent_capabilities(agent_id, capabilities)
        all_violations.extend(capability_assessment.violations)
        all_warnings.extend(capability_assessment.warnings)
        all_recommendations.extend(capability_assessment.recommendations)
        
        # 3. Check resource limits
        resource_assessment = self.access_controller.check_resource_limits(agent_id)
        all_violations.extend(resource_assessment.violations)
        all_warnings.extend(resource_assessment.warnings)
        
        # 4. Task-specific validation
        task_violations = await self._validate_task_specific_security(agent_id, task_type, sanitized_input)
        all_violations.extend(task_violations)
        
        # Create comprehensive assessment
        assessment = SecurityAssessment(
            passed=len(all_violations) == 0 or not any(v.severity == SecurityLevel.CRITICAL for v in all_violations),
            violations=all_violations,
            warnings=all_warnings,
            recommendations=all_recommendations
        )
        
        # Log violations if configured
        if self.security_config['log_violations']:
            await self._log_security_assessment(agent_id, task_type, assessment)
        
        # Store in history
        self.violation_history.extend(all_violations)
        
        return assessment
    
    async def _validate_task_specific_security(self, agent_id: str, task_type: str, 
                                             input_data: Any) -> List[SecurityViolation]:
        """Validate security for specific task types."""
        violations = []
        
        # File operations require special validation
        if task_type == 'file_operations':
            if isinstance(input_data, dict) and 'file_path' in input_data:
                file_path = input_data['file_path']
                # Determine security level based on agent capabilities
                security_level = SecurityLevel.MEDIUM  # Default
                
                file_assessment = self.access_controller.validate_file_access(
                    agent_id, file_path, security_level
                )
                violations.extend(file_assessment.violations)
        
        # Code generation requires input validation
        elif task_type == 'code_generation':
            if isinstance(input_data, dict) and 'code' in input_data:
                code = input_data['code']
                if isinstance(code, str):
                    # Check for potentially dangerous code patterns
                    dangerous_code_patterns = [
                        r'exec\s*\(',
                        r'eval\s*\(',
                        r'__import__',
                        r'subprocess\.',
                        r'os\.system',
                    ]
                    
                    for pattern in dangerous_code_patterns:
                        if re.search(pattern, code, re.IGNORECASE):
                            violations.append(SecurityViolation(
                                violation_type=VulnerabilityType.INJECTION,
                                severity=SecurityLevel.HIGH,
                                description=f"Potentially dangerous code pattern in generated code: {pattern}",
                                agent_id=agent_id,
                                task_type=task_type,
                                remediation="Review and sanitize generated code"
                            ))
        
        return violations
    
    async def _log_security_assessment(self, agent_id: str, task_type: str, 
                                     assessment: SecurityAssessment):
        """Log security assessment results."""
        if assessment.violations:
            logger.warning(f"Security violations for agent {agent_id} task {task_type}: "
                          f"{len(assessment.violations)} violations")
            
            for violation in assessment.violations:
                logger.warning(f"  {violation.severity.value.upper()}: {violation.description}")
        
        if assessment.warnings:
            for warning in assessment.warnings:
                logger.info(f"Security warning for agent {agent_id}: {warning}")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        # Analyze violation history
        violation_counts = {}
        severity_counts = {}
        agent_violations = {}
        
        for violation in self.violation_history:
            # Count by type
            violation_type = violation.violation_type.value
            violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1
            
            # Count by severity
            severity = violation.severity.value
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            # Count by agent
            if violation.agent_id:
                agent_violations[violation.agent_id] = agent_violations.get(violation.agent_id, 0) + 1
        
        return {
            'summary': {
                'total_violations': len(self.violation_history),
                'unique_violation_types': len(violation_counts),
                'agents_with_violations': len(agent_violations),
                'security_config': self.security_config
            },
            'violation_analysis': {
                'by_type': violation_counts,
                'by_severity': severity_counts,
                'by_agent': agent_violations
            },
            'top_violations': [
                violation.to_dict() for violation in 
                sorted(self.violation_history, key=lambda v: v.timestamp, reverse=True)[:10]
            ],
            'recommendations': self._generate_security_recommendations(violation_counts, severity_counts)
        }
    
    def _generate_security_recommendations(self, violation_counts: Dict[str, int], 
                                         severity_counts: Dict[str, int]) -> List[str]:
        """Generate security recommendations based on violation patterns."""
        recommendations = []
        
        # Check for common violation patterns
        if violation_counts.get('injection', 0) > 5:
            recommendations.append("Implement stricter input validation and sanitization")
        
        if violation_counts.get('resource_exhaustion', 0) > 3:
            recommendations.append("Review and adjust resource limits")
        
        if violation_counts.get('unauthorized_access', 0) > 2:
            recommendations.append("Strengthen access control policies")
        
        if severity_counts.get('critical', 0) > 0:
            recommendations.append("Immediately review and address critical security violations")
        
        if severity_counts.get('high', 0) > 5:
            recommendations.append("Implement additional monitoring for high-severity operations")
        
        return recommendations
    
    async def run_security_scan(self, agent_registry) -> SecurityAssessment:
        """
        Run comprehensive security scan of the agent system.
        
        Args:
            agent_registry: Agent registry to scan
            
        Returns:
            SecurityAssessment with scan results
        """
        all_violations = []
        all_warnings = []
        all_recommendations = []
        
        logger.info("Starting comprehensive security scan")
        
        # Scan all registered agents
        for agent_info in agent_registry.list_all_agents():
            agent_assessment = self.capability_validator.validate_agent_capabilities(
                agent_info.agent_id, 
                getattr(agent_info, 'capabilities', [])
            )
            
            all_violations.extend(agent_assessment.violations)
            all_warnings.extend(agent_assessment.warnings)
            all_recommendations.extend(agent_assessment.recommendations)
        
        # Check system configuration
        config_violations = await self._scan_system_configuration()
        all_violations.extend(config_violations)
        
        logger.info(f"Security scan completed: {len(all_violations)} violations found")
        
        return SecurityAssessment(
            passed=len(all_violations) == 0,
            violations=all_violations,
            warnings=all_warnings,
            recommendations=all_recommendations
        )
    
    async def _scan_system_configuration(self) -> List[SecurityViolation]:
        """Scan system configuration for security issues."""
        violations = []
        
        # Check if strict mode is enabled
        if not self.security_config.get('strict_mode', False):
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.UNAUTHORIZED_ACCESS,
                severity=SecurityLevel.MEDIUM,
                description="Strict security mode is disabled",
                remediation="Enable strict security mode for production"
            ))
        
        # Check if violation logging is enabled
        if not self.security_config.get('log_violations', False):
            violations.append(SecurityViolation(
                violation_type=VulnerabilityType.DATA_EXPOSURE,
                severity=SecurityLevel.LOW,
                description="Security violation logging is disabled",
                remediation="Enable violation logging for security monitoring"
            ))
        
        return violations


# Global security validator instance
_global_validator: Optional[SecurityValidator] = None


def get_security_validator() -> SecurityValidator:
    """Get or create global security validator instance."""
    global _global_validator
    
    if _global_validator is None:
        _global_validator = SecurityValidator()
    
    return _global_validator


async def validate_agent_security(agent_id: str, task_type: str, input_data: Any, 
                                capabilities: List[str]) -> SecurityAssessment:
    """Convenience function for agent security validation."""
    validator = get_security_validator()
    return await validator.validate_agent_execution(agent_id, task_type, input_data, capabilities)


def get_security_report() -> Dict[str, Any]:
    """Get current security report."""
    validator = get_security_validator()
    return validator.get_security_report() 