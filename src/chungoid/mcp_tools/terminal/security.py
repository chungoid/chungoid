"""
Terminal Security Module

Command classification and sandboxing for secure terminal operations.
Provides risk assessment and security controls for command execution.
"""

import logging
import re
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk levels for command classification."""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CommandPattern:
    """Pattern for command classification."""
    pattern: str
    risk_level: RiskLevel
    description: str
    allowed_contexts: Optional[Set[str]] = None


class CommandClassifier:
    """Classifies terminal commands based on security risk patterns."""
    
    def __init__(self):
        self.patterns = self._load_security_patterns()
        
    def _load_security_patterns(self) -> List[CommandPattern]:
        """Load command security patterns for classification."""
        return [
            # Safe commands - basic operations
            CommandPattern(r"^(pwd|ls|dir|echo|cat|head|tail|wc|grep|find|which|whoami|date|uptime)(\s|$)", 
                         RiskLevel.SAFE, "Basic read-only operations"),
            CommandPattern(r"^(git\s+(status|log|diff|show|branch|remote|config\s+--get))", 
                         RiskLevel.SAFE, "Safe git read operations"),
            CommandPattern(r"^(python\s+--version|node\s+--version|npm\s+--version|pip\s+--version)", 
                         RiskLevel.SAFE, "Version checking commands"),
            
            # Low risk - development operations
            CommandPattern(r"^(mkdir|touch|cp|mv)\s+[^/]", 
                         RiskLevel.LOW, "Basic file operations (relative paths)"),
            CommandPattern(r"^(python\s+-m\s+pip\s+(list|show|check))", 
                         RiskLevel.LOW, "Pip read operations"),
            CommandPattern(r"^(npm\s+(list|info|outdated|audit))", 
                         RiskLevel.LOW, "NPM read operations"),
            CommandPattern(r"^(pytest|python\s+-m\s+pytest)\s+", 
                         RiskLevel.LOW, "Test execution"),
            
            # Medium risk - package management and builds
            CommandPattern(r"^(python\s+-m\s+pip\s+(install|uninstall|upgrade))", 
                         RiskLevel.MEDIUM, "Pip package management"),
            CommandPattern(r"^(npm\s+(install|uninstall|update|run))", 
                         RiskLevel.MEDIUM, "NPM package management"),
            CommandPattern(r"^(make|cmake|gcc|g\+\+|javac|mvn|gradle)", 
                         RiskLevel.MEDIUM, "Build commands"),
            CommandPattern(r"^(git\s+(add|commit|push|pull|merge|clone))", 
                         RiskLevel.MEDIUM, "Git modification operations"),
            
            # High risk - system modifications
            CommandPattern(r"^(sudo|su)\s+", 
                         RiskLevel.HIGH, "Privilege escalation"),
            CommandPattern(r"^(rm|del)\s+.*(-r|-rf|--recursive)", 
                         RiskLevel.HIGH, "Recursive deletion"),
            CommandPattern(r"^(chmod|chown|chgrp)\s+", 
                         RiskLevel.HIGH, "Permission modifications"),
            CommandPattern(r"^(systemctl|service|systemd)", 
                         RiskLevel.HIGH, "System service management"),
            
            # Critical risk - dangerous operations
            CommandPattern(r"^(rm|del)\s+.*(/|\*|~)", 
                         RiskLevel.CRITICAL, "Dangerous deletion patterns"),
            CommandPattern(r"^(format|fdisk|mkfs|dd|wipefs)", 
                         RiskLevel.CRITICAL, "Disk formatting/manipulation"),
            CommandPattern(r"^(curl|wget).*(\||>|>>|\$\(|\`)", 
                         RiskLevel.CRITICAL, "Network download with execution"),
            CommandPattern(r"^(nc|netcat|ncat)\s+.*-[el]", 
                         RiskLevel.CRITICAL, "Network listeners"),
            CommandPattern(r"^(eval|exec)\s+", 
                         RiskLevel.CRITICAL, "Dynamic code execution"),
        ]
    
    def classify(self, command: str) -> Dict[str, Any]:
        """Classify a command and return risk assessment."""
        command = command.strip()
        if not command:
            return {
                "risk_level": "unknown",
                "description": "Empty command",
                "matched_patterns": [],
                "recommendations": ["Provide a valid command"]
            }
        
        matched_patterns = []
        highest_risk = RiskLevel.SAFE
        
        # Check against all patterns
        for pattern in self.patterns:
            if re.search(pattern.pattern, command, re.IGNORECASE):
                matched_patterns.append({
                    "pattern": pattern.pattern,
                    "risk_level": pattern.risk_level.value,
                    "description": pattern.description
                })
                
                # Track highest risk level
                if self._risk_priority(pattern.risk_level) > self._risk_priority(highest_risk):
                    highest_risk = pattern.risk_level
        
        # If no patterns matched, classify as unknown (medium risk)
        if not matched_patterns:
            highest_risk = RiskLevel.MEDIUM
            matched_patterns.append({
                "pattern": "unrecognized",
                "risk_level": "medium",
                "description": "Command not recognized by security patterns"
            })
        
        # Generate recommendations based on risk level
        recommendations = self._generate_recommendations(highest_risk, command)
        
        # Additional analysis
        analysis = self._analyze_command_features(command)
        
        return {
            "risk_level": highest_risk.value,
            "description": f"Command classified as {highest_risk.value} risk",
            "matched_patterns": matched_patterns,
            "recommendations": recommendations,
            "analysis": analysis,
            "command": command,
        }
    
    def _risk_priority(self, risk_level: RiskLevel) -> int:
        """Get numeric priority for risk level comparison."""
        priorities = {
            RiskLevel.SAFE: 0,
            RiskLevel.LOW: 1,
            RiskLevel.MEDIUM: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return priorities.get(risk_level, 2)
    
    def _generate_recommendations(self, risk_level: RiskLevel, command: str) -> List[str]:
        """Generate security recommendations based on risk level."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "CRITICAL: This command poses severe security risks",
                "Consider alternative approaches or manual execution",
                "Ensure you understand all implications before proceeding"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "HIGH RISK: This command requires careful review",
                "Verify the command is necessary and safe",
                "Consider running in a sandboxed environment"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "MEDIUM RISK: Review command before execution",
                "Ensure this operation is intended",
                "Check for unintended side effects"
            ])
        elif risk_level == RiskLevel.LOW:
            recommendations.append("LOW RISK: Generally safe operation")
        else:
            recommendations.append("SAFE: No significant security concerns")
        
        # Add specific recommendations based on command content
        if "install" in command.lower():
            recommendations.append("Package installation: Verify package source and integrity")
        if "rm" in command.lower() or "del" in command.lower():
            recommendations.append("File deletion: Double-check file paths and backup important data")
        if "sudo" in command.lower():
            recommendations.append("Privilege escalation: Ensure operation requires elevated privileges")
            
        return recommendations
    
    def _analyze_command_features(self, command: str) -> Dict[str, Any]:
        """Analyze command features for additional context."""
        analysis = {
            "has_pipes": "|" in command,
            "has_redirects": any(op in command for op in [">", ">>", "<"]),
            "has_background": "&" in command,
            "has_variables": "$" in command or "`" in command,
            "has_wildcards": any(char in command for char in ["*", "?", "["]),
            "has_network": any(cmd in command.lower() for cmd in ["curl", "wget", "ssh", "scp", "rsync"]),
            "has_sudo": "sudo" in command.lower(),
            "word_count": len(command.split()),
            "char_count": len(command),
        }
        
        # Risk indicators
        risk_indicators = []
        if analysis["has_pipes"] and analysis["has_variables"]:
            risk_indicators.append("Complex piping with variable expansion")
        if analysis["has_network"] and analysis["has_redirects"]:
            risk_indicators.append("Network operations with output redirection")
        if analysis["has_sudo"] and analysis["has_wildcards"]:
            risk_indicators.append("Privileged operations with wildcards")
        if analysis["word_count"] > 20:
            risk_indicators.append("Very long command - review carefully")
            
        analysis["risk_indicators"] = risk_indicators
        return analysis


class SecuritySandbox:
    """Manages security sandboxing for terminal command execution."""
    
    def __init__(self):
        self.sandbox_enabled = self._check_sandbox_availability()
        
    def _check_sandbox_availability(self) -> bool:
        """Check if sandboxing capabilities are available."""
        # Check for common sandboxing tools
        sandbox_tools = ["firejail", "bubblewrap", "systemd-run"]
        
        for tool in sandbox_tools:
            try:
                import subprocess
                subprocess.run([tool, "--version"], 
                             capture_output=True, timeout=5, check=False)
                logger.info(f"Sandbox tool available: {tool}")
                return True
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
                
        logger.warning("No sandbox tools found - running without sandboxing")
        return False
    
    def setup(self, classification: Dict[str, Any], working_directory: Path) -> Dict[str, Any]:
        """Setup sandbox based on command classification."""
        risk_level = classification.get("risk_level", "medium")
        
        sandbox_config = {
            "enabled": self.sandbox_enabled,
            "risk_level": risk_level,
            "restrictions": [],
            "allowed_paths": [str(working_directory)],
            "blocked_paths": [],
            "network_access": True,
            "filesystem_access": "read-write",
        }
        
        # Configure restrictions based on risk level
        if risk_level == "critical":
            sandbox_config.update({
                "network_access": False,
                "filesystem_access": "read-only",
                "restrictions": [
                    "no-network",
                    "no-system-calls",
                    "read-only-filesystem",
                    "no-privilege-escalation"
                ]
            })
        elif risk_level == "high":
            sandbox_config.update({
                "network_access": False,
                "restrictions": [
                    "no-network",
                    "no-privilege-escalation",
                    "limited-filesystem"
                ]
            })
        elif risk_level == "medium":
            sandbox_config.update({
                "restrictions": [
                    "no-privilege-escalation",
                    "limited-network"
                ]
            })
        
        # Add common system paths to blocked list for high/critical risk
        if risk_level in ["high", "critical"]:
            sandbox_config["blocked_paths"].extend([
                "/etc", "/usr/bin", "/usr/sbin", "/sbin", "/bin",
                "/var", "/opt", "/root"
            ])
        
        logger.info(f"Sandbox configured for {risk_level} risk level")
        return sandbox_config


# Standalone security functions for direct use

async def terminal_classify_command(
    command: str,
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Classify a terminal command for security risk assessment.
    
    Args:
        command: Command to classify
        context: Optional execution context for enhanced classification
        
    Returns:
        Dict containing classification results
    """
    try:
        classifier = CommandClassifier()
        result = classifier.classify(command)
        
        if context:
            result["context"] = context
            
        result["timestamp"] = logger.info(f"Command classified: {command[:50]}... -> {result['risk_level']}")
        return result
        
    except Exception as e:
        logger.error(f"Command classification failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "risk_level": "unknown",
            "command": command,
        }


async def terminal_check_permissions(
    command: str,
    working_directory: Optional[str] = None,
    project_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Check permissions and access rights for command execution.
    
    Args:
        command: Command to check
        working_directory: Working directory for execution
        project_path: Project directory path
        
    Returns:
        Dict containing permission check results
    """
    try:
        work_dir = Path(working_directory).resolve() if working_directory else Path.cwd()
        
        permissions = {
            "working_directory_writable": os.access(work_dir, os.W_OK),
            "working_directory_readable": os.access(work_dir, os.R_OK),
            "working_directory_executable": os.access(work_dir, os.X_OK),
            "user_id": os.getuid() if hasattr(os, 'getuid') else None,
            "effective_user_id": os.geteuid() if hasattr(os, 'geteuid') else None,
            "group_id": os.getgid() if hasattr(os, 'getgid') else None,
        }
        
        # Check if command requires specific permissions
        requires_write = any(op in command.lower() for op in ["install", "update", "rm", "del", "mv", "cp"])
        requires_network = any(cmd in command.lower() for cmd in ["curl", "wget", "git", "ssh"])
        requires_sudo = "sudo" in command.lower()
        
        recommendations = []
        if requires_write and not permissions["working_directory_writable"]:
            recommendations.append("Command requires write access - check directory permissions")
        if requires_sudo:
            recommendations.append("Command requires elevated privileges - ensure necessary")
        if requires_network:
            recommendations.append("Command requires network access - verify connectivity")
            
        return {
            "success": True,
            "permissions": permissions,
            "requirements": {
                "requires_write": requires_write,
                "requires_network": requires_network,
                "requires_sudo": requires_sudo,
            },
            "recommendations": recommendations,
            "working_directory": str(work_dir),
            "command": command,
        }
        
    except Exception as e:
        logger.error(f"Permission check failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "command": command,
        }


async def terminal_sandbox_status(
    project_path: Optional[str] = None,
    project_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Get current sandbox status and available security features.
    
    Args:
        project_path: Project directory path
        project_id: Project identifier
        
    Returns:
        Dict containing sandbox status information
    """
    try:
        sandbox = SecuritySandbox()
        
        # Check available security tools
        security_tools = {}
        tools_to_check = ["firejail", "bubblewrap", "systemd-run", "docker", "podman"]
        
        for tool in tools_to_check:
            try:
                import subprocess
                result = subprocess.run([tool, "--version"], 
                                     capture_output=True, timeout=5, check=False)
                security_tools[tool] = {
                    "available": result.returncode == 0,
                    "version": result.stdout.decode().strip()[:100] if result.returncode == 0 else None
                }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                security_tools[tool] = {"available": False, "version": None}
        
        return {
            "success": True,
            "sandbox_enabled": sandbox.sandbox_enabled,
            "available_tools": security_tools,
            "recommended_tools": ["firejail", "bubblewrap"],
            "project_path": project_path,
            "project_id": project_id,
            "security_features": {
                "command_classification": True,
                "permission_checking": True,
                "resource_monitoring": True,
                "timeout_protection": True,
            }
        }
        
    except Exception as e:
        logger.error(f"Sandbox status check failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
        } 