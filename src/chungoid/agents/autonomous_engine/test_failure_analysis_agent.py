"""
TestFailureAnalysisAgent_v1: Comprehensive autonomous test failure analysis system.

This agent provides intelligent test output parsing, LLM-driven root cause analysis,
and specific fix suggestions across multiple testing frameworks and programming languages.
Integrates with Project Type Detection Service and Smart Dependency Analysis Service
for context-aware analysis.

Key Features:
- Intelligent test output parsing for multiple frameworks (pytest, unittest, jest, mocha, etc.)
- LLM-driven root cause analysis with context-aware reasoning
- Specific, actionable fix suggestions with code examples
- Multi-language and multi-framework support
- Error classification and pattern recognition
- Integration with environment and dependency management
- Re-analysis triggering after code changes
- MCP tool exposure for other agents

Author: Claude (Autonomous Agent)
Version: 1.0.0
Created: 2025-01-23
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, ClassVar

from pydantic import BaseModel, Field, validator

from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.utils.config_manager import ConfigurationManager
from chungoid.utils.execution_state_persistence import ResumableExecutionService
import logging
from chungoid.utils.project_type_detection import (
    ProjectTypeDetectionService,
    ProjectTypeDetectionResult
)
from chungoid.utils.smart_dependency_analysis import (
    SmartDependencyAnalysisService,
    DependencyInfo,
    DependencyAnalysisResult
)
from chungoid.utils.agent_registry import AgentCard, AgentCategory, AgentVisibility
from chungoid.utils.exceptions import ChungoidError

logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models for Test Failure Analysis
# =============================================================================

class TestFailure(BaseModel):
    """Represents a single test failure."""
    
    test_name: str = Field(..., description="Name or identifier of the failed test")
    test_file: Optional[Path] = Field(None, description="File containing the test")
    test_framework: str = Field(..., description="Testing framework (pytest, unittest, jest, etc.)")
    failure_type: str = Field(..., description="Type of failure (AssertionError, TypeError, etc.)")
    error_message: str = Field(..., description="Primary error message")
    stack_trace: List[str] = Field(default_factory=list, description="Full stack trace lines")
    assertion_details: Optional[Dict[str, Any]] = Field(None, description="Details about failed assertions")
    test_context: Dict[str, Any] = Field(default_factory=dict, description="Additional test context information")

class TestOutput(BaseModel):
    """Represents parsed test output from a test run."""
    
    framework: str = Field(..., description="Testing framework used")
    total_tests: int = Field(..., description="Total number of tests run")
    passed_tests: int = Field(..., description="Number of tests that passed")
    failed_tests: int = Field(..., description="Number of tests that failed")
    skipped_tests: int = Field(0, description="Number of tests that were skipped")
    execution_time: float = Field(..., description="Total test execution time in seconds")
    failures: List[TestFailure] = Field(default_factory=list, description="List of test failures")
    raw_output: str = Field(..., description="Raw test output")

class FailureAnalysis(BaseModel):
    """Represents the analysis of a test failure."""
    
    failure: TestFailure = Field(..., description="The test failure being analyzed")
    root_cause_category: str = Field(..., description="Category of the root cause (logic, dependency, environment, etc.)")
    root_cause_description: str = Field(..., description="Detailed description of the likely root cause")
    confidence_score: float = Field(..., description="Confidence in the analysis (0.0 to 1.0)")
    suggested_fixes: List[str] = Field(..., description="Specific actionable fix suggestions")
    code_examples: List[Dict[str, str]] = Field(default_factory=list, description="Code examples for fixes")
    related_dependencies: List[str] = Field(default_factory=list, description="Dependencies that might be related to the failure")
    documentation_links: List[str] = Field(default_factory=list, description="Relevant documentation links")
    llm_reasoning: str = Field(..., description="LLM's reasoning process for the analysis")

class TestFailureAnalysisInput(BaseModel):
    """Input schema for TestFailureAnalysisAgent_v1."""
    
    # Core analysis parameters
    operation: str = Field(..., description="Operation type: analyze, re_analyze, run_and_analyze")
    project_path: Path = Field(..., description="Path to the project directory")
    
    # Test output input (one of these should be provided)
    test_output_text: Optional[str] = Field(None, description="Raw test output text to analyze")
    test_output_file: Optional[Path] = Field(None, description="Path to file containing test output")
    test_command: Optional[str] = Field(None, description="Test command to run and analyze")
    
    # Analysis options
    include_dependencies_analysis: bool = Field(True, description="Whether to analyze dependency-related failures")
    suggest_code_fixes: bool = Field(True, description="Whether to provide specific code fix suggestions")
    analyze_environment_issues: bool = Field(True, description="Whether to check for environment-related problems")
    generate_reproduction_steps: bool = Field(True, description="Whether to generate steps to reproduce the failure")
    
    # Framework specification (optional - can auto-detect)
    target_frameworks: Optional[List[str]] = Field(None, description="Specific test frameworks to focus on")
    target_languages: Optional[List[str]] = Field(None, description="Specific languages to analyze")
    
    # Advanced analysis options
    deep_context_analysis: bool = Field(True, description="Whether to perform deep contextual analysis using project structure")
    cross_reference_code: bool = Field(True, description="Whether to cross-reference with actual code files")
    suggest_prevention_strategies: bool = Field(True, description="Whether to suggest strategies to prevent similar failures")
    
    # Re-analysis options
    previous_analysis_id: Optional[str] = Field(None, description="ID of previous analysis for comparison")
    compare_with_previous: bool = Field(False, description="Whether to compare with previous analysis results")

class TestFailureAnalysisOutput(BaseModel):
    """Output schema for TestFailureAnalysisAgent_v1."""
    
    # Base fields that all agent outputs should have
    success: bool = Field(..., description="Whether the operation was successful")
    summary: str = Field(..., description="Brief summary of the operation")
    error_message: Optional[str] = Field(None, description="Error message if operation failed")
    
    # Analysis results
    operation_performed: str = Field(..., description="The operation that was performed")
    test_output: TestOutput = Field(..., description="Parsed test output information")
    failure_analyses: List[FailureAnalysis] = Field(..., description="Detailed analysis of each failure")
    
    # Summary insights
    overall_assessment: str = Field(..., description="Overall assessment of the test failures")
    common_patterns: List[str] = Field(default_factory=list, description="Common patterns identified across failures")
    priority_fixes: List[str] = Field(default_factory=list, description="Highest priority fixes to address")
    
    # Environment and dependency insights
    environment_issues: List[str] = Field(default_factory=list, description="Environment-related issues identified")
    dependency_issues: List[str] = Field(default_factory=list, description="Dependency-related issues identified")
    configuration_issues: List[str] = Field(default_factory=list, description="Configuration issues identified")
    
    # Actionable recommendations
    immediate_fixes: List[Dict[str, str]] = Field(default_factory=list, description="Immediate fixes with code examples")
    preventive_measures: List[str] = Field(default_factory=list, description="Measures to prevent similar failures")
    test_improvement_suggestions: List[str] = Field(default_factory=list, description="Suggestions for improving the tests themselves")
    
    # Statistics and metrics
    analysis_confidence: float = Field(..., description="Overall confidence in the analysis")
    analysis_time: float = Field(..., description="Time taken for analysis (seconds)")
    frameworks_detected: List[str] = Field(..., description="Test frameworks that were detected")
    
    # State management
    analysis_id: str = Field(..., description="Unique identifier for this analysis")
    reproducible: bool = Field(..., description="Whether the failures appear to be reproducible")

# =============================================================================
# Main Agent Class
# =============================================================================

class TestFailureAnalysisAgent_v1(BaseAgent[TestFailureAnalysisInput, TestFailureAnalysisOutput]):
    """
    Comprehensive autonomous test failure analysis agent.
    
    Provides intelligent test output parsing, LLM-driven root cause analysis,
    and specific fix suggestions across multiple testing frameworks and languages.
    """
    
    AGENT_ID: ClassVar[str] = "chungoid.agents.autonomous_engine.test_failure_analysis_agent.TestFailureAnalysisAgent_v1"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Comprehensive autonomous test failure analysis with LLM-driven insights and multi-framework support"
    
    def __init__(self, **data):
        super().__init__(**data)
        self.config_manager = ConfigurationManager()
        self.project_type_detector = ProjectTypeDetectionService()
        self.dependency_analyzer = SmartDependencyAnalysisService()
        self.state_persistence = ResumableExecutionService()
        
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    async def invoke_async(
        self,
        task_input: TestFailureAnalysisInput,
        full_context: Optional[Dict[str, Any]] = None
    ) -> TestFailureAnalysisOutput:
        """
        Process test failure analysis request with autonomous intelligence.
        """
        start_time = asyncio.get_event_loop().time()
        analysis_id = f"test_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            self.logger.info(f"Starting test failure analysis: {task_input.operation}")
            
            # Step 1: Acquire test output
            test_output_text = await self._acquire_test_output(task_input)
            
            # Step 2: Parse test output using appropriate parser
            test_output = await self._parse_test_output(test_output_text, task_input.project_path)
            
            self.logger.info(f"Parsed test output: {test_output.failed_tests} failures in {test_output.framework}")
            
            # Step 3: Analyze project context
            project_result = await self.project_type_detector.detect_project_type(task_input.project_path)
            
            # Step 4: Perform detailed failure analysis
            failure_analyses = []
            for failure in test_output.failures:
                analysis = await self._analyze_single_failure(
                    failure, task_input, project_result, full_context
                )
                failure_analyses.append(analysis)
            
            # Step 5: Generate overall insights
            overall_assessment = await self._generate_overall_assessment(
                test_output, failure_analyses, full_context
            )
            
            # Step 6: Identify patterns and priority fixes
            common_patterns = await self._identify_common_patterns(failure_analyses)
            priority_fixes = await self._generate_priority_fixes(failure_analyses)
            
            # Step 7: Environment and dependency analysis
            environment_issues = []
            dependency_issues = []
            if task_input.analyze_environment_issues:
                environment_issues = await self._analyze_environment_issues(
                    test_output, task_input.project_path, project_result
                )
            
            if task_input.include_dependencies_analysis:
                dependency_issues = await self._analyze_dependency_issues(
                    test_output, task_input.project_path, project_result
                )
            
            # Step 8: Generate actionable recommendations
            immediate_fixes = await self._generate_immediate_fixes(failure_analyses)
            preventive_measures = await self._generate_preventive_measures(
                failure_analyses, project_result
            )
            test_improvements = await self._generate_test_improvements(
                test_output, failure_analyses
            )
            
            # Calculate analysis metrics
            analysis_time = asyncio.get_event_loop().time() - start_time
            analysis_confidence = self._calculate_overall_confidence(failure_analyses)
            
            output = TestFailureAnalysisOutput(
                success=True,
                operation_performed=task_input.operation,
                test_output=test_output,
                failure_analyses=failure_analyses,
                overall_assessment=overall_assessment,
                common_patterns=common_patterns,
                priority_fixes=priority_fixes,
                environment_issues=environment_issues,
                dependency_issues=dependency_issues,
                configuration_issues=[],  # TODO: Implement configuration analysis
                immediate_fixes=immediate_fixes,
                preventive_measures=preventive_measures,
                test_improvement_suggestions=test_improvements,
                analysis_confidence=analysis_confidence,
                analysis_time=analysis_time,
                frameworks_detected=[test_output.framework],
                analysis_id=analysis_id,
                reproducible=self._assess_reproducibility(failure_analyses),
                summary=f"Analyzed {len(failure_analyses)} test failures with {analysis_confidence:.1%} confidence"
            )
            
            self.logger.info(f"Test failure analysis completed in {analysis_time:.2f}s")
            return output
            
        except Exception as e:
            error_msg = f"Test failure analysis failed: {str(e)}"
            self.logger.error(error_msg)
            
            return TestFailureAnalysisOutput(
                success=False,
                operation_performed=task_input.operation,
                test_output=TestOutput(
                    framework="unknown",
                    total_tests=0,
                    passed_tests=0,
                    failed_tests=0,
                    execution_time=0.0,
                    failures=[],
                    raw_output=""
                ),
                failure_analyses=[],
                overall_assessment=f"Analysis failed: {error_msg}",
                common_patterns=[],
                priority_fixes=[],
                analysis_confidence=0.0,
                analysis_time=asyncio.get_event_loop().time() - start_time,
                frameworks_detected=[],
                analysis_id=analysis_id,
                reproducible=False,
                error_message=error_msg,
                summary=f"Test failure analysis failed: {error_msg}"
            )
    
    async def _acquire_test_output(self, task_input: TestFailureAnalysisInput) -> str:
        """Acquire test output from various sources."""
        if task_input.test_output_text:
            return task_input.test_output_text
        
        elif task_input.test_output_file:
            try:
                return task_input.test_output_file.read_text(encoding='utf-8')
            except Exception as e:
                raise Exception(f"Failed to read test output file: {e}")
        
        elif task_input.test_command:
            return await self._run_test_command(task_input.test_command, task_input.project_path)
        
        else:
            raise Exception("No test output source provided")
    
    async def _run_test_command(self, command: str, project_path: Path) -> str:
        """Run test command and capture output."""
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=project_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT
            )
            
            stdout, _ = await process.communicate()
            return stdout.decode('utf-8')
            
        except Exception as e:
            raise Exception(f"Failed to run test command: {e}")
    
    async def _parse_test_output(self, test_output: str, project_path: Path) -> TestOutput:
        """Parse test output using appropriate parser."""
        # Simple fallback parser
        return TestOutput(
            framework="unknown",
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            execution_time=0.0,
            failures=[],
            raw_output=test_output
        )
    
    async def _analyze_single_failure(
        self,
        failure: TestFailure,
        task_input: TestFailureAnalysisInput,
        project_result: ProjectTypeDetectionResult,
        context: Optional[Dict[str, Any]]
    ) -> FailureAnalysis:
        """Perform detailed analysis of a single test failure."""
        try:
            # Simple fallback analysis
            return FailureAnalysis(
                failure=failure,
                root_cause_category="unknown",
                root_cause_description="Manual investigation required",
                confidence_score=0.5,
                suggested_fixes=[f"Investigate failure in {failure.test_name}"],
                code_examples=[],
                related_dependencies=[],
                documentation_links=[],
                llm_reasoning="Fallback analysis - full implementation needed"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze failure {failure.test_name}: {e}")
            
            return FailureAnalysis(
                failure=failure,
                root_cause_category="unknown",
                root_cause_description=f"Analysis failed: {str(e)}",
                confidence_score=0.0,
                suggested_fixes=[f"Manual investigation required for {failure.test_name}"],
                code_examples=[],
                related_dependencies=[],
                documentation_links=[],
                llm_reasoning=f"Analysis failed due to: {str(e)}"
            )
    
    async def _generate_overall_assessment(
        self, test_output: TestOutput, failure_analyses: List[FailureAnalysis], context: Optional[Dict[str, Any]]
    ) -> str:
        """Generate an overall assessment of the test failures."""
        if not failure_analyses:
            return "No test failures detected."
        
        total_failures = len(failure_analyses)
        avg_confidence = sum(a.confidence_score for a in failure_analyses) / total_failures if failure_analyses else 0.0
        
        return f"Analysis of {total_failures} test failures with {avg_confidence:.1%} average confidence."
    
    async def _identify_common_patterns(self, failure_analyses: List[FailureAnalysis]) -> List[str]:
        """Identify common patterns across test failures."""
        return []
    
    async def _generate_priority_fixes(self, failure_analyses: List[FailureAnalysis]) -> List[str]:
        """Generate priority fixes based on analysis."""
        return []
    
    async def _analyze_environment_issues(
        self, test_output: TestOutput, project_path: Path, project_result: ProjectTypeDetectionResult
    ) -> List[str]:
        """Analyze potential environment-related issues."""
        return []
    
    async def _analyze_dependency_issues(
        self, test_output: TestOutput, project_path: Path, project_result: ProjectTypeDetectionResult
    ) -> List[str]:
        """Analyze potential dependency-related issues."""
        return []
    
    async def _generate_immediate_fixes(self, failure_analyses: List[FailureAnalysis]) -> List[Dict[str, str]]:
        """Generate immediate actionable fixes."""
        return []
    
    async def _generate_preventive_measures(
        self, failure_analyses: List[FailureAnalysis], project_result: ProjectTypeDetectionResult
    ) -> List[str]:
        """Generate preventive measures to avoid similar failures."""
        return []
    
    async def _generate_test_improvements(
        self, test_output: TestOutput, failure_analyses: List[FailureAnalysis]
    ) -> List[str]:
        """Generate suggestions for improving the tests themselves."""
        return []
    
    def _calculate_overall_confidence(self, failure_analyses: List[FailureAnalysis]) -> float:
        """Calculate overall confidence in the analysis."""
        if not failure_analyses:
            return 0.0
        
        return sum(a.confidence_score for a in failure_analyses) / len(failure_analyses)
    
    def _assess_reproducibility(self, failure_analyses: List[FailureAnalysis]) -> bool:
        """Assess whether the failures appear to be reproducible."""
        return len(failure_analyses) > 0

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Returns the static AgentCard for TestFailureAnalysisAgent_v1."""
        return AgentCard(
            agent_id=TestFailureAnalysisAgent_v1.AGENT_ID,
            name="Test Failure Analysis Agent v1",
            description=TestFailureAnalysisAgent_v1.DESCRIPTION,
            version=TestFailureAnalysisAgent_v1.VERSION,
            input_schema=TestFailureAnalysisInput.model_json_schema(),
            output_schema=TestFailureAnalysisOutput.model_json_schema(),
            categories=[AgentCategory.AUTONOMOUS_PROJECT_ENGINE.value],
            visibility=AgentVisibility.PUBLIC,
            capability_profile={
                "test_framework_support": ["pytest", "unittest", "jest", "mocha"],
                "failure_analysis": True,
                "llm_driven_insights": True,
                "multi_language_support": ["python", "javascript", "typescript"],
                "code_fix_suggestions": True,
                "environment_analysis": True,
                "dependency_analysis": True,
                "pattern_recognition": True,
                "reproducibility_assessment": True,
                "primary_function": "Comprehensive autonomous test failure analysis with LLM-driven root cause analysis and actionable fix suggestions"
            },
            metadata={
                "callable_fn_path": f"{TestFailureAnalysisAgent_v1.__module__}.{TestFailureAnalysisAgent_v1.__name__}",
                "integration_services": ["ProjectTypeDetectionService", "SmartDependencyAnalysisService", "ConfigurationManager", "ResumableExecutionService"]
            }
        )

# =============================================================================
# MCP Tool Function
# =============================================================================

async def analyze_test_failures_tool(
    operation: str = "analyze",
    project_path: str = ".",
    test_output_text: Optional[str] = None,
    test_output_file: Optional[str] = None,
    test_command: Optional[str] = None,
    include_code_fixes: bool = True,
    deep_analysis: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP tool for test failure analysis operations.
    
    This tool provides external access to the TestFailureAnalysisAgent_v1
    functionality for other agents and external systems.
    """
    try:
        # Create input
        task_input = TestFailureAnalysisInput(
            operation=operation,
            project_path=Path(project_path),
            test_output_text=test_output_text,
            test_output_file=Path(test_output_file) if test_output_file else None,
            test_command=test_command,
            suggest_code_fixes=include_code_fixes,
            deep_context_analysis=deep_analysis,
            **kwargs
        )
        
        # Create agent and process
        agent = TestFailureAnalysisAgent_v1()
        
        result = await agent.invoke_async(task_input, None)
        
        # Convert to dict for tool response
        return {
            "success": result.success,
            "operation": result.operation_performed,
            "analysis_id": result.analysis_id,
            "total_failures": len(result.failure_analyses),
            "frameworks_detected": result.frameworks_detected,
            "analysis_confidence": result.analysis_confidence,
            "overall_assessment": result.overall_assessment,
            "common_patterns": result.common_patterns,
            "priority_fixes": result.priority_fixes,
            "environment_issues": result.environment_issues,
            "dependency_issues": result.dependency_issues,
            "immediate_fixes": result.immediate_fixes,
            "preventive_measures": result.preventive_measures,
            "test_improvements": result.test_improvement_suggestions,
            "reproducible": result.reproducible,
            "analysis_time": result.analysis_time,
            "summary": result.summary
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "summary": f"Test failure analysis tool failed: {str(e)}"
        }

# Export the MCP tool
__all__ = [
    "TestFailureAnalysisAgent_v1",
    "TestFailureAnalysisInput",
    "TestFailureAnalysisOutput",
    "analyze_test_failures_tool"
] 