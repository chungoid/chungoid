"""
Evaluation Protocols Module

Week 6 Implementation: Production Readiness & Comprehensive Evaluation

This module contains protocols for comprehensive evaluation and production readiness:
- AutonomousExecutionEvaluatorProtocol: Comprehensive evaluation framework with real tool metrics
- ProductionDeploymentProtocol: Production deployment preparation and validation
- SystemValidationProtocol: Final integration testing and validation

These protocols provide comprehensive evaluation capabilities for production deployment,
including real tool metrics collection, performance analysis, and continuous improvement.
"""

from .autonomous_execution_evaluator import (
    AutonomousExecutionEvaluatorProtocol,
    EvaluationMetric,
    EvaluationResult,
    EvaluationConfig,
    MetricType,
    EvaluationSeverity,
    PerformanceAnalysis,
    OptimizationRecommendation
)

__all__ = [
    # Autonomous Execution Evaluator Protocol
    "AutonomousExecutionEvaluatorProtocol",
    "EvaluationMetric",
    "EvaluationResult", 
    "EvaluationConfig",
    "MetricType",
    "EvaluationSeverity",
    "PerformanceAnalysis",
    "OptimizationRecommendation"
] 