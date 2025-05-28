"""
Intelligence MCP Tools Module

This module provides MCP-compatible wrapper functions for intelligence tools
that handle the 'context' and 'domain' parameters agents expect to pass.

The wrappers maintain clean architectural separation between:
- Core intelligence functions (domain-focused, rigid signatures)  
- MCP tool interface (agent-friendly, flexible context parameters)
"""

from .mcp_wrappers import (
    adaptive_learning_analyze,
    create_strategy_experiment,
    apply_learning_recommendations,
    create_intelligent_recovery_plan,
    predict_potential_failures,
    analyze_historical_patterns,
    get_real_time_performance_analysis,
    optimize_agent_resolution_mcp,
    generate_performance_recommendations,
)

__all__ = [
    "adaptive_learning_analyze",
    "create_strategy_experiment", 
    "apply_learning_recommendations",
    "create_intelligent_recovery_plan",
    "predict_potential_failures",
    "analyze_historical_patterns",
    "get_real_time_performance_analysis",
    "optimize_agent_resolution_mcp",
    "generate_performance_recommendations",
]
