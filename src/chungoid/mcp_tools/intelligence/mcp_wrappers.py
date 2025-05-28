"""
MCP Wrapper Functions for Intelligence Tools

This module provides MCP-compatible wrapper functions for intelligence tools that
handle the 'context' and 'domain' parameters agents expect to pass, while mapping
them appropriately to the actual function signatures.

The wrappers maintain clean architectural separation between:
- Core intelligence functions (domain-focused, rigid signatures)
- MCP tool interface (agent-friendly, flexible context parameters)

This resolves the "unexpected keyword argument 'context'" errors while preserving
both the clean intelligence architecture and MCP compatibility.
"""

import logging
from typing import Any, Dict, List, Optional

# Import the actual intelligence functions
from chungoid.intelligence.adaptive_learning_system import (
    adaptive_learning_analyze as _adaptive_learning_analyze,
    create_strategy_experiment as _create_strategy_experiment, 
    apply_learning_recommendations as _apply_learning_recommendations,
)
from chungoid.intelligence.advanced_replanning_intelligence import (
    create_intelligent_recovery_plan as _create_intelligent_recovery_plan,
    predict_potential_failures as _predict_potential_failures,
    analyze_historical_patterns as _analyze_historical_patterns,
)
from chungoid.runtime.performance_optimizer import (
    get_real_time_performance_analysis as _get_real_time_performance_analysis,
    optimize_agent_resolution_mcp as _optimize_agent_resolution_mcp,
    generate_performance_recommendations as _generate_performance_recommendations,
)

logger = logging.getLogger(__name__)


def _extract_project_context(context: Optional[Dict[str, Any]], domain: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract project context information from MCP context parameter.
    
    Args:
        context: Context dictionary from agent call
        domain: Domain/agent ID for additional context
        
    Returns:
        Dict containing extracted project context
    """
    if not context:
        return {}
        
    project_context = {}
    
    # Extract project identification
    if "project_id" in context:
        project_context["project_id"] = context["project_id"]
    elif "project_analysis" in context and isinstance(context["project_analysis"], dict):
        project_analysis = context["project_analysis"]
        if "project_id" in project_analysis:
            project_context["project_id"] = project_analysis["project_id"]
    
    # Extract project path information  
    if "project_path" in context:
        project_context["project_path"] = context["project_path"]
    elif "project_analysis" in context and isinstance(context["project_analysis"], dict):
        project_analysis = context["project_analysis"]
        if "project_path" in project_analysis:
            project_context["project_path"] = project_analysis["project_path"]
    
    # Add domain as context if provided
    if domain:
        project_context["requesting_agent"] = domain
        project_context["analysis_domain"] = domain
    
    return project_context


# ============================================================================
# Adaptive Learning System Wrappers
# ============================================================================

async def adaptive_learning_analyze(
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    project_id: Optional[str] = None,
    time_window_days: int = 30,
    enable_cross_project: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for adaptive_learning_analyze.
    
    Handles context and domain parameters from agents while mapping them
    to the appropriate parameters for the core adaptive learning function.
    
    Args:
        context: Contextual information from agent (project analysis, current state, etc.)
        domain: Agent domain/ID for context
        project_id: Specific project to analyze (extracted from context if not provided)
        time_window_days: Number of days of historical data to analyze
        enable_cross_project: Whether to include cross-project learning
        **kwargs: Additional parameters (for forward compatibility)
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    try:
        # Extract project context from MCP parameters
        project_context = _extract_project_context(context, domain)
        
        # Use project_id from context if not explicitly provided
        if project_id is None and "project_id" in project_context:
            project_id = project_context["project_id"]
        
        # Extract time window from context if provided
        if context and "time_window_days" in context:
            time_window_days = context["time_window_days"]
        elif context and "analysis_config" in context:
            analysis_config = context["analysis_config"] 
            if isinstance(analysis_config, dict) and "time_window_days" in analysis_config:
                time_window_days = analysis_config["time_window_days"]
        
        # Extract cross-project setting from context if provided
        if context and "enable_cross_project" in context:
            enable_cross_project = context["enable_cross_project"]
        
        logger.info(f"[MCP-Intelligence] Adaptive learning analysis for project {project_id or 'all'} by {domain or 'unknown'}")
        
        # Call the core function with mapped parameters
        result = await _adaptive_learning_analyze(
            project_id=project_id,
            time_window_days=time_window_days,
            enable_cross_project=enable_cross_project
        )
        
        # Enhance result with MCP context information
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "context_provided": context is not None,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Adaptive learning analysis failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain,
                "context_provided": context is not None
            }
        }


async def create_strategy_experiment(
    baseline_strategy: Dict[str, Any],
    test_strategy: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    success_metric: str = "success_rate",
    duration_days: int = 7,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for create_strategy_experiment.
    
    Args:
        baseline_strategy: Current baseline strategy configuration
        test_strategy: New strategy variant to test
        context: Contextual information from agent
        domain: Agent domain/ID for context
        success_metric: Metric to optimize for
        duration_days: How long to run the experiment
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing experiment details
    """
    try:
        # Extract context information
        project_context = _extract_project_context(context, domain)
        
        # Extract success metric from context if provided
        if context and "success_metric" in context:
            success_metric = context["success_metric"]
        elif context and "experiment_config" in context:
            experiment_config = context["experiment_config"]
            if isinstance(experiment_config, dict) and "success_metric" in experiment_config:
                success_metric = experiment_config["success_metric"]
        
        # Extract duration from context if provided
        if context and "duration_days" in context:
            duration_days = context["duration_days"]
        elif context and "experiment_config" in context:
            experiment_config = context["experiment_config"]
            if isinstance(experiment_config, dict) and "duration_days" in experiment_config:
                duration_days = experiment_config["duration_days"]
        
        logger.info(f"[MCP-Intelligence] Creating strategy experiment for {domain or 'unknown'}")
        
        # Call the core function
        result = await _create_strategy_experiment(
            baseline_strategy=baseline_strategy,
            test_strategy=test_strategy,
            success_metric=success_metric,
            duration_days=duration_days
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Strategy experiment creation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


async def apply_learning_recommendations(
    recommendations: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    auto_apply_threshold: float = 0.9,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for apply_learning_recommendations.
    
    Args:
        recommendations: List of recommendations to process
        context: Contextual information from agent
        domain: Agent domain/ID for context
        auto_apply_threshold: Confidence threshold for automatic application
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing application results
    """
    try:
        # Extract context information
        project_context = _extract_project_context(context, domain)
        
        # Extract auto-apply threshold from context if provided
        if context and "auto_apply_threshold" in context:
            auto_apply_threshold = context["auto_apply_threshold"]
        elif context and "application_config" in context:
            application_config = context["application_config"]
            if isinstance(application_config, dict) and "auto_apply_threshold" in application_config:
                auto_apply_threshold = application_config["auto_apply_threshold"]
        
        logger.info(f"[MCP-Intelligence] Applying learning recommendations for {domain or 'unknown'}")
        
        # Call the core function
        result = await _apply_learning_recommendations(
            recommendations=recommendations,
            auto_apply_threshold=auto_apply_threshold
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Learning recommendations application failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


# ============================================================================
# Advanced Replanning Intelligence Wrappers  
# ============================================================================

async def create_intelligent_recovery_plan(
    failure_context: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    project_context: Optional[Dict[str, Any]] = None,
    enable_research: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for create_intelligent_recovery_plan.
    
    Args:
        failure_context: Information about the failure to recover from
        context: Additional contextual information from agent
        domain: Agent domain/ID for context
        project_context: Project context (extracted from context if not provided)
        enable_research: Whether to enable autonomous research
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing recovery plan
    """
    try:
        # Extract and merge project context
        extracted_context = _extract_project_context(context, domain)
        if project_context is None:
            project_context = extracted_context
        else:
            project_context.update(extracted_context)
        
        # Extract enable_research from context if provided
        if context and "enable_research" in context:
            enable_research = context["enable_research"]
        elif context and "recovery_config" in context:
            recovery_config = context["recovery_config"]
            if isinstance(recovery_config, dict) and "enable_research" in recovery_config:
                enable_research = recovery_config["enable_research"]
        
        logger.info(f"[MCP-Intelligence] Creating recovery plan for {domain or 'unknown'}")
        
        # Call the core function
        result = await _create_intelligent_recovery_plan(
            failure_context=failure_context,
            project_context=project_context,
            enable_research=enable_research
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Recovery plan creation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


async def predict_potential_failures(
    current_context: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    execution_plan: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for predict_potential_failures.
    
    Args:
        current_context: Current execution context
        context: Additional contextual information from agent
        domain: Agent domain/ID for context
        execution_plan: Optional execution plan (extracted from context if not provided)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing failure predictions
    """
    try:
        # Extract project context
        project_context = _extract_project_context(context, domain)
        
        # Extract execution plan from context if not provided
        if execution_plan is None and context and "execution_plan" in context:
            execution_plan = context["execution_plan"]
        elif execution_plan is None and context and "current_plan" in context:
            execution_plan = context["current_plan"]
        
        logger.info(f"[MCP-Intelligence] Predicting failures for {domain or 'unknown'}")
        
        # Call the core function
        result = await _predict_potential_failures(
            current_context=current_context,
            execution_plan=execution_plan
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Failure prediction failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


async def analyze_historical_patterns(
    failure_context: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    project_context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for analyze_historical_patterns.
    
    Args:
        failure_context: Information about the failure to analyze
        context: Additional contextual information from agent
        domain: Agent domain/ID for context
        project_context: Project context (extracted from context if not provided)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing historical pattern analysis
    """
    try:
        # Extract and merge project context
        extracted_context = _extract_project_context(context, domain)
        if project_context is None:
            project_context = extracted_context
        else:
            project_context.update(extracted_context)
        
        logger.info(f"[MCP-Intelligence] Analyzing historical patterns for {domain or 'unknown'}")
        
        # Call the core function
        result = await _analyze_historical_patterns(
            failure_context=failure_context,
            project_context=project_context
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Historical pattern analysis failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


# ============================================================================
# Performance Optimizer Wrappers
# ============================================================================

async def get_real_time_performance_analysis(
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for get_real_time_performance_analysis.
    
    Args:
        context: Contextual information from agent
        domain: Agent domain/ID for context
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing current performance metrics and analysis
    """
    try:
        # Extract project context
        project_context = _extract_project_context(context, domain)
        
        logger.info(f"[MCP-Intelligence] Getting performance analysis for {domain or 'unknown'}")
        
        # Call the core function (no parameters needed)
        result = await _get_real_time_performance_analysis()
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Performance analysis failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


async def optimize_agent_resolution_mcp(
    task_type: str,
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    required_capabilities: Optional[List[str]] = None,
    prefer_autonomous: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for optimize_agent_resolution_mcp.
    
    Args:
        task_type: Type of task requiring agent resolution
        context: Contextual information from agent
        domain: Agent domain/ID for context
        required_capabilities: List of required capabilities
        prefer_autonomous: Whether to prefer autonomous agents
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing resolution result and optimization metrics
    """
    try:
        # Extract project context
        project_context = _extract_project_context(context, domain)
        
        # Extract required capabilities from context if not provided
        if required_capabilities is None and context and "required_capabilities" in context:
            required_capabilities = context["required_capabilities"]
        elif required_capabilities is None and context and "capabilities" in context:
            required_capabilities = context["capabilities"]
        
        # Extract prefer_autonomous from context if provided
        if context and "prefer_autonomous" in context:
            prefer_autonomous = context["prefer_autonomous"]
        
        logger.info(f"[MCP-Intelligence] Optimizing agent resolution for {domain or 'unknown'}")
        
        # Call the core function
        result = await _optimize_agent_resolution_mcp(
            task_type=task_type,
            required_capabilities=required_capabilities,
            prefer_autonomous=prefer_autonomous
        )
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Agent resolution optimization failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        }


async def generate_performance_recommendations(
    context: Optional[Dict[str, Any]] = None,
    domain: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    MCP wrapper for generate_performance_recommendations.
    
    Args:
        context: Contextual information from agent
        domain: Agent domain/ID for context
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing performance recommendations and insights
    """
    try:
        # Extract project context
        project_context = _extract_project_context(context, domain)
        
        logger.info(f"[MCP-Intelligence] Generating performance recommendations for {domain or 'unknown'}")
        
        # Call the core function (no parameters needed)
        result = await _generate_performance_recommendations()
        
        # Enhance result with MCP context
        if result.get("success"):
            result["mcp_context"] = {
                "requesting_agent": domain,
                "project_context": project_context
            }
        
        return result
        
    except Exception as e:
        logger.error(f"[MCP-Intelligence] Performance recommendations generation failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "mcp_context": {
                "requesting_agent": domain
            }
        } 