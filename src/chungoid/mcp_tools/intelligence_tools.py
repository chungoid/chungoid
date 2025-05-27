"""
Intelligence Tools Module

Centralized interface for intelligence system MCP tools.
Provides direct access to adaptive learning, advanced replanning, and performance optimization
capabilities for agent refinement cycles.
"""

import logging
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Import intelligence functions
try:
    from chungoid.intelligence.adaptive_learning_system import (
        adaptive_learning_analyze,
        create_strategy_experiment,
        apply_learning_recommendations,
    )
    from chungoid.intelligence.advanced_replanning_intelligence import (
        create_intelligent_recovery_plan,
        predict_potential_failures,
        analyze_historical_patterns,
    )
    from chungoid.runtime.performance_optimizer import (
        get_real_time_performance_analysis,
        optimize_agent_resolution_mcp,
        generate_performance_recommendations,
    )
    INTELLIGENCE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Intelligence functions not available: {e}")
    INTELLIGENCE_AVAILABLE = False


class IntelligenceToolsInterface:
    """
    Centralized interface for intelligence system tools.
    
    Provides unified access to adaptive learning, replanning intelligence,
    and performance optimization capabilities for agent refinement.
    """
    
    def __init__(self):
        self.available = INTELLIGENCE_AVAILABLE
        
    async def analyze_patterns_and_learn(
        self,
        project_id: Optional[str] = None,
        time_window_days: int = 30,
        enable_cross_project: bool = True
    ) -> Dict[str, Any]:
        """Analyze execution patterns and generate learning insights."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await adaptive_learning_analyze(
            project_id=project_id,
            time_window_days=time_window_days,
            enable_cross_project=enable_cross_project
        )
    
    async def create_strategy_experiment(
        self,
        baseline_strategy: Dict[str, Any],
        test_strategy: Dict[str, Any],
        success_metric: str,
        duration_days: int = 7
    ) -> Dict[str, Any]:
        """Create A/B testing experiments for strategy optimization."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await create_strategy_experiment(
            baseline_strategy=baseline_strategy,
            test_strategy=test_strategy,
            success_metric=success_metric,
            duration_days=duration_days
        )
    
    async def apply_learning_recommendations(
        self,
        recommendations: List[Dict[str, Any]],
        auto_apply_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Apply learning recommendations from pattern analysis."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await apply_learning_recommendations(
            recommendations=recommendations,
            auto_apply_threshold=auto_apply_threshold
        )
    
    async def create_recovery_plan(
        self,
        failure_context: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create comprehensive recovery plans for failures."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await create_intelligent_recovery_plan(
            failure_context=failure_context,
            project_context=project_context
        )
    
    async def predict_failures(
        self,
        current_context: Dict[str, Any],
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict potential failures based on current context."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await predict_potential_failures(
            current_context=current_context,
            execution_plan=execution_plan
        )
    
    async def analyze_historical_patterns(
        self,
        failure_context: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze historical patterns for planning insights."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await analyze_historical_patterns(
            failure_context=failure_context,
            project_context=project_context
        )
    
    async def get_performance_analysis(self) -> Dict[str, Any]:
        """Get real-time performance metrics and analysis."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await get_real_time_performance_analysis()
    
    async def optimize_agent_resolution(
        self,
        task_type: str,
        required_capabilities: Optional[List[str]] = None,
        prefer_autonomous: bool = True
    ) -> Dict[str, Any]:
        """Optimize agent resolution with performance monitoring."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await optimize_agent_resolution_mcp(
            task_type=task_type,
            required_capabilities=required_capabilities,
            prefer_autonomous=prefer_autonomous
        )
    
    async def get_performance_recommendations(self) -> Dict[str, Any]:
        """Generate performance optimization recommendations."""
        if not self.available:
            return {"success": False, "error": "Intelligence functions not available"}
        
        return await generate_performance_recommendations()
    
    def get_available_tools(self) -> List[str]:
        """Get list of available intelligence tools."""
        if not self.available:
            return []
        
        return [
            "analyze_patterns_and_learn",
            "create_strategy_experiment", 
            "apply_learning_recommendations",
            "create_recovery_plan",
            "predict_failures",
            "analyze_historical_patterns",
            "get_performance_analysis",
            "optimize_agent_resolution",
            "get_performance_recommendations"
        ]
    
    def get_tool_descriptions(self) -> Dict[str, str]:
        """Get descriptions of available intelligence tools."""
        return {
            "analyze_patterns_and_learn": "Analyze execution patterns and generate learning insights",
            "create_strategy_experiment": "Create A/B testing experiments for strategy optimization",
            "apply_learning_recommendations": "Apply learning recommendations from pattern analysis",
            "create_recovery_plan": "Create comprehensive recovery plans for failures",
            "predict_failures": "Predict potential failures based on current context",
            "analyze_historical_patterns": "Analyze historical patterns for planning insights",
            "get_performance_analysis": "Get real-time performance metrics and analysis",
            "optimize_agent_resolution": "Optimize agent resolution with performance monitoring",
            "get_performance_recommendations": "Generate performance optimization recommendations"
        }


# Global intelligence tools interface instance
intelligence_tools = IntelligenceToolsInterface()


# Convenience functions for direct access
async def quick_learning_analysis(project_id: Optional[str] = None) -> Dict[str, Any]:
    """Quick learning analysis for current project."""
    return await intelligence_tools.analyze_patterns_and_learn(project_id=project_id)


async def quick_performance_check() -> Dict[str, Any]:
    """Quick performance analysis check."""
    return await intelligence_tools.get_performance_analysis()


async def quick_recovery_plan(error_type: str, error_message: str) -> Dict[str, Any]:
    """Quick recovery plan generation for common errors."""
    failure_context = {
        "error_type": error_type,
        "message": error_message,
        "timestamp": "current"
    }
    return await intelligence_tools.create_recovery_plan(failure_context)


def get_intelligence_status() -> Dict[str, Any]:
    """Get status of intelligence tools availability."""
    return {
        "available": intelligence_tools.available,
        "tools_count": len(intelligence_tools.get_available_tools()),
        "available_tools": intelligence_tools.get_available_tools(),
        "tool_descriptions": intelligence_tools.get_tool_descriptions()
    } 