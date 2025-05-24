"""
Dynamic Tool Discovery & Manifest System

Provides intelligent tool selection, composition, and discovery capabilities
for the Chungoid MCP tool ecosystem. Enables autonomous tool chaining
and performance-based tool recommendations.

Features:
- Rich tool metadata with capability descriptions
- Usage patterns and best practices
- Historical performance tracking  
- Dynamic capability matching
- Intelligent tool composition recommendations
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ToolCategory(Enum):
    """Categories for organizing MCP tools."""
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    TERMINAL = "terminal"
    CONTENT = "content"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"


class UsageComplexity(Enum):
    """Complexity levels for tool usage."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    EXPERT = "expert"


@dataclass
class ToolCapability:
    """Describes a specific capability of a tool."""
    name: str
    description: str
    input_types: List[str]
    output_types: List[str]
    examples: List[str]
    prerequisites: List[str] = None
    
    def __post_init__(self):
        if self.prerequisites is None:
            self.prerequisites = []


@dataclass
class UsagePattern:
    """Describes common usage patterns for tools."""
    pattern_name: str
    description: str
    tool_sequence: List[str]
    use_cases: List[str]
    success_rate: float = 0.0
    avg_execution_time: float = 0.0
    complexity: UsageComplexity = UsageComplexity.MODERATE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['complexity'] = self.complexity.value
        return data


@dataclass
class ToolMetrics:
    """Performance and usage metrics for a tool."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_execution_time: float = 0.0
    last_used: Optional[str] = None
    error_patterns: List[str] = None
    
    def __post_init__(self):
        if self.error_patterns is None:
            self.error_patterns = []
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_executions == 0:
            return 0.0
        return (self.successful_executions / self.total_executions) * 100
    
    def update_execution(self, success: bool, execution_time: float, error: Optional[str] = None):
        """Update metrics with new execution data."""
        self.total_executions += 1
        if success:
            self.successful_executions += 1
        else:
            self.failed_executions += 1
            if error and error not in self.error_patterns:
                self.error_patterns.append(error)
        
        # Update average execution time
        if self.avg_execution_time == 0:
            self.avg_execution_time = execution_time
        else:
            self.avg_execution_time = (self.avg_execution_time + execution_time) / 2
        
        self.last_used = datetime.now().isoformat()


@dataclass
class ToolManifest:
    """Complete manifest entry for an MCP tool."""
    tool_name: str
    display_name: str
    description: str
    category: ToolCategory
    capabilities: List[ToolCapability]
    usage_patterns: List[UsagePattern]
    metrics: ToolMetrics
    dependencies: List[str] = None
    related_tools: List[str] = None
    tags: List[str] = None
    complexity: UsageComplexity = UsageComplexity.MODERATE
    project_aware: bool = True
    security_level: str = "standard"
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.related_tools is None:
            self.related_tools = []
        if self.tags is None:
            self.tags = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['category'] = self.category.value
        data['complexity'] = self.complexity.value
        data['capabilities'] = [asdict(cap) for cap in self.capabilities]
        data['usage_patterns'] = [pattern.to_dict() for pattern in self.usage_patterns]
        data['metrics'] = asdict(self.metrics)
        return data


class DynamicToolDiscovery:
    """Dynamic tool discovery and recommendation system."""
    
    def __init__(self, manifest_file: Optional[str] = None):
        self.manifests: Dict[str, ToolManifest] = {}
        self.manifest_file = manifest_file or "tool_manifests.json"
        self.load_manifests()
    
    def register_tool(self, manifest: ToolManifest) -> bool:
        """Register a new tool in the discovery system."""
        try:
            self.manifests[manifest.tool_name] = manifest
            logger.info(f"Registered tool: {manifest.tool_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register tool {manifest.tool_name}: {e}")
            return False
    
    def find_tools_by_capability(self, capability_name: str) -> List[ToolManifest]:
        """Find tools that provide a specific capability."""
        matching_tools = []
        for manifest in self.manifests.values():
            for cap in manifest.capabilities:
                if capability_name.lower() in cap.name.lower() or capability_name.lower() in cap.description.lower():
                    matching_tools.append(manifest)
                    break
        return matching_tools
    
    def find_tools_by_category(self, category: ToolCategory) -> List[ToolManifest]:
        """Find tools in a specific category."""
        return [manifest for manifest in self.manifests.values() if manifest.category == category]
    
    def find_tools_by_tags(self, tags: List[str]) -> List[ToolManifest]:
        """Find tools that match any of the given tags."""
        matching_tools = []
        for manifest in self.manifests.values():
            if any(tag in manifest.tags for tag in tags):
                matching_tools.append(manifest)
        return matching_tools
    
    def get_tool_recommendations(
        self, 
        context: Dict[str, Any],
        task_description: str,
        max_recommendations: int = 5
    ) -> List[Tuple[ToolManifest, float]]:
        """Get intelligent tool recommendations based on context and task."""
        recommendations = []
        
        for manifest in self.manifests.values():
            score = self._calculate_recommendation_score(manifest, context, task_description)
            if score > 0:
                recommendations.append((manifest, score))
        
        # Sort by score (descending) and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]
    
    def get_tool_composition_suggestions(
        self, 
        target_tools: List[str],
        context: Dict[str, Any]
    ) -> List[UsagePattern]:
        """Suggest tool composition patterns for achieving complex tasks."""
        suggestions = []
        
        # Find usage patterns that involve the target tools
        for manifest in self.manifests.values():
            for pattern in manifest.usage_patterns:
                if any(tool in pattern.tool_sequence for tool in target_tools):
                    suggestions.append(pattern)
        
        # Sort by success rate and complexity
        suggestions.sort(key=lambda x: (x.success_rate, -x.complexity.value), reverse=True)
        return suggestions
    
    def update_tool_metrics(
        self, 
        tool_name: str, 
        success: bool, 
        execution_time: float, 
        error: Optional[str] = None
    ) -> bool:
        """Update performance metrics for a tool."""
        try:
            if tool_name in self.manifests:
                self.manifests[tool_name].metrics.update_execution(success, execution_time, error)
                logger.debug(f"Updated metrics for {tool_name}: success={success}, time={execution_time}")
                return True
            else:
                logger.warning(f"Tool not found for metrics update: {tool_name}")
                return False
        except Exception as e:
            logger.error(f"Failed to update metrics for {tool_name}: {e}")
            return False
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report for all tools."""
        report = {
            "total_tools": len(self.manifests),
            "categories": {},
            "top_performers": [],
            "tools_needing_attention": [],
            "timestamp": datetime.now().isoformat(),
        }
        
        # Category breakdown
        for category in ToolCategory:
            tools_in_category = self.find_tools_by_category(category)
            if tools_in_category:
                report["categories"][category.value] = {
                    "count": len(tools_in_category),
                    "avg_success_rate": sum(t.metrics.success_rate for t in tools_in_category) / len(tools_in_category),
                    "total_executions": sum(t.metrics.total_executions for t in tools_in_category),
                }
        
        # Top performers (high success rate and usage)
        performers = [(name, manifest) for name, manifest in self.manifests.items() 
                     if manifest.metrics.total_executions > 0]
        performers.sort(key=lambda x: (x[1].metrics.success_rate, x[1].metrics.total_executions), reverse=True)
        
        report["top_performers"] = [
            {
                "tool_name": name,
                "success_rate": manifest.metrics.success_rate,
                "total_executions": manifest.metrics.total_executions,
                "avg_execution_time": manifest.metrics.avg_execution_time,
            }
            for name, manifest in performers[:10]
        ]
        
        # Tools needing attention (low success rate or high error rate)
        attention_needed = [
            {
                "tool_name": name,
                "success_rate": manifest.metrics.success_rate,
                "error_count": len(manifest.metrics.error_patterns),
                "total_executions": manifest.metrics.total_executions,
            }
            for name, manifest in self.manifests.items()
            if manifest.metrics.success_rate < 80 and manifest.metrics.total_executions > 5
        ]
        
        report["tools_needing_attention"] = sorted(
            attention_needed, 
            key=lambda x: x["success_rate"]
        )
        
        return report
    
    def save_manifests(self) -> bool:
        """Save tool manifests to file."""
        try:
            manifest_data = {
                name: manifest.to_dict() 
                for name, manifest in self.manifests.items()
            }
            
            with open(self.manifest_file, 'w') as f:
                json.dump(manifest_data, f, indent=2)
            
            logger.info(f"Saved {len(self.manifests)} tool manifests to {self.manifest_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save manifests: {e}")
            return False
    
    def load_manifests(self) -> bool:
        """Load tool manifests from file."""
        try:
            manifest_path = Path(self.manifest_file)
            if not manifest_path.exists():
                logger.info("No existing manifest file found, starting with empty registry")
                self._initialize_default_manifests()
                return True
            
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            for name, data in manifest_data.items():
                # Reconstruct manifest from dict data
                # This is a simplified version - full implementation would properly deserialize
                manifest = self._reconstruct_manifest_from_dict(name, data)
                if manifest:
                    self.manifests[name] = manifest
            
            logger.info(f"Loaded {len(self.manifests)} tool manifests")
            return True
        except Exception as e:
            logger.error(f"Failed to load manifests: {e}")
            self._initialize_default_manifests()
            return False
    
    def _calculate_recommendation_score(
        self, 
        manifest: ToolManifest, 
        context: Dict[str, Any], 
        task_description: str
    ) -> float:
        """Calculate recommendation score for a tool based on context and task."""
        score = 0.0
        
        # Base score from success rate
        score += manifest.metrics.success_rate * 0.3
        
        # Boost score for recently used tools
        if manifest.metrics.last_used:
            # Simple recency boost - could be more sophisticated
            score += 10.0
        
        # Boost score for tools with matching capabilities
        task_lower = task_description.lower()
        for capability in manifest.capabilities:
            if any(keyword in task_lower for keyword in capability.name.lower().split()):
                score += 25.0
            if any(keyword in task_lower for keyword in capability.description.lower().split()):
                score += 15.0
        
        # Context matching
        project_type = context.get('project_type', '')
        if project_type and project_type in manifest.tags:
            score += 20.0
        
        # Penalize for complexity if not needed
        if manifest.complexity == UsageComplexity.EXPERT and 'simple' in task_lower:
            score -= 10.0
        
        return max(0.0, score)
    
    def _reconstruct_manifest_from_dict(self, name: str, data: Dict[str, Any]) -> Optional[ToolManifest]:
        """Reconstruct ToolManifest from dictionary data."""
        try:
            # This is a simplified reconstruction - full implementation would be more robust
            metrics = ToolMetrics(**data.get('metrics', {}))
            
            # For now, create a basic manifest
            manifest = ToolManifest(
                tool_name=name,
                display_name=data.get('display_name', name),
                description=data.get('description', ''),
                category=ToolCategory(data.get('category', 'development')),
                capabilities=[],  # Would reconstruct properly in full implementation
                usage_patterns=[],  # Would reconstruct properly in full implementation
                metrics=metrics,
                dependencies=data.get('dependencies', []),
                related_tools=data.get('related_tools', []),
                tags=data.get('tags', []),
                complexity=UsageComplexity(data.get('complexity', 'moderate')),
                project_aware=data.get('project_aware', True),
                security_level=data.get('security_level', 'standard'),
            )
            
            return manifest
        except Exception as e:
            logger.error(f"Failed to reconstruct manifest for {name}: {e}")
            return None
    
    def _initialize_default_manifests(self):
        """Initialize with default tool manifests for known tools."""
        # This would be populated with actual tool manifests
        # For now, just log that we're starting fresh
        logger.info("Initializing with default tool manifests")


# Global tool discovery instance
tool_discovery = DynamicToolDiscovery()


# Tool manifest generation and registration functions

async def generate_tool_manifest(
    tool_name: str,
    description: str,
    category: ToolCategory,
    capabilities: List[ToolCapability],
    project_aware: bool = True,
    security_level: str = "standard",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Generate a tool manifest for dynamic discovery and recommendation.
    
    Args:
        tool_name: Name of the tool
        description: Tool description
        category: Tool category
        capabilities: List of tool capabilities
        project_aware: Whether tool supports project context
        security_level: Security level (safe, standard, high, critical)
        tags: Additional tags for categorization
        
    Returns:
        Dict containing tool manifest information
    """
    try:
        manifest = ToolManifest(
            tool_name=tool_name,
            display_name=tool_name.replace('_', ' ').title(),
            description=description,
            category=category,
            capabilities=capabilities,
            usage_patterns=[],
            metrics=ToolMetrics(),
            tags=tags or [],
            project_aware=project_aware,
            security_level=security_level,
        )
        
        # Register with discovery system
        success = tool_discovery.register_tool(manifest)
        
        return {
            "success": success,
            "tool_name": tool_name,
            "manifest": manifest.to_dict(),
            "registered": success,
        }
        
    except Exception as e:
        logger.error(f"Failed to generate manifest for {tool_name}: {e}")
        return {
            "success": False,
            "error": str(e),
            "tool_name": tool_name,
        }


async def discover_tools(
    query: str,
    context: Optional[Dict[str, Any]] = None,
    category: Optional[str] = None,
    max_results: int = 10,
) -> Dict[str, Any]:
    """
    Discover tools based on query and context.
    
    Args:
        query: Search query or task description
        context: Execution context for recommendations
        category: Optional category filter
        max_results: Maximum number of results to return
        
    Returns:
        Dict containing discovered tools and recommendations
    """
    try:
        results = {
            "query": query,
            "context": context or {},
            "discovered_tools": [],
            "recommendations": [],
            "composition_suggestions": [],
        }
        
        # Find tools by capability
        capability_matches = tool_discovery.find_tools_by_capability(query)
        
        # Find tools by category if specified
        if category:
            try:
                cat_enum = ToolCategory(category.lower())
                category_matches = tool_discovery.find_tools_by_category(cat_enum)
            except ValueError:
                category_matches = []
        else:
            category_matches = []
        
        # Combine and deduplicate results
        all_matches = list({tool.tool_name: tool for tool in capability_matches + category_matches}.values())
        
        results["discovered_tools"] = [
            {
                "tool_name": tool.tool_name,
                "display_name": tool.display_name,
                "description": tool.description,
                "category": tool.category.value,
                "success_rate": tool.metrics.success_rate,
                "complexity": tool.complexity.value,
                "tags": tool.tags,
            }
            for tool in all_matches[:max_results]
        ]
        
        # Get intelligent recommendations if context provided
        if context:
            recommendations = tool_discovery.get_tool_recommendations(
                context, query, max_results
            )
            
            results["recommendations"] = [
                {
                    "tool_name": tool.tool_name,
                    "display_name": tool.display_name,
                    "recommendation_score": score,
                    "success_rate": tool.metrics.success_rate,
                    "description": tool.description,
                }
                for tool, score in recommendations
            ]
        
        return {
            "success": True,
            **results,
        }
        
    except Exception as e:
        logger.error(f"Tool discovery failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
        }


async def get_tool_composition_recommendations(
    target_tools: List[str],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Get recommendations for tool composition and chaining.
    
    Args:
        target_tools: List of tools to find composition patterns for
        context: Optional execution context
        
    Returns:
        Dict containing composition recommendations
    """
    try:
        suggestions = tool_discovery.get_tool_composition_suggestions(
            target_tools, context or {}
        )
        
        return {
            "success": True,
            "target_tools": target_tools,
            "context": context or {},
            "composition_patterns": [
                {
                    "pattern_name": pattern.pattern_name,
                    "description": pattern.description,
                    "tool_sequence": pattern.tool_sequence,
                    "use_cases": pattern.use_cases,
                    "success_rate": pattern.success_rate,
                    "complexity": pattern.complexity.value,
                    "avg_execution_time": pattern.avg_execution_time,
                }
                for pattern in suggestions
            ],
        }
        
    except Exception as e:
        logger.error(f"Tool composition recommendation failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_tools": target_tools,
        }


async def get_tool_performance_analytics() -> Dict[str, Any]:
    """
    Get comprehensive analytics on tool performance and usage.
    
    Returns:
        Dict containing performance analytics
    """
    try:
        report = tool_discovery.get_performance_report()
        
        return {
            "success": True,
            "analytics": report,
            "insights": _generate_performance_insights(report),
        }
        
    except Exception as e:
        logger.error(f"Performance analytics failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


def _generate_performance_insights(report: Dict[str, Any]) -> List[str]:
    """Generate actionable insights from performance data."""
    insights = []
    
    if report.get("tools_needing_attention"):
        insights.append(f"{len(report['tools_needing_attention'])} tools have success rates below 80% and may need optimization")
    
    if report.get("top_performers"):
        best_category = max(report.get("categories", {}).items(), 
                          key=lambda x: x[1].get("avg_success_rate", 0), default=("", {}))
        if best_category[0]:
            insights.append(f"'{best_category[0]}' category shows highest average success rate")
    
    total_executions = sum(cat.get("total_executions", 0) for cat in report.get("categories", {}).values())
    if total_executions > 100:
        insights.append(f"System has processed {total_executions} total tool executions across all categories")
    
    return insights 