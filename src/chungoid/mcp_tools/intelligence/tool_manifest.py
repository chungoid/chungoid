"""
Dynamic Tool Discovery & Manifest System - PURE INTELLIGENT SYSTEM

  *** CRITICAL SYSTEM DIRECTIVE: NO FALLBACKS ALLOWED ***

This module provides intelligent tool selection, composition, and discovery capabilities
for the Chungoid MCP tool ecosystem. It enables autonomous tool chaining and
performance-based tool recommendations through PURE INTELLIGENT METHODS ONLY.

STRICT INTELLIGENCE-ONLY RULES:
- NO hardcoded tool lists or patterns based on agent types
- NO "if discovery fails, suggest these tools" fallback logic
- NO backwards compatibility with simple/rule-based systems
- ALL tool recommendations MUST come from intelligent analysis
- FAILURE modes return clear errors rather than degrading to fallbacks

Features:
- Rich tool metadata with capability descriptions
- Usage patterns and best practices derived from intelligent analysis
- Historical performance tracking for continuous learning
- Dynamic capability matching using intelligent algorithms
- Intelligent tool composition recommendations (NO HARDCODED PATTERNS)

If intelligent discovery/composition fails, the system MUST fail gracefully with
clear error messages. This preserves the integrity of the intelligent system and
prevents degradation to simplistic rule-based behavior.
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
            # Reconstruct metrics
            metrics_data = data.get('metrics', {})
            metrics = ToolMetrics(**metrics_data)
            
            # Reconstruct capabilities properly
            capabilities = []
            capabilities_data = data.get('capabilities', [])
            for cap_data in capabilities_data:
                capability = ToolCapability(
                    name=cap_data.get('name', ''),
                    description=cap_data.get('description', ''),
                    input_types=cap_data.get('input_types', []),
                    output_types=cap_data.get('output_types', []),
                    examples=cap_data.get('examples', []),
                    prerequisites=cap_data.get('prerequisites', [])
                )
                capabilities.append(capability)
            
            # Reconstruct usage patterns properly
            usage_patterns = []
            patterns_data = data.get('usage_patterns', [])
            for pattern_data in patterns_data:
                pattern = UsagePattern(
                    pattern_name=pattern_data.get('pattern_name', ''),
                    description=pattern_data.get('description', ''),
                    tool_sequence=pattern_data.get('tool_sequence', []),
                    use_cases=pattern_data.get('use_cases', []),
                    success_rate=pattern_data.get('success_rate', 0.0),
                    avg_execution_time=pattern_data.get('avg_execution_time', 0.0),
                    complexity=UsageComplexity(pattern_data.get('complexity', 'moderate'))
                )
                usage_patterns.append(pattern)
            
            # Create properly reconstructed manifest
            manifest = ToolManifest(
                tool_name=name,
                display_name=data.get('display_name', name),
                description=data.get('description', ''),
                category=ToolCategory(data.get('category', 'development')),
                capabilities=capabilities,  # Now properly reconstructed!
                usage_patterns=usage_patterns,  # Now properly reconstructed!
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
        logger.info("🚀🚀🚀 INITIALIZING TOOL MANIFESTS - CRITICAL FOR INTELLIGENT AGENT BEHAVIOR 🚀🚀🚀")
        
        try:
            # Import inside function to avoid circular import
            logger.info("📋 CALLING COMPREHENSIVE MANIFEST INITIALIZATION...")
            
            # Call the comprehensive initialization that includes auto-manifests
            result = self._call_manifest_initialization()
            
            if result and result.get("success"):
                total_tools = result.get("total_tools", 0)
                total_available = result.get("total_available", 0)
                coverage_percentage = result.get("coverage_percentage", 0)
                rich_manifests = result.get("rich_manifests", 0)
                auto_manifests = result.get("auto_manifests", 0)
                operational_status = result.get("operational_status", "UNKNOWN")
                
                logger.info("🎉🎉🎉 COMPREHENSIVE TOOL MANIFEST INITIALIZATION COMPLETE! 🎉🎉🎉")
                logger.info(f"📊 FINAL COVERAGE: {total_tools}/{total_available} tools ({coverage_percentage:.1f}%)")
                logger.info(f"🎯 BREAKDOWN:")
                logger.info(f"   ✅ RICH MANIFESTS: {rich_manifests} tools with detailed intelligence")
                logger.info(f"   🔧 AUTO-MANIFESTS: {auto_manifests} tools with basic intelligence")
                logger.info(f"🚀 OPERATIONAL STATUS: {operational_status}")
                
                if operational_status == "FULLY_OPERATIONAL":
                    logger.info("🧠 INTELLIGENT TOOL DISCOVERY IS FULLY OPERATIONAL!")
                    logger.info("🔥 AGENTS HAVE COMPLETE SOPHISTICATED TOOL SELECTION!")
                elif operational_status == "MOSTLY_OPERATIONAL":
                    logger.info("🧠 INTELLIGENT TOOL DISCOVERY IS MOSTLY OPERATIONAL!")
                    logger.info("🔥 AGENTS HAVE EXTENSIVE INTELLIGENT TOOL SELECTION!")
                elif operational_status == "PARTIALLY_OPERATIONAL":
                    logger.warning("⚠️  INTELLIGENT TOOL DISCOVERY IS PARTIALLY OPERATIONAL!")
                    logger.warning("🚨 AGENTS HAVE LIMITED INTELLIGENT TOOL SELECTION!")
                else:
                    logger.error("❌ INTELLIGENT TOOL DISCOVERY IS BARELY OPERATIONAL!")
                    logger.error("🚨 AGENTS WILL MOSTLY USE BASIC TOOL SELECTION!")
            else:
                logger.error("❌ COMPREHENSIVE INITIALIZATION FAILED!")
                error_msg = result.get("error", "Unknown error") if result else "No result returned"
                logger.error(f"💥 ERROR: {error_msg}")
                
        except ImportError as e:
            logger.error("❌❌❌ CRITICAL: MANIFEST INITIALIZATION MODULE NOT FOUND! ❌❌❌")
            logger.error(f"💥 IMPORT ERROR: {e}")
            logger.error("⚠️  TOOL DISCOVERY WILL BE COMPLETELY BROKEN!")
            logger.error("🚨 AGENTS WILL OPERATE IN DUMB MODE!")
            
        except Exception as e:
            logger.error("❌❌❌ UNEXPECTED ERROR DURING MANIFEST INITIALIZATION! ❌❌❌")
            logger.error(f"💥 ERROR: {e}")
            logger.error("⚠️  TOOL DISCOVERY SYSTEM COMPROMISED!")
            
        # Always log final status
        final_count = len(self.manifests)
        if final_count > 0:
            logger.info(f"✅ FINAL STATUS: {final_count} TOOLS AVAILABLE FOR INTELLIGENT DISCOVERY")
        else:
            logger.error("❌ FINAL STATUS: ZERO TOOLS AVAILABLE - SYSTEM IS BROKEN!")
            logger.error("🚨 ALL AGENTS WILL USE GENERIC FALLBACK TOOLS!")

    def _call_manifest_initialization(self):
        """Call manifest initialization avoiding circular imports."""
        try:
            from .manifest_initialization import initialize_all_tool_manifests
            
            logger.info("📋 CALLING COMPREHENSIVE MANIFEST INITIALIZATION...")
            
            # Call the comprehensive initialization that includes auto-manifests
            return initialize_all_tool_manifests(self)
            
        except Exception as e:
            logger.error(f"💥 MANIFEST INITIALIZATION FAILED: {e}")
            return {"success": False, "error": str(e)}


    def get_discovery_health_status(self) -> Dict[str, Any]:
        """Get blatant health status of tool discovery system."""
        total_tools = len(self.manifests)
        
        # Use known tool count to avoid circular import
        total_available = 67  # Known tool count from __all__ list in __init__.py
        coverage_percentage = (total_tools / total_available * 100) if total_available > 0 else 0
        
        if total_tools == 0:
            logger.error("💀💀💀 TOOL DISCOVERY SYSTEM IS DEAD! 💀💀💀")
            logger.error("🚨 ZERO TOOLS REGISTERED - AGENTS OPERATING IN DUMB MODE!")
            return {
                "healthy": False,
                "status": "CRITICAL_FAILURE",
                "total_tools": 0,
                "total_available": total_available,
                "coverage_percentage": 0.0,
                "message": "NO TOOLS REGISTERED - SYSTEM BROKEN",
                "impact": "AGENTS FALL BACK TO GENERIC TOOLS"
            }
        elif coverage_percentage >= 95:
            logger.info(f"✅ TOOL DISCOVERY FULLY OPERATIONAL: {total_tools}/{total_available} tools ({coverage_percentage:.1f}%)")
            return {
                "healthy": True,
                "status": "FULLY_OPERATIONAL",
                "total_tools": total_tools,
                "total_available": total_available,
                "coverage_percentage": coverage_percentage,
                "message": "COMPLETE INTELLIGENT TOOL DISCOVERY",
                "impact": "AGENTS HAVE FULL CAPABILITIES"
            }
        elif coverage_percentage >= 80:
            logger.info(f"✅ TOOL DISCOVERY MOSTLY OPERATIONAL: {total_tools}/{total_available} tools ({coverage_percentage:.1f}%)")
            return {
                "healthy": True,
                "status": "MOSTLY_OPERATIONAL",
                "total_tools": total_tools,
                "total_available": total_available,
                "coverage_percentage": coverage_percentage,
                "message": "EXTENSIVE INTELLIGENT TOOL DISCOVERY",
                "impact": "AGENTS HAVE EXCELLENT CAPABILITIES"
            }
        elif coverage_percentage >= 50:
            logger.warning(f"⚠️  TOOL DISCOVERY PARTIALLY OPERATIONAL: {total_tools}/{total_available} tools ({coverage_percentage:.1f}%)")
            return {
                "healthy": False,
                "status": "PARTIALLY_OPERATIONAL",
                "total_tools": total_tools,
                "total_available": total_available,
                "coverage_percentage": coverage_percentage,
                "message": "LIMITED INTELLIGENT TOOL DISCOVERY",
                "impact": "AGENTS HAVE REDUCED CAPABILITIES"
            }
        else:
            logger.warning(f"⚠️  TOOL DISCOVERY BARELY OPERATIONAL: {total_tools}/{total_available} tools ({coverage_percentage:.1f}%)")
            return {
                "healthy": False,
                "status": "BARELY_OPERATIONAL",
                "total_tools": total_tools,
                "total_available": total_available,
                "coverage_percentage": coverage_percentage,
                "message": "MINIMAL INTELLIGENT TOOL DISCOVERY",
                "impact": "AGENTS MOSTLY USE BASIC TOOLS"
            }


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
        # Check if tool discovery is healthy first
        total_registered = len(tool_discovery.manifests)
        if total_registered == 0:
            logger.error(f"💀 TOOL DISCOVERY FAILURE: Query '{query}' - NO TOOLS REGISTERED!")
            logger.error("🚨 AGENT WILL FALL BACK TO GENERIC TOOLS!")
            return {
                "success": False,
                "error": "NO_TOOLS_REGISTERED",
                "query": query,
                "context": context or {},
                "discovered_tools": [],
                "recommendations": [],
                "composition_suggestions": [],
                "health_status": "CRITICAL_FAILURE"
            }
            
        logger.debug(f"🔍 TOOL DISCOVERY: Searching for '{query}' in {total_registered} registered tools")
        
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
        
        # Log discovery results
        discovered_count = len(results["discovered_tools"])
        recommendation_count = len(results["recommendations"])
        
        if discovered_count > 0 or recommendation_count > 0:
            logger.info(f"✅ TOOL DISCOVERY SUCCESS: '{query}' → {discovered_count} tools, {recommendation_count} recommendations")
            results["health_status"] = "OPERATIONAL"
        else:
            logger.warning(f"⚠️  TOOL DISCOVERY EMPTY: '{query}' → No tools found (from {total_registered} registered)")
            results["health_status"] = "EMPTY_RESULTS"
        
        return {
            "success": True,
            **results,
        }
        
    except Exception as e:
        logger.error(f"💥 TOOL DISCOVERY ERROR: Query '{query}' failed - {e}")
        return {
            "success": False,
            "error": str(e),
            "query": query,
            "health_status": "ERROR"
        }


async def get_tool_composition_recommendations(
    target_tools: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Get recommendations for tool composition and chaining.
    
    PURE INTELLIGENT SYSTEM: Uses ONLY intelligent discovery mechanisms.
    NO fallbacks, NO hardcoded patterns, PURE intelligence only.
    
    Args:
        target_tools: List of tools to find composition patterns for (optional)
        context: Optional execution context (can include agent_id, task_type, etc.)
        **kwargs: Additional parameters from agent calls
        
    Returns:
        Dict containing composition recommendations
    """
    try:
        # Handle None parameters gracefully
        if target_tools is None:
            target_tools = []
        if context is None:
            context = {}
        
        # PURE INTELLIGENT ENHANCEMENT: If no target_tools provided, discover them intelligently
        if not target_tools and context:
            logger.info(f"[PURE INTELLIGENT] Auto-discovering tools from context: {context}")
            
            # Extract task information from context
            task_type = context.get("task_type", "")
            agent_id = context.get("agent_id", "")
            
            # PURE INTELLIGENT discovery - no fallbacks
            if task_type or agent_id:
                discovery_query = task_type if task_type else f"tools for {agent_id}"
                logger.info(f"🔍 AUTO-DISCOVERING TOOLS: '{discovery_query}'")
                
                discovery_result = await discover_tools(
                    query=discovery_query,
                    context=context,
                    max_results=5
                )
                
                if discovery_result.get("success") and discovery_result.get("discovered_tools"):
                    target_tools = [tool["tool_name"] for tool in discovery_result["discovered_tools"]]
                    logger.info(f"✅ INTELLIGENT DISCOVERY SUCCESS: Found {len(target_tools)} tools: {target_tools}")
                else:
                    # NO FALLBACKS - pure intelligence only
                    health_status = discovery_result.get("health_status", "UNKNOWN")
                    logger.warning(f"❌ PURE INTELLIGENT DISCOVERY FAILED: Query '{discovery_query}' - Status: {health_status}")
                    
                    if health_status == "CRITICAL_FAILURE":
                        logger.error("💀 TOOL REGISTRY IS EMPTY - SYSTEM IS BROKEN!")
                    elif health_status == "EMPTY_RESULTS":
                        logger.warning("🔍 NO TOOLS MATCH QUERY - SEARCH CRITERIA TOO SPECIFIC?")
                    
                    return {
                        "success": False,
                        "error": "Pure intelligent discovery failed - no tools could be discovered intelligently",
                        "target_tools": [],
                        "context": context,
                        "intelligence_level": "discovery_failed",
                        "health_status": health_status,
                        "message": "PURE INTELLIGENT SYSTEM: No hardcoded fallbacks available. Intelligent discovery required."
                    }

        # Continue with intelligent composition only if we have tools
        if not target_tools:
            return {
                "success": False,
                "error": "No target tools provided and intelligent discovery unsuccessful",
                "target_tools": [],
                "context": context,
                "intelligence_level": "no_tools",
                "message": "PURE INTELLIGENT SYSTEM: Requires either explicit tools or successful intelligent discovery."
            }

        suggestions = tool_discovery.get_tool_composition_suggestions(
            target_tools, context or {}
        )
        
        # Enhanced intelligent response
        result = {
            "success": True,
            "target_tools": target_tools,
            "context": context,
            "intelligently_discovered": len(target_tools) > 0 and not kwargs.get("explicit_tools", False),
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
        
        # Add intelligent recommendations if we have context
        if context and target_tools:
            try:
                # Get performance insights for the recommended tools
                performance_report = tool_discovery.get_performance_report()
                
                # Add tool-specific recommendations
                tool_recommendations = []
                for tool in target_tools:
                    if tool in tool_discovery.manifests:
                        manifest = tool_discovery.manifests[tool]
                        tool_recommendations.append({
                            "tool_name": tool,
                            "success_rate": manifest.metrics.success_rate,
                            "complexity": manifest.complexity.value,
                            "recommended_use_cases": [cap.name for cap in manifest.capabilities[:3]]
                        })
                
                result["tool_recommendations"] = tool_recommendations
                result["intelligence_level"] = "advanced"
                
            except Exception as e:
                logger.warning(f"Could not generate advanced recommendations: {e}")
                result["intelligence_level"] = "basic"
        else:
            result["intelligence_level"] = "basic"
        
        logger.info(f"[PURE INTELLIGENT] Generated {len(result['composition_patterns'])} composition patterns for {len(target_tools)} tools")
        return result

    except Exception as e:
        logger.error(f"Pure intelligent tool composition failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "target_tools": target_tools or [],
            "context": context or {},
            "intelligence_level": "error",
            "message": "PURE INTELLIGENT SYSTEM: An error occurred during intelligent processing."
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


# Health check function for agents
async def get_tool_discovery_health() -> Dict[str, Any]:
    """
    Get blatant health status of the tool discovery system.
    Agents can call this to verify intelligent tool discovery is working.
    """
    health_status = tool_discovery.get_discovery_health_status()
    
    # Add additional runtime checks
    try:
        # Test capability search
        test_result = tool_discovery.find_tools_by_capability("test")
        health_status["capability_search_working"] = True
        
        # Test category search
        category_result = tool_discovery.find_tools_by_category(ToolCategory.DATABASE)
        health_status["category_search_working"] = True
        
        # Test manifest loading
        health_status["manifests_loaded"] = len(tool_discovery.manifests) > 0
        
        if health_status["healthy"]:
            logger.info("✅ TOOL DISCOVERY HEALTH CHECK: SYSTEM OPERATIONAL")
        else:
            logger.error(f"❌ TOOL DISCOVERY HEALTH CHECK: {health_status['status']}")
            
    except Exception as e:
        logger.error(f"💥 TOOL DISCOVERY HEALTH CHECK FAILED: {e}")
        health_status.update({
            "healthy": False,
            "status": "HEALTH_CHECK_FAILED",
            "error": str(e)
        })
    
    return health_status 