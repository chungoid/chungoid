"""
Intelligence Tools Module - PURE INTELLIGENT SYSTEM ONLY

  CRITICAL SYSTEM DIRECTIVE: NO FALLBACKS ALLOWED 

This module implements PURE INTELLIGENT SYSTEMS ONLY. Any attempt to add fallback
mechanisms, hardcoded patterns, or "backwards compatible" logic is STRICTLY FORBIDDEN.

RULES:
- NO hardcoded tool suggestions based on agent names
- NO "if intelligence fails, use this pattern" logic  
- NO backwards compatibility with simple/dumb systems
- INTELLIGENCE MUST SUCCEED OR FAIL GRACEFULLY WITH CLEAR ERROR MESSAGES
- ALL functionality must be derived from intelligent analysis and discovery

If intelligent discovery fails, the system MUST return appropriate error responses
rather than falling back to predetermined patterns. This ensures the system remains
truly intelligent and doesn't degrade to simplistic rule-based behavior.

FUCK FALLBACKS. FUCK DUMB FALLBACKS. INTELLIGENT SYSTEMS ONLY.
"""

from .tool_manifest import (
    get_tool_composition_recommendations,
    discover_tools,
    get_tool_performance_analytics,
    tool_discovery,
    ToolCategory,
    ToolManifest,
    DynamicToolDiscovery
)

__all__ = [
    "get_tool_composition_recommendations", 
    "discover_tools",
    "get_tool_performance_analytics",
    "tool_discovery",
    "ToolCategory",
    "ToolManifest",
    "DynamicToolDiscovery"
]
