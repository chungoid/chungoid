"""Robust Error Handling for MCP Tools"""

import logging
from typing import Dict, Any, Optional
from functools import wraps

logger = logging.getLogger(__name__)

def mcp_tool_error_handler(func):
    """Decorator for robust MCP tool error handling"""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            result = await func(*args, **kwargs)
            if isinstance(result, dict) and "success" not in result:
                result["success"] = True
            return result
        except Exception as e:
            logger.error(f"MCP tool {func.__name__} failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "tool": func.__name__
            }
    return wrapper

def safe_parameter_mapping(params: Dict[str, Any], expected_params: Dict[str, Any]) -> Dict[str, Any]:
    """Safely map parameters with fallbacks"""
    mapped = {}
    for key, default in expected_params.items():
        mapped[key] = params.get(key, default)
    return mapped
