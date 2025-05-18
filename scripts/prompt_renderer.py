import re
from pathlib import Path
from typing import Any, Dict, Union
import yaml

_PLACEHOLDER_RE = re.compile(r"{{\s*([^{}]+?)\s*}}")  # capture everything between braces minus braces themselves


def _render_text(raw: str, context: Dict[str, Any]) -> str:  # noqa: D401
    """Replace placeholders like ``{{ var }}`` in *raw* using *context* dict.

    The renderer is intentionally minimal – it does **not** evaluate arbitrary
    expressions.  It supports dotted look-ups (e.g. ``last_status.status``)
    to one nesting level by walking the context mapping.
    """

    def _replace(match: re.Match[str]) -> str:  # type: ignore[type-var]
        expr = match.group(1).strip()
        parts = expr.split(".")
        value: Any = context
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                # Unresolved placeholder – keep as-is so linter can catch.
                return match.group(0)
        return str(value)

    return _PLACEHOLDER_RE.sub(_replace, raw)


def _render_obj(obj: Any, context: Dict[str, Any]) -> Any:  # noqa: D401
    if isinstance(obj, str):
        return _render_text(obj, context)
    if isinstance(obj, list):
        return [_render_obj(i, context) for i in obj]
    if isinstance(obj, dict):
        return {k: _render_obj(v, context) for k, v in obj.items()}
    return obj


def render_yaml_prompt(path: Union[str, Path], context: Dict[str, Any]) -> Dict[str, Any]:  # noqa: D401
    """Load *path* (YAML) and render all string fields via placeholder substitution."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))  # type: ignore[assignment]
    return _render_obj(data, context) 