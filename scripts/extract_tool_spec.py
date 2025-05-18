#!/usr/bin/env python3
"""Extract MCP tool specifications and export them as JSON files.

This utility introspects ``chungoid-core`` to discover tools returned by
``ChungoidEngine.get_mcp_tools`` (control layer) and validates each
tool definition against the canonical JSON-Schema defined at
``schemas/tool_spec_schema.json``.

Outputs are written under ``dev/docs/reference/tool_specs/<tool>.json`` so they
can be published with the rendered documentation.  A Markdown index file
``dev/docs/reference/tool_specs_index.md`` is also (re)generated with :download: links
for easy browsing.

Usage (local):
    python dev/scripts/extract_tool_spec.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

import jsonschema  # type: ignore

# ---------------------------------------------------------------------------
# Compatibility shim (must come BEFORE control-layer imports)
# ---------------------------------------------------------------------------


class _StubModule:
    """Minimal placeholder to satisfy `import`."""

    def __init__(self, name: str):
        self.__name__ = name

    def __getattr__(self, item):  # noqa: D401, B007
        # Return no-op lambda for any attribute access.
        return lambda *args, **kwargs: None


# Pre-register stub for chungoid.utils.analysis_utils if missing so that
# 'import summarise_code' in utils/__init__.py doesn't explode.
if "chungoid.utils.analysis_utils" not in sys.modules:
    sys.modules["chungoid.utils.analysis_utils"] = _StubModule(
        "chungoid.utils.analysis_utils"
    )


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "schemas" / "tool_spec_schema.json"
TOOL_SPECS_JSON_DIR = ROOT / "dev" / "docs" / "reference" / "tool_specs"
INDEX_MD = ROOT / "dev" / "docs" / "reference" / "tool_specs_index.md"

HEADER_MD = (
    "# Tool Specifications\n\n"
    "Below is the list of MCP tools currently exposed by the Chungoid core. "
    "Each link downloads a machine-readable JSON file that conforms to "
    "`tool_spec_schema.json`.\n"
)


def _get_engine_class():
    """Dynamically import ChungoidEngine without hard-coding sys.path."""
    engine_path = ROOT / "chungoid-core" / "src" / "chungoid" / "engine.py"
    if not engine_path.exists():
        raise FileNotFoundError("Could not locate chungoid-core/src/chungoid/engine.py")
    spec = importlib.util.spec_from_file_location("chungoid_core_engine", engine_path)
    if spec is None or spec.loader is None:
        raise ImportError("Failed to build import spec for ChungoidEngine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # type: ignore[arg-type]
    try:
        spec.loader.exec_module(module)  # type: ignore[union-attr]
    except ImportError as exc:
        # Fallback: stub out missing summarise_code during utils import.
        if "summarise_code" in str(exc):
            ana_mod = sys.modules.get("chungoid.utils.analysis_utils")
            if ana_mod is None:
                ana_mod = _StubModule("chungoid.utils.analysis_utils")
                sys.modules["chungoid.utils.analysis_utils"] = ana_mod
            setattr(ana_mod, "summarise_code", lambda *args, **kwargs: "")
            # Retry import once
            spec.loader.exec_module(module)  # type: ignore[union-attr]
        else:
            raise
    return module.ChungoidEngine  # type: ignore[attr-defined]


def collect_tool_specs() -> List[Dict[str, Any]]:
    """Return list of tool spec dicts obtained from ChungoidEngine."""
    ChungoidEngine = _get_engine_class()
    # Bypass heavy __init__ by instantiating via __new__
    engine_instance = ChungoidEngine.__new__(ChungoidEngine)  # type: ignore[call-arg]
    tools: List[Dict[str, Any]] = ChungoidEngine.get_mcp_tools(engine_instance)  # type: ignore[arg-type]
    # Ensure description/key ordering maybe; but just return list.
    return tools


def validate_specs(specs: List[Dict[str, Any]]) -> List[str]:
    """Validate each spec against the JSON schema. Return list of errors."""
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    errors: List[str] = []
    for spec in specs:
        for err in validator.iter_errors(spec):
            errors.append(f"{spec.get('name', '<unknown>')}: {err.message}")
    return errors


def export_specs(specs: List[Dict[str, Any]]) -> List[Path]:
    TOOL_SPECS_JSON_DIR.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for spec in specs:
        fname = f"{spec['name']}.json"
        dest = TOOL_SPECS_JSON_DIR / fname
        dest.write_text(json.dumps(spec, indent=2), encoding="utf-8")
        written.append(dest)
    return written


def write_index(spec_files: List[Path]) -> None:
    lines = [HEADER_MD, ""]
    if not spec_files:
        lines.append("*(No tool specs found – control layer returned empty list)*\n")
    else:
        index_parent_dir = INDEX_MD.parent
        for path in sorted(spec_files):
            rel_link = path.relative_to(index_parent_dir)
            link_str = str(rel_link).replace("\\", "/")
            tool_name = path.stem
            lines.append(f"* :download:`{tool_name} <{link_str}>`")
        lines.append("")

    current_date = datetime.now().strftime("%Y-%m-%d")
    footer = [
        "",
        "---",
        "*This is a living document.*",
        f"*Last updated: {current_date} by Documentation Automation Script*"
    ]
    lines.extend(footer)
    
    INDEX_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    specs = collect_tool_specs()
    errors = validate_specs(specs)
    if errors:
        for e in errors:
            print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)

    spec_files = export_specs(specs)
    write_index(spec_files)
    print(f"Exported {len(spec_files)} tool spec(s) → {TOOL_SPECS_JSON_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

__all__ = [
    "collect_tool_specs",
    "export_specs",
    "write_index",
] 