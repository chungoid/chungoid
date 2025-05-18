#!/usr/bin/env python3
"""Export JSON-Schema files into the documentation tree.

This helper copies every `*.json` file from the project-level ``schemas/``
directory into ``dev/docs/reference/schemas`` so that Sphinx can expose them as
static assets (downloadable from Read-the-Docs or GitHub Pages builds).

It also writes/overwrites a Markdown index file ``dev/docs/reference/schemas_index.md``
that lists each schema with a direct **:download:** link pointing to the
copied asset.  The file is referenced from the main ``index.rst`` so that
it shows up in the rendered documentation sidebar.

Designed for use in CI (see docs-build workflow) but can be run locally:

    python dev/scripts/export_jsonschemas.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import List
from types import ModuleType
import importlib.util, types
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]  # /chungoid-mcp
SCHEMA_SRC = ROOT / "schemas"
REFLECTION_SCHEMA_NAME = "a2a_reflection_schema.json"
DOCS_OUTPUT_SCHEMAS_DIR = ROOT / "dev" / "docs" / "reference" / "schemas"
INDEX_MD = ROOT / "dev" / "docs" / "reference" / "schemas_index.md"

def _generate_reflection_schema() -> Path | None:
    """Generate Reflection model JSON schema to schemas/ directory.

    Returns the Path written, or None if generation failed.
    """
    dest = SCHEMA_SRC / REFLECTION_SCHEMA_NAME
    try:
        # Dynamically import reflection_store.py without requiring chromadb.
        ref_path = ROOT / "chungoid-core" / "src" / "chungoid" / "utils" / "reflection_store.py"
        if not ref_path.exists():
            return None

        # Stub chromadb and submodule so reflection_store import doesn't error.
        chroma_stub = ModuleType("chromadb")
        chroma_api_stub = ModuleType("chromadb.api")
        # Minimal placeholder classes so `from chromadb.api import ClientAPI, Collection` succeeds.
        class _Dummy:  # noqa: D401
            pass

        chroma_api_stub.ClientAPI = _Dummy  # type: ignore[attr-defined]
        chroma_api_stub.Collection = _Dummy  # type: ignore[attr-defined]

        sys.modules.setdefault("chromadb", chroma_stub)
        sys.modules.setdefault("chromadb.api", chroma_api_stub)

        # Stub out chungoid.utils and its chroma_client_factory to satisfy relative imports
        utils_stub = ModuleType("chungoid.utils")
        chroma_factory_stub = ModuleType("chungoid.utils.chroma_client_factory")

        def _dummy_get_client(*_args, **_kwargs):  # noqa: D401
            class _DummyClient:  # noqa: D401
                pass

            return _DummyClient()

        chroma_factory_stub.get_client = _dummy_get_client  # type: ignore[attr-defined]

        sys.modules.setdefault("chungoid", ModuleType("chungoid"))
        sys.modules.setdefault("chungoid.utils", utils_stub)
        sys.modules.setdefault("chungoid.utils.chroma_client_factory", chroma_factory_stub)

        # Load module under its package name so relative imports work
        spec = importlib.util.spec_from_file_location("chungoid.utils.reflection_store", ref_path)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        Reflection = getattr(mod, "Reflection", None)
        if Reflection is None:
            return None
        schema_json = Reflection.schema_json(indent=2)  # type: ignore[attr-defined]

        SCHEMA_SRC.mkdir(exist_ok=True)
        dest.write_text(schema_json, encoding="utf-8")
        return dest
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Could not generate reflection schema: {exc}", file=sys.stderr)
        return None

HEADER_MD = """# JSON Schemas\n\nThe following machine-readable contracts are bundled with the Chungoid\ncodebase and are available for download when the documentation is built.\n\n> **Tip**: Right-click a filename and choose *Save link as…* to grab the raw\n> JSON file.\n"""

def copy_schema_files() -> List[Path]:
    """Copy all ``*.json`` from *schemas* → *dev/docs/reference/schemas*.

    Returns a list of destination paths that were (re)written."""
    DOCS_OUTPUT_SCHEMAS_DIR.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for src in SCHEMA_SRC.glob("*.json"):
        dest = DOCS_OUTPUT_SCHEMAS_DIR / src.name
        shutil.copy2(src, dest)
        copied.append(dest)
    return copied


def write_index(schema_files: List[Path]) -> None:
    """Render a Markdown bullet list with Sphinx **:download:** links."""
    lines = [HEADER_MD, ""]
    if not schema_files:
        lines.append("*(No schema files found)*\n")
    else:
        index_parent_dir = INDEX_MD.parent
        for path in sorted(schema_files):
            rel_link = path.relative_to(index_parent_dir)
            link_str = str(rel_link).replace("\\", "/")
            lines.append(f"* :download:`{path.name} <{link_str}>`")
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
    # Ensure dynamic Reflection schema is available before copy step.
    _generate_reflection_schema()

    if not SCHEMA_SRC.exists():
        print(f"[ERROR] Source dir {SCHEMA_SRC} is missing", file=sys.stderr)
        sys.exit(1)

    copied = copy_schema_files()
    write_index(copied)
    print(f"Exported {len(copied)} schema(s) → {DOCS_OUTPUT_SCHEMAS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

__all__ = [
    "copy_schema_files",
    "write_index",
] 