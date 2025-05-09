"""Package that ships JSON/YAML schema files used at runtime.

This file exists solely so that `setuptools` treats the *schemas* directory as
package data, ensuring the files are included in wheels and editable installs.
"""

from importlib import resources as _resources

# Re-export a convenience helper so callers can do
# `stage_flow_schema_json = schemas.read_text("stage_flow_schema.json")`

def read_text(name: str) -> str:  # pragma: no cover
    """Read a bundled schema file as text."""
    return _resources.files(__name__).joinpath(name).read_text(encoding="utf-8")

__all__ = ["read_text"] 