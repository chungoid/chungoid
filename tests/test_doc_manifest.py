import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root

# In the meta-monorepo the core package lives under ./chungoid-core/.
# In the standalone public repo the pyproject lives at repo root.
_core_path = ROOT / "chungoid-core" / "pyproject.toml"
PYPROJECT = _core_path if _core_path.exists() else ROOT / "pyproject.toml"

LLMS_DIR = ROOT / "dev" / "llms-txt"

# Public repo may not ship the offline doc cache – skip the test in that case.
if not LLMS_DIR.exists():  # pragma: no cover
    pytest.skip("Offline library docs directory missing", allow_module_level=True)

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

def _dependencies() -> list[tuple[str, str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    deps = data.get("project", {}).get("dependencies", [])  # type: ignore[index]
    result: list[tuple[str, str]] = []
    for dep in deps:
        if "==" in dep:
            name, ver = dep.split("==", 1)
            result.append((name, ver))
    return result


def test_library_doc_manifests_exist():
    missing: list[str] = []
    for name, ver in _dependencies():
        # First look for a manifest matching the exact pinned version.
        manifest = LLMS_DIR / name / ver / "manifest.yaml"

        if not manifest.is_file():
            # Fallback: many pre-fetched offline docs are stored under a "latest" directory.
            manifest = LLMS_DIR / name / "latest" / "manifest.yaml"

        if not manifest.is_file():
            # Only flag as missing if any docs directory exists for that library.
            if (LLMS_DIR / name).exists():
                missing.append(f"{name}=={ver}")

    if missing:
        pytest.fail(
            "The following dependencies are missing documentation manifests. "
            "Run sync_library_docs.py or fix pipeline: " + ", ".join(missing)
        ) 