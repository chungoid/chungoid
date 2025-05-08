import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent  # repo root
PYPROJECT = ROOT / "chungoid-core" / "pyproject.toml"
LLMS_DIR = ROOT / "dev" / "llms-txt"

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

pytest.xfail("Doc manifest coverage not enforced in CI yet.", allow_module_level=True)

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
        manifest = LLMS_DIR / name / ver / "manifest.yaml"
        if not manifest.is_file():
            missing.append(f"{name}=={ver}")
    if missing:
        pytest.fail(
            "The following dependencies are missing documentation manifests. "
            "Run sync_library_docs.py or fix pipeline: " + ", ".join(missing)
        ) 