import textwrap
from pathlib import Path

import pytest

# Import helper fns from script
from importlib.util import spec_from_file_location, module_from_spec

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "dev" / "scripts" / "validate_planning_files.py"

# Skip the entire module if the helper script is absent (e.g., in slim public repo)
if not SCRIPT_PATH.exists():  # pragma: no cover
    pytest.skip("validate_planning_files.py not present", allow_module_level=True)

spec = spec_from_file_location("validate_planning", SCRIPT_PATH)
validate_planning = module_from_spec(spec)
spec.loader.exec_module(validate_planning)

validate_file = validate_planning.validate_file
check_pairs = validate_planning.check_pairs


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


def test_validate_file_success(tmp_path: Path):
    md = _write(
        tmp_path,
        "demo_blueprint.md",
        """---\ntitle: Demo\ncategory: blueprint\nowner: test\ncreated: 2025-05-08\nstatus: draft\n---\ncontent""",
    )
    errors = validate_file(md)
    assert errors == []


def test_validate_file_missing_key(tmp_path: Path):
    bad = _write(
        tmp_path,
        "bad_roadmap.md",
        """---\ntitle: Bad\ncategory: roadmap\ncreated: 2025-05-08\nstatus: draft\n---\ntext""",
    )
    errors = validate_file(bad)
    assert any("missing keys" in e for e in errors)


def test_check_pairs(tmp_path: Path):
    # create only roadmap, expect error for missing blueprint
    r = _write(
        tmp_path,
        "feature_roadmap.md",
        """---\ntitle: Feature Roadmap\ncategory: roadmap\nowner: t\ncreated: 2025-05-08\nstatus: draft\n---\n""",
    )
    errors = check_pairs([r])
    assert "blueprint" in errors[0]

    # add matching blueprint, errors cleared
    b = _write(
        tmp_path,
        "feature_blueprint.md",
        """---\ntitle: Feature Blueprint\ncategory: blueprint\nowner: t\ncreated: 2025-05-08\nstatus: draft\n---\n""",
    )
    errors2 = check_pairs([r, b])
    assert errors2 == [] 