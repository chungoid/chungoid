import json
from pathlib import Path

import yaml
from jsonschema import validate, Draft202012Validator

# Resolve repo root (three parents up from tests/unit/)
ROOT = Path(__file__).resolve().parents[3]

# The schema lives under ./schemas/ in meta repo and under ./chungoid-core/schemas/ in public repo.
_schema = ROOT / "schemas" / "stage_flow_schema.json"
if not _schema.exists():
    _schema = ROOT / "chungoid-core" / "schemas" / "stage_flow_schema.json"

# Example flow likewise.
_example = ROOT / "dev" / "examples" / "sample_flow.yaml"
if not _example.exists():
    _example = ROOT / "chungoid-core" / "dev" / "examples" / "sample_flow.yaml"

SCHEMA_PATH = _schema
EXAMPLE_PATH = _example


def test_example_flow_validates():
    schema = json.loads(SCHEMA_PATH.read_text())
    data = yaml.safe_load(EXAMPLE_PATH.read_text())
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    assert not errors, f"Schema validation errors: {[e.message for e in errors]}" 