import json
from pathlib import Path

import yaml
from jsonschema import validate, Draft202012Validator

SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "stage_flow_schema.json"
EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "dev" / "examples" / "sample_flow.yaml"


def test_example_flow_validates():
    schema = json.loads(SCHEMA_PATH.read_text())
    data = yaml.safe_load(EXAMPLE_PATH.read_text())
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    assert not errors, f"Schema validation errors: {[e.message for e in errors]}" 