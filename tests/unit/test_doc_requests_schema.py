import yaml, jsonschema, pathlib

SCHEMA_PATH = pathlib.Path(__file__).resolve().parents[3] / "schemas" / "doc_requests_schema.yaml"

schema = yaml.safe_load(SCHEMA_PATH.read_text())


def test_doc_requests_schema_allows_minimal_entry():
    sample = [{
        "library_id": "fastapi/docs",
        "reason": "Context7 returned 404"
    }]
    jsonschema.validate(sample, schema) 