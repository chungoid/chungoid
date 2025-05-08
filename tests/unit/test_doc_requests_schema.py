import yaml, jsonschema, pathlib, itertools

# Locate schema file by walking up parent directories
def _find_schema(start: pathlib.Path) -> pathlib.Path:
    for parent in itertools.chain([start], start.parents):
        candidate = parent / "schemas" / "doc_requests_schema.yaml"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("Could not locate doc_requests_schema.yaml in ancestors")

SCHEMA_PATH = _find_schema(pathlib.Path(__file__).resolve())
schema = yaml.safe_load(SCHEMA_PATH.read_text())


def test_doc_requests_schema_allows_minimal_entry():
    sample = [{
        "library_id": "fastapi/docs",
        "reason": "Context7 returned 404"
    }]
    jsonschema.validate(sample, schema) 