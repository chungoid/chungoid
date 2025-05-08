import importlib, sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_core_snapshot_builds_dict():
    mod = importlib.import_module("dev.scripts.embed_core_snapshot")
    snapshot = mod._build_snapshot()  # type: ignore[attr-defined]

    required = {"type", "core_commit", "core_version", "created", "stage_files", "tool_specs"}
    assert required.issubset(snapshot), f"Missing keys: {required - snapshot.keys()}"
    assert snapshot["type"] == "core_snapshot" 