import importlib, sys, pathlib, importlib.util, pytest, itertools
ROOT = pathlib.Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_core_snapshot_builds_dict():
    spec = importlib.util.find_spec("dev.scripts.embed_core_snapshot")
    if spec is None:
        here = pathlib.Path(__file__).resolve()
        script_path = None
        for parent in itertools.chain([here.parent], here.parents):
            cand = parent / "dev" / "scripts" / "embed_core_snapshot.py"
            if cand.is_file():
                script_path = cand
                break
        if script_path is None:
            pytest.skip("embed_core_snapshot script not available in this repository layout", allow_module_level=True)
        spec = importlib.util.spec_from_file_location("embed_core_snapshot", script_path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["embed_core_snapshot"] = mod  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
    else:
        mod = importlib.import_module("dev.scripts.embed_core_snapshot")

    snapshot = mod._build_snapshot()  # type: ignore[attr-defined]

    required = {"type", "core_commit", "core_version", "created", "stage_files", "tool_specs"}
    assert required.issubset(snapshot), f"Missing keys: {required - snapshot.keys()}"
    assert snapshot["type"] == "core_snapshot" 