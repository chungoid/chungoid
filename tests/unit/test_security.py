import pytest
from pathlib import Path
from utils.security import safe_resolve, PathTraversalError


def test_safe_resolve_inside(tmp_path: Path):
    base = tmp_path / "project"
    base.mkdir()
    # create nested path
    nested = base / "sub" / "file.txt"
    nested.parent.mkdir()
    nested.write_text("hello")

    resolved = safe_resolve(base, "sub/file.txt")
    assert resolved == nested.resolve()


def test_safe_resolve_traversal(tmp_path: Path):
    base = tmp_path / "project"
    base.mkdir()
    # Attempt traversal out of base dir
    with pytest.raises(PathTraversalError):
        safe_resolve(base, "../other/outside.txt") 