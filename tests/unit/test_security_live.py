import os
from pathlib import Path

import pytest

from chungoid.utils.security import safe_resolve, PathTraversalError


def test_safe_resolve_within_base(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    # normal path
    resolved = safe_resolve(base, "sub/file.txt")
    assert resolved == (base / "sub/file.txt").resolve()
    # create dirs to ensure no error
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.touch()
    assert resolved.exists()


def test_safe_resolve_escape(tmp_path):
    base = tmp_path / "root"
    base.mkdir()
    with pytest.raises(PathTraversalError):
        safe_resolve(base, "../evil.txt")


@pytest.mark.parametrize("missing_base", ["missing_dir", "missing_dir/nested"])

def test_safe_resolve_missing_base(tmp_path, missing_base):
    with pytest.raises(FileNotFoundError):
        safe_resolve(tmp_path / missing_base, "a.txt") 