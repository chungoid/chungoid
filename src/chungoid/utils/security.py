from __future__ import annotations

"""Security helpers for path operations.

Currently only exposes `safe_resolve`, a helper that resolves a path while
ensuring it stays within an allowed base directory.  The intention is to give
stronger guarantees against path-traversal attempts when user-supplied
relative paths are combined with the project directory.
"""
from pathlib import Path

__all__: list[str] = ["safe_resolve", "PathTraversalError"]


class PathTraversalError(ValueError):
    """Raised when a resolved path escapes the allowed base directory."""


def safe_resolve(base_dir: Path | str, relative_path: Path | str) -> Path:
    """Return an absolute path that must remain inside *base_dir*.

    Parameters
    ----------
    base_dir
        The directory that acts as the root.  Must already exist and be a
        directory (checked at runtime).
    relative_path
        The user-supplied path that should be *within* ``base_dir``.  The path
        is **not** allowed to contain traversals that would escape the base
        directory (e.g. ``../../etc/passwd``).

    Returns
    -------
    Path
        The fully-resolved absolute path.

    Raises
    ------
    PathTraversalError
        If the resolved path is outside ``base_dir``.
    FileNotFoundError
        If ``base_dir`` does not exist.
    NotADirectoryError
        If ``base_dir`` is not a directory.
    """
    base_dir_path = Path(base_dir).resolve()
    if not base_dir_path.exists():
        raise FileNotFoundError(f"Base directory '{base_dir}' does not exist.")
    if not base_dir_path.is_dir():
        raise NotADirectoryError(
            f"Base path '{base_dir_path}' is not a directory; cannot perform safe resolution."
        )

    abs_path = (base_dir_path / relative_path).resolve()

    try:
        abs_path.relative_to(base_dir_path)
    except ValueError as exc:
        # Attempted traversal detected
        raise PathTraversalError(
            f"Resolved path '{abs_path}' escapes the base directory '{base_dir_path}'."
        ) from exc

    return abs_path 