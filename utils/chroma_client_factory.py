"""Factory for creating ChromaDB client instances (Persistent or HTTP).

This small helper isolates the low-level logic needed to initialise a
`chromadb` client so that the rest of the codebase only needs to call
`get_client` with a desired mode.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings

__all__ = ["get_client"]


def _normalise_mode(mode: str) -> str:
    """Return lower-cased normalised mode name."""
    if not mode:
        raise ValueError("Chroma mode may not be empty/null")
    mode_l = mode.lower()
    if mode_l not in {"persistent", "http"}:
        raise ValueError(f"Unsupported Chroma mode: {mode}")
    return mode_l


def get_client(
    mode: str,
    project_path: Path,
    *,
    settings: Optional[Settings] = None,
    server_url: Optional[str] = None,
) -> ClientAPI:
    """Return an initialised Chroma client.

    Parameters
    ----------
    mode
        "persistent" or "http" (case-insensitive).
    project_path
        Root directory of the project. Used to locate `.chungoid/chroma_db`
        when `mode == "persistent"`.
    settings
        Optional explicit `chromadb.Settings` object.  If *None*, a default
        `Settings()` is used.
    server_url
        Required only when `mode == "http"`.  Should look like
        "http://host:port" (schema optional).
    """
    mode_n = _normalise_mode(mode)
    s = settings or Settings()

    if mode_n == "persistent":
        db_dir = (project_path / ".chungoid" / "chroma_db").resolve()
        db_dir.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(db_dir), settings=s)

    # ---- HTTP client ----
    if not server_url:
        raise ValueError("server_url must be provided when mode == 'http'")

    # Remove protocol if present & split host/port
    server_url = server_url.replace("http://", "").replace("https://", "")
    if ":" not in server_url:
        # default port
        host, port_s = server_url, "8000"
    else:
        host, port_s = server_url.split(":", 1)
    try:
        port = int(port_s)
    except ValueError as exc:
        raise ValueError(f"Invalid port in server_url '{server_url}'") from exc

    return chromadb.HttpClient(host=host, port=port, settings=s) 