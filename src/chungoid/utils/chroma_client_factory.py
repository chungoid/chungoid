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

    # Remove protocol if present
    url_no_protocol = server_url.replace("http://", "").replace("https://", "")

    # Split host:port from path (database name)
    host_port_str, *path_segments = url_no_protocol.split("/", 1)
    database = path_segments[0] if path_segments else None

    # Split host from port
    if ":" not in host_port_str:
        # default port
        host, port_str = host_port_str, "8000"
    else:
        host, port_str = host_port_str.split(":", 1)
    
    try:
        port = int(port_str)
    except ValueError as exc:
        raise ValueError(f"Invalid port '{port_str}' in server_url '{server_url}'") from exc

    # Pass database to HttpClient, it has its own default if None is passed.
    return chromadb.HttpClient(host=host, port=port, settings=s, database=database) 