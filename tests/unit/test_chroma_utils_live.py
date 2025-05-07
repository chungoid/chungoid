from importlib import reload
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils import chroma_utils
import utils.chroma_utils as cu


def _fake_get_client(mode, project_dir, server_url=None):
    """Return a sentinel object marking which mode was requested."""
    return SimpleNamespace(mode=mode, project=str(project_dir), server_url=server_url)


@pytest.fixture(autouse=True)
def fake_chroma_client(monkeypatch):
    # Do NOT patch anything here; just neutralise the original autouse fixture.
    yield


@pytest.fixture(autouse=True)
def patch_factory(monkeypatch):
    # Patch the factory before each test
    # prevent global conftest from patching chroma_utils.get_chroma_client
    monkeypatch.setattr(chroma_utils, "_factory_get_client", _fake_get_client)
    # Ensure singleton cleared
    chroma_utils._client = None
    chroma_utils._client_project_context = None
    yield
    chroma_utils._client = None
    chroma_utils._client_project_context = None


def test_persistent_mode(tmp_path, monkeypatch):
    # Set config to persistent
    cfg_yaml = """chromadb:\n  mode: persistent\n"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    from utils import config_loader

    config_loader._config = None
    config_loader.load_config(str(cfg_path))
    # Set project context
    chroma_utils.set_chroma_project_context(tmp_path)

    client = chroma_utils.get_chroma_client()
    assert client.mode == "persistent"


def test_http_mode_env(monkeypatch, tmp_path):
    # No config file, but env forces http
    monkeypatch.setenv("CHUNGOID_CHROMA_MODE", "http")
    from chungoid.utils import config_loader
    from chungoid.utils import chroma_utils

    config_loader._config = None
    config_loader.load_config(config_path=str(tmp_path / "nonexistent_config.yaml"))

    client = chroma_utils.get_chroma_client()
    assert client.mode == "http"

    monkeypatch.delenv("CHUNGOID_CHROMA_MODE")


def test_persistent_mode_from_config(tmp_path):
    # Set config to persistent
    cfg_yaml = """chromadb:\n  mode: persistent\n"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    from utils import config_loader

    config_loader._config = None
    config_loader.load_config(str(cfg_path))
    # Set project context
    chroma_utils.set_chroma_project_context(tmp_path)

    client = chroma_utils.get_chroma_client()
    assert client.mode == "persistent" 