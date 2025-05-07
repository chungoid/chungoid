from importlib import reload
from pathlib import Path
from types import SimpleNamespace

import pytest

from chungoid.utils import chroma_utils
import chungoid.utils.chroma_utils as cu
import chromadb


def _fake_get_client(mode, project_dir, server_url=None):
    """Return a sentinel object marking which mode was requested."""
    return SimpleNamespace(mode=mode, project=str(project_dir), server_url=server_url)


@pytest.fixture(autouse=True)
def fake_chroma_client(monkeypatch):
    # Do NOT patch anything here; just neutralise the original autouse fixture.
    yield


@pytest.fixture()
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


def test_persistent_mode(tmp_path, monkeypatch, patch_factory):
    # Set config to persistent
    cfg_yaml = """chromadb:\n  mode: persistent\n"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    from chungoid.utils import config_loader

    config_loader._config = None
    config_loader.load_config(str(cfg_path))
    # Set project context
    chroma_utils.set_chroma_project_context(tmp_path)

    client = chroma_utils.get_chroma_client()
    assert client.mode == "persistent"


def test_http_mode_env(monkeypatch, tmp_path):
    # from chungoid.utils import chroma_utils # Only import chroma_utils # No longer needed here if autouse is off for this test
    # from chungoid.utils.chroma_utils import _factory_get_client as original_factory # Get original factory # No longer needed

    # This test now MOCKS config_loader.get_config DIRECTLY within chroma_utils
    # and also MOCKS chromadb.HttpClient to prevent actual connection attempts.
    from unittest.mock import MagicMock, patch
    from chungoid.utils import chroma_utils # Import here for clarity
    import chromadb # Required for isinstance check

    # Define the mock config that get_config within chroma_utils will return
    mock_config_http = {
        "chromadb": {
            "mode": "http",
            "server_url": "http://localhost:8000/test_http_direct_patch" 
        },
        "logging": {"level": "DEBUG"}
    }

    def mock_get_config_for_chroma_utils_module():
        return mock_config_http

    # Patch get_config AS IT IS USED BY chroma_utils.py
    monkeypatch.setattr(chroma_utils, "get_config", mock_get_config_for_chroma_utils_module)
    
    # Crucially, reset any cached client and context in chroma_utils
    chroma_utils._client = None
    chroma_utils._client_mode = None
    chroma_utils._client_project_context = None
    chroma_utils.clear_chroma_project_context() # Ensure project context is None for http mode

    # ---- DEBUGGING ----
    # print(f"DEBUG: chroma_utils.get_config() returns: {chroma_utils.get_config()}")
    # print(f"DEBUG: chroma_utils._factory_get_client is: {chroma_utils._factory_get_client}")
    # print(f"DEBUG: cu._factory_get_client is: {cu._factory_get_client}")
    # ---- END DEBUGGING ----

    # Mock chromadb.HttpClient to avoid actual network calls
    mock_http_client_instance = MagicMock(spec=chromadb.HttpClient)

    with patch('chromadb.HttpClient', return_value=mock_http_client_instance) as mock_http_client_class:
        client = chroma_utils.get_chroma_client()

    assert client is not None, "HTTP Client should not be None with direct get_config patch"
    mock_http_client_class.assert_called_once() # Verify HttpClient was called
    assert client == mock_http_client_instance, "Returned client should be the mocked HttpClient instance"
    # Further assertions can be made on mock_http_client_class.call_args if needed,
    # e.g., to check host, port, database, settings. For example:
    # call_args = mock_http_client_class.call_args
    # assert call_args.kwargs['host'] == 'localhost'
    # assert call_args.kwargs['port'] == 8000
    # assert call_args.kwargs['database'] == 'test_http_direct_patch'

    # Test with CHROMA_MODE env var (legacy, should still work if get_config handles it)
    monkeypatch.setenv("CHROMA_MODE", "http")


def test_persistent_mode_from_config(tmp_path, patch_factory):
    # Set config to persistent
    cfg_yaml = """chromadb:\n  mode: persistent\n"""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(cfg_yaml)

    from chungoid.utils import config_loader

    config_loader._config = None
    config_loader.load_config(str(cfg_path))
    # Set project context
    chroma_utils.set_chroma_project_context(tmp_path)

    client = chroma_utils.get_chroma_client()
    assert client.mode == "persistent" 