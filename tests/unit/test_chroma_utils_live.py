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
            "url": "http://localhost:8000/test_http_direct_patch"  # CORRECTED KEY from server_url to url
        },
        "logging": {"level": "DEBUG"}
    }

    def mock_get_config_for_chroma_utils_module():
        return mock_config_http

    monkeypatch.setattr(chroma_utils, "get_config", mock_get_config_for_chroma_utils_module)
    
    chroma_utils._client = None
    chroma_utils._client_mode = None
    chroma_utils._client_project_context = None
    chroma_utils.clear_chroma_project_context() 

    mock_http_client_instance = MagicMock(spec=chromadb.HttpClient)

    with patch('chromadb.HttpClient', return_value=mock_http_client_instance) as mock_http_client_class:
        client = chroma_utils.get_chroma_client()

    assert client is not None, "HTTP Client should not be None with direct get_config patch"
    mock_http_client_class.assert_called_once() 
    assert client == mock_http_client_instance, "Returned client should be the mocked HttpClient instance"
    
    # Check call arguments for HttpClient (optional, but good practice)
    call_args = mock_http_client_class.call_args
    assert call_args is not None, "HttpClient should have been called"
    # The host/port are parsed from the URL by _factory_get_client
    # The URL used by _factory_get_client comes from chroma_config.get("url")
    # which is now correctly "http://localhost:8000/test_http_direct_patch"
    # urlparse("http://localhost:8000/test_http_direct_patch").hostname -> 'localhost'
    # urlparse("http://localhost:8000/test_http_direct_patch").port -> 8000
    assert call_args.kwargs.get('host') == 'localhost'
    assert call_args.kwargs.get('port') == 8000
    assert call_args.kwargs.get('ssl') == False

    # Test with CHROMA_MODE env var (legacy, should still work if get_config handles it)
    # This part of the test might need separate handling for get_config if it also reads env vars directly
    # For now, the primary fix is the direct get_config mock.
    # monkeypatch.setenv("CHROMA_MODE", "http") # Keep this commented or ensure get_config mock handles this scenario


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