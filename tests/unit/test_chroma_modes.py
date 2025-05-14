import pytest
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, ANY
import logging
# import urllib.parse # Not currently needed with _factory_get_client mock
# import sys # Not currently needed

from chungoid.utils import chroma_utils
# from chungoid.utils import config_loader # Not directly used by test logic if get_config is patched in chroma_utils
import chromadb
from chromadb.config import Settings # Ensure Settings is imported for type hints if needed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pytestmark = pytest.mark.legacy # Ensure removed


class TestChromaModes:
    """Test different ChromaDB client instantiation modes (Pytest style)."""

    # Removed setUp and tearDown as individual tests will manage their context
    # and tmp_path fixture is used for persistent mode.

    # UNSKIPPED
    @patch("chungoid.utils.chroma_utils.get_config")
    @patch("chungoid.utils.chroma_utils._factory_get_client") 
    def test_http_mode(self, mock_factory_get_client: MagicMock, mock_get_config: MagicMock): # Removed tmp_path as it's not used
        """Test HTTP mode client instantiation, ensuring factory is called."""
        mock_config_http = {"chromadb": {"mode": "http", "url": "http://localhost:8000"}}
        mock_get_config.return_value = mock_config_http
        mock_http_client_instance = MagicMock(spec=chromadb.HttpClient)
        mock_factory_get_client.return_value = mock_http_client_instance

        # Reset singleton state for test isolation
        chroma_utils._client = None
        chroma_utils._client_mode = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context() 

        client = chroma_utils.get_chroma_client()

        mock_get_config.assert_called_once()
        mock_factory_get_client.assert_called_once_with(
            mode="http", 
            server_url="http://localhost:8000", 
            project_dir=Path(".") 
        )
        assert client == mock_http_client_instance, "Returned client should be the one from the factory"

    # UNSKIPPED
    @patch("chromadb.PersistentClient") 
    @patch("os.makedirs")
    @patch("chungoid.utils.chroma_utils.get_config")
    def test_persistent_mode(self, mock_get_config: MagicMock, mock_makedirs: MagicMock, mock_persistent_client_ctor: MagicMock, tmp_path: Path):
        """Test persistent mode client instantiation, checking os.makedirs and client constructor."""
        project_dir = tmp_path / "my_project"
        project_dir.mkdir()
        expected_db_path_in_factory = project_dir.resolve() / ".chungoid" / "chroma_db"

        mock_config_persistent = {"chromadb": {"mode": "persistent"}}
        mock_get_config.return_value = mock_config_persistent
        
        mock_returned_instance = MagicMock(spec_set=True)
        mock_persistent_client_ctor.return_value = mock_returned_instance

        chroma_utils._client = None
        chroma_utils._client_mode = None
        chroma_utils._client_project_context = None
        chroma_utils.set_chroma_project_context(project_dir) 

        client = chroma_utils.get_chroma_client()

        mock_get_config.assert_called_once()
        mock_makedirs.assert_called_with(str(expected_db_path_in_factory), exist_ok=True)
        mock_persistent_client_ctor.assert_called_once_with(path=str(expected_db_path_in_factory), settings=ANY)
        assert client == mock_returned_instance, "Returned client should be the instance from the mocked PersistentClient constructor"
        
        chroma_utils.clear_chroma_project_context()

    @patch('chungoid.utils.chroma_utils.get_config')
    def test_http_mode_direct(self, mock_get_config: MagicMock, tmp_path: Path):
        """Test HTTP mode client instantiation - simplified, testing get_config mock (Pytest style)."""
        mock_config_http = {"chromadb": {"mode": "http", "url": "http://localhost:8000"}}
        mock_get_config.return_value = mock_config_http
        
        chroma_utils._client = None
        chroma_utils._client_mode = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context()

        print(f"TEST_DIRECT: chroma_utils module: {chroma_utils}")
        print(f"TEST_DIRECT: chroma_utils.__file__: {chroma_utils.__file__}")
        print(f"TEST_DIRECT: id(chroma_utils.get_chroma_client) in test: {id(chroma_utils.get_chroma_client)}")
        print(f"TEST_DIRECT: id(mock_get_config) in test: {id(mock_get_config)}") 
        print(f"TEST_DIRECT: mock_get_config is chroma_utils.get_config in test before call: {mock_get_config is chroma_utils.get_config}")

        print("DEBUG_DIRECT: In test_http_mode_direct, about to call get_chroma_client()")
        try:
            chroma_utils.get_chroma_client() 
            print("DEBUG_DIRECT: In test_http_mode_direct, call to get_chroma_client() completed.")
        except Exception as e:
            print(f"DEBUG_DIRECT: In test_http_mode_direct, get_chroma_client() raised an exception: {e}")
        
        print(f"TEST_DIRECT: mock_get_config is chroma_utils.get_config in test after call: {mock_get_config is chroma_utils.get_config}")
        mock_get_config.assert_called_once()

# if __name__ == "__main__": # This is a unittest pattern, not needed for pytest
#     unittest.main() 