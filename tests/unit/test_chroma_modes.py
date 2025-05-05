import unittest
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import logging
import pytest

from utils import chroma_utils
from utils import config_loader
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

pytestmark = pytest.mark.legacy


class TestChromaModes(unittest.TestCase):
    """Verify get_chroma_client honour the new `chromadb.mode` config key."""

    def setUp(self):
        # Fresh temp project dirs for persistent mode
        self.temp_root = Path(tempfile.mkdtemp())
        self.project_dir = self.temp_root / "proj"
        self.project_dir.mkdir()

        # Reset singleton + context between tests
        chroma_utils._client = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context()

    def tearDown(self):
        chroma_utils._client = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context()
        shutil.rmtree(self.temp_root)

    @patch("chromadb.HttpClient")
    @patch("utils.config_loader.get_config")
    def test_http_mode(self, mock_get_config, mock_http_ctor):
        """`mode: http` returns an HttpClient with server_url parsing."""
        cfg = {
            "chromadb": {
                "mode": "http",
                "server_url": "testhost:9999",
            }
        }
        mock_get_config.return_value = cfg
        mock_http_client_instance = MagicMock()
        mock_http_ctor.return_value = mock_http_client_instance

        client = chroma_utils.get_chroma_client()

        self.assertIs(client, mock_http_client_instance)
        mock_http_ctor.assert_called_once_with(host="testhost", port=9999)

    @patch("chromadb.PersistentClient")
    @patch("os.makedirs")
    @patch("utils.config_loader.get_config")
    def test_persistent_mode(self, mock_get_config, mock_makedirs, mock_persist_ctor):
        """`mode: persistent` requires project context and creates proper path."""
        cfg = {"chromadb": {"mode": "persistent"}}
        mock_get_config.return_value = cfg
        mock_pc_instance = MagicMock()
        mock_persist_ctor.return_value = mock_pc_instance

        # context must be set first
        chroma_utils.set_chroma_project_context(self.project_dir)
        client = chroma_utils.get_chroma_client()

        self.assertIs(client, mock_pc_instance)
        expected_dir = self.project_dir / ".chungoid" / "chroma_db"
        mock_makedirs.assert_called_once_with(expected_dir, exist_ok=True)
        mock_persist_ctor.assert_called_once_with(path=str(expected_dir))


if __name__ == "__main__":
    unittest.main() 