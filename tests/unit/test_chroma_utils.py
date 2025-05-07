import unittest
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import logging
import pytest
pytestmark = pytest.mark.legacy

# Correct import path assuming tests are run from the project root or configured PYTHONPATH
from chungoid.utils import chroma_utils
from chungoid.utils import config_loader
import chromadb

# Configure logging for tests
logging.basicConfig(level=logging.INFO) # Use INFO to reduce noise unless debugging
logger = logging.getLogger(__name__)

class TestChromaUtils(unittest.TestCase):

    def setUp(self):
        """Set up test environment."""
        logger.debug("Setting up TestChromaUtils test...")
        # Create a temporary directory for fake projects
        self.test_dir = tempfile.mkdtemp()
        logger.debug(f"Created temp dir: {self.test_dir}")
        self.project_dir_1 = Path(self.test_dir) / "project1"
        self.project_dir_2 = Path(self.test_dir) / "project2"
        self.project_dir_1.mkdir()
        self.project_dir_2.mkdir()

        # Reset the singleton client and context before each test
        chroma_utils._client = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context()
        self.assertIsNone(chroma_utils._current_project_directory, "Context should be clear initially")

    def tearDown(self):
        """Clean up test environment."""
        logger.debug("Tearing down TestChromaUtils test...")
        # Reset the singleton client and context after each test
        chroma_utils._client = None
        chroma_utils._client_project_context = None
        chroma_utils.clear_chroma_project_context()
        # Remove the temporary directory
        shutil.rmtree(self.test_dir)
        logger.debug(f"Removed temp dir: {self.test_dir}")

    def test_set_and_clear_project_context(self):
        """Test setting and clearing the project context."""
        logger.info("Running test_set_and_clear_project_context")
        self.assertIsNone(chroma_utils._current_project_directory)
        chroma_utils.set_chroma_project_context(self.project_dir_1)
        self.assertEqual(chroma_utils._current_project_directory, self.project_dir_1.resolve())
        chroma_utils.clear_chroma_project_context()
        self.assertIsNone(chroma_utils._current_project_directory)

    @patch('chungoid.utils.config_loader.get_config')
    @patch('chromadb.HttpClient')  # Patch HttpClient constructor; spec not needed inside
    def test_get_chroma_client_http(self, mock_http_client_constructor, mock_get_config):
        """Test get_chroma_client returns HttpClient when configured."""
        logger.info("Running test_get_chroma_client_http")
        # Configure mocks
        mock_get_config.return_value = {
            "chromadb": {"client_type": "http", "host": "testhost", "port": 9999}
        }
        # Create a simple mock instance to be returned by the patched constructor
        mock_http_client_instance = MagicMock()
        mock_http_client_constructor.return_value = mock_http_client_instance

        # Call the function
        client = chroma_utils.get_chroma_client()

        # Assertions
        self.assertIs(client, mock_http_client_instance)
        mock_http_client_constructor.assert_called_once_with(host="testhost", port=9999)
        self.assertIsNone(chroma_utils._client_project_context)

        # Call again, should return same instance
        client_again = chroma_utils.get_chroma_client()
        self.assertIs(client_again, client)
        mock_http_client_constructor.assert_called_once() # Constructor not called again

    @patch('chungoid.utils.config_loader.get_config')
    @patch('chromadb.PersistentClient')  # Patch PersistentClient constructor
    @patch('os.makedirs') # Mock os.makedirs to avoid actual FS operations
    def test_get_chroma_client_persistent_success(self, mock_makedirs, mock_persistent_client_constructor, mock_get_config):
        """Test get_chroma_client returns PersistentClient when configured and context is set."""
        logger.info("Running test_get_chroma_client_persistent_success")
        # Configure mocks
        mock_get_config.return_value = {"chromadb": {"client_type": "persistent"}}
        mock_persistent_client_instance = MagicMock()
        mock_persistent_client_constructor.return_value = mock_persistent_client_instance

        # Set project context
        chroma_utils.set_chroma_project_context(self.project_dir_1)

        # Call the function
        client = chroma_utils.get_chroma_client()

        # Assertions
        self.assertIs(client, mock_persistent_client_instance)
        expected_path = str(self.project_dir_1.resolve() / ".chungoid" / "chroma_db")
        # Check if makedirs was called correctly by the helper
        mock_makedirs.assert_called_once_with(Path(expected_path), exist_ok=True)
        # Check if PersistentClient was called with the correct path by the helper
        mock_persistent_client_constructor.assert_called_once_with(path=expected_path)
        self.assertEqual(chroma_utils._client_project_context, self.project_dir_1.resolve())

        # Call again, should return same instance
        client_again = chroma_utils.get_chroma_client()
        self.assertIs(client_again, client)
        mock_persistent_client_constructor.assert_called_once() # Constructor not called again

    @patch('chungoid.utils.config_loader.get_config')
    def test_get_chroma_client_persistent_no_context(self, mock_get_config):
        """Test get_chroma_client returns None and logs error if persistent but no context."""
        logger.info("Running test_get_chroma_client_persistent_no_context")
        mock_get_config.return_value = {"chromadb": {"client_type": "persistent"}}
        self.assertIsNone(chroma_utils._current_project_directory, "Context should be None initially")

        with self.assertLogs(chroma_utils.logger, level='ERROR') as log_cm:
            client = chroma_utils.get_chroma_client()

        self.assertIsNone(client, "Client should be None when context is missing")
        self.assertTrue(any("project context directory not set" in msg for msg in log_cm.output),
                        "Error message about missing context not logged")

    @patch('chungoid.utils.config_loader.get_config')
    @patch('chromadb.PersistentClient')
    @patch('os.makedirs')
    def test_get_chroma_client_persistent_context_change(self, mock_makedirs, mock_persistent_client, mock_get_config):
        """Test get_chroma_client re-initializes if context changes."""
        logger.info("Running test_get_chroma_client_persistent_context_change")
        mock_get_config.return_value = {"chromadb": {"client_type": "persistent"}}
        mock_persistent_instance_1 = MagicMock()
        mock_persistent_instance_2 = MagicMock()
        mock_persistent_client.side_effect = [mock_persistent_instance_1, mock_persistent_instance_2]

        # --- First context ---
        chroma_utils.set_chroma_project_context(self.project_dir_1)
        client1 = chroma_utils.get_chroma_client()
        self.assertIs(client1, mock_persistent_instance_1)
        expected_path_1 = str(self.project_dir_1.resolve() / ".chungoid" / "chroma_db")
        mock_persistent_client.assert_called_with(path=expected_path_1)
        mock_makedirs.assert_called_with(Path(expected_path_1), exist_ok=True)
        call_count_after_1 = mock_persistent_client.call_count
        self.assertEqual(call_count_after_1, 1)
        makedirs_calls_after_1 = mock_makedirs.call_count

        # --- Change context --- (This implicitly clears the old client in get_chroma_client)
        chroma_utils.set_chroma_project_context(self.project_dir_2)
        with self.assertLogs(chroma_utils.logger, level='WARNING') as log_cm:
             client2 = chroma_utils.get_chroma_client()

        self.assertIs(client2, mock_persistent_instance_2)
        self.assertIsNot(client1, client2)
        expected_path_2 = str(self.project_dir_2.resolve() / ".chungoid" / "chroma_db")
        mock_persistent_client.assert_called_with(path=expected_path_2) # Check last call
        mock_makedirs.assert_called_with(Path(expected_path_2), exist_ok=True) # Check last call
        self.assertEqual(mock_persistent_client.call_count, 2, "Constructor should be called again")
        self.assertEqual(mock_makedirs.call_count, makedirs_calls_after_1 + 1, "makedirs should be called again")
        self.assertTrue(any("context changed" in msg for msg in log_cm.output),
                        "Warning about context change not logged")

    @patch('chungoid.utils.config_loader.get_config')
    @patch('chungoid.utils.chroma_utils.get_persistent_chroma_client') # Patch the helper
    def test_get_chroma_client_persistent_uses_helper(self, mock_get_persistent_helper, mock_get_config):
        """Verify get_chroma_client calls the dedicated get_persistent_chroma_client helper."""
        logger.info("Running test_get_chroma_client_persistent_uses_helper")
        mock_get_config.return_value = {"chromadb": {"client_type": "persistent"}}
        mock_helper_instance = MagicMock()
        mock_get_persistent_helper.return_value = mock_helper_instance

        chroma_utils.set_chroma_project_context(self.project_dir_1)
        client = chroma_utils.get_chroma_client()

        self.assertIs(client, mock_helper_instance)
        mock_get_persistent_helper.assert_called_once_with(self.project_dir_1.resolve())


if __name__ == '__main__':
    # To run tests from the command line within the chungoid-core directory:
    # python -m unittest tests/unit/test_chroma_utils.py
    unittest.main()
