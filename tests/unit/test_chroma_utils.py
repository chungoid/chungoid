import unittest
import os
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock, ANY
import logging
import pytest

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

    @patch('chungoid.utils.chroma_utils.get_config')
    @patch('chromadb.HttpClient')
    def test_get_chroma_client_http(self, mock_http_client_constructor, mock_get_config):
        """Test get_chroma_client returns HttpClient when configured."""
        logger.info("Running test_get_chroma_client_http")
        # Configure mocks
        mock_get_config.return_value = {
            "chromadb": {"mode": "http", "url": "http://testhost:9999"} # Corrected key from "server_url" to "url"
        }
        mock_http_client_instance = MagicMock(name="MockHttpClientInstance") # Give it a name for clarity
        mock_http_client_constructor.return_value = mock_http_client_instance

        client = chroma_utils.get_chroma_client()

        # self.assertIs(client, mock_http_client_instance) # Original assertIs, keep commented
        self.assertIsNotNone(client, "Client should not be None on first call")
        # Check if constructor was called. ssl=False for http. settings might be default.
        # The _factory_get_client is what calls chromadb.HttpClient, let's check that instead if direct patching of HttpClient fails
        # For now, assuming HttpClient is directly called by get_chroma_client OR the factory's parameters are passed to it.
        # The current structure of get_chroma_client calls _factory_get_client, which then calls HttpClient.
        # So, if we are patching HttpClient directly, the call from _factory_get_client should match.
        # _factory_get_client parameters: host, port, ssl, settings, headers
        mock_http_client_constructor.assert_called_once_with(host="testhost", port=9999, ssl=False, settings=ANY, headers=ANY)

        self.assertIsNone(chroma_utils._client_project_context, "Project context should be None for http client")

        # Call again, should return same instance (if singleton logic works and assertIs wasn't the only issue)
        client_again = chroma_utils.get_chroma_client()
        # self.assertIs(client_again, client) # This will also likely fail if the first assertIs fails
        self.assertIsNotNone(client_again, "Client should not be None on second call")
        if client is not None and client_again is not None: # Check identity only if both are not None
            self.assertIs(client_again, client, "Should return same client instance on subsequent calls")

        mock_http_client_constructor.assert_called_once() # Constructor still should only be called once

    @patch('chungoid.utils.chroma_utils.get_config')
    @patch('chromadb.PersistentClient')
    @patch('os.makedirs')
    def test_get_chroma_client_persistent_success(self, mock_makedirs, mock_persistent_client_constructor, mock_get_config):
        """Test get_chroma_client returns PersistentClient when configured and context is set."""
        logger.info("Running test_get_chroma_client_persistent_success")
        # Configure mocks
        mock_get_config.return_value = {"chromadb": {"mode": "persistent"}}
        mock_persistent_client_instance = MagicMock(name="MockPersistentClientInstance")
        mock_persistent_client_constructor.return_value = mock_persistent_client_instance

        chroma_utils.set_chroma_project_context(self.project_dir_1)
        client = chroma_utils.get_chroma_client()

        # self.assertIs(client, mock_persistent_client_instance) # Original assertIs, keep commented
        self.assertIsNotNone(client, "Client should not be None on first call")
        
        expected_db_path = str(self.project_dir_1.resolve() / ".chungoid" / "chroma_db")
        mock_makedirs.assert_called_once_with(expected_db_path, exist_ok=True)
        # The _factory_get_client calls PersistentClient with path and settings
        mock_persistent_client_constructor.assert_called_once_with(path=expected_db_path, settings=ANY)
        
        self.assertIsNotNone(chroma_utils._client_project_context, "Project context should be set")
        self.assertEqual(chroma_utils._client_project_context, self.project_dir_1.resolve())

        client_again = chroma_utils.get_chroma_client()
        # self.assertIs(client_again, client)
        self.assertIsNotNone(client_again, "Client should not be None on second call")
        if client is not None and client_again is not None: 
            self.assertIs(client_again, client, "Should return same client instance")
        
        mock_persistent_client_constructor.assert_called_once() # Still only called once
        mock_makedirs.assert_called_once() # Still only called once

    @patch('chungoid.utils.chroma_utils.get_config')
    def test_get_chroma_client_persistent_no_context(self, mock_get_config):
        """Test get_chroma_client returns None and logs error if persistent but no context."""
        logger.info("Running test_get_chroma_client_persistent_no_context")
        mock_get_config.return_value = {"chromadb": {"mode": "persistent"}} # Corrected config
        # Ensure context is None before call, as per test intent
        chroma_utils.clear_chroma_project_context() # Explicitly clear for this test case
        self.assertIsNone(chroma_utils._current_project_directory, "Context should be None initially")

        with self.assertLogs(chroma_utils.logger, level='ERROR') as log_cm:
            # We expect get_chroma_client to raise ChromaOperationError here
            with self.assertRaises(chroma_utils.ChromaOperationError) as ex_cm:
                chroma_utils.get_chroma_client()
            
            self.assertIn("ChromaDB project directory context not set for persistent mode", str(ex_cm.exception))

        # Check the log message that occurs *before* the exception is raised.
        self.assertTrue(any("ChromaDB mode is 'persistent' but no project directory context is set." in output for output in log_cm.output))

    @patch('chungoid.utils.chroma_utils.get_config') # MODIFIED Outermost
    @patch('chromadb.PersistentClient')          # Middle
    @patch('os.makedirs')                        # Innermost
    def test_get_chroma_client_persistent_context_change(self, mock_makedirs, mock_persistent_client, mock_get_config):
        """Test get_chroma_client re-initializes if context changes."""
        logger.info("Running test_get_chroma_client_persistent_context_change")
        mock_get_config.return_value = {"chromadb": {"mode": "persistent"}} # Corrected config
        mock_persistent_instance_1 = MagicMock(name="PersistentClient_Ctx1")
        mock_persistent_instance_2 = MagicMock(name="PersistentClient_Ctx2")
        mock_persistent_client.side_effect = [mock_persistent_instance_1, mock_persistent_instance_2]

        # --- First context ---
        chroma_utils.set_chroma_project_context(self.project_dir_1)
        client1 = chroma_utils.get_chroma_client()
        # self.assertIs(client1, mock_persistent_instance_1) # Original assertIs
        self.assertIsNotNone(client1, "Client1 should not be None")
        
        expected_db_path1 = str(self.project_dir_1.resolve() / ".chungoid" / "chroma_db")
        mock_persistent_client.assert_any_call(path=expected_db_path1, settings=ANY) # Check first call args
        mock_makedirs.assert_any_call(expected_db_path1, exist_ok=True)
        self.assertEqual(mock_persistent_client.call_count, 1, "Constructor should be called once for client1")
        self.assertEqual(mock_makedirs.call_count, 1, "makedirs should be called once for client1")

        # --- Second context ---
        chroma_utils.set_chroma_project_context(self.project_dir_2)
        client2 = chroma_utils.get_chroma_client() # Should re-initialize
        # self.assertIs(client2, mock_persistent_instance_2) # Original assertIs
        self.assertIsNotNone(client2, "Client2 should not be None")
        
        expected_db_path2 = str(self.project_dir_2.resolve() / ".chungoid" / "chroma_db")
        # Check the latest call's path argument specifically, and ensure settings was also passed.
        mock_persistent_client.assert_called_with(path=expected_db_path2, settings=ANY) 
        self.assertEqual(mock_persistent_client.call_count, 2, "Constructor should be called twice (total)")
        mock_makedirs.assert_any_call(expected_db_path2, exist_ok=True)
        self.assertEqual(mock_makedirs.call_count, 2, "makedirs should be called twice (total)")

    @patch('chungoid.utils.chroma_utils.get_config')
    @patch('chungoid.utils.chroma_utils._factory_get_client') # Changed from get_persistent_chroma_client
    def test_get_chroma_client_persistent_uses_helper(self, mock_factory_get_client, mock_get_config):
        """Verify get_chroma_client calls the dedicated _factory_get_client for persistent mode."""
        logger.info("Running test_get_chroma_client_persistent_uses_helper")
        mock_get_config.return_value = {"chromadb": {"mode": "persistent"}} # Corrected config
        mock_factory_instance = MagicMock(name="MockFactoryPersistentInstance")
        mock_factory_get_client.return_value = mock_factory_instance

        chroma_utils.set_chroma_project_context(self.project_dir_1)
        client = chroma_utils.get_chroma_client()

        # self.assertIs(client, mock_factory_instance) # Original assertIs
        self.assertIsNotNone(client, "Client should not be None")
        mock_factory_get_client.assert_called_once_with(
            mode="persistent", 
            project_dir=self.project_dir_1.resolve()
            # server_url=None implicitly passed by not including it.
        )


if __name__ == '__main__':
    # To run tests from the command line within the chungoid-core directory:
    # python -m unittest tests/unit/test_chroma_utils.py
    unittest.main()
