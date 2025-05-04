import unittest
from unittest.mock import patch, MagicMock
import os

# Add project root to path to allow importing utils
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Module to test
from utils import chroma_utils

# We need to import the actual exception for patching side_effect
import chromadb.errors


class TestChromaUtils(unittest.TestCase):

    def setUp(self):
        # Reset the cached client before each test to ensure isolation
        chroma_utils._client = None

    # 1. Test with environment variables set
    @patch.dict(os.environ, {'CHROMA_HOST': 'env_host', 'CHROMA_PORT': '9999'})
    @patch('utils.chroma_utils.chromadb.HttpClient')
    def test_get_chroma_client_with_env_vars(self, mock_http_client):
        """Test get_chroma_client uses environment variables if set."""
        mock_instance = MagicMock()
        # Simulate successful connection (heartbeat returns a value)
        mock_instance.heartbeat.return_value = 12345
        mock_http_client.return_value = mock_instance

        client = chroma_utils.get_chroma_client()

        self.assertIsNotNone(client)
        mock_http_client.assert_called_once_with(host='env_host', port=9999)
        self.assertIs(client, mock_instance) # Ensure the created instance is returned

        # Test singleton behavior
        client2 = chroma_utils.get_chroma_client()
        self.assertIs(client, client2) # Should return the cached client
        mock_http_client.assert_called_once() # Should not be called again

    # 2. Test with environment variables NOT set (using defaults)
    @patch.dict(os.environ, {}, clear=True) # Ensure env vars are cleared for this test
    @patch('utils.chroma_utils.chromadb.HttpClient')
    def test_get_chroma_client_with_defaults(self, mock_http_client):
        """Test get_chroma_client uses default host/port if env vars are not set."""
        mock_instance = MagicMock()
        mock_http_client.return_value = mock_instance

        # Reset singleton before test
        chroma_utils._client = None
        client = chroma_utils.get_chroma_client()

        self.assertIsNotNone(client)
        # Verify defaults are used (assuming 'localhost' and 8000 are defaults in the code)
        mock_http_client.assert_called_once_with(host='localhost', port=8000)
        self.assertIs(client, mock_instance)

        # Test singleton behavior
        client2 = chroma_utils.get_chroma_client()
        self.assertIs(client, client2)
        mock_http_client.assert_called_once()

    # 3. Test connection error handling
    @patch.dict(os.environ, {'CHROMA_HOST': 'bad_host', 'CHROMA_PORT': '1111'})
    # Simulate HttpClient raising an error on instantiation or heartbeat
    @patch('utils.chroma_utils.chromadb.HttpClient')
    def test_get_chroma_client_connection_error(self, mock_http_client):
        """Test get_chroma_client handles connection errors gracefully."""
        # Simulate instantiation failing with a generic exception
        mock_http_client.side_effect = Exception("Connection refused") # Use generic Exception

        # Reset singleton before test
        chroma_utils._client = None
        client = chroma_utils.get_chroma_client()

        self.assertIsNone(client, "Client should be None on connection error")
        mock_http_client.assert_called_once_with(host='bad_host', port=1111)

    # 4. Test scenario where environment variables are partially set (should use defaults for missing ones)
    @patch.dict(os.environ, {'CHROMA_HOST': 'partial_host'}, clear=True) # Only HOST is set
    @patch('utils.chroma_utils.chromadb.HttpClient')
    def test_get_chroma_client_partial_env_vars(self, mock_http_client):
        """Test get_chroma_client uses defaults for partially set env vars."""
        mock_instance = MagicMock()
        mock_http_client.return_value = mock_instance

        # Reset singleton before test
        chroma_utils._client = None
        client = chroma_utils.get_chroma_client()

        self.assertIsNotNone(client)
        # Host from env, Port should be default (8000)
        mock_http_client.assert_called_once_with(host='partial_host', port=8000)
        self.assertIs(client, mock_instance)


if __name__ == '__main__':
    unittest.main()
