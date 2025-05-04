import unittest
from unittest.mock import patch, MagicMock
import asyncio  # <<< Import asyncio for running async tests

# Add project root to path to allow importing utils
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils import chroma_utils


class TestChromaUtils(unittest.TestCase):
    def setUp(self):
        # Reset the shared client before each test
        chroma_utils._client = None

    @patch("utils.chroma_utils.get_config")  # <<< Mock get_config
    @patch("chromadb.HttpClient")
    async def test_get_chroma_client_http(self, mock_http_client, mock_get_config):
        """Test get_chroma_client initialization with http type from config."""
        # Configure the mock to return specific config values
        mock_get_config.return_value = {
            "chromadb": {
                "client_type": "http",
                "host": "testhost",
                "port": 9999,
                "persist_path": "./unused",
            }
        }

        # Reset the internal client for testing isolation
        chroma_utils._client = None
        mock_instance = MagicMock()
        mock_http_client.return_value = mock_instance

        client = await chroma_utils.get_chroma_client()

        self.assertIsNotNone(client)
        mock_http_client.assert_called_once_with(host="testhost", port=9999)
        mock_instance.heartbeat.assert_called_once()
        mock_get_config.assert_called_once()  # Ensure config was fetched
        # Test singleton behavior
        client2 = await chroma_utils.get_chroma_client()
        self.assertIs(client, client2)
        mock_http_client.assert_called_once()  # Client should not be created again
        mock_get_config.assert_called_once()  # Config should not be fetched again (depends on loader caching)

    @patch("utils.chroma_utils.get_config")  # <<< Mock get_config
    @patch("chromadb.PersistentClient")
    @patch("os.path.exists", return_value=True)  # Assume path exists
    @patch("os.makedirs")  # Mock makedirs
    async def test_get_chroma_client_persistent(
        self, mock_makedirs, mock_exists, mock_persistent_client, mock_get_config
    ):
        """Test get_chroma_client initialization with persistent type from config."""
        mock_get_config.return_value = {
            "chromadb": {
                "client_type": "persistent",
                "host": "unused",
                "port": 0,
                "persist_path": "./test_chroma_data",
            }
        }

        chroma_utils._client = None
        mock_instance = MagicMock()
        mock_persistent_client.return_value = mock_instance

        client = await chroma_utils.get_chroma_client()

        self.assertIsNotNone(client)
        mock_persistent_client.assert_called_once_with(path="./test_chroma_data")
        mock_instance.list_collections.assert_called_once()  # Basic check for persistent
        mock_exists.assert_called_once_with("./test_chroma_data")
        mock_makedirs.assert_not_called()  # Because mock_exists returned True
        mock_get_config.assert_called_once()

        client2 = await chroma_utils.get_chroma_client()
        self.assertIs(client, client2)
        mock_persistent_client.assert_called_once()
        mock_get_config.assert_called_once()

    @patch("utils.chroma_utils.get_config")  # <<< Mock get_config
    async def test_get_chroma_client_invalid_type(self, mock_get_config):
        """Test get_chroma_client with an invalid client type from config."""
        mock_get_config.return_value = {"chromadb": {"client_type": "invalid"}}

        chroma_utils._client = None
        client = await chroma_utils.get_chroma_client()
        self.assertIsNone(client)  # Should return None on config error
        mock_get_config.assert_called_once()

    # Test connection error during http init
    @patch("utils.chroma_utils.get_config")
    @patch("chromadb.HttpClient", side_effect=ConnectionError("Test connection failed"))
    async def test_get_chroma_client_http_connection_error(self, mock_http_client, mock_get_config):
        """Test http client init fails gracefully on connection error."""
        mock_get_config.return_value = {
            "chromadb": {"client_type": "http", "host": "badhost", "port": 1234}
        }
        chroma_utils._client = None
        client = await chroma_utils.get_chroma_client()
        self.assertIsNone(client)
        mock_http_client.assert_called_once_with(host="badhost", port=1234)
        mock_get_config.assert_called_once()

    # TODO: Add tests for other core functions like get_or_create_collection, add_documents etc., mocking the client methods.


# Note: Running async tests might require an async test runner like pytest-asyncio or similar setup.
# This structure uses standard unittest for simplicity, assuming direct async calls work in the test environment.

# Use asyncio test runner if possible
if __name__ == "__main__":
    # Basic runner needs adjustment for async tests.
    # Wrap test methods in asyncio.run or use an async test runner like pytest-asyncio
    suite = unittest.TestSuite()
    for test_name in unittest.defaultTestLoader.getTestCaseNames(TestChromaUtils):
        # Check if the test method is async
        test_method = getattr(TestChromaUtils(test_name), test_name)
        if asyncio.iscoroutinefunction(test_method):
            # Wrap async test methods
            suite.addTest(
                unittest.FunctionTestCase(lambda: asyncio.run(test_method()))
            )  # Needs instance? Revisit runner.
        else:
            suite.addTest(TestChromaUtils(test_name))
    # This basic runner might still have issues with async setup/teardown.
    # Recommend using pytest with pytest-asyncio for robust async testing.
    print("Running tests (basic runner - consider pytest-asyncio for robust async tests):")
    # unittest.main() # Original simple runner
    runner = unittest.TextTestRunner()
    runner.run(suite)
