import pytest
from unittest.mock import MagicMock
import chromadb
import os # Add os import
import logging

# Autouse fixture that patches utils.chroma_utils.get_chroma_client so
# no test ever tries to connect to a real Chroma server.

def pytest_configure(config):
    """Force flow registry to use memory mode during tests.""" # Simplified docstring
    os.environ["FLOW_REGISTRY_MODE"] = "memory"
    
    # Attempt to configure Chroma settings for local/testing mode
    # REMOVED ALL CHROMA ENV VAR SETTINGS AND DEFAULT_SETTINGS MODIFICATIONS
    # try:
    #     from chromadb.config import Settings, DEFAULT_SETTINGS
    #     # Environment variables first
    #     os.environ["CHROMA_API_IMPL"] = "chromadb.api.segment.SegmentAPI"
    #     os.environ["CHROMA_DB_IMPL"] = "duckdb+parquet"
    #     os.environ["CHROMA_IS_PERSISTENT"] = "TRUE" # Note: Chroma's Settings uses a bool, not str
    #     os.environ["CHROMA_ALLOW_RESET"] = "true"  # Note: Chroma's Settings uses a bool, not str

    #     # Attempt to directly modify a global settings object or influence future instances
    #     # This is highly speculative.
    #     if hasattr(chromadb, 'settings') and isinstance(chromadb.settings, Settings): # Old way, might not exist
    #         logger.debug("Attempting to patch chromadb.settings (old global)")
    #         chromadb.settings.chroma_api_impl = "chromadb.api.segment.SegmentAPI"
    #         chromadb.settings.chroma_db_impl = "duckdb+parquet"
    #         chromadb.settings.is_persistent = True
    #         chromadb.settings.allow_reset = True
        
    #     # Try to modify the DEFAULT_SETTINGS object which is more likely to be used by new Settings() instances
    #     logger.debug(f"Original DEFAULT_SETTINGS.is_persistent: {getattr(DEFAULT_SETTINGS, 'is_persistent', 'N/A')}")
    #     logger.debug(f"Original DEFAULT_SETTINGS.chroma_api_impl: {getattr(DEFAULT_SETTINGS, 'chroma_api_impl', 'N/A')}")
        
    #     DEFAULT_SETTINGS.is_persistent = True
    #     DEFAULT_SETTINGS.chroma_api_impl = "chromadb.api.segment.SegmentAPI"
    #     DEFAULT_SETTINGS.chroma_db_impl = "duckdb+parquet"
    #     DEFAULT_SETTINGS.allow_reset = True
    #     logger.debug(f"Patched DEFAULT_SETTINGS.is_persistent to: {DEFAULT_SETTINGS.is_persistent}")
    #     logger.debug(f"Patched DEFAULT_SETTINGS.chroma_api_impl to: {DEFAULT_SETTINGS.chroma_api_impl}")

    # except ImportError:
    #     print("CONTEST.PY: Could not import chromadb.config.Settings to adjust Chroma settings.")
    #     pass


# @pytest.fixture(autouse=True) # <<< COMMENT OUT AUTOUSE AGAIN
def fake_chroma_client(monkeypatch):
    from chungoid.utils import chroma_utils as cu

    fake_client = MagicMock(spec=chromadb.ClientAPI)
    fake_collection = MagicMock(spec=chromadb.Collection)

    # Common behaviour defaults
    fake_client.get_or_create_collection.return_value = fake_collection
    fake_client.get_collection.return_value = fake_collection
    fake_client.list_collections.return_value = []
    fake_collection.add.return_value = None
    fake_collection.query.return_value = {
        "ids": [[]],
        "metadatas": [[]],
        "documents": [[]],
        "distances": [[]],
    }
    fake_collection.count.return_value = 0

    # Attempt to patch the new name. This might still raise an AttributeError
    # if conftest itself is using a different chroma_utils instance, but it's the target.
    # This whole try/except might be irrelevant now that autouse is off, 
    # but keeping it won't harm for this step.
    try:
        # Assuming we will rename get_chroma_client_DEBUG_VERSION back to get_chroma_client in chroma_utils.py next
        monkeypatch.setattr(cu, "get_chroma_client", lambda: fake_client) 
    except AttributeError:
        print("CONTEST.PY: get_chroma_client not found on cu module by conftest.py during manual call (autouse is off)")
        pass # Allow tests to proceed and fail at their own points

    yield fake_client  # Tests may use it if they like 