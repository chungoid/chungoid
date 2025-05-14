import pytest
from unittest.mock import MagicMock
import chromadb

# Autouse fixture that patches utils.chroma_utils.get_chroma_client so
# no test ever tries to connect to a real Chroma server.

# @pytest.fixture(autouse=True) # <<< COMMENTED OUT AUTOUSE
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