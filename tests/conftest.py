import pytest
from unittest.mock import MagicMock
import chromadb

# Autouse fixture that patches utils.chroma_utils.get_chroma_client so
# no test ever tries to connect to a real Chroma server.
@pytest.fixture(autouse=True)
def fake_chroma_client(monkeypatch):
    from utils import chroma_utils as cu

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

    # Patch the singleton accessor to always return our fake
    monkeypatch.setattr(cu, "get_chroma_client", lambda: fake_client)

    yield fake_client  # Tests may use it if they like 