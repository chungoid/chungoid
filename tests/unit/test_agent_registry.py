# chungoid-core/tests/unit/utils/test_agent_registry.py
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone
import json

# Ensure this import path is correct based on your project structure and how pytest handles it.
# You might need to adjust it if chungoid-core is not directly on PYTHONPATH during tests.
from chungoid.utils.agent_registry import AgentRegistry, AgentCard 
from chungoid.utils.chroma_client_factory import get_client # If you want to mock get_client behavior too

class TestAgentRegistry(unittest.TestCase):

    @patch('chungoid.utils.agent_registry.get_client') # Mock at the source of usage
    def test_add_and_get_agent_with_complex_metadata(self, mock_get_client):
        # 1. Setup Mocks
        mock_chroma_collection = MagicMock()
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_get_client.return_value = mock_chroma_client

        # 2. Initialize AgentRegistry (it will use the mocked client)
        registry = AgentRegistry(project_root=Path("."), chroma_mode="persistent")

        # 3. Create an AgentCard with various data types
        now = datetime.now(timezone.utc)
        test_card_data = {
            "agent_id": "test_agent_001",
            "name": "Test Agent",
            "description": "A test agent with full fields.",
            "stage_focus": "testing",
            "capabilities": ["test_capability_1", "test_capability_2"],
            "tool_names": ["test_tool_A", "test_tool_B"],
            "metadata": {
                "custom_key": "custom_value", 
                "nested_level": 1,
                "is_active": True
            },
            "created": now,
        }
        agent_card = AgentCard(**test_card_data)

        # Mock the _exists check to simulate agent not existing initially
        mock_chroma_collection.get.return_value = {"ids": []} # For _exists check
        
        # 4. Call registry.add()
        registry.add(agent_card, overwrite=False)

        # 5. Assertions for 'add'
        # Check that chromadb.add was called once
        mock_chroma_collection.add.assert_called_once()
        
        # Get the arguments passed to collection.add
        args, kwargs = mock_chroma_collection.add.call_args
        added_ids = kwargs.get('ids')
        added_documents = kwargs.get('documents')
        added_metadatas = kwargs.get('metadatas')

        self.assertEqual(added_ids, [agent_card.agent_id])
        self.assertEqual(added_documents, [agent_card.description])
        
        expected_chroma_meta = {
            "agent_id": agent_card.agent_id,
            "name": agent_card.name,
            "stage_focus": agent_card.stage_focus,
            "created": agent_card.created.isoformat(), # Pydantic model_dump serializes datetime
            "_capabilities_str": ",".join(agent_card.capabilities),
            "_tool_names_str": ",".join(agent_card.tool_names),
            "_agent_card_metadata_json": json.dumps(agent_card.metadata),
        }
        # Note: Pydantic's model_dump for datetime includes microseconds and tzinfo by default.
        # Chroma stores datetime as string. Ensure comparison is consistent.
        # For simplicity here, we assume the structure. In a real test, you might need to parse
        # 'created' from added_metadatas[0] if it's not directly isoformat.
        # Or, more robustly, check each key-value pair.
        
        self.assertEqual(len(added_metadatas), 1)
        actual_meta = added_metadatas[0]
        
        self.assertEqual(actual_meta.get("agent_id"), expected_chroma_meta["agent_id"])
        self.assertEqual(actual_meta.get("name"), expected_chroma_meta["name"])
        self.assertEqual(actual_meta.get("stage_focus"), expected_chroma_meta["stage_focus"])
        # Datetime comparison needs care if not using exact string match from pydantic dump
        self.assertTrue("created" in actual_meta) # Check presence at least
        self.assertEqual(actual_meta.get("_capabilities_str"), expected_chroma_meta["_capabilities_str"])
        self.assertEqual(actual_meta.get("_tool_names_str"), expected_chroma_meta["_tool_names_str"])
        self.assertEqual(actual_meta.get("_agent_card_metadata_json"), expected_chroma_meta["_agent_card_metadata_json"])


        # 6. Setup mock for 'get'
        # Simulate what ChromaDB's get would return
        mock_chroma_collection.get.reset_mock() # Reset from the _exists call
        mock_chroma_collection.get.return_value = {
            "ids": [agent_card.agent_id],
            "documents": [agent_card.description],
            "metadatas": [added_metadatas[0]] # Use the metadata that was supposedly added
        }

        # 7. Call registry.get()
        retrieved_card = registry.get(agent_card.agent_id)

        # 8. Assertions for 'get'
        self.assertIsNotNone(retrieved_card)
        self.assertEqual(retrieved_card.agent_id, agent_card.agent_id)
        self.assertEqual(retrieved_card.name, agent_card.name)
        self.assertEqual(retrieved_card.description, agent_card.description)
        self.assertEqual(retrieved_card.stage_focus, agent_card.stage_focus)
        self.assertEqual(retrieved_card.capabilities, agent_card.capabilities)
        self.assertEqual(retrieved_card.tool_names, agent_card.tool_names)
        self.assertEqual(retrieved_card.metadata, agent_card.metadata) # Crucial check
        self.assertEqual(retrieved_card.created, agent_card.created)


if __name__ == '__main__':
    unittest.main()