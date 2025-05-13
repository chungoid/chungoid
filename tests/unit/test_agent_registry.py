# chungoid-core/tests/unit/utils/test_agent_registry.py
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime, timezone
import json
from typing import Any, Dict, Optional # Added Dict, Optional, Any

# Ensure this import path is correct based on your project structure and how pytest handles it.
# You might need to adjust it if chungoid-core is not directly on PYTHONPATH during tests.
from chungoid.utils.agent_registry import AgentRegistry, AgentCard
from chungoid.utils.chroma_client_factory import get_client # If you want to mock get_client behavior too

# Helper to create consistent AgentCard data
def create_test_card_data(agent_id: str, mcp_tool_schemas: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    now = datetime.now(timezone.utc)
    data = {
        "agent_id": agent_id,
        "name": f"Test Agent {agent_id}",
        "description": "A test agent.",
        "stage_focus": "testing",
        "capabilities": ["test_capability_1", "test_capability_2"],
        "tool_names": ["test_tool_A", "test_tool_B"],
        "metadata": {"custom_key": "custom_value"},
        "created": now,
        "mcp_tool_input_schemas": mcp_tool_schemas # Explicitly add
    }
    return data

class TestAgentRegistry(unittest.TestCase):

    @patch('chungoid.utils.agent_registry.get_client') # Mock at the source of usage
    def test_add_and_get_agent_without_mcp_schemas(self, mock_get_client):
        # 1. Setup Mocks
        mock_chroma_collection = MagicMock()
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_get_client.return_value = mock_chroma_client

        # 2. Initialize AgentRegistry
        registry = AgentRegistry(project_root=Path("."), chroma_mode="persistent")

        # 3. Create an AgentCard without mcp_tool_input_schemas
        test_card_data = create_test_card_data("test_agent_001", mcp_tool_schemas=None) # Explicitly None
        agent_card = AgentCard(**test_card_data)

        # Mock the _exists check
        mock_chroma_collection.get.return_value = {"ids": []}

        # 4. Call registry.add()
        registry.add(agent_card, overwrite=False)

        # 5. Assertions for 'add'
        mock_chroma_collection.add.assert_called_once()
        args, kwargs = mock_chroma_collection.add.call_args
        added_metadatas = kwargs.get('metadatas')

        self.assertEqual(len(added_metadatas), 1)
        actual_meta = added_metadatas[0]

        # Verify essential fields are present
        self.assertEqual(actual_meta.get("agent_id"), agent_card.agent_id)
        self.assertEqual(actual_meta.get("name"), agent_card.name)
        self.assertEqual(actual_meta.get("stage_focus"), agent_card.stage_focus)
        self.assertIn("created", actual_meta)
        self.assertEqual(actual_meta.get("_capabilities_str"), ",".join(agent_card.capabilities))
        self.assertEqual(actual_meta.get("_tool_names_str"), ",".join(agent_card.tool_names))
        self.assertEqual(actual_meta.get("_agent_card_metadata_json"), json.dumps(agent_card.metadata))
        # Assert that the mcp tool schema key is NOT present
        self.assertNotIn("_mcp_tool_input_schemas_json", actual_meta)


        # 6. Setup mock for 'get'
        mock_chroma_collection.get.reset_mock()
        mock_chroma_collection.get.return_value = {
            "ids": [agent_card.agent_id],
            "documents": [agent_card.description],
            "metadatas": [added_metadatas[0]]
        }

        # 7. Call registry.get()
        retrieved_card = registry.get(agent_card.agent_id)

        # 8. Assertions for 'get'
        self.assertIsNotNone(retrieved_card)
        self.assertEqual(retrieved_card.agent_id, agent_card.agent_id)
        self.assertEqual(retrieved_card.name, agent_card.name)
        self.assertEqual(retrieved_card.description, agent_card.description)
        self.assertEqual(retrieved_card.capabilities, agent_card.capabilities)
        self.assertEqual(retrieved_card.tool_names, agent_card.tool_names)
        self.assertEqual(retrieved_card.metadata, agent_card.metadata)
        self.assertEqual(retrieved_card.created, agent_card.created)
        # Assert mcp_tool_input_schemas is None after retrieval
        self.assertIsNone(retrieved_card.mcp_tool_input_schemas)

    @patch('chungoid.utils.agent_registry.get_client')
    def test_add_and_get_agent_with_mcp_tool_schemas(self, mock_get_client):
        # 1. Setup Mocks
        mock_chroma_collection = MagicMock()
        mock_chroma_client = MagicMock()
        mock_chroma_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_get_client.return_value = mock_chroma_client

        # 2. Initialize AgentRegistry
        registry = AgentRegistry(project_root=Path("."), chroma_mode="persistent")

        # 3. Create an AgentCard WITH mcp_tool_input_schemas
        tool_schemas = {
            "tool1": {"param1": "string", "param2": "int"},
            "tool2": {"argA": "boolean"}
        }
        test_card_data = create_test_card_data("test_agent_002", mcp_tool_schemas=tool_schemas)
        agent_card = AgentCard(**test_card_data)

        # Mock the _exists check
        mock_chroma_collection.get.return_value = {"ids": []}

        # 4. Call registry.add()
        registry.add(agent_card, overwrite=False)

        # 5. Assertions for 'add'
        mock_chroma_collection.add.assert_called_once()
        args, kwargs = mock_chroma_collection.add.call_args
        added_metadatas = kwargs.get('metadatas')

        self.assertEqual(len(added_metadatas), 1)
        actual_meta = added_metadatas[0]

        # Verify essential fields are present (as before)
        self.assertEqual(actual_meta.get("agent_id"), agent_card.agent_id)
        # ... other basic field checks if desired ...

        # Assert that the mcp tool schema key IS present and correctly serialized
        self.assertIn("_mcp_tool_input_schemas_json", actual_meta)
        self.assertEqual(actual_meta["_mcp_tool_input_schemas_json"], json.dumps(tool_schemas))

        # 6. Setup mock for 'get'
        mock_chroma_collection.get.reset_mock()
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
        # ... other basic field checks ...

        # Assert mcp_tool_input_schemas is correctly deserialized
        self.assertIsNotNone(retrieved_card.mcp_tool_input_schemas)
        self.assertEqual(retrieved_card.mcp_tool_input_schemas, tool_schemas)


if __name__ == '__main__':
    unittest.main()