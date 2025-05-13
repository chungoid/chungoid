import unittest
from unittest.mock import patch, MagicMock
import uuid
import json
from datetime import datetime, timezone

# Adjust import path based on your structure
from chungoid.utils.a2a_utils import generate_correlation_id, get_reflections_by_correlation_id
from chungoid.utils.reflection_store import Reflection, ReflectionStore # Assuming Reflection class is here


class TestA2AUtils(unittest.TestCase):

    @patch('uuid.uuid4')
    def test_generate_correlation_id(self, mock_uuid4):
        # Arrange
        expected_uuid = uuid.UUID('12345678-1234-5678-1234-567812345678')
        mock_uuid4.return_value = expected_uuid

        # Act
        correlation_id = generate_correlation_id()

        # Assert
        mock_uuid4.assert_called_once()
        self.assertEqual(correlation_id, str(expected_uuid))
        self.assertIsInstance(correlation_id, str)
        self.assertEqual(len(correlation_id), 36)

    def test_get_reflections_by_correlation_id_found(self):
        # Arrange
        target_correlation_id = "corr-123"
        mock_store_instance = MagicMock(spec=ReflectionStore)
        current_time = datetime.now(timezone.utc)

        # These are the Reflection objects that store.query() would return
        mock_query_results = [
             Reflection(
                 message_id='r1', conversation_id='conv1', agent_id='agent1', 
                 content_type='tool_call', content='doc1', timestamp=current_time,
                 metadata={'event': 'tool_call', 'agent_id': 'agent1', 'extra': '''{"correlation_id": "corr-123", "other": "data"}'''}
             ),
             Reflection(
                 message_id='r2', conversation_id='conv1', agent_id='agent2',
                 content_type='tool_call', content='doc2', timestamp=current_time,
                 metadata={'event': 'tool_call', 'agent_id': 'agent2', 'extra': '''{"correlation_id": "corr-abc", "other": "data"}'''}
             ),
             Reflection(
                 message_id='r3', conversation_id='conv1', agent_id='agent1',
                 content_type='thought', content='doc3', timestamp=current_time,
                 metadata={'event': 'thought', 'agent_id': 'agent1', 'extra': '''{"correlation_id": "corr-123", "thought_step": 1}'''}
             ),
        ]
        mock_store_instance.query.return_value = mock_query_results # Mock store.query()
        
        # Act
        found_reflections = get_reflections_by_correlation_id(mock_store_instance, target_correlation_id)
        
        # Assert
        mock_store_instance.query.assert_called_once_with(limit=2000) # Verify query was called correctly
        self.assertEqual(len(found_reflections), 2)
        found_reflections.sort(key=lambda r: r.message_id)
        self.assertEqual(found_reflections[0].message_id, 'r1') 
        self.assertEqual(found_reflections[1].message_id, 'r3')

    def test_get_reflections_by_correlation_id_not_found(self):
        # Arrange
        target_correlation_id = "corr-not-present"
        mock_store_instance = MagicMock(spec=ReflectionStore)
        current_time = datetime.now(timezone.utc)
        mock_query_results = [
             Reflection(message_id='r1', conversation_id='conv1', agent_id='agent1', content_type='tool_call', content='doc1', timestamp=current_time, metadata={'event': 'tool_call', 'agent_id': 'agent1', 'extra': '''{"correlation_id": "corr-123"}'''}),
        ]
        mock_store_instance.query.return_value = mock_query_results # Mock store.query()
        # Act
        found_reflections = get_reflections_by_correlation_id(mock_store_instance, target_correlation_id)
        # Assert
        mock_store_instance.query.assert_called_once_with(limit=2000)
        self.assertEqual(len(found_reflections), 0)

    def test_get_reflections_by_correlation_id_malformed_extra(self):
        # Arrange
        target_correlation_id = "corr-123"
        mock_store_instance = MagicMock(spec=ReflectionStore)
        current_time = datetime.now(timezone.utc)

        mock_query_results = [
            Reflection(
                message_id='r1', conversation_id='conv1', agent_id='agent1', 
                content_type='tool_call', content='doc1', timestamp=current_time,
                metadata={'event': 'tool_call', 'agent_id': 'agent1', 'extra': '''{"correlation_id": "corr-123"}'''} # Correct one
            ),
            Reflection(
                message_id='r2', conversation_id='conv1', agent_id='agent2', 
                content_type='tool_call', content='doc2', timestamp=current_time, 
                metadata={'event': 'tool_call', 'agent_id': 'agent2', 'extra': 'not json'} # Invalid JSON
            ),
            Reflection(
                message_id='r3', conversation_id='conv1', agent_id='agent1', 
                content_type='thought', content='doc3', timestamp=current_time,
                metadata={'event': 'thought', 'agent_id': 'agent1', 'extra': None} # Extra is None
            ),
             Reflection(
                message_id='r4', conversation_id='conv1', agent_id='agent1',
                content_type='thought', content='doc3', timestamp=current_time,
                metadata=None # Metadata is None
            ),
        ]
        mock_store_instance.query.return_value = mock_query_results # Mock store.query()
        # Act
        found_reflections = get_reflections_by_correlation_id(mock_store_instance, target_correlation_id)
        # Assert
        mock_store_instance.query.assert_called_once_with(limit=2000)
        self.assertEqual(len(found_reflections), 1)
        self.assertEqual(found_reflections[0].message_id, 'r1')

if __name__ == '__main__':
    unittest.main() 