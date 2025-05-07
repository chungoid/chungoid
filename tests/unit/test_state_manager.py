import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import json
import os
import sys
import logging
import chromadb
import filelock
import datetime
import pytest

# Add project root to path to allow importing utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chungoid.utils.state_manager import StateManager, StatusFileError
from chungoid.utils.exceptions import ChromaOperationError # Import correct error

# Define constants for test data
INITIAL_STATUS_CONTENT = '{"runs": []}'
STATUS_CONTENT_RUN0_DONE = '{"current_stage": 1.0, "runs": [{"run_id": "run_0", "status_updates": [{"timestamp": "2023-01-01T10:00:00Z", "stage": 0.0, "status": "DONE", "artifacts": ["a.txt"]}]}]}'
STATUS_CONTENT_RUN0_FAIL = '{"current_stage": 0.0, "runs": [{"run_id": "run_0", "status_updates": [{"timestamp": "2023-01-01T10:00:00Z", "stage": 0.0, "status": "FAIL", "artifacts": []}]}]}'
STATUS_CONTENT_INVALID_JSON = '{"runs": [}'

# Mark the entire file as legacy until StateManager refactor is complete
pytestmark = pytest.mark.legacy

class TestStateManager(unittest.TestCase):
    test_target_dir = "./test_sm_target" # Use relative path for simplicity
    dummy_stages_dir_str = "./dummy_stages_for_test"
    # Derive other paths based on test_target_dir
    mock_chungoid_dir = Path(test_target_dir) / ".chungoid"
    mock_status_path = mock_chungoid_dir / "project_status.json"
    mock_lock_path = mock_status_path.with_suffix(".json.lock")

    @patch('chungoid.utils.chroma_utils.get_chroma_client')
    def setUp(self, mock_get_chroma):
        self.mock_chroma_client = MagicMock(spec=chromadb.ClientAPI)
        self.mock_collection = MagicMock(spec=chromadb.Collection)
        self.mock_chroma_client.get_or_create_collection.return_value = self.mock_collection
        mock_get_chroma.return_value = self.mock_chroma_client
        # No dummy dir creation needed anymore

    def tearDown(self):
        pass # No filesystem changes to clean up

    # --- Helper REMOVED ---

    # --- Initialization Tests ---
    @patch('filelock.FileLock')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open, read_data=STATUS_CONTENT_RUN0_DONE)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_init_success_file_exists(self, mock_is_dir, mock_exists, mock_open_func, mock_makedirs, mock_lock):
        """Test successful init when status file exists."""
        mock_exists.return_value = True # Status file exists
        # Mock is_dir for both target_dir and stages_dir
        def is_dir_side_effect(path_instance):
            if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()):
                return True
            if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()):
                return True # Mock stages dir as existing
            return False
        mock_is_dir.side_effect = is_dir_side_effect

        with patch('json.load', return_value=json.loads(STATUS_CONTENT_RUN0_DONE)):
            sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
            self.assertIsNotNone(sm)
            mock_open_func.assert_called_with(str(self.mock_status_path), 'r', encoding='utf-8')

    @patch('filelock.FileLock')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_init_success_file_does_not_exist(self, mock_is_dir, mock_exists, mock_open_func, mock_makedirs, mock_lock):
        """Test successful init when status file doesn't exist (returns default)."""
        mock_exists.return_value = False # Status file does NOT exist
        # Mock is_dir for both target_dir and stages_dir
        def is_dir_side_effect(path_instance):
            if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()):
                return True
            if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()):
                return True # Mock stages dir as existing
            return False
        mock_is_dir.side_effect = is_dir_side_effect

        sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
        self.assertIsNotNone(sm)
        # Check _read_status_file returns default without error
        status = sm._read_status_file()
        self.assertEqual(status, {"runs": []})
        mock_makedirs.assert_called_once_with(self.mock_chungoid_dir, parents=True, exist_ok=True)

    @patch.object(Path, 'is_dir')
    def test_init_target_dir_does_not_exist(self, mock_is_dir):
        """Test init raises ValueError if target directory doesn't exist."""
        mock_is_dir.return_value = False # Target dir does not exist
        with self.assertRaisesRegex(ValueError, "Target project directory not found"):
            StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)

    @patch.object(Path, 'is_dir')
    def test_init_stages_dir_does_not_exist(self, mock_is_dir):
        """Test init raises ValueError if stages directory doesn't exist."""
        # Mock target dir exists, but stages dir does not
        def is_dir_side_effect(path_instance):
            if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()):
                return True
            if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()):
                return False # Mock stages dir as NOT existing
            return False
        mock_is_dir.side_effect = is_dir_side_effect
        with self.assertRaisesRegex(ValueError, "Server stages directory not found"):
            StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)

    @patch('filelock.FileLock')
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open, read_data=STATUS_CONTENT_INVALID_JSON)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_init_status_file_corrupted(self, mock_is_dir, mock_exists, mock_open_func, mock_makedirs, mock_lock):
        """Test _read_status_file raises StatusFileError if status file is invalid JSON."""
        mock_exists.return_value = True
        def is_dir_side_effect(path_instance):
             if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()): return True
             if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()): return True
             return False
        mock_is_dir.side_effect = is_dir_side_effect

        with patch('json.load', side_effect=json.JSONDecodeError("err", "doc", 0)):
             sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
             with self.assertRaises(StatusFileError):
                 sm._read_status_file()

    # --- Status Update Tests ---
    @patch('filelock.FileLock')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open, read_data=INITIAL_STATUS_CONTENT)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    @patch('datetime.datetime')
    def test_update_project_status(self, mock_dt, mock_is_dir, mock_exists, mock_open_func, mock_json_dump, mock_lock):
        """Test update_project_status correctly adds a new status entry and writes file."""
        mock_exists.return_value = True
        def is_dir_side_effect(path_instance):
             if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()): return True
             if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()): return True
             return False
        mock_is_dir.side_effect = is_dir_side_effect
        with patch('json.load', return_value=json.loads(INITIAL_STATUS_CONTENT)):
            sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)

        fixed_timestamp = "2023-02-15T12:00:00Z"
        mock_dt.now.return_value.isoformat.return_value = fixed_timestamp

        success = sm.update_status(stage=0.0, status="RUNNING", artifacts=["out.log"])
        self.assertTrue(success)

        # Verify write call
        mock_open_func.assert_called_with(str(self.mock_status_path), 'w', encoding='utf-8')
        mock_json_dump.assert_called_once()
        written_data = mock_json_dump.call_args[0][0]
        self.assertEqual(len(written_data['runs']), 1)
        self.assertEqual(len(written_data['runs'][0]['status_updates']), 1)
        last_update = written_data['runs'][0]['status_updates'][0]
        self.assertEqual(last_update['timestamp'], fixed_timestamp)
        self.assertEqual(last_update['stage'], 0.0)
        self.assertEqual(last_update['status'], "RUNNING")
        self.assertEqual(last_update['artifacts'], ["out.log"])

    # --- Get Next Stage Tests ---
    @patch('filelock.FileLock')
    @patch('builtins.open', new_callable=mock_open, read_data=STATUS_CONTENT_RUN0_DONE)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_get_next_stage_simple(self, mock_is_dir, mock_exists, mock_open_func, mock_lock):
        """Test get_next_stage advances correctly."""
        mock_exists.return_value = True
        def is_dir_side_effect(path_instance):
             if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()): return True
             if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()): return True
             return False
        mock_is_dir.side_effect = is_dir_side_effect
        with patch('json.load', return_value=json.loads(STATUS_CONTENT_RUN0_DONE)):
            sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)

        next_stage = sm.get_next_stage()
        self.assertEqual(next_stage, 1.0)

    @patch('filelock.FileLock')
    @patch('builtins.open', new_callable=mock_open, read_data=STATUS_CONTENT_RUN0_FAIL)
    @patch.object(Path, 'exists')
    @patch.object(Path, 'is_dir')
    def test_get_next_stage_fail(self, mock_is_dir, mock_exists, mock_open_func, mock_lock):
        """Test get_next_stage returns None if last status was FAIL."""
        mock_exists.return_value = True
        def is_dir_side_effect(path_instance):
             if str(path_instance.resolve()) == str(Path(self.test_target_dir).resolve()): return True
             if str(path_instance.resolve()) == str(Path(self.dummy_stages_dir_str).resolve()): return True
             return False
        mock_is_dir.side_effect = is_dir_side_effect
        with patch('json.load', return_value=json.loads(STATUS_CONTENT_RUN0_FAIL)):
            sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)

        next_stage = sm.get_next_stage()
        self.assertIsNone(next_stage)

    # --- ChromaDB related tests (simplified) ---
    @patch.object(Path, 'is_dir', return_value=True) # Assume dirs exist
    def test_add_reflection_success(self, mock_isdir):
        """Test add_reflection calls Chroma client add."""
        sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
        success = sm.add_reflection(stage=1.0, status="PASS", reflection_text="Test reflection")
        self.assertTrue(success)
        self.mock_chroma_client.get_or_create_collection.assert_called_once_with(name=sm._REFLECTIONS_COLLECTION_NAME)
        self.mock_collection.add.assert_called_once()
        # Add more detailed assertions on call_args if needed

    @patch.object(Path, 'is_dir', return_value=True)
    def test_add_reflection_chroma_error(self, mock_isdir):
        """Test add_reflection handles Chroma errors."""
        self.mock_collection.add.side_effect = chromadb.exceptions.ChromaError("DB fail")
        sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
        with self.assertRaises(ChromaOperationError):
            sm.add_reflection(stage=1.0, status="PASS", reflection_text="Fail reflection")

    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_reflection_context_from_chroma_success(self, mock_isdir):
        """Test get_reflection_context_from_chroma queries and formats."""
        mock_results = {
            'ids': [['id1']],
            'metadatas': [[{'stage_number': 0.0, 'status': 'DONE'}]],
            'documents': [['doc1']],
            'distances': [[0.1]]
        }
        self.mock_collection.query.return_value = mock_results
        sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
        results = sm.get_reflection_context_from_chroma(query="test query")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document"], "doc1")
        self.mock_chroma_client.get_collection.assert_called_once_with(name=sm._REFLECTIONS_COLLECTION_NAME)
        self.mock_collection.query.assert_called_once()
        # Check query args
        call_args, call_kwargs = self.mock_collection.query.call_args
        self.assertEqual(call_kwargs.get("query_texts"), ["test query"])
        self.assertEqual(call_kwargs.get("n_results"), 3) # Default n_results

    @patch.object(Path, 'is_dir', return_value=True)
    def test_store_artifact_context_in_chroma_success(self, mock_isdir):
        """Test store_artifact_context_in_chroma adds artifact data."""
        sm = StateManager(self.test_target_dir, server_stages_dir=self.dummy_stages_dir_str)
        success = sm.store_artifact_context_in_chroma(stage_number=1.0, rel_path="a/b.txt", content="artifact content")
        self.assertTrue(success)
        self.mock_chroma_client.get_or_create_collection.assert_called_once_with(name=sm._CONTEXT_COLLECTION_NAME)
        self.mock_collection.add.assert_called_once()
        # Add more detailed assertions on call_args if needed

# Add more tests as needed: e.g. get_all_reflections, error cases for chroma methods

if __name__ == "__main__":
    unittest.main()
