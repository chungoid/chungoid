import unittest
from unittest.mock import patch, MagicMock, mock_open, PropertyMock
from pathlib import Path
import json
import os
import sys
import logging
import chromadb
import filelock
import datetime
import pytest
import shutil

# Add project root to path to allow importing utils
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from chungoid.utils.state_manager import StateManager, StatusFileError
from chungoid.utils.exceptions import ChromaOperationError # Import correct error
from chungoid.schemas.common_enums import StageStatus # Added import
from chungoid.schemas.errors import AgentErrorDetails # Added import
from chungoid.schemas.flows import PausedRunDetails # <<< Added import

# Define constants for test data
INITIAL_STATUS_CONTENT = '{"runs": []}'
STATUS_CONTENT_RUN0_DONE = '{"current_stage": 1.0, "runs": [{"run_id": "run_0", "status_updates": [{"timestamp": "2023-01-01T10:00:00Z", "stage": 0.0, "status": "DONE", "artifacts": ["a.txt"]}]}]}'
STATUS_CONTENT_RUN0_FAIL = '{"current_stage": 0.0, "runs": [{"run_id": "run_0", "status_updates": [{"timestamp": "2023-01-01T10:00:00Z", "stage": 0.0, "status": "FAIL", "artifacts": []}]}]}'
STATUS_CONTENT_INVALID_JSON = '{"runs": [}'

# Mark the entire file as legacy until StateManager refactor is complete
pytestmark = pytest.mark.legacy

@pytest.mark.legacy
@patch('pathlib.Path.exists') 
@patch('pathlib.Path.open', new_callable=mock_open)
class TestStateManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_target_dir = Path("./test_sm_target").resolve()
        cls.dummy_stages_dir = Path("./dummy_stages_for_test").resolve()
        cls.mock_status_path = cls.test_target_dir / ".chungoid" / "project_status.json"
        cls.test_target_dir_str = str(cls.test_target_dir)
        cls.dummy_stages_dir_str = str(cls.dummy_stages_dir)

        # Ensure dummy dirs exist for tests that might need them
        cls.test_target_dir.mkdir(parents=True, exist_ok=True)
        (cls.test_target_dir / ".chungoid").mkdir(exist_ok=True)
        cls.dummy_stages_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        # Clean up created directories
        if cls.test_target_dir.exists():
            shutil.rmtree(cls.test_target_dir)
        if cls.dummy_stages_dir.exists():
            shutil.rmtree(cls.dummy_stages_dir)
            
    @patch('filelock.FileLock')
    @patch.object(Path, 'mkdir')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'exists', return_value=False) # Assume file does NOT exist
    @patch.object(Path, 'is_dir') # Changed: No return_value here
    def test_init_success_file_does_not_exist(self, mock_is_dir, mock_method_exists, mock_builtin_open, mock_mkdir, mock_lock, mock_class_path_open, mock_class_path_exists): # Added mock_class_path_open, mock_class_path_exists; renamed mock_exists
        """Test successful init when status file doesn't exist (returns default)."""
        # Mocks applied via decorators
        mock_is_dir.side_effect = [True, True] # Explicitly for target_dir.is_dir() and server_stages_dir.is_dir()
        
        sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
        self.assertIsNotNone(sm)
        status = sm._read_status_file()
        self.assertEqual(status, {"runs": []})
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch.object(Path, 'is_dir', return_value=False)
    def test_init_target_dir_does_not_exist(self, mock_is_dir, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test init raises ValueError if target directory doesn't exist."""
        with self.assertRaisesRegex(ValueError, "Target project directory not found"):
           StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)

    @patch.object(Path, 'is_dir') # Patch Path.is_dir directly
    @patch('os.mkdir') # Patch os.mkdir to prevent FileExistsError
    def test_init_stages_dir_does_not_exist(self, mock_os_mkdir, mock_path_is_dir_method, mock_class_path_open, mock_class_path_exists): # Added class mock args, added mock_os_mkdir
        """Test init logs warning if stages directory doesn't exist but continues."""
        
        # StateManager checks target_dir.is_dir() then server_stages_dir.is_dir()
        # We want the first to be True, the second False for this test.
        mock_path_is_dir_method.side_effect = [True, False]
        
        # StateManager should log a warning but NOT raise an error
        with patch('logging.getLogger') as mock_get_logger, \
             patch.object(StateManager, '_read_status_file', return_value={"runs": []}) as mock_read_status: # Mock _read_status_file
            mock_logger_instance = MagicMock()
            mock_get_logger.return_value = mock_logger_instance
            StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
        
        # Check is_dir was called twice (for target and stages dir)
        self.assertEqual(mock_path_is_dir_method.call_count, 2)
        # Assert the warning was logged
        mock_logger_instance.warning.assert_any_call(
            f"Server stages directory not found or not a directory: {self.dummy_stages_dir_str}. Operations requiring it may fail."
        )
        # Ensure os.mkdir was NOT called in this specific error path (it's called *after* the check)
        # Correction: mkdir IS called later for .chungoid dir, so we patch it but don't assert it wasn't called.
        # mock_os_mkdir.assert_not_called()

    @patch('filelock.FileLock')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open, read_data=STATUS_CONTENT_INVALID_JSON)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_init_status_file_corrupted(self, mock_is_dir, mock_open_func, mock_makedirs, mock_lock, mock_class_path_open, mock_class_path_exists): # Removed mock_method_exists
        """Test _read_status_file raises StatusFileError if status file is invalid JSON."""
        # Mock the CLASS level Path.exists to return True for the status file check inside _read_status_file
        mock_class_path_exists.return_value = True
        
        # Mock Path.open used within _read_status_file to return corrupted data
        # Use the class-level mock (mock_class_path_open) passed as argument
        mock_class_path_open.return_value.__enter__.return_value.read.return_value = STATUS_CONTENT_INVALID_JSON
        
        with self.assertRaisesRegex(StatusFileError, "Invalid JSON in status file"):
             # StateManager init calls _read_status_file, which should now raise the error
             StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
        
        # Ensure the class-level exists mock was called
        mock_class_path_exists.assert_called()
        # Ensure Path.open was called (using the class-level mock)
        mock_class_path_open.assert_called()

    @patch('filelock.FileLock')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('chungoid.utils.state_manager.Path.exists') # Add patch for Path.exists
    @patch.object(Path, 'is_dir', return_value=True)
    @patch('chungoid.utils.state_manager.datetime') # Patch datetime where it is used
    @patch('os.fsync') 
    def test_update_project_status(self, mock_fsync, mock_datetime_class, mock_is_dir, mock_method_path_exists, mock_builtin_open_func, mock_json_dump, mock_lock, mock_class_path_open, mock_class_path_exists): # Added class mock args, renamed mocks
        """Test update_project_status correctly adds a new status entry and writes file."""
        fixed_timestamp_dt = datetime.datetime(2023, 2, 15, 12, 0, 0, tzinfo=datetime.timezone.utc)
        
        # Configure the mocked datetime class
        mock_datetime_class.now.return_value = fixed_timestamp_dt
        mock_datetime_class.timezone.utc = datetime.timezone.utc # Ensure utc is available
    
        # Mock Path.exists: False during init read, True during update read?
        # Let's make it simpler: assume init read finds nothing, update read finds nothing (first run)
        mock_method_path_exists.side_effect = lambda: False # No args, return False

        # Configure the class-level mock for Path.open to handle reads/writes
        # Read returns empty initially, Write captures data
        read_mock = mock_open(read_data='').return_value
        write_mock = mock_open().return_value
        def open_side_effect(*args, **kwargs):
            if args[0] == 'r':
                return read_mock
            elif args[0] == 'w':
                return write_mock
            return mock_open().return_value # Default mock
        mock_class_path_open.side_effect = open_side_effect

        sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)

        # Call the method which uses Path.open('w') internally via _write_status_file
        success = sm.update_status(stage=0.0, status=StageStatus.SUCCESS.value, artifacts=["out.log"])
        self.assertTrue(success)
        
        # Assert datetime.datetime.now was called with the correct timezone argument
        mock_datetime_class.now.assert_any_call(datetime.timezone.utc)

        # Expected data should reflect creating the first run (run_id 0)
        expected_data = {"runs": [{"run_id": 0, "start_timestamp": fixed_timestamp_dt.isoformat(), "status_updates": []}]}
        run_entry = expected_data["runs"][0]
        
        run_entry["status_updates"].append({
            "stage": 0.0,
            "status": StageStatus.SUCCESS.value,
            "timestamp": fixed_timestamp_dt.isoformat(), 
            "artifacts": ["out.log"],
        })
        # Assert json.dump called with the correct data and the write mock file handle
        mock_json_dump.assert_called_once_with(expected_data, write_mock, indent=2)
        mock_lock.return_value.__enter__.assert_called()
        mock_lock.return_value.__exit__.assert_called()
        mock_fsync.assert_called_once()

    @patch('filelock.FileLock')
    @patch('json.dump')
    @patch('builtins.open', new_callable=mock_open)
    @patch('chungoid.utils.state_manager.datetime')
    @patch('os.fsync')
    def test_update_project_status_fail_with_details(self, mock_fsync, mock_datetime_class, mock_builtin_open_func, mock_json_dump, mock_lock, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test update_status stores AgentErrorDetails on FAIL."""
        fixed_timestamp_dt = datetime.datetime(2023, 2, 15, 13, 0, 0, tzinfo=datetime.timezone.utc)
        mock_datetime_class.now.return_value = fixed_timestamp_dt
        mock_datetime_class.timezone.utc = datetime.timezone.utc

        error_instance = AgentErrorDetails(
            error_type="ValueError",
            message="Something went wrong",
            traceback="Traceback...\\n...",
            agent_id="test_agent",
            stage_id="1.5"
        )
        # Expect the JSON string representation in the final data, with indentation
        error_json_str = error_instance.model_dump_json(indent=2) 

        # Assume status file already has run 0
        existing_status_content = '{"runs": [{"run_id": 0, "start_timestamp": "2023-02-15T12:00:00+00:00", "status_updates": []}]}'
        
        # Mock Path.exists: True for the read inside update_status
        # Use mock_class_path_exists from class decorator here
        mock_class_path_exists.return_value = True
        
        # Configure the class-level mock for Path.open to handle reads/writes
        read_mock = mock_open(read_data=existing_status_content).return_value
        write_mock = mock_open().return_value
        def open_side_effect(*args, **kwargs):
            if args[0] == 'r':
                return read_mock
            elif args[0] == 'w':
                return write_mock
            return mock_open().return_value # Default mock
        mock_class_path_open.side_effect = open_side_effect

        # Patch is_dir needed for StateManager init
        with patch('chungoid.utils.state_manager.Path.is_dir', return_value=True):
            sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)

            # Ensure that when update_status calls _read_status_file, it gets the existing content
            with patch.object(sm, '_read_status_file', return_value=json.loads(existing_status_content)):
                success = sm.update_status(
                    stage=1.5,
                    status=StageStatus.FAILURE.value, 
                    artifacts=[],
                    reason="Agent failed",
                    error_details=error_instance
                )
            self.assertTrue(success)

            # Prepare expected data for json.dump assertion
            expected_data = json.loads(existing_status_content) # This will have the 12:00 timestamp
            expected_entry = {
                "stage": 1.5,
                "status": StageStatus.FAILURE.value,
                "timestamp": fixed_timestamp_dt.isoformat(), 
                "artifacts": [],
                "reason": "Agent failed",
                "error_details": error_json_str # Compare string with string
            }
            # The update should append to the *existing* run 0
            expected_data["runs"][0]["status_updates"].append(expected_entry)

            # Assert json.dump was called with the correct structure and the mocked Path.open file handle for writing
            mock_json_dump.assert_called_once_with(expected_data, write_mock, indent=2)

            mock_datetime_class.now.assert_any_call(datetime.timezone.utc)
            mock_fsync.assert_called_once()
            mock_lock.return_value.__enter__.assert_called()
            mock_lock.return_value.__exit__.assert_called()

    @patch('filelock.FileLock')
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_next_stage_simple(self, mock_is_dir, mock_lock, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test get_next_stage advances correctly."""
        sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)

        # Patch _read_status_file to return the desired data *when get_next_stage calls it*
        parsed_data = json.loads(STATUS_CONTENT_RUN0_DONE)
        
        # Mock the glob call to return dummy stage files
        mock_stage0 = MagicMock(spec=Path)
        mock_stage0.is_file.return_value = True
        mock_stage0.stem = "stage0"
        mock_stage1 = MagicMock(spec=Path)
        mock_stage1.is_file.return_value = True
        mock_stage1.stem = "stage1"
        
        # Patch pathlib.Path.glob directly
        with patch.object(sm, '_read_status_file', return_value=parsed_data), \
             patch('pathlib.Path.glob', return_value=[mock_stage0, mock_stage1]) as mock_glob: # Patch pathlib.Path.glob
            
            next_stage = sm.get_next_stage()
            # Highest completed = 0.0 (from RUN0_DONE). Available = [0.0, 1.0]. Lowest > 0.0 is 1.0
            self.assertEqual(next_stage, 1.0)
            mock_glob.assert_called_once_with("stage*.yaml")

    @patch('filelock.FileLock')
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_next_stage_fail(self, mock_is_dir, mock_lock, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test get_next_stage returns correct stage based on FAIL content."""
        sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
        
        parsed_data = json.loads(STATUS_CONTENT_RUN0_FAIL)

        # Mock the glob call to return dummy stage files
        mock_stage0 = MagicMock(spec=Path)
        mock_stage0.is_file.return_value = True
        mock_stage0.stem = "stage0"
        mock_stage1_5 = MagicMock(spec=Path)
        mock_stage1_5.is_file.return_value = True
        mock_stage1_5.stem = "stage1.5"

        with patch.object(sm, '_read_status_file', return_value=parsed_data), \
             patch('pathlib.Path.glob', return_value=[mock_stage0, mock_stage1_5]) as mock_glob: # Patch pathlib.Path.glob
            
            # Logic changed: get_next_stage finds lowest available > highest completed.
            # In RUN0_FAIL, highest completed is -1. Available = [0.0, 1.5]. Lowest > -1 is 0.0
            next_stage = sm.get_next_stage() 
            self.assertEqual(next_stage, 0.0) # Expect 0.0 as next stage
            mock_glob.assert_called_once_with("stage*.yaml")

    @patch.object(Path, 'is_dir', return_value=True)
    @patch('chungoid.utils.state_manager.chroma_utils.query_documents')
    def test_get_reflection_context_from_chroma_success(self, mock_query_documents, mock_isdir, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test get_reflection_context_from_chroma queries and formats."""
        # Updated mock_query_results to match the expected output format of chroma_utils.query_documents
        mock_formatted_results = [{
            'id': 'id1',
            'document': 'doc1',
            'metadata': {'stage_number': 0.0, 'status': 'DONE'},
            'distance': 0.1,
            'embedding': None  # Assuming embedding is not strictly checked here or is None
        }]
        mock_query_documents.return_value = mock_formatted_results
        
        # Ensure filelock is mocked for StateManager init
        with patch('filelock.FileLock'):
            sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
            results = sm.get_reflection_context_from_chroma(query="test query")
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["document"], "doc1")

        # Assert that chroma_utils.query_documents was called correctly
        mock_query_documents.assert_called_once_with(
            collection_name=sm._REFLECTIONS_COLLECTION_NAME,
            query_texts=["test query"],
            n_results=3, # Default n_results in get_reflection_context_from_chroma
            where_filter=None, # Default filter
            include=['metadatas', 'documents', 'distances']
        )

    @patch.object(Path, 'is_dir', return_value=True)
    @patch('chungoid.utils.state_manager.chroma_utils.add_or_update_document') # Corrected patch target
    def test_store_artifact_context_in_chroma_success(self, mock_add_or_update_document, mock_isdir, mock_class_path_open, mock_class_path_exists): # Added class mock args
        """Test store_artifact_context_in_chroma adds artifact data."""
        mock_add_or_update_document.return_value = True # Simulate successful operation

        # Ensure filelock is mocked for StateManager init
        with patch('filelock.FileLock'):
            sm = StateManager(self.test_target_dir_str, server_stages_dir=self.dummy_stages_dir_str)
            success = sm.store_artifact_context_in_chroma(stage_number=1.0, rel_path="a/b.txt", content="artifact content")
        
        self.assertTrue(success)
        # Assert that chroma_utils.add_or_update_document was called
        mock_add_or_update_document.assert_called_once()
        
        args, kwargs = mock_add_or_update_document.call_args
        self.assertEqual(kwargs['collection_name'], sm._CONTEXT_COLLECTION_NAME)
        self.assertIn('doc_id', kwargs) # add_or_update_document takes doc_id
        self.assertEqual(kwargs['document_content'], "artifact content")
        self.assertIn('metadata', kwargs)
        self.assertEqual(kwargs['metadata']['stage_number'], 1.0)
        self.assertEqual(kwargs['metadata']['relative_path'], "a/b.txt")

# Add more tests as needed

# --- Pause/Resume State Tests --- #

class TestStateManagerPauseResume(unittest.TestCase):
    # Use setUp to create mocks needed specifically for these tests
    def setUp(self):
        self.test_target_dir = Path("./test_sm_target_pause_resume").resolve()
        self.chungoid_dir = self.test_target_dir / ".chungoid"
        self.paused_runs_dir = self.chungoid_dir / "paused_runs"
        self.status_file_path_obj = self.chungoid_dir / "project_status.json"
        self.test_run_id = "paused-run-123"
        self.paused_file_path_obj = self.paused_runs_dir / f"{self.test_run_id}.json"
        self.dummy_stages_dir = self.test_target_dir / "dummy_stages"
        if not self.dummy_stages_dir.exists(): # Create if not exists, for local test runs
            self.dummy_stages_dir.mkdir(parents=True, exist_ok=True)

        # Patch methods that interfere with isolated testing
        self.patch_read = patch.object(StateManager, '_read_status_file', return_value={"runs": []})
        self.patch_write = patch.object(StateManager, '_write_status_file')
        self.patch_logger_factory = patch('logging.getLogger') 
        
        # Patch external dependencies directly to prevent real calls during StateManager init
        self.patch_persistent_client = patch('chromadb.PersistentClient', return_value=MagicMock()) 
        self.patch_filelock_class = patch('filelock.FileLock', return_value=MagicMock())      

        self.mock_read = self.patch_read.start()
        self.mock_write = self.patch_write.start()
        self.mock_get_logger = self.patch_logger_factory.start()
        self.mock_logger = MagicMock() # Create a mock logger instance
        self.mock_get_logger.return_value = self.mock_logger # Make getLogger return our mock
        
        self.mock_persistent_client = self.patch_persistent_client.start()
        self.mock_filelock_class = self.patch_filelock_class.start()
        
        # StateManager will be initialized with use_locking=True by default, 
        # but our patch to filelock.FileLock will ensure it gets a mock.
        # Chroma client will also be a mock.
        self.sm_for_pause_tests = StateManager(str(self.test_target_dir), server_stages_dir=str(self.dummy_stages_dir))
        self.sm_for_pause_tests.status_file_path = self.status_file_path_obj # Ensure it uses our Path obj

        # <<< ADDED: Create mocks needed for new strategy >>>
        self.mock_status_parent_dir = MagicMock(spec=Path, name="MockStatusParentDir")
        self.mock_paused_runs_dir = MagicMock(spec=Path, name="MockPausedRunsDir")
        self.mock_paused_file_path = MagicMock(spec=Path, name="MockPausedFilePath")
        # Chain the mocks: parent / "paused_runs" -> paused_dir / "filename.json" -> file_path
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path
        # <<< END ADDED >>>

    def tearDown(self):
        self.patch_read.stop()
        self.patch_write.stop()
        self.patch_logger_factory.stop() 
        self.patch_persistent_client.stop()
        self.patch_filelock_class.stop()
        
        # Clean up test directory 
        if self.test_target_dir.exists():
            shutil.rmtree(self.test_target_dir)
        # dummy_stages_dir is inside test_target_dir, so it's removed with it.

    def create_paused_details(self) -> PausedRunDetails:
        "Helper to create a sample PausedRunDetails object."
        return PausedRunDetails(
            run_id=self.test_run_id,
            flow_id="test-flow-id", # Added missing flow_id
            paused_at_stage_id="stage_error",
            timestamp=datetime.datetime(2024, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc), # Fixed timestamp
            context_snapshot={"key": "value", "outputs": {}},
            error_details=AgentErrorDetails(
                error_type="ValueError",
                message="Agent failed",
                traceback="Traceback...",
                agent_id="agent_x",
                stage_id="stage_error"
            ),
            reason="Agent Error"
        )

    def test_save_paused_flow_state_success(self):
        """Test successfully saving paused flow state (isolated)."""
        paused_details = self.create_paused_details()
        expected_json = paused_details.model_dump_json(indent=2)
        
        # Reset mocks for this test (needed if __truediv__ is called multiple times across tests)
        self.mock_status_parent_dir.reset_mock()
        self.mock_paused_runs_dir.reset_mock()
        self.mock_paused_file_path.reset_mock()
        # Re-establish mock chaining needed for this test
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path

        m_open = mock_open()
        
        # Mock the `parent` property to return our mock parent directory
        prop_mock = PropertyMock(return_value=self.mock_status_parent_dir)
        with patch.object(Path, 'parent', new_callable=PropertyMock, return_value=self.mock_status_parent_dir) as mock_parent_prop, \
             patch.object(self.mock_paused_file_path, 'open', m_open): # Patch open on the final mock file path
            
            # Ensure the patch applies specifically to the status file path object instance?
            # No, patching Path.parent globally is simpler for the context manager duration.
            # StateManager will call `self.status_file_path.parent` which hits the patch.
            success = self.sm_for_pause_tests.save_paused_flow_state(paused_details)
        
        self.assertTrue(success)
        
        # Assert that status_file_path.parent was accessed
        # We can't directly assert on the global patch easily, 
        # but we check the subsequent calls which prove parent was mocked.
        
        # Assert that the first division (mock_status_parent_dir / "paused_runs") was called
        self.mock_status_parent_dir.__truediv__.assert_called_once_with("paused_runs")
        
        # Assert mkdir was called on the result of the first division (mock_paused_runs_dir)
        self.mock_paused_runs_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)

        # Assert that the second division (mock_paused_runs_dir / f"{paused_details.run_id}.json") was called
        self.mock_paused_runs_dir.__truediv__.assert_called_once_with(f"{paused_details.run_id}.json")
        
        # Assert open was called on the result of the second division (mock_paused_file_path)
        m_open.assert_called_once_with("w", encoding="utf-8")
        m_open().write.assert_called_once_with(expected_json)
        self.mock_logger.info.assert_any_call(f"Saving paused flow state for run_id '{self.test_run_id}' to {self.mock_paused_file_path}")

    def test_load_paused_flow_state_success(self):
        """Test successfully loading paused flow state (isolated)."""
        paused_details_orig = self.create_paused_details()
        paused_json_content = paused_details_orig.model_dump_json()

        # Reset mocks
        self.mock_status_parent_dir.reset_mock()
        self.mock_paused_runs_dir.reset_mock()
        self.mock_paused_file_path.reset_mock()
        # Re-establish mock chaining needed for this test
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path
        
        # Configure final mock file path behavior
        self.mock_paused_file_path.exists.return_value = True
        self.mock_paused_file_path.is_file.return_value = True
        self.mock_paused_file_path.read_text.return_value = paused_json_content
        
        # Mock the `parent` property 
        with patch.object(Path, 'parent', new_callable=PropertyMock, return_value=self.mock_status_parent_dir):
            loaded_details = self.sm_for_pause_tests.load_paused_flow_state(self.test_run_id)
        
        self.assertIsNotNone(loaded_details)
        self.assertEqual(loaded_details, paused_details_orig)
        
        # Assert path generation calls
        self.mock_status_parent_dir.__truediv__.assert_called_once_with("paused_runs")
        self.mock_paused_runs_dir.__truediv__.assert_called_once_with(f"{self.test_run_id}.json")

        # Assert checks on the final mock path
        self.mock_paused_file_path.exists.assert_called_once()
        self.mock_paused_file_path.is_file.assert_called_once()
        self.mock_paused_file_path.read_text.assert_called_once()
        self.mock_logger.info.assert_any_call(f"Loading paused flow state for run_id '{self.test_run_id}' from {self.mock_paused_file_path}")

    def test_load_paused_flow_state_not_found(self):
        """Test loading returns None when file doesn't exist (isolated)."""
        # Reset mocks
        self.mock_status_parent_dir.reset_mock()
        self.mock_paused_runs_dir.reset_mock()
        self.mock_paused_file_path.reset_mock()
        # Re-establish mock chaining needed for this test
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path
        
        # Configure final mock file path behavior
        self.mock_paused_file_path.exists.return_value = False # File does not exist
        
        # Mock the `parent` property
        with patch.object(Path, 'parent', new_callable=PropertyMock, return_value=self.mock_status_parent_dir):
            loaded_details = self.sm_for_pause_tests.load_paused_flow_state(self.test_run_id)
        
        self.assertIsNone(loaded_details)
        
        # Assert path generation calls
        self.mock_status_parent_dir.__truediv__.assert_called_once_with("paused_runs")
        self.mock_paused_runs_dir.__truediv__.assert_called_once_with(f"{self.test_run_id}.json")

        # Assert checks and calls on the final mock path
        # Ensure exists() is checked correctly (called once in this path for load)
        self.assertEqual(self.mock_paused_file_path.exists.call_count, 1)
        self.mock_paused_file_path.is_file.assert_not_called() 
        self.mock_paused_file_path.unlink.assert_not_called() 
        # Updated expected log message to match actual implementation
        self.mock_logger.info.assert_any_call(f"Paused state file for run_id '{self.test_run_id}' not found at {self.mock_paused_file_path}.")

    def test_delete_paused_flow_state_success(self):
        """Test successfully deleting an existing paused state file (isolated)."""
        # Reset mocks
        self.mock_status_parent_dir.reset_mock()
        self.mock_paused_runs_dir.reset_mock()
        self.mock_paused_file_path.reset_mock()
        # Re-establish mock chaining needed for this test
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path
        
        # Configure final mock file path behavior
        self.mock_paused_file_path.exists.return_value = True
        self.mock_paused_file_path.is_file.return_value = True
        
        # Mock the `parent` property
        with patch.object(Path, 'parent', new_callable=PropertyMock, return_value=self.mock_status_parent_dir):
            success = self.sm_for_pause_tests.delete_paused_flow_state(self.test_run_id)
        
        self.assertTrue(success)
        
        # Assert path generation calls
        self.mock_status_parent_dir.__truediv__.assert_called_once_with("paused_runs")
        self.mock_paused_runs_dir.__truediv__.assert_called_once_with(f"{self.test_run_id}.json")
        
        # Assert checks and calls on the final mock path
        self.mock_paused_file_path.exists.assert_called_once()
        self.mock_paused_file_path.is_file.assert_called_once()
        self.mock_paused_file_path.unlink.assert_called_once()
        self.mock_logger.info.assert_any_call(f"Deleting paused flow state file for run_id '{self.test_run_id}' at {self.mock_paused_file_path}")

    def test_delete_paused_flow_state_not_found(self):
        """Test deleting returns True when the file doesn't exist (isolated)."""
        # Reset mocks
        self.mock_status_parent_dir.reset_mock()
        self.mock_paused_runs_dir.reset_mock()
        self.mock_paused_file_path.reset_mock()
        # Re-establish mock chaining needed for this test
        self.mock_status_parent_dir.__truediv__.return_value = self.mock_paused_runs_dir
        self.mock_paused_runs_dir.__truediv__.return_value = self.mock_paused_file_path
        
        # Configure final mock file path behavior
        self.mock_paused_file_path.exists.return_value = False # File does not exist
        
        # Mock the `parent` property
        with patch.object(Path, 'parent', new_callable=PropertyMock, return_value=self.mock_status_parent_dir):
            success = self.sm_for_pause_tests.delete_paused_flow_state(self.test_run_id)

        self.assertTrue(success)
        
        # Assert path generation calls
        self.mock_status_parent_dir.__truediv__.assert_called_once_with("paused_runs")
        self.mock_paused_runs_dir.__truediv__.assert_called_once_with(f"{self.test_run_id}.json")
        
        # Assert checks and calls on the final mock path
        # Ensure exists() is checked correctly (called twice in this path)
        self.assertEqual(self.mock_paused_file_path.exists.call_count, 2)
        self.mock_paused_file_path.is_file.assert_not_called() 
        self.mock_paused_file_path.unlink.assert_not_called() 
        # Updated expected log message to match actual implementation
        self.mock_logger.info.assert_any_call(f"Paused state file for run_id '{self.test_run_id}' not found at {self.mock_paused_file_path}. No deletion needed.")

# <<< Add tests for get_or_create_current_run_id >>>
class TestStateManagerGetRunId(unittest.TestCase):

    def setUp(self):
        self.test_target_dir = Path("./test_sm_target_getid").resolve()
        self.dummy_stages_dir = self.test_target_dir / "dummy_stages"
        if not self.dummy_stages_dir.exists():
            self.dummy_stages_dir.mkdir(parents=True, exist_ok=True)
        
        # Patch dependencies needed for StateManager init
        self.patch_persistent_client = patch('chromadb.PersistentClient', return_value=MagicMock())
        self.patch_filelock_class = patch('filelock.FileLock', return_value=MagicMock())
        self.patch_logger_factory = patch('logging.getLogger') 

        self.mock_persistent_client = self.patch_persistent_client.start()
        self.mock_filelock_class = self.patch_filelock_class.start()
        self.mock_get_logger = self.patch_logger_factory.start()
        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger
        
        # Need to mock _read_status_file for StateManager init, can reuse later
        self.patch_read = patch.object(StateManager, '_read_status_file')
        self.mock_read = self.patch_read.start()
        self.mock_read.return_value = {"runs": []} # Default for init
        
        self.sm = StateManager(str(self.test_target_dir), server_stages_dir=str(self.dummy_stages_dir))

    def tearDown(self):
        self.patch_read.stop()
        self.patch_logger_factory.stop()
        self.patch_persistent_client.stop()
        self.patch_filelock_class.stop()
        if self.test_target_dir.exists():
            shutil.rmtree(self.test_target_dir)

    def test_get_run_id_no_runs(self):
        """Test returns 0 if status file has no runs."""
        self.mock_read.return_value = {"runs": []}
        # Reset call count after init
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertEqual(run_id, 0)
        # Fix: Check only the call made by the method itself
        # self.assertEqual(self.mock_read.call_count, 3)
        self.mock_read.assert_called_once()

    def test_get_run_id_existing_runs(self):
        """Test returns the highest run_id from existing runs."""
        status_data = {
            "runs": [
                {"run_id": 0, "status_updates": []},
                {"run_id": 2, "status_updates": []}, # Intentionally out of order
                {"run_id": 1, "status_updates": []}
            ]
        }
        self.mock_read.return_value = status_data
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertEqual(run_id, 2)
        self.mock_read.assert_called_once()

    def test_get_run_id_single_run(self):
        """Test returns correct run_id with only one run."""
        status_data = {"runs": [{"run_id": 5, "status_updates": []}]}
        self.mock_read.return_value = status_data
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertEqual(run_id, 5)
        self.mock_read.assert_called_once()

    def test_get_run_id_ignores_invalid_entries(self):
        """Test returns highest valid run_id, ignoring invalid entries."""
        # Case 1: Valid run_id exists alongside invalid entries
        status_data_mixed = {"runs": [{"run_id": 0}, {"no_run_id": True}, {"run_id": "abc"}, {"run_id": 1}]}
        self.mock_read.return_value = status_data_mixed
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertEqual(run_id, 1, "Failed when valid run_id mixed with invalid")
        self.mock_read.assert_called_once()

    def test_get_run_id_returns_none_if_only_invalid(self):
        """Test returns None if only invalid run_ids are found."""
        # Case 1: Only invalid keys / types
        status_data_all_invalid_keys = {"runs": [{"no_run_id": True}, {"other_key": 1}]}
        self.mock_read.return_value = status_data_all_invalid_keys
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertIsNone(run_id, "Failed when only invalid keys present")
        self.mock_read.assert_called_once()

        # Case 2: Highest run_id has invalid type
        status_data_highest_invalid_type = {"runs": [{"run_id": 0}, {"run_id": "abc"}]}
        self.mock_read.return_value = status_data_highest_invalid_type
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        # Max key will find {"run_id": 0} because -1 (for "abc") < 0.
        # So, the function should return 0 here, not None.
        # Let's rethink this case. If the max object found has an invalid run_id, it should error.
        # If max finds {"run_id": 0} because "abc" yields -1, it returns 0. Seems correct.
        # What if max finds {"run_id": "abc"} because it was the only one? e.g. {"runs": [{"run_id": "abc"}]} -> max key -1 -> returns None.
        # What if max finds {"run_id": -5}? e.g. {"runs": [{"run_id": -5}]} -> max key -1 -> returns None.
        # The logic seems to return None if the found max run_id fails validation (None, not int, < 0).
        
        # Let's test the None path more directly:
        status_data_highest_is_negative = {"runs": [{"run_id": -5}]}
        self.mock_read.return_value = status_data_highest_is_negative
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertIsNone(run_id, "Failed when highest run_id is negative")
        self.mock_read.assert_called_once()

        status_data_highest_is_none = {"runs": [{"run_id": None}]}
        self.mock_read.return_value = status_data_highest_is_none
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertIsNone(run_id, "Failed when highest run_id is None")
        self.mock_read.assert_called_once()

        status_data_highest_is_wrong_type = {"runs": [{"run_id": "abc"}]}
        self.mock_read.return_value = status_data_highest_is_wrong_type
        self.mock_read.reset_mock()
        run_id = self.sm.get_or_create_current_run_id()
        self.assertIsNone(run_id, "Failed when highest run_id is wrong type")
        self.mock_read.assert_called_once()

    def test_get_run_id_status_file_error(self):
        """Test returns None if reading status file fails."""
        self.mock_read.side_effect = StatusFileError("Read failed")
        run_id = self.sm.get_or_create_current_run_id()
        self.assertIsNone(run_id)
        self.mock_logger.error.assert_any_call("Failed to get current run_id due to StatusFileError: Read failed")

if __name__ == "__main__":
    unittest.main()
