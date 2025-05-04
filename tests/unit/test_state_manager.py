import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

# Add project root to path to allow importing utils
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.state_manager import StateManager, StatusFileError


class TestStateManager(unittest.TestCase):
    def setUp(self):
        """Set up for test methods."""
        self.test_dir = Path("./test_sm_dir")
        self.test_dir.mkdir(exist_ok=True)
        self.server_stages_dir = Path("./test_stages")
        self.server_stages_dir.mkdir(exist_ok=True)
        # Create dummy stage files
        (self.server_stages_dir / "stage0.yaml").touch()
        (self.server_stages_dir / "stage1.yaml").touch()
        self.status_file = self.test_dir / ".chungoid" / "project_status.json"
        self.lock_file = Path(f"{self.status_file}.lock")
        # Ensure .chungoid dir exists for setup
        (self.test_dir / ".chungoid").mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up after test methods."""
        if self.lock_file.exists():
            self.lock_file.unlink()
        if self.status_file.exists():
            self.status_file.unlink()
        if (self.test_dir / ".chungoid").exists():
            (self.test_dir / ".chungoid").rmdir()
        if self.test_dir.exists():
            self.test_dir.rmdir()
        if (self.server_stages_dir / "stage0.yaml").exists():
            (self.server_stages_dir / "stage0.yaml").unlink()
        if (self.server_stages_dir / "stage1.yaml").exists():
            (self.server_stages_dir / "stage1.yaml").unlink()
        if self.server_stages_dir.exists():
            self.server_stages_dir.rmdir()

    def test_init_success(self):
        """Test successful initialization."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        self.assertTrue((self.test_dir / ".chungoid").exists())
        self.assertIsNotNone(sm)
        self.assertIsNone(sm.chroma_client)  # Chroma client init deferred

    def test_init_no_stages_dir_failure(self):
        """Test initialization fails if server_stages_dir doesn't exist."""
        with self.assertRaises(ValueError):
            StateManager(str(self.test_dir), "./nonexistent_stages_dir")

    @patch("builtins.open", new_callable=mock_open, read_data='[{"stage": 0.0, "status": "DONE"}]')
    @patch("filelock.FileLock")  # Mock FileLock
    def test_read_status_file_success(self, mock_lock, mock_file):
        """Test reading a valid status file."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        status = sm._read_status_file()
        self.assertEqual(len(status), 1)
        self.assertEqual(status[0]["stage"], 0.0)
        # mock_file.assert_called_once_with(sm.status_file_path, "r", encoding="utf-8")
        # FileLock interaction is mocked, direct assertion might be complex

    @patch("builtins.open", side_effect=IOError("Read error"))
    @patch("filelock.FileLock")
    def test_read_status_file_io_error(self, mock_lock, mock_file):
        """Test handling IOError during read."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir), use_locking=False)
        with self.assertRaises(StatusFileError):
            sm._read_status_file()

    # --- Tests for get_next_stage ---
    @patch(
        "utils.state_manager.StateManager._read_status_file",
        return_value={\"runs\": [{\"run_id\": 0, \"status_updates\": [{\"stage\": 0.0, \"status\": \"DONE\"}]}]}
    )
    def test_get_next_stage_after_zero(self, mock_read):
        """Test get_next_stage after stage 0 -> returns next available."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        next_stage = sm.get_next_stage()
        self.assertEqual(next_stage, 1.0)

    @patch(
        "utils.state_manager.StateManager._read_status_file",
        return_value={\"runs\": []}
    )
    def test_get_next_stage_empty_status(self, mock_read):
        """Test get_next_stage with no previous status -> returns lowest available."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        next_stage = sm.get_next_stage()
        self.assertEqual(next_stage, 0.0)

    @patch(
        "utils.state_manager.StateManager._read_status_file",
        return_value={\"runs\": [{\"run_id\": 0, \"status_updates\": [{\"stage\": 1.0, \"status\": \"DONE\"}]}]}
    )
    def test_get_next_stage_end_of_line(self, mock_read):
        """Test get_next_stage when last available stage is done -> loops to 0."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        next_stage = sm.get_next_stage()
        self.assertEqual(next_stage, 0.0)  # Loop back behavior

    @patch(
        "utils.state_manager.StateManager._read_status_file",
        return_value=[{"stage": 0.0, "status": "FAIL"}],
    )
    def test_get_next_stage_after_fail(self, mock_read):
        """Test get_next_stage after a failed stage -> raises RuntimeError."""
        sm = StateManager(str(self.test_dir), str(self.server_stages_dir))
        with self.assertRaises(RuntimeError):
            sm.get_next_stage()

    # TODO: Add tests for update_status (mocking _read/_write)
    # TODO: Add tests for _get_chroma_client interaction (requires async runner)
    # TODO: Add tests for context/reflection methods (requires async runner and mocking chroma_utils)


if __name__ == "__main__":
    unittest.main()
