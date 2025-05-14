import unittest
import tempfile
from pathlib import Path
import shutil
from unittest.mock import patch, MagicMock
import pytest

from chungoid.utils.state_manager import StateManager, StatusFileError

# Setup logging

class TestStateManagerCore(unittest.TestCase):
    """Covers status read / write paths of StateManager without hitting real Chroma."""

    def setUp(self):
        # temp project
        self.project_dir = Path(tempfile.mkdtemp())
        # create stage definitions directory expected by StateManager (server side)
        self.server_prompts_dir = self.project_dir / "server_prompts"
        stages_dir = self.server_prompts_dir / "stages"
        stages_dir.mkdir(parents=True)
        # minimal stage0.yaml so get_next_stage works
        (stages_dir / "stage0.yaml").write_text("system_prompt: foo\nuser_prompt: bar\n", encoding="utf-8")

    def tearDown(self):
        shutil.rmtree(self.project_dir)

    @patch("chungoid.utils.chroma_utils.get_chroma_client")
    def test_status_file_initialization_and_update(self, mock_get_client):
        # patch chroma client with minimal mock
        mock_get_client.return_value = MagicMock(list_collections=lambda: [])

        sm = StateManager(
            target_directory=str(self.project_dir),
            server_stages_dir=str(self.server_prompts_dir / "stages"),
            use_locking=False,
        )
        # initial status should be empty runs
        self.assertEqual(sm.get_full_status(), {"runs": []})

        # perform update
        ok = sm.update_status(stage=0, status="DONE", artifacts=["a.txt"])
        self.assertTrue(ok)
        status = sm.get_full_status()
        self.assertEqual(len(status["runs"]), 1)
        self.assertEqual(status["runs"][0]["status_updates"][0]["stage"], 0.0)
        self.assertEqual(status["runs"][0]["status_updates"][0]["status"], "DONE")
        self.assertIn("a.txt", status["runs"][0]["status_updates"][0]["artifacts"])

if __name__ == "__main__":
    unittest.main() 