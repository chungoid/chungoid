import unittest
import tempfile
from pathlib import Path
import shutil
from utils.prompt_manager import PromptManager, PromptLoadError
import pytest

pytestmark = pytest.mark.legacy

class TestPromptManagerCore(unittest.TestCase):
    """Covers loading & rendering logic of PromptManager with minimal YAML."""

    def setUp(self):
        # temp dir with stage definitions and common template
        self.temp_dir = Path(tempfile.mkdtemp())
        self.stages_dir = self.temp_dir / "stages"
        self.stages_dir.mkdir()
        self.common_file = self.temp_dir / "common.yaml"

        # minimal common template
        self.common_file.write_text("""preamble: |\n  PRE\npostamble: |\n  POST\n""", encoding="utf-8")

        # stage0 definition (valid)
        (self.stages_dir / "stage0.yaml").write_text(
            """
            system_prompt: "Greetings {{ name }}"
            user_prompt: "What is up?"
            """,
            encoding="utf-8",
        )
        # stage1 missing keys (invalid)
        (self.stages_dir / "stage1.yaml").write_text(
            """
            system_prompt: "Only system prompt"
            """,
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_render_prompt_success(self):
        pm = PromptManager(server_stages_dir=str(self.stages_dir), common_template_path=str(self.common_file))
        rendered = pm.get_rendered_prompt(0, {"name": "Alice"})
        # Ensure context substituted and common pre/post not empty
        self.assertIn("Alice", rendered)
        self.assertIn("PRE", rendered)
        self.assertIn("POST", rendered)

    def test_load_stage_missing_keys_raises(self):
        # Expect PromptLoadError on init because stage1 lacks 'user_prompt'
        with self.assertRaises(PromptLoadError):
            PromptManager(server_stages_dir=str(self.stages_dir), common_template_path=str(self.common_file))

if __name__ == "__main__":
    unittest.main() 