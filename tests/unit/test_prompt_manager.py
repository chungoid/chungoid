import unittest
from pathlib import Path

# Add project root to path to allow importing utils
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.prompt_manager import PromptManager, PromptLoadError

# Sample YAML content
COMMON_YAML = """
preamble: |-
  COMMON PREAMBLE
  You are Chungoid.
postamble: |-
  COMMON POSTAMBLE
  Follow instructions carefully.
"""

STAGE0_YAML = """
name: Stage 0 - Test
prompt_details: |-
  Details for stage {{ stage_number }}.
  Reflections: {{ reflections_summary | default('None') }}
"""

STAGE1_YAML = """
name: Stage 1 - Test
prompt_details: |-
  Stage 1 details.
"""


class TestPromptManager(unittest.TestCase):
    def setUp(self):
        """Set up temporary directories and files."""
        self.base_dir = Path("./test_pm_dir")
        self.stages_dir = self.base_dir / "stages"
        self.common_path = self.base_dir / "common.yaml"

        self.stages_dir.mkdir(parents=True, exist_ok=True)

        with open(self.common_path, "w") as f:
            f.write(COMMON_YAML)
        with open(self.stages_dir / "stage0.yaml", "w") as f:
            f.write(STAGE0_YAML)
        with open(self.stages_dir / "stage1.yaml", "w") as f:
            f.write(STAGE1_YAML)

    def tearDown(self):
        """Clean up temporary files and directories."""
        for f in self.stages_dir.glob("*"):
            f.unlink()
        self.stages_dir.rmdir()
        self.common_path.unlink()
        self.base_dir.rmdir()

    def test_init_success(self):
        """Test successful initialization and loading."""
        pm = PromptManager(str(self.stages_dir), str(self.common_path))
        self.assertIsNotNone(pm)
        self.assertIn(0, pm.stage_definitions)
        self.assertIn(1, pm.stage_definitions)
        self.assertIn("preamble", pm.common_template)

    def test_init_missing_stages_dir(self):
        """Test initialization fails if stages directory is missing."""
        with self.assertRaises(PromptLoadError):
            PromptManager("./nonexistent_stages", str(self.common_path))

    def test_init_missing_common_file(self):
        """Test initialization fails if common template file is missing."""
        with self.assertRaises(PromptLoadError):
            PromptManager(str(self.stages_dir), "./nonexistent_common.yaml")

    def test_get_stage_definition(self):
        """Test retrieving a loaded stage definition."""
        pm = PromptManager(str(self.stages_dir), str(self.common_path))
        stage0_def = pm.get_stage_definition(0)
        self.assertEqual(stage0_def["name"], "Stage 0 - Test")

        with self.assertRaises(PromptLoadError):  # Test non-existent stage
            pm.get_stage_definition(99)

    def test_get_rendered_prompt_stage0(self):
        """Test rendering a prompt for stage 0 with context."""
        pm = PromptManager(str(self.stages_dir), str(self.common_path))
        context = {"reflections_summary": "Previous run was okay."}
        prompt = pm.get_rendered_prompt(0, context_data=context)

        self.assertIn("COMMON PREAMBLE", prompt)
        self.assertIn("Details for stage 0.", prompt)
        self.assertIn("Reflections: Previous run was okay.", prompt)
        self.assertIn("COMMON POSTAMBLE", prompt)

    def test_get_rendered_prompt_stage1_no_context(self):
        """Test rendering a prompt for stage 1 without extra context."""
        pm = PromptManager(str(self.stages_dir), str(self.common_path))
        prompt = pm.get_rendered_prompt(1)

        self.assertIn("COMMON PREAMBLE", prompt)
        self.assertIn("Stage 1 details.", prompt)
        self.assertIn("Reflections: None", prompt)  # Default value if not provided
        self.assertIn("COMMON POSTAMBLE", prompt)

    def test_render_error(self):
        """Test handling of Jinja template rendering errors."""
        # Create a bad stage file with invalid Jinja syntax
        with open(self.stages_dir / "stage_bad.yaml", "w") as f:
            f.write("name: Bad Stage\nprompt_details: Hello {{ invalid syntax \}")

        pm = PromptManager(str(self.stages_dir), str(self.common_path))
        with self.assertRaises(PromptLoadError):  # Expecting PromptLoadError on render fail
            pm.get_rendered_prompt("bad")

    # TODO: Add tests for YAML loading errors (invalid format, not dict)


if __name__ == "__main__":
    unittest.main()
