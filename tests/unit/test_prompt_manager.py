import unittest
from unittest.mock import patch, MagicMock, mock_open
import yaml
from pathlib import Path
import sys
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError

pytestmark = pytest.mark.legacy

# --- Test Data ---
COMMON_YAML_CONTENT = """
preamble: COMMON PREAMBLE
postamble: COMMON POSTAMBLE
"""

STAGE0_YAML_CONTENT = """
system_prompt: System prompt for stage 0. Context: {{ context_data.data | default('No Context') }}
prompt_details: Details for stage 0
user_prompt: User prompt for stage 0.
"""

STAGE1_YAML_CONTENT = """
system_prompt: System prompt for stage 1.
prompt_details: Details for stage 1. Value: {{ context_data.value }}
user_prompt: User prompt for stage 1. Value: {{ context_data.value }}
"""

STAGE_FLOAT_YAML_CONTENT = """
system_prompt: System prompt for stage 0.5.
prompt_details: Float stage
user_prompt: User prompt float stage.
"""

STAGE_INVALID_YAML_CONTENT = ": This is not valid YAML"
STAGE_MISSING_KEY_YAML_CONTENT = "user_prompt: Only user prompt"


class TestPromptManager(unittest.TestCase):

    mock_stages_dir = "/fake/stages"
    mock_common_path = "/fake/common.yaml"

    # --- Initialization Tests ---
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file')
    @patch.object(Path, 'is_dir')
    def test_init_success(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test successful initialization loads common and stage files."""
        # Mock paths
        mock_stages_path = MagicMock(spec=Path)
        mock_common_path_obj = MagicMock(spec=Path)
        mock_stage0_path = MagicMock(spec=Path, name="stage0.yaml")
        mock_stage1_path = MagicMock(spec=Path, name="stage1.yaml")
        mock_stage_float_path = MagicMock(spec=Path, name="stage0.5.yaml")
        mock_other_path = MagicMock(spec=Path, name="other.txt")

        mock_is_dir.return_value = True # Stages dir exists
        mock_is_file.return_value = True # Common file exists
        mock_glob.return_value = [mock_stage0_path, mock_stage1_path, mock_stage_float_path, mock_other_path]

        # Side effect for Path() constructor
        def path_constructor_side_effect(p):
            if p == self.mock_stages_dir: return mock_stages_path
            if p == self.mock_common_path: return mock_common_path_obj
            # Add more specific path returns if needed by internal logic
            return MagicMock(spec=Path) # Default mock

        # Side effect for open
        def open_side_effect(p, *args, **kwargs):
            p_str = str(p)
            if p_str == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
            if p_str.endswith("stage0.yaml"): return mock_open(read_data=STAGE0_YAML_CONTENT)()
            if p_str.endswith("stage1.yaml"): return mock_open(read_data=STAGE1_YAML_CONTENT)()
            if p_str.endswith("stage0.5.yaml"): return mock_open(read_data=STAGE_FLOAT_YAML_CONTENT)()
            raise FileNotFoundError(p_str)
        mock_open_func.side_effect = open_side_effect

        # Side effect for yaml.safe_load
        def yaml_side_effect(stream):
            content = stream.read()
            if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
            if content == STAGE0_YAML_CONTENT: return yaml.safe_load(STAGE0_YAML_CONTENT)
            if content == STAGE1_YAML_CONTENT: return yaml.safe_load(STAGE1_YAML_CONTENT)
            if content == STAGE_FLOAT_YAML_CONTENT: return yaml.safe_load(STAGE_FLOAT_YAML_CONTENT)
            raise yaml.YAMLError("Mock YAML error")
        mock_safe_load.side_effect = yaml_side_effect

        with patch('pathlib.Path', side_effect=path_constructor_side_effect):
            pm = PromptManager(self.mock_stages_dir, self.mock_common_path)

            self.assertIsNotNone(pm)
            self.assertIn("preamble", pm.common_template)
            self.assertEqual(len(pm.stage_definitions), 3)
            self.assertIn(0, pm.stage_definitions)
            self.assertIn(1, pm.stage_definitions)
            self.assertIn(0.5, pm.stage_definitions)

    @patch.object(Path, 'is_dir', return_value=False) # Stages dir does NOT exist
    def test_init_missing_stages_dir(self, mock_is_dir):
        """Test init fails if stages directory doesn't exist."""
        with self.assertRaisesRegex(PromptLoadError, "Resolved server stage template directory not found"):
             # Need to mock the Path constructor to return the mock with is_dir=False
             with patch('pathlib.Path') as mock_path_constructor:
                 mock_path_instance = MagicMock(spec=Path)
                 mock_path_instance.resolve.return_value = mock_path_instance
                 mock_path_instance.is_dir.return_value = False
                 mock_path_constructor.return_value = mock_path_instance
                 PromptManager(self.mock_stages_dir, self.mock_common_path)

    @patch.object(Path, 'is_dir', return_value=True) # Stages dir exists
    @patch.object(Path, 'is_file', return_value=False) # Common file does NOT exist
    def test_init_missing_common_file(self, mock_is_file, mock_is_dir):
        """Test init fails if common template file doesn't exist."""
        with self.assertRaisesRegex(PromptLoadError, "Resolved common template file not found"):
             # Mock Path constructor again
             with patch('pathlib.Path') as mock_path_constructor:
                 def path_side_effect(p):
                     instance = MagicMock(spec=Path)
                     instance.resolve.return_value = instance
                     if p == self.mock_stages_dir:
                         instance.is_dir.return_value = True
                     elif p == self.mock_common_path:
                         instance.is_file.return_value = False # Mock common as not file
                     else:
                         instance.is_dir.return_value = False
                         instance.is_file.return_value = False
                     return instance
                 mock_path_constructor.side_effect = path_side_effect
                 PromptManager(self.mock_stages_dir, self.mock_common_path)

    @patch('yaml.safe_load', side_effect=yaml.YAMLError("Bad YAML"))
    @patch('builtins.open', new_callable=mock_open, read_data=STAGE_INVALID_YAML_CONTENT)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_init_invalid_yaml_syntax(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test init fails if a stage YAML file has syntax errors."""
        mock_bad_stage_path = MagicMock(spec=Path, name="stage_bad.yaml")
        mock_glob.return_value = [mock_bad_stage_path]

        # Mock Path constructor
        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path)
                instance.resolve.return_value = instance
                instance.name = Path(p).name # Set name for glob results
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage_bad.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect

            # Mock open to return bad content for the specific file
            original_open = open
            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage_bad.yaml"): return mock_open(read_data=STAGE_INVALID_YAML_CONTENT)()
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                return original_open(p, *args, **kwargs)
            mock_open_func.side_effect = open_side_effect

            with self.assertRaisesRegex(PromptLoadError, "Failed to load 1 stage definition file"):
                PromptManager(self.mock_stages_dir, self.mock_common_path)

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_init_missing_required_key(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test init fails if a stage YAML misses required keys."""
        mock_missing_stage_path = MagicMock(spec=Path, name="stage_missing.yaml")
        mock_glob.return_value = [mock_missing_stage_path]

        # Mock Path constructor
        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path)
                instance.resolve.return_value = instance
                instance.name = Path(p).name
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage_missing.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect

            # Mock open
            original_open = open
            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage_missing.yaml"): return mock_open(read_data=STAGE_MISSING_KEY_YAML_CONTENT)()
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                return original_open(p, *args, **kwargs)
            mock_open_func.side_effect = open_side_effect

            # Mock safe_load
            def yaml_side_effect(stream):
                content = stream.read()
                if content == STAGE_MISSING_KEY_YAML_CONTENT: return yaml.safe_load(STAGE_MISSING_KEY_YAML_CONTENT)
                if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
                raise yaml.YAMLError("Mock YAML error")
            mock_safe_load.side_effect = yaml_side_effect

            with self.assertRaisesRegex(PromptLoadError, "missing required string keys:.*prompt_details"):
                 # The missing key might be prompt_details now based on current validator
                 PromptManager(self.mock_stages_dir, self.mock_common_path)

    # --- Get Stage Definition Tests ---
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_stage_definition(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test retrieving definitions for int, float, and missing stages."""
        mock_stage0_path = MagicMock(spec=Path, name="stage0.yaml")
        mock_stage_float_path = MagicMock(spec=Path, name="stage0.5.yaml")
        mock_glob.return_value = [mock_stage0_path, mock_stage_float_path]

        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path); instance.resolve.return_value = instance; instance.name = Path(p).name
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage0.yaml"): instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage0.5.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect

            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage0.yaml"): return mock_open(read_data=STAGE0_YAML_CONTENT)()
                if str(p).endswith("stage0.5.yaml"): return mock_open(read_data=STAGE_FLOAT_YAML_CONTENT)()
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                raise FileNotFoundError(p)
            mock_open_func.side_effect = open_side_effect

            def yaml_side_effect(stream):
                content = stream.read()
                if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
                if content == STAGE0_YAML_CONTENT: return yaml.safe_load(STAGE0_YAML_CONTENT)
                if content == STAGE_FLOAT_YAML_CONTENT: return yaml.safe_load(STAGE_FLOAT_YAML_CONTENT)
                raise yaml.YAMLError("Mock YAML error")
            mock_safe_load.side_effect = yaml_side_effect

            pm = PromptManager(self.mock_stages_dir, self.mock_common_path)
            stage0_def = pm.get_stage_definition(0)
            self.assertIn("prompt_details", stage0_def)
            stage0_5_def = pm.get_stage_definition(0.5)
            self.assertIn("prompt_details", stage0_5_def)
            with self.assertRaises(PromptLoadError):
                pm.get_stage_definition(99)

    # --- Rendering Tests ---
    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_rendered_prompt(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test rendering prompts with and without context."""
        mock_stage0_path = MagicMock(spec=Path, name="stage0.yaml")
        mock_stage1_path = MagicMock(spec=Path, name="stage1.yaml")
        mock_glob.return_value = [mock_stage0_path, mock_stage1_path]

        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path); instance.resolve.return_value = instance; instance.name = Path(p).name
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage0.yaml"): instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage1.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect

            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage0.yaml"): return mock_open(read_data=STAGE0_YAML_CONTENT)()
                if str(p).endswith("stage1.yaml"): return mock_open(read_data=STAGE1_YAML_CONTENT)()
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                raise FileNotFoundError(p)
            mock_open_func.side_effect = open_side_effect

            def yaml_side_effect(stream):
                content = stream.read()
                if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
                if content == STAGE0_YAML_CONTENT: return yaml.safe_load(STAGE0_YAML_CONTENT)
                if content == STAGE1_YAML_CONTENT: return yaml.safe_load(STAGE1_YAML_CONTENT)
                raise yaml.YAMLError("Mock YAML error")
            mock_safe_load.side_effect = yaml_side_effect

            pm = PromptManager(self.mock_stages_dir, self.mock_common_path)

            context0 = {"data": "Some Context"}
            prompt0 = pm.get_rendered_prompt(0, context_data=context0)
            self.assertIn("COMMON PREAMBLE", prompt0)
            self.assertIn("Context: Some Context", prompt0)
            self.assertIn("User prompt for stage 0.", prompt0)
            self.assertIn("COMMON POSTAMBLE", prompt0)

            prompt0_no_ctx = pm.get_rendered_prompt(0)
            self.assertIn("Context: No Context", prompt0_no_ctx)

            context1 = {"value": 123}
            prompt1 = pm.get_rendered_prompt(1, context_data=context1)
            self.assertIn("Details for stage 1. Value: 123", prompt1)
            self.assertIn("User prompt for stage 1. Value: 123", prompt1)
            self.assertIn("COMMON POSTAMBLE", prompt1)

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_rendered_prompt_missing_context(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test PromptRenderError is raised if required context is missing."""
        mock_stage1_path = MagicMock(spec=Path, name="stage1.yaml")
        mock_glob.return_value = [mock_stage1_path]
        # ... (Similar mock setup as test_get_rendered_prompt) ...
        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path); instance.resolve.return_value = instance; instance.name = Path(p).name
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage1.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect
            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage1.yaml"): return mock_open(read_data=STAGE1_YAML_CONTENT)()
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                raise FileNotFoundError(p)
            mock_open_func.side_effect = open_side_effect
            def yaml_side_effect(stream):
                content = stream.read()
                if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
                if content == STAGE1_YAML_CONTENT: return yaml.safe_load(STAGE1_YAML_CONTENT)
                raise yaml.YAMLError("Mock YAML error")
            mock_safe_load.side_effect = yaml_side_effect

            pm = PromptManager(self.mock_stages_dir, self.mock_common_path)
            with self.assertRaises(PromptRenderError):
                 pm.get_rendered_prompt(1, context_data={}) # Missing 'value'

    @patch('yaml.safe_load')
    @patch('builtins.open', new_callable=mock_open)
    @patch.object(Path, 'glob')
    @patch.object(Path, 'is_file', return_value=True)
    @patch.object(Path, 'is_dir', return_value=True)
    def test_get_rendered_prompt_bad_template(self, mock_is_dir, mock_is_file, mock_glob, mock_open_func, mock_safe_load):
        """Test PromptRenderError is raised for invalid Jinja syntax in template."""
        mock_bad_stage_path = MagicMock(spec=Path, name="stage_bad.yaml")
        mock_glob.return_value = [mock_bad_stage_path]
        # ... (Similar mock setup as test_init_invalid_yaml_syntax but load STAGE0 for common) ...
        with patch('pathlib.Path') as mock_path_constructor:
            def path_side_effect(p):
                instance = MagicMock(spec=Path); instance.resolve.return_value = instance; instance.name = Path(p).name
                if p == self.mock_stages_dir: instance.is_dir.return_value = True
                elif p == self.mock_common_path: instance.is_file.return_value = True
                elif p == str(Path(self.mock_stages_dir) / "stage_bad.yaml"): instance.is_file.return_value = True
                else: instance.is_dir.return_value = False; instance.is_file.return_value = False
                return instance
            mock_path_constructor.side_effect = path_side_effect
            def open_side_effect(p, *args, **kwargs):
                if str(p).endswith("stage_bad.yaml"): return mock_open(read_data="prompt_details: {{ bad variable } }")() # Invalid Jinja
                if str(p) == self.mock_common_path: return mock_open(read_data=COMMON_YAML_CONTENT)()
                raise FileNotFoundError(p)
            mock_open_func.side_effect = open_side_effect
            def yaml_side_effect(stream):
                content = stream.read()
                if content == COMMON_YAML_CONTENT: return yaml.safe_load(COMMON_YAML_CONTENT)
                if content == "prompt_details: {{ bad variable } }": return {"prompt_details": "{{ bad variable } }"} # Load the bad template str
                raise yaml.YAMLError("Mock YAML error")
            mock_safe_load.side_effect = yaml_side_effect

            pm = PromptManager(self.mock_stages_dir, self.mock_common_path)
            with self.assertRaises(PromptRenderError):
                pm.get_rendered_prompt(0) # Assuming stage 0 maps to stage_bad.yaml here based on glob mock

if __name__ == "__main__":
    unittest.main()
