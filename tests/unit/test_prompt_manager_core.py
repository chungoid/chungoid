import pytest
from pathlib import Path
from chungoid.utils.prompt_manager import PromptManager, PromptLoadError

# Original unittest.TestCase and pytestmark removed

@pytest.fixture
def prompt_manager_core_setup_valid(tmp_path: Path):
    """Sets up a valid prompt structure for successful rendering tests."""
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    common_file = tmp_path / "common.yaml"

    common_file.write_text(
        """preamble: |\n  PRE\npostamble: |\n  POST\n""", encoding="utf-8"
    )

    (stages_dir / "stage0.yaml").write_text(
        """
        system_prompt: "Greetings {{ context_data.name }}"
        user_prompt: "What is up?"
        prompt_details: "Details for stage 0"
        """,
        encoding="utf-8",
    )
    return stages_dir, common_file


@pytest.fixture
def prompt_manager_core_setup_invalid_stage(tmp_path: Path):
    """Sets up a prompt structure with one invalid stage file for error testing."""
    stages_dir = tmp_path / "stages"
    stages_dir.mkdir()
    common_file = tmp_path / "common.yaml"

    common_file.write_text(
        """preamble: |\n  PRE\npostamble: |\n  POST\n""", encoding="utf-8"
    )

    # Valid stage0 (might not be loaded if invalid stage1 causes early init failure)
    (stages_dir / "stage0.yaml").write_text(
        """
        system_prompt: "Greetings {{ context_data.name }}"
        user_prompt: "What is up?"
        prompt_details: "Details for stage 0"
        """,
        encoding="utf-8",
    )

    # Invalid stage1 (missing user_prompt and prompt_details)
    (stages_dir / "stage1.yaml").write_text(
        """system_prompt: "Only system prompt"        
        """,  # Note: Original had '"""system_prompt: "Only system prompt"        """' which has extra quote. Corrected.
        encoding="utf-8"
    )
    return stages_dir, common_file


def test_core_render_prompt_success(prompt_manager_core_setup_valid):
    """Tests successful prompt rendering with valid setup."""
    stages_dir, common_file = prompt_manager_core_setup_valid
    pm = PromptManager(
        server_stages_dir=str(stages_dir), common_template_path=str(common_file)
    )
    rendered = pm.get_rendered_prompt(0, {"name": "Alice"})
    assert "Alice" in rendered
    assert "PRE" in rendered
    assert "POST" in rendered


def test_core_load_stage_missing_keys_raises(prompt_manager_core_setup_invalid_stage):
    """Tests that PromptLoadError is raised if a stage file is missing required keys."""
    stages_dir, common_file = prompt_manager_core_setup_invalid_stage
    with pytest.raises(PromptLoadError):
        PromptManager(
            server_stages_dir=str(stages_dir), common_template_path=str(common_file)
        ) 