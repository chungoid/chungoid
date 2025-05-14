import pytest
from pathlib import Path
# import yaml # No longer directly needed for YAMLError check if PromptLoadError wraps it well
from chungoid.utils.prompt_manager import PromptManager, PromptLoadError, PromptRenderError

# --- Test Data (Constants) ---
COMMON_YAML_CONTENT = """
preamble: COMMON PREAMBLE
postamble: COMMON POSTAMBLE
"""

STAGE0_YAML_CONTENT = """
system_prompt: |-
  System prompt for stage 0. Context: {{ context_data.data | default('No Context') }}
prompt_details: Details for stage 0
user_prompt: |-
  User prompt for stage 0.
"""

STAGE1_YAML_CONTENT = """
system_prompt: |-
  System prompt for stage 1.
prompt_details: |-
  Details for stage 1. Value: {{ context_data.value }}
user_prompt: |-
  User prompt for stage 1. Value: {{ context_data.value }}
"""

STAGE_FLOAT_YAML_CONTENT = """
system_prompt: |-
  System prompt for stage 0.5.
prompt_details: Float stage
user_prompt: |-
  User prompt float stage.
"""

STAGE_INVALID_YAML_SYNTAX_CONTENT = ": This is not valid YAML # Induces a parser error"

STAGE_MISSING_KEY_YAML_CONTENT = """
system_prompt: |-
  System prompt, but missing other required keys.
# 'prompt_details' is often required for validation within PromptManager.
# 'user_prompt' is also typically required.
# Omitting 'user_prompt' to test missing key validation
"""

STAGE_WITH_BAD_JINJA_CONTENT = """
system_prompt: |-
  Hello {{ name badly } }
user_prompt: User prompt for bad jinja
prompt_details: Details for bad jinja stage
"""

# --- Fixture ---
@pytest.fixture
def pm_base_setup(tmp_path: Path):
    """Provides a base directory structure for prompt manager tests."""
    base_dir = tmp_path / "pm_tests"
    base_dir.mkdir()
    stages_dir = base_dir / "stages"
    # stages_dir is NOT created here; tests create it if they need it populated,
    # or test for its absence.
    common_file = base_dir / "common.yaml"
    # common_file is also NOT created here by default.
    return base_dir, stages_dir, common_file

# --- Initialization Tests ---
def test_init_success(pm_base_setup):
    """Test successful initialization loads common and stage files."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir() # Create for this test
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.yaml").write_text(STAGE0_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage1.yaml").write_text(STAGE1_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.5.yaml").write_text(STAGE_FLOAT_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "other.txt").write_text("ignored content", encoding="utf-8")

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))
    assert pm is not None
    assert "preamble" in pm.common_template
    assert len(pm.stage_definitions) == 3
    assert 0 in pm.stage_definitions
    assert 1 in pm.stage_definitions
    assert 0.5 in pm.stage_definitions

def test_init_missing_stages_dir(pm_base_setup):
    """Test init fails if stages directory doesn't exist."""
    base_dir, stages_dir, common_file = pm_base_setup
    # Ensure stages_dir does NOT exist (fixture doesn't create it by default)
    # If it was created by another test using same tmp_path session, it might exist.
    # However, pytest typically provides a unique tmp_path per test function.
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    
    with pytest.raises(PromptLoadError, match="Resolved server stage template directory not found: .*"):
        PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

def test_init_missing_common_file(pm_base_setup):
    """Test init fails if common template file doesn't exist."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir() # Create stages_dir
    # Ensure common_file does NOT exist
    (stages_dir / "stage0.yaml").write_text(STAGE0_YAML_CONTENT, encoding="utf-8")

    with pytest.raises(PromptLoadError, match=r"Could not load common template: YAML file not found: .*"):
        PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

def test_init_invalid_yaml_syntax(pm_base_setup):
    """Test init fails if a stage YAML file has syntax errors."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.yaml").write_text(STAGE_INVALID_YAML_SYNTAX_CONTENT, encoding="utf-8")

    with pytest.raises(PromptLoadError, match=r"Failed to load 1 stage definition file\(s\)\. See logs for details\."):
        PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

def test_init_missing_required_key(pm_base_setup):
    """Test init fails if a stage YAML misses required keys (e.g., user_prompt)."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    # Write to a file name that will be processed by PromptManager
    (stages_dir / "stage0.yaml").write_text(STAGE_MISSING_KEY_YAML_CONTENT, encoding="utf-8")
    
    # It should match the generic error raised when any file fails to load.
    with pytest.raises(PromptLoadError, match=r"Failed to load 1 stage definition file\(s\)\. See logs for details\."):
        PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

# --- Get Stage Definition Tests ---
def test_get_stage_definition(pm_base_setup):
    """Test retrieving definitions for int, float, and missing stages."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.yaml").write_text(STAGE0_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.5.yaml").write_text(STAGE_FLOAT_YAML_CONTENT, encoding="utf-8")

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

    stage0_def = pm.get_stage_definition(0)
    assert stage0_def is not None
    assert stage0_def["system_prompt"].startswith("System prompt for stage 0")

    stage_float_def = pm.get_stage_definition(0.5)
    assert stage_float_def is not None
    assert stage_float_def["prompt_details"] == "Float stage"

    with pytest.raises(PromptLoadError, match="Stage 99 definition not found"):
        pm.get_stage_definition(99)

# --- Render Prompt Tests ---
def test_get_rendered_prompt_with_context(pm_base_setup):
    """Test successful rendering with context and common template elements."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage1.yaml").write_text(STAGE1_YAML_CONTENT, encoding="utf-8")

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))
    context = {"value": "TEST_VALUE"}
    rendered = pm.get_rendered_prompt(1, context_data=context)
    
    assert "COMMON PREAMBLE" in rendered
    assert "COMMON POSTAMBLE" in rendered
    assert "System prompt for stage 1." in rendered
    assert "User prompt for stage 1. Value: TEST_VALUE" in rendered

def test_get_rendered_prompt_missing_context_with_default(pm_base_setup):
    """Test rendering when context is missing but template has a default filter."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    (stages_dir / "stage0.yaml").write_text(STAGE0_YAML_CONTENT, encoding="utf-8") # Uses {{ context_data.data | default('No Context') }}

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))
    rendered_default = pm.get_rendered_prompt(0, context_data={}) 
    assert "Context: No Context" in rendered_default

def test_get_rendered_prompt_missing_context_no_default(pm_base_setup):
    """Test rendering when context is missing and template has no default (Jinja renders empty)."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    # STAGE1_YAML_CONTENT uses {{ context_data.value }} without a default
    # Rename the file to be processed, e.g., stage1.yaml (or a unique valid number)
    (stages_dir / "stage1.yaml").write_text(STAGE1_YAML_CONTENT.replace("stage 1", "stage1_no_default"), encoding="utf-8")

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))
    # Use the actual stage number (float or int) instead of string
    rendered_no_default = pm.get_rendered_prompt(1, context_data={}) 
    # prompt_details is not part of the final string
    # assert "Details for stage1_no_default. Value: " in rendered_no_default 
    assert "{{ context_data.value }}" not in rendered_no_default # Placeholder should not be literally present
    # Check that the user prompt part (which uses the value) renders it as empty, including its trailing newline
    expected_user_prompt_render = "User prompt for stage1_no_default. Value: \n"
    assert expected_user_prompt_render in rendered_no_default

def test_get_rendered_prompt_bad_jinja_template(pm_base_setup):
    """Test rendering fails with PromptRenderError if Jinja syntax is invalid."""
    base_dir, stages_dir, common_file = pm_base_setup
    stages_dir.mkdir()
    common_file.write_text(COMMON_YAML_CONTENT, encoding="utf-8")
    # Write to a file name that will be processed
    (stages_dir / "stage0.yaml").write_text(STAGE_WITH_BAD_JINJA_CONTENT, encoding="utf-8")

    pm = PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(common_file))

    # Update the stage identifier to the number used (0.0 for stage0.yaml)
    # The regex should still match as it's about TemplateSyntaxError
    # Made regex more general for stage number (0 or 0.0) and less strict on quotes
    with pytest.raises(PromptRenderError, match=r"Error rendering prompt for stage 0(?:\\.0)?: TemplateSyntaxError - expected token 'end of print statement', got 'badly'"):
        pm.get_rendered_prompt(0, context_data={"name": "Test"})
