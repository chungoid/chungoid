import pytest
from pathlib import Path

from chungoid.utils.prompt_manager import PromptManager


def create_prompt_tree(tmp_path: Path):
    root = tmp_path / "prompts"
    stages_dir = root / "stages"
    stages_dir.mkdir(parents=True)

    # common template
    (root / "common.yaml").write_text("""\
    preamble: PRE
    postamble: POST
    """)

    # Stage 0 file
    (stages_dir / "stage0.yaml").write_text(
        """\
    system_prompt: "Sys {{ context_data.value }}"
    user_prompt: "User {{ context_data.value }}"
    prompt_details: Something
    """
    )

    return root, stages_dir


@pytest.fixture()

def pm(tmp_path):
    root, stages_dir = create_prompt_tree(tmp_path)
    return PromptManager(server_stages_dir=str(stages_dir), common_template_path=str(root / "common.yaml"))


def test_load_definitions(pm):
    assert 0 in pm.stage_definitions
    stage0 = pm.get_stage_definition(0)
    assert stage0["user_prompt"].startswith("User")


def test_render_prompt(pm):
    rendered = pm.get_rendered_prompt(0, {"value": "X"})
    assert "Sys X" in rendered and "User X" in rendered 