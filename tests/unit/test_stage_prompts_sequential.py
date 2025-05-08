import yaml, pathlib, re
from itertools import chain

PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[2] / "server_prompts" / "stages"

# Gather all stage prompt files (*excluding* templates or non-stage extras)
STAGE_FILES = [p for p in PROMPTS_DIR.glob("stage*.yaml") if p.is_file()]

import pytest

@pytest.mark.parametrize("prompt_path", STAGE_FILES, ids=[p.name for p in STAGE_FILES])
def test_system_prompt_contains_sequential_thinking(prompt_path):
    data = yaml.safe_load(prompt_path.read_text())
    # system_prompt is mandatory and should reference sequential thinking
    assert re.search(r"sequential thinking", data.get("system_prompt", ""), re.IGNORECASE), f"Missing sequential thinking in {prompt_path.name}" 