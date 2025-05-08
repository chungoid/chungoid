import yaml, re, pathlib

# `tests/unit/` → `tests` (parents[1]) → project root (`chungoid-core`) (parents[2])
# We only need to go up two levels from the test file to reach the `chungoid-core` package root,
# then append `server_prompts/stages/...`.
STAGE_PATH = pathlib.Path(__file__).resolve().parents[2] / "server_prompts" / "stages" / "stage_minus1_goal_draft.yaml"

def test_sequential_thinking_block_present():
    data = yaml.safe_load(STAGE_PATH.read_text())
    assert "sequential thinking" in data["system_prompt"].lower() 