import yaml, re, pathlib

STAGE_PATH = pathlib.Path(__file__).resolve().parents[3] / "server_prompts" / "stages" / "stage_minus1_goal_draft.yaml"

def test_sequential_thinking_block_present():
    data = yaml.safe_load(STAGE_PATH.read_text())
    assert "sequential thinking" in data["system_prompt"].lower() 