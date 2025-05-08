import yaml, pathlib, re

# Locate prompt file relative to project root
PROMPT_PATH = pathlib.Path(__file__).resolve().parents[2] / "server_prompts" / "stages" / "stage_minus1_goal_draft.yaml"


def test_stage_minus1_prompt_contract():
    """Validate critical fields and contract phrases in Stage –1 prompt."""
    data = yaml.safe_load(PROMPT_PATH.read_text())

    # Basic metadata
    assert data["id"] == "stage_-1_goal_draft"
    assert data["agent"] == "goal_drafter"

    # System prompt must instruct sequential thinking
    assert re.search(r"sequential thinking", data["system_prompt"], re.IGNORECASE)

    # prompt_details must mention execution contract and artifact filenames
    details_lower = data["prompt_details"].lower()
    assert "execution contract" in details_lower
    assert "goal_draft.md" in details_lower
    assert "goal_questions.json" in details_lower 