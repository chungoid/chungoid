import pytest
import textwrap

from chungoid.runtime.orchestrator import ExecutionPlan


def test_validator_rejects_missing_agent_id():
    bad_yaml = textwrap.dedent(
        """
        start_stage: s1
        stages:
          s1:
            # agent_id missing on purpose
            input: "hi"
        """
    )
    with pytest.raises(ValueError, match="agent_id"):
        ExecutionPlan.from_yaml(bad_yaml)


def test_validator_rejects_missing_start_stage():
    bad_yaml = textwrap.dedent(
        """
        stages:
          s1:
            agent_id: foo
        """
    )
    with pytest.raises(ValueError):
        ExecutionPlan.from_yaml(bad_yaml) 