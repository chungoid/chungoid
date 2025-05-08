"""Simple Stage-Flow executor (MVP).

Loads a Stage-Flow YAML file that conforms to ``schemas/stage_flow_schema.json``
and executes its stages in order, delegating business logic to *agents* supplied
at runtime via a registry (a mapping of ``agent_name -> callable``).

The executor purposefully avoids coupling to the broader Chungoid core; it can
be used in unit tests with plain Python callables or in production by passing
wrappers around real MCP tools.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import yaml
import jsonschema  # type: ignore

from chungoid.utils.agent_resolver import AgentProvider, DictAgentProvider, RegistryAgentProvider  # noqa: E501

ROOT = Path(__file__).resolve().parents[3]  # /chungoid-core/src/chungoid/ -> ../../..
SCHEMA_PATH = ROOT / "schemas" / "stage_flow_schema.json"

StageDict = Dict[str, Any]
AgentCallable = Callable[[StageDict], Dict[str, Any]]  # returns arbitrary dict


class FlowValidationError(Exception):
    """Raised when the YAML does not match the Stage-Flow schema."""


class UnknownAgentError(Exception):
    """Raised when an agent referenced in the flow is not found in the registry."""


class FlowExecutor:
    """Executes Stage-Flow YAML definitions."""

    def __init__(self, agent_registry: Dict[str, AgentCallable] | AgentProvider, *, max_steps: int = 100) -> None:
        # Backward-compat: accept raw dict or an AgentProvider implementation.
        if isinstance(agent_registry, dict):
            self.agent_provider: AgentProvider = DictAgentProvider(agent_registry)
        else:
            self.agent_provider = agent_registry
        self.max_steps = max_steps

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------

    def run(self, yaml_path: Path | str) -> List[str]:
        """Execute the flow and return the ordered list of completed stage names."""
        data = self._load_and_validate(yaml_path)
        stages_executed: List[str] = []

        current_name: Optional[str] = data["start_stage"]
        steps = 0
        while current_name:
            if steps >= self.max_steps:
                raise RuntimeError("Stage-flow exceeded maximum allowed steps (possible cycle)")
            steps += 1

            stage = data["stages"].get(current_name)
            if stage is None:
                raise KeyError(f"Stage '{current_name}' not found in YAML definition")

            stages_executed.append(current_name)

            # Determine agent identifier
            if "agent_id" in stage:
                agent_name: str = stage["agent_id"]
            else:
                agent_name = stage["agent"]
            agent_fn = self.agent_provider.get(agent_name)
            if agent_fn is None:
                raise UnknownAgentError(f"Agent '{agent_name}' not registered")

            agent_result = agent_fn(stage)

            # Determine next stage
            next_field = stage.get("next")
            if next_field in (None, "null"):
                break  # Flow completed

            if isinstance(next_field, str):
                current_name = next_field
            elif isinstance(next_field, dict):
                condition_key = next_field.get("condition")
                if condition_key is None:
                    raise ValueError(f"Conditional 'next' missing 'condition' key in stage '{current_name}'")
                cond_value = bool(agent_result.get(condition_key))
                branch_key = "true" if cond_value else "false"
                current_name = next_field.get(branch_key)
                if current_name is None:
                    raise ValueError(
                        f"Conditional 'next' in stage '{current_name}' missing branch '{branch_key}'"
                    )
            else:
                raise TypeError(f"Unsupported 'next' type in stage '{current_name}': {type(next_field)}")

        return stages_executed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_yaml(path: Path | str) -> Dict[str, Any]:
        with Path(path).open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)  # type: ignore[return-value]

    def _load_and_validate(self, path: Path | str) -> Dict[str, Any]:
        data = self._load_yaml(path)
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        try:
            jsonschema.validate(data, schema)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise FlowValidationError(str(exc)) from exc
        return data


# ----------------------------- CLI helper ---------------------------------


def _cli() -> None:  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run a Stage-Flow YAML with dummy agents.")
    parser.add_argument("yaml_path", type=str, help="Path to flow YAML file")
    args = parser.parse_args()

    # Very simple demo agents
    def pass_agent(stage: StageDict) -> Dict[str, Any]:
        return {}

    def goal_analyzer(_: StageDict) -> Dict[str, Any]:
        return {"has_clear_goals": True}

    registry = {
        "goal_analyzer": goal_analyzer,
        "goal_refiner": pass_agent,
        "tech_stack_advisor": pass_agent,
        "project_bootstrapper": pass_agent,
    }

    executor = FlowExecutor(registry)
    result = executor.run(Path(args.yaml_path))
    print("Flow completed, stages executed:", result)


if __name__ == "__main__":  # pragma: no cover
    _cli() 