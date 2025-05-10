"""Minimal execution runtime skeleton (Phase-6 scaffold).

This module will gradually evolve into the production‐grade orchestrator that
executes Flow YAML graphs.  For now we expose a *SyncOrchestrator* that can run
very simple flows in-process so we can start adding tests and build up the API
surface incrementally.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import datetime as _dt
import yaml
from pydantic import BaseModel, Field, ConfigDict

__all__ = [
    "StageSpec",
    "ExecutionPlan",
    "SyncOrchestrator",
    "AsyncOrchestrator",
]


# ---------------------------------------------------------------------------
# DSL → Python models
# ---------------------------------------------------------------------------


class StageSpec(BaseModel):
    """Specification of a single stage inside a flow."""

    agent_id: str = Field(..., description="ID of the agent to invoke for this stage")
    input: Optional[str] = Field(None, description="Expression or literal passed to the agent")

    model_config = ConfigDict(extra="allow")


class ExecutionPlan(BaseModel):
    """Validated, structured representation of the Flow YAML."""

    id: str
    created: _dt.datetime = Field(default_factory=_dt.datetime.utcnow)
    start_stage: str
    stages: Dict[str, StageSpec]

    @classmethod
    def from_yaml(cls, yaml_text: str, flow_id: str | None = None) -> "ExecutionPlan":
        """Parse the *yaml_text* of a FlowCard and convert it to a plan."""

        data = yaml.safe_load(yaml_text)
        if not isinstance(data, dict):
            raise ValueError("Flow YAML must map keys → values")

        # Validate against JSON schema (if jsonschema is installed)
        try:
            _validate_dsl(data)
        except Exception as exc:
            # Rewrap to avoid leaking jsonschema dependency to callers
            raise ValueError(f"Flow DSL validation error: {exc}") from exc

        # Fallback check (redundant once schema validated but keeps mypy happy)
        if "stages" not in data or "start_stage" not in data:
            raise ValueError("Flow YAML missing required 'stages' or 'start_stage' key")

        return cls(
            id=flow_id or "<unknown>",
            start_stage=data["start_stage"],
            stages=data["stages"],
        )


# ---------------------------------------------------------------------------
# Orchestrator (sync only for now)
# ---------------------------------------------------------------------------


class SyncOrchestrator:
    """Very small synchronous orchestrator.

    It doesn't do agent routing yet – instead, it simply walks through the
    stages in breadth-first order starting from *start_stage* and returns a log
    of visited stage names.  This is enough for early unit tests and will be
    replaced by real agent invocation later.
    """

    def __init__(self, plan: ExecutionPlan) -> None:
        self.plan = plan

    def run(self, *, max_hops: int = 64, context: dict | None = None) -> List[str]:
        """Execute the plan and return a list of stage names that were 'run'.
        If a stage has `next_if`, evaluate conditions using the provided context.
        """
        visited: List[str] = []
        current = self.plan.start_stage
        context = context or {}
        for _ in range(max_hops):
            visited.append(current)
            stage_spec = self.plan.stages[current]
            next_stage = None
            if hasattr(stage_spec, "next_if") and getattr(stage_spec, "next_if", None):
                cond_map = getattr(stage_spec, "next_if")
                if isinstance(cond_map, dict) and cond_map:
                    for cond, candidate in cond_map.items():
                        if cond == "always":
                            next_stage = candidate
                            break
                        # Enhanced evaluator: support input ==, !=, >, <, >=, <= value
                        parsed = self._parse_condition(cond)
                        if parsed:
                            key, op, val = parsed
                            if key in context:
                                ctx_val = context[key]
                                # Try to convert both to float if possible
                                try:
                                    ctx_val_num = float(ctx_val)
                                    val_num = float(val)
                                    compare_type = "numeric"
                                except Exception:
                                    ctx_val_num = ctx_val
                                    val_num = val
                                    compare_type = "string"
                                if self._eval_condition(ctx_val_num, op, val_num):
                                    next_stage = candidate
                                    break
                    # fallback: if nothing matched, next_stage remains None
            if not next_stage:
                next_stage = getattr(stage_spec, "next", None)  # type: ignore[attr-defined]
            if not next_stage:
                break
            if next_stage not in self.plan.stages:
                raise KeyError(f"Unknown next stage '{next_stage}' referenced from '{current}'")
            current = next_stage
        return visited

    @staticmethod
    def _parse_condition(cond: str):
        # Only allow simple conditions: <key> <op> <value>
        import re
        pattern = r"^(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+)$"
        m = re.match(pattern, cond)
        if not m:
            return None
        key, op, val = m.groups()
        val = val.strip().strip('"\'')
        return key, op, val

    @staticmethod
    def _eval_condition(left, op, right):
        if op == "==":
            return left == right
        if op == "!=":
            return left != right
        if op == ">":
            return left > right
        if op == "<":
            return left < right
        if op == ">=":
            return left >= right
        if op == "<=":
            return left <= right
        return False


# ---------------------------------------------------------------------------
# Async variant (placeholder – internally reuses SyncOrchestrator for now)
# ---------------------------------------------------------------------------


class AsyncOrchestrator:
    """Asynchronous wrapper around *SyncOrchestrator*.

    This class will evolve into a *true* async engine (awaiting agent IO, etc.).
    For now it simply delegates to the synchronous implementation while keeping
    an `async def run()` API so we can start writing async tests and integrate
    into FastAPI endpoints.
    """

    def __init__(self, plan: ExecutionPlan) -> None:  # noqa: D401
        self._delegate = SyncOrchestrator(plan)

    async def run(self, *, max_hops: int = 64, context: dict | None = None) -> List[str]:
        """Run the plan asynchronously and return visited stage names."""

        # In a future commit this will perform `await agent.call()` etc.
        # For now we just run synchronously inside the event loop.
        return self._delegate.run(max_hops=max_hops, context=context)


try:
    import jsonschema
    from functools import lru_cache
    from pathlib import Path

    _SCHEMA_PATH = Path(__file__).resolve().parents[3] / "schemas" / "execution_dsl.json"

    @lru_cache(maxsize=1)
    def _load_schema() -> dict:  # pragma: no cover
        with _SCHEMA_PATH.open("r", encoding="utf-8") as fp:
            import json

            return json.load(fp)

    def _validate_dsl(data: dict) -> None:  # pragma: no cover
        schema = _load_schema()
        jsonschema.validate(instance=data, schema=schema)


except ModuleNotFoundError:  # jsonschema not available – skip runtime validation

    def _validate_dsl(data: dict) -> None:  # type: ignore[override]
        return 