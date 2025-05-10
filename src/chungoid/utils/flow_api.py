from fastapi import APIRouter, Body, Header, HTTPException
from pathlib import Path
from chungoid.runtime.orchestrator import ExecutionPlan, AsyncOrchestrator
from .flow_registry import FlowRegistry
from .flow_registry_singleton import _flow_registry  # Use the shared registry

def get_router(api_key_checker):
    router = APIRouter()

    @router.post("/run/{flow_id}", tags=["flow"])
    async def run_flow(
        flow_id: str,
        context: dict = Body(default_factory=dict),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
    ):
        """Execute a flow by ID with optional context (input, user, etc)."""
        api_key_checker(x_api_key)
        card = _flow_registry.get(flow_id)
        if not card:
            raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")
        plan = ExecutionPlan.from_yaml(card.yaml_text, flow_id=flow_id)
        orch = AsyncOrchestrator(plan)
        visited = await orch.run(context=context)
        return {"visited": visited, "flow_id": flow_id}

    return router 