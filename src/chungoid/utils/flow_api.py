from fastapi import APIRouter, Body, Header, HTTPException, Depends
import os
from pathlib import Path
from chungoid.runtime.orchestrator import ExecutionPlan, AsyncOrchestrator
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.state_manager import StateManager
from chungoid.utils.metrics_store import MetricsStore
from .flow_registry import FlowRegistry
from .flow_registry_singleton import _flow_registry

def get_agent_provider() -> AgentProvider:
    return AgentProvider()

def get_state_manager() -> StateManager:
    return StateManager(project_root=Path.cwd())

def get_metrics_store() -> MetricsStore:
    return MetricsStore(project_root=Path.cwd())

def get_config() -> dict:
    return {"logging": {"level": os.getenv("LOG_LEVEL", "INFO")}}

def get_router(api_key_checker):
    router = APIRouter()

    @router.post("/run/{flow_id}", tags=["flow"])
    async def run_flow(
        flow_id: str,
        context: dict = Body(default_factory=dict),
        x_api_key: str | None = Header(None, alias="X-API-Key"),
        agent_provider: AgentProvider = Depends(get_agent_provider),
        state_manager: StateManager = Depends(get_state_manager),
        metrics_store: MetricsStore = Depends(get_metrics_store),
        config: dict = Depends(get_config),
    ):
        """Execute a flow by ID with optional context (input, user, etc)."""
        api_key_checker(x_api_key)
        card = _flow_registry.get(flow_id)
        if not card:
            raise HTTPException(status_code=404, detail=f"Flow {flow_id} not found")
        plan = ExecutionPlan.from_yaml(card.yaml_text, flow_id=flow_id)
        orch = AsyncOrchestrator(
            config=config, 
            agent_provider=agent_provider, 
            state_manager=state_manager,
            metrics_store=metrics_store,
            master_planner_reviewer_agent_id=None
        )
        result_context = await orch.run(plan=plan, context=context)
        return {"status": "ok", "result": result_context}

    return router 