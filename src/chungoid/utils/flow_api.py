from fastapi import APIRouter, Body, Header, HTTPException, Depends
import os
from pathlib import Path
from chungoid.runtime.orchestrator import AsyncOrchestrator
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
        
        # The card.yaml_text contains the plan.
        # AsyncOrchestrator.run can take goal_str or flow_yaml_path.
        # We don't have a path, but we have the content.
        # We'll need to either save card.yaml_text to a temp file and pass the path,
        # or modify AsyncOrchestrator.run to accept yaml_content directly,
        # or (simplest for now) assume that if a flow_id is passed to run_flow,
        # it corresponds to a master_plan_id that the orchestrator can load.
        # For now, let's assume the orchestrator's run method can handle a "goal_str" which could be the yaml text content for simplicity,
        # or better, it should handle master_plan_id directly.
        # The AsyncOrchestrator's run method currently accepts flow_yaml_path or master_plan_id or goal_str.
        # Since we have flow_id, we can treat it as master_plan_id.

        # If card.yaml_path is available and preferred:
        # flow_path_to_pass = card.yaml_path 
        # Otherwise, if we must use master_plan_id:
        master_plan_id_to_pass = flow_id

        orch = AsyncOrchestrator(
            config=config, 
            agent_provider=agent_provider, 
            state_manager=state_manager,
            metrics_store=metrics_store,
            master_planner_reviewer_agent_id="system.master_planner_reviewer_agent_v1" # Use a placeholder string ID
        )
        # result_context = await orch.run(plan=plan, context=context) # OLD WAY
        result_context = await orch.run(master_plan_id=master_plan_id_to_pass, initial_context=context) # NEW WAY
        return {"status": "ok", "result": result_context}

    return router 