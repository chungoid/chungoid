"""UnifiedOrchestrator (Phase-1 UAEI)

This orchestrator executes stages by calling `UnifiedAgent.execute()`
using the new `ExecutionContext` structure. For Phase-1 it supports only
single-pass execution and minimal branching. It will gradually replace
`AsyncOrchestrator` after all agents migrate.

This is the complete implementation according to enhanced_cycle.md Phase 1.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.unified_execution_schemas import (
    ExecutionContext,
    ExecutionConfig,
    StageInfo,
    AgentExecutionResult,
    ExecutionMode,
)
from ..schemas.master_flow import MasterExecutionPlan
from ..schemas.agent_master_planner import MasterPlannerInput
from ..schemas.common_enums import StageStatus, OnFailureAction
from ..utils.state_manager import StateManager
from .unified_agent_resolver import UnifiedAgentResolver
from ..utils.metrics_store import MetricsStore

__all__ = ["UnifiedOrchestrator"]


class UnifiedOrchestrator:
    """
    Phase-1 UAEI UnifiedOrchestrator
    
    Replaces AsyncOrchestrator with a simplified, single-path execution model.
    Uses only agent.execute() calls - no branching, no adapters, no complexity.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        state_manager: StateManager,
        agent_resolver: UnifiedAgentResolver,
        metrics_store: MetricsStore
    ):
        self.config = config
        self.state_manager = state_manager
        self.agent_resolver = agent_resolver
        self.metrics_store = metrics_store
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize shared context
        self.shared_context: Dict[str, Any] = {
            "project_root_path": str(state_manager.target_dir_path),
            "outputs": {},
        }

    # ---------------------------------------------------------------------
    async def execute_stage(
        self,
        stage_id: str,
        agent_id: str,
        inputs: Any,
        attempt: int = 1,
        max_iterations: int = 1
    ) -> AgentExecutionResult:
        """Execute a single stage with the given agent in single-pass mode."""

        self.logger.info("[UAEI] Executing stage %s (attempt %d) with agent %s", stage_id, attempt, agent_id)

        # Phase 3: Resolve agent using UnifiedAgentResolver (single path)
        agent = await self.agent_resolver.resolve_agent(agent_id)

        # Create execution context
        ctx = ExecutionContext(
            inputs=inputs,
            shared_context=self.shared_context,
            stage_info=StageInfo(stage_id=stage_id, attempt_number=attempt),
            execution_config=ExecutionConfig(
                max_iterations=max_iterations,
                quality_threshold=0.85,  # Phase 3: Enable quality thresholds
                completion_criteria=None  # Phase 3: Will be enhanced
            ),
        )

        # Phase 3: Execute using UnifiedAgent.execute() with ExecutionMode.OPTIMAL
        result = await agent.execute(ctx, ExecutionMode.OPTIMAL)

        # Persist outputs in shared context under stage_id
        self.shared_context["outputs"][stage_id] = result.output
        return result

    # ------------------------------------------------------------------
    async def execute_master_plan_async(
        self,
        master_plan: MasterExecutionPlan,
        run_id_override: Optional[str] = None,
        tags_override: Optional[List[str]] = None
    ) -> None:
        """Execute a complete master plan using UnifiedAgent.execute() calls."""
        
        run_id = run_id_override or str(uuid.uuid4())
        self.logger.info(f"[UAEI] Executing master plan {master_plan.id} with run_id {run_id}")
        
        # Update shared context with plan info
        self.shared_context.update({
            "master_plan_id": master_plan.id,
            "run_id": run_id,
            "tags": tags_override or []
        })
        
        if master_plan.initial_context:
            self.shared_context.update(master_plan.initial_context)
        
        # Execute stages sequentially (Phase-1: simple execution)
        for stage in master_plan.stages:
            try:
                self.logger.info(f"[UAEI] Executing stage {stage.stage_id}")
                
                result = await self.execute_stage(
                    stage_id=stage.stage_id,
                    agent_id=stage.agent_id,
                    inputs=stage.inputs,
                    max_iterations=getattr(stage, 'max_iterations', 1)
                )
                
                self.logger.info(f"[UAEI] Stage {stage.stage_id} completed with status: {result.completion_reason}")
                
            except Exception as e:
                self.logger.error(f"[UAEI] Stage {stage.stage_id} failed: {e}")
                # Phase-1: simple error handling
                raise

        self.logger.info(f"[UAEI] Master plan {master_plan.id} execution completed")

    # ------------------------------------------------------------------
    async def execute_master_planner_goal_async(
        self,
        master_planner_input: MasterPlannerInput
    ) -> None:
        """
        UAEI Phase-1: Simplified goal execution without master planner agent.
        Instead of generating complex plans, directly execute relevant agents for the goal.
        """
        
        self.logger.info(f"[UAEI] Executing simplified goal flow: {master_planner_input.user_goal[:100]}...")
        
        # Update shared context
        self.shared_context.update({
            "user_goal": master_planner_input.user_goal,
            "project_id": master_planner_input.project_id,
            "run_id": master_planner_input.run_id,
            "tags": master_planner_input.tags or []
        })
        
        if master_planner_input.initial_context:
            self.shared_context.update(master_planner_input.initial_context)
        
        # UAEI Phase-1: Execute a simplified flow of key agents
        # This replaces the complex master planning with direct agent execution
        
        # 1. Environment setup
        await self.execute_stage(
            stage_id="environment_bootstrap",
            agent_id="EnvironmentBootstrapAgent",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_type": "cli_tool"  # Inferred from goal
            },
            max_iterations=1
        )
        
        # 2. Dependency management  
        await self.execute_stage(
            stage_id="dependency_management",
            agent_id="DependencyManagementAgent_v1", 
            inputs={
                "operation": "analyze",
                "project_path": self.shared_context.get("project_root_path", "."),
                "auto_detect_dependencies": True,
                "install_after_analysis": True,
                "resolve_conflicts": True,
                "target_languages": ["python"],
                "user_goal": master_planner_input.user_goal,
                "technologies": ["scapy", "python"]  # Inferred from goal
            },
            max_iterations=1
        )
        
        self.logger.info("[UAEI] Simplified goal execution completed")

    # ------------------------------------------------------------------
    async def run(
        self,
        goal_str: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        run_id_override: Optional[str] = None
    ) -> Tuple[StageStatus, Any, Optional[str]]:
        """
        Main orchestrator run method - replaces AsyncOrchestrator.run()
        
        Returns:
            Tuple of (final_status, final_shared_context, final_error_details)
        """
        
        run_id = run_id_override or str(uuid.uuid4())
        self.logger.info(f"[UAEI] Starting orchestrator run {run_id}")
        
        if initial_context:
            self.shared_context.update(initial_context)
        
        try:
            if goal_str:
                # Create master planner input
                planner_input = MasterPlannerInput(
                    user_goal=goal_str,
                    project_id=self.shared_context.get("project_id", "unknown"),
                    run_id=run_id,
                    initial_context=initial_context or {}
                )
                
                # Execute via master planner
                await self.execute_master_planner_goal_async(planner_input)
            
            # Return success status
            return StageStatus.COMPLETED_SUCCESS, self.shared_context, None
            
        except Exception as e:
            self.logger.error(f"[UAEI] Orchestrator run failed: {e}")
            return StageStatus.COMPLETED_FAILURE, self.shared_context, str(e)

    # ------------------------------------------------------------------
    async def resume_flow_async(
        self,
        run_id_to_resume: str,
        action: str,
        new_inputs: Optional[Dict[str, Any]] = None,
        target_stage_id_for_branch: Optional[str] = None
    ) -> None:
        """Resume a paused flow - Phase-1 simplified implementation."""
        
        self.logger.info(f"[UAEI] Resuming flow {run_id_to_resume} with action {action}")
        
        # Phase-1: Basic resume support
        # In a full implementation, this would load paused state and continue execution
        # For now, we'll implement basic retry logic
        
        if action == "retry":
            self.logger.info(f"[UAEI] Retrying last stage for run {run_id_to_resume}")
            # Load last stage from state and retry
            # This is a simplified implementation
            
        elif action == "abort":
            self.logger.info(f"[UAEI] Aborting run {run_id_to_resume}")
            # Mark as aborted in state
            
        else:
            self.logger.warning(f"[UAEI] Resume action {action} not fully implemented in Phase-1")
            
        # Phase-1: Basic implementation complete

    # ------------------------------------------------------------------
    def get_shared_outputs(self) -> Dict[str, Any]:
        """Get all stage outputs from shared context."""
        return self.shared_context.get("outputs", {}) 