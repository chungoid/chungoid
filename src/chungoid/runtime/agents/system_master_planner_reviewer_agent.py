"""
Implementation of the MasterPlannerReviewerAgent.

This agent is invoked when an autonomous flow encounters an error or needs review.
It analyzes the situation and suggests a course of action.
"""

import logging
import uuid
from typing import Dict, Any, Optional
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    ReviewerActionType
)
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

class MasterPlannerReviewerAgent:
    """
    System agent responsible for reviewing failed/paused MasterExecutionPlans and suggesting next steps.
    """
    AGENT_ID = "system.master_planner_reviewer_agent"
    AGENT_NAME = "Master Planner Reviewer Agent"
    AGENT_DESCRIPTION = ("Reviews failed or paused autonomous execution plans and suggests recovery actions, "
                       "such as retrying a stage, modifying the plan, or escalating to a user.")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"MasterPlannerReviewerAgent initialized with config: {self.config}")

    def get_agent_card(self) -> AgentCard:
        """Returns the AgentCard definition for this agent."""
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            capabilities=[
                "master_plan_review",
                "error_analysis",
                "recovery_suggestion",
                "flow_escalation"
            ],
            input_schema=MasterPlannerReviewerInput.model_json_schema(),
            output_schema=MasterPlannerReviewerOutput.model_json_schema(),
            tags=["system", "planner", "reviewer", "error-handling", "autonomous-flow"],
            # mcp_tool_input_schemas: if this agent exposes tools, define them here
        )

    async def invoke_async(
        self, 
        inputs: MasterPlannerReviewerInput, 
        full_context: Optional[Dict[str, Any]] = None # Full orchestrator context if available
    ) -> MasterPlannerReviewerOutput:
        """
        Main invocation point for the agent.
        Receives the current plan, pause/error details, and full context.
        Returns a suggested action.
        """
        logger.info(f"MasterPlannerReviewerAgent invoked. Pause Status: {inputs.pause_status.value}, Paused Stage: {inputs.paused_stage_id}")
        logger.debug(f"Reviewer Input: {inputs.model_dump_json(indent=2)}")
        if full_context:
            logger.debug(f"Reviewer Full Context Keys: {list(full_context.keys())}")

        error_details = inputs.triggering_error_details
        current_plan = inputs.current_master_plan
        paused_stage_id = inputs.paused_stage_id
        # context_at_pause = inputs.full_context_at_pause # Available if needed

        suggestion_type = ReviewerActionType.ESCALATE_TO_USER # Default
        suggestion_details = {
            "message_to_user": f"Flow paused at stage '{paused_stage_id}' due to '{inputs.pause_status.value}'. No specific recovery suggestion identified.",
            "original_error": error_details.model_dump() if error_details else None
        }
        reasoning = "Default response: Escalating to user as no specific recovery heuristic matched."
        confidence = 0.2

        if error_details:
            error_msg = error_details.message.lower() if error_details.message else ""
            error_type = error_details.error_type.lower() if error_details.error_type else ""

            # Heuristic 1: Simple retry for "transient" sounding errors if retry count is low
            # This needs access to retry counts for the stage, which isn't directly in MasterPlannerReviewerInput.
            # For now, let's assume a simple case. The orchestrator might manage actual retry counts.
            if "timeout" in error_msg or "transient" in error_msg or "network issue" in error_msg:
                suggestion_type = ReviewerActionType.RETRY_STAGE_AS_IS
                suggestion_details = {"target_stage_id": paused_stage_id}
                reasoning = "Error message suggests a transient issue. Suggesting a simple retry."
                confidence = 0.6
                # In a more advanced version, we'd check stage retry counts from context_at_pause or plan status.

            # Heuristic 2: Input validation error - suggest modifying input
            # This is a common pattern.
            elif "validationerror" in error_type or "invalid input" in error_msg or "parameter missing" in error_msg:
                suggestion_type = ReviewerActionType.RETRY_STAGE_WITH_MODIFIED_INPUT
                # For now, we can't concretely suggest *what* input to modify without more context
                # or LLM capabilities. So, we highlight the need for input modification.
                suggestion_details = {
                    "target_stage_id": paused_stage_id,
                    "modification_needed": "Review and correct inputs for the failed stage.",
                    "original_error_summary": f"{error_type}: {error_msg[:100]}"
                }
                reasoning = "Error suggests an input validation issue. Stage may need corrected inputs to proceed."
                confidence = 0.5
                # Future: Could try to extract problematic fields if error message is structured.

            # Heuristic 3: Agent not found or configuration error
            elif "agentnotfound" in error_type or "config error" in error_msg:
                suggestion_type = ReviewerActionType.MODIFY_MASTER_PLAN
                suggestion_details = {
                    "target_stage_id": paused_stage_id,
                    "suggested_plan_change": f"Agent '{error_details.agent_id}' for stage '{paused_stage_id}' might be missing or misconfigured. Consider replacing the agent or correcting its configuration in the plan.",
                    "original_error_summary": f"{error_type}: {error_msg[:100]}"
                }
                reasoning = "Error indicates a problem with agent resolution or configuration. Plan modification might be needed."
                confidence = 0.45

            # Heuristic 4: Unmet success criteria
            elif inputs.pause_status == FlowPauseStatus.SUCCESS_CRITERIA_FAILED:
                suggestion_type = ReviewerActionType.ESCALATE_TO_USER # Often needs human judgment
                suggestion_details = {
                     "message_to_user": f"Stage '{paused_stage_id}' completed but failed its success criteria. Please review the stage output and plan.",
                     "original_error": error_details.model_dump() if error_details else {"summary": "Success criteria failed."}
                }
                reasoning = "Stage completed but did not meet defined success criteria. Human review is advised."
                confidence = 0.7 # High confidence in escalating this

            # Heuristic 5: Critical, unrecoverable errors (example)
            elif "critical" in error_msg or "unrecoverable" in error_type:
                suggestion_type = ReviewerActionType.ESCALATE_TO_USER # Defaulting to user for abort.
                                                                    # Could also be NO_ACTION_SUGGESTED if we want orchestrator to abort directly.
                suggestion_details = {
                    "message_to_user": f"Critical unrecoverable error encountered at stage '{paused_stage_id}'. Recommend aborting flow. Error: {error_type} - {error_msg}",
                    "recommend_abort": True,
                    "original_error": error_details.model_dump() if error_details else None
                }
                reasoning = "Error appears critical and unrecoverable. Escalating to user with recommendation to abort."
                confidence = 0.8
        
        elif inputs.pause_status == FlowPauseStatus.CLARIFICATION_NEEDED:
            suggestion_type = ReviewerActionType.ESCALATE_TO_USER
            suggestion_details = {
                "message_to_user": f"Flow paused at stage '{paused_stage_id}' for user clarification. Please provide the requested input.",
                 "clarification_request_details": inputs.paused_run_details.get("clarification_request") 
            }
            reasoning = "Flow is paused awaiting user clarification as per plan design."
            confidence = 0.95


        # TODO: Add more heuristics:
        # - Check for stages that are marked as "optional" in the plan (if such a field exists).
        #   If an optional stage fails, suggest ReviewerActionType.PROCEED_AS_IS.
        # - Analyze current_master_plan structure for dependencies.
        # - If an agent can provide structured output for "what went wrong", use that.
        # - Consider using an LLM for more nuanced suggestions if configured.

        logger.info(f"Reviewer suggesting: {suggestion_type.value}, Reasoning: {reasoning}")
        return MasterPlannerReviewerOutput(
            suggestion_type=suggestion_type,
            suggestion_details=suggestion_details,
            reasoning=reasoning,
            confidence_score=confidence
        )

    # Sync invoke for simpler testing or if used in a sync context
    def invoke(
        self, 
        inputs: MasterPlannerReviewerInput, 
        full_context: Optional[Dict[str, Any]] = None
    ) -> MasterPlannerReviewerOutput:
        logger.info("MasterPlannerReviewerAgent (sync invoke) called.")
        # For now, sync invoke just calls the async one and waits if in a test or simple script.
        # In a real async environment, this shouldn't block. 
        # This is a simplification for now; a true sync version might be needed or this method removed.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # This is tricky if called from a running loop that isn't this one's.
                # For simple cases or tests, asyncio.run_coroutine_threadsafe might be an option
                # or just raising an error that sync invoke from running loop is not supported directly.
                # For now, let's assume if loop is running, we might be in a test that can handle it.
                # Or this agent is always expected to be called via its async interface by the orchestrator.
                logger.warning("Sync invoke called from a running event loop. This might lead to issues.")
                # Fallback to a simple call, hoping for the best in test scenarios.
                # This is NOT robust for general purpose sync calls from async contexts.
                future = asyncio.run_coroutine_threadsafe(self.invoke_async(inputs, full_context), loop)
                return future.result(timeout=30) # Add a timeout
            else:
                return asyncio.run(self.invoke_async(inputs, full_context))
        except RuntimeError as e:
            # RuntimeError: asyncio.run() cannot be called from a running event loop
            # This can happen in tests using pytest-asyncio if not careful
            logger.error(f"RuntimeError in sync invoke, likely due to asyncio loop state: {e}. Re-trying with direct await if possible or simple call.")
            # This fallback is mostly for making some tests pass easily if they call sync invoke.
            # It's not a good general pattern.
            # If in an async context already, one should await invoke_async.
            # This sync invoke is more for cases where there is NO loop or it's a separate thread's loop.
            # The orchestrator should always use invoke_async.
            
            # Simplistic fallback for common test pattern where a test function is async but calls sync
            # This is hacky
            async def run_it(): return await self.invoke_async(inputs, full_context)
            try: # Python 3.7+ has asyncio.get_running_loop()
                asyncio.get_running_loop() # Check if loop is running
                # If we are here, it means a loop is running. Awaiting directly might work if the caller is async. 
                # But this is a sync function. This path is problematic.
                logger.warning("Sync invoke called from within an async function with a running loop. This is bad practice. invoke_async should be used.")
                # Attempting a crude execution for tests, not for production
                # This will likely fail or behave unexpectedly in many async scenarios.
                # Create a new loop just for this if no loop is running or if the current one is problematic.
                # However, creating nested event loops is generally an anti-pattern.
                # The best solution is for the caller to use invoke_async if in an async context.
                # For now, let's just log and return a default error to highlight the issue.
                return MasterPlannerReviewerOutput(
                    suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
                    reasoning=f"Sync invoke called in problematic async state ({e}). Please use invoke_async.",
                    confidence_score=0.0
                )
            except RuntimeError: # No loop running
                return asyncio.run(run_it())
        except Exception as e:
            logger.exception(f"Exception in MasterPlannerReviewerAgent sync invoke: {e}")
            return MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
                reasoning=f"Failed during sync execution: {e}",
                confidence_score=0.0
            )

# It's good practice to have a way to easily get the card or register the agent.
# This might be done in an __init__.py of the agents module or a dedicated registry script.

def get_agent_card_static() -> AgentCard:
    """Static method to get the agent card without instantiating the agent."""
    return MasterPlannerReviewerAgent().get_agent_card() # Instantiate briefly to get card

# Example of how it might be registered (conceptual)
# from chungoid.utils.agent_registry import default_agent_registry
# if __name__ == '__main__':
#     agent = MasterPlannerReviewerAgent()
#     default_agent_registry.add(agent.get_agent_card(), overwrite=True)
#     print(f"Agent {agent.AGENT_ID} card registered/updated.")

# Example Usage (for direct testing):
if __name__ == '__main__':
    from chungoid.schemas.errors import AgentErrorDetails
    import json

    # Create a dummy error detail
    err_details = AgentErrorDetails(
        error_type="TestError",
        message="This is a test error for the reviewer agent.",
        agent_id="test_agent",
        stage_id="stage_alpha"
    )

    # Create dummy input
    reviewer_input = MasterPlannerReviewerInput(
        original_goal="Test the MasterPlannerReviewerAgent",
        failed_master_plan_json=json.dumps({"id": "plan_123", "start_stage": "stage_alpha", "stages": {"stage_alpha": {"agent_id": "test_agent"}}}),
        failed_stage_id="stage_alpha",
        error_details=err_details,
        full_context_snapshot={"global_var": "hello", "outputs": {"prev_stage": {"data": "some_output"}}}
    )

    # Initialize and invoke the agent
    agent = MasterPlannerReviewerAgent()
    output = agent.invoke(reviewer_input)

    print("Reviewer Agent Output:")
    print(output.model_dump_json(indent=2)) 