import logging
import uuid
from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    ReviewerActionType
)
from chungoid.utils.agent_registry import AgentCard

logger = logging.getLogger(__name__)

AGENT_ID = "system.master_planner_reviewer"
AGENT_DESCRIPTION = (
    "System agent that reviews failed autonomous execution plan stages, analyzes the error, "
    "and suggests a course of action (e.g., retry, modify plan, escalate to user)."
)

# AgentCard definition for registration
AGENT_CARD = AgentCard(
    agent_id=AGENT_ID,
    name="Master Planner Reviewer Agent",
    description=AGENT_DESCRIPTION,
    stage_focus="Autonomous Error Handling & Reflection",
    capabilities=[
        "Analyze autonomous execution failures",
        "Suggest recovery actions (retry, modify, escalate)",
        "Process error details and execution context"
    ],
    tags=["system", "planner", "reviewer", "error_handling", "reflection", "autonomous"],
    # For a system agent directly invoked by orchestrator, tool_names might be less relevant
    # unless it *uses* other MCP tools internally, which is not the case for the no-op version.
    tool_names=[], 
    # Schemas for direct invocation if it were a standard callable agent exposed via generic MCP tool
    # These are more for documentation/discovery if used in such a way.
    # The orchestrator will likely call its method with specific Python types.
    input_schema=MasterPlannerReviewerInput.model_json_schema(),
    output_schema=MasterPlannerReviewerOutput.model_json_schema(),
    # This agent is *called by* the orchestrator, it doesn't expose its own MCP tools in the typical sense.
    mcp_tool_input_schemas={},
    metadata={
        "version": "1.0.0",
        "layer": "ControlLayer",
        "system_critical": True,
        "invocation_method": "direct_class_method_call_by_orchestrator"
    }
)

class MasterPlannerReviewerAgent:
    """System agent responsible for reviewing failed autonomous execution plans and suggesting recovery actions."""

    def __init__(self, config: dict = None):
        self.config = config or {}
        logger.info(f"MasterPlannerReviewerAgent initialized with config: {self.config}")

    async def async_invoke(self, inputs: MasterPlannerReviewerInput) -> MasterPlannerReviewerOutput:
        """Receives details of a failed plan/stage and suggests a course of action."""
        logger.info(f"MasterPlannerReviewerAgent invoked for failed stage: {inputs.failed_stage_id}")
        logger.debug(f"Reviewer Input: {inputs.model_dump_json(indent=2)}")

        # Basic no-op logic: always escalate to user
        suggestion_id = f"suggestion_{uuid.uuid4()}"
        default_justification = (
            f"MasterPlannerReviewerAgent (no-op implementation) received error in stage '{inputs.failed_stage_id}'. "
            f"Error type: {inputs.error_details.error_type}. Escalating for user review."
        )
        
        output = MasterPlannerReviewerOutput(
            suggestion_id=suggestion_id,
            suggested_action=ReviewerActionType.ESCALATE_TO_USER,
            action_details={"message_to_user": default_justification},
            confidence_score=0.1, # Low confidence for no-op
            justification=default_justification
        )

        logger.info(f"MasterPlannerReviewerAgent suggestion: Escalate to user for stage {inputs.failed_stage_id}")
        logger.debug(f"Reviewer Output: {output.model_dump_json(indent=2)}")
        return output

    # Conventional sync invoke for easier direct testing if needed, though orchestrator uses async
    def invoke(self, inputs: MasterPlannerReviewerInput) -> MasterPlannerReviewerOutput:
        import asyncio
        return asyncio.run(self.async_invoke(inputs))

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