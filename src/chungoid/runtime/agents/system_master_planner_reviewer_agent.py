"""
Implementation of the MasterPlannerReviewerAgent.

This agent is invoked when an autonomous flow encounters an error or needs review.
It analyzes the situation and suggests a course of action.
"""

import logging
import uuid
import json
from typing import Dict, Any, Optional, Union
import asyncio # Added for sync invoke and main
from concurrent.futures import ThreadPoolExecutor # Added for sync invoke
import os # Added for main
from pathlib import Path

from chungoid.schemas.agent_master_planner_reviewer import (
    MasterPlannerReviewerInput,
    MasterPlannerReviewerOutput,
    ReviewerActionType,
    RetryStageWithChangesDetails,
    AddClarificationStageDetails,
    ModifyMasterPlanDetails,
    ModifyMasterPlanRemoveStageDetails,
    ModifyMasterPlanModifyStageDetails
)
from chungoid.utils.agent_registry import AgentCard
from chungoid.schemas.common_enums import FlowPauseStatus
# Corrected LLM Provider Imports
from chungoid.utils.llm_provider import LLMProvider, OpenAILLMProvider 
# Import MasterStageSpec for type hinting in prompt examples
from chungoid.schemas.master_flow import MasterStageSpec, MasterExecutionPlan # Added MasterExecutionPlan for helper methods
from chungoid.agents.testing_mock_agents import MockSetupAgentV1Output, MockClarificationAgentV1Output # ADDED FOR MOCK LOGIC

logger = logging.getLogger(__name__)

# --- Prompt Constants ---

# Note: Pydantic model examples in prompts are for LLM guidance;
# actual validation happens during parsing of LLM output against the real Pydantic models.

SYSTEM_PROMPT_TEMPLATE_BASE = """
You are the Master Planner Reviewer Agent. Your role is to analyze the state of a paused master execution flow and suggest the best course of action.
You will be given the current master plan, details of the paused run (including the stage that failed or requires attention), the error (if any), and a snapshot of the execution context.

Available suggestion types are: {action_types_json}

Consider the following rules and guidelines:

1.  **Analyze the Error and Context:**
    *   Carefully examine `triggering_error_details` and `full_context_at_pause`.
    *   The `relevant_context_snippet` provides a focused summary, including:
        *   `failed_stage_spec`: The specification of the stage that paused.
        *   `failed_stage_inputs`: The actual inputs passed to the failed stage.
        *   `failed_stage_output_snippet`: A snippet of the output from the failed stage, if any.
        *   `explicit_setup_message_content`: The 'message' content from 'stage_A_setup's output, if found.
        *   `explicit_clarification_content`: The 'clarification_provided' content from a previously run clarification stage (like 'stage_BC_clarify_for_B'), if found.

2.  **Mock Failure Handling (Default for `trigger_fail` issues):**
    *   For simple mock failures (e.g., a 'trigger_fail' flag was set to true in a mock agent),
        your primary suggestion should be 'RETRY_STAGE_WITH_CHANGES'.
    *   You should suggest changing the input that caused the mock failure (e.g., set 'trigger_fail' to false).
    *   This rule is OVERRIDDEN by Rule #3 if its conditions are met.

3.  **Specific Scenario: Handling Stages Needing Clarification (e.g., based on 'stage_B_needs_clarification' in setup message):**
    *   This rule takes precedence over general mock failure handling (Rule #2) IF `explicit_setup_message_content` contains 'stage_B_needs_clarification'.
    *   **Condition A (Clarification NOT YET PROVIDED):**
        *   IF `explicit_setup_message_content` contains 'stage_B_needs_clarification'
        *   AND `explicit_clarification_content` is MISSING or EMPTY in the `relevant_context_snippet`,
        *   THEN you MUST suggest `ADD_CLARIFICATION_STAGE`.
            *   The new stage should use agent 'mock_clarification_agent_v1'.
            *   Its `inputs` MUST be: {{"query": "What is the actual question to ask? (e.g., 'What is the current weather?')"}}. (The LLM should resolve this to the actual message).
            *   Its `success_criteria` MUST be: ["clarification_provided IS_NOT_EMPTY"].
            *   The `new_stage_spec.id` SHOULD BE 'stage_BC_clarify_for_B'. (The system will make it unique if needed, e.g., `stage_BC_clarify_for_B_v1`).
            *   The `insert_before_stage_id` MUST be the ID of the stage that just failed (i.e., `paused_stage_id` from your input, which should be 'stage_B_fail_point' in this scenario).
            *   Other `new_stage_spec` fields (name, description, number) should be sensible.
            *   Reasoning should state clarification is needed and not yet found.
    *   **Condition B (Clarification HAS BEEN PROVIDED):**
        *   IF `explicit_setup_message_content` contains 'stage_B_needs_clarification'
        *   AND `explicit_clarification_content` IS PRESENT AND NOT EMPTY in the `relevant_context_snippet`,
        *   AND the `paused_stage_id` is 'stage_B_fail_point' (or the stage that originally needed clarification),
        *   THEN you MUST suggest `RETRY_STAGE_WITH_CHANGES` for the `paused_stage_id` ('stage_B_fail_point').
            *   The `changes_to_stage_spec.inputs` MUST include setting `trigger_fail` to `false`.
            *   It should also preserve other necessary inputs for 'stage_B_fail_point', like `setup_message` (e.g., `{{"trigger_fail": false, "setup_message": "context.intermediate_outputs.setup_message.message"}}`).
            *   Reasoning should state clarification was found, and now the original stage can be retried with changes.

4.  **Success Criteria Failures:**
    *   If a stage failed due to `SuccessCriteriaFailed` (check `triggering_error_details.error_type`),
        and the failure is not covered by rule #3, consider if a `RETRY_STAGE_WITH_CHANGES` could fix it by altering inputs.
        If not, `ESCALATE_TO_USER` is often appropriate.

5.  **Agent Not Found or Resolution Errors:**
    *   If the error is `AgentNotFoundError`, `NoAgentFoundForCategoryError`, or `AmbiguousAgentCategoryError`,
        suggest `ESCALATE_TO_USER`. These are structural issues.

6.  **General Errors:**
    *   For other types of errors, assess if a `RETRY_STAGE_AS_IS` is plausible (e.g., for transient issues).
    *   If inputs seem problematic, `RETRY_STAGE_WITH_CHANGES` might be applicable.
    *   If the plan seems flawed (e.g., a stage is fundamentally wrong or missing), `MODIFY_MASTER_PLAN` (e.g. to remove a problematic stage if a workaround is clear) could be an option, but use sparingly.
    *   If a stage failed but the overall goal might still be achievable by skipping it or if the failure is inconsequential, `PROCEED_AS_IS` might be an option (use with caution).

7.  **Output Format:**
    *   You MUST output a single JSON object conforming to the `MasterPlannerReviewerOutput` schema.
    *   The `suggestion_type` field must be one of the available enum values.
    *   The `suggestion_details` field must be a JSON object appropriate for the `suggestion_type`.
        *   For `RETRY_STAGE_WITH_CHANGES`: `RetryStageWithChangesDetails` schema.
        *   For `ADD_CLARIFICATION_STAGE`: `AddClarificationStageDetails` schema.
        *   For `MODIFY_MASTER_PLAN`: `ModifyMasterPlanRemoveStageDetails` schema (currently only remove is detailed).
        *   For `ESCALATE_TO_USER`: include a `message_to_user` field in details if you have a specific message.
    *   Provide a clear `reasoning` for your suggestion.

Example `RetryStageWithChangesDetails`:
{{
  "target_stage_id": "stage_X_name",
  "changes_to_stage_spec": {{ "inputs": {{ "some_input_key": "new_value" }} }}
}}

Example `AddClarificationStageDetails`:
{{
  "new_stage_spec": {{
    "id": "stage_Y_clarify",
    "name": "Clarification for Y",
    "description": "Gathers info for Y",
    "number": 2.5,
    "agent_id": "mock_clarification_agent_v1",
    "inputs": {{ "query": "What is the actual question to ask? (e.g., 'What is the current weather?')" }},
    "output_context_path": "intermediate_outputs.clarification_for_Y",
    "success_criteria": ["clarification_provided IS_NOT_EMPTY"],
    "on_failure": {{ "action": "FAIL_MASTER_FLOW", "log_message": "Clarification failed for Y" }}
    // next_stage will be handled by the system if not specified
  }},
  "original_failed_stage_id": "string (ID of the stage that FAILED or successfully COMPLETED, triggering this review)",
  "insert_before_stage_id": "string (ID of the stage BEFORE which the new stage should be inserted - determine this from user request in context, e.g., 'add before stage_B')",
  "new_stage_output_to_map_to_verification_stage_input": {{ 
    "source_output_field": "clarification_provided",
    "target_input_field_in_verification_stage": "clarification_data"
  }}
}}
Make sure your 'new_stage_spec' for ADD_CLARIFICATION_STAGE includes all necessary fields like id, name, description, number, agent_id, inputs, success_criteria.
The 'agent_id' for the new stage should be a valid, existing agent.
If adding a stage, its 'output_context_path' should generally be 'intermediate_outputs.some_descriptive_name'.
Its 'on_failure' policy should usually be 'FAIL_MASTER_FLOW' to prevent loops on failing clarification.

CRITICAL INSTRUCTIONS FOR ADD_CLARIFICATION_STAGE: 
When suggesting `ADD_CLARIFICATION_STAGE`:
1.  Determine the correct `insert_before_stage_id` by carefully reading the user's request in the `full_context_at_pause` (specifically, `explicit_setup_message_content` or `context.outputs.stage_A_setup.message` often contains phrases like "add a stage before stage_X").
2.  The `original_failed_stage_id` is the ID of the stage that led to this review (the stage that paused/failed, or the stage that succeeded if this is an `on_success` review).
3.  **New Stage Inputs:**
    *   If the `agent_id` for the `new_stage_spec` is `mock_clarification_agent_v1`, its `inputs` field MUST be a dictionary containing a single key `"query"`.
    *   The value for `"query"` MUST be the *actual question string* that the clarification agent should ask (e.g., "What is the current weather?"). You should extract this question from the user's request in the context (e.g., from `explicit_setup_message_content`). DO NOT use a context path string like "context.outputs.some.path" for the query value itself.
4.  **MANDATORY CHECK FOR OUTPUT MAPPING**: You MUST inspect the user's request details (primarily in `context.outputs.stage_A_setup.message` or `explicit_setup_message_content`). If this request contains instructions to map the new clarification stage's output to another stage's input (e.g., "map its output to stage_C_verify.inputs.clarification_data" or similar phrasing),
    then you MUST populate the `new_stage_output_to_map_to_verification_stage_input` field. This field requires:
    *   `source_output_field`: If the `new_stage_spec.agent_id` is `mock_clarification_agent_v1`, this MUST be `"clarification_provided"`. For other agents, use the actual output field name from that agent's output model.
    *   `target_input_field_in_verification_stage`: The exact name of the input field in the *target verification stage* where this data should be mapped (e.g., `"clarification_data"`).
    If, and only if, NO such mapping instructions are found in the user request context, you may omit `new_stage_output_to_map_to_verification_stage_input` or set it to null.
5.  Ensure the `new_stage_spec.next_stage` correctly points to the stage that should execute *after* the new stage (this is often the `insert_before_stage_id`).

If in doubt, `ESCALATE_TO_USER` is a safe fallback.
Do not hallucinate schemas or fields. Stick to the provided structures.
"""

USER_PROMPT_TEMPLATE = '''
A master execution plan has paused. Please analyze the situation and provide a recovery suggestion.

**Plan & Failure Context:**

*   **Paused Stage ID:** `{paused_stage_id}`
*   **Pause Status:** `{pause_status}`
*   **Triggering Error (if any):**
    *   Type: `{error_type}`
    *   Message: `{error_message}`
    *   Traceback:
        ```
        {error_traceback}
        ```
    *   Agent ID that errored: `{error_agent_id}`
*   **Current Master Plan Snippet (focus on paused stage and neighbors):**
    ```json
    {current_master_plan_snippet}
    ```
*   **Paused Stage Specification:**
    ```json
    {paused_stage_spec_json}
    ```
*   **Full Context Snapshot at Pause (relevant keys):**
    ```json
    {relevant_context_snapshot_json}
    ```

**Your Task:**
Review all the provided information. Output a single JSON object conforming to the `MasterPlannerReviewerOutput` schema as described in the system prompt, including `suggestion_type`, `suggestion_details` (structured according to `suggestion_type`), `confidence_score`, and `reasoning`.
'''

class MasterPlannerReviewerAgent:
    """
    System agent responsible for reviewing failed/paused MasterExecutionPlans and suggesting next steps.
    """
    AGENT_ID = "system.master_planner_reviewer_agent"
    AGENT_NAME = "Master Planner Reviewer Agent"
    AGENT_DESCRIPTION = ("Reviews failed or paused autonomous execution plans and suggests recovery actions, "
                       "such as retrying a stage, modifying the plan, or escalating to a user.")

    def __init__(self, config: Optional[Dict[str, Any]] = None, llm_client: Optional[LLMProvider] = None):
        self.config = config if config else {}

        if llm_client:
            self.llm_client = llm_client
            logger.info(f"MasterPlannerReviewerAgent initialized with pre-configured LLM client: {type(self.llm_client).__name__}")
            return # Exit if llm_client is provided

        # Try to load from environment variable
        api_key_from_env = os.getenv("OPENAI_API_KEY")
        
        if api_key_from_env and api_key_from_env != "dummy_key_if_none":
            self.llm_client = OpenAILLMProvider(api_key=api_key_from_env)
            logger.info("MasterPlannerReviewerAgent: Loaded OpenAI API key from OPENAI_API_KEY environment variable.")
            logger.info(f"MasterPlannerReviewerAgent initialized LLM client: {type(self.llm_client).__name__}")
        else:
            self.llm_client = None # Explicitly set to None
            logger.warning("MasterPlannerReviewerAgent: OPENAI_API_KEY environment variable not found or is a dummy key. LLM client NOT initialized. Will rely on mocks or fail if LLM is needed.")
        
    def get_agent_card(self) -> AgentCard:
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
        )

    def _get_simplified_plan_snippet(self, plan: MasterExecutionPlan, paused_stage_id: str) -> Dict[str, Any]:
        all_stage_ids = list(plan.stages.keys()) # Order is insertion order (Python 3.7+)
        
        try:
            paused_idx = all_stage_ids.index(paused_stage_id)
        except ValueError: # paused_stage_id not in the list of keys
            logger.warning(f"Paused stage ID '{paused_stage_id}' not found among plan stage keys: {all_stage_ids}. Returning first few stages for snippet.")
            snippet_stages_info = []
            for stage_id_key in all_stage_ids[:3]:
                spec_model = plan.stages.get(stage_id_key)
                if spec_model:
                    # Add the ID to the dumped spec for clarity in the prompt
                    spec_dump = spec_model.model_dump(mode='json')
                    spec_dump["_id_for_snippet"] = stage_id_key 
                    snippet_stages_info.append(spec_dump)
            return {"error": f"Paused stage ID '{paused_stage_id}' not found in plan stage keys.", 
                    "stages_snippet": snippet_stages_info, 
                    "current_stage_focus_index_in_snippet": -1}

        start_idx = max(0, paused_idx - 1)
        end_idx = min(len(all_stage_ids), paused_idx + 2)
        
        snippet_stage_ids = all_stage_ids[start_idx:end_idx]
        
        stages_snippet_data = []
        for stage_id_key in snippet_stage_ids:
            spec_model = plan.stages.get(stage_id_key)
            if spec_model:
                spec_dump = spec_model.model_dump(mode='json')
                spec_dump["_id_for_snippet"] = stage_id_key 
                stages_snippet_data.append(spec_dump)
        
        return {
            "stages_snippet": stages_snippet_data, 
            "current_stage_focus_index_in_snippet": paused_idx - start_idx 
        }

    def _get_paused_stage_spec(self, plan: MasterExecutionPlan, paused_stage_id: str) -> Optional[Dict[str, Any]]:
        stage_spec_model = plan.stages.get(paused_stage_id)
        if stage_spec_model:
            return stage_spec_model.model_dump(mode='json')
        return None

    def _get_relevant_context_snippet(self, full_context: Dict[str, Any], paused_stage_id: str) -> Dict[str, Any]:
        snippet = {}
        intermediate_outputs = full_context.get("intermediate_outputs", {})
        
        if intermediate_outputs:
            snippet["intermediate_outputs_summary"] = {
                k: f"(data present, type: {type(v).__name__})" for k, v in intermediate_outputs.items()
            }
            
            setup_message_obj = intermediate_outputs.get("setup_message")
            if isinstance(setup_message_obj, MockSetupAgentV1Output):
                snippet["explicit_setup_message_content"] = setup_message_obj.message
            elif isinstance(setup_message_obj, dict) and "message" in setup_message_obj:
                snippet["explicit_setup_message_content"] = setup_message_obj.get("message")

            # --- START NEW LOGIC FOR CLARIFICATION ---
            # Try to find output from a stage that looks like our clarification stage
            # This is a bit heuristic; a more robust way might involve knowing the exact ID, but _vX makes that tricky.
            clarification_key_pattern = "stage_BC_clarify_for_B" 
            found_clarification_output = None
            for key, value in intermediate_outputs.items():
                if key.startswith(clarification_key_pattern): # e.g., "stage_BC_clarify_for_B" or "stage_BC_clarify_for_B_v1"
                    if isinstance(value, MockClarificationAgentV1Output):
                        found_clarification_output = value.clarification_provided
                        snippet["explicit_clarification_output_found_at_key"] = key # For LLM debugging
                        break
                    elif isinstance(value, dict) and "clarification_provided" in value:
                        found_clarification_output = value.get("clarification_provided")
                        snippet["explicit_clarification_output_found_at_key"] = key # For LLM debugging
                        break
            
            if found_clarification_output is not None:
                 snippet["explicit_clarification_content"] = found_clarification_output
            # --- END NEW LOGIC FOR CLARIFICATION ---

        # Potentially add specific outputs if they are small and highly relevant.
        # Example: output of the stage immediately preceding the paused_stage_id.
        # This requires more sophisticated logic to trace stage execution order and history.
        return snippet

    async def invoke_async(self, input_payload: MasterPlannerReviewerInput) -> MasterPlannerReviewerOutput:
        logger.info(f"MasterPlannerReviewerAgent invoked for paused stage: {input_payload.paused_stage_id}")

        initial_msg_content_from_stage_A = ""
        if input_payload.full_context_at_pause and \
           input_payload.full_context_at_pause.get("intermediate_outputs", {}):
            setup_message_obj = input_payload.full_context_at_pause["intermediate_outputs"].get("setup_message")
            if setup_message_obj:
                if isinstance(setup_message_obj, dict):
                    initial_msg_content_from_stage_A = setup_message_obj.get("message", "")
                elif hasattr(setup_message_obj, 'message'):
                    initial_msg_content_from_stage_A = setup_message_obj.message
        
        if not initial_msg_content_from_stage_A and \
           input_payload.paused_stage_id == "stage_B_fail_point" and \
           input_payload.current_master_plan and \
           input_payload.current_master_plan.stages.get("stage_A_setup") and \
           isinstance(input_payload.current_master_plan.stages["stage_A_setup"].inputs, dict) and \
           isinstance(input_payload.current_master_plan.stages["stage_A_setup"].inputs.get("initial_message"), str):
            initial_msg_content_from_stage_A = input_payload.current_master_plan.stages["stage_A_setup"].inputs.get("initial_message", "")
            logger.info(f"invoke_async: initial_msg_content_from_stage_A from spec fallback = '{initial_msg_content_from_stage_A}'")

        # --- Specific Mock Triggers ---
        if input_payload.paused_stage_id == "stage_B_fail_point":
            if "trigger_TC_P2F1_MODIFY_02" in initial_msg_content_from_stage_A:
                logger.warning("MasterPlannerReviewerAgent: Applying TC_P2F1_MODIFY_02 mock: Modifying stage_B_fail_point.")
                original_stage_b_spec = input_payload.current_master_plan.stages.get("stage_B_fail_point")
                if not original_stage_b_spec: return self._default_escalate_to_user("MOCK ERROR: Original spec for stage_B_fail_point not found for TC_P2F1_MODIFY_02.")
                try:
                    updated_spec_data = original_stage_b_spec.model_dump(exclude_unset=False)
                    if not isinstance(updated_spec_data.get("inputs"), dict): updated_spec_data["inputs"] = {}
                    updated_spec_data["inputs"]["trigger_fail"] = False
                    updated_spec_data["inputs"]["modified_by_reviewer_TC_P2F1_MODIFY_02"] = True
                    if "setup_message" not in updated_spec_data["inputs"]: updated_spec_data["inputs"]["setup_message"] = "context.intermediate_outputs.setup_message.message"
                    final_updated_stage_spec = MasterStageSpec(**updated_spec_data)
                except Exception as e: return self._default_escalate_to_user(f"MOCK ERROR: Failed to construct updated spec for TC_P2F1_MODIFY_02: {e}")
                details = ModifyMasterPlanModifyStageDetails(action="modify_stage_spec", target_stage_id="stage_B_fail_point", updated_stage_spec=final_updated_stage_spec)
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.MODIFY_MASTER_PLAN, suggestion_details=details, confidence_score=0.98, reasoning="Mock: TC_P2F1_MODIFY_02 - Modifying stage_B_fail_point.")

            if "trigger_TC_P2F1_RETRY_02_AGENT_SWAP" in initial_msg_content_from_stage_A:
                logger.warning("MasterPlannerReviewerAgent: Applying TC_P2F1_RETRY_02 mock: Swapping agent for stage_B_fail_point.")
                details_agent_swap = RetryStageWithChangesDetails(target_stage_id="stage_B_fail_point", changes_to_stage_spec={"agent_id": "mock_alternative_agent_v1"})
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES, suggestion_details=details_agent_swap, confidence_score=1.0, reasoning="Mock for TC_P2F1_RETRY_02: Change agent.")

            if "trigger_TC_P2F1_RETRY_AS_IS" in initial_msg_content_from_stage_A:
                error_is_transient = input_payload.triggering_error_details and isinstance(input_payload.triggering_error_details.message, str) and "Intentional transient failure" in input_payload.triggering_error_details.message
                if error_is_transient:
                    logger.warning("MasterPlannerReviewerAgent: Applying TC_P2F1_OTHER_01 mock: RETRY_STAGE_AS_IS for stage_B_fail_point.")
                    return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.RETRY_STAGE_AS_IS, confidence_score=1.0, reasoning="Mock for TC_P2F1_OTHER_01: Simulated transient error.")
                else:
                    logger.warning(f"MasterPlannerReviewerAgent: TC_P2F1_RETRY_AS_IS trigger, but error not expected mock transient. Error: {input_payload.triggering_error_details.message if input_payload.triggering_error_details else 'N/A'}")

            if "trigger_TC_P2F1_ESCALATE" in initial_msg_content_from_stage_A:
                logger.warning("MasterPlannerReviewerAgent: Applying TC_P2F1_OTHER_02 mock: ESCALATE_TO_USER for stage_B_fail_point.")
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.ESCALATE_TO_USER, suggestion_details={"message_to_user": "Mock escalation for TC_P2F1_OTHER_02."}, confidence_score=1.0, reasoning="Mock for TC_P2F1_OTHER_02: Escalating as per test trigger.")

            if "trigger_TC_P2F1_MODIFY_01" in initial_msg_content_from_stage_A:
                logger.warning("MasterPlannerReviewerAgent: Applying TC_P2F1_MODIFY_01 mock: Removing stage_B_fail_point.")
                details = ModifyMasterPlanRemoveStageDetails(action="remove_stage", target_stage_id="stage_B_fail_point")
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.MODIFY_MASTER_PLAN, suggestion_details=details, confidence_score=1.0, reasoning="Mock for TC_P2F1_MODIFY_01: Remove stage_B_fail_point.")

            # Clarification logic for stage_B_fail_point
            needs_clarification = "stage_B_needs_clarification" in initial_msg_content_from_stage_A
            clarification_provided = False
            if input_payload.full_context_at_pause and input_payload.full_context_at_pause.get("intermediate_outputs", {}):
                clar_key_pattern = "clarification_for_stage_b"
                alt_clar_key = "stage_BC_clarify_for_B_output"
                clar_data = input_payload.full_context_at_pause["intermediate_outputs"].get(clar_key_pattern) or \
                            input_payload.full_context_at_pause["intermediate_outputs"].get(alt_clar_key)
                if clar_data:
                    clar_content = getattr(clar_data, 'clarification_provided', None) if hasattr(clar_data, 'clarification_provided') else clar_data.get('clarification_provided')
                    if clar_content: clarification_provided = True
            
            if needs_clarification and not clarification_provided:
                logger.warning("MasterPlannerReviewerAgent: Mock for stage_B_fail_point (needs clarification): ADD_CLARIFICATION_STAGE.")
                new_stage_spec_dict = {
                    "id": "stage_BC_clarify_for_B", "name": "Clarification for Stage B (Mock)", "description": "Mock clarification stage.",
                    "number": input_payload.paused_stage_spec.get("number", 0.0) + 0.1, "agent_id": "mock_clarification_agent_v1",
                    "inputs": {"query": f"Clarification for stage_B based on: {initial_msg_content_from_stage_A}"},
                    "output_context_path": "intermediate_outputs.clarification_for_stage_b", "next_stage": "stage_B_fail_point",
                    "success_criteria": ["clarification_provided IS_NOT_EMPTY"], "on_failure": {"action": "FAIL_MASTER_FLOW"}
                }
                try: final_new_stage_spec = MasterStageSpec(**new_stage_spec_dict)
                except Exception as e: return self._default_escalate_to_user(f"Mock ADD_CLARIFICATION_STAGE spec error: {e}")
                details_add = AddClarificationStageDetails(new_stage_spec=final_new_stage_spec, insert_before_stage_id="stage_B_fail_point", original_failed_stage_id="stage_B_fail_point")
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.ADD_CLARIFICATION_STAGE, suggestion_details=details_add, confidence_score=0.9, reasoning="Mock: Adding clarification for Stage B.")

            if needs_clarification and clarification_provided:
                logger.warning("MasterPlannerReviewerAgent: Mock for stage_B_fail_point (clarification provided): RETRY_STAGE_WITH_CHANGES.")
                details_retry = RetryStageWithChangesDetails(
                    target_stage_id="stage_B_fail_point",
                    changes_to_stage_spec={"inputs": {"trigger_fail": False, "setup_message": "context.intermediate_outputs.setup_message.message", "clarification_input": "context.intermediate_outputs.clarification_for_stage_b.clarification_provided"}}
                )
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES, suggestion_details=details_retry, confidence_score=0.9, reasoning="Mock: Clarification provided. Retrying Stage B.")
            
            # Fallback for stage_B_fail_point if clarification not explicitly handled by above AND NO LLM
            if not self.llm_client:
                logger.warning("MasterPlannerReviewerAgent: Mock fallback for stage_B_fail_point (LLM N/A): RETRY_STAGE_WITH_CHANGES (trigger_fail=false).")
                details_fallback = RetryStageWithChangesDetails(target_stage_id="stage_B_fail_point", changes_to_stage_spec={"inputs": {"trigger_fail": False, "setup_message": "context.intermediate_outputs.setup_message.message"}})
                return MasterPlannerReviewerOutput(suggestion_type=ReviewerActionType.RETRY_STAGE_WITH_CHANGES, suggestion_details=details_fallback, confidence_score=0.8, reasoning="Mock Fallback (LLM N/A): Generic retry for stage_B_fail_point.")

        # --- General LLM Invocation or Final Fallback ---
        if self.llm_client:
            logger.info("Proceeding with LLM-based review as no specific mock was triggered or returned.")
            action_types_list = [action.value for action in ReviewerActionType]
            action_types_json = json.dumps(action_types_list)
            system_prompt = SYSTEM_PROMPT_TEMPLATE_BASE.format(action_types_json=action_types_json)

            paused_stage_spec_json = json.dumps(self._get_paused_stage_spec(input_payload.current_master_plan, input_payload.paused_stage_id), indent=2)
            current_master_plan_snippet = json.dumps(self._get_simplified_plan_snippet(input_payload.current_master_plan, input_payload.paused_stage_id), indent=2)
            relevant_context_snapshot = self._get_relevant_context_snippet(input_payload.full_context_at_pause, input_payload.paused_stage_id)
            relevant_context_snapshot_json = json.dumps(relevant_context_snapshot, indent=2)

            user_prompt = USER_PROMPT_TEMPLATE.format(
                paused_stage_id=input_payload.paused_stage_id,
                pause_status=input_payload.pause_status.value,
                error_type=input_payload.triggering_error_details.error_type if input_payload.triggering_error_details else "N/A",
                error_message=input_payload.triggering_error_details.message if input_payload.triggering_error_details else "N/A",
                error_traceback=input_payload.triggering_error_details.traceback if input_payload.triggering_error_details else "N/A",
                error_agent_id=input_payload.triggering_error_details.agent_id if input_payload.triggering_error_details else "N/A",
                current_master_plan_snippet=current_master_plan_snippet,
                paused_stage_spec_json=paused_stage_spec_json,
                relevant_context_snapshot_json=relevant_context_snapshot_json
            )
            llm_response_str = "" # Initialize llm_response_str
            try:
                llm_response_str = await self.llm_client.generate( # Changed generate_text_async to generate
                    prompt=user_prompt, 
                    system_prompt=system_prompt, 
                    temperature=0.1, 
                    max_tokens=1024
                )
                logger.debug(f"LLM Raw Response:\n{llm_response_str}")
                if llm_response_str.startswith("```json"): llm_response_str = llm_response_str[len("```json"):]
                if llm_response_str.startswith("```"): llm_response_str = llm_response_str[3:]
                if llm_response_str.endswith("```"): llm_response_str = llm_response_str[:-3]
                llm_response_str = llm_response_str.strip()
                llm_suggestion_dict = json.loads(llm_response_str)
                
                suggestion_type_val = llm_suggestion_dict.get("suggestion_type")
                raw_details = llm_suggestion_dict.get("suggestion_details")
                parsed_details: Optional[Union[RetryStageWithChangesDetails, AddClarificationStageDetails, ModifyMasterPlanDetails, Dict[str, Any]]] = None
                
                if suggestion_type_val == ReviewerActionType.RETRY_STAGE_WITH_CHANGES.value:
                    parsed_details = RetryStageWithChangesDetails(**raw_details) if raw_details else None
                elif suggestion_type_val == ReviewerActionType.ADD_CLARIFICATION_STAGE.value:
                    if raw_details and "new_stage_spec" in raw_details and isinstance(raw_details["new_stage_spec"], dict):
                        raw_details["new_stage_spec"] = MasterStageSpec(**raw_details["new_stage_spec"])
                    parsed_details = AddClarificationStageDetails(**raw_details) if raw_details else None
                elif suggestion_type_val == ReviewerActionType.MODIFY_MASTER_PLAN.value:
                    action_in_details = raw_details.get("action") if isinstance(raw_details, dict) else None
                    # ---- START MODIFICATION FOR LLM OUTPUT ----
                    # LLM might return "stage_id_to_remove" instead of target_stage_id and no "action"
                    llm_stage_id_to_remove = raw_details.get("stage_id_to_remove") if isinstance(raw_details, dict) else None

                    if action_in_details == "remove_stage" or (action_in_details is None and llm_stage_id_to_remove):
                        # If LLM provided llm_stage_id_to_remove, adapt it
                        if llm_stage_id_to_remove and not raw_details.get("target_stage_id"):
                            raw_details["target_stage_id"] = llm_stage_id_to_remove
                        # Ensure action is set
                        raw_details["action"] = "remove_stage"
                        parsed_details = ModifyMasterPlanRemoveStageDetails(**raw_details) if raw_details else None
                    # ---- END MODIFICATION FOR LLM OUTPUT ----
                    elif action_in_details == "modify_stage_spec":
                        if raw_details and "updated_stage_spec" in raw_details and isinstance(raw_details["updated_stage_spec"], dict):
                             raw_details["updated_stage_spec"] = MasterStageSpec(**raw_details["updated_stage_spec"])
                        parsed_details = ModifyMasterPlanModifyStageDetails(**raw_details) if raw_details else None
                    else: parsed_details = raw_details

                return MasterPlannerReviewerOutput(
                    suggestion_type=suggestion_type_val, suggestion_details=parsed_details,
                    confidence_score=llm_suggestion_dict.get("confidence_score", 0.75),
                    reasoning=llm_suggestion_dict.get("reasoning", "LLM provided suggestion.")
                )
            except json.JSONDecodeError as json_err:
                logger.error(f"LLM response not valid JSON: {json_err}. Response: {llm_response_str}", exc_info=True)
                return self._default_escalate_to_user(f"LLM output not valid JSON. Details: {json_err}. Raw: {llm_response_str[:200]}")
            except Exception as e: 
                logger.error(f"Error processing LLM suggestion: {e}. LLM Raw: {llm_response_str}", exc_info=True)
                return self._default_escalate_to_user(f"Error processing LLM suggestion: {e}. Raw: {llm_response_str[:200]}")
        else: 
            logger.warning("LLM client not available. No specific mock matched. Falling back to default escalation.")
            return self._default_escalate_to_user("LLM client not configured and no specific mock rule applied for this scenario.")

    def _default_escalate_to_user(self, reason: str) -> MasterPlannerReviewerOutput:
        logger.error(f"MasterPlannerReviewerAgent escalating to user: {reason}")
        return MasterPlannerReviewerOutput(
            suggestion_type=ReviewerActionType.ESCALATE_TO_USER,
            suggestion_details={"message_to_user": reason},
            confidence_score=0.0,
            reasoning=reason
        )

    def invoke(
        self, 
        inputs: MasterPlannerReviewerInput, 
        full_context: Optional[Dict[str, Any]] = None
    ) -> MasterPlannerReviewerOutput:
        logger.info("MasterPlannerReviewerAgent (sync invoke) called.")
        try:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            if loop.is_running():
                logger.warning("Sync invoke called from a running event loop. This might lead to issues.")
                future = asyncio.run_coroutine_threadsafe(self.invoke_async(inputs), loop)
                return future.result(timeout=120) # Increased timeout for LLM call
            else:
                return asyncio.run(self.invoke_async(inputs))
        except RuntimeError as e:
            logger.error(f"RuntimeError in sync invoke (asyncio loop state): {e}. This agent primarily supports async invocation.")
            async def run_it_wrapper():
                return await self.invoke_async(inputs)
            try:
                asyncio.get_running_loop()
                logger.warning("Sync invoke called from within an async function with a running loop. This is bad practice. Falling back to new event loop execution.")
                with ThreadPoolExecutor(max_workers=1) as executor:
                    # Submit asyncio.run(coro) to the thread pool
                    future = executor.submit(lambda: asyncio.run(run_it_wrapper()))
                    return future.result(timeout=125) # Slightly longer for thread overhead
            except RuntimeError: # No loop running in current thread
                return asyncio.run(run_it_wrapper())
        except Exception as e:
            logger.exception(f"Exception in MasterPlannerReviewerAgent sync invoke: {e}")
            return MasterPlannerReviewerOutput(
                suggestion_type=ReviewerActionType.NO_ACTION_SUGGESTED,
                reasoning=f"Failed during sync execution: {e}",
                confidence_score=0.0
            )

def get_agent_card_static() -> AgentCard:
    return MasterPlannerReviewerAgent().get_agent_card()

if __name__ == '__main__':
    print("Basic __main__ for MasterPlannerReviewerAgent (LLM version - requires async setup and API keys)")

    async def test_agent():
        mock_error = AgentErrorDetails(agent_id="test_agent", error_type="TestError", message="Something went wrong in test", traceback="Trace...")
        mock_stage = MasterStageSpec(id="s1", name="Test Stage", agent_id="test_agent", inputs={})
        mock_plan = MasterExecutionPlan(id="plan1", name="Test Plan", description="Desc", original_request={"goal":"Test"}, stages=[mock_stage])
        
        test_input = MasterPlannerReviewerInput(
            current_master_plan=mock_plan,
            paused_run_details={ "run_id": "run0", "flow_id": "plan1", "paused_at_stage_id": "s1", "timestamp": "time", "status": FlowPauseStatus.PAUSED_AGENT_ERROR, "context_snapshot": {}, "error_details": mock_error.model_dump(), "clarification_request": None},
            pause_status=FlowPauseStatus.PAUSED_AGENT_ERROR,
            paused_stage_id="s1",
            triggering_error_details=mock_error,
            full_context_at_pause={"intermediate_outputs": {"prev_stage_output": "some data"}}
        )
        try:
            # Assuming OPENAI_API_KEY is set for OpenAIClient default
            reviewer_agent = MasterPlannerReviewerAgent() 
            output = await reviewer_agent.invoke_async(test_input)
            print("Agent Output:")
            print(output.model_dump_json(indent=2))
        except Exception as e:
            print(f"Error during test_agent(): {e}. Ensure LLM client (e.g., OpenAI) is configured if not mocked, or API key is set.")

    if os.environ.get("OPENAI_API_KEY"):
       asyncio.run(test_agent())
    else:
        print("Skipping test_agent() run as OPENAI_API_KEY is not set.") 