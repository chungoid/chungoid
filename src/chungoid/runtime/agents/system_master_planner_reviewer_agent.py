"""
Implementation of the MasterPlannerReviewerAgent.

This agent is invoked when an autonomous flow encounters an error or needs review.
It analyzes the situation and suggests a course of action.
"""

from doctest import master
from itertools import tee
import logging
import uuid
import json
from typing import ClassVar, List
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

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
from chungoid.schemas.errors import AgentErrorDetails
from chungoid.utils.agent_registry import AgentCard
from chungoid.schemas.common_enums import FlowPauseStatus
from chungoid.protocols.base.protocol_interface import ProtocolPhase
# Corrected LLM Provider Imports
from chungoid.utils.llm_provider import LLMProvider # REMOVED OpenAILLMProvider
# Import MasterStageSpec for type hinting in prompt examples
from chungoid.schemas.master_flow import MasterStageSpec, MasterExecutionPlan
from networkx import spanner # Added MasterExecutionPlan for helper methods
# from chungoid.runtime.agents.mocks.testing_mock_agents import MockSetupAgentV1Output, MockClarificationAgentV1Output # ADDED FOR MOCK LOGIC

# Registry-first architecture import
from chungoid.registry import register_system_agent

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
from chungoid.utils.agent_registry import AgentCard, AgentToolSpec # MODIFIED: Changed import path
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
# from chungoid.utils.security import is_safe_path # For path safety checks # REMOVED
from chungoid.schemas.errors import AgentErrorDetails # For structured errors
from chungoid.schemas.orchestration import SharedContext # ADDED
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.registry import register_system_agent

logger = logging.getLogger(__name__)

# --- Prompt Constants ---

# Note: Pydantic model examples in prompts are for LLM guidance;
# actual validation happens during parsing of LLM output against the real Pydantic models.

SYSTEM_PROMPT_TEMPLATE_BASE = r"""
You are tee Master spanner Reviewer Agent. Your role is to analyze the state of a paused master execution flow and suggest the best course of action.
You will be given the current master plan, details of the paused run (including the stage that failed or requires attention), the error (if any), and a snapshot of the execution context.

Available suggestion types are: {action_types_json}\n\nConsider the following rules and guidelines:\n\n1.  **Analyze the Error and Context:**
    *   Carefully examine \`triggering_error_details\` and \`full_context_at_pause\`.
    *   The \`relevant_context_snippet\` provides a focused summary, including:
        *   \\`failed_stage_spec\\`: The specification of the stage that paused.
        *   \\`failed_stage_inputs\\`: The actual inputs passed to the failed stage.
        *   \\`failed_stage_output_snippet\\`: A snippet of the output from the failed stage, if any.
        *   \\`explicit_setup_message_content\\`: The 'message' content from 'stage_A_setup\\\'s output, if found.
        *   \\`explicit_clarification_content\\`: The 'clarification_provided' content from a previously run clarification stage (like 'stage_BC_clarify_for_B'), if found.\n\n2.  **Mock Failure Handling (Default for \\`trigger_fail\\` issues):**
    *   For simple mock failures (e.g., a \\`trigger_fail\\` flag was set to true in a mock agent),
        your primary suggestion should be 'RETRY_STAGE_WITH_CHANGES'.
    *   You should suggest changing the input that caused the mock failure (e.g., set \\`trigger_fail\\` to false).
    *   This rule is OVERRIDDEN by Rule #3 if its conditions are met.\n\n3.  **Specific Scenario: Handling Stages Needing Clarification (e.g., based on \\`stage_B_needs_clarification\\` in setup message):**
    *   This rule takes precedence over general mock failure handling (Rule #2) IF \\`explicit_setup_message_content\\` contains \\`stage_B_needs_clarification\\`.
    *   **Condition A (Clarification NOT YET PROVIDED):**
        *   IF \\`explicit_setup_message_content\\` contains \\`stage_B_needs_clarification\\`
        *   AND \\`explicit_clarification_content\\` is MISSING or EMPTY in the \\`relevant_context_snippet\\`,
        *   THEN you MUST suggest \\`ADD_CLARIFICATION_STAGE\\`.
            *   The new stage should use an appropriate agent capable of user clarification (e.g., an agent categorized for \\`human_interaction\\` or \\`system_intervention\\` if suitable for plan-level clarification, such as \\`SystemInterventionAgent_v1\\` if the planner needs input from the operator).
            *   Its \\`inputs\\` MUST conform to the chosen agent\\\'s schema. For example, if using an agent like \\`SystemInterventionAgent_v1\\`, it might be: {{{{ \"prompt_message_for_user\": \"What is the actual question to ask? (e.g., \'What is the current weather?\')\" }}}}. (The LLM should resolve this to the actual message and adapt inputs for the chosen agent).
            *   Its \\`success_criteria\\` MUST be appropriate for the chosen agent (e.g., [\"human_response IS_NOT_EMPTY\"] or [\"clarification_provided IS_NOT_EMPTY\"]).
            *   The \\`new_stage_spec.id\\` SHOULD BE \\`stage_BC_clarify_for_B\\`. (The system will make it unique if needed, e.g., \\`stage_BC_clarify_for_B_v1\\`).
            *   The \\`insert_before_stage_id\\` MUST be the ID of the stage that just failed (i.e., \\`paused_stage_id\\` from your input, which should be \\`stage_B_fail_point\\` in this scenario).
            *   Other \\`new_stage_spec\\` fields (name, description, number) should be sensible.
            *   Reasoning should state clarification is needed and not yet found.
    *   **Condition B (Clarification HAS BEEN PROVIDED):**
        *   IF \\`explicit_setup_message_content\\` contains \\`stage_B_needs_clarification\\`
        *   AND \\`explicit_clarification_content\\` IS PRESENT AND NOT EMPTY in the \\`relevant_context_snippet\\`,
        *   AND the \\`paused_stage_id\\` is \\`stage_B_fail_point\\` (or the stage that originally needed clarification),
        *   THEN you MUST suggest \\`RETRY_STAGE_WITH_CHANGES\\` for the \\`paused_stage_id\\` (\\`stage_B_fail_point\\`).
            *   The \\`changes_to_stage_spec.inputs\\` MUST include setting \\`trigger_fail\\` to \\`false\\`.
            *   It should also preserve other necessary inputs for \\`stage_B_fail_point\\`, like \\`setup_message\\` (e.g., \\`{{{{ \"trigger_fail\": false, \"setup_message\": \"context.intermediate_outputs.setup_message.message\" }}}}\\`).
            *   Reasoning should state clarification was found, and now the original stage can be retried with changes.\n\n4.  **Success Criteria Failures:**
    *   If a stage failed due to \\`SuccessCriteriaFailed\\` (check \\`triggering_error_details.error_type\\`),
        and the failure is not covered by rule #3, consider if a \\`RETRY_STAGE_WITH_CHANGES\\` could fix it by altering inputs.
        If not, \\`ESCALATE_TO_USER\\` is often appropriate.\n\n5.  **Agent Not Found or Resolution Errors:**
    *   If the error is \\`AgentNotFoundError\\`, \\`NoAgentFoundForCategoryError\\`, or \\`AmbiguousAgentCategoryError\\`,
        suggest \\`ESCALATE_TO_USER\\`. These are structural issues.\n\n6.  **General Errors:**
    *   For other types of errors, assess if a \\`RETRY_STAGE_AS_IS\\` is plausible (e.g., for transient issues).
    *   If inputs seem problematic, \\`RETRY_STAGE_WITH_CHANGES\\` might be applicable.
    *   If the plan seems flawed (e.g., a stage is fundamentally wrong or missing), \\`MODIFY_MASTER_PLAN\\` (e.g. to remove a problematic stage if a workaround is clear) could be an option, but use sparingly.
    *   If a stage failed but the overall goal might still be achievable by skipping it or if the failure is inconsequential, \\`PROCEED_AS_IS\\` might be an option (use with caution).\n\n7.  **Pydantic ValidationError Handling:**
    *   If \\`triggering_error_details.error_type\\` is \\`pydantic_core._pydantic_core.ValidationError\\` or \\`ValidationError\\` (from Pydantic itself):
        *   Examine the \\`triggering_error_details.message\\`. It will list the missing or invalid fields.
        *   Your primary suggestion MUST be \\`RETRY_STAGE_WITH_CHANGES\\`.
        *   In \\`changes_to_stage_spec.inputs\\`, you MUST include ALL fields that were originally passed to the stage PLUS the fields identified as missing or needing correction from the error message.
        *   **For missing or invalid fields:**
            *   **Attempt to resolve from context:** If the correct value can be determined from \\`full_context_at_pause\\` (e.g., \\`project_id\\` might be available as \\`full_context_at_pause.data.project_id\\`, or an output from a previous stage like \\`full_context_at_pause.outputs.some_previous_stage.relevant_field_name\\`), you MUST provide the value as a context path string (e.g., \\`\"{{context.data.project_id}}\"\\` or \\`\"{{context.outputs.some_previous_stage.relevant_field_name}}\"\\`).
            *   **Provide concrete values if known:** If a field requires a specific literal string, boolean, number, etc., and you can confidently determine that value (e.g., a default \\`target_file_path\\`), provide that concrete value.
            *   **DO NOT USE VAGUE PLACEHOLDERS:** You MUST NOT use placeholders like \\`\"TODO_RESOLVE_FROM_CONTEXT_OR_USER\"\\`. These are not resolvable and will cause further errors.
            *   **If unresolvable, ESCALATE:** If a required field\\\'s value cannot be confidently determined from context or general knowledge, and it\\\'s critical for the stage to proceed, you MUST suggest \\`ESCALATE_TO_USER\\` and clearly state which field(s) are missing and why they could not be determined.
        *   Ensure the \\`inputs\\` dictionary in \\`changes_to_stage_spec\\` is FLAT, as per the GOOD/BAD examples below.
        *   Your \\`reasoning\\` should clearly state which fields were missing/invalid and how you are attempting to fix them (e.g., \\`\"Added missing 'project_id' field, resolving from '{{context.data.project_id}}'\"\\`).\n\n8.  **Output Format:**
    *   You MUST output a single JSON object conforming to the \\`MasterPlannerReviewerOutput\\` schema.
    *   The \\`suggestion_type\\` field must be one of the available enum values.
    *   The \\`suggestion_details\\` field must be a JSON object appropriate for the \\`suggestion_type\\`.
        *   For \\`RETRY_STAGE_WITH_CHANGES\\`: \\`RetryStageWithChangesDetails\\` schema.
        *   For \\`ADD_CLARIFICATION_STAGE\\`: \\`AddClarificationStageDetails\\` schema.
        *   For \\`MODIFY_MASTER_PLAN\\`: \\`ModifyMasterPlanRemoveStageDetails\\` schema (currently only remove is detailed).
        *   For \\`ESCALATE_TO_USER\\`: include a \\`message_to_user\\` field in details if you have a specific message.
    *   Provide a clear \\`reasoning\\` for your suggestion.\n\nExample \\`RetryStageWithChangesDetails\\`:\nThe \\`changes_to_stage_spec.inputs\\` field is CRITICAL. Its value MUST be a FLAT dictionary where keys are the direct input field names of the target agent, and values are their new values.\n\nBAD EXAMPLE (Causes Errors):\n{{{{ \\
  \"target_stage_id\": \"stage_X_name\",\\
  \"changes_to_stage_spec\": {{{{ \\
    \"inputs\": {{{{ \\
      \"inputs\": {{{{ \\
        \"some_input_key\": \"new_value\", \\
        \"another_key\": \"another_value\"\\
      }}}} \\
    }}}} \\
  }}}} \\
}}}}\n\nGOOD EXAMPLE (Correct Structure - FLAT inputs):\n{{{{ \\
  \"target_stage_id\": \"stage_X_name\",\\
  \"changes_to_stage_spec\": {{{{ \\
    \"inputs\": {{{{ \\
      \"some_input_key_for_agent_X\": \"new_value\", \\
      \"another_input_key_for_agent_X\": \"another_value\",\\
      \"project_id\": \"actual_project_id_if_needed\", \\
      \"task_description\": \"actual_description_if_needed\", \\
      \"target_file_path\": \"actual_path_if_needed\" \\
    }}}} \\
  }}}} \\
}}}}\n\nExample \\`AddClarificationStageDetails\\`:\n{{{{ \\
  \"new_stage_spec\": {{{{ \\
    \"id\": \"stage_Y_clarify\",\\
    \"name\": \"Clarification for Y\",\\
    \"description\": \"Gathers info for Y\",\\
    \"number\": 2.5,\\
    \"agent_id\": \"SystemInterventionAgent_v1\",\\
    \"inputs\": {{ \"prompt_message_for_user\": \"What is the actual question to ask? (e.g., \'What is the current weather?\')\" }},\\
    \"output_context_path\": \"intermediate_outputs.clarification_for_Y\",\\
    \"success_criteria\": [\"human_response IS_NOT_EMPTY\"],\\
    \"on_failure\": {{ \"action\": \"FAIL_MASTER_FLOW\", \"log_message\": \"Clarification failed for Y\" }}
    // next_stage will be handled by the system if not specified\\
  }}}},\\
  \"original_failed_stage_id\": \"string (ID of the stage that FAILED or successfully COMPLETED, triggering this review)\",\\
  \"insert_before_stage_id\": \"string (ID of the stage BEFORE which the new stage should be inserted - determine this from user request in context, e.g., \'add before stage_B\')\",\\
  \"new_stage_output_to_map_to_verification_stage_input\": {{ \\
    \"source_output_field\": \"human_response\",\\
    \"target_input_field_in_verification_stage\": \"clarification_data\"\\
  }}\\
}}}}\nMake sure your \'new_stage_spec\' for ADD_CLARIFICATION_STAGE includes all necessary fields like id, name, description, number, agent_id, inputs, success_criteria.\\
The \'agent_id\' for the new stage should be a valid, existing agent.\\
If adding a stage, its \'output_context_path\' should generally be \'intermediate_outputs.some_descriptive_name\'.\\
Its \'on_failure\' policy should usually be \'FAIL_MASTER_FLOW\' to prevent loops on failing clarification.\\n\nCRITICAL INSTRUCTIONS FOR ADD_CLARIFICATION_STAGE: \\
When suggesting \\`ADD_CLARIFICATION_STAGE\\`:\\
1.  Determine the correct \\`insert_before_stage_id\\` by carefully reading the user\\\'s request in the \\`full_context_at_pause\\` (specifically, \\`explicit_setup_message_content\\` or \\`context.outputs.stage_A_setup.message\\` often contains phrases like \\`\"add a stage before stage_X\"\\`).\\
2.  The \\`original_failed_stage_id\\` is the ID of the stage that led to this review (the stage that paused/failed, or the stage that succeeded if this is an \\`on_success\\` review).\\
3.  **New Stage Inputs:**\\
    *   If the \\`agent_id\\` for the \\`new_stage_spec\\` is, for example, \\`SystemInterventionAgent_v1\\`, its \\`inputs\\` field MUST be a dictionary containing a key like \\`\"prompt_message_for_user\"\\`. Adapt inputs to the chosen agent\\\'s schema.\\
    *   The value for this prompt/query MUST be the *actual question string* that the clarification agent should ask (e.g., \\`\"What is the current weather?\"\\`). You should extract this question from the user\\\'s request in the context (e.g., from \\`explicit_setup_message_content\\`). DO NOT use a context path string like \\`\"context.outputs.some.path\"\\` for the query value itself.\\
4.  **MANDATORY CHECK FOR OUTPUT MAPPING**: You MUST inspect the user\\\'s request details (primarily in \\`context.outputs.stage_A_setup.message\\` or \\`explicit_setup_message_content\\`). If this request contains instructions to map the new clarification stage\\\'s output to another stage\\\'s input (e.g., \\`\"map its output to stage_C_verify.inputs.clarification_data\"\\` or similar phrasing),\\
    then you MUST populate the \\`new_stage_output_to_map_to_verification_stage_input\\` field. This field requires:\\
    *   \\`source_output_field\\`: Use the actual output field name from the chosen clarification agent\\\'s output model (e.g., \\`\"human_response\"\\` for \\`SystemInterventionAgent_v1\\`, or \\`\"clarification_provided\"\\` for other hypothetical agents).\\
    *   \\`target_input_field_in_verification_stage\\`: The exact name of the input field in the *target verification stage* where this data should be mapped (e.g., \\`\"clarification_data\"\\`).\\
    If, and only if, NO such mapping instructions are found in the user request context, you may omit \\`new_stage_output_to_map_to_verification_stage_input\\` or set it to null.\\
5.  Ensure the \\`new_stage_spec.next_stage\\` correctly points to the stage that should execute *after* the new stage (this is often the \\`insert_before_stage_id\\`).\\

If in doubt, \\`ESCALATE_TO_USER\\` is a safe fallback.\\
Do not hallucinate schemas or fields. Stick to the provided structures.
"""

USER_PROMPT_TEMPLATE = """
The master execution plan has been paused. Please analyze the situation and provide a suggestion.

**Paused Stage ID:** {paused_stage_id}
**Pause Status:** {pause_status}

**Triggering Error Details (if any):**
- Error Type: {error_type}
- Agent ID: {error_agent_id}
- Message: {error_message}
- Traceback (snippet):
{error_traceback}

**Current Master Plan (Snippet focusing on paused stage and its neighbors):**
```json
{current_master_plan_snippet}
```

**Full Specification of Paused Stage ({paused_stage_id}):**
```json
{paused_stage_spec_json}
```

**Relevant Context Snapshot at Pause:**
(This includes a summary of intermediate outputs and any explicitly extracted setup/clarification messages)
```json
{relevant_context_snapshot_json}
```

Based on all the information above and the rules provided in the system prompt, please formulate your suggestion as a JSON object.
"""

@register_system_agent(capabilities=["plan_review", "error_analysis", "recovery_suggestions"])
class MasterPlannerReviewerAgent(ProtocolAwareAgent):
    '''
    System agent responsible for reviewing failed/paused MasterExecutionPlans and suggesting next steps.
    '''
    AGENT_ID: ClassVar[str] = "system.master_planner_reviewer_agent_v1"
    AGENT_NAME: ClassVar[str] = "Master Planner Reviewer Agent"
    AGENT_DESCRIPTION: ClassVar[str] = ("Reviews failed or paused autonomous execution plans and suggests recovery actions, "
                        "including retrying stages, adding clarification stages, or escalating to users.")
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    CAPABILITIES: ClassVar[List[str]] = ["plan_review", "error_analysis", "recovery_suggestions"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL
    
    # AUTONOMOUS: Protocol-aware configuration
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["plan_review", "error_recovery"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["error_recovery"]

    def __init__(self, 
                 llm_provider: LLMProvider = None,
                 prompt_manager: PromptManager = None,
                 config: Optional[Dict[str, Any]] = None, 
                 llm_client: Optional[LLMProvider] = None, 
                 llm_manager: Optional[Any] = None,
                 system_context: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        super().__init__(
            llm_provider=llm_provider or llm_client,
            prompt_manager=prompt_manager,
            agent_id=agent_id or self.AGENT_ID,
            system_context=system_context,
            **kwargs
        )
        # Store config as private attribute if needed for backward compatibility
        self._config = config if config else {}
        # Store LLM provider internally without creating a Pydantic field
        self._llm_client = llm_provider or llm_client or llm_manager # Directly assign the passed llm_client (which should be an LLMManager instance)
        # Add missing _logger attribute
        self._logger = logging.getLogger(self.AGENT_ID)

        if self._llm_client:
            logger.info(f"MasterPlannerReviewerAgent initialized with provided LLM client/manager: {type(self._llm_client).__name__}")
        elif llm_manager: # Accept llm_manager as an alternative name for dependency injection
            self._llm_client = llm_manager
            logger.info(f"MasterPlannerReviewerAgent initialized with provided llm_manager: {type(self._llm_client).__name__}")
        else:
            logger.warning("MasterPlannerReviewerAgent: No LLM client/manager provided during initialization. LLM capabilities will be unavailable.")
        
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
            if isinstance(setup_message_obj, dict) and "message" in setup_message_obj:
                snippet["explicit_setup_message_content"] = setup_message_obj.get("message")

            # --- START NEW LOGIC FOR CLARIFICATION ---
            # Try to find output from a stage that looks like our clarification stage
            # This is a bit heuristic; a more robust way might involve knowing the exact ID, but _vX makes that tricky.
            clarification_key_pattern = "stage_BC_clarify_for_B" 
            found_clarification_output = None
            for key, value in intermediate_outputs.items():
                if key.startswith(clarification_key_pattern): # e.g., "stage_BC_clarify_for_B" or "stage_BC_clarify_for_B_v1"
                    if isinstance(value, dict) and "clarification_provided" in value:
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

    # ADDED: Protocol-aware execution method (hybrid approach)
    async def execute_with_protocols(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using appropriate protocol with fallback to traditional method.
        Follows AI agent best practices for hybrid execution.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                self._logger.warning("Protocol execution failed, falling back to traditional method")
                raise ProtocolExecutionError("Pure protocol execution failed")
                
        except Exception as e:
            self._logger.warning(f"Protocol execution error: {e}, falling back to traditional method")
            raise ProtocolExecutionError("Pure protocol execution failed")

    # ADDED: Protocol phase execution logic
    async def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input: Any) -> Any:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }

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

    async def __call__(self, input_payload: MasterPlannerReviewerInput) -> MasterPlannerReviewerOutput:
        """Makes the agent instance directly callable, invoking its async logic."""
        logger.debug(f"MasterPlannerReviewerAgent instance __call__ invoked with input_payload: {type(input_payload)}")
        raise ProtocolExecutionError("Pure protocol execution failed")

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