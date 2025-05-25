from __future__ import annotations

import json
import logging
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

from typing import Any, List, Dict, Optional, ClassVar
import uuid
from datetime import datetime, timezone

from chungoid.schemas.agent_master_planner import (
    MasterPlannerInput,
    MasterPlannerOutput,
)
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec, EnhancedMasterExecutionPlan, EnhancedMasterStageSpec
from chungoid.schemas.user_goal_schemas import UserGoalRequest
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase

from pathlib import Path # For conceptual PCMA instantiation

# Collection constants
EXECUTION_PLANS_COLLECTION = "execution_plans"

# Placeholder for LLM client - replace with actual implementation path
# from chungoid.utils.llm_clients import get_llm_client, LLMInterface

logger = logging.getLogger(__name__)

# --- Begin Enhanced Task-Type Orchestration Prompt ---
# Replacing the hardcoded agent_id system prompt with task-type orchestration
ENHANCED_AUTONOMOUS_SYSTEM_PROMPT = """
You are the Master Project Orchestrator, an expert AI system responsible for creating 
task-driven execution plans that leverage autonomous agent capabilities while preserving 
specialized expertise.

**AUTONOMOUS EXECUTION PHILOSOPHY:**
Generate plans using TASK TYPES and CAPABILITY REQUIREMENTS instead of hardcoded agent IDs. 
The orchestrator will automatically select the best autonomous-capable agent for each task, 
ensuring specialized expertise is preserved through autonomous execution.

**REQUIRED JSON SCHEMA:**
You MUST return a JSON object with this EXACT structure:

```json
{
  "id": "plan_unique_id",
  "name": "Plan Name",
  "description": "Plan description",
  "version": "2.0.0",
  "initial_stage": "first_stage_id",
  "project_id": "project_identifier",
  "stages": {
    "stage_id_1": {
      "id": "stage_id_1",
      "name": "Stage Name",
      "description": "Stage description",
      "task_type": "requirements_analysis",
      "required_capabilities": ["requirements_analysis", "stakeholder_analysis"],
      "preferred_execution": "autonomous",
      "fallback_agent_id": "ProductAnalystAgent_v1",
      "inputs": {
        "user_goal": "goal text",
        "project_context": "context"
      },
      "success_criteria": ["criteria1", "criteria2"],
      "next_stage": "stage_id_2"
    },
    "stage_id_2": {
      "id": "stage_id_2",
      "name": "Code Generation",
      "description": "Generate the required code",
      "task_type": "code_generation",
      "required_capabilities": ["code_generation", "implementation"],
      "preferred_execution": "autonomous",
      "fallback_agent_id": "SmartCodeGeneratorAgent_v1",
      "inputs": {
        "project_id": "{context.project_id}",
        "task_description": "Create a simple Python hello world script based on requirements",
        "target_file_path": "hello_world.py",
        "programming_language": "python"
      },
      "success_criteria": ["code_generated", "tests_pass"],
      "next_stage": null
    }
  }
}
```

**TASK TYPE VOCABULARY:**
- requirements_analysis: Analyze user goals, extract requirements, stakeholder analysis
- environment_setup: Bootstrap development environment, dependency management
- architecture_design: System architecture, component design, blueprint generation
- code_generation: Generate code modules, components, implementations
- file_operations: File system operations, directory management, file manipulation
- documentation: Generate project documentation, API docs, README files
- risk_assessment: Analyze risks, identify issues, optimization opportunities
- quality_validation: Review artifacts, validate quality, architectural review
- dependency_management: Package management, conflict resolution, version optimization
- testing: Test generation, execution, validation

**AUTONOMOUS AGENT CAPABILITIES:**
- ProductAnalystAgent_v1: requirements_analysis, stakeholder_analysis, documentation
- ArchitectAgent_v1: architecture_design, system_planning, blueprint_generation
- EnvironmentBootstrapAgent: environment_setup, dependency_management, project_bootstrapping
- DependencyManagementAgent_v1: dependency_analysis, package_management, conflict_resolution
- ProjectDocumentationAgent_v1: documentation_generation, project_analysis
- ProactiveRiskAssessorAgent_v1: risk_assessment, deep_investigation, impact_analysis
- BlueprintReviewerAgent_v1: review_protocol, quality_validation, architectural_review
- CodeDebuggingAgent_v1: code_debugging, error_analysis, automated_fixes
- SmartCodeGeneratorAgent_v1: code_generation, implementation, module_creation
- SystemRequirementsGatheringAgent_v1: requirements_gathering, stakeholder_analysis

**CONCRETE AGENT FALLBACKS:**
- SystemFileSystemAgent_v1: file_operations, directory_management (implements invoke_async)
- SystemInterventionAgent_v1: human_interaction, system_intervention

**PLAN GENERATION DIRECTIVE:**
1. Analyze user goal and identify required task types
2. For each task, specify task_type and required_capabilities
3. Prefer autonomous execution when capabilities match
4. Provide concrete fallbacks for specialized operations
5. Ensure task dependencies and execution order are logical
6. Return ONLY the JSON object - no additional text or formatting

The output MUST be a single JSON object following the exact schema shown above.
"""

DEFAULT_USER_PROMPT_TEMPLATE = (
    'User Goal: "{user_goal_string}"\n'
    'Target Platform: {target_platform_string}\n\n'
    "Project Context (Optional):\n"
    "{project_context_summary_string}\n\n"
    r"Current `MasterExecutionPlan` (if any, for modification requests):\n"
    r"```json\n"
    r"{existing_plan_json_if_any}\n"
    r"```\n\n"
    r"Based on the user goal (and existing plan if provided), "
    r"generate the complete `MasterExecutionPlan` JSON object."
)
# --- End Embedded Prompt ---


# Placeholder LLM Interface (replace with actual client)
# class MockLLMClient:  # implements LLMInterface # This will be removed or made external
#     async def generate_json(
#         self, system_prompt: str, user_prompt: str, temperature: float = 0.1
#     ) -> Dict[str, Any]:
#         logger.warning(
#             "MockLLMClient.generate_json called. "
#             "Returning a predefined example plan for ANY goal."
#         )
#         # This mock will return the 'show-config' plan structure for any input,
#         # to allow testing the parsing logic.
#         # In a real scenario, this would be the LLM's JSON output string, then parsed.

#         # Simulate LLM returning the JSON for the "show-config" plan from previous static version
#         plan_id_mock = f"mock_llm_plan_{str(uuid.uuid4())[:4]}"
#         mock_plan_dict = {
#             "id": plan_id_mock,
#             "name": "Mock LLM Plan for: User Goal",  # Will be replaced by actual goal later
#             "description": (
#                 "This is a mock plan generated by MockLLMClient for "
#                 "testing purposes."
#             ),
#             "start_stage": "define_show_config_spec_mock",  # Using mock stage names
#             "stages": {
#                 "define_show_config_spec_mock": {
#                     "name": "Define 'show-config' CLI Command Specification (Mock)",
#                     "agent_id": "MockSystemInterventionAgent_v1",
#                     "output_context_path": "stage_outputs.define_show_config_spec_mock",
#                     "number": 1.0,
#                     "inputs": {
#                         "prompt_message_for_user": "Proceed with initial project setup based on gathered requirements?"
#                     },
#                     "success_criteria": [
#                         "'{context.shared_data.show_config_specification_mock}' != None"
#                     ],  # Simplified
#                     "next_stage": "implement_show_config_logic_mock",
#                 },
#                 "implement_show_config_logic_mock": {
#                     "name": "Implement 'show-config' CLI Logic (Mock)",
#                     "agent_id": "MockCodeGeneratorAgent_v1",
#                     "output_context_path": (
#                         "stage_outputs.implement_show_config_logic_mock"
#                     ),
#                     "number": 2.0,
#                     "inputs": {
#                         "target_file_path": "chungoid-core/src/chungoid/cli.py",
#                         "code_specification_prompt": (
#                             "MockLLM: Implement based on "
#                             "{{ context.shared_data.show_config_specification_mock }}"
#                         ),
#                     },
#                     "success_criteria": [
#                         (
#                             "'{context.outputs.implement_show_config_logic_mock.code_changes_applied}' "
#                             "== True"
#                         )
#                     ],  # Simplified
#                     "next_stage": "FINAL_STEP",  # Simplified for mock
#                 },
#             },
#         }
#         # Simulate delay
#         import asyncio

#         await asyncio.sleep(0.1)
#         return mock_plan_dict


# Registry-first architecture import
from chungoid.registry import register_system_agent

@register_system_agent(capabilities=["multi_agent_coordination", "deep_planning", "workflow_orchestration"])
class MasterPlannerAgent(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "SystemMasterPlannerAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Master Planner Agent"
    AGENT_VERSION: ClassVar[str] = "0.2.0"  # Updated version
    CAPABILITIES: ClassVar[List[str]] = ["multi_agent_coordination", "deep_planning", "workflow_orchestration"]  # Added required CAPABILITIES
    DESCRIPTION: ClassVar[str] = (
        "Generates a MasterExecutionPlan based on a high-level user goal using an "
        "LLM."
    )  # Updated
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC

    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["autonomous_team_formation", "enhanced_deep_planning"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["enhanced_deep_planning", "goal_tracking"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ["agent_communication", "context_sharing", "tool_validation"]

    NEW_BLUEPRINT_TO_FLOW_PROMPT_NAME: ClassVar[str] = "blueprint_to_flow_agent_v1.yaml"

    # MODIFIED: Declared fields
    system_prompt: str

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager):
        initial_data = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager,
            "system_prompt": ENHANCED_AUTONOMOUS_SYSTEM_PROMPT
        }
        # Pydantic's BaseModel.__init__ (called via super() chain) will use these
        # to populate fields from ProtocolAwareAgent (llm_provider, prompt_manager)
        super().__init__(**initial_data)

        # Post-initialization checks or logic can go here if needed,
        # but basic field presence/type for declared non-optional fields 
        # is handled by Pydantic during super().__init__.
        # (a non-optional field) was None after super().__init__.

    def _attempt_json_repair(self, malformed_json: str) -> Optional[str]:
        """
        Attempt to repair common JSON formatting issues that can occur with LLM responses.
        
        Args:
            malformed_json: The potentially malformed JSON string from LLM
            
        Returns:
            Repaired JSON string if successful, None if repair failed
        """
        if not malformed_json or not malformed_json.strip():
            return None
            
        try:
            # Common repair strategies
            repaired = malformed_json.strip()
            
            # 1. Handle truncated responses by removing incomplete JSON elements
            # Look for the last complete stage or section
            if repaired.endswith('"'):
                # Remove the trailing incomplete quote
                repaired = repaired[:-1]
                logger.debug("Removed trailing incomplete quote")
            
            # Find the last complete stage entry by looking for complete stage blocks
            # Remove any incomplete final stage entry
            lines = repaired.split('\n')
            complete_lines = []
            in_stage = False
            stage_depth = 0
            
            for line in lines:
                stripped = line.strip()
                if '"write_' in stripped and not stripped.endswith('",'):
                    # This is likely the start of an incomplete stage - truncate here
                    logger.debug("Found incomplete stage entry, truncating at this point")
                    break
                complete_lines.append(line)
            
            repaired = '\n'.join(complete_lines)
            
            # 2. Handle unterminated strings by closing them
            if repaired.count('"') % 2 != 0:
                # Odd number of quotes - add closing quote at the end
                logger.debug("Attempting to fix unterminated string by adding closing quote")
                repaired = repaired + '"'
            
            # 3. Handle missing closing braces/brackets
            open_braces = repaired.count('{') - repaired.count('}')
            open_brackets = repaired.count('[') - repaired.count(']')
            
            if open_braces > 0:
                logger.debug(f"Adding {open_braces} missing closing braces")
                repaired = repaired + '}' * open_braces
                
            if open_brackets > 0:
                logger.debug(f"Adding {open_brackets} missing closing brackets")
                repaired = repaired + ']' * open_brackets
            
            # 4. Handle trailing commas before closing braces/brackets
            repaired = repaired.replace(',}', '}').replace(',]', ']')
            
            # 5. Try to extract valid JSON if the response has extra text
            if not repaired.startswith('{'):
                # Look for the first '{' character
                start_idx = repaired.find('{')
                if start_idx != -1:
                    repaired = repaired[start_idx:]
                    logger.debug("Extracted JSON from response with extra prefix text")
            
            # 6. Handle incomplete stage entries by removing trailing commas and unclosed content
            # This is a more aggressive approach for truncated responses
            if '"stages"' in repaired:
                try:
                    # Find the stages section and ensure it's properly closed
                    stages_start = repaired.find('"stages"')
                    stages_content = repaired[stages_start:]
                    
                    # Look for incomplete stage definitions (missing closing braces)
                    # and truncate before them
                    import re
                    # Remove any incomplete stage at the end
                    repaired = re.sub(r',\s*"[^"]+"\s*:\s*{\s*[^}]*$', '', repaired)
                    
                    # Ensure stages section is properly closed
                    if not repaired.rstrip().endswith('}'):
                        # Count open braces in stages section to determine how many to close
                        stages_section = repaired[stages_start:]
                        open_in_stages = stages_section.count('{') - stages_section.count('}')
                        if open_in_stages > 0:
                            repaired += '}' * open_in_stages
                            
                except Exception as regex_error:
                    logger.debug(f"Advanced repair failed, falling back to basic repair: {regex_error}")
            
            # Test if the repair worked
            json.loads(repaired)
            return repaired
            
        except (json.JSONDecodeError, Exception) as e:
            logger.debug(f"JSON repair attempt failed: {e}")
            return None

    # ADDED: Protocol-aware execution method (hybrid approach)
    async def execute_with_protocols(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using appropriate protocol with fallback to traditional method.
        For MasterPlannerAgent, this delegates to invoke_async to ensure proper output format.
        """
        try:
            # For MasterPlannerAgent, we need to call invoke_async to get proper MasterPlannerOutput
            logger.info(f"MasterPlannerAgent executing via protocol delegation to invoke_async")
            
            # Convert task_input to MasterPlannerInput if needed
            if hasattr(task_input, 'dict'):
                task_input_dict = task_input.dict()
            elif hasattr(task_input, 'model_dump'):
                task_input_dict = task_input.model_dump()
            elif isinstance(task_input, dict):
                task_input_dict = task_input
            else:
                # Try to create MasterPlannerInput from the task_input
                from ...schemas.agent_master_planner import MasterPlannerInput
                task_input_dict = {"user_goal": str(task_input)}
            
            # Create MasterPlannerInput from the task input
            from ...schemas.agent_master_planner import MasterPlannerInput
            if isinstance(task_input, MasterPlannerInput):
                planner_input = task_input
            else:
                planner_input = MasterPlannerInput(**task_input_dict)
            
            # Call the actual invoke_async method to get proper MasterPlannerOutput
            result = await self.invoke_async(planner_input, full_context)
            
            logger.info(f"MasterPlannerAgent protocol execution completed successfully")
            return result
                
        except Exception as e:
            logger.error(f"MasterPlannerAgent protocol execution failed: {e}")
            # Re-raise the exception to maintain error handling behavior
            raise

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
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

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        """Extract agent output from protocol execution results."""
        # For MasterPlannerAgent, this method is not used since we delegate directly to invoke_async
        # But we keep it for compatibility
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }

    async def invoke_async(
        self,
        inputs: MasterPlannerInput,
        full_context: Optional[Dict[str, Any]] = None,
    ) -> MasterPlannerOutput:
        logger.info(
            f"MasterPlannerAgent (LLM-driven) invoked with goal: "
            f"{inputs.user_goal}"
        )

        user_goal_str = inputs.user_goal
        project_context_summary = (
            ""  # Placeholder, could be passed in MasterPlannerInput if needed
        )
        existing_plan_json = "{}"  # Placeholder for modification workflows
        target_platform_str = "Not specified" # ADDED default for target_platform

        if inputs.original_request and inputs.original_request.key_constraints:
            try:
                project_context_summary = json.dumps(
                    inputs.original_request.key_constraints, indent=2
                )
            except TypeError:
                project_context_summary = str(inputs.original_request.key_constraints)
        
        if inputs.original_request and inputs.original_request.target_platform: # ADDED target_platform handling
            target_platform_str = inputs.original_request.target_platform

        # TODO: Add logic for handling existing_plan_json if modification is supported

        current_prompt_name_to_use: Optional[str] = None
        prompt_data_for_llm: Dict[str, Any] = {}
        current_system_prompt: str = ""
        current_user_prompt: str = ""

        if inputs.blueprint_doc_id:
            logger.info(f"MasterPlannerAgent invoked in Blueprint-to-Flow mode for blueprint: {inputs.blueprint_doc_id}")
            current_prompt_name_to_use = self.NEW_BLUEPRINT_TO_FLOW_PROMPT_NAME
            
            blueprint_content: Optional[str] = None
            reviewer_feedback_content: Optional[str] = None

            try:
                logger.info(f"Attempting to fetch blueprint content for doc_id: {inputs.blueprint_doc_id} using PCMA.")
                # Placeholder for actual PCMA retrieval - these would be real calls
                retrieved_blueprint = None  # Retrieved from PCMA
                
                if retrieved_blueprint and retrieved_blueprint.status == "SUCCESS" and retrieved_blueprint.content:
                    blueprint_content = str(retrieved_blueprint.content)
                    logger.info(f"Successfully fetched blueprint content for {inputs.blueprint_doc_id}.")
                else:
                    logger.error(f"Blueprint content not found or empty for doc_id: {inputs.blueprint_doc_id} in project {inputs.project_id}. Status: {retrieved_blueprint.status if retrieved_blueprint else 'N/A'}")
                    return MasterPlannerOutput(
                        master_plan_json="{}", # Default empty plan on error
                        error_message=f"Blueprint content not found for doc_id: {inputs.blueprint_doc_id}. Status: {retrieved_blueprint.status if retrieved_blueprint else 'N/A'}"
                    )

                if inputs.blueprint_reviewer_feedback_doc_id:
                    logger.info(f"Attempting to fetch reviewer feedback for doc_id: {inputs.blueprint_reviewer_feedback_doc_id} using PCMA.")
                    retrieved_feedback = None  # Retrieved from PCMA
                    
                    if retrieved_feedback and retrieved_feedback.status == "SUCCESS" and retrieved_feedback.content:
                        reviewer_feedback_content = str(retrieved_feedback.content)
                        logger.info(f"Successfully fetched reviewer feedback for {inputs.blueprint_reviewer_feedback_doc_id}.")
                    else:
                        logger.warning(f"Reviewer feedback {inputs.blueprint_reviewer_feedback_doc_id} not found, content empty, or retrieval failed. Status: {retrieved_feedback.status if retrieved_feedback else 'N/A'}. Proceeding without it.")
                
                prompt_data_for_llm = {
                    "blueprint_content": blueprint_content,
                    "reviewer_feedback_content": reviewer_feedback_content or "No feedback provided."
                    # Add other necessary fields for this prompt like project_id, available_agents if schema expects them
                }
                # The system prompt for blueprint-to-flow might be implicitly handled by the prompt_name in PromptManager

            except Exception as e:
                logger.error(f"Error fetching context from PCMA for Blueprint-to-Flow mode: {e}", exc_info=True)
                return MasterPlannerOutput(
                    master_plan_json="{}",
                    error_message=f"PCMA context retrieval error: {e}"
                )
        else: # Fallback to existing user_goal to plan logic
            logger.info(f"MasterPlannerAgent invoked in UserGoal-to-Flow mode for goal: {inputs.user_goal}")
            # Use existing prompt logic for user_goal
            current_system_prompt = self.system_prompt
            current_user_prompt = DEFAULT_USER_PROMPT_TEMPLATE.format(
                user_goal_string=user_goal_str,
                target_platform_string=target_platform_str,
                project_context_summary_string=project_context_summary,
                existing_plan_json_if_any=existing_plan_json,
            )

        logger.debug(
            f"MasterPlannerAgent System Prompt (effective):\n{current_system_prompt}"
        )
        logger.debug(
            f"MasterPlannerAgent User Prompt (effective):\n{current_user_prompt}"
        )

        try:
            # MODIFIED: Use self.llm_provider with higher token limit
            llm_response_str = await self.llm_provider.generate(
                system_prompt=current_system_prompt,
                prompt=current_user_prompt,
                temperature=0.1,       # Consistent temperature
                max_tokens=4000,       # Reduced for gpt-3.5-turbo compatibility (max 4096)
                response_format={"type": "json_object"}
            )
            logger.debug(f"Raw LLM JSON response: {llm_response_str}")

            # Step 2: Parse the string response as JSON with error recovery
            try:
                llm_generated_plan_dict = json.loads(llm_response_str)
            except json.JSONDecodeError as json_error:
                logger.warning(f"Initial JSON parse failed: {json_error}. Attempting to repair JSON...")
                # Try to repair common JSON issues
                repaired_json = self._attempt_json_repair(llm_response_str)
                if repaired_json:
                    llm_generated_plan_dict = json.loads(repaired_json)
                    logger.info("Successfully repaired and parsed JSON response")
                else:
                    raise json_error  # Re-raise original error if repair failed

            # --- ADDED: Handle nested structure from OpenAI ---
            # OpenAI sometimes wraps the plan in an "execution_plan" field
            if "execution_plan" in llm_generated_plan_dict and isinstance(llm_generated_plan_dict["execution_plan"], dict):
                logger.info("Detected nested execution_plan structure from LLM, extracting plan...")
                nested_plan = llm_generated_plan_dict["execution_plan"]
                # Preserve any top-level fields that aren't in the nested plan
                for key, value in llm_generated_plan_dict.items():
                    if key != "execution_plan" and key not in nested_plan:
                        nested_plan[key] = value
                llm_generated_plan_dict = nested_plan
                logger.info("Successfully extracted plan from nested structure")
            # --- END nested structure handling ---

            # --- ADDED: Ensure 'id' is present, a non-empty string ---
            current_id = llm_generated_plan_dict.get("id")
            if not isinstance(current_id, str) or not current_id.strip():
                new_plan_id = uuid.uuid4().hex
                logger.warning(
                    f"LLM-generated plan was missing an 'id', had an empty 'id', or 'id' was not a string. "
                    f"Original id: '{current_id}'. Assigning a new UUID: {new_plan_id}"
                )
                llm_generated_plan_dict["id"] = new_plan_id
            # --- END 'id' ensuring block ---

            # --- Inject stage IDs from dictionary keys ---
            stages_from_llm = llm_generated_plan_dict.get("stages")
            if isinstance(stages_from_llm, dict):
                for stage_key, stage_spec_dict in stages_from_llm.items():
                    if isinstance(stage_spec_dict, dict):
                        stage_spec_dict["id"] = stage_key
                        
                        # Sanitize clarification_checkpoint: if present and not a dict, set to None
                        if "clarification_checkpoint" in stage_spec_dict and not isinstance(stage_spec_dict["clarification_checkpoint"], dict):
                            logger.warning(
                                f"LLM provided a non-dictionary value for clarification_checkpoint in stage '{stage_key}'. "
                                f"Received: {stage_spec_dict['clarification_checkpoint']}. Setting to None."
                            )
                            stage_spec_dict["clarification_checkpoint"] = None
                            
                llm_generated_plan_dict["stages"] = stages_from_llm
            # --- End stage ID injection and sanitization ---

            # ADDED: Handle common LLM mistake of using 'initial_stage' instead of 'start_stage'
            # ENHANCED: For EnhancedMasterExecutionPlan, we want 'initial_stage' as primary
            if "start_stage" in llm_generated_plan_dict and "initial_stage" not in llm_generated_plan_dict:
                logger.info("Found 'start_stage' in LLM response, converting to 'initial_stage' for enhanced plan.")
                llm_generated_plan_dict["initial_stage"] = llm_generated_plan_dict.pop("start_stage")
            elif "initial_stage" in llm_generated_plan_dict and "start_stage" not in llm_generated_plan_dict:
                logger.info("Found 'initial_stage' in LLM response, setting 'start_stage' for compatibility.")
                llm_generated_plan_dict["start_stage"] = llm_generated_plan_dict["initial_stage"]

            # Ensure 'original_request' from MasterPlannerInput is added to the plan
            # if the LLM didn't include it (which it likely won't if not explicitly prompted)
            if (
                "original_request" not in llm_generated_plan_dict
                and inputs.original_request
            ):
                llm_generated_plan_dict["original_request"] = (
                    inputs.original_request.model_dump()
                )

            # Update name and description if the mock LLM didn't use the actual goal
            if "Mock LLM Plan for: User Goal" in llm_generated_plan_dict.get(
                "name", ""
            ):
                llm_generated_plan_dict["name"] = f"Plan for: {user_goal_str}"
            if (
                "This is a mock plan generated by MockLLMClient"
                in llm_generated_plan_dict.get("description", "")
            ):
                llm_generated_plan_dict["description"] = (
                    f"Master plan autonomously generated for goal: "
                    f"{user_goal_str}"
                )

            # Ensure project_id from input is added to the plan data if not present
            if inputs.project_id and "project_id" not in llm_generated_plan_dict:
                llm_generated_plan_dict["project_id"] = inputs.project_id

            # Parse the LLM's response for ENHANCED PLAN GENERATION ONLY
            # NO LEGACY SUPPORT - Pure task-type orchestration
            enhanced_plan = EnhancedMasterExecutionPlan.model_validate(llm_generated_plan_dict)

            if inputs.original_request:
                enhanced_plan.original_request = inputs.original_request

            logger.info(f"MasterPlannerAgent successfully generated ENHANCED plan: {enhanced_plan.id} for project {enhanced_plan.project_id}")

            # --- Store generated enhanced plan to PCMA --- 
            generated_plan_artifact_id: Optional[str] = None
            stored_in_collection_name: Optional[str] = None

            if inputs.project_id: # Only store if project_id is available
                try:
                    # Placeholder for PCMA storage logic
                    store_output = None  # Result from PCMA storage
                    
                    if store_output and store_output.status == "SUCCESS":
                        generated_plan_artifact_id = store_output.document_id
                        stored_in_collection_name = EXECUTION_PLANS_COLLECTION
                        logger.info(f"Enhanced MasterExecutionPlan stored successfully in PCMA. Doc ID: {generated_plan_artifact_id}")
                    else:
                        logger.error(f"Failed to store Enhanced MasterExecutionPlan in PCMA. Status: {store_output.status if store_output else 'N/A'}, Message: {store_output.message if store_output else 'N/A'}")
                
                except Exception as e_store:
                    logger.error(f"Exception during PCMA storage of Enhanced MasterExecutionPlan: {e_store}", exc_info=True)
            else:
                logger.warning("project_id not provided in MasterPlannerInput, enhanced plan will not be stored in PCMA.")

            return MasterPlannerOutput(
                master_plan_json=enhanced_plan.model_dump_json(indent=2),
                confidence_score=0.85,  # Higher confidence for enhanced autonomous plans
                planner_notes="Enhanced autonomous plan generated with task-type orchestration.",
                generated_plan_artifact_id=generated_plan_artifact_id,
                stored_in_collection=stored_in_collection_name
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            return MasterPlannerOutput(
                master_plan_json="",
                error_message=f"LLM output was not valid JSON: {e}",
                planner_notes="LLM output parsing failed.",
            )
        except Exception as e:  # Catch Pydantic validation errors and other issues
            logger.error(
                f"Error processing LLM response or validating plan: {e}",
                exc_info=True,
            )
            return MasterPlannerOutput(
                master_plan_json="",
                error_message=f"Error generating or validating plan: {str(e)}",
                planner_notes="Plan generation/validation failed.",
            )


def get_agent_card_static() -> AgentCard:
    """Returns the static AgentCard for the MasterPlannerAgent."""
    return AgentCard(
        agent_id=MasterPlannerAgent.AGENT_ID,
        name=MasterPlannerAgent.AGENT_NAME,
        version=MasterPlannerAgent.VERSION,  # Ensure this matches class version
        description=MasterPlannerAgent.DESCRIPTION,  # Ensure this matches
        category=MasterPlannerAgent.CATEGORY,
        visibility=MasterPlannerAgent.VISIBILITY,
        input_schema=MasterPlannerInput.model_json_schema(),
        output_schema=MasterPlannerOutput.model_json_schema(),
    )


async def main_test():
    logging.basicConfig(level=logging.INFO)
    logger.info("Running MasterPlannerAgent (LLM-driven) test...")
    planner = MasterPlannerAgent()

    # Test 1: Simple goal
    test_goal_1 = UserGoalRequest(
        goal_description="Implement a new feature foo_bar.",
        target_platform="chungoid-mcp"
    )
    test_input_1 = MasterPlannerInput(
        user_goal=test_goal_1.goal_description,
        original_request=test_goal_1
    )
    logger.info(f"--- Test 1: Goal: {test_goal_1.goal_description} ---")
    output_1 = await planner.invoke_async(test_input_1)

    if output_1.error_message:
        print(f"Error: {output_1.error_message}")
    else:
        print("Generated Master Plan JSON (Test 1):")
        print(output_1.master_plan_json)
        try:
            parsed_plan_1 = EnhancedMasterExecutionPlan.model_validate_json(
                output_1.master_plan_json
            )
            print("\nEnhanced Plan 1 successfully parsed.")
            print(f"Plan ID: {parsed_plan_1.id}, Name: {parsed_plan_1.name}")
        except Exception as e:
            print(f"\nError parsing generated enhanced plan 1: {e}")

    # Test 2: Another goal to ensure mock is not hardcoded to one specific input text
    test_goal_2 = UserGoalRequest(
        goal_description="Refactor the authentication module.",
        target_platform="chungoid-mcp",
        key_constraints={"details": "auth module is in src/auth"},
    )
    test_input_2 = MasterPlannerInput(
        user_goal=test_goal_2.goal_description,
        original_request=test_goal_2
    )
    logger.info(f"--- Test 2: Goal: {test_goal_2.goal_description} ---")
    output_2 = await planner.invoke_async(test_input_2)

    if output_2.error_message:
        print(f"Error: {output_2.error_message}")
    else:
        print("\nGenerated Master Plan JSON (Test 2):")
        print(output_2.master_plan_json)
        try:
            parsed_plan_2 = EnhancedMasterExecutionPlan.model_validate_json(
                output_2.master_plan_json
            )
            print("\nEnhanced Plan 2 successfully parsed.")
            print(f"Plan ID: {parsed_plan_2.id}, Name: {parsed_plan_2.name}")
            if parsed_plan_2.original_request:
                print(
                    f"Original request in plan 2: "
                    f"{parsed_plan_2.original_request.goal_description}"
                )
                print(
                    f"Key constraints in plan 2: "
                    f"{parsed_plan_2.original_request.key_constraints}"
                )

        except Exception as e:
            print(f"\nError parsing generated enhanced plan 2: {e}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main_test())
