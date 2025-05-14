from typing import Any, Dict, List, Optional
import uuid
import logging
from pathlib import Path
import json
import asyncio

from pydantic import BaseModel, Field

# Chungoid specific imports
from chungoid.schemas.user_goal_schemas import UserGoalRequest
from chungoid.schemas.master_flow import MasterExecutionPlan, MasterStageSpec
from chungoid.utils.agent_registry import AgentCard # Removed AGENT_REGISTRY import
from chungoid.utils.agent_resolver import AgentProvider # For type hinting if passed
from chungoid.utils.llm_provider import LLMProvider, MockLLMProvider # Added LLMProvider imports

# TODO: Import LLM interaction utilities once available/decided

logger = logging.getLogger(__name__)

AGENT_ID = "master_planner_agent"
AGENT_DESCRIPTION = (
    "Takes a high-level user goal and generates a multi-stage execution plan "
    "by decomposing the goal, selecting appropriate agents for sub-tasks, "
    "sequencing them, and formatting the output as a MasterExecutionPlan."
)

# Define the path to the prompt templates directory relative to this file
PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "server_prompts" / "master_planner"

class MasterPlannerAgent:
    """
    Agent responsible for taking a UserGoalRequest and producing a MasterExecutionPlan.
    It uses a multi-step LLM-assisted process for:
    1. Goal Decomposition
    2. Agent/Tool Selection per Task
    3. Dependency Analysis & Sequencing
    4. Plan Formatting
    """

    def __init__(self, agent_provider: AgentProvider, llm_provider: LLMProvider): # Updated llm_provider type
        self.agent_provider = agent_provider
        self.llm_provider = llm_provider
        
        # Load prompts
        try:
            self.decomposition_prompt_template: str = (PROMPTS_DIR / "decomposition_prompt.txt").read_text()
            self.agent_selection_prompt_template: str = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text()
            self.sequencing_prompt_template: str = (PROMPTS_DIR / "sequencing_prompt.txt").read_text()
            logger.info("MasterPlannerAgent: All prompts loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"MasterPlannerAgent: Failed to load one or more prompt files from {PROMPTS_DIR}. Error: {e}")
            # Depending on strictness, could raise an error or use fallback default prompts.
            # For now, let it proceed and potentially fail at usage if prompts are missing.
            self.decomposition_prompt_template = "Error: Decomposition prompt not loaded."
            self.agent_selection_prompt_template = "Error: Agent selection prompt not loaded."
            self.sequencing_prompt_template = "Error: Sequencing prompt not loaded."
            # raise # Or re-raise to prevent agent initialization with missing critical components

    async def _decompose_goal(self, user_goal: UserGoalRequest) -> List[str]:
        """Step 1: Decompose the high-level user goal into smaller, actionable tasks."""
        logger.info(f"Decomposing goal: {user_goal.goal_description}")

        if "Error:" in self.decomposition_prompt_template:
            logger.error("Decomposition prompt was not loaded. Cannot proceed with goal decomposition.")
            return []

        prompt = self.decomposition_prompt_template.format(
            goal_description=user_goal.goal_description,
            target_platform=user_goal.target_platform or "Not specified",
            key_constraints=str(user_goal.key_constraints) if user_goal.key_constraints else "None"
        )

        try:
            raw_response = await self.llm_provider.generate(
                prompt,
                # model_id="specific_model_for_decomposition", # Optional: specify model
                max_tokens=1000 # Adjust as needed
            )
            
            if not raw_response or raw_response.strip() == "":
                logger.warning("LLM returned an empty response for goal decomposition.")
                return []

            # Basic parsing for a numbered list:
            # "1. Task one.\n2. Task two.\n..."
            # or "- Task one\n- Task two"
            decomposed_tasks: List[str] = []
            for line in raw_response.strip().split('\\n'):
                line = line.strip()
                if not line:
                    continue
                # Remove common list prefixes (e.g., "1. ", "- ", "* ")
                if line.startswith(tuple(f"{i}. " for i in range(1, 100))): # Handles "1. ", "2. ", ... "99. "
                    line = line.split('.', 1)[1].strip()
                elif line.startswith(("- ", "* ")):
                    line = line[2:].strip()
                
                if line: # Ensure line is not empty after stripping prefixes
                    decomposed_tasks.append(line)
            
            if not decomposed_tasks:
                logger.warning(f"Could not parse any tasks from LLM decomposition response: {raw_response}")
                return []

            logger.info(f"Decomposed into {len(decomposed_tasks)} tasks.")
            return decomposed_tasks

        except Exception as e:
            logger.error(f"Error during LLM call for goal decomposition: {e}", exc_info=True)
            return [] # Return empty list on error

    async def _select_agent_for_task(self, task_description: str, original_user_goal: UserGoalRequest) -> Dict[str, Any]:
        """Step 2: For a decomposed task, select the most suitable agent(s)."""
        logger.info(f"Selecting agent for task: {task_description}")

        if "Error:" in self.agent_selection_prompt_template:
            logger.error("Agent selection prompt was not loaded. Cannot proceed.")
            return {"selected_agent_ids": ["error_agent_selection_prompt_not_loaded"], "justification": "Prompt load error"}

        candidate_agents_details_formatted = ""
        candidate_cards: List[AgentCard] = []

        if hasattr(self.agent_provider, 'search_agents'):
            logger.info(f"Attempting to find agents for task: '{task_description}' using agent provider.")
            try:
                # Ensure the call to search_agents is awaited if it's an async method
                candidate_cards_with_scores = await self.agent_provider.search_agents(
                    query_text=task_description, 
                    n_results=5 # TODO: Make n_results configurable if needed
                )
                # Assuming candidate_cards_with_scores is a list of (AgentCard, score) tuples
                candidate_cards = [card for card, score in candidate_cards_with_scores]
                logger.info(f"Agent search for task '{task_description}' found {len(candidate_cards)} candidates.")
            except Exception as e:
                logger.error(f"Error calling agent_provider.search_agents for task '{task_description}': {e}")
                candidate_cards = [] # Proceed as if no agents found
        else:
            logger.warning("Agent provider does not have 'search_agents' method. Cannot dynamically find agents.")
            # Fallback behavior will be triggered if candidate_cards remains empty

        if candidate_cards:
            candidate_details_list = []
            for card in candidate_cards:
                doc_parts = []
                doc_parts.append(f"- Agent ID: {card.agent_id}")
                doc_parts.append(f"  Name: {card.name}")
                if card.description:
                    doc_parts.append(f"  Description: {card.description}")
                if card.capabilities:
                    doc_parts.append(f"  Capabilities: {', '.join(card.capabilities)}")
                if card.tags:
                    doc_parts.append(f"  Tags: {', '.join(card.tags)}")
                
                if card.input_schema and isinstance(card.input_schema, dict):
                    summary = card.input_schema.get('title', card.input_schema.get('description', 'N/A'))
                    doc_parts.append(f"  Expected Input Summary: {summary}")
                if card.output_schema and isinstance(card.output_schema, dict):
                    summary = card.output_schema.get('title', card.output_schema.get('description', 'N/A'))
                    doc_parts.append(f"  Expected Output Summary: {summary}")
                
                # Optionally, summarize mcp_tool_input_schemas if relevant for selection context
                # if card.mcp_tool_input_schemas:
                #     for tool_name, schema in card.mcp_tool_input_schemas.items():
                #         if isinstance(schema, dict):
                #             schema_summary = schema.get('title', schema.get('description', 'No summary'))
                #             doc_parts.append(f"  Exposed MCP Tool '{tool_name}' Input: {schema_summary}")

                candidate_details_list.append("\n".join(doc_parts))
            
            candidate_agents_details_formatted = "\n\n".join(candidate_details_list)
        else:
            logger.warning(f"Agent search for task '{task_description}' returned no candidates or provider lacks search. Falling back to hardcoded list.")
            candidate_agents_details_formatted = """- Agent ID: generic_task_agent
  Description: A generic agent capable of performing various tasks.
  Capabilities: Can execute general instructions.
  Expected Input Summary: text prompt
  Expected Output Summary: text result
- Agent ID: file_writer_agent
  Description: Writes content to a file.
  Capabilities: File I/O, content creation.
  Expected Input Summary: file_path, content
  Expected Output Summary: status_message"""            

        prompt = self.agent_selection_prompt_template.format(
            original_user_goal_description=original_user_goal.goal_description, # Ensure placeholder matches prompt
            current_decomposed_task_description=task_description,
            candidate_agents_details_formatted=candidate_agents_details_formatted
        )

        try:
            raw_response = await self.llm_provider.generate(
                prompt,
                # model_id="specific_model_for_agent_selection", # Optional
                max_tokens=500 
            )

            if not raw_response or raw_response.strip() == "":
                logger.warning("LLM returned an empty response for agent selection.")
                return {"selected_agent_ids": ["error_llm_empty_response"], "justification": "LLM empty response"}

            # --- Step 2b: Parse LLM Response --- 
            # Expecting JSON: {"selected_agent_ids": ["id1", "id2"] | "NO_SUITABLE_AGENT", "justification": "..."}
            try:
                parsed_response = json.loads(raw_response)
                selected_ids = parsed_response.get("selected_agent_ids")
                justification = parsed_response.get("justification", "No justification provided.")

                if selected_ids == "NO_SUITABLE_AGENT" or not selected_ids:
                    logger.info(f"LLM indicated no suitable agent for task '{task_description}'. Justification: {justification}")
                    return {"selected_agent_ids": ["NO_SUITABLE_AGENT"], "justification": justification}
                
                if isinstance(selected_ids, list) and len(selected_ids) > 0:
                    logger.info(f"LLM selected agent(s) {selected_ids} for task '{task_description}'. Justification: {justification}")
                    return {"selected_agent_ids": selected_ids, "justification": justification}
                else:
                    logger.error(f"LLM response for agent selection had unexpected format for selected_agent_ids: {selected_ids}. Raw: {raw_response}")
                    return {"selected_agent_ids": ["error_llm_bad_format"], "justification": "LLM response format error"}

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON response from LLM for agent selection: {e}. Raw response: {raw_response}", exc_info=True)
                return {"selected_agent_ids": ["error_json_decode"], "justification": f"JSON decode error: {e}"}
            
        except Exception as e:
            logger.error(f"Error during LLM call for agent selection: {e}", exc_info=True)
            return {"selected_agent_ids": ["error_llm_call_failed"], "justification": f"LLM call error: {e}"}

    async def _sequence_tasks(self, tasks_with_agents: List[Dict[str, Any]], original_user_goal: UserGoalRequest) -> List[Dict[str, Any]]:
        """Step 3: Determine the correct order for tasks and establish next_stage relationships."""
        logger.info(f"Sequencing {len(tasks_with_agents)} tasks for goal: {original_user_goal.goal_description}")

        if "Error:" in self.sequencing_prompt_template:
            logger.error("Sequencing prompt was not loaded. Returning tasks in original order.")
            # Fallback to original order with temporary IDs if prompt is missing
            ordered_tasks_fallback = []
            for i, task_data in enumerate(tasks_with_agents):
                task_data['temp_task_id'] = f"task_{i}" # Ensure original tasks get temp_task_id
                task_data['next_temporary_task_id'] = f"task_{i+1}" if i < len(tasks_with_agents) - 1 else "FINAL_STEP"
                ordered_tasks_fallback.append(task_data)
            return ordered_tasks_fallback

        # Assign temporary IDs to tasks for the LLM to reference
        tasks_for_prompt = []
        temp_id_to_task_map: Dict[str, Dict[str, Any]] = {}
        for i, task_data in enumerate(tasks_with_agents):
            temp_id = f"T{i}"
            task_data['temp_task_id'] = temp_id # Store temp_id in the original dict
            tasks_for_prompt.append(f"{temp_id}: {task_data['task_description']} (Assigned Agent: {task_data['selected_agent_id']})")
            temp_id_to_task_map[temp_id] = task_data
        
        tasks_formatted_str = "\\n".join(tasks_for_prompt)

        prompt = self.sequencing_prompt_template.format(
            goal_description=original_user_goal.goal_description,
            tasks_to_sequence=tasks_formatted_str
        )

        try:
            raw_response = await self.llm_provider.generate(
                prompt,
                max_tokens=max(100, len(tasks_with_agents) * 20) # Adjust token based on number of tasks
            )

            if not raw_response or raw_response.strip() == "":
                logger.warning("LLM returned an empty response for task sequencing. Returning tasks in original order.")
                # Ensure temp_task_id and next_temporary_task_id are set for fallback
                for i, task_data in enumerate(tasks_with_agents):
                    if 'temp_task_id' not in task_data: # Should have been set before try block
                        task_data['temp_task_id'] = f"T{i}" # Defensive
                    task_data['next_temporary_task_id'] = tasks_with_agents[i+1]['temp_task_id'] if i < len(tasks_with_agents) - 1 else "FINAL_STEP"
                return tasks_with_agents # Original list, now with temp_task_ids and next_temporary_task_id

            # Expected LLM output: comma-separated list of temporary task IDs in order, e.g., "T1,T0,T2"
            # Or one ID per line.
            ordered_temp_ids: List[str] = []
            for line in raw_response.strip().split('\n'):
                ids_in_line = [tid.strip() for tid in line.split(',') if tid.strip()]
                ordered_temp_ids.extend(ids_in_line)
            
            # Validate and reorder tasks
            sequenced_tasks: List[Dict[str, Any]] = []
            valid_ids_count = 0
            for temp_id in ordered_temp_ids:
                if temp_id in temp_id_to_task_map:
                    sequenced_tasks.append(temp_id_to_task_map[temp_id])
                    valid_ids_count +=1
                else:
                    logger.warning(f"LLM returned an unknown temporary task ID '{temp_id}' in sequencing. It will be ignored.")
            
            if valid_ids_count != len(tasks_with_agents):
                logger.warning(
                    f"LLM sequencing did not return all original task IDs. Expected {len(tasks_with_agents)}, got {valid_ids_count} valid ones. "
                    f"Resulting sequence might be incomplete or original order might be preferred."
                )
                # Optional: Fallback to original order if sequencing is critically incomplete
                # For now, proceed with the subset, or consider a more robust fallback.
                if not sequenced_tasks: # If LLM returned no valid IDs
                    logger.error("LLM returned no valid task IDs for sequencing. Falling back to original order.")
                    # Ensure temp_task_id and next_temporary_task_id are set for fallback
                    for i, task_data in enumerate(tasks_with_agents):
                        if 'temp_task_id' not in task_data: # Should have been set before try block
                            task_data['temp_task_id'] = f"T{i}" # Defensive
                        task_data['next_temporary_task_id'] = tasks_with_agents[i+1]['temp_task_id'] if i < len(tasks_with_agents) - 1 else "FINAL_STEP"
                    return tasks_with_agents # Original list, now with temp_task_ids and next_temporary_task_id

            # Update next_temporary_task_id for the newly sequenced tasks
            for i, task_data in enumerate(sequenced_tasks):
                task_data['next_temporary_task_id'] = sequenced_tasks[i+1]['temp_task_id'] if i < len(sequenced_tasks) - 1 else "FINAL_STEP"
            
            logger.info(f"Tasks sequenced by LLM. Original count: {len(tasks_with_agents)}, Sequenced count: {len(sequenced_tasks)}.")
            return sequenced_tasks

        except Exception as e:
            logger.error(f"Error during LLM call for task sequencing: {e}. Returning tasks in original order.", exc_info=True)
            # Fallback to original order with temporary IDs if not already processed by initial fallback
            # Also ensure next_temporary_task_id is set
            for i, task_data in enumerate(tasks_with_agents):
                if 'temp_task_id' not in task_data: 
                    task_data['temp_task_id'] = f"T{i}" 
                task_data['next_temporary_task_id'] = tasks_with_agents[i+1]['temp_task_id'] if i < len(tasks_with_agents) - 1 else "FINAL_STEP"
            return tasks_with_agents # Return original if temp_task_ids were already assigned

    def _format_plan(self, ordered_tasks_with_agents: List[Dict[str, Any]], user_goal: UserGoalRequest) -> MasterExecutionPlan:
        logger.info(f"Formatting plan for {len(ordered_tasks_with_agents)} tasks.")
        
        stages: Dict[str, MasterStageSpec] = {}
        start_stage_id: Optional[str] = None
        
        for i, task_data in enumerate(ordered_tasks_with_agents):
            current_stage_id = task_data['temp_task_id'] 
            if i == 0:
                start_stage_id = current_stage_id

            next_stage_id_in_plan: Optional[str] = None
            if task_data['next_temporary_task_id'] != "FINAL_STEP":
                next_stage_id_in_plan = task_data['next_temporary_task_id']

            # Define inputs for the stage
            stage_inputs: Dict[str, Any] = {
                "task_description": task_data['task_description'] # Literal task description
            }
            
            if i == 0:
                # For the very first task, also add the original goal's full details via context path
                stage_inputs["user_goal_details"] = "context.original_request" # Access full UserGoalRequest
            else:
                # For subsequent tasks, make all outputs of the immediate previous stage available under a standard key
                previous_stage_id = ordered_tasks_with_agents[i-1]['temp_task_id']
                stage_inputs["previous_stage_outputs"] = previous_stage_id # Orchestrator resolves this to context.outputs[previous_stage_id]
            
            # New: Allow specific input mapping from previous stages or original_request if defined in task_data
            # This assumes task_data might have an 'explicit_inputs' dict like:
            # "explicit_inputs": {
            #   "current_stage_input_name1": "context.outputs.some_previous_stage_id.specific_output_key",
            #   "current_stage_input_name2": "context.original_request.some_goal_attribute",
            #   "current_stage_input_name3": "literal_value_for_input"
            # }
            if "explicit_inputs" in task_data and isinstance(task_data["explicit_inputs"], dict):
                for input_name, source_specifier in task_data["explicit_inputs"].items():
                    stage_inputs[input_name] = source_specifier # source_specifier is the context path or literal

            # Define placeholder outputs (can be refined if planner has more info)
            stage_outputs: Dict[str, Any] = {
                # Example: Agent might be expected to produce a report file path
                # "output_report_path": "path/to/generated/report.txt",
                # "_mcp_generated_artifacts_relative_paths_": ["path/to/generated/report.txt"]
            }

            stage_spec = MasterStageSpec(
                name=f"Stage {i}: {task_data['task_description'][:100]}", # Prepend Stage number
                agent_id=task_data['selected_agent_id'],
                inputs=stage_inputs, 
                outputs=stage_outputs, # Add outputs
                number=float(i),
                next_stage=next_stage_id_in_plan,
            )
            stages[current_stage_id] = stage_spec

        if not start_stage_id and stages: 
            start_stage_id = list(stages.keys())[0]

        # --- Plan Validation (Basic) ---
        if stages and start_stage_id and start_stage_id not in stages:
            logger.error(f"MasterPlan Validation Error: start_stage '{start_stage_id}' not found in stages. Plan may be invalid.")
            # Potentially mark plan as invalid or return an error plan here in a future iteration.

        all_stage_ids = set(stages.keys())
        for stage_id, stage_spec_val in stages.items():
            if stage_spec_val.next_stage and stage_spec_val.next_stage != "FINAL_STEP" and stage_spec_val.next_stage not in all_stage_ids:
                logger.error(f"MasterPlan Validation Error: Stage '{stage_id}' has next_stage '{stage_spec_val.next_stage}' which is not a valid stage ID. Plan may be invalid.")

        plan_id = f"mep_{user_goal.goal_id}_{str(uuid.uuid4())}"

        plan = MasterExecutionPlan(
            id=plan_id,
            name=f"Plan for: {user_goal.goal_description[:50]}...",
            description=f"Autogenerated plan for user goal: {user_goal.goal_description}",
            start_stage=start_stage_id if start_stage_id else "", 
            stages=stages,
            original_request=user_goal # Store the original request object itself
        )
        return plan

    async def execute(self, user_goal_request: UserGoalRequest) -> MasterExecutionPlan:
        """
        Main execution method for the Master Planner Agent.
        Takes a UserGoalRequest and returns a MasterExecutionPlan.
        """
        logger.info(f"MasterPlannerAgent received request: {user_goal_request.goal_id} - {user_goal_request.goal_description}")

        # Step 1: Decompose Goal
        decomposed_tasks_desc = await self._decompose_goal(user_goal_request)
        if not decomposed_tasks_desc:
            logger.error("Goal decomposition failed or returned no tasks. Cannot create a plan.")
            return MasterExecutionPlan(
                id=f"error_plan_decomp_{user_goal_request.goal_id}_{str(uuid.uuid4())}",
                name=f"Failed Plan (Decomposition): {user_goal_request.goal_description[:30]}...",
                description="Goal decomposition resulted in no tasks. No execution plan generated.",
                start_stage="",
                stages={},
                original_request=user_goal_request
            )

        # Step 2: Select Agent for Each Task
        tasks_with_agents_data: List[Dict[str, Any]] = []
        for task_desc in decomposed_tasks_desc:
            selection_result = await self._select_agent_for_task(task_desc, user_goal_request)
            
            # Use the first agent if multiple are returned, or handle NO_SUITABLE_AGENT
            agent_id_to_use = "placeholder_for_unassigned_task" # Default
            if selection_result.get("selected_agent_ids") and isinstance(selection_result["selected_agent_ids"], list) and selection_result["selected_agent_ids"][0] != "NO_SUITABLE_AGENT" and selection_result["selected_agent_ids"][0] != "error_agent_selection_prompt_not_loaded" and not selection_result["selected_agent_ids"][0].startswith("error_"):
                agent_id_to_use = selection_result["selected_agent_ids"][0]
            elif selection_result.get("selected_agent_ids") and selection_result["selected_agent_ids"][0] == "NO_SUITABLE_AGENT":
                 logger.warning(f"No suitable agent identified by LLM for task: {task_desc}. Using placeholder.")
                 # agent_id_to_use remains placeholder_for_unassigned_task
            else:
                logger.error(f"Agent selection failed for task '{task_desc}' or returned error. Result: {selection_result}. Using placeholder.")
                # agent_id_to_use remains placeholder_for_unassigned_task

            tasks_with_agents_data.append({
                "task_description": task_desc,
                "selected_agent_id": agent_id_to_use,
                "justification": selection_result.get("justification", "N/A")
            })
        
        # After selection, check if any tasks actually got a real agent assigned.
        # If all tasks ended up with the placeholder, then agent selection effectively failed.
        meaningful_assignments = [ 
            task for task in tasks_with_agents_data 
            if task["selected_agent_id"] != "placeholder_for_unassigned_task"
        ]

        if not meaningful_assignments:
            logger.error("Agent selection phase resulted in no tasks with assigned agents. Cannot create a meaningful plan.")
            return MasterExecutionPlan(
                id=f"error_plan_selection_{user_goal_request.goal_id}_{str(uuid.uuid4())}",
                name=f"Failed Plan (Agent Selection): {user_goal_request.goal_description[:30]}...",
                description="Agent selection phase resulted in no tasks with assigned agents.",
                start_stage="",
                stages={},
                original_request=user_goal_request
            )
        
        # Step 3: Sequence Tasks - Only proceed if there are tasks with actual agents
        # Pass the original tasks_with_agents_data which includes placeholders, 
        # as _sequence_tasks might still need to see all original tasks for context, 
        # but the decision to proceed is based on meaningful_assignments.
        # Or, alternatively, pass meaningful_assignments if _sequence_tasks should only see those.
        # For now, let's assume _sequence_tasks can handle placeholders or they are filtered later.
        # The critical part is the guard *before* calling _sequence_tasks.
        ordered_tasks_with_agents = await self._sequence_tasks(tasks_with_agents_data, user_goal_request)
        if not ordered_tasks_with_agents:
            logger.error("Task sequencing resulted in no ordered tasks. Cannot create a plan.")
            return MasterExecutionPlan(
                id=f"error_plan_sequencing_{user_goal_request.goal_id}_{str(uuid.uuid4())}",
                name=f"Failed Plan (Sequencing): {user_goal_request.goal_description[:30]}...",
                description="Task sequencing phase resulted in no ordered tasks.",
                start_stage="",
                stages={},
                original_request=user_goal_request
            )

        # Step 4: Format Plan
        final_plan = self._format_plan(ordered_tasks_with_agents, user_goal_request)
        
        logger.info(f"MasterPlannerAgent successfully generated plan: {final_plan.id}")
        return final_plan

# Agent Card Definition
master_planner_agent_card = AgentCard(
    agent_id=AGENT_ID,
    name="Master Planner Agent",
    description=AGENT_DESCRIPTION,
    input_schema=UserGoalRequest.model_json_schema(),
    output_schema=MasterExecutionPlan.model_json_schema(),
    # callable_fn_path="chungoid.runtime.agents.master_planner_agent.MasterPlannerAgent", # Path to class
    # For now, let's assume it's registered differently or direct instantiation
    version="0.1.0",
    tags=["planning", "orchestration", "meta-agent"],
    dependencies=[] # e.g., specific LLM models or libraries if known
)

# TODO: Register this agent card with the AGENT_REGISTRY
# This might happen at application startup or through a dedicated registration script.
# Example (conceptual, depends on AGENT_REGISTRY's API):
# AGENT_REGISTRY.add(master_planner_agent_card)

# Example of how this agent might be invoked (outside this file, e.g., by ChungoidEngine)
async def example_usage():
    # Mock dependencies
    # Use MockLLMProvider for testing
    
    # --- Mock AgentProvider and its _registry for dynamic agent listing ---
    mock_registry = MagicMock()
    sample_card_1_data = {
        "agent_id": "generic_task_agent_ex", 
        "name": "Generic Task Agent Example", 
        "description": "Can do generic things.",
        "capabilities": ["general"], 
        "tags": [],
        "input_schema": {"type": "object", "title": "Generic Input", "description": "Takes a text prompt."},
        "output_schema": {"type": "object", "title": "Generic Output", "description": "Returns a text result."}
    }
    sample_card_2_data = {
        "agent_id": "file_writer_agent_ex", 
        "name": "File Writer Example", 
        "description": "Writes to files.",
        "capabilities": ["file_io"], 
        "tags": [],
        "input_schema": {"type": "object", "title": "File Write Input", "description": "Takes path and content."},
        "output_schema": {"type": "object", "title": "File Write Output", "description": "Returns status."}
    }
    # Need to import AgentCard for this example usage if we are creating instances.
    # However, AgentRegistry.list() should return AgentCard instances, so mock its return value directly.
    # For simplicity, we mock the return of registry.list() directly with dictionaries that _select_agent_for_task can parse
    # as if they were AgentCard objects (duck typing for the attributes it accesses).
    # A more thorough mock would have registry.list() return MagicMock(spec=AgentCard) instances.
    from chungoid.utils.agent_registry import AgentCard # Import for constructing mock cards
    mock_registry.list.return_value = [
        AgentCard(**sample_card_1_data),
        AgentCard(**sample_card_2_data)
    ]

    mock_agent_provider = MagicMock(spec=AgentProvider)
    mock_agent_provider._registry = mock_registry # Assign the mock registry

    # --- Construct mock LLM responses ---
    # 1. Decomposition Prompt
    decomposition_prompt_key = (PROMPTS_DIR / "decomposition_prompt.txt").read_text().format(
        goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        target_platform="Python CLI",
        key_constraints=str({"storage_format": "JSON", "user_interaction": "command-line arguments"})
    ).strip()
    mock_decomposition_response = """1. Design Naming_Scheme storage format.
2. Implement CLI argument parsing.
3. Implement Naming_Scheme creation function.
4. Implement Naming_Scheme retrieval function."""

    # 2. Agent Selection Prompt (using the dynamically generated agent list)
    # Create the expected formatted string from our sample cards for keying the mock response
    formatted_sample_agents_for_prompt = []
    for card_data in [sample_card_1_data, sample_card_2_data]:
        capabilities_list = card_data["capabilities"] if card_data.get("capabilities") else card_data.get("tags", [])
        capabilities_str = ", ".join(capabilities_list) if capabilities_list else 'N/A'
        input_desc = card_data["input_schema"].get('description', card_data["input_schema"].get('title', 'No description'))
        output_desc = card_data["output_schema"].get('description', card_data["output_schema"].get('title', 'No description'))
        details = (
            f"- Agent ID: {card_data['agent_id']}\\n"
            f"  Name: {card_data['name']}\\n"
            f"  Description: {card_data['description'] or 'N/A'}\\n"
            f"  Capabilities: {capabilities_str}\\n" 
            f"  Input Summary: {input_desc}\\n"
            f"  Output Summary: {output_desc}"
        )
        formatted_sample_agents_for_prompt.append(details)
    candidate_agents_details_formatted_key = "\\n\\n".join(formatted_sample_agents_for_prompt)

    # Mock responses for each decomposed task. Let's assume 4 tasks.
    # Task 0: "Design Naming_Scheme storage format."
    agent_selection_prompt_key_task0 = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text().format(
        original_user_goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        current_decomposed_task_description="Design Naming_Scheme storage format.",
        candidate_agents_details_formatted=candidate_agents_details_formatted_key
    ).strip()
    mock_agent_selection_response_task0 = json.dumps({"selected_agent_ids": ["generic_task_agent_ex"], "justification": "Good for design tasks."})

    # Task 1: "Implement CLI argument parsing."
    agent_selection_prompt_key_task1 = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text().format(
        original_user_goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        current_decomposed_task_description="Implement CLI argument parsing.",
        candidate_agents_details_formatted=candidate_agents_details_formatted_key
    ).strip()
    mock_agent_selection_response_task1 = json.dumps({"selected_agent_ids": ["generic_task_agent_ex"], "justification": "Good for CLI logic."})

    # Task 2: "Implement Naming_Scheme creation function."
    agent_selection_prompt_key_task2 = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text().format(
        original_user_goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        current_decomposed_task_description="Implement Naming_Scheme creation function.",
        candidate_agents_details_formatted=candidate_agents_details_formatted_key
    ).strip()
    mock_agent_selection_response_task2 = json.dumps({"selected_agent_ids": ["file_writer_agent_ex"], "justification": "Good for creation with storage."})
    
    # Task 3: "Implement Naming_Scheme retrieval function."
    agent_selection_prompt_key_task3 = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text().format(
        original_user_goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        current_decomposed_task_description="Implement Naming_Scheme retrieval function.",
        candidate_agents_details_formatted=candidate_agents_details_formatted_key
    ).strip()
    mock_agent_selection_response_task3 = json.dumps({"selected_agent_ids": ["generic_task_agent_ex"], "justification": "Good for retrieval logic, maybe file_writer could also read."})

    # 3. Sequencing Prompt (using the decomposed tasks from above)
    tasks_for_sequencing_key_list = [
        f"T0: Design Naming_Scheme storage format. (Assigned Agent: generic_task_agent_ex)",
        f"T1: Implement CLI argument parsing. (Assigned Agent: generic_task_agent_ex)",
        f"T2: Implement Naming_Scheme creation function. (Assigned Agent: file_writer_agent_ex)",
        f"T3: Implement Naming_Scheme retrieval function. (Assigned Agent: generic_task_agent_ex)"
    ]
    tasks_for_sequencing_key_str = "\\n".join(tasks_for_sequencing_key_list)
    sequencing_prompt_key = (PROMPTS_DIR / "sequencing_prompt.txt").read_text().format(
        goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        tasks_to_sequence=tasks_for_sequencing_key_str
    ).strip()
    mock_sequencing_response = "T0,T1,T2,T3" # Assume LLM keeps original order for simplicity here

    mock_llm_provider = MockLLMProvider(predefined_responses={
        decomposition_prompt_key: mock_decomposition_response,
        agent_selection_prompt_key_task0: mock_agent_selection_response_task0,
        agent_selection_prompt_key_task1: mock_agent_selection_response_task1,
        agent_selection_prompt_key_task2: mock_agent_selection_response_task2,
        agent_selection_prompt_key_task3: mock_agent_selection_response_task3,
        sequencing_prompt_key: mock_sequencing_response
    })

    planner = MasterPlannerAgent(agent_provider=mock_agent_provider, llm_provider=mock_llm_provider)
    
    sample_goal = UserGoalRequest(
        goal_description="Develop a Python CLI tool to manage a personal Naming_Scheme document.",
        target_platform="Python CLI",
        key_constraints={"storage_format": "JSON", "user_interaction": "command-line arguments"}
    )
    
    print("\n--- Running MasterPlannerAgent example_usage ---")
    try:
        generated_plan = await planner.execute(sample_goal)
        print(f"Generated Plan ID: {generated_plan.id}")
        print(f"Plan Name: {generated_plan.name}")
        print(f"Start Stage: {generated_plan.start_stage}")
        print(f"Number of stages: {len(generated_plan.stages)}")

        for stage_id, stage_spec in generated_plan.stages.items():
            print(f"  Stage ID: {stage_id}")
            print(f"    Name: {stage_spec.name}")
            print(f"    Agent ID: {stage_spec.agent_id}")
            print(f"    Number: {stage_spec.number}")
            print(f"    Next Stage: {stage_spec.next_stage}")
            print(f"    Inputs: {stage_spec.inputs}")

        # --- Assertions ---
        # 1. Check mock_registry call
        mock_registry.list.assert_called_once_with(limit=20)

        # 2. Check LLM calls (using the mock_llm_provider's internal tracking if available, or by checking side effects)
        #    For MockLLMProvider, we can check if all predefined keys were accessed.
        #    This requires MockLLMProvider to track which keys were used or for us to check its call history.
        #    Let's assume MockLLMProvider might not have advanced call tracking, so we verify by the successful plan generation for now.
        #    A more robust test would involve a MockLLMProvider that records calls.
        
        # 3. Assert plan structure
        assert generated_plan.id is not None
        assert generated_plan.name.startswith("Plan for: Develop a Python CLI tool")
        assert len(generated_plan.stages) == 4 # Based on 4 decomposed tasks
        assert generated_plan.start_stage is not None
        
        # 4. Assert stage details (example for the first and last stage based on mock sequencing "T0,T1,T2,T3")
        #    Need to find the actual stage IDs as they are UUID-based
        stage_ids_in_order = []
        current_s_id = generated_plan.start_stage
        while current_s_id:
            stage_ids_in_order.append(current_s_id)
            current_s_id = generated_plan.stages[current_s_id].next_stage
        
        assert len(stage_ids_in_order) == 4

        first_stage_id = stage_ids_in_order[0]
        first_stage_spec = generated_plan.stages[first_stage_id]
        assert first_stage_spec.name.startswith("Design Naming_Scheme storage format.")
        assert first_stage_spec.agent_id == "generic_task_agent_ex"
        assert first_stage_spec.number == 0.0
        assert first_stage_spec.next_stage == stage_ids_in_order[1]

        last_stage_id = stage_ids_in_order[3]
        last_stage_spec = generated_plan.stages[last_stage_id]
        assert last_stage_spec.name.startswith("Implement Naming_Scheme retrieval function.")
        assert last_stage_spec.agent_id == "generic_task_agent_ex"
        assert last_stage_spec.number == 3.0
        assert last_stage_spec.next_stage is None

        print("MasterPlannerAgent example_usage assertions passed.")

    except Exception as e:
        print(f"Error generating plan in example_usage: {e}")
        raise # Re-raise to fail the example if it's acting as a test
    finally:
        print("--- Finished MasterPlannerAgent example_usage ---")

if __name__ == "__main__":
    import asyncio
    from unittest.mock import MagicMock
    # This is for development/testing of the agent structure itself
    # It requires a running event loop if any methods are async (which execute is)
    # For now, commenting out direct run, as it needs proper mock setup or providers.
    # asyncio.run(example_usage())
    print(f"MasterPlannerAgent file created. Agent ID: {AGENT_ID}")
    print("Agent Card (JSON Schema for Input):")
    # print(json.dumps(master_planner_agent_card.input_schema, indent=2)) # Pydantic v2
    print("Agent Card (JSON Schema for Output):")
    # print(json.dumps(master_planner_agent_card.output_schema, indent=2)) # Pydantic v2
    pass 