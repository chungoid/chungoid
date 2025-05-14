import pytest
from unittest.mock import MagicMock, AsyncMock, ANY
import json

from chungoid.runtime.agents.master_planner_agent import MasterPlannerAgent, PROMPTS_DIR
from chungoid.schemas.user_goal_schemas import UserGoalRequest
from chungoid.utils.llm_provider import MockLLMProvider, LLMProvider
from chungoid.utils.agent_resolver import AgentProvider
from chungoid.utils.agent_registry import AgentCard

# --- Fixtures ---

@pytest.fixture
def mock_llm_provider() -> MockLLMProvider:
    return MockLLMProvider(predefined_responses={})

@pytest.fixture
def mock_agent_provider() -> MagicMock:
    provider = MagicMock(spec=AgentProvider)
    mock_registry = MagicMock()
    sample_card_1_data = {
        "agent_id": "generic_task_agent_test", 
        "name": "Generic Task Agent Test", 
        "description": "Test generic agent.",
        "capabilities": ["general_test"], 
        "tags": [],
        "input_schema": {"title": "Generic Test Input", "description": "Input for generic test agent"},
        "output_schema": {"title": "Generic Test Output", "description": "Output from generic test agent"}
    }
    sample_card_1 = AgentCard(**sample_card_1_data)
    mock_registry.list.return_value = [sample_card_1]
    provider.search_agents = AsyncMock(return_value=[(sample_card_1, 0.9)])
    provider._registry = mock_registry
    return provider

@pytest.fixture
def planner(mock_agent_provider: AgentProvider, mock_llm_provider: LLMProvider) -> MasterPlannerAgent:
    return MasterPlannerAgent(agent_provider=mock_agent_provider, llm_provider=mock_llm_provider)

# --- Test Class ---

class TestMasterPlannerAgent:

    @pytest.mark.asyncio
    async def test_decompose_goal_success(self, planner: MasterPlannerAgent, mock_llm_provider: MockLLMProvider):
        sample_goal = UserGoalRequest(
            goal_description="Test goal to decompose.",
            target_platform="Test Platform",
            key_constraints={"format": "test"}
        )
        decomposition_prompt_template = (PROMPTS_DIR / "decomposition_prompt.txt").read_text()
        expected_prompt = decomposition_prompt_template.format(
            goal_description=sample_goal.goal_description,
            target_platform=sample_goal.target_platform,
            key_constraints=str(sample_goal.key_constraints)
        ).strip()
        mock_llm_response = "1. First task.\\n2. Second task.\\n- Third task (different prefix)"
        mock_llm_provider.predefined_responses[expected_prompt] = mock_llm_response
        decomposed_tasks = await planner._decompose_goal(sample_goal)
        assert decomposed_tasks == ["First task.", "Second task.", "Third task (different prefix)"]

    @pytest.mark.asyncio
    async def test_select_agent_success_dynamic_agents(self, planner: MasterPlannerAgent, mock_agent_provider: MagicMock, mock_llm_provider: MockLLMProvider):
        task_desc = "Test task for agent selection."
        original_goal = UserGoalRequest(goal_description="Original test goal.")
        
        card_data_from_search = mock_agent_provider.search_agents.return_value[0][0]
        capabilities_str = ", ".join(card_data_from_search.capabilities)
        
        # Match agent's logic: prefer title over description for schema summaries
        input_schema_dict = card_data_from_search.input_schema or {}
        output_schema_dict = card_data_from_search.output_schema or {}
        input_desc = input_schema_dict.get('title', input_schema_dict.get('description', 'N/A'))
        output_desc = output_schema_dict.get('title', output_schema_dict.get('description', 'N/A'))

        # Ensure each part is stripped and then joined to avoid subtle whitespace issues
        details_lines = [
            f"- Agent ID: {card_data_from_search.agent_id}",
            f"  Name: {card_data_from_search.name}",
            f"  Description: {card_data_from_search.description}",
            f"  Capabilities: {capabilities_str}",
            f"  Expected Input Summary: {input_desc}",
            f"  Expected Output Summary: {output_desc}"
        ]
        expected_formatted_agent_details = "\n".join(line.strip() for line in details_lines).strip()
    
        agent_selection_prompt_template = (PROMPTS_DIR / "agent_selection_prompt.txt").read_text().strip() # Strip template here
        
        expected_prompt = agent_selection_prompt_template.format(
            original_user_goal_description=original_goal.goal_description,
            current_decomposed_task_description=task_desc,
            candidate_agents_details_formatted=expected_formatted_agent_details.strip() # Strip details again just in case
        ).strip() # Strip final result

        mock_llm_response_json = {"selected_agent_ids": ["generic_task_agent_test"], "justification": "This agent is perfect for testing."}
        mock_llm_provider.predefined_responses[expected_prompt] = json.dumps(mock_llm_response_json)
        
        generate_spy = AsyncMock(wraps=mock_llm_provider.generate)
        mock_llm_provider.generate = generate_spy

        result = await planner._select_agent_for_task(task_desc, original_goal)

        # Debugging prompt comparison - MOVED AFTER THE CALL THAT TRIGGERS THE SPY
        actual_call_args = generate_spy.call_args
        if actual_call_args:
            actual_prompt_for_llm = actual_call_args[0][0]
            print(f"DEBUG: Expected Prompt last 10: '{expected_prompt[-10:]}' | Hex: {''.join(f'{ord(c):02x}' for c in expected_prompt[-10:])}")
            print(f"DEBUG: Actual   Prompt last 10: '{actual_prompt_for_llm[-10:]}' | Hex: {''.join(f'{ord(c):02x}' for c in actual_prompt_for_llm[-10:])}")
            if expected_prompt != actual_prompt_for_llm:
                # Find first differing character
                for i, (e_char, a_char) in enumerate(zip(expected_prompt, actual_prompt_for_llm)):
                    if e_char != a_char:
                        print(f"DEBUG: Prompts differ at index {i}: expected '{e_char}' (0x{ord(e_char):02x}), actual '{a_char}' (0x{ord(a_char):02x})")
                        # Print context around difference
                        print(f"DEBUG: Expected context: ...{expected_prompt[max(0, i-20):i+20]}...") # Increased context
                        print(f"DEBUG: Actual context  : ...{actual_prompt_for_llm[max(0, i-20):i+20]}...") # Increased context
                        break # This break is now correctly inside the for loop
                if len(expected_prompt) != len(actual_prompt_for_llm):
                    print(f"DEBUG: Prompts differ in length. Expected: {len(expected_prompt)}, Actual: {len(actual_prompt_for_llm)}")
                    print(f"DEBUG: Expected ends with: \n''{expected_prompt[-40:]}''") # More end context
                    print(f"DEBUG: Actual   ends with: \n''{actual_prompt_for_llm[-40:]}''")   # More end context
        else:
            print("DEBUG: generate_spy was not called.")

        mock_agent_provider.search_agents.assert_called_once_with(query_text=task_desc, n_results=5)
        generate_spy.assert_called_once_with(expected_prompt, max_tokens=500)
        assert result == mock_llm_response_json

    @pytest.mark.asyncio
    async def test_sequence_tasks_success(self, planner: MasterPlannerAgent, mock_llm_provider: MockLLMProvider):
        original_goal = UserGoalRequest(goal_description="Goal for sequencing.")
        tasks_with_agents = [
            {"task_description": "Task A", "selected_agent_id": "agent1", "justification": "..."},
            {"task_description": "Task B", "selected_agent_id": "agent2", "justification": "..."},
            {"task_description": "Task C", "selected_agent_id": "agent3", "justification": "..."}
        ]
        expected_tasks_for_prompt_str = (
            "T0: Task A (Assigned Agent: agent1)\\n"
            "T1: Task B (Assigned Agent: agent2)\\n"
            "T2: Task C (Assigned Agent: agent3)"
        )
        sequencing_prompt_template = (PROMPTS_DIR / "sequencing_prompt.txt").read_text()
        expected_prompt = sequencing_prompt_template.format(
            goal_description=original_goal.goal_description,
            tasks_to_sequence=expected_tasks_for_prompt_str
        ).strip()
        mock_llm_response = "T1,T2,T0"
        mock_llm_provider.predefined_responses[expected_prompt] = mock_llm_response
        
        result = await planner._sequence_tasks(tasks_with_agents, original_goal)
        
        assert len(result) == 3
        assert result[0]["task_description"] == "Task B" and result[0]["temp_task_id"] == "T1" and result[0]["next_temporary_task_id"] == "T2"
        assert result[1]["task_description"] == "Task C" and result[1]["temp_task_id"] == "T2" and result[1]["next_temporary_task_id"] == "T0"
        assert result[2]["task_description"] == "Task A" and result[2]["temp_task_id"] == "T0" and result[2]["next_temporary_task_id"] == "FINAL_STEP"

    def test_format_plan_success(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(
            goal_id="test_goal_123",
            goal_description="Create a wonderful new app.",
            target_platform="cloud",
            key_constraints={"budget": "low"}
        )
        ordered_tasks = [
            {"task_description": "Design the UI", "selected_agent_id": "ui_designer_agent", "temp_task_id": "T0_design", "next_temporary_task_id": "T1_backend"},
            {"task_description": "Develop backend API", "selected_agent_id": "backend_dev_agent", "temp_task_id": "T1_backend", "next_temporary_task_id": "T2_frontend"},
            {"task_description": "Develop frontend UI", "selected_agent_id": "frontend_dev_agent", "temp_task_id": "T2_frontend", "next_temporary_task_id": "FINAL_STEP"}
        ]
        plan = planner._format_plan(ordered_tasks, user_goal)
        assert plan.id.startswith("mep_test_goal_123_")
        assert plan.name == "Plan for: Create a wonderful new app...."
        assert plan.original_request == user_goal
        assert len(plan.stages) == 3
        assert plan.start_stage == "T0_design"
        
        stage0 = plan.stages["T0_design"]
        assert stage0.name == "Stage 0: Design the UI"
        assert stage0.agent_id == "ui_designer_agent" and stage0.next_stage == "T1_backend"
        assert stage0.inputs == {"task_description": "Design the UI", "user_goal_details": "context.original_request"}
        
        stage1 = plan.stages["T1_backend"]
        assert stage1.name == "Stage 1: Develop backend API"
        assert stage1.agent_id == "backend_dev_agent" 
        assert stage1.next_stage == "T2_frontend"
        assert stage1.inputs == {"task_description": "Develop backend API", "previous_stage_outputs": "T0_design"}

        stage2 = plan.stages["T2_frontend"]
        assert stage2.name == "Stage 2: Develop frontend UI"
        assert stage2.agent_id == "frontend_dev_agent"
        assert stage2.next_stage is None
        assert stage2.inputs == {"task_description": "Develop frontend UI", "previous_stage_outputs": "T1_backend"}

    def test_format_plan_with_explicit_inputs_original_request(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(goal_id="g1", goal_description="Test goal")
        task_data = [{
            "temp_task_id": "t1",
            "task_description": "Task 1",
            "selected_agent_id": "agent_A",
            "next_temporary_task_id": "FINAL_STEP",
            "explicit_inputs": {
                "goal_desc_explicit": "context.original_request.goal_description",
                "fixed_val": "literal_here"
            }
        }]
        plan = planner._format_plan(task_data, user_goal)
        assert plan.stages["t1"].inputs["goal_desc_explicit"] == "context.original_request.goal_description"
        assert plan.stages["t1"].inputs["fixed_val"] == "literal_here"
        assert plan.stages["t1"].inputs["user_goal_details"] == "context.original_request"
        assert plan.stages["t1"].inputs["task_description"] == "Task 1"

    def test_format_plan_with_explicit_inputs_previous_outputs(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(goal_id="g1", goal_description="Test goal")
        task_data = [
            {
                "temp_task_id": "t1", "task_description": "Task 1", 
                "selected_agent_id": "agent_A", "next_temporary_task_id": "t2",
            },
            {
                "temp_task_id": "t2", "task_description": "Task 2", 
                "selected_agent_id": "agent_B", "next_temporary_task_id": "FINAL_STEP",
                "explicit_inputs": {
                    "data_from_t1": "context.outputs.t1.some_output_key",
                    "task_specific_literal": 123
                }
            }
        ]
        plan = planner._format_plan(task_data, user_goal)
        stage_t2_inputs = plan.stages["t2"].inputs
        assert stage_t2_inputs["data_from_t1"] == "context.outputs.t1.some_output_key"
        assert stage_t2_inputs["task_specific_literal"] == 123
        assert stage_t2_inputs["previous_stage_outputs"] == "t1"
        assert stage_t2_inputs["task_description"] == "Task 2"

    def test_format_plan_with_explicit_inputs_override_defaults(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(goal_id="g1", goal_description="Test goal")
        task_data = [
            { 
                "temp_task_id": "t1", "task_description": "Original T1 Desc", 
                "selected_agent_id": "agent_A", "next_temporary_task_id": "t2",
                "explicit_inputs": {
                    "task_description": "Overridden T1 Desc", 
                    "user_goal_details": "context.original_request.goal_id" 
                }
            },
            { 
                "temp_task_id": "t2", "task_description": "Original T2 Desc", 
                "selected_agent_id": "agent_B", "next_temporary_task_id": "FINAL_STEP",
                "explicit_inputs": {
                    "task_description": "Overridden T2 Desc", 
                    "previous_stage_outputs": "context.outputs.t1.full_object" 
                }
            }
        ]
        plan = planner._format_plan(task_data, user_goal)
        
        stage_t1_inputs = plan.stages["t1"].inputs
        assert stage_t1_inputs["task_description"] == "Overridden T1 Desc"
        assert stage_t1_inputs["user_goal_details"] == "context.original_request.goal_id"
        
        stage_t2_inputs = plan.stages["t2"].inputs
        assert stage_t2_inputs["task_description"] == "Overridden T2 Desc"
        assert stage_t2_inputs["previous_stage_outputs"] == "context.outputs.t1.full_object"

    def test_format_plan_with_invalid_explicit_inputs_type(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(goal_id="g1", goal_description="Test goal")
        task_data = [{
            "temp_task_id": "t1",
            "task_description": "Task 1",
            "selected_agent_id": "agent_A",
            "next_temporary_task_id": "FINAL_STEP",
            "explicit_inputs": "this_is_not_a_dict" 
        }]
        plan = planner._format_plan(task_data, user_goal)
        assert plan.stages["t1"].inputs["task_description"] == "Task 1"
        assert plan.stages["t1"].inputs["user_goal_details"] == "context.original_request"
        assert len(plan.stages["t1"].inputs) == 2

    def test_format_plan_explicit_inputs_missing(self, planner: MasterPlannerAgent):
        user_goal = UserGoalRequest(goal_id="g1", goal_description="Test goal")
        task_data = [{
            "temp_task_id": "t1",
            "task_description": "Task 1",
            "selected_agent_id": "agent_A",
            "next_temporary_task_id": "FINAL_STEP"
        }]
        plan = planner._format_plan(task_data, user_goal)
        assert plan.stages["t1"].inputs["task_description"] == "Task 1"
        assert plan.stages["t1"].inputs["user_goal_details"] == "context.original_request"
        assert len(plan.stages["t1"].inputs) == 2

    @pytest.mark.asyncio
    async def test_execute_full_success_path(self, planner: MasterPlannerAgent, mock_llm_provider: MockLLMProvider, mock_agent_provider: MagicMock):
        user_goal = UserGoalRequest(
            goal_id="exec_goal_001",
            goal_description="Execute a full plan successfully."
        )

        decomposed_tasks_desc = ["Task Alpha", "Task Beta"]
        selection_results = [
            {"selected_agent_ids": ["agent_alpha"], "justification": "Alpha selected"},
            {"selected_agent_ids": ["agent_beta"], "justification": "Beta selected"}
        ]
        tasks_after_selection_for_sequencing = [
            {"task_description": "Task Alpha", "selected_agent_id": "agent_alpha", "justification": "Alpha selected"},
            {"task_description": "Task Beta", "selected_agent_id": "agent_beta", "justification": "Beta selected"}
        ]
        ordered_tasks_after_sequencing_for_formatting = [
            {
                "task_description": "Task Beta", "selected_agent_id": "agent_beta", "justification": "Beta selected",
                "temp_task_id": "T0_beta", "next_temporary_task_id": "T1_alpha"
            },
            {
                "task_description": "Task Alpha", "selected_agent_id": "agent_alpha", "justification": "Alpha selected",
                "temp_task_id": "T1_alpha", "next_temporary_task_id": "FINAL_STEP"
            }
        ]

        planner._decompose_goal = AsyncMock(return_value=decomposed_tasks_desc)
        planner._select_agent_for_task = AsyncMock(side_effect=selection_results) 
        planner._sequence_tasks = AsyncMock(return_value=ordered_tasks_after_sequencing_for_formatting)
        
        final_plan = await planner.execute(user_goal)

        planner._decompose_goal.assert_called_once_with(user_goal)
        assert planner._select_agent_for_task.call_count == len(decomposed_tasks_desc)
        planner._select_agent_for_task.assert_any_call(decomposed_tasks_desc[0], user_goal)
        planner._select_agent_for_task.assert_any_call(decomposed_tasks_desc[1], user_goal)
        
        planner._sequence_tasks.assert_called_once_with(tasks_after_selection_for_sequencing, user_goal)

        assert final_plan.id.startswith(f"mep_{user_goal.goal_id}_")
        assert final_plan.name == f"Plan for: {user_goal.goal_description[:50]}..."
        assert len(final_plan.stages) == 2
        assert final_plan.start_stage == "T0_beta"

        assert final_plan.stages["T0_beta"].name == "Stage 0: Task Beta"
        assert final_plan.stages["T0_beta"].agent_id == "agent_beta"
        assert final_plan.stages["T0_beta"].next_stage == "T1_alpha"

        assert final_plan.stages["T1_alpha"].name == "Stage 1: Task Alpha"
        assert final_plan.stages["T1_alpha"].agent_id == "agent_alpha"
        assert final_plan.stages["T1_alpha"].next_stage is None
        assert final_plan.original_request == user_goal

    @pytest.mark.asyncio
    async def test_execute_decomposition_fails(self, planner: MasterPlannerAgent, caplog):
        user_goal = UserGoalRequest(goal_id="exec_goal_002", goal_description="Decomposition fails.")
        planner._decompose_goal = AsyncMock(return_value=[])
        planner._select_agent_for_task = AsyncMock()
        planner._sequence_tasks = AsyncMock()

        import logging 
        with caplog.at_level(logging.WARNING, logger='chungoid.runtime.agents.master_planner_agent'):
            final_plan = await planner.execute(user_goal)

        planner._decompose_goal.assert_called_once_with(user_goal)
        planner._select_agent_for_task.assert_not_called()
        planner._sequence_tasks.assert_not_called()

        assert final_plan.id.startswith(f"error_plan_decomp_{user_goal.goal_id}_")
        assert final_plan.name == f"Failed Plan (Decomposition): {user_goal.goal_description[:30]}..."
        assert final_plan.description == "Goal decomposition resulted in no tasks. No execution plan generated."
        assert len(final_plan.stages) == 0
        assert final_plan.start_stage == ""
        assert final_plan.original_request == user_goal
        assert "Goal decomposition failed or returned no tasks. Cannot create a plan." in caplog.text

    @pytest.mark.asyncio
    async def test_execute_agent_selection_fails_all(self, planner: MasterPlannerAgent, caplog):
        user_goal = UserGoalRequest(goal_id="exec_goal_003", goal_description="All agent selections fail.")
        decomposed_tasks = ["Task 1", "Task 2"]
        planner._decompose_goal = AsyncMock(return_value=decomposed_tasks)
        planner._select_agent_for_task = AsyncMock(return_value={"selected_agent_ids": ["NO_SUITABLE_AGENT"], "justification": "None found"})
        planner._sequence_tasks = AsyncMock()

        import logging
        with caplog.at_level(logging.WARNING, logger='chungoid.runtime.agents.master_planner_agent'):
            final_plan = await planner.execute(user_goal)

        planner._decompose_goal.assert_called_once_with(user_goal)
        assert planner._select_agent_for_task.call_count == len(decomposed_tasks)
        planner._sequence_tasks.assert_not_called()

        assert final_plan.id.startswith(f"error_plan_selection_{user_goal.goal_id}_")
        assert final_plan.name == f"Failed Plan (Agent Selection): {user_goal.goal_description[:30]}..."
        assert final_plan.description == "Agent selection phase resulted in no tasks with assigned agents."
        assert len(final_plan.stages) == 0
        assert final_plan.original_request == user_goal
        assert "Agent selection phase resulted in no tasks with assigned agents. Cannot create a meaningful plan." in caplog.text

    @pytest.mark.asyncio
    async def test_execute_sequencing_fails(self, planner: MasterPlannerAgent, caplog):
        user_goal = UserGoalRequest(goal_id="exec_goal_004", goal_description="Sequencing fails.")
        decomposed_tasks = ["Task One", "Task Two"]
        selection_results = [
            {"selected_agent_ids": ["agent_one"], "justification": "Selected"},
            {"selected_agent_ids": ["agent_two"], "justification": "Selected"}
        ]
        
        planner._decompose_goal = AsyncMock(return_value=decomposed_tasks)
        planner._select_agent_for_task = AsyncMock(side_effect=selection_results)
        planner._sequence_tasks = AsyncMock(return_value=[])

        import logging
        with caplog.at_level(logging.ERROR, logger='chungoid.runtime.agents.master_planner_agent'): 
            final_plan = await planner.execute(user_goal)

        planner._decompose_goal.assert_called_once_with(user_goal)
        assert planner._select_agent_for_task.call_count == len(decomposed_tasks)
        planner._sequence_tasks.assert_called_once() 

        assert final_plan.id.startswith(f"error_plan_sequencing_{user_goal.goal_id}_")
        assert final_plan.name == f"Failed Plan (Sequencing): {user_goal.goal_description[:30]}..."
        assert final_plan.description == "Task sequencing phase resulted in no ordered tasks."
        assert len(final_plan.stages) == 0
        assert final_plan.original_request == user_goal
        assert "Task sequencing resulted in no ordered tasks. Cannot create a plan." in caplog.text

    @pytest.mark.asyncio
    async def test_execute_calls_sub_methods(self, planner: MasterPlannerAgent, mock_llm_provider: MockLLMProvider, mock_agent_provider: MagicMock):
        user_goal = UserGoalRequest(goal_id="exec_call_test", goal_description="Test execute flow.")
        planner._decompose_goal = AsyncMock(return_value=["Task 1"])
        planner._select_agent_for_task = AsyncMock(return_value={"selected_agent_ids": ["agent_1"], "justification": "Selected"})
        planner._sequence_tasks = AsyncMock(return_value=[
            {"task_description": "Task 1", "selected_agent_id": "agent_1", "temp_task_id": "T0", "next_temporary_task_id": "FINAL_STEP"}
        ])
        format_plan_spy = MagicMock(wraps=planner._format_plan)
        planner._format_plan = format_plan_spy
        
        await planner.execute(user_goal)

        planner._decompose_goal.assert_called_once_with(user_goal)
        planner._select_agent_for_task.assert_called_once_with("Task 1", user_goal)
        planner._sequence_tasks.assert_called_once_with(
            [{"task_description": "Task 1", "selected_agent_id": "agent_1", "justification": "Selected"}], 
            user_goal
        )
        format_plan_spy.assert_called_once_with(
            [{"task_description": "Task 1", "selected_agent_id": "agent_1", "temp_task_id": "T0", "next_temporary_task_id": "FINAL_STEP"}],
            user_goal
        )
    
    pass 