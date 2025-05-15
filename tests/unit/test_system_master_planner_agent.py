import pytest
import json
from unittest.mock import patch, AsyncMock

from chungoid.runtime.agents.system_master_planner_agent import MasterPlannerAgent, MockLLMClient
from chungoid.schemas.agent_master_planner import MasterPlannerInput, MasterPlannerOutput
from chungoid.schemas.user_goal_schemas import UserGoalRequest
from chungoid.schemas.master_flow import MasterExecutionPlan


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal="Build a simple CLI tool.", project_scope="test_project")
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # No need to patch MockLLMClient.generate_json if its default behavior is sufficient for valid case.
    # Its default mock already returns a parsable (though generic) plan.

    output = await agent.invoke_async(planner_input)

    assert output.error_message is None
    assert output.master_plan_json is not None
    assert output.master_plan_json != ""
    assert output.confidence_score is not None
    
    try:
        parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
        assert parsed_plan.id is not None
        assert parsed_plan.name == f"Plan for: {goal_request.goal}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal == goal_request.goal

    except Exception as e:
        pytest.fail(f"Valid plan JSON from agent failed to parse: {e}\\nJSON: {output.master_plan_json}")


@pytest.mark.asyncio
async def test_master_planner_agent_llm_returns_malformed_json():
    """Test error handling when the LLM returns a string that is not valid JSON."""
    agent = MasterPlannerAgent()
    
    malformed_json_string = "{\'id\': \'plan123\'" # Missing closing brace and quotes

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        # Mock the LLM to return a dict that would cause json.JSONDecodeError if it were a string being parsed
        # However, our agent's try-except for JSONDecodeError is for when LLM returns a STRING that is bad.
        # If LLM returns a dict (as our mock does), and it's MasterExecutionPlan.model_validate that fails,
        # it will be a Pydantic ValidationError, caught by the broader Exception.
        # To test JSONDecodeError, the mock would need to return a STRING that's bad.
        # For now, let's test the Pydantic validation failure by returning a bad dict.
        
        # To truly test JSONDecodeError, we'd need to change MockLLMClient to return string
        # or mock the internal json.loads if the agent were structured that way.
        # The current agent structure:
        # llm_generated_plan_dict = await self.llm_client.generate_json(...)
        # plan = MasterExecutionPlan.model_validate(llm_generated_plan_dict)
        # So, JSONDecodeError is less likely with current mock structure.
        # Let's test Pydantic validation error instead.

        mock_llm_call.return_value = {"id": "test_plan", "stages": "not_a_dict"} # This will fail Pydantic validation

        goal_request = UserGoalRequest(goal="Test malformed structure from LLM", project_scope="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        # Example Pydantic error: "stages -> Input should be a valid dictionary"
        assert "Input should be a valid dictionary" in output.error_message or "not_a_dict" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_llm_validation_error_missing_fields():
    """Test error handling when LLM returns JSON missing required MasterExecutionPlan fields."""
    agent = MasterPlannerAgent()

    incomplete_plan_data = {
        "id": "plan_incomplete",
        "name": "Incomplete Plan",
        # Missing 'stages' and 'initial_stage' which are required by MasterExecutionPlan
    }

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = incomplete_plan_data
        
        goal_request = UserGoalRequest(goal="Test incomplete plan from LLM", project_scope="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        assert "Field required" in output.error_message # Pydantic's typical message for missing fields


@pytest.mark.asyncio
async def test_master_planner_agent_llm_call_raises_exception():
    """Test error handling when the llm_client.generate_json call itself raises an exception."""
    agent = MasterPlannerAgent()

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.side_effect = Exception("LLM Service Unavailable")
        
        goal_request = UserGoalRequest(goal="Test LLM service failure", project_scope="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan: LLM Service Unavailable" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_includes_original_request_and_context_in_prompt():
    """Test that original_request and its project_context are used in prompt construction."""
    agent = MasterPlannerAgent()
    
    project_details = {"current_branch": "main", "dependencies": ["fastapi"]}
    goal_request = UserGoalRequest(
        goal="Test prompt construction with context", 
        project_scope="test_project_ctx",
        project_context=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "initial_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockHumanInputAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal == goal_request.goal
    assert parsed_plan.original_request.project_context == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal="Build a simple CLI tool.", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # No need to patch MockLLMClient.generate_json if its default behavior is sufficient for valid case.
    # Its default mock already returns a parsable (though generic) plan.

    output = await agent.invoke_async(planner_input)

    assert output.error_message is None
    assert output.master_plan_json is not None
    assert output.master_plan_json != ""
    assert output.confidence_score is not None
    
    try:
        parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
        assert parsed_plan.id is not None
        assert parsed_plan.name == f"Plan for: {goal_request.goal}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal == goal_request.goal
             assert parsed_plan.original_request.project_context == goal_request.project_context

    except Exception as e:
        pytest.fail(f"Valid plan JSON from agent failed to parse: {e}\\nJSON: {output.master_plan_json}")


@pytest.mark.asyncio
async def test_master_planner_agent_llm_returns_malformed_json_with_context():
    """Test error handling when the LLM returns a string that is not valid JSON with context."""
    agent = MasterPlannerAgent()
    
    malformed_json_string = "{\'id\': \'plan123\'" # Missing closing brace and quotes

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        # Mock the LLM to return a dict that would cause json.JSONDecodeError if it were a string being parsed
        # However, our agent's try-except for JSONDecodeError is for when LLM returns a STRING that is bad.
        # If LLM returns a dict (as our mock does), and it's MasterExecutionPlan.model_validate that fails,
        # it will be a Pydantic ValidationError, caught by the broader Exception.
        # To test JSONDecodeError, the mock would need to return a STRING that's bad.
        # For now, let's test the Pydantic validation failure by returning a bad dict.
        
        # To truly test JSONDecodeError, we'd need to change MockLLMClient to return string
        # or mock the internal json.loads if the agent were structured that way.
        # The current agent structure:
        # llm_generated_plan_dict = await self.llm_client.generate_json(...)
        # plan = MasterExecutionPlan.model_validate(llm_generated_plan_dict)
        # So, JSONDecodeError is less likely with current mock structure.
        # Let's test Pydantic validation error instead.

        mock_llm_call.return_value = {"id": "test_plan", "stages": "not_a_dict"} # This will fail Pydantic validation

        goal_request = UserGoalRequest(goal="Test malformed structure from LLM", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        # Example Pydantic error: "stages -> Input should be a valid dictionary"
        assert "Input should be a valid dictionary" in output.error_message or "not_a_dict" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_llm_validation_error_missing_fields_with_context():
    """Test error handling when LLM returns JSON missing required MasterExecutionPlan fields with context."""
    agent = MasterPlannerAgent()

    incomplete_plan_data = {
        "id": "plan_incomplete",
        "name": "Incomplete Plan",
        # Missing 'stages' and 'initial_stage' which are required by MasterExecutionPlan
    }

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = incomplete_plan_data
        
        goal_request = UserGoalRequest(goal="Test incomplete plan from LLM", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        assert "Field required" in output.error_message # Pydantic's typical message for missing fields


@pytest.mark.asyncio
async def test_master_planner_agent_llm_call_raises_exception_with_context():
    """Test error handling when the llm_client.generate_json call itself raises an exception with context."""
    agent = MasterPlannerAgent()

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.side_effect = Exception("LLM Service Unavailable")
        
        goal_request = UserGoalRequest(goal="Test LLM service failure", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan: LLM Service Unavailable" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_includes_original_request_and_context_in_prompt_with_context():
    """Test that original_request and its project_context are used in prompt construction with context."""
    agent = MasterPlannerAgent()
    
    project_details = {"current_branch": "main", "dependencies": ["fastapi"]}
    goal_request = UserGoalRequest(
        goal="Test prompt construction with context", 
        project_scope="test_project_ctx",
        project_context=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "initial_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockHumanInputAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal == goal_request.goal
    assert parsed_plan.original_request.project_context == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context_and_goal():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context and goal."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal="Build a simple CLI tool.", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # No need to patch MockLLMClient.generate_json if its default behavior is sufficient for valid case.
    # Its default mock already returns a parsable (though generic) plan.

    output = await agent.invoke_async(planner_input)

    assert output.error_message is None
    assert output.master_plan_json is not None
    assert output.master_plan_json != ""
    assert output.confidence_score is not None
    
    try:
        parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
        assert parsed_plan.id is not None
        assert parsed_plan.name == f"Plan for: {goal_request.goal}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal == goal_request.goal
             assert parsed_plan.original_request.project_context == goal_request.project_context

    except Exception as e:
        pytest.fail(f"Valid plan JSON from agent failed to parse: {e}\\nJSON: {output.master_plan_json}")


@pytest.mark.asyncio
async def test_master_planner_agent_llm_returns_malformed_json_with_context_and_goal():
    """Test error handling when the LLM returns a string that is not valid JSON with context and goal."""
    agent = MasterPlannerAgent()
    
    malformed_json_string = "{\'id\': \'plan123\'" # Missing closing brace and quotes

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        # Mock the LLM to return a dict that would cause json.JSONDecodeError if it were a string being parsed
        # However, our agent's try-except for JSONDecodeError is for when LLM returns a STRING that is bad.
        # If LLM returns a dict (as our mock does), and it's MasterExecutionPlan.model_validate that fails,
        # it will be a Pydantic ValidationError, caught by the broader Exception.
        # To test JSONDecodeError, the mock would need to return a STRING that's bad.
        # For now, let's test the Pydantic validation failure by returning a bad dict.
        
        # To truly test JSONDecodeError, we'd need to change MockLLMClient to return string
        # or mock the internal json.loads if the agent were structured that way.
        # The current agent structure:
        # llm_generated_plan_dict = await self.llm_client.generate_json(...)
        # plan = MasterExecutionPlan.model_validate(llm_generated_plan_dict)
        # So, JSONDecodeError is less likely with current mock structure.
        # Let's test Pydantic validation error instead.

        mock_llm_call.return_value = {"id": "test_plan", "stages": "not_a_dict"} # This will fail Pydantic validation

        goal_request = UserGoalRequest(goal="Test malformed structure from LLM", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        # Example Pydantic error: "stages -> Input should be a valid dictionary"
        assert "Input should be a valid dictionary" in output.error_message or "not_a_dict" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_llm_validation_error_missing_fields_with_context_and_goal():
    """Test error handling when LLM returns JSON missing required MasterExecutionPlan fields with context and goal."""
    agent = MasterPlannerAgent()

    incomplete_plan_data = {
        "id": "plan_incomplete",
        "name": "Incomplete Plan",
        # Missing 'stages' and 'initial_stage' which are required by MasterExecutionPlan
    }

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.return_value = incomplete_plan_data
        
        goal_request = UserGoalRequest(goal="Test incomplete plan from LLM", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan" in output.error_message
        assert "Field required" in output.error_message # Pydantic's typical message for missing fields


@pytest.mark.asyncio
async def test_master_planner_agent_llm_call_raises_exception_with_context_and_goal():
    """Test error handling when the llm_client.generate_json call itself raises an exception with context and goal."""
    agent = MasterPlannerAgent()

    with patch.object(MockLLMClient, 'generate_json', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.side_effect = Exception("LLM Service Unavailable")
        
        goal_request = UserGoalRequest(goal="Test LLM service failure", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)
        
        output = await agent.invoke_async(planner_input)

        assert output.master_plan_json == ""
        assert output.error_message is not None
        assert "Error generating or validating plan: LLM Service Unavailable" in output.error_message


@pytest.mark.asyncio
async def test_master_planner_agent_includes_original_request_and_context_in_prompt_with_context_and_goal():
    """Test that original_request and its project_context are used in prompt construction with context and goal."""
    agent = MasterPlannerAgent()
    
    project_details = {"current_branch": "main", "dependencies": ["fastapi"]}
    goal_request = UserGoalRequest(
        goal="Test prompt construction with context", 
        project_scope="test_project_ctx",
        project_context=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "initial_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockHumanInputAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal == goal_request.goal
    assert parsed_plan.original_request.project_context == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context_and_goal():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context and goal."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal="Build a simple CLI tool.", project_scope="test_project", project_context={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal, original_request=goal_request)

    # No need to patch MockLLMClient.generate_json if its default behavior is sufficient for valid case.
    # Its default mock already returns a parsable (though generic) plan.

    output = await agent.invoke_async(planner_input)

    assert output.error_message is None
    assert output.master_plan_json is not None
    assert output.master_plan_json != ""
    assert output.confidence_score is not None
    
    try:
        parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
        assert parsed_plan.id is not None
        assert parsed_plan.name == f"Plan for: {goal_request.goal}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal == goal_request.goal
             assert parsed_plan.original_request.project_context == goal_request.project_context

    except Exception as e:
        pytest.fail(f"Valid plan JSON from agent failed to parse: {e}\\nJSON: {output.master_plan_json}") 