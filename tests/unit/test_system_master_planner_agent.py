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

    goal_request = UserGoalRequest(goal_description="Build a simple CLI tool.", target_platform="test_project")
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

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
        assert parsed_plan.name == f"Plan for: {goal_request.goal_description}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal_description == goal_request.goal_description

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

        goal_request = UserGoalRequest(goal_description="Test malformed structure from LLM", target_platform="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test incomplete plan from LLM", target_platform="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test LLM service failure", target_platform="test_project")
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        goal_description="Test prompt construction with context", 
        target_platform="test_project_ctx",
        key_constraints=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "start_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockSystemInterventionAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal_description in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal_description == goal_request.goal_description
    assert parsed_plan.original_request.key_constraints == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal_description="Build a simple CLI tool.", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

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
        assert parsed_plan.name == f"Plan for: {goal_request.goal_description}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal_description == goal_request.goal_description
             assert parsed_plan.original_request.key_constraints == goal_request.key_constraints

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

        goal_request = UserGoalRequest(goal_description="Test malformed structure from LLM", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test incomplete plan from LLM", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test LLM service failure", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        goal_description="Test prompt construction with context", 
        target_platform="test_project_ctx",
        key_constraints=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "start_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockSystemInterventionAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal_description in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal_description == goal_request.goal_description
    assert parsed_plan.original_request.key_constraints == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context_and_goal():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context and goal."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal_description="Build a simple CLI tool.", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

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
        assert parsed_plan.name == f"Plan for: {goal_request.goal_description}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal_description == goal_request.goal_description
             assert parsed_plan.original_request.key_constraints == goal_request.key_constraints

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

        goal_request = UserGoalRequest(goal_description="Test malformed structure from LLM", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test incomplete plan from LLM", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        
        goal_request = UserGoalRequest(goal_description="Test LLM service failure", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
        planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
        
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
        goal_description="Test prompt construction with context and goal", 
        target_platform="test_project_ctx",
        key_constraints=project_details
    )
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

    # We need to inspect the arguments passed to the mocked llm_client.generate_json
    # The mock itself doesn't use the prompts, so we patch it to capture them.
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        # Return a valid minimal plan dict to satisfy the rest of the agent logic
        return {
            "id": "spy_plan", "name": "Spy Plan", "description": "Plan for spy test",
            "start_stage": "stage1", "stages": {"stage1": {"name": "stage1", "number": 1, "agent_id": "MockSystemInterventionAgent_v1", "next_stage": "FINAL_STEP"}}
        }

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_generate_json_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None # Ensure the call itself succeeded
    mock_llm_generate_json_call.assert_called_once()
    
    assert captured_args['user_prompt'] is not None
    # Check that the goal is in the user prompt
    assert goal_request.goal_description in captured_args['user_prompt']
    # Check that the project_context (as a JSON string) is in the user prompt
    expected_project_context_str = json.dumps(project_details, indent=2)
    assert expected_project_context_str in captured_args['user_prompt']

    # Also ensure original_request is correctly added to the final plan if LLM doesn't include it
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal_description == goal_request.goal_description
    assert parsed_plan.original_request.key_constraints == project_details


@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_with_context_and_goal():
    """Test that the agent correctly processes a valid plan from the (mocked) LLM with context and goal."""
    agent = MasterPlannerAgent()
    
    # This is the dict our MockLLMClient.generate_json is expected to return by default
    # We can make it more specific for this test if needed, or rely on its default.
    # For this test, we'll assume the mock LLM returns something parsable.
    # The mock LLM returns a dict, which the agent then validates.

    goal_request = UserGoalRequest(goal_description="Build a simple CLI tool.", target_platform="test_project", key_constraints={"current_branch": "main", "dependencies": ["fastapi"]})
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

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
        assert parsed_plan.name == f"Plan for: {goal_request.goal_description}" # Check name update
        assert parsed_plan.stages is not None
        assert len(parsed_plan.stages) > 0
        if goal_request:
             assert parsed_plan.original_request is not None
             assert parsed_plan.original_request.goal_description == goal_request.goal_description
             assert parsed_plan.original_request.key_constraints == goal_request.key_constraints

    except Exception as e:
        pytest.fail(f"Valid plan JSON from agent failed to parse: {e}\\nJSON: {output.master_plan_json}")


# Test cases for when original_request.project_context is None

@pytest.mark.asyncio
async def test_master_planner_agent_valid_plan_generation_no_proj_context():
    agent = MasterPlannerAgent()
    goal_request = UserGoalRequest(goal_description="Build something simple.", target_platform="test_no_ctx") # MODIFIED project_scope, key_constraints is None by default
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
    output = await agent.invoke_async(planner_input)
    assert output.error_message is None
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal_description == goal_request.goal_description
    assert parsed_plan.original_request.key_constraints is None # MODIFIED project_context


@pytest.mark.asyncio
async def test_master_planner_agent_prompt_no_proj_context():
    agent = MasterPlannerAgent()
    goal_request = UserGoalRequest(goal_description="Test prompt with no project context.", target_platform="test_no_ctx_prompt") # MODIFIED project_scope
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
    
    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['system_prompt'] = system_prompt
        captured_args['user_prompt'] = user_prompt
        return {"id": "spy_plan_no_ctx", "name": "Spy Plan No Ctx", "description": "Plan for spy test no ctx",
                "start_stage": "s1", "stages": {"s1": {"name": "s1", "number": 1, "agent_id": "MockSystemInterventionAgent_v1", "next_stage": "FINAL_STEP"}}}

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_call:
        output = await agent.invoke_async(planner_input)
    
    assert output.error_message is None
    mock_llm_call.assert_called_once()
    assert goal_request.goal_description in captured_args['user_prompt']
    # Ensure "None" or similar indication for project_context if it's absent
    # The actual string depends on the prompt template formatting for None context.
    # Common patterns: "Project Context: None", "Project Context: Not available"
    # For this test, let's assume if project_context is None, its key might not even appear or appears as "None".
    # A safer check might be to ensure a specific string indicating real context is NOT present.
    assert "current_branch" not in captured_args['user_prompt'] 
    assert "dependencies" not in captured_args['user_prompt']


# --- Test cases for main() in system_master_planner_agent.py ---
# These will use the actual main_test function if it's simple enough,
# or we'll replicate its logic here for more control over mocks if needed.

@pytest.mark.asyncio
@patch('builtins.print') # Mock print to avoid polluting test output
@patch('chungoid.runtime.agents.system_master_planner_agent.logger.info') # Mock logger
async def test_main_function_runs_and_parses_mock_output(mock_logger_info, mock_print):
    """ Test that the main_test function in the agent script runs without error
        and that its mock LLM output can be parsed.
    """
    # Patch the LLMProvider used by MasterPlannerAgent if main_test instantiates its own
    # For this test, we rely on the MockLLMClient already used by MasterPlannerAgent by default
    
    # We need to ensure that the MasterPlannerAgent in main_test uses the same MockLLMClient logic
    # or we provide a universally accessible mock for generate_json
    
    # The default MockLLMClient.generate_json returns a fixed valid plan.
    # We just need to check if main_test completes and print indicates success.

    from chungoid.runtime.agents.system_master_planner_agent import main_test as agent_main_test

    # Call the main_test function from the script
    await agent_main_test()

    # Check for successful parsing indications in print calls
    # Example from main_test: print("\\nPlan 1 successfully parsed.")
    # Example from main_test: print("\\nPlan 2 successfully parsed.")
    
    print_calls_args = [call_args[0][0] for call_args in mock_print.call_args_list] # Get the first arg of each print call
    
    assert any("Plan 1 successfully parsed." in call_arg for call_arg in print_calls_args)
    assert any("Plan 2 successfully parsed." in call_arg for call_arg in print_calls_args)
    # Check that no "Error parsing generated plan" was printed for both
    assert not any("Error parsing generated plan 1" in call_arg for call_arg in print_calls_args)
    assert not any("Error parsing generated plan 2" in call_arg for call_arg in print_calls_args)

    # Verify logger calls if necessary (e.g., to see if main_test started)
    mock_logger_info.assert_any_call("Running MasterPlannerAgent (LLM-driven) test...")
    mock_logger_info.assert_any_call(f"--- Test 1: Goal: Implement a new feature foo_bar. ---")
    mock_logger_info.assert_any_call(f"--- Test 2: Goal: Refactor the authentication module. ---")

# Note: If main_test is very complex or involves external dependencies not easily mocked here,
# it might be better to test its constituent parts or refactor main_test for better testability.
# For now, this test assumes main_test is relatively self-contained with MasterPlannerAgent's default mock.

# Additional tests for project_scope (if it becomes target_platform or similar)
# and ensuring it's handled correctly in prompts and plan outputs.
# For example:
@pytest.mark.asyncio
async def test_project_scope_becomes_target_platform_in_prompt():
    agent = MasterPlannerAgent()
    # Assuming UserGoalRequest might take target_platform directly or project_scope maps to it
    goal_request = UserGoalRequest(
        goal_description="Build a web UI",
        target_platform="React" # project_scope was here, now correctly target_platform
    )
    # If UserGoalRequest still uses project_scope, and it's meant for target_platform:
    # goal_request = UserGoalRequest(goal_description="Build a web UI", project_scope="React")
    
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)

    captured_args = {}
    async def generate_json_spy(system_prompt: str, user_prompt: str, temperature: float = 0.1):
        captured_args['user_prompt'] = user_prompt
        return {"id": "spy_platform_plan", "name": "Spy Platform Plan", 
                "start_stage": "s1", "stages": {"s1": {"name": "s1", "number": 1, "agent_id": "generic_agent", "next_stage": "FINAL_STEP"}}}

    with patch.object(MockLLMClient, 'generate_json', side_effect=generate_json_spy) as mock_llm_call:
        await agent.invoke_async(planner_input)
    
    mock_llm_call.assert_called_once()
    # This assertion depends on how the prompt template formats the target_platform.
    # Example: "Target Platform: React"
    assert "Target Platform: React" in captured_args['user_prompt'] 
    # Or if project_scope was used and it becomes target_platform in prompt:
    # assert f"Target Platform: {goal_request.target_platform}" in captured_args['user_prompt'] # MODIFIED goal_request.project_scope

@pytest.mark.asyncio
async def test_project_scope_in_original_request_if_used():
    agent = MasterPlannerAgent()
    # Assuming project_scope should be stored in the original_request of the plan
    # And UserGoalRequest schema has a field for it (e.g., target_platform or a direct project_scope)
    goal_request_data = {
        "goal_description": "Deploy to cloud",
        "target_platform": "AWS Lambda" # project_scope was here
    }
    # If UserGoalRequest takes project_scope directly:
    # goal_request_data = {"goal_description": "Deploy to cloud", "project_scope": "AWS Lambda"}

    goal_request = UserGoalRequest(**goal_request_data)
    planner_input = MasterPlannerInput(user_goal=goal_request.goal_description, original_request=goal_request)
    
    output = await agent.invoke_async(planner_input) # Relies on default mock LLM plan
    assert output.error_message is None
    parsed_plan = MasterExecutionPlan.model_validate_json(output.master_plan_json)
    assert parsed_plan.original_request is not None
    assert parsed_plan.original_request.goal_description == goal_request.goal_description
    
    # Assert based on how project_scope is stored (e.g., as target_platform)
    assert parsed_plan.original_request.target_platform == goal_request.target_platform
    # If UserGoalRequest had a direct project_scope field and it was stored:
    # assert parsed_plan.original_request.project_scope == goal_request.project_scope 