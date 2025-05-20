import pytest
import json
import uuid
from unittest.mock import AsyncMock
from typing import List, Optional, Dict, Any # Added Dict, Any

# Agent and schema imports
from chungoid.agents.autonomous_engine.code_debugging_agent import CodeDebuggingAgent_v1, DebuggingTaskInput, DebuggingTaskOutput
from chungoid.schemas.common import ConfidenceScore

# Mock imports
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
# No specific collection constants needed for direct use by debugger agent, context comes via ARCA

# Mock LLM response for CodeDebuggingAgent
MOCK_DEBUGGER_LLM_FIX_PROPOSED_RESPONSE = json.dumps({
    "proposed_solution_type": "MODIFIED_SNIPPET",
    "proposed_code_changes": "def fixed_function():\\n    return 42",
    "explanation_of_fix": "The bug was due to an off-by-one error. The proposed snippet corrects this.",
    "confidence_score_value": 0.8,
    "confidence_score_explanation": "Moderately high confidence based on clarity of error and fix.",
    "areas_of_uncertainty": ["Impact on performance not fully assessed."],
    "status_message": "Fix proposed.", # Matches a field in the old DebuggingTaskOutput, new one has overall `status`
    # New schema fields:
    "confidence_score": 0.8, # Simplified for direct parsing by agent if it expects a float now
    # The agent's output schema is DebuggingTaskOutput which has a structured confidence_score
    # but the prompt output was simplified in one version. The agent will parse this and create the object.
    # Let's assume the LLM gives a simple float for `confidence_score` based on prompt, 
    # and the agent constructs the ConfidenceScore object.
    # However, the design doc output schema for LLM expects a full confidence object.
    # Let's align mock LLM with the prompt's output_schema for LLM directly:
    # "assessment_confidence": {
    #     "value": 0.8,
    #     "level": "High", 
    #     "explanation": "Reasoning..."
    # }
    # The CodeDebuggingAgent prompt output_schema is DebuggingAgentLLMOutput
    # It defines assessment_confidence as a dict, not DebuggingTaskOutput directly.
    # The agent then maps this to its DebuggingTaskOutput.confidence_score (ConfidenceScore object).
    "assessment_confidence": { # This is what the code_debugging_agent_v1_prompt.yaml output_schema expects
        "value": 0.8,
        "level": "High",
        "explanation": "Confident due to clear error."
    }
})

MOCK_DEBUGGER_LLM_NO_FIX_RESPONSE = json.dumps({
    "proposed_solution_type": "NO_FIX_IDENTIFIED",
    "proposed_code_changes": None,
    "explanation_of_fix": "Unable to determine the root cause with the provided context.",
    "assessment_confidence": {
        "value": 0.3,
        "level": "Low",
        "explanation": "Low confidence in 'no fix' due to limited context."
    },
    "areas_of_uncertainty": ["The actual execution flow is unclear."],
    "status_message": "No fix identified."
})

@pytest.fixture
def mock_llm_provider_debugger(): # Specific fixture name
    mock = AsyncMock(spec=LLMProvider)
    # Default to success, can be overridden in tests
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=MOCK_DEBUGGER_LLM_FIX_PROPOSED_RESPONSE)
    return mock

@pytest.fixture
def mock_prompt_manager_debugger():
    mock = AsyncMock(spec=PromptManager)
    return mock

@pytest.fixture
def mock_pcma_debugger(): # Debugger gets context via ARCA, but it now takes PCMA in init
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: Optional[str] = None):
        # This mock might be called if the debugger itself tries to fetch more context, though not its primary design.
        # For now, assume it doesn't make direct calls for its core task based on current implementation.
        mock_doc = AsyncMock()
        mock_doc.document_content = f"Debugger mock content for {doc_id}"
        mock_doc.id = doc_id
        return mock_doc
    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    return mock

@pytest.mark.asyncio
async def test_code_debugging_agent_proposes_fix_success(mock_llm_provider_debugger, mock_prompt_manager_debugger, mock_pcma_debugger):
    agent = CodeDebuggingAgent_v1(
        llm_provider=mock_llm_provider_debugger,
        prompt_manager=mock_prompt_manager_debugger,
        project_chroma_manager=mock_pcma_debugger # Pass pcma_debugger
    )

    task_input = DebuggingTaskInput(
        project_id="test_debugger_project",
        task_id="test_debugger_task_fix_123",
        faulty_code_path="src/buggy_module.py",
        faulty_code_snippet="def buggy_function():\n    return 1 / 0",
        failed_test_reports=[
            {
                "test_name": "test_buggy_function_zero_division",
                "error_message": "ZeroDivisionError: division by zero",
                "stack_trace": "Traceback... at src/buggy_module.py line 2 in buggy_function"
            }
        ],
        relevant_loprd_requirements_ids=["loprd_req_001"], # Content will be in prompt_data
        relevant_blueprint_section_ids=["bp_sec_abc"],   # Content will be in prompt_data
        # Mocked content for LOPRD/Blueprint will be directly in prompt_render_data by ARCA
        # The DebuggingAgent doesn't fetch them itself from these IDs, it expects the string content.
        # So, the `prompt_render_data` prepared before calling the LLM is key.
        loprd_requirements_content_str="LOPRD Requirement: Function must not crash.",
        blueprint_sections_content_str="Blueprint: buggy_function handles critical calculations."
    )

    output = await agent.invoke_async(task_input)

    assert output.status == "SUCCESS_FIX_PROPOSED"
    assert output.proposed_solution_type == "MODIFIED_SNIPPET"
    assert output.proposed_code_changes == "def fixed_function():\\n    return 42"
    assert "off-by-one error" in output.explanation_of_fix
    assert output.confidence_score is not None
    assert output.confidence_score.value == 0.8
    assert output.confidence_score.level == "High"
    assert "Impact on performance" in output.areas_of_uncertainty[0]

    # Verify LLM call
    mock_llm_provider_debugger.generate_text_async_with_prompt_manager.assert_called_once()
    call_args = mock_llm_provider_debugger.generate_text_async_with_prompt_manager.call_args
    assert call_args.kwargs['prompt_name'] == CodeDebuggingAgent_v1.PROMPT_NAME
    prompt_data = call_args.kwargs['prompt_data']
    assert prompt_data['faulty_code_path'] == "src/buggy_module.py"
    assert "buggy_function" in prompt_data['faulty_code_snippet']
    assert "ZeroDivisionError" in prompt_data['failed_test_reports_str']
    assert "LOPRD Requirement: Function must not crash." in prompt_data['loprd_requirements_str'] # Matches agent's prompt_render_data key
    assert "Blueprint: buggy_function handles critical calculations." in prompt_data['blueprint_sections_str'] # Matches agent's prompt_render_data key
    
    print(f"Code Debugging Agent fix proposed test completed. Output: {output.model_dump_json(indent=2)}")


@pytest.mark.asyncio
async def test_code_debugging_agent_no_fix_identified(mock_llm_provider_debugger, mock_prompt_manager_debugger, mock_pcma_debugger):
    agent = CodeDebuggingAgent_v1(
        llm_provider=mock_llm_provider_debugger,
        prompt_manager=mock_prompt_manager_debugger,
        project_chroma_manager=mock_pcma_debugger
    )
    mock_llm_provider_debugger.generate_text_async_with_prompt_manager.return_value = MOCK_DEBUGGER_LLM_NO_FIX_RESPONSE

    task_input = DebuggingTaskInput(
        project_id="test_debugger_project_nofix",
        task_id="test_debugger_task_nofix_456",
        faulty_code_path="src/complex_bug.py",
        faulty_code_snippet="def complex_bug():\n    # ... very complex logic ...",
        failed_test_reports=[{"test_name": "test_complex", "error_message": "Mystery Error", "stack_trace": "..."}],
        relevant_loprd_requirements_ids=[],
        relevant_blueprint_section_ids=[],
        loprd_requirements_content_str="N/A",
        blueprint_sections_content_str="N/A"
    )

    output = await agent.invoke_async(task_input)

    assert output.status == "FAILURE_NO_FIX_IDENTIFIED"
    assert output.proposed_solution_type == "NO_FIX_IDENTIFIED"
    assert output.proposed_code_changes is None
    assert "Unable to determine the root cause" in output.explanation_of_fix
    assert output.confidence_score.value == 0.3

    print(f"Code Debugging Agent no fix test completed. Output: {output.model_dump_json(indent=2)}")

# Add more tests:
# - LLM parsing failures
# - LLM call exceptions
# - Different `proposed_solution_type` from LLM (e.g., CODE_PATCH, NEEDS_MORE_CONTEXT)
# - Input validation (e.g., missing required fields in DebuggingTaskInput, though Pydantic handles this)
# - Test with `previous_debugging_attempts` provided in input. 