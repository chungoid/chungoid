import pytest
import json
import uuid
from unittest.mock import AsyncMock
from typing import List, Optional, Dict, Any # Added Dict, Any

# Agent and schema imports
from chungoid.agents.autonomous_engine.code_debugging_agent import (
    CodeDebuggingAgent_v1, 
    DebuggingTaskInput, 
    DebuggingTaskOutput,
    FailedTestReport  # ADDED import
)
from chungoid.schemas.common import ConfidenceScore

# Mock imports
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
# No specific collection constants needed for direct use by debugger agent, context comes via ARCA

# ADDED: Fix Pydantic forward references - must be done after all imports
ProjectChromaManagerAgent_v1.model_rebuild()  # Must be rebuilt first
CodeDebuggingAgent_v1.model_rebuild()  # Then rebuild agents that depend on it

# Mock LLM response for CodeDebuggingAgent
MOCK_DEBUGGER_LLM_FIX_PROPOSED_RESPONSE = json.dumps({
    "proposed_solution_type": "MODIFIED_SNIPPET",
    "proposed_code_changes": "def fixed_function():\\n    return 42",
    "explanation_of_fix": "The bug was due to an off-by-one error. The proposed snippet corrects this.",
    "confidence_score_obj": {  # FIXED: Use confidence_score_obj as expected by agent
        "value": 0.8,
        "level": "High",
        "explanation": "Confident due to clear error."
    },
    "areas_of_uncertainty": ["Impact on performance not fully assessed."],
    "suggestions_for_ARCA": "Consider adding unit tests for edge cases."
})

MOCK_DEBUGGER_LLM_NO_FIX_RESPONSE = json.dumps({
    "proposed_solution_type": "NO_FIX_IDENTIFIED",
    "proposed_code_changes": None,
    "explanation_of_fix": "Unable to determine the root cause with the provided context.",
    "confidence_score_obj": {  # FIXED: Use confidence_score_obj as expected by agent
        "value": 0.3,
        "level": "Low",
        "explanation": "Low confidence in 'no fix' due to limited context."
    },
    "areas_of_uncertainty": ["The actual execution flow is unclear."],
    "suggestions_for_ARCA": "Need more context about the error."
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
        prompt_manager=mock_prompt_manager_debugger
    )

    task_input = DebuggingTaskInput(
        project_id="test_debugger_project",
        task_id="test_debugger_task_fix_123",
        faulty_code_path="src/buggy_module.py",
        faulty_code_snippet="def buggy_function():\n    return 1 / 0",
        failed_test_reports=[
            FailedTestReport(
                test_name="test_buggy_function_zero_division",
                error_message="ZeroDivisionError: division by zero",
                stack_trace="Traceback... at src/buggy_module.py line 2 in buggy_function"
            )
        ],
        relevant_loprd_requirements_ids=["loprd_req_001"],
        relevant_blueprint_section_ids=["bp_sec_abc"]
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
    assert call_args.kwargs['prompt_name'] == CodeDebuggingAgent_v1.PROMPT_TEMPLATE_NAME
    prompt_data = call_args.kwargs['prompt_render_data']
    assert prompt_data['faulty_code_path'] == "src/buggy_module.py"
    assert "buggy_function" in prompt_data['faulty_code_snippet']
    assert "ZeroDivisionError" in prompt_data['failed_test_reports_str']
    assert "loprd_req_001" in prompt_data['relevant_loprd_requirements_ids_str']
    assert "bp_sec_abc" in prompt_data['relevant_blueprint_section_ids_str']
    
    print(f"Code Debugging Agent fix proposed test completed. Output: {output.model_dump_json(indent=2)}")


@pytest.mark.asyncio
async def test_code_debugging_agent_no_fix_identified(mock_llm_provider_debugger, mock_prompt_manager_debugger, mock_pcma_debugger):
    agent = CodeDebuggingAgent_v1(
        llm_provider=mock_llm_provider_debugger,
        prompt_manager=mock_prompt_manager_debugger
    )
    mock_llm_provider_debugger.generate_text_async_with_prompt_manager.return_value = MOCK_DEBUGGER_LLM_NO_FIX_RESPONSE

    task_input = DebuggingTaskInput(
        project_id="test_debugger_project_nofix",
        task_id="test_debugger_task_nofix_456",
        faulty_code_path="src/complex_bug.py",
        faulty_code_snippet="def complex_bug():\n    # ... very complex logic ...",
        failed_test_reports=[
            FailedTestReport(
                test_name="test_complex", 
                error_message="Mystery Error", 
                stack_trace="..."
            )
        ],
        relevant_loprd_requirements_ids=[],
        relevant_blueprint_section_ids=[]
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