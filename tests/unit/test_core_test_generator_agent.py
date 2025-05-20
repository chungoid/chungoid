import pytest
import json
from unittest.mock import AsyncMock, patch

# Updated import for the agent
from chungoid.runtime.agents.core_test_generator_agent import CoreTestGeneratorAgent_v1
from chungoid.schemas.agent_test_generator import TestGeneratorAgentInput, TestGeneratorAgentOutput
# Imports for mocked dependencies
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
from chungoid.schemas.common import ConfidenceScore # Though not directly in output, good for context

# Default mock LLM response for successful test generation
MOCK_LLM_SUCCESS_RESPONSE = json.dumps({
    "generated_test_code_string": "def test_mock_success():\\n    assert True"
})

@pytest.fixture
def mock_llm_provider():
    mock = AsyncMock(spec=LLMProvider)
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=MOCK_LLM_SUCCESS_RESPONSE)
    return mock

@pytest.fixture
def mock_prompt_manager():
    mock = AsyncMock(spec=PromptManager)
    # We might need to mock specific prompt_manager methods if the agent uses them directly
    # For now, generate_text_async_with_prompt_manager is on llm_provider
    return mock

@pytest.fixture
def mock_pcma():
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    # Default mock for get_document_by_id
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: str):
        mock_doc = AsyncMock()
        if "loprd" in doc_id:
            mock_doc.document_content = "Mock LOPRD requirement: Must be awesome."
        elif "blueprint" in doc_id:
            mock_doc.document_content = "Mock Blueprint section: Component X architecture."
        else:
            mock_doc.document_content = "Mock generic document content."
        mock_doc.id = doc_id
        return mock_doc
    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    # Mock for store_document_content (conceptual, as agent doesn't implement it yet)
    mock.store_document_content = AsyncMock(return_value=str(uuid.uuid4()))
    return mock


@pytest.mark.asyncio
async def test_test_generator_agent_create_tests_success(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test successful test generation with mocked dependencies."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    code_str = "def my_function(x):\\n    return x * 2"
    file_path = "source_module.py"
    test_file_path = "test_source_module.py"
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_success",
        task_id="test_task_success",
        code_to_test=code_str,
        file_path_of_code=file_path,
        target_test_file_path=test_file_path,
        programming_language="python",
        test_framework_preference="pytest",
        relevant_loprd_requirements_ids=["loprd_req_001"],
        relevant_blueprint_section_ids=["bp_sec_abc"]
    )

    output = await agent.invoke_async(inputs)

    assert output.status == "SUCCESS"
    assert output.error_message is None
    assert output.target_test_file_path == test_file_path
    assert output.generated_test_code_string is not None
    assert "test_mock_success" in output.generated_test_code_string # From MOCK_LLM_SUCCESS_RESPONSE
    
    # Check that PCMA was called for LOPRD and Blueprint context
    assert any(call.kwargs.get('doc_id') == "loprd_req_001" for call in mock_pcma.get_document_by_id.call_args_list)
    assert any(call.kwargs.get('doc_id') == "bp_sec_abc" for call in mock_pcma.get_document_by_id.call_args_list)

    # Check that LLM provider was called
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()
    call_args = mock_llm_provider.generate_text_async_with_prompt_manager.call_args
    assert call_args.kwargs['prompt_manager'] == mock_prompt_manager
    assert call_args.kwargs['prompt_name'] == CoreTestGeneratorAgent_v1.PROMPT_NAME
    assert "Mock LOPRD requirement: Must be awesome." in call_args.kwargs['prompt_data']['loprd_requirements_content']
    assert "Mock Blueprint section: Component X architecture." in call_args.kwargs['prompt_data']['blueprint_sections_content']


@pytest.mark.asyncio
async def test_test_generator_agent_with_related_context(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test test generation when related_files_context is provided."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    code_str = "from .models import Item\\ndef process_item(item: Item):\\n    return item.name"
    file_path = "services.py"
    test_file_path = "test_services.py"
    related_ctx = {"models.py": "class Item:\\n    name: str"}

    inputs = TestGeneratorAgentInput(
        project_id="test_proj_related_ctx",
        task_id="test_task_related_ctx",
        code_to_test=code_str,
        file_path_of_code=file_path,
        target_test_file_path=test_file_path,
        related_files_context=related_ctx,
        programming_language="python",
        relevant_loprd_requirements_ids=[], # No LOPRD/BP for this specific test focus
        relevant_blueprint_section_ids=[]
    )
    
    # Specific mock for PCMA to return None for LOPRD/BP for this test
    async def _no_context_get_doc(project_id: str, doc_id: str, collection_name: str):
        return None 
    mock_pcma.get_document_by_id.side_effect = _no_context_get_doc


    output = await agent.invoke_async(inputs)
    
    assert output.status == "SUCCESS"
    assert output.generated_test_code_string is not None
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()
    
    prompt_data = mock_llm_provider.generate_text_async_with_prompt_manager.call_args.kwargs['prompt_data']
    assert code_str in prompt_data['code_to_test']
    assert file_path in prompt_data['file_path_of_code']
    assert "models.py" in prompt_data['related_files_context_str']
    assert related_ctx["models.py"] in prompt_data['related_files_context_str']
    assert "No LOPRD requirements provided or found." in prompt_data['loprd_requirements_content']
    assert "No Blueprint sections provided or found." in prompt_data['blueprint_sections_content']


@pytest.mark.asyncio
async def test_test_generator_agent_llm_returns_invalid_json(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when LLM response is not valid JSON."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = "This is not JSON"
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_invalid_json", task_id="test_task_invalid_json",
        code_to_test="def f(): pass", file_path_of_code="f.py", target_test_file_path="test_f.py",
        relevant_loprd_requirements_ids=[], relevant_blueprint_section_ids=[]
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_OUTPUT_PARSING"
    assert "LLM response not valid JSON" in output.error_message
    assert output.generated_test_code_string is None
    assert "This is not JSON" in output.llm_full_response

@pytest.mark.asyncio
async def test_test_generator_agent_llm_json_missing_key(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when LLM JSON response is missing the required key."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = json.dumps({"other_key": "some_value"})
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_missing_key", task_id="test_task_missing_key",
        code_to_test="def f(): pass", file_path_of_code="f.py", target_test_file_path="test_f.py",
        relevant_loprd_requirements_ids=[], relevant_blueprint_section_ids=[]
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_OUTPUT_PARSING" # Or a more specific error if we add one
    assert "LLM JSON output missing required key: generated_test_code_string" in output.error_message
    assert output.generated_test_code_string is None
    assert "other_key" in output.llm_full_response


@pytest.mark.asyncio
async def test_test_generator_agent_llm_call_exception(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when the LLM call itself raises an exception."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.side_effect = Exception("LLM service down")
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_llm_exception", task_id="test_task_llm_exception",
        code_to_test="def f(): pass", file_path_of_code="f.py", target_test_file_path="test_f.py",
        relevant_loprd_requirements_ids=[], relevant_blueprint_section_ids=[]
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM"
    assert "LLM interaction failed: LLM service down" in output.error_message
    assert output.generated_test_code_string is None

@pytest.mark.asyncio
async def test_test_generator_agent_pcma_loprd_retrieval_fails(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test behavior when PCMA fails to retrieve LOPRD context."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    
    async def mock_get_doc_failure(project_id: str, doc_id: str, collection_name: str):
        if "loprd" in doc_id:
            raise ValueError("Simulated PCMA LOPRD retrieval error")
        mock_doc = AsyncMock() # For blueprint
        mock_doc.document_content = "Mock Blueprint section: Component Y architecture."
        return mock_doc
        
    mock_pcma.get_document_by_id.side_effect = mock_get_doc_failure
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_pcma_fail", task_id="test_task_pcma_fail",
        code_to_test="def my_func(): pass", file_path_of_code="m.py", target_test_file_path="test_m.py",
        relevant_loprd_requirements_ids=["loprd_fail_id"],
        relevant_blueprint_section_ids=["bp_ok_id"]
    )

    await agent.invoke_async(inputs) # Call the agent

    # Check that the prompt_data sent to LLM reflects the LOPRD retrieval issue
    prompt_data = mock_llm_provider.generate_text_async_with_prompt_manager.call_args.kwargs['prompt_data']
    assert "Failed to retrieve LOPRD content for id loprd_fail_id: Simulated PCMA LOPRD retrieval error" in prompt_data['loprd_requirements_content']
    assert "Mock Blueprint section: Component Y architecture." in prompt_data['blueprint_sections_content']
    # The agent should still proceed and the LLM call should be made

@pytest.mark.asyncio
async def test_test_generator_agent_pcma_blueprint_retrieval_fails(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test behavior when PCMA fails to retrieve Blueprint context."""
    agent = CoreTestGeneratorAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    
    async def mock_get_doc_failure(project_id: str, doc_id: str, collection_name: str):
        if "blueprint" in doc_id:
            raise ValueError("Simulated PCMA Blueprint retrieval error")
        mock_doc = AsyncMock() # For LOPRD
        mock_doc.document_content = "Mock LOPRD requirement: Must be functional."
        return mock_doc
        
    mock_pcma.get_document_by_id.side_effect = mock_get_doc_failure
    
    inputs = TestGeneratorAgentInput(
        project_id="test_proj_pcma_bp_fail", task_id="test_task_pcma_bp_fail",
        code_to_test="def my_func_bp(): pass", file_path_of_code="m_bp.py", target_test_file_path="test_m_bp.py",
        relevant_loprd_requirements_ids=["loprd_ok_id"],
        relevant_blueprint_section_ids=["bp_fail_id"]
    )

    await agent.invoke_async(inputs) # Call the agent

    prompt_data = mock_llm_provider.generate_text_async_with_prompt_manager.call_args.kwargs['prompt_data']
    assert "Mock LOPRD requirement: Must be functional." in prompt_data['loprd_requirements_content']
    assert "Failed to retrieve Blueprint content for id bp_fail_id: Simulated PCMA Blueprint retrieval error" in prompt_data['blueprint_sections_content']

# Add a dummy import for uuid if not already present from other tests/fixtures for 'str(uuid.uuid4())' usage
import uuid 