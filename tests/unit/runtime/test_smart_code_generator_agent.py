import pytest
import json
import uuid # Added for task_id
from unittest.mock import patch, AsyncMock
from typing import Optional, Any, List # Added List

from chungoid.runtime.agents.core_code_generator_agent import CodeGeneratorAgent
from chungoid.schemas.agent_code_generator import CodeGeneratorAgentInput, CodeGeneratorAgentOutput

# Imports for mocked dependencies
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1

# Default mock LLM response for successful code generation
MOCK_LLM_CODE_SUCCESS_RESPONSE = json.dumps({
    "generated_code_string": "# Mocked generated code\\ndef hello_mock():\\n    print('Hello from mocked generated code!')"
})

# Reusable fixtures from the other test file (assuming they are in a conftest.py or defined here)
@pytest.fixture
def mock_llm_provider():
    mock = AsyncMock(spec=LLMProvider)
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=MOCK_LLM_CODE_SUCCESS_RESPONSE)
    return mock

@pytest.fixture
def mock_prompt_manager():
    mock = AsyncMock(spec=PromptManager)
    return mock

@pytest.fixture
def mock_pcma():
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    # Default mock for get_document_by_id
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: Optional[str] = None):
        mock_doc = AsyncMock()
        if doc_id == "spec_new_hello":
            mock_doc.document_content = "Create a Python function `hello()` that prints 'Hello, World!'."
        elif doc_id == "spec_modify_bar":
            mock_doc.document_content = "Add a docstring to the function Foo.bar_method: 'This is bar_method.'"
        elif doc_id == "existing_foo_code":
            mock_doc.document_content = "class Foo:\\n    def bar_method(self):\\n        pass"
        elif "loprd" in doc_id:
            mock_doc.document_content = "Mock LOPRD requirement content for code gen."
        elif "blueprint" in doc_id:
            mock_doc.document_content = "Mock Blueprint section content for code gen."
        else:
            # Default for unspecified doc_ids to avoid errors if they are optional and not explicitly handled
            mock_doc.document_content = f"Mock content for doc_id: {doc_id}"
            # return None # Or raise an error if all doc_ids should be explicitly mocked
        mock_doc.id = doc_id
        return mock_doc
    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    mock.store_document_content = AsyncMock(return_value=str(uuid.uuid4())) # For conceptual storage
    return mock

@pytest.mark.asyncio
async def test_code_generator_agent_create_code_success(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test successful code generation for a new file with mocked dependencies."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    file_path = "new_hello.py"
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_create",
        task_id=f"task_create_{str(uuid.uuid4())}",
        code_specification_doc_id="spec_new_hello",
        target_file_path=file_path,
        programming_language="python",
        # Assuming no existing code, LOPRD, or Blueprint for this simple creation case by default
        # loprd_requirement_ids=None, 
        # blueprint_section_ids=None,
    )

    output = await agent.invoke_async(inputs)

    assert output.status == "SUCCESS"
    assert output.error_message is None
    assert output.target_file_path == file_path
    assert output.generated_code_string is not None
    assert "hello_mock" in output.generated_code_string # From MOCK_LLM_CODE_SUCCESS_RESPONSE
    
    mock_pcma.get_document_by_id.assert_any_call(project_id="test_proj_create", doc_id="spec_new_hello")
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()
    call_args = mock_llm_provider.generate_text_async_with_prompt_manager.call_args
    prompt_data = call_args.kwargs['prompt_data']
    assert "Create a Python function `hello()` that prints 'Hello, World!'." in prompt_data['code_specification_content']
    assert "No existing code provided." in prompt_data['existing_code_content']

@pytest.mark.asyncio
async def test_code_generator_agent_modify_code_success(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test successful code modification with context and mocked dependencies."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    file_path = "existing_module.py"
    related_ctx = {"helper.py": "def some_helper(): return True"}
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_modify",
        task_id=f"task_modify_{str(uuid.uuid4())}",
        code_specification_doc_id="spec_modify_bar",
        target_file_path=file_path,
        existing_code_doc_id="existing_foo_code",
        related_files_context=related_ctx,
        programming_language="python",
        loprd_requirement_ids=["loprd_for_modify"],
        blueprint_section_ids=["bp_for_modify"]
    )

    output = await agent.invoke_async(inputs)
    
    assert output.status == "SUCCESS"
    assert output.generated_code_string is not None
    assert "hello_mock" in output.generated_code_string # From MOCK_LLM_CODE_SUCCESS_RESPONSE

    mock_pcma.get_document_by_id.assert_any_call(project_id="test_proj_modify", doc_id="spec_modify_bar")
    mock_pcma.get_document_by_id.assert_any_call(project_id="test_proj_modify", doc_id="existing_foo_code")
    mock_pcma.get_document_by_id.assert_any_call(project_id="test_proj_modify", doc_id="loprd_for_modify")
    mock_pcma.get_document_by_id.assert_any_call(project_id="test_proj_modify", doc_id="bp_for_modify")

    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()
    prompt_data = mock_llm_provider.generate_text_async_with_prompt_manager.call_args.kwargs['prompt_data']
    assert "Add a docstring to the function Foo.bar_method: 'This is bar_method.'" in prompt_data['code_specification_content']
    assert "class Foo:" in prompt_data['existing_code_content'] # From existing_foo_code mock
    assert "Mock LOPRD requirement content for code gen." in prompt_data['loprd_context_string']
    assert "Mock Blueprint section content for code gen." in prompt_data['blueprint_context_string']
    assert "helper.py" in prompt_data['related_files_context_str']
    assert related_ctx["helper.py"] in prompt_data['related_files_context_str']

@pytest.mark.asyncio
async def test_code_generator_agent_llm_returns_invalid_json(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when LLM response is not valid JSON."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = "This is not JSON"
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_invalid", task_id="task_invalid",
        code_specification_doc_id="spec_invalid", target_file_path="test.py"
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_OUTPUT_PARSING"
    assert "LLM response not valid JSON" in output.error_message
    assert output.generated_code_string is None
    assert "This is not JSON" in output.llm_full_response

@pytest.mark.asyncio
async def test_code_generator_agent_llm_json_missing_key(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when LLM JSON response is missing the required key."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = json.dumps({"other_key": "value"})
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_m_key", task_id="task_m_key",
        code_specification_doc_id="spec_m_key", target_file_path="test.py"
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_OUTPUT_PARSING"
    assert "LLM JSON output missing required key: generated_code_string" in output.error_message
    assert output.generated_code_string is None
    assert "other_key" in output.llm_full_response

@pytest.mark.asyncio
async def test_code_generator_agent_llm_call_exception(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test failure when the LLM call itself raises an exception."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    mock_llm_provider.generate_text_async_with_prompt_manager.side_effect = Exception("LLM network error")
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_llm_ex", task_id="task_llm_ex",
        code_specification_doc_id="spec_llm_ex", target_file_path="test.py"
    )
        
    output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM"
    assert "LLM interaction failed: LLM network error" in output.error_message
    assert output.generated_code_string is None

@pytest.mark.asyncio
async def test_code_generator_agent_pcma_retrieval_fails(mock_llm_provider, mock_prompt_manager, mock_pcma):
    """Test that agent proceeds if some PCMA context retrieval fails but logs error in prompt data."""
    agent = CodeGeneratorAgent(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        project_chroma_manager=mock_pcma
    )
    
    async def mock_get_doc_partial_failure(project_id: str, doc_id: str, collection_name: Optional[str] = None):
        if doc_id == "spec_pcma_fail":
            return AsyncMock(document_content="Working spec for PCMA fail test", id=doc_id)
        elif doc_id == "loprd_pcma_fail":
            raise ValueError("Simulated PCMA LOPRD retrieval error for code gen")
        # Allow blueprint and existing code to be found or return default mock for them
        mock_doc = AsyncMock()
        mock_doc.id = doc_id
        if doc_id == "bp_pcma_ok":
            mock_doc.document_content = "Blueprint context is OK."
        elif doc_id == "existing_code_ok":
            mock_doc.document_content = "Existing code is OK."
        else:
            mock_doc.document_content = "Default mock content."
        return mock_doc
        
    mock_pcma.get_document_by_id.side_effect = mock_get_doc_partial_failure
    
    inputs = CodeGeneratorAgentInput(
        project_id="test_proj_pcma_some_fail", 
        task_id="task_pcma_some_fail",
        code_specification_doc_id="spec_pcma_fail", 
        target_file_path="test_some_fail.py",
        existing_code_doc_id="existing_code_ok",
        loprd_requirement_ids=["loprd_pcma_fail"],
        blueprint_section_ids=["bp_pcma_ok"]
    )

    output = await agent.invoke_async(inputs)
    # We expect the agent to still try to generate code, even if some context is missing
    assert output.status == "SUCCESS" 
    assert "hello_mock" in output.generated_code_string # LLM call should still proceed

    prompt_data = mock_llm_provider.generate_text_async_with_prompt_manager.call_args.kwargs['prompt_data']
    assert "Working spec for PCMA fail test" in prompt_data['code_specification_content']
    assert "Failed to retrieve LOPRD context for IDs ['loprd_pcma_fail']: Simulated PCMA LOPRD retrieval error" in prompt_data['loprd_context_string']
    assert "Blueprint context is OK." in prompt_data['blueprint_context_string']
    assert "Existing code is OK." in prompt_data['existing_code_content'] 