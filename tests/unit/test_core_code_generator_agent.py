import pytest
from unittest.mock import patch, AsyncMock

from chungoid.runtime.agents.core_code_generator_agent import CodeGeneratorAgent, MockCodeLLMClient
from chungoid.schemas.agent_code_generator import CodeGeneratorAgentInput, CodeGeneratorAgentOutput

@pytest.mark.asyncio
async def test_code_generator_agent_create_code_success():
    """Test successful code generation for a new file."""
    agent = CodeGeneratorAgent()
    task_desc = "Create a Python function `hello()` that prints 'Hello, World!'."
    file_path = "new_hello.py"
    inputs = CodeGeneratorAgentInput(
        task_description=task_desc,
        target_file_path=file_path,
        programming_language="python"
    )

    # MockLLMClient's default behavior will be used.
    output = await agent.invoke_async(inputs)

    assert output.status == "SUCCESS"
    assert output.error_message is None
    assert output.target_file_path == file_path
    assert output.generated_code_string is not None
    assert "print('Hello from generated code!')" in output.generated_code_string # From mock
    assert task_desc[:50] in output.generated_code_string # Check if task was in prompt part of mock code
    assert output.llm_confidence is not None
    assert output.usage_metadata is not None

@pytest.mark.asyncio
async def test_code_generator_agent_modify_code_success():
    """Test successful code generation when modifying existing code with context."""
    agent = CodeGeneratorAgent()
    task_desc = "Add a docstring to the function Foo.bar_method: 'This is bar_method.'" 
    file_path = "existing_module.py"
    existing_code = "class Foo:\n    def bar_method(self):\n        pass"
    related_context = {"helper.py": "def some_helper(): return True"}
    
    inputs = CodeGeneratorAgentInput(
        task_description=task_desc,
        target_file_path=file_path,
        code_to_modify=existing_code,
        related_files_context=related_context,
        programming_language="python"
    )

    # Spy on the LLM call to check prompt components
    captured_prompts = {}
    async def generate_code_spy(system_prompt: str, user_prompt: str, code_context: Optional[str] = None):
        captured_prompts['system'] = system_prompt
        captured_prompts['user'] = user_prompt
        captured_prompts['code_context'] = code_context
        # Return default mock LLM output
        mock_llm = MockCodeLLMClient()
        return await mock_llm.generate_code(system_prompt, user_prompt, code_context)

    with patch.object(MockCodeLLMClient, 'generate_code', side_effect=generate_code_spy) as mock_llm_call:
        output = await agent.invoke_async(inputs)
    
    assert output.status == "SUCCESS"
    assert output.generated_code_string is not None
    mock_llm_call.assert_called_once()

    # Check that inputs were part of the prompt passed to LLM
    assert task_desc in captured_prompts['user']
    assert existing_code in captured_prompts['user']
    assert "helper.py" in captured_prompts['user']
    assert related_context["helper.py"] in captured_prompts['user']
    assert existing_code == captured_prompts['code_context'] # Check code_context specifically

@pytest.mark.asyncio
async def test_code_generator_agent_llm_returns_invalid_code_response():
    """Test failure when LLM response does not contain a valid code string."""
    agent = CodeGeneratorAgent()
    inputs = CodeGeneratorAgentInput(task_description="test", target_file_path="test.py")

    with patch.object(MockCodeLLMClient, 'generate_code', new_callable=AsyncMock) as mock_llm_call:
        # Simulate LLM returning a dict without 'generated_code' or with a non-string type
        mock_llm_call.return_value = {"confidence": 0.5, "raw_response": "Something else"} 
        
        output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_GENERATION"
    assert output.error_message == "LLM did not return a valid code string in its response."
    assert output.generated_code_string is None
    assert str(mock_llm_call.return_value) in output.llm_full_response

@pytest.mark.asyncio
async def test_code_generator_agent_llm_call_exception():
    """Test failure when the LLM call itself raises an exception."""
    agent = CodeGeneratorAgent()
    inputs = CodeGeneratorAgentInput(task_description="test", target_file_path="test.py")

    with patch.object(MockCodeLLMClient, 'generate_code', new_callable=AsyncMock) as mock_llm_call:
        mock_llm_call.side_effect = Exception("LLM network error")
        
        output = await agent.invoke_async(inputs)

    assert output.status == "FAILURE_LLM_GENERATION"
    assert "LLM interaction failed: LLM network error" in output.error_message
    assert output.generated_code_string is None 