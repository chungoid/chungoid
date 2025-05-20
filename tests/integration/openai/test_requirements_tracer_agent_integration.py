import pytest
import os
import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Optional

from chungoid.agents.autonomous_engine.requirements_tracer_agent import (
    RequirementsTracerAgent_v1,
    RequirementsTracerInput,
    RequirementsTracerOutput,
)
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    RetrieveArtifactOutput,
    TRACEABILITY_REPORTS_COLLECTION,
    StoreArtifactOutput,
)
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.llm_provider import LLMProvider

# Get the OpenAI API Key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REASON_TO_SKIP = "OPENAI_API_KEY not found in environment variables. Skipping integration test."

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for each test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def prompt_manager_instance():
    # Adjust the path as necessary to locate your actual prompts directory
    # This assumes the tests are run from a context where this relative path is valid.
    # For robust CI, consider absolute paths or a more sophisticated path resolution.
    return PromptManager(prompt_directory_paths=["chungoid-core/server_prompts"])

@pytest.fixture(scope="module")
async def llm_provider_instance(prompt_manager_instance):
    if not OPENAI_API_KEY:
        pytest.skip(REASON_TO_SKIP)
    
    provider = LLMProvider(prompt_manager=prompt_manager_instance)
    yield provider
    # Clean up the client after all tests in the module are done
    await provider.close_client()

@pytest.fixture
def mock_pcma_integration():
    mock_pcma_instance = AsyncMock(spec=ProjectChromaManagerAgent_v1)

    # This function will be the side_effect for pcma.retrieve_artifact
    async def _mock_retrieve_artifact_for_test(base_collection_name: str, document_id: str):
        content = "Default mock content from _mock_retrieve_artifact_for_test"
        if document_id == "source_doc_integration_test":
            content = "This is the source LOPRD requirement for integration testing: The system must be awesome."
        elif document_id == "target_doc_integration_test":
            content = "This is the target Blueprint section for integration testing: Component X enables awesomeness."
        
        return RetrieveArtifactOutput(
            document_id=document_id,
            content=content,
            metadata={"retrieved_by": "mock_pcma_integration_test", "base_collection_name_used": base_collection_name},
            status="SUCCESS"
        )
    
    # Set the side_effect for the correct method name
    mock_pcma_instance.retrieve_artifact.side_effect = _mock_retrieve_artifact_for_test

    # Mock store_document_content (actual method is store_artifact)
    mock_store_output = StoreArtifactOutput(
        document_id="mock_trace_report_id_12345", 
        status="SUCCESS",
        message="Traceability report stored successfully by mock."
    )
    # Correct the attribute being mocked to 'store_artifact'
    mock_pcma_instance.store_artifact.return_value = mock_store_output
    
    return mock_pcma_instance

@pytest.mark.skipif(not OPENAI_API_KEY, reason=REASON_TO_SKIP)
@pytest.mark.integration_openai  # Custom marker for OpenAI integration tests
@pytest.mark.asyncio
async def test_requirements_tracer_agent_live_llm_call(
    llm_provider_instance: LLMProvider,  # Use the real LLMProvider
    prompt_manager_instance: PromptManager, # Use the real PromptManager
    mock_pcma_integration: ProjectChromaManagerAgent_v1,
):
    """
    Tests the RequirementsTracerAgent_v1 with a live call to the OpenAI API.
    Mocks PCMA for document retrieval and storage.
    """
    agent = RequirementsTracerAgent_v1(
        llm_provider=llm_provider_instance,
        prompt_manager=prompt_manager_instance,
        project_chroma_manager=mock_pcma_integration,
    )

    project_id = "test_integration_project_live_llm"
    source_artifact_doc_id = "source_doc_integration_test"
    target_artifact_doc_id = "target_doc_integration_test"

    input_data = RequirementsTracerInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        source_artifact_type="LOPRD",
        target_artifact_doc_id=target_artifact_doc_id,
        target_artifact_type="Blueprint"
    )

    print(f"\nAttempting live LLM call for RequirementsTracerAgent_v1 with prompt: {RequirementsTracerAgent_v1.PROMPT_TEMPLATE_NAME} v{RequirementsTracerAgent_v1.VERSION}")

    result: RequirementsTracerOutput = None
    try:
        result = await agent.invoke_async(input_data)
    except Exception as e:
        pytest.fail(f"Agent invocation with live LLM failed: {e}")

    print(f"LLM call completed. Agent output: {result.model_dump_json(indent=2) if result else 'No result'}")

    assert result is not None, "Agent did not return a result."
    assert result.status == "SUCCESS", f"Agent returned status '{result.status}' with message: {result.message}"
    assert result.project_id == project_id
    assert result.traceability_report_doc_id == "mock_trace_report_id_12345" # From mocked PCMA store
    assert result.message == f"Traceability report generated. Stored as doc_id: {result.traceability_report_doc_id}"

    # Verify PCMA retrieve_artifact calls
    # The agent determines the collection name internally based on artifact type.
    # We need to know what those resolved collection names are to assert correctly.
    # From RequirementsTracerAgent_v1:
    # LOPRD -> loprd_artifacts_collection
    # Blueprint -> blueprint_artifacts_collection
    mock_pcma_integration.retrieve_artifact.assert_any_call(
        base_collection_name="loprd_artifacts_collection", 
        document_id=source_artifact_doc_id
    )
    mock_pcma_integration.retrieve_artifact.assert_any_call(
        base_collection_name="blueprint_artifacts_collection", 
        document_id=target_artifact_doc_id
    )
    assert mock_pcma_integration.retrieve_artifact.call_count == 2

    # Verify PCMA store_artifact call
    mock_pcma_integration.store_artifact.assert_called_once()
    store_call = mock_pcma_integration.store_artifact.call_args
    assert store_call is not None, "store_artifact was not called with any arguments"
    
    # The store_artifact method is called with a single keyword argument 'args'
    # which is an instance of StoreArtifactInput.
    stored_artifact_input_args = store_call.kwargs.get('args')
    assert stored_artifact_input_args is not None, "store_artifact was not called with 'args' keyword argument"

    assert stored_artifact_input_args.base_collection_name == TRACEABILITY_REPORTS_COLLECTION
    assert isinstance(stored_artifact_input_args.artifact_content, str)
    assert len(stored_artifact_input_args.artifact_content) > 0
    
    # Check metadata fields within the StoreArtifactInput object
    assert stored_artifact_input_args.metadata["project_id"] == project_id
    assert "confidence_value" in stored_artifact_input_args.metadata
    assert isinstance(stored_artifact_input_args.metadata["confidence_value"], float)
    assert stored_artifact_input_args.metadata["source_artifact_id"] == source_artifact_doc_id
    assert stored_artifact_input_args.metadata["target_artifact_id"] == target_artifact_doc_id

    print("Live LLM call test for RequirementsTracerAgent_v1 passed.")

# To run this test:
# 1. Ensure OPENAI_API_KEY is set in your environment variables or .env file.
# 2. Run pytest, potentially targeting this specific test or marker:
#    pytest -m integration_openai
#    pytest chungoid-core/tests/integration/openai/test_requirements_tracer_agent_integration.py
