import pytest
import json
import uuid
from unittest.mock import AsyncMock

# Agent and schema imports
from chungoid.agents.autonomous_engine.product_analyst_agent import ProductAnalystAgent_v1
from chungoid.schemas.agent_product_analyst import ProductAnalystAgentInput, ProductAnalystAgentOutput
from chungoid.schemas.loprd import LOPRD # For validating mocked LOPRD output
from chungoid.schemas.common import ConfidenceScore

# Mock imports
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, LOPRD_ARTIFACTS_COLLECTION

# Mock LLM response for ProductAnalystAgent
MOCK_PA_LLM_SUCCESS_RESPONSE = json.dumps({
    "loprd_artifact": {
        # Minimal valid LOPRD structure for testing
        "document_metadata": {
            "document_id": "loprd_mock_doc_id",
            "version": "1.0",
            "project_name": "Mock Project",
            "creation_date": "2023-01-01",
            "last_modified_date": "2023-01-01",
            "authors": ["ProductAnalystAgent_v1"],
            "status": "DRAFT"
        },
        "project_overview": {
            "project_goal": "Mock goal from LLM.",
            "target_audience": ["mock_users"],
            "scope": {"in_scope": ["mock_feature"], "out_of_scope": []}
        },
        "functional_requirements": [],
        "non_functional_requirements": [],
        "user_stories": [],
        "acceptance_criteria_global": []
    },
    "confidence_score": {
        "value": 0.85,
        "level": "High",
        "reasoning": "Confident in the mock LOPRD structure.",
        "method": "LLM Self-Assessment (ProductAnalyst)"
    }
})

@pytest.fixture
def mock_llm_provider_pa(): # Specific fixture name to avoid conflict if run with other tests
    mock = AsyncMock(spec=LLMProvider)
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=MOCK_PA_LLM_SUCCESS_RESPONSE)
    return mock

@pytest.fixture
def mock_prompt_manager_pa():
    mock = AsyncMock(spec=PromptManager)
    return mock

@pytest.fixture
def mock_pcma_pa():
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: str):
        mock_doc = AsyncMock()
        if doc_id == "refined_goal_doc_id_example":
            mock_doc.document_content = "This is the refined user goal content."
        elif doc_id == "assumptions_doc_id_example":
            mock_doc.document_content = "These are the assumptions."
        elif doc_id == "feedback_doc_id_example":
            mock_doc.document_content = "This is ARCA feedback."
        else:
            mock_doc.document_content = f"Default content for {doc_id}"
        mock_doc.id = doc_id
        return mock_doc
    
    async def _mock_store_document_content(project_id: str, collection_name: str, document_content: any, metadata: dict, document_id:str = None, document_relative_path:str=None):
        # Mock the storage and return the ID that would be used/generated
        return document_id if document_id else str(uuid.uuid4())

    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    mock.store_document_content = AsyncMock(side_effect=_mock_store_document_content)
    return mock

@pytest.mark.asyncio
async def test_product_analyst_agent_success(mock_llm_provider_pa, mock_prompt_manager_pa, mock_pcma_pa):
    agent = ProductAnalystAgent_v1(
        llm_provider=mock_llm_provider_pa,
        prompt_manager=mock_prompt_manager_pa,
        project_chroma_manager=mock_pcma_pa
    )

    task_input = ProductAnalystAgentInput(
        project_id="test_pa_project",
        task_id="test_pa_task_123",
        refined_user_goal_doc_id="refined_goal_doc_id_example",
        assumptions_and_ambiguities_doc_id="assumptions_doc_id_example",
        project_context_doc_ids=["feedback_doc_id_example"] # Example, can be empty
    )

    output = await agent.invoke_async(task_input)

    assert output.status == "SUCCESS"
    assert output.loprd_document_id is not None
    assert output.agent_confidence_score is not None
    assert output.agent_confidence_score.value == 0.85
    assert output.agent_confidence_score.level == "High"
    
    # Verify PCMA calls
    mock_pcma_pa.get_document_by_id.assert_any_call(project_id="test_pa_project", doc_id="refined_goal_doc_id_example")
    mock_pcma_pa.get_document_by_id.assert_any_call(project_id="test_pa_project", doc_id="assumptions_doc_id_example")
    mock_pcma_pa.get_document_by_id.assert_any_call(project_id="test_pa_project", doc_id="feedback_doc_id_example")
    
    # Verify LLM call
    mock_llm_provider_pa.generate_text_async_with_prompt_manager.assert_called_once()
    call_args = mock_llm_provider_pa.generate_text_async_with_prompt_manager.call_args
    assert call_args.kwargs['prompt_name'] == ProductAnalystAgent_v1.PROMPT_NAME
    assert "This is the refined user goal content." in call_args.kwargs['prompt_data']['refined_user_goal_content']

    # Verify PCMA store call
    # The agent stores the LOPRD. We check if store_document_content was called for the LOPRD collection.
    # This requires knowing the output.loprd_document_id, which is fine.
    
    # Find the call to store_document_content that matches LOPRD_ARTIFACTS_COLLECTION
    loprd_store_call = None
    for call in mock_pcma_pa.store_document_content.call_args_list:
        if call.kwargs.get('collection_name') == LOPRD_ARTIFACTS_COLLECTION:
            loprd_store_call = call
            break
    
    assert loprd_store_call is not None, "PCMA store_document_content not called for LOPRD_ARTIFACTS_COLLECTION"
    
    # Check some metadata from the store call
    stored_loprd_content_str = loprd_store_call.kwargs['document_content']
    stored_metadata = loprd_store_call.kwargs['metadata']
    
    # Validate the stored LOPRD structure (rudimentary check)
    stored_loprd_dict = json.loads(stored_loprd_content_str)
    assert stored_loprd_dict['document_metadata']['document_id'] == "loprd_mock_doc_id" # from mock LLM response
    assert stored_loprd_dict['project_overview']['project_goal'] == "Mock goal from LLM."

    assert stored_metadata['artifact_type'] == "LOPRD_JSON"
    assert stored_metadata['project_id'] == "test_pa_project"
    assert stored_metadata['confidence_value'] == 0.85 # Check confidence stored in metadata
    assert stored_metadata['confidence_level'] == "High"

    print(f"Product Analyst Agent test completed. Output: {output.model_dump_json(indent=2)}")

# Add more tests here for failure cases, different inputs, etc.
# e.g., test_product_analyst_agent_llm_failure
# e.g., test_product_analyst_agent_pcma_retrieval_failure
# e.g., test_product_analyst_agent_invalid_loprd_from_llm 