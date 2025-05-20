import pytest
import json
import uuid
from unittest.mock import AsyncMock
from typing import List # Added List

# Agent and schema imports
from chungoid.agents.autonomous_engine.proactive_risk_assessor_agent import ProactiveRiskAssessorAgent_v1, ProactiveRiskAssessorInput, ProactiveRiskAssessorOutput
from chungoid.schemas.common import ConfidenceScore

# Mock imports
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import (
    ProjectChromaManagerAgent_v1,
    LOPRD_ARTIFACTS_COLLECTION,
    BLUEPRINT_ARTIFACTS_COLLECTION,
    EXECUTION_PLANS_COLLECTION,
    RISK_ASSESSMENT_REPORTS_COLLECTION,
    OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION
)

# Mock LLM response for ProactiveRiskAssessorAgent
MOCK_PRAA_LLM_SUCCESS_RESPONSE = json.dumps({
    "risk_assessment_report_md": "# Mock Risk Report\\n- Risk 1: Mock risk details.",
    "optimization_opportunities_report_md": "# Mock Optimization Report\\n- Opt 1: Mock optimization details.",
    "assessment_confidence": {
        "value": 0.9,
        "level": "High",
        "reasoning": "Confident in the mock risk and optimization assessment.",
        "method": "LLM Self-Assessment (PRAA)"
    }
})

@pytest.fixture
def mock_llm_provider_praa(): # Specific fixture name
    mock = AsyncMock(spec=LLMProvider)
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=MOCK_PRAA_LLM_SUCCESS_RESPONSE)
    return mock

@pytest.fixture
def mock_prompt_manager_praa():
    mock = AsyncMock(spec=PromptManager)
    return mock

@pytest.fixture
def mock_pcma_praa():
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: str):
        mock_doc = AsyncMock()
        content = f"Mock content for {doc_id} from {collection_name}"
        if doc_id == "loprd_to_assess":
            content = "## Mock LOPRD Content\\n- Goal: Assess this."
        elif doc_id == "blueprint_to_assess":
            content = "### Mock Blueprint Content\\n- Component A designed."
        elif doc_id == "plan_to_assess":
            content = "#### Mock Plan Content\\n- Step 1: Execute."
        elif doc_id == "context_loprd_for_blueprint":
             content = "Contextual LOPRD for blueprint assessment."
        mock_doc.document_content = content
        mock_doc.id = doc_id
        return mock_doc
    
    async def _mock_store_document_content(project_id: str, collection_name: str, document_content: any, metadata: dict, document_id:str = None, document_relative_path:str=None):
        return document_id if document_id else str(uuid.uuid4())

    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    mock.store_document_content = AsyncMock(side_effect=_mock_store_document_content)
    return mock

@pytest.mark.asyncio
async def test_proactive_risk_assessor_agent_assess_loprd_success(mock_llm_provider_praa, mock_prompt_manager_praa, mock_pcma_praa):
    agent = ProactiveRiskAssessorAgent_v1(
        llm_provider=mock_llm_provider_praa,
        prompt_manager=mock_prompt_manager_praa,
        project_chroma_manager=mock_pcma_praa
    )

    task_input = ProactiveRiskAssessorInput(
        project_id="test_praa_project_loprd",
        task_id="test_praa_task_loprd_123",
        artifact_id="loprd_to_assess",
        artifact_type="LOPRD"
    )

    output = await agent.invoke_async(task_input)

    assert output.status == "SUCCESS"
    assert output.risk_assessment_report_doc_id is not None
    assert output.optimization_suggestions_report_doc_id is not None
    assert output.confidence_score is not None
    assert output.confidence_score.value == 0.9
    assert output.llm_full_response == MOCK_PRAA_LLM_SUCCESS_RESPONSE

    # Verify PCMA get calls
    mock_pcma_praa.get_document_by_id.assert_any_call(
        project_id="test_praa_project_loprd", 
        doc_id="loprd_to_assess", 
        collection_name=LOPRD_ARTIFACTS_COLLECTION
    )
    
    # Verify LLM call
    mock_llm_provider_praa.generate_text_async_with_prompt_manager.assert_called_once()
    call_args = mock_llm_provider_praa.generate_text_async_with_prompt_manager.call_args
    assert call_args.kwargs['prompt_name'] == ProactiveRiskAssessorAgent_v1.PROMPT_NAME
    prompt_data = call_args.kwargs['prompt_data']
    assert prompt_data['analysis_focus'] == "LOPRD"
    assert "## Mock LOPRD Content" in prompt_data['loprd_json_content']
    assert "N/A - Blueprint not provided" in prompt_data['project_blueprint_md_content']

    # Verify PCMA store calls
    risk_store_call = next(c for c in mock_pcma_praa.store_document_content.call_args_list if c.kwargs['collection_name'] == RISK_ASSESSMENT_REPORTS_COLLECTION)
    opt_store_call = next(c for c in mock_pcma_praa.store_document_content.call_args_list if c.kwargs['collection_name'] == OPTIMIZATION_SUGGESTION_REPORTS_COLLECTION)
    
    assert risk_store_call is not None
    assert "# Mock Risk Report" in risk_store_call.kwargs['document_content']
    assert risk_store_call.kwargs['metadata']['assessed_artifact_id'] == "loprd_to_assess"

    assert opt_store_call is not None
    assert "# Mock Optimization Report" in opt_store_call.kwargs['document_content']
    assert opt_store_call.kwargs['metadata']['assessed_artifact_type'] == "LOPRD"
    
    print(f"PRAA LOPRD test completed. Output: {output.model_dump_json(indent=2)}")


@pytest.mark.asyncio
async def test_proactive_risk_assessor_agent_assess_blueprint_success(mock_llm_provider_praa, mock_prompt_manager_praa, mock_pcma_praa):
    agent = ProactiveRiskAssessorAgent_v1(
        llm_provider=mock_llm_provider_praa,
        prompt_manager=mock_prompt_manager_praa,
        project_chroma_manager=mock_pcma_praa
    )

    task_input = ProactiveRiskAssessorInput(
        project_id="test_praa_project_bp",
        task_id="test_praa_task_bp_456",
        artifact_id="blueprint_to_assess",
        artifact_type="Blueprint",
        loprd_document_id_for_blueprint_context="context_loprd_for_blueprint"
    )

    output = await agent.invoke_async(task_input)

    assert output.status == "SUCCESS"
    assert output.risk_assessment_report_doc_id is not None

    # Verify PCMA get calls for blueprint and its LOPRD context
    mock_pcma_praa.get_document_by_id.assert_any_call(
        project_id="test_praa_project_bp", 
        doc_id="blueprint_to_assess", 
        collection_name=BLUEPRINT_ARTIFACTS_COLLECTION
    )
    mock_pcma_praa.get_document_by_id.assert_any_call(
        project_id="test_praa_project_bp", 
        doc_id="context_loprd_for_blueprint", 
        collection_name=LOPRD_ARTIFACTS_COLLECTION
    )

    # Verify LLM call content
    call_args = mock_llm_provider_praa.generate_text_async_with_prompt_manager.call_args
    prompt_data = call_args.kwargs['prompt_data']
    assert prompt_data['analysis_focus'] == "Blueprint"
    assert "Contextual LOPRD for blueprint assessment." in prompt_data['loprd_json_content']
    assert "### Mock Blueprint Content" in prompt_data['project_blueprint_md_content']

    print(f"PRAA Blueprint test completed. Output: {output.model_dump_json(indent=2)}")

# Add more tests: assess MasterExecutionPlan, LLM failures, PCMA failures, etc. 