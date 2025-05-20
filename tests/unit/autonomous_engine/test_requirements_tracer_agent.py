import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

from chungoid.agents.autonomous_engine.requirements_tracer_agent import RequirementsTracerAgent_v1, RequirementsTracerAgentInput, RequirementsTracerAgentOutput
from chungoid.schemas.project_chronicle import DocumentContent
from chungoid.utils.prompt_manager import PromptManager
from chungoid.llm_provider.llm_provider import LLMProvider
from chungoid.integrations.project_chroma_manager.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, PCMA_COLLECTION_NAMES

@pytest.fixture
def mock_llm_provider():
    return AsyncMock(spec=LLMProvider)

@pytest.fixture
def mock_prompt_manager():
    pm = MagicMock(spec=PromptManager)
    pm.get_prompt_template.return_value = "Test prompt template for {source_artifact_content} and {target_artifact_content}"
    return pm

@pytest.fixture
def mock_pcma_agent():
    pcma = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    pcma.TRACEABILITY_REPORTS_COLLECTION = PCMA_COLLECTION_NAMES.TRACEABILITY_REPORTS.value
    return pcma

@pytest.fixture
def requirements_tracer_agent(mock_llm_provider, mock_prompt_manager, mock_pcma_agent):
    agent = RequirementsTracerAgent_v1(
        llm_provider=mock_llm_provider,
        prompt_manager=mock_prompt_manager,
        pcma_agent=mock_pcma_agent
    )
    return agent

@pytest.mark.asyncio
async def test_invoke_async_success(requirements_tracer_agent, mock_llm_provider, mock_prompt_manager, mock_pcma_agent):
    """
    Tests the successful invocation of RequirementsTracerAgent_v1.
    """
    project_id = "test_project_id"
    source_artifact_doc_id = "source_doc_123"
    target_artifact_doc_id = "target_doc_456"
    
    input_data = RequirementsTracerAgentInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        target_artifact_doc_id=target_artifact_doc_id
    )

    mock_source_content = DocumentContent(doc_id=source_artifact_doc_id, content="This is the source content.", metadata={"type": "requirement"})
    mock_target_content = DocumentContent(doc_id=target_artifact_doc_id, content="This is the target code content.", metadata={"type": "code_module"})

    mock_pcma_agent.get_document_by_id.side_effect = [
        mock_source_content,
        mock_target_content
    ]

    llm_response_data = {
        "traceability_report": "Detailed traceability report linking source to target.",
        "confidence_score": 0.95
    }
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = llm_response_data

    mock_stored_doc_id = "trace_report_789"
    mock_pcma_agent.store_document_content.return_value = mock_stored_doc_id

    # Act
    result = await requirements_tracer_agent.invoke_async(input_data)

    # Assert
    assert isinstance(result, RequirementsTracerAgentOutput)
    assert result.project_id == project_id
    assert result.traceability_report_doc_id == mock_stored_doc_id
    assert result.status == "success"
    assert result.message == "Requirements traceability report generated and stored successfully."

    # Verify PCMA calls
    mock_pcma_agent.get_document_by_id.assert_any_call(
        project_id=project_id,
        doc_id=source_artifact_doc_id,
        collection_name=None # Agent determines collection or doesn't need it for generic get
    )
    mock_pcma_agent.get_document_by_id.assert_any_call(
        project_id=project_id,
        doc_id=target_artifact_doc_id,
        collection_name=None # Agent determines collection or doesn't need it for generic get
    )
    assert mock_pcma_agent.get_document_by_id.call_count == 2
    
    # Verify LLM call
    expected_prompt_render_data = {
        "source_artifact_content": mock_source_content.content,
        "target_artifact_content": mock_target_content.content
    }
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once_with(
        prompt_name=RequirementsTracerAgent_v1.AGENT_NAME,
        prompt_version="v1", # Assuming default or agent-defined version
        prompt_render_data=expected_prompt_render_data,
        project_id=project_id,
        calling_agent_id=RequirementsTracerAgent_v1.AGENT_ID,
        expected_json_schema=RequirementsTracerAgent_v1.DEFAULT_OUTPUT_SCHEMA,
        prompt_sub_path="autonomous_engine"
    )

    # Verify PCMA store call
    expected_report_content = llm_response_data["traceability_report"]
    expected_metadata = {
        "source_artifact_doc_id": source_artifact_doc_id,
        "target_artifact_doc_id": target_artifact_doc_id,
        "confidence_score": llm_response_data["confidence_score"],
        "agent_id": RequirementsTracerAgent_v1.AGENT_ID,
    }
    mock_pcma_agent.store_document_content.assert_called_once_with(
        project_id=project_id,
        content=expected_report_content,
        collection_name=mock_pcma_agent.TRACEABILITY_REPORTS_COLLECTION,
        metadata=expected_metadata,
        doc_id_prefix="trace_report"
    )

@pytest.mark.asyncio
async def test_invoke_async_pcma_get_document_fails(requirements_tracer_agent, mock_pcma_agent):
    """
    Tests behavior when PCMA fails to retrieve a document.
    """
    project_id = "test_project_id_fail"
    source_artifact_doc_id = "source_doc_fail"
    target_artifact_doc_id = "target_doc_fail"
    
    input_data = RequirementsTracerAgentInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        target_artifact_doc_id=target_artifact_doc_id
    )

    mock_pcma_agent.get_document_by_id.side_effect = ValueError("Failed to retrieve document")

    # Act
    result = await requirements_tracer_agent.invoke_async(input_data)

    # Assert
    assert isinstance(result, RequirementsTracerAgentOutput)
    assert result.project_id == project_id
    assert result.traceability_report_doc_id is None
    assert result.status == "error"
    assert "Failed to retrieve document content" in result.message
    assert "ValueError: Failed to retrieve document" in result.message

    mock_pcma_agent.get_document_by_id.assert_called_once() # Should fail on the first call

@pytest.mark.asyncio
async def test_invoke_async_llm_call_fails(requirements_tracer_agent, mock_llm_provider, mock_pcma_agent):
    """
    Tests behavior when the LLM call fails.
    """
    project_id = "test_project_id_llm_fail"
    source_artifact_doc_id = "source_doc_llm_fail"
    target_artifact_doc_id = "target_doc_llm_fail"
    
    input_data = RequirementsTracerAgentInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        target_artifact_doc_id=target_artifact_doc_id
    )

    mock_source_content = DocumentContent(doc_id=source_artifact_doc_id, content="Source", metadata={})
    mock_target_content = DocumentContent(doc_id=target_artifact_doc_id, content="Target", metadata={})
    mock_pcma_agent.get_document_by_id.side_effect = [mock_source_content, mock_target_content]
    
    mock_llm_provider.generate_text_async_with_prompt_manager.side_effect = Exception("LLM API error")

    # Act
    result = await requirements_tracer_agent.invoke_async(input_data)

    # Assert
    assert isinstance(result, RequirementsTracerAgentOutput)
    assert result.project_id == project_id
    assert result.traceability_report_doc_id is None
    assert result.status == "error"
    assert "Error during LLM call" in result.message
    assert "Exception: LLM API error" in result.message
    
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()

@pytest.mark.asyncio
async def test_invoke_async_llm_returns_invalid_json(requirements_tracer_agent, mock_llm_provider, mock_pcma_agent):
    """
    Tests behavior when the LLM returns a malformed JSON or not the expected structure.
    """
    project_id = "test_project_id_json_fail"
    source_artifact_doc_id = "source_doc_json_fail"
    target_artifact_doc_id = "target_doc_json_fail"
    
    input_data = RequirementsTracerAgentInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        target_artifact_doc_id=target_artifact_doc_id
    )

    mock_source_content = DocumentContent(doc_id=source_artifact_doc_id, content="Source", metadata={})
    mock_target_content = DocumentContent(doc_id=target_artifact_doc_id, content="Target", metadata={})
    mock_pcma_agent.get_document_by_id.side_effect = [mock_source_content, mock_target_content]
    
    # Simulate LLM returning data that doesn't match Pydantic model (e.g., missing 'confidence_score')
    llm_malformed_response_data = { 
        "traceability_report": "Report without confidence."
        # "confidence_score": 0.8 # Missing
    }
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = llm_malformed_response_data

    # Act
    result = await requirements_tracer_agent.invoke_async(input_data)

    # Assert
    assert isinstance(result, RequirementsTracerAgentOutput)
    assert result.project_id == project_id
    assert result.traceability_report_doc_id is None
    assert result.status == "error"
    assert "Failed to parse LLM response or response did not match expected schema" in result.message
    # The specific Pydantic error might be too detailed/brittle for this test, 
    # but checking for part of the message is good.
    assert "validation error" in result.message.lower()
    
    mock_llm_provider.generate_text_async_with_prompt_manager.assert_called_once()
    mock_pcma_agent.store_document_content.assert_not_called() # Should not store if parsing fails

@pytest.mark.asyncio
async def test_invoke_async_pcma_store_document_fails(requirements_tracer_agent, mock_llm_provider, mock_pcma_agent):
    """
    Tests behavior when PCMA fails to store the document.
    """
    project_id = "test_project_id_store_fail"
    source_artifact_doc_id = "source_doc_store_fail"
    target_artifact_doc_id = "target_doc_store_fail"
    
    input_data = RequirementsTracerAgentInput(
        project_id=project_id,
        source_artifact_doc_id=source_artifact_doc_id,
        target_artifact_doc_id=target_artifact_doc_id
    )

    mock_source_content = DocumentContent(doc_id=source_artifact_doc_id, content="Source", metadata={})
    mock_target_content = DocumentContent(doc_id=target_artifact_doc_id, content="Target", metadata={})
    mock_pcma_agent.get_document_by_id.side_effect = [mock_source_content, mock_target_content]

    llm_response_data = {
        "traceability_report": "Valid report.",
        "confidence_score": 0.9
    }
    mock_llm_provider.generate_text_async_with_prompt_manager.return_value = llm_response_data
    
    mock_pcma_agent.store_document_content.side_effect = Exception("ChromaDB unavailable")

    # Act
    result = await requirements_tracer_agent.invoke_async(input_data)

    # Assert
    assert isinstance(result, RequirementsTracerAgentOutput)
    assert result.project_id == project_id
    assert result.traceability_report_doc_id is None
    assert result.status == "error"
    assert "Failed to store traceability report" in result.message
    assert "Exception: ChromaDB unavailable" in result.message
    
    mock_pcma_agent.store_document_content.assert_called_once() 