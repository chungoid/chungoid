import pytest
import json
import uuid
from unittest.mock import AsyncMock, MagicMock, call # Added call
from typing import List, Dict, Any, Optional # Added Dict, Any, Optional

# Agent and schema imports
from chungoid.agents.autonomous_engine.automated_refinement_coordinator_agent import (
    AutomatedRefinementCoordinatorAgent_v1,
    ARCAReviewInput,
    ARCAReviewOutput,
    # Assuming other internal schemas/enums might be needed for deeper testing later
)
from chungoid.schemas.common import ConfidenceScore, ArtifactType
from chungoid.schemas.agent_code_debugging import DebuggingTaskInput # For when ARCA calls debugger
from chungoid.schemas.agent_logs import ARCALogEntry # For log_event_to_pcma

# Mock imports
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1, AGENT_REFLECTIONS_AND_LOGS_COLLECTION
from chungoid.agents.autonomous_engine.proactive_risk_assessor_agent import ProactiveRiskAssessorAgent_v1 # For mocking PRAA if ARCA calls it
from chungoid.agents.autonomous_engine.code_debugging_agent import CodeDebuggingAgent_v1 # For mocking debugger
# Potentially more agent mocks if ARCA orchestrates them directly: SmartCodeIntegrationAgent, SystemTestRunnerAgent etc.

@pytest.fixture
def mock_llm_provider_arca():
    mock = AsyncMock(spec=LLMProvider)
    # ARCA's primary LLM use is for its own decision making, if any (prompt not shown yet)
    # For now, assume it doesn't make direct LLM calls in the simplest path, or mock a generic response.
    mock.generate_text_async_with_prompt_manager = AsyncMock(return_value=json.dumps({"decision": "proceed"}))
    return mock

@pytest.fixture
def mock_prompt_manager_arca():
    mock = AsyncMock(spec=PromptManager)
    return mock

@pytest.fixture
def mock_pcma_arca():
    mock = AsyncMock(spec=ProjectChromaManagerAgent_v1)
    async def _mock_get_document_by_id(project_id: str, doc_id: str, collection_name: Optional[str] = None):
        mock_doc = AsyncMock()
        if doc_id == "artifact_to_review_id":
            mock_doc.document_content = "Content of artifact being reviewed by ARCA."
            mock_doc.metadata = {"artifact_type": "LOPRD", "confidence_score_value": 0.95, "name": "TestLOPRD"}
        elif doc_id == "praa_risk_report_id":
            mock_doc.document_content = "Mock PRAA Risk Report Content"
            mock_doc.metadata = {"artifact_type": "RiskAssessmentReport"}
        elif doc_id == "faulty_code_doc_id":
            mock_doc.document_content = "def bad_function(): return 1/0"
            mock_doc.metadata = {"artifact_type": "CodeModule"}
        else:
            mock_doc.document_content = f"Default ARCA mock content for {doc_id}"
            mock_doc.metadata = {"artifact_type": "Unknown"}
        mock_doc.id = doc_id
        return mock_doc

    async def _mock_store_document_content(project_id: str, collection_name: str, document_content: any, metadata: dict, document_id:str = None, document_relative_path:str=None):
        return document_id if document_id else str(uuid.uuid4())
    
    mock.get_document_by_id = AsyncMock(side_effect=_mock_get_document_by_id)
    mock.store_document_content = AsyncMock(side_effect=_mock_store_document_content) # For logs mainly
    # Mock log_arca_event - this is an internal PCMA method called by ARCA
    # In PCMA, log_arca_event uses store_document_content, so mocking store_document_content is often enough
    # but we can also mock the higher-level method if needed for specific checks.
    mock.log_arca_event = AsyncMock(return_value=MagicMock(status="SUCCESS", log_id=str(uuid.uuid4())))
    return mock

@pytest.fixture
def mock_code_debugging_agent():
    mock_agent = AsyncMock(spec=CodeDebuggingAgent_v1)
    # Define a default successful response for the debugger
    async def _mock_invoke_debugger(task_input: DebuggingTaskInput, full_context: Optional[Dict[str, Any]] = None):
        return MagicMock(
            status="SUCCESS_FIX_PROPOSED",
            proposed_code_changes="def good_function(): return 1+1",
            confidence_score=0.9
        )
    mock_agent.invoke_async = _mock_invoke_debugger
    return mock_agent

# Minimal ARCA test - checks initialization and a very simple non-failure path
@pytest.mark.asyncio
async def test_arca_initialization_and_simple_loprd_approve(mock_llm_provider_arca, mock_prompt_manager_arca, mock_pcma_arca):
    # ARCA also needs other agents passed to its __init__ for orchestration
    # These would be mocked as well. For this simple test, let's assume they are not hit.
    # Or pass MagicMock() for them if the __init__ requires them.
    mock_agent_names = [
        "ProductAnalystAgent_v1", "ArchitectAgent_v1", "BlueprintReviewerAgent_v1",
        "BlueprintToFlowAgent_v1", "SmartCodeGeneratorAgent_v1", "SmartCodeIntegrationAgent_v1",
        "SmartTestGeneratorAgent_v1", "SystemTestRunnerAgent_v1", "ProjectDocumentationAgent_v1",
        "CodeDebuggingAgent_v1", "ProactiveRiskAssessorAgent_v1", "RequirementsTracerAgent_v1"
    ]
    mock_other_agents = {name: MagicMock() for name in mock_agent_names}

    agent = AutomatedRefinementCoordinatorAgent_v1(
        llm_provider=mock_llm_provider_arca,
        prompt_manager=mock_prompt_manager_arca,
        project_chroma_manager=mock_pcma_arca,
        agent_references=mock_other_agents,
        # Pass other required ARCA init params if any, e.g. cycle_id from a higher context
        # For unit test, we can set a mock cycle_id
        initial_cycle_id_override="test_cycle_001" 
    )

    task_input = ARCAReviewInput(
        project_id="test_arca_project_simple",
        task_id="test_arca_task_simple_123",
        artifact_doc_id="artifact_to_review_id",
        artifact_type=ArtifactType.LOPRD,
        # For this simple case, assume PRAA report is good or not applicable
        praa_risk_assessment_report_doc_id=None, # "praa_risk_report_id_good",
        praa_optimization_report_doc_id=None,
        rta_traceability_report_doc_id=None,
        # No code module path or test failure for LOPRD review path
    )

    # Mock PCMA for PRAA report if it were to be fetched
    # For this test, we assume quality_metric is high enough to not trigger detailed PRAA/RTA processing or LLM decision
    # The current ARCA logic for LOPRDs directly goes to logging if confidence is high.
    
    output = await agent.invoke_async(task_input)

    assert output.status == "SUCCESS_CYCLE_ELEMENT_APPROVED" # ARCA approves if confidence is high
    assert output.decision_summary is not None
    assert "LOPRD artifact_to_review_id approved based on high confidence" in output.decision_summary.lower()
    assert output.next_actions_recommended is not None
    # Depending on ARCA logic, next_actions might suggest moving to ArchitectAgent

    # Verify PCMA logging calls
    # ARCA logs an INVOCATION_START and then a DECISION_MADE or similar event.
    assert mock_pcma_arca.log_arca_event.call_count >= 1 # At least invocation start
    
    # Example check for a specific log event type if needed:
    # log_calls = mock_pcma_arca.log_arca_event.call_args_list
    # assert any(call.args[2].event_type == "ARCA_INVOCATION_START" for call in log_calls)
    # assert any(call.args[2].event_type == "ARCA_DECISION_MADE" and "approved" in call.args[2].details.get("summary","").lower() for call in log_calls)

    print(f"ARCA simple LOPRD approve test completed. Output: {output.model_dump_json(indent=2)}")


@pytest.mark.asyncio
async def test_arca_handles_code_test_failure_invokes_debugger(mock_llm_provider_arca, mock_prompt_manager_arca, mock_pcma_arca, mock_code_debugging_agent):
    mock_agent_names_for_debug_test = [
        "ProductAnalystAgent_v1", "ArchitectAgent_v1", "BlueprintReviewerAgent_v1",
        "BlueprintToFlowAgent_v1", "SmartCodeGeneratorAgent_v1", "SmartCodeIntegrationAgent_v1",
        "SmartTestGeneratorAgent_v1", "SystemTestRunnerAgent_v1", "ProjectDocumentationAgent_v1",
        "ProactiveRiskAssessorAgent_v1", "RequirementsTracerAgent_v1"
    ] # All agents except CodeDebuggingAgent
    mock_other_agents = {name: MagicMock() for name in mock_agent_names_for_debug_test}
    mock_other_agents["CodeDebuggingAgent_v1"] = mock_code_debugging_agent # Add the specific mock for debugger

    agent = AutomatedRefinementCoordinatorAgent_v1(
        llm_provider=mock_llm_provider_arca,
        prompt_manager=mock_prompt_manager_arca,
        project_chroma_manager=mock_pcma_arca,
        agent_references=mock_other_agents,
        initial_cycle_id_override="test_cycle_debug_001"
    )

    failed_test_report = {
        "test_suite_path": "/tests/test_module.py",
        "failed_tests": [
            {"test_name": "test_division_by_zero", "error_message": "ZeroDivisionError", "stack_trace": "..."}
        ],
        "summary": {"total": 1, "failed": 1, "passed": 0}
    }

    task_input = ARCAReviewInput(
        project_id="test_arca_debug_project",
        task_id="test_arca_task_debug_456",
        artifact_doc_id="faulty_code_doc_id", # Doc ID of the code that failed tests
        artifact_type=ArtifactType.CodeModule_TestFailure,
        code_module_file_path="src/module.py",
        failed_test_report_details=failed_test_report,
        # Assume LOPRD/Blueprint IDs are available for context if needed by debugger
        relevant_loprd_doc_ids_for_context=["loprd_ctx_for_debug"],
        relevant_blueprint_doc_ids_for_context=["bp_ctx_for_debug"]
    )

    # Mock PCMA for fetching LOPRD/Blueprint context for debugger
    async def _get_doc_for_debugger_context(project_id: str, doc_id: str, collection_name: Optional[str] = None):
        mock_doc = AsyncMock()
        if doc_id == "faulty_code_doc_id":
            mock_doc.document_content = "def bad_function(): return 1/0"
        elif doc_id == "loprd_ctx_for_debug":
            mock_doc.document_content = "LOPRD: Function should return a number."
        elif doc_id == "bp_ctx_for_debug":
            mock_doc.document_content = "Blueprint: bad_function is part of CoreLogic."
        else:
            mock_doc.document_content = "Default context"
        mock_doc.id = doc_id
        return mock_doc
    mock_pcma_arca.get_document_by_id.side_effect = _get_doc_for_debugger_context

    output = await agent.invoke_async(task_input)

    assert output.status == "REFINEMENT_ATTEMPTED_DEBUGGING"
    mock_code_debugging_agent.invoke_async.assert_called_once()
    debugger_input: DebuggingTaskInput = mock_code_debugging_agent.invoke_async.call_args[0][0]
    
    assert debugger_input.project_id == "test_arca_debug_project"
    assert debugger_input.faulty_code_path == "src/module.py"
    assert debugger_input.faulty_code_snippet == "def bad_function(): return 1/0" # Fetched via PCMA
    assert len(debugger_input.failed_test_reports) == 1
    assert debugger_input.failed_test_reports[0]['test_name'] == "test_division_by_zero"
    assert debugger_input.relevant_loprd_requirements_ids == ["loprd_ctx_for_debug"]
    assert debugger_input.relevant_blueprint_section_ids == ["bp_ctx_for_debug"]
    
    # ARCA should have logged events
    assert mock_pcma_arca.log_arca_event.call_count >= 2 # Invocation + Debugger Invoked (at least)

    print(f"ARCA code test failure handling test completed. Output: {output.model_dump_json(indent=2)}")

# Add more tests for ARCA: 
# - Different artifact types (Blueprint, Plan)
# - PRAA/RTA integration paths
# - LLM-driven decision making (if ARCA has its own complex prompts)
# - Failure to retrieve context from PCMA
# - Debugging loops (max attempts)
# - Escalation paths 