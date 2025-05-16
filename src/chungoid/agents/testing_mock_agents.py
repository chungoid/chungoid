import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

from pydantic import BaseModel, Field

from chungoid.utils.agent_registry import AgentCard # Assuming AgentCard is here
# from chungoid.schemas.common_enums import AgentCategory # If needed for cards

logger = logging.getLogger(__name__)

# --- MockSetupAgentV1 ---

class MockSetupAgentV1Input(BaseModel):
    request_clarification_downstream: bool = Field(False, description="If true, output message will trigger ADD_STAGE test case.")
    initial_message: str = Field(..., description="Initial message from the setup stage.")
    output_file_path: Optional[str] = Field(None, description="Path to the output file.")
    output_file_content: Optional[str] = Field(None, description="Content to write to the output file.")

class MockSetupAgentV1Output(BaseModel):
    message: str = Field(..., description="Output message from setup.")
    output_file_written: bool = Field(False, description="Indicates if the output file was written.")

class MockSetupAgentV1:
    AGENT_ID = "mock_setup_agent_v1"
    AGENT_NAME = "Mock Setup Agent V1"
    AGENT_DESCRIPTION = "A mock agent that performs a simple setup step and outputs a message."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockSetupAgentV1Output:
        parsed_inputs = MockSetupAgentV1Input(**inputs)
        logger.info(f"Mock Setup Agent V1 invoked with request_clarification_downstream={parsed_inputs.request_clarification_downstream}.")
        logger.info(f"Mock Setup Agent V1: parsed_inputs.initial_message = '{parsed_inputs.initial_message}'")

        output_message = parsed_inputs.initial_message
        if parsed_inputs.request_clarification_downstream:
            output_message += " This setup indicates stage_B_needs_clarification."

        # If an output file path is provided, write the content to it
        file_written = False
        if parsed_inputs.output_file_path and parsed_inputs.output_file_content is not None:
            try:
                # Ensure project_root_dir is available in config, or fallback
                # This part needs to be robust if the agent is run outside a full orchestrator context
                # For now, let's assume it's primarily for integration tests where config might be sparse.
                # A better approach would be for the orchestrator to pass a resolved base path if needed.
                
                # Assuming current working directory is project root for this mock, which is often the case in tests
                # A more robust solution would involve the orchestrator providing a proper working directory or path resolver.
                target_path = Path(parsed_inputs.output_file_path)
                if not target_path.is_absolute():
                    # This is a simplification; ideally, the agent gets a base path from the orchestrator config
                    # For dummy_project, this would be relative to dummy_project if run from chungoid-mcp
                    base_dir = Path(".") # Fallback to current working directory
                    # A more robust method to get project root if available from context or config
                    # if full_context and full_context.get("project_config", {}).get("project_root_dir"):
                    # base_dir = Path(full_context["project_config"]["project_root_dir"])
                    
                    target_path = base_dir / parsed_inputs.output_file_path

                target_path.parent.mkdir(parents=True, exist_ok=True)
                with open(target_path, "w") as f:
                    f.write(parsed_inputs.output_file_content)
                logger.info(f"Mock Setup Agent V1: Wrote content to {target_path}")
                file_written = True
            except Exception as e:
                logger.error(f"Mock Setup Agent V1: Error writing output file '{parsed_inputs.output_file_path}': {e}")

        logger.info(f"Mock Setup Agent V1: final output_message to be returned = '{output_message}'")
        return MockSetupAgentV1Output(message=output_message, output_file_written=file_written)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            input_schema=MockSetupAgentV1Input.model_json_schema(),
            output_schema=MockSetupAgentV1Output.model_json_schema(),
            categories=["testing", "mock"],
            capability_profile={"mock_operations": ["setup"]}
        )

# --- MockFailPointAgentV1 ---

class MockFailPointAgentV1Input(BaseModel):
    setup_message: str = Field(..., description="Message from the setup stage.")
    trigger_fail: bool = Field(False, description="If true, the agent will raise an exception.")
    modified_by_reviewer_TC_P2F1_MODIFY_02: Optional[bool] = Field(None, description="Marker set by reviewer for TC_P2F1_MODIFY_02.")

class MockFailPointAgentV1Output(BaseModel):
    processed_message: str = Field(..., description="Message after processing, if successful.")

# Class-level dictionary to track invocation counts for simulating transient errors
# Key: "run_id:stage_id", Value: invocation_count
_MOCK_FAIL_POINT_INVOCATION_COUNTS: Dict[str, int] = {}

class MockFailPointAgentV1:
    AGENT_ID = "mock_fail_point_agent_v1"
    AGENT_NAME = "Mock Fail Point Agent V1"
    AGENT_DESCRIPTION = "A mock agent that can be configured to fail or succeed, for testing error handling."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockFailPointAgentV1Output:
        parsed_inputs = MockFailPointAgentV1Input(**inputs)
        
        run_id = "unknown_run"
        stage_id = "unknown_stage"
        if full_context:
            run_id = full_context.get("run_id", "unknown_run_in_context")
            stage_id = full_context.get("current_stage_id", "unknown_stage_in_context")

        invocation_key = f"{run_id}:{stage_id}"
        current_attempt = _MOCK_FAIL_POINT_INVOCATION_COUNTS.get(invocation_key, 0) + 1
        _MOCK_FAIL_POINT_INVOCATION_COUNTS[invocation_key] = current_attempt

        logger.info(f"{self.AGENT_NAME} invoked (Attempt {current_attempt} for {invocation_key}). Inputs: trigger_fail={parsed_inputs.trigger_fail}, setup_message='{parsed_inputs.setup_message}', modified_marker={parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}.")

        # For TC_P2F1_OTHER_01 (RETRY_STAGE_AS_IS), we want it to fail on attempt 1 if trigger_fail is true,
        # then succeed on attempt 2 (the retry) if trigger_fail is still true.
        if parsed_inputs.trigger_fail and current_attempt == 1:
            # Check if this is the RETRY_AS_IS test case via a hint in setup_message if needed,
            # or assume any first attempt with trigger_fail=true might be part of this test.
            # For simplicity, we'll assume if trigger_fail is true, the first attempt fails to allow RETRY_AS_IS to work.
            logger.warning(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}) is intentionally failing as trigger_fail is True and it's the first attempt.")
            # To ensure the test resets for multiple `chungoid flow run` commands, clear the specific key after this run if it fails.
            # This is tricky as the agent instance doesn't know if the overall flow will retry or terminate.
            # For now, let this state persist for the lifetime of the Python process running the mocks.
            # For robust testing across CLI calls, this dictionary should be cleared by a test fixture or similar.
            
            # MODIFICATION: Change error message if setup_message indicates an LLM test for TC_P2F1_LLM_MODIFY_01
            if "TC_P2F1_LLM_MODIFY_01" in parsed_inputs.setup_message:
                logger.info(f"{self.AGENT_NAME}: Detected LLM test TC_P2F1_LLM_MODIFY_01 in setup_message. Raising specific error.")
                raise ValueError(f"LLM Test: Intentional failure for stage_B_fail_point to trigger reviewer (TC_P2F1_LLM_MODIFY_01 attempt {current_attempt} for {invocation_key}).")
            else:
                raise ValueError(f"Intentional transient failure for testing RETRY_STAGE_AS_IS (Attempt {current_attempt} for {invocation_key}).")
        elif parsed_inputs.trigger_fail and current_attempt > 1:
            logger.info(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}): trigger_fail is True, but this is attempt > 1, so succeeding (simulating transient error resolved)." )
            # Fall through to success logic
        elif parsed_inputs.trigger_fail: # Should not be reached if current_attempt logic is exhaustive
             logger.warning(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}) is intentionally failing as trigger_fail is True (unexpected state)." )
             raise ValueError(f"Intentional failure for {self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}).")

        current_trigger_fail_val = parsed_inputs.trigger_fail
        current_setup_message_val = parsed_inputs.setup_message
        current_modified_marker_val = parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02

        processed_base = f"Fail point processed message. Trigger fail was: {current_trigger_fail_val}. Setup: '{current_setup_message_val}'."
        
        logger.info(f"{self.AGENT_NAME}: Base message constructed: \"{processed_base}\"")
        logger.info(f"{self.AGENT_NAME}: BEFORE IF: current_modified_marker_val is '{current_modified_marker_val}' (type: {type(current_modified_marker_val)})")

        final_processed_message = processed_base
        if current_modified_marker_val is not None:
            append_string = f" Modified by reviewer TC_P2F1_MODIFY_02: {current_modified_marker_val}."
            logger.info(f"{self.AGENT_NAME}: Marker IS NOT None. Appending: \"{append_string}\"")
            final_processed_message += append_string
        else:
            logger.info(f"{self.AGENT_NAME}: Marker IS None. NOT appending modifier string.")
        
        logger.info(f"Final 'processed_message' being returned by {self.AGENT_NAME}: \"{final_processed_message}\"")
        return MockFailPointAgentV1Output(processed_message=final_processed_message)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            input_schema=MockFailPointAgentV1Input.model_json_schema(),
            output_schema=MockFailPointAgentV1Output.model_json_schema(),
            categories=["testing", "mock", "failure-simulation"],
            capability_profile={"mock_operations": ["conditional_failure", "processing"]}
        )

# --- MockVerifyAgentV1 ---

class MockVerifyAgentV1Input(BaseModel):
    fail_point_data: Optional[str] = Field(None, description="Data from stage B, if it succeeded or was recovered.")
    setup_data: Optional[str] = Field(None, description="Data from stage A.")
    clarification_data: Optional[str] = Field(None, description="Data from an added clarification stage, if any.")

class MockVerifyAgentV1Output(BaseModel):
    verification_notes: str = Field(..., description="Notes from the verification stage.")
    inputs_received: Dict[str, Any] = Field(..., description="A copy of the inputs received by the agent for logging/assertion.")


class MockVerifyAgentV1:
    AGENT_ID = "mock_verify_agent_v1"
    AGENT_NAME = "Mock Verification Agent V1"
    AGENT_DESCRIPTION = "A mock agent that logs its inputs for verifying flow state after potential modifications."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockVerifyAgentV1Output:
        parsed_inputs = MockVerifyAgentV1Input(**inputs)
        logger.info(f"{self.AGENT_NAME} invoked. Inputs received: {parsed_inputs.model_dump_json()}")
        notes = f"Verification stage reached. Setup: '{parsed_inputs.setup_data}', FailPoint: '{parsed_inputs.fail_point_data}', Clarification: '{parsed_inputs.clarification_data}'"
        return MockVerifyAgentV1Output(verification_notes=notes, inputs_received=parsed_inputs.model_dump())

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            input_schema=MockVerifyAgentV1Input.model_json_schema(),
            output_schema=MockVerifyAgentV1Output.model_json_schema(),
            categories=["testing", "mock", "verification"],
            capability_profile={"mock_operations": ["logging", "state_check"]}
        )

# --- MockClarificationAgentV1 ---

class MockClarificationAgentV1Input(BaseModel):
    query: str = Field(..., description="The query or question needing clarification.")
    original_context_hint: Optional[Dict[str, Any]] = Field(None, description="Hint from original context that might be relevant.")

class MockClarificationAgentV1Output(BaseModel):
    clarification_provided: str = Field(..., description="The clarification provided by the (mocked) user/process.")
    details: str = Field("Mocked clarification based on query.", description="Additional details about the clarification.")

class MockClarificationAgentV1:
    AGENT_ID = "mock_clarification_agent_v1"
    AGENT_NAME = "Mock Clarification Agent V1"
    AGENT_DESCRIPTION = "A mock agent that provides a canned clarification response."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockClarificationAgentV1Output:
        parsed_inputs = MockClarificationAgentV1Input(**inputs)
        logger.info(f"{self.AGENT_NAME} invoked with query: '{parsed_inputs.query}'.")
        clarification = f"Mocked clarification for query: '{parsed_inputs.query}'. Original hint: {parsed_inputs.original_context_hint}"
        return MockClarificationAgentV1Output(clarification_provided=clarification)

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            input_schema=MockClarificationAgentV1Input.model_json_schema(),
            output_schema=MockClarificationAgentV1Output.model_json_schema(),
            categories=["testing", "mock", "clarification"],
            capability_profile={"mock_operations": ["provide_data"]}
        )

# --- MockAlternativeAgentV1 ---

class MockAlternativeAgentV1Input(BaseModel):
    setup_message: str = Field(..., description="Message from the setup stage.")
    trigger_fail: bool = Field(False, description="This agent ignores this field but logs it.")
    modified_by_reviewer_TC_P2F1_MODIFY_02: Optional[bool] = Field(None, description="Marker, logged if present.")

class MockAlternativeAgentV1Output(BaseModel):
    alternative_processed_message: str = Field(..., description="Message after processing by the alternative agent.")
    original_trigger_fail_value: bool

class MockAlternativeAgentV1:
    AGENT_ID = "mock_alternative_agent_v1"
    AGENT_NAME = "Mock Alternative Agent V1"
    AGENT_DESCRIPTION = "A mock agent used as an alternative target for RETRY_STAGE_WITH_CHANGES (agent_id swap)."

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config if config else {}
        logger.info(f"{self.AGENT_NAME} initialized.")

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockAlternativeAgentV1Output:
        parsed_inputs = MockAlternativeAgentV1Input(**inputs)
        
        logger.info(f"{self.AGENT_NAME} invoked. This agent ALWAYS SUCCEEDS regardless of 'trigger_fail'.")
        logger.info(f"{self.AGENT_NAME}: Received trigger_fail={parsed_inputs.trigger_fail}, setup_message='{parsed_inputs.setup_message}', modified_marker={parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}.")

        alt_message = f"Alternative agent processed successfully. Original setup: '{parsed_inputs.setup_message}'. TriggerFail input was: {parsed_inputs.trigger_fail}."
        if parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02 is not None:
            alt_message += f" Saw modified_marker: {parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}."
            
        return MockAlternativeAgentV1Output(
            alternative_processed_message=alt_message,
            original_trigger_fail_value=parsed_inputs.trigger_fail
        )

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.AGENT_DESCRIPTION,
            input_schema=MockAlternativeAgentV1Input.model_json_schema(),
            output_schema=MockAlternativeAgentV1Output.model_json_schema(),
            categories=["testing", "mock"],
            capability_profile={"mock_operations": ["alternative_processing"]}
        )

# Helper list for registration script
ALL_MOCK_TESTING_AGENTS = [
    MockSetupAgentV1,
    MockFailPointAgentV1,
    MockVerifyAgentV1,
    MockClarificationAgentV1, # Added new agent
    MockAlternativeAgentV1
]

def get_all_mock_testing_agent_cards() -> List[AgentCard]:
    return [agent().get_agent_card() for agent in ALL_MOCK_TESTING_AGENTS] 