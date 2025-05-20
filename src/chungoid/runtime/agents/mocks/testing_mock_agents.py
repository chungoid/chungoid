import logging
from typing import Optional, Dict, Any, List, Literal, Union, NewType, Annotated, ClassVar, TypeVar, Generic
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator, ConfigDict, FilePath, DirectoryPath
import asyncio
import time
import os
import json
import uuid

from chungoid.schemas.common import ArbitraryModel, InputOutputContextPathStr # Assuming this exists and is correct
from chungoid.schemas.common_enums import StageStatus # For mock context
from chungoid.utils.agent_registry_meta import AgentCategory # AgentCategory is here
from chungoid.utils.agent_registry import AgentCard # AgentCard for get_agent_card_static
from chungoid.schemas.errors import AgentErrorDetails # CORRECTED IMPORT

# Corrected import path
from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema 

logger = logging.getLogger(__name__)

from chungoid.utils.agent_registry_meta import AgentCategory # This was fixed - REMOVED AgentProfile

# region Mock Agent Input/Output Schemas

# --- MockSetupAgentV1 ---

class MockSetupAgentV1Input(BaseModel):
    request_clarification_downstream: bool = Field(False, description="If true, output message will trigger ADD_STAGE test case.")
    initial_message: str = Field(..., description="Initial message from the setup stage.")
    output_file_path: Optional[str] = Field(None, description="Path to the output file.")
    output_file_content: Optional[str] = Field(None, description="Content to write to the output file.")

class MockSetupAgentV1Output(BaseModel):
    message: str = Field(..., description="Output message from setup.")
    output_file_written: bool = Field(False, description="Indicates if the output file was written.")

class MockSetupAgentV1(BaseAgent[MockSetupAgentV1Input, MockSetupAgentV1Output]):
    AGENT_ID: ClassVar[str] = "mock_setup_agent_v1"
    AGENT_NAME: ClassVar[str] = "Mock Setup Agent V1"
    AGENT_DESCRIPTION: ClassVar[str] = "A mock agent that performs a simple setup step and outputs a message."

    def __init__(self, config: Optional[Dict[str, Any]] = None, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        
        current_system_context = system_context
        self._logger_instance = current_system_context.get("logger") if current_system_context and isinstance(current_system_context.get("logger"), logging.Logger) else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger_instance.info(f"{self.__class__.__name__} instance created. Config via _config_internal (if needed). Logger set.")

    async def __call__(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockSetupAgentV1Output:
        self._logger_instance.debug(f"{self.AGENT_NAME} __call__ invoked.")
        # MockSetupAgentV1's invoke_async expects Dict, not the pydantic model directly for its first arg.
        return await self.invoke_async(inputs, full_context)

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockSetupAgentV1Output:
        parsed_inputs = MockSetupAgentV1Input(**inputs)
        self._logger_instance.info(f"Mock Setup Agent V1 invoked with request_clarification_downstream={parsed_inputs.request_clarification_downstream}.")
        self._logger_instance.info(f"Mock Setup Agent V1: parsed_inputs.initial_message = '{parsed_inputs.initial_message}'")

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
                self._logger_instance.info(f"Mock Setup Agent V1: Wrote content to {target_path}")
                file_written = True
            except Exception as e:
                self._logger_instance.error(f"Mock Setup Agent V1: Error writing output file '{parsed_inputs.output_file_path}': {e}")

        self._logger_instance.info(f"Mock Setup Agent V1: final output_message to be returned = '{output_message}'")
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
    failure_message_override: Optional[str] = Field(None, description="Allows overriding the default failure message.")
    modified_by_reviewer_TC_P2F1_MODIFY_02: Optional[bool] = Field(None, description="Marker set by reviewer for TC_P2F1_MODIFY_02.")

class MockFailPointAgentV1Output(BaseModel):
    processed_message: str = Field(..., description="Message after processing, if successful.")

# Class-level dictionary to track invocation counts for simulating transient errors
# Key: "run_id:stage_id", Value: invocation_count
_MOCK_FAIL_POINT_INVOCATION_COUNTS: Dict[str, int] = {}

class MockFailPointAgentV1(BaseAgent[MockFailPointAgentV1Input, MockFailPointAgentV1Output]):
    AGENT_ID: ClassVar[str] = "mock_fail_point_agent_v1"
    AGENT_NAME: ClassVar[str] = "Mock Fail Point Agent V1"
    AGENT_DESCRIPTION: ClassVar[str] = "A mock agent that can be configured to fail or succeed, for testing error handling."

    def __init__(self, config: Optional[Dict[str, Any]] = None, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        
        current_system_context = system_context
        self._logger_instance = current_system_context.get("logger") if current_system_context and isinstance(current_system_context.get("logger"), logging.Logger) else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger_instance.info(f"{self.__class__.__name__} instance created. Config via _config_internal (if needed). Logger set.")

        # Hack: Clear the global/static dictionary on new instance creation FOR TESTING ONLY.
        # This should NOT be in production code. Awaiting proper run_id/stage_id based counters.
        # MockFailPointAgentV1._MOCK_FAIL_POINT_INVOCATION_COUNTS.clear() 
        # self._logger_instance.warning("MockFailPointAgentV1._MOCK_FAIL_POINT_INVOCATION_COUNTS cleared on new instance (DEBUG HACK)")

    async def __call__(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockFailPointAgentV1Output:
        self._logger_instance.debug(f"{self.AGENT_NAME} __call__ invoked.")
        # MockFailPointAgentV1's invoke_async expects Dict for its first arg.
        return await self.invoke_async(inputs, full_context)

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockFailPointAgentV1Output:
        parsed_inputs = MockFailPointAgentV1Input(**inputs)
        
        self._logger_instance.debug(f"{self.AGENT_NAME}: INVOKE_ASYNC parsed_inputs.failure_message_override: '{parsed_inputs.failure_message_override}'")

        # --- Enhanced run_id and stage_id extraction for debugging ---
        derived_run_id = "unknown_run_initial"
        derived_stage_id = "unknown_stage_initial"

        # Access config and system_context via _config_internal and _system_context_internal from BaseAgent
        current_config = self._config_internal
        current_system_context = self._system_context_internal

        if current_config and "system_context" in current_config: # This check seems convoluted, system_context is usually separate
            # This block might indicate a misunderstanding of how system_context is typically passed/accessed.
            # Usually, system_context is directly available via self._system_context_internal.
            # The config dictionary passed to the agent might also contain a *copy* or reference to system_context parts.
            self._logger_instance.debug(f"{self.AGENT_NAME}: Found 'system_context' key within self._config_internal. This is unusual.")
            # Let's prioritize self._system_context_internal if available
            if current_system_context:
                derived_run_id = current_system_context.get("run_id", "unknown_run_from_direct_sys_context")
                derived_stage_id = current_system_context.get("current_stage_id", "unknown_stage_from_direct_sys_context")
                self._logger_instance.debug(f"{self.AGENT_NAME}: Used self._system_context_internal. run_id='{derived_run_id}', stage_id='{derived_stage_id}'")
            else: # Fallback to the nested one if direct one is missing
                cfg_sys_ctx = current_config.get("system_context", {})
                derived_run_id = cfg_sys_ctx.get("run_id", "unknown_run_from_config_system_context")
                derived_stage_id = cfg_sys_ctx.get("current_stage_id", "unknown_stage_from_config_system_context")
                self._logger_instance.debug(f"{self.AGENT_NAME}: Used system_context nested in self._config_internal. run_id='{derived_run_id}', stage_id='{derived_stage_id}'")
        elif current_system_context: # Preferred way to access system context
            derived_run_id = current_system_context.get("run_id", "unknown_run_from_sys_context")
            derived_stage_id = current_system_context.get("current_stage_id", "unknown_stage_from_sys_context")
            self._logger_instance.debug(f"{self.AGENT_NAME}: Used self._system_context_internal. run_id='{derived_run_id}', stage_id='{derived_stage_id}'")
        elif full_context: # Fallback if no internal system_context
            derived_run_id = full_context.get("run_id", "unknown_run_from_full_context")
            # current_stage_id is typically NOT directly in full_context, but check just in case
            derived_stage_id = full_context.get("current_stage_id", "unknown_stage_from_full_context")
            self._logger_instance.debug(f"{self.AGENT_NAME}: Used full_context for run_id/stage_id. run_id='{derived_run_id}', stage_id='{derived_stage_id}' (Note: stage_id from full_context is unusual)")
        else:
            self._logger_instance.warning(f"{self.AGENT_NAME}: Neither self.config.system_context nor full_context available for run_id/stage_id derivation.")
        # --- End enhanced extraction ---

        invocation_key = f"{derived_run_id}:{derived_stage_id}"
        
        self._logger_instance.debug(f"{self.AGENT_NAME}: Constructed invocation_key: '{invocation_key}'")
        self._logger_instance.debug(f"{self.AGENT_NAME}: State of _MOCK_FAIL_POINT_INVOCATION_COUNTS (module global) before get: {_MOCK_FAIL_POINT_INVOCATION_COUNTS}")
        
        current_attempt = _MOCK_FAIL_POINT_INVOCATION_COUNTS.get(invocation_key, 0) + 1
        _MOCK_FAIL_POINT_INVOCATION_COUNTS[invocation_key] = current_attempt
        
        self._logger_instance.debug(f"{self.AGENT_NAME}: State of _MOCK_FAIL_POINT_INVOCATION_COUNTS (module global) after set: {_MOCK_FAIL_POINT_INVOCATION_COUNTS}")


        self._logger_instance.info(f"{self.AGENT_NAME} invoked (Attempt {current_attempt} for {invocation_key}). Inputs: trigger_fail={parsed_inputs.trigger_fail}, setup_message='{parsed_inputs.setup_message}', modified_marker={parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}.")

        # For TC_P2F1_OTHER_01 (RETRY_STAGE_AS_IS), we want it to fail on attempt 1 if trigger_fail is true,
        # then succeed on attempt 2 (the retry) if trigger_fail is still true.
        if parsed_inputs.trigger_fail and current_attempt == 1:
            # Check if this is the RETRY_AS_IS test case via a hint in setup_message if needed,
            # or assume any first attempt with trigger_fail=true might be part of this test.
            self._logger_instance.debug(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}): Checking failure logic. failure_message_override='{parsed_inputs.failure_message_override}'")
            
            # MODIFICATION: Change error message if setup_message indicates an LLM test for TC_P2F1_LLM_MODIFY_01
            if "TC_P2F1_LLM_MODIFY_01" in parsed_inputs.setup_message:
                self._logger_instance.info(f"{self.AGENT_NAME}: Detected LLM test TC_P2F1_LLM_MODIFY_01 in setup_message. Raising specific error.")
                error_msg_to_raise = f"LLM Test: Intentional failure for stage_B_fail_point to trigger reviewer (TC_P2F1_LLM_MODIFY_01 attempt {current_attempt} for {invocation_key})."
            else: # This is the path for TC_P2F1_OTHER_01 (RETRY_AS_IS)
                error_msg_to_raise = parsed_inputs.failure_message_override or \
                                     f"Intentional transient failure for testing RETRY_STAGE_AS_IS (Attempt {current_attempt} for {invocation_key})."

            self._logger_instance.warning(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}) is intentionally failing. Message: '{error_msg_to_raise}'")
            raise ValueError(error_msg_to_raise) # Use ValueError for consistency with other mock error
        elif parsed_inputs.trigger_fail and current_attempt > 1:
            self._logger_instance.info(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}): trigger_fail is True, but this is attempt > 1, so succeeding (simulating transient error resolved)." )
            # MODIFICATION FOR SUCCESS CRITERIA:
            # The success criteria for stage_B_fail_point in test_reviewer_flow_v1.yaml for TC_P2F1_OTHER_01 expects this specific message.
            final_processed_message = "Passed after transient failure mock for TC_P2F1_OTHER_01. Original Setup: " + parsed_inputs.setup_message
            self._logger_instance.info(f"Final 'processed_message' being returned by {self.AGENT_NAME} after successful retry: \"{final_processed_message}\"")
            return MockFailPointAgentV1Output(processed_message=final_processed_message)
        elif parsed_inputs.trigger_fail: # Should not be reached if current_attempt logic is exhaustive
             self._logger_instance.warning(f"{self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}) is intentionally failing as trigger_fail is True (unexpected state)." )
             raise ValueError(f"Intentional failure for {self.AGENT_NAME} (Attempt {current_attempt} for {invocation_key}).")

        current_trigger_fail_val = parsed_inputs.trigger_fail
        current_setup_message_val = parsed_inputs.setup_message
        current_modified_marker_val = parsed_inputs.modified_by_reviewer_TC_P2F1_MODIFY_02

        processed_base = f"Fail point processed message. Trigger fail was: {current_trigger_fail_val}. Setup: '{current_setup_message_val}'."
        
        self._logger_instance.info(f"{self.AGENT_NAME}: Base message constructed: \"{processed_base}\"")
        self._logger_instance.info(f"{self.AGENT_NAME}: BEFORE IF: current_modified_marker_val is '{current_modified_marker_val}' (type: {type(current_modified_marker_val)})" )

        final_processed_message = processed_base
        if current_modified_marker_val is not None:
            append_string = f" Modified by reviewer TC_P2F1_MODIFY_02: {current_modified_marker_val}."
            self._logger_instance.info(f"{self.AGENT_NAME}: Marker IS NOT None. Appending: \"{append_string}\"")
            final_processed_message += append_string
        else:
            self._logger_instance.info(f"{self.AGENT_NAME}: Marker IS None. NOT appending modifier string.")
        
        self._logger_instance.info(f"Final 'processed_message' being returned by {self.AGENT_NAME}: \"{final_processed_message}\"")
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


class MockVerifyAgentV1(BaseAgent[MockVerifyAgentV1Input, MockVerifyAgentV1Output]):
    AGENT_ID: ClassVar[str] = "mock_verify_agent_v1"
    AGENT_NAME: ClassVar[str] = "Mock Verification Agent V1"
    AGENT_DESCRIPTION: ClassVar[str] = "A mock agent that logs its inputs for verifying flow state after potential modifications."

    def __init__(self, config: Optional[Dict[str, Any]] = None, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        
        current_system_context = system_context
        self._logger_instance = current_system_context.get("logger") if current_system_context and isinstance(current_system_context.get("logger"), logging.Logger) else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger_instance.info(f"{self.__class__.__name__} instance created. Config via _config_internal (if needed). Logger set.")

    async def __call__(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockVerifyAgentV1Output:
        self._logger_instance.debug(f"{self.AGENT_NAME} __call__ invoked.")
        # MockVerifyAgentV1's invoke_async expects Dict for its first arg.
        return await self.invoke_async(inputs, full_context)

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockVerifyAgentV1Output:
        parsed_inputs = MockVerifyAgentV1Input(**inputs)
        self._logger_instance.info(f"{self.AGENT_NAME} invoked. Inputs received: {parsed_inputs.model_dump_json()}")
        
        # Construct the verification notes
        # For TC_P2F1_OTHER_01, stage_C_verify success criteria expects a specific string part.
        notes = f"Verification stage reached. Setup: '{parsed_inputs.setup_data}'. "
        notes += f"Stage B output after retry: {parsed_inputs.fail_point_data}. " # Added prefix and adjusted structure
        notes += f"Clarification: '{parsed_inputs.clarification_data}'."
        
        self._logger_instance.info(f"{self.AGENT_NAME}: Constructed verification_notes: '{notes}'")
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

class MockClarificationAgentV1(BaseAgent[MockClarificationAgentV1Input, MockClarificationAgentV1Output]):
    AGENT_ID: ClassVar[str] = "mock_clarification_agent_v1"
    AGENT_NAME: ClassVar[str] = "Mock Clarification Agent V1"
    AGENT_DESCRIPTION: ClassVar[str] = "A mock agent that provides a canned clarification response."

    def __init__(self, config: Optional[Dict[str, Any]] = None, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        
        current_system_context = system_context
        self._logger_instance = current_system_context.get("logger") if current_system_context and isinstance(current_system_context.get("logger"), logging.Logger) else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger_instance.info(f"{self.__class__.__name__} instance created. Config via _config_internal (if needed). Logger set.")

    async def __call__(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockClarificationAgentV1Output:
        self._logger_instance.debug(f"{self.AGENT_NAME} __call__ invoked.")
        # MockClarificationAgentV1's invoke_async expects Dict for its first arg.
        return await self.invoke_async(inputs, full_context)

    async def invoke_async(self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None) -> MockClarificationAgentV1Output:
        parsed_inputs = MockClarificationAgentV1Input(**inputs)
        self._logger_instance.info(f"{self.AGENT_NAME} invoked with query: '{parsed_inputs.query}'.")
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

class MockAlternativeAgentV1Input(ArbitraryModel):
    setup_message: str = Field(..., description="A message from the setup stage.")
    trigger_fail: bool = Field(default=False, description="If true, this agent should simulate a failure.")
    modified_by_reviewer_TC_P2F1_MODIFY_02: Optional[str] = Field(None, description="Marker set by reviewer for TC_P2F1_MODIFY_02")

class MockAlternativeAgentV1Output(ArbitraryModel):
    alternative_processed_message: str
    original_trigger_fail_value: bool

class MockAlternativeAgentV1(BaseAgent[MockAlternativeAgentV1Input, MockAlternativeAgentV1Output]):
    AGENT_ID: ClassVar[str] = "mock_alternative_agent_v1"
    AGENT_NAME: ClassVar[str] = "Mock Alternative Agent V1"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "A mock agent that always succeeds, used as an alternative by reviewer suggestions."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.TESTING_MOCK
    PROFILE: ClassVar[Dict[str, Any]] = {
        "version_semver": "1.0.0",
        "primitive_support": ["mock_success"],
        "dependencies": []
    }
    InputSchema: ClassVar[type] = MockAlternativeAgentV1Input
    OutputSchema: ClassVar[type] = MockAlternativeAgentV1Output

    def __init__(self, config: Optional[Dict[str, Any]] = None, system_context: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, system_context=system_context)
        
        current_system_context = system_context
        self._logger_instance = current_system_context.get("logger") if current_system_context and isinstance(current_system_context.get("logger"), logging.Logger) else logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger_instance.info(f"{self.__class__.__name__} instance created. Config via _config_internal (if needed). Logger set.")

    async def __call__(self, inputs: MockAlternativeAgentV1Input, full_context: Optional[Dict[str, Any]] = None) -> MockAlternativeAgentV1Output:
        self._logger_instance.debug(f"{self.AGENT_NAME} __call__ invoked directly with Pydantic model.")
        # MockAlternativeAgentV1's invoke_async *does* expect the Pydantic model directly.
        return await self.invoke_async(inputs, full_context)

    async def invoke_async(self, inputs: MockAlternativeAgentV1Input, full_context: Optional[Dict[str, Any]] = None) -> MockAlternativeAgentV1Output:
        run_id_to_log = self._system_context_internal.get('run_id') if self._system_context_internal else None
        stage_id_to_log = self._system_context_internal.get('current_stage_id') if self._system_context_internal else None # Note: 'current_stage_id' is the key used by orchestrator
        print(f"MOCK_ALTERNATIVE_AGENT_V1: INVOKED! Run ID from system_context: {run_id_to_log}, Stage ID from system_context: {stage_id_to_log}")
        self._logger_instance.info(f"{self.AGENT_NAME} invoked. This agent ALWAYS SUCCEEDS regardless of 'trigger_fail'.")
        # inputs is already a validated MockAlternativeAgentV1Input instance
        print(f"MOCK_ALTERNATIVE_AGENT_V1: Received inputs: trigger_fail={inputs.trigger_fail}, setup_message='{inputs.setup_message}', modified_marker={inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}")

        alt_message = f"Alternative agent processed successfully. Original setup: '{inputs.setup_message}'. TriggerFail input was: {inputs.trigger_fail}."
        if inputs.modified_by_reviewer_TC_P2F1_MODIFY_02 is not None:
            alt_message += f" Saw modified_marker: {inputs.modified_by_reviewer_TC_P2F1_MODIFY_02}."
        
        print(f"MOCK_ALTERNATIVE_AGENT_V1: Returning message: \"{alt_message}\"")
        return MockAlternativeAgentV1Output(
            alternative_processed_message=alt_message,
            original_trigger_fail_value=inputs.trigger_fail
        )

    def get_agent_card(self) -> AgentCard:
        return AgentCard(
            agent_id=self.AGENT_ID,
            name=self.AGENT_NAME,
            description=self.DESCRIPTION,
            input_schema=self.InputSchema.model_json_schema(),
            output_schema=self.OutputSchema.model_json_schema(),
            categories=[self.CATEGORY.value],
            capability_profile=self.PROFILE
        )

# --- ADD IMPORT FOR THE MISSING MOCK AGENT ---
from chungoid.runtime.agents.mocks.mock_system_requirements_gathering_agent import MockSystemRequirementsGatheringAgent
from chungoid.runtime.agents.mocks.mock_human_input_agent import MockHumanInputAgent # ADDED IMPORT
# --- END ADD IMPORT ---

# Helper list for registration script
ALL_MOCK_TESTING_AGENTS = [
    MockSetupAgentV1,
    MockFailPointAgentV1,
    MockVerifyAgentV1,
    MockClarificationAgentV1, # Added new agent
    MockAlternativeAgentV1,
    MockSystemRequirementsGatheringAgent, # <<< ADDED THE AGENT HERE
    MockHumanInputAgent # ADDED MockHumanInputAgent to the list
]

def get_all_mock_testing_agent_cards() -> List[AgentCard]:
    return [agent().get_agent_card() for agent in ALL_MOCK_TESTING_AGENTS]

def get_mock_agent_fallback_map() -> Dict[str, Any]: # Changed type hint for now, will be AgentCallable
    """Returns a map of mock agent IDs to their classes for fallback registration."""
    # For RegistryAgentProvider, we typically provide the class itself, 
    # or a pre-configured instance, or a factory/bound method.
    # Providing the class is simplest if the provider can instantiate it.
    # Based on current RegistryAgentProvider logic, providing the class is fine.
    # Or, if agents need specific default construction, an instance can be provided.
    # Let's provide the class for now.
    return {agent.AGENT_ID: agent for agent in ALL_MOCK_TESTING_AGENTS} 