from typing import Any, Dict, Optional, ClassVar
import logging
from pydantic import BaseModel

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
from chungoid.runtime.agents.agent_base import InputSchema, OutputSchema
from chungoid.schemas.orchestration import SharedContext
# Attempt to import providers for type hinting, but allow Any if it causes cycles during early init
try:
    from chungoid.utils.llm_provider import LLMProvider
    from chungoid.utils.prompt_manager import PromptManager
    from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
except ImportError:
    LLMProvider = Any
    PromptManager = Any
    ProjectChromaManagerAgent_v1 = Any


logger = logging.getLogger(__name__)

class NoOpInput(BaseModel):
    """Input schema for NoOpAgent_v1. Accepts any passthrough data."""
    passthrough_data: Optional[Dict[str, Any]] = None

class NoOpOutput(BaseModel):
    """Output schema for NoOpAgent_v1."""
    message: str
    passthrough_data: Optional[Dict[str, Any]] = None

class NoOpAgent_v1(ProtocolAwareAgent[NoOpInput, NoOpOutput]):
    """
    A No-Operation Agent. It logs its invocation and returns a success message.
    It primarily serves as a placeholder in execution plans where an action is
    defined but no concrete operation needs to be performed by a specialized agent.
    """
    AGENT_ID: ClassVar[str] = "NoOpAgent_v1"
    AGENT_VERSION: ClassVar[str] = "1.0"

    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ["simple_operations"]
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ["status_reporting"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ["agent_communication", "context_sharing"]

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        project_chroma_manager: Optional[ProjectChromaManagerAgent_v1] = None,
        system_context: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
        agent_id: Optional[str] = None,
    ):
        kwargs_for_super = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager,
        }
        if project_chroma_manager:
            kwargs_for_super["project_chroma_manager"] = project_chroma_manager
        if system_context is not None:
            kwargs_for_super["system_context"] = system_context
        if config is not None:
            kwargs_for_super["config"] = config
        
        effective_agent_id = agent_id if agent_id is not None else self.AGENT_ID
        kwargs_for_super["agent_id"] = effective_agent_id
        
        super().__init__(**kwargs_for_super)
        logger.info(f"NoOpAgent_v1 (ID: {self.agent_id}) initialized.")

    # ADDED: Protocol-aware execution method (hybrid approach)
    async def execute_with_protocols(self, task_input: NoOpInput, full_context: Optional[Dict[str, Any]] = None) -> NoOpOutput:
        """
        Execute using appropriate protocol with fallback to traditional method.
        Follows AI agent best practices for hybrid execution.
        """
        try:
            # Determine primary protocol for this agent
            primary_protocol = self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "simple_operations"
            
            protocol_task = {
                "task_input": task_input.dict() if hasattr(task_input, 'dict') else task_input,
                "full_context": full_context,
                "goal": f"Execute {self.AGENT_NAME} specialized task"
            }
            
            protocol_result = self.execute_with_protocol(protocol_task, primary_protocol)
            
            if protocol_result["overall_success"]:
                return self._extract_output_from_protocol_result(protocol_result, task_input)
            else:
                logger.warning("Protocol execution failed, falling back to traditional method")
                return await self.invoke_async(task_input, full_context)
                
        except Exception as e:
            logger.warning(f"Protocol execution error: {e}, falling back to traditional method")
            return await self.invoke_async(task_input, full_context)

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input: NoOpInput) -> NoOpOutput:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return NoOpOutput(
            message=f"NoOpAgent_v1 executed via protocol: {protocol_result.get('protocol_name')}",
            passthrough_data=task_input.passthrough_data
        )

    # MAINTAINED: Original invoke_async method for backward compatibility
    async def invoke_async(
        self,
        task_input: NoOpInput,
        full_context: Optional[SharedContext] = None,
    ) -> NoOpOutput:
        logger.info(
            f"NoOpAgent_v1 (ID: {self.agent_id}) invoked. Input: {task_input}. Context: {full_context}"
        )
        return NoOpOutput(
            message=f"NoOpAgent_v1 (ID: {self.agent_id}) executed successfully.",
            passthrough_data=task_input.passthrough_data
        )

    @classmethod
    def get_input_schema(cls) -> type[InputSchema]:
        return NoOpInput

    @classmethod
    def get_output_schema(cls) -> type[OutputSchema]:
        return NoOpOutput 