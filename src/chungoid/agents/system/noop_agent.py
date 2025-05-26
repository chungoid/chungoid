from typing import Any, Dict, Optional, ClassVar, List
import logging
from pydantic import BaseModel
import time

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
# REMOVED: BaseAgent import - no longer exists
# from chungoid.runtime.agents.agent_base import InputSchema, OutputSchema
from chungoid.schemas.orchestration import SharedContext
# Attempt to import providers for type hinting, but allow Any if it causes cycles during early init
try:
    from chungoid.utils.llm_provider import LLMProvider
    from chungoid.utils.prompt_manager import PromptManager
    # REMOVED: ProjectChromaManagerAgent_v1 import - replaced with MCP tools
    # from chungoid.agents.autonomous_engine.project_chroma_manager_agent import ProjectChromaManagerAgent_v1
except ImportError:
    LLMProvider = Any
    PromptManager = Any

# Define ProjectChromaManagerAgent_v1 as Any since it doesn't exist
ProjectChromaManagerAgent_v1 = Any

logger = logging.getLogger(__name__)

class NoOpInput(BaseModel):
    """Input schema for NoOpAgent_v1. Accepts any passthrough data."""
    passthrough_data: Optional[Dict[str, Any]] = None

class NoOpOutput(BaseModel):
    """Output schema for NoOpAgent_v1."""
    message: str
    passthrough_data: Optional[Dict[str, Any]] = None

from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.protocols.base.protocol_interface import ProtocolPhase

# Registry-first architecture import
from chungoid.registry import register_system_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
)

@register_system_agent(capabilities=["simple_operations", "status_reporting"])
class NoOpAgent_v1(UnifiedAgent):
    """
    A No-Operation Agent. It logs its invocation and returns a success message.
    It primarily serves as a placeholder in execution plans where an action is
    defined but no concrete operation needs to be performed by a specialized agent.
    """
    AGENT_ID: ClassVar[str] = "NoOpAgent_v1"
    AGENT_VERSION: ClassVar[str] = "1.0"
    CAPABILITIES: ClassVar[List[str]] = ["simple_operations", "status_reporting", "complex_analysis"]

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
        logger.info(f"NoOpAgent_v1 (ID: {self.get_id()}) initialized.")

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

    # ------------------------------------------------------------------
    # Legacy implementation preserved for Phase-2 consolidation ----------
    # ------------------------------------------------------------------
    async def _legacy_execute_impl(
        self,
        task_input: NoOpInput,
        full_context: Optional[SharedContext] = None,
    ) -> NoOpOutput:
        """Legacy implementation consolidated into unified execute method."""
        logger.info(
            f"NoOpAgent_v1 (ID: {self.get_id()}) invoked. Input: {task_input}. Context: {full_context}"
        )

        # Handle both Pydantic model and dictionary inputs
        if isinstance(task_input, dict):
            passthrough_data = task_input.get("passthrough_data")
        else:
            passthrough_data = task_input.passthrough_data

        return NoOpOutput(
            message=f"NoOpAgent_v1 (ID: {self.get_id()}) executed successfully.",
            passthrough_data=passthrough_data,
        )

    @classmethod
    def get_input_schema(cls) -> type:
        return NoOpInput

    @classmethod
    def get_output_schema(cls) -> type:
        return NoOpOutput

    # ------------------------------------------------------------------
    # Phase 3 UAEI implementation --------------------------------------
    # ------------------------------------------------------------------
    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """Phase 3 UAEI implementation - Core no-op logic for single iteration."""
        self.logger.info(f"[NoOp] Starting iteration {iteration + 1}")
        
        try:
            # Convert inputs to proper type
            if isinstance(context.inputs, dict):
                task_input = NoOpInput(**context.inputs)
            elif isinstance(context.inputs, NoOpInput):
                task_input = context.inputs
            else:
                # Fallback for other types
                task_input = NoOpInput(
                    passthrough_data=getattr(context.inputs, 'passthrough_data', None)
                )

            # Execute no-op operation
            output = await self._legacy_execute_impl(task_input, context.shared_context)
            
            quality_score = 1.0
            tools_used = []
            
        except Exception as e:
            self.logger.error(f"NoOp execution failed: {str(e)}")
            
            # Create error output
            output = NoOpOutput(
                message=f"NoOp execution failed: {str(e)}",
                passthrough_data=None
            )
            
            quality_score = 0.1
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used="simple_operations"
        ) 