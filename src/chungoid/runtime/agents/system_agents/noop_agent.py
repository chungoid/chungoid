from typing import Any, Dict, Optional, ClassVar
import logging
from pydantic import BaseModel

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
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

class NoOpAgent_v1(BaseAgent[NoOpInput, NoOpOutput]):
    """
    A No-Operation Agent. It logs its invocation and returns a success message.
    It primarily serves as a placeholder in execution plans where an action is
    defined but no concrete operation needs to be performed by a specialized agent.
    """
    AGENT_ID: ClassVar[str] = "NoOpAgent_v1"
    AGENT_VERSION: ClassVar[str] = "1.0"

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