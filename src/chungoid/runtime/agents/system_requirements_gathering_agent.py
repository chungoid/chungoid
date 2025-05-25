from __future__ import annotations

import logging
class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

import asyncio
import uuid
from typing import Any, Dict, Optional, ClassVar
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
from chungoid.runtime.agents.agent_base import BaseAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptDefinition
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility

logger = logging.getLogger(__name__)

class SystemRequirementsGatheringInput(BaseModel):
    user_goal: str = Field(..., description="The high-level user goal.")
    project_context_summary: Optional[str] = Field(None, description="Optional summary of the existing project context.")
    # Add other fields as necessary, e.g., existing requirements, constraints

class SystemRequirementsGatheringOutput(BaseModel):
    refined_requirements_document_id: Optional[str] = Field(None, description="ID of the document artifact containing the refined requirements (e.g., in ChromaDB).")
    requirements_summary: str = Field(..., description="A textual summary of the gathered and refined requirements.")
    # Add other fields, e.g., structured requirements data

class SystemRequirementsGatheringAgent_v1(ProtocolAwareAgent[SystemRequirementsGatheringInput, SystemRequirementsGatheringOutput]):
    AGENT_ID: ClassVar[str] = "SystemRequirementsGatheringAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Requirements Gathering Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Gathers and refines system requirements based on an initial user goal or problem statement."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Or PUBLIC if it can be invoked directly

    # Declare fields for dependencies injected in __init__
    llm_provider: LLMProvider
    prompt_manager: PromptManager
    loprd_generation_prompt_template_obj: Optional[PromptDefinition] = None
    loprd_generation_inline_fallback_prompt: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True
    
    # ADDED: Protocol definitions following AI agent best practices
    PRIMARY_PROTOCOLS: ClassVar[list[str]] = ['requirements_gathering', 'stakeholder_analysis']
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ['documentation', 'validation']
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'goal_tracking']

    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager, 
                 system_context: Optional[Dict[str, Any]] = None):
        # Pass llm_provider and prompt_manager to super().__init__
        init_kwargs = {
            "llm_provider": llm_provider,
            "prompt_manager": prompt_manager
        }
        if system_context is not None:
            init_kwargs["system_context"] = system_context
        
        super().__init__(**init_kwargs)
        # Pydantic now handles assignment of llm_provider, prompt_manager
        if not self.llm_provider: # Should be caught by Pydantic if not optional
            raise ValueError("LLMProvider is required for SystemRequirementsGatheringAgent_v1")
        if not self.prompt_manager: # Should be caught by Pydantic if not optional
            raise ValueError("PromptManager is required for SystemRequirementsGatheringAgent_v1")

        # Load the LOPRD generation prompt using PromptManager
        try:
            self.loprd_generation_prompt_template_obj = self.prompt_manager.get_prompt_definition(
                prompt_name="system_requirements_gathering_v1",
                prompt_version="1.0",
                sub_path="autonomous_engine"
            )
            self.loprd_generation_inline_fallback_prompt = None
        except FileNotFoundError:
            logger.error("LOPRD prompt file 'autonomous_engine/system_requirements_gathering_agent_v1_prompt.yaml' not found by PromptManager.")
            self.loprd_generation_prompt_template_obj = None 
            self.loprd_generation_inline_fallback_prompt = """
            You are an expert requirements analyst. Generate a JSON LOPRD for: {user_goal}. Context: {project_context_summary}.
            JSON should include: loprd_id, document_version, user_goal_received, execution_summary, detailed_requirements, acceptance_criteria, relevant_technologies, potential_risks.
            """
            logger.warning("Using a basic inline fallback prompt for LOPRD generation.")
        except Exception as e:
            logger.error(f"Error loading LOPRD prompt: {e}", exc_info=True)
            self.loprd_generation_prompt_template_obj = None
            self.loprd_generation_inline_fallback_prompt = "Error loading prompt. Goal: {user_goal}."
            logger.warning("Using error fallback prompt for LOPRD generation.")

    async def execute(self, task_input, full_context: Optional[Dict[str, Any]] = None):
        """
        Execute using pure protocol architecture.
        No fallback - protocol execution only for clean, maintainable code.
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
                # Enhanced error handling instead of fallback
                error_msg = f"Protocol execution failed for {self.AGENT_NAME}: {protocol_result.get('error', 'Unknown error')}"
                self._logger.error(error_msg)
                raise ProtocolExecutionError(error_msg)
                
        except Exception as e:
            error_msg = f"Pure protocol execution failed for {self.AGENT_NAME}: {e}"
            self._logger.error(error_msg)
            raise ProtocolExecutionError(error_msg)

    # ADDED: Protocol phase execution logic
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute agent-specific logic for each protocol phase."""
        
        # Generic phase handling - can be overridden by specific agents
        if phase.name in ["discovery", "analysis", "planning", "execution", "validation"]:
            return self._execute_generic_phase(phase)
        else:
            self._logger.warning(f"Unknown protocol phase: {phase.name}")
            return {"phase_completed": True, "method": "fallback"}

    def _execute_generic_phase(self, phase: ProtocolPhase) -> Dict[str, Any]:
        """Execute generic phase logic suitable for most agents."""
        return {
            "phase_name": phase.name,
            "status": "completed", 
            "outputs": {"generic_result": f"Phase {phase.name} completed"},
            "method": "generic_protocol_execution"
        }

    def _extract_output_from_protocol_result(self, protocol_result: Dict[str, Any], task_input) -> Any:
        """Extract agent output from protocol execution results."""
        # Generic extraction - should be overridden by specific agents
        return {
            "status": "SUCCESS",
            "message": "Task completed via protocol execution",
            "protocol_used": protocol_result.get("protocol_name"),
            "execution_time": protocol_result.get("execution_time", 0),
            "phases_completed": len([p for p in protocol_result.get("phases", []) if p.get("success", False)])
        }

    @classmethod
    def get_agent_card_static(cls) -> AgentCard:
        """Returns the static AgentCard for this agent."""
        return AgentCard(
            agent_id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            version=cls.VERSION,
            description=cls.DESCRIPTION,
            category=cls.CATEGORY,
            visibility=cls.VISIBILITY,
            input_schema=SystemRequirementsGatheringInput.model_json_schema(),
            output_schema=SystemRequirementsGatheringOutput.model_json_schema(),
            dependencies=["LLMProvider", "PromptManager"],
            init_args=["llm_provider", "prompt_manager"]
        )

# Example of how it might be used (for testing this file directly)
# async def main():
#     # Mock dependencies
#     class MockLLMProvider(LLMProvider):
#         async def generate_text(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
#             return f"LLM mock response for: {user_prompt}"
#         async def generate_json(self, system_prompt: str, user_prompt: str, **kwargs) -> Dict[str, Any]:
#             return {"summary": f"LLM mock JSON for: {user_prompt}"}

#     class MockPromptManager(PromptManager):
#         def __init__(self): super().__init__(Path(".")) # Dummy path
#         def get_prompt_template(self, template_name: str):
#             # return a mock template
#             class MockTemplate:
#                 def render(self, **kwargs) -> str: return f"Rendered prompt with {kwargs}"
#             return MockTemplate()

#     llm_provider = MockLLMProvider()
#     prompt_manager = MockPromptManager()
#     agent = SystemRequirementsGatheringAgent_v1(llm_provider=llm_provider, prompt_manager=prompt_manager)
    
#     test_input = SystemRequirementsGatheringInput(user_goal="Build a todo app.")
#     output = await agent.invoke_async(test_input)
#     print(f"Agent Output: {output.model_dump_json(indent=2)}")
#     print(f"Agent Card: {agent.get_agent_card_static().model_dump_json(indent=2)}")

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main()) 