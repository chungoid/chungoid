from __future__ import annotations

import logging
from typing import Any, Dict, Optional, ClassVar

from pydantic import BaseModel, Field, ValidationError

from chungoid.runtime.agents.agent_base import BaseAgent, InputSchema, OutputSchema
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
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

class SystemRequirementsGatheringAgent_v1(BaseAgent[SystemRequirementsGatheringInput, SystemRequirementsGatheringOutput]):
    AGENT_ID: ClassVar[str] = "SystemRequirementsGatheringAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Requirements Gathering Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Gathers and refines system requirements based on an initial user goal or problem statement."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.INTERNAL # Or PUBLIC if it can be invoked directly

    # Declare fields for dependencies injected in __init__
    llm_provider: LLMProvider
    prompt_manager: PromptManager

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, system_context: Optional[Dict[str, Any]] = None):
        # Pass llm_provider and prompt_manager to super().__init__ for Pydantic validation
        super().__init__(
            system_context=system_context, 
            llm_provider=llm_provider, 
            prompt_manager=prompt_manager
        )
        if not llm_provider: # These checks are now somewhat redundant if Pydantic validates them as required
            raise ValueError("LLMProvider is required for SystemRequirementsGatheringAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for SystemRequirementsGatheringAgent_v1")
        # self.llm_provider and self.prompt_manager are now set by Pydantic during super().__init__
        # self._logger_instance is available from BaseAgent

        # Example: Load a specific prompt for this agent
        # self.prompt_template = self.prompt_manager.get_prompt_template("system_requirements_gathering.md")

    async def invoke_async(
        self, inputs: Dict[str, Any], full_context: Optional[Dict[str, Any]] = None
    ) -> SystemRequirementsGatheringOutput:
        try:
            parsed_inputs = SystemRequirementsGatheringInput(**inputs)
        except ValidationError as e:
            logger.error(f"Input validation error for {self.AGENT_ID} ({self.AGENT_NAME}): {{e}}", exc_info=True)
            raise # Re-raise for the orchestrator to handle

        logger.info(f"{self.AGENT_NAME} ({self.AGENT_ID}) invoked with goal: {{parsed_inputs.user_goal}}")

        # 1. Render prompt using inputs and prompt_template
        # rendered_prompt = self.prompt_template.render(
        #     user_goal=parsed_inputs.user_goal,
        #     project_context_summary=parsed_inputs.project_context_summary
        # )

        # 2. Call LLM
        # llm_response_str = await self.llm_provider.generate_text(
        #     system_prompt="You are an expert requirements analyst.", # Or from prompt_manager
        #     user_prompt=rendered_prompt,
        #     # model="gpt-4-turbo", # Or from config
        #     temperature=0.5
        # )

        # 3. Parse LLM response (e.g., if it's structured JSON or Markdown)
        # For now, a placeholder response:
        requirements_summary_placeholder = f"Refined requirements for goal: '{{parsed_inputs.user_goal}}'. Context: {{parsed_inputs.project_context_summary or 'N/A'}}."
        
        # 4. Optionally store detailed requirements as an artifact (e.g., in ProjectChromaManagerAgent)
        # refined_requirements_document_id = await self._store_requirements_artifact(...)

        logger.info(f"Requirements summary generated: {{requirements_summary_placeholder[:100]}}...")
        
        return SystemRequirementsGatheringOutput(
            # refined_requirements_document_id=refined_requirements_document_id,
            requirements_summary=requirements_summary_placeholder
        )

    @classmethod
    def get_agent_card_static(cls) -> AgentCard:
        return AgentCard(
            agent_id=cls.AGENT_ID,
            name=cls.AGENT_NAME,
            version=cls.VERSION,
            description=cls.DESCRIPTION,
            category=cls.CATEGORY.value,
            visibility=cls.VISIBILITY.value,
            input_schema=SystemRequirementsGatheringInput.model_json_schema(),
            output_schema=SystemRequirementsGatheringOutput.model_json_schema(),
            # Add capability_profile, tags, etc. as needed
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