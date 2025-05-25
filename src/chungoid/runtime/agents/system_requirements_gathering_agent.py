from __future__ import annotations

import logging
import asyncio
import uuid
from typing import Any, Dict, Optional, ClassVar, Type, List
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
from chungoid.registry import register_system_agent

logger = logging.getLogger(__name__)

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

class SystemRequirementsGatheringInput(BaseModel):
    user_goal: str = Field(..., description="The high-level user goal.")
    project_context: Optional[str] = Field(None, description="Optional summary of the existing project context.")

class SystemRequirementsGatheringOutput(BaseModel):
    requirements_document_id: Optional[str] = Field(None, description="ID of the document artifact containing the refined requirements.")
    requirements_summary: str = Field(..., description="A textual summary of the gathered and refined requirements.")
    functional_requirements: List[str] = Field(default_factory=list, description="List of functional requirements extracted from the user goal.")
    technical_requirements: List[str] = Field(default_factory=list, description="List of technical requirements and constraints.")
    acceptance_criteria: List[str] = Field(default_factory=list, description="List of acceptance criteria for the project.")
    status: str = Field(default="SUCCESS", description="Status of the requirements gathering process.")

@register_system_agent(capabilities=["requirements_analysis", "stakeholder_analysis", "documentation"])
class SystemRequirementsGatheringAgent_v1(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "SystemRequirementsGatheringAgent_v1"
    AGENT_NAME: ClassVar[str] = "System Requirements Gathering Agent"
    VERSION: ClassVar[str] = "1.0.0"
    DESCRIPTION: ClassVar[str] = "Autonomously gathers and refines system requirements based on user goals using LLM-driven analysis."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.REQUIREMENTS_ANALYSIS
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[SystemRequirementsGatheringInput]] = SystemRequirementsGatheringInput
    OUTPUT_SCHEMA: ClassVar[Type[SystemRequirementsGatheringOutput]] = SystemRequirementsGatheringOutput
    
    # AUTONOMOUS: Protocol-aware configuration
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["requirements_analysis", "stakeholder_analysis"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["documentation", "validation"]

    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager, 
                 system_context: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        if not llm_provider:
            raise ValueError("LLMProvider is required for SystemRequirementsGatheringAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for SystemRequirementsGatheringAgent_v1")

        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            agent_id=agent_id or self.AGENT_ID,
            system_context=system_context,
            config=config,
            **kwargs
        )

        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        self._logger = logging.getLogger(self.AGENT_ID)
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized as autonomous protocol-aware agent.")

    async def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocol phase logic for requirements gathering."""
        if phase.name == "analysis" or phase == ProtocolPhase.ANALYSIS:
            # Perform requirements analysis
            user_goal = context.get("user_goal", "")
            requirements = await self._analyze_requirements(user_goal)
            return {"requirements": requirements, "phase": "analysis"}
        elif phase.name == "validation" or phase == ProtocolPhase.VALIDATION:
            # Validate requirements
            requirements = context.get("requirements", {})
            validation_result = self._validate_requirements(requirements)
            return {"validation": validation_result, "phase": "validation"}
        elif phase.name == "synthesis" or phase == ProtocolPhase.SYNTHESIS:
            # Synthesize final requirements
            requirements = context.get("requirements", {})
            validation = context.get("validation", {})
            final_requirements = self._synthesize_requirements(requirements, validation)
            return {"final_requirements": final_requirements, "phase": "synthesis"}
        else:
            # Default phase handling
            phase_name = phase.name if hasattr(phase, 'name') else str(phase)
            return {"phase": phase_name, "status": "completed"}

    async def _analyze_requirements(self, user_goal: str) -> Dict[str, Any]:
        """Analyze user goal to extract requirements."""
        try:
            if self.llm_provider:
                # Use LLM for sophisticated analysis
                prompt = f"""
                Analyze the following user goal and extract detailed requirements:
                
                Goal: {user_goal}
                
                Please provide:
                1. Functional requirements
                2. Technical requirements  
                3. Acceptance criteria
                4. Potential risks
                
                Format as JSON.
                """
                
                response = await self.llm_provider.generate(
                    prompt=prompt,
                    system_prompt="You are a requirements analyst. Provide structured analysis.",
                    response_format="json_object"
                )
                
                if response and response.content:
                    import json
                    try:
                        return json.loads(response.content)
                    except json.JSONDecodeError:
                        pass
            
            # Fallback to basic analysis
            return self._generate_fallback_requirements_dict(user_goal)
            
        except Exception as e:
            self.logger.error(f"Error in requirements analysis: {e}")
            return self._generate_fallback_requirements_dict(user_goal)

    def _validate_requirements(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Validate requirements for completeness and consistency."""
        validation = {
            "is_complete": True,
            "is_consistent": True,
            "issues": [],
            "confidence": 0.8
        }
        
        # Check for required fields
        required_fields = ["functional_requirements", "technical_requirements"]
        for field in required_fields:
            if field not in requirements or not requirements[field]:
                validation["is_complete"] = False
                validation["issues"].append(f"Missing {field}")
        
        # Check for consistency
        if "functional_requirements" in requirements and "technical_requirements" in requirements:
            func_reqs = requirements["functional_requirements"]
            tech_reqs = requirements["technical_requirements"]
            if isinstance(func_reqs, list) and isinstance(tech_reqs, list):
                if len(func_reqs) == 0 or len(tech_reqs) == 0:
                    validation["is_consistent"] = False
                    validation["issues"].append("Empty requirements lists")
        
        return validation

    def _synthesize_requirements(self, requirements: Dict[str, Any], validation: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize final requirements based on analysis and validation."""
        final_requirements = requirements.copy()
        
        # Add validation metadata
        final_requirements["validation_status"] = validation
        final_requirements["synthesis_timestamp"] = datetime.now(timezone.utc).isoformat()
        final_requirements["confidence_score"] = validation.get("confidence", 0.8)
        
        # Ensure all required fields are present
        if "acceptance_criteria" not in final_requirements:
            final_requirements["acceptance_criteria"] = [
                "System meets functional requirements",
                "System meets technical requirements",
                "System passes validation tests"
            ]
        
        return final_requirements

    def _generate_fallback_requirements_dict(self, user_goal: str) -> Dict[str, Any]:
        """Generate basic requirements when LLM analysis fails."""
        return {
            "functional_requirements": [
                f"Implement core functionality for: {user_goal}",
                "Provide user-friendly interface",
                "Handle basic error scenarios"
            ],
            "technical_requirements": [
                "Use appropriate programming language",
                "Follow coding best practices",
                "Include basic error handling"
            ],
            "complexity_assessment": "medium",
            "estimated_effort": "Basic implementation effort required",
            "key_technologies": ["python"],
            "potential_risks": ["Implementation complexity", "User requirements clarity"]
        }

    def _generate_requirements_summary(self, user_goal: str, functional_reqs: List[str], 
                                     technical_reqs: List[str], acceptance_criteria: List[str]) -> str:
        """Generate a comprehensive summary of gathered requirements."""
        
        summary_parts = [
            f"Requirements Analysis for: {user_goal}",
            "",
            "FUNCTIONAL REQUIREMENTS:",
        ]
        
        for i, req in enumerate(functional_reqs, 1):
            summary_parts.append(f"  {i}. {req}")
        
        summary_parts.extend([
            "",
            "TECHNICAL REQUIREMENTS:",
        ])
        
        for i, req in enumerate(technical_reqs, 1):
            summary_parts.append(f"  {i}. {req}")
        
        summary_parts.extend([
            "",
            "ACCEPTANCE CRITERIA:",
        ])
        
        for i, criteria in enumerate(acceptance_criteria, 1):
            summary_parts.append(f"  {i}. {criteria}")
        
        summary_parts.extend([
            "",
            f"Total Requirements: {len(functional_reqs)} functional, {len(technical_reqs)} technical",
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        ])
        
        return "\n".join(summary_parts)

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Return the agent card for this agent."""
        return AgentCard(
            agent_id="SystemRequirementsGatheringAgent_v1",
            name="System Requirements Gathering Agent",
            description="Autonomously gathers and refines system requirements using protocol-driven analysis",
            category=AgentCategory.REQUIREMENTS_ANALYSIS,
            version="1.0.0",
            capabilities=["requirements_analysis", "stakeholder_analysis", "documentation"],
            input_schema=SystemRequirementsGatheringInput,
            output_schema=SystemRequirementsGatheringOutput
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