from __future__ import annotations

import logging
import json
import uuid
import asyncio
from typing import Any, Dict, Optional, List, Type, ClassVar
from pydantic import BaseModel, Field
from datetime import datetime, timezone
from pathlib import Path

from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent
from chungoid.protocols.base.protocol_interface import ProtocolPhase
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager, PromptRenderError
from chungoid.schemas.chroma_agent_io_schemas import StoreArtifactInput, StoreArtifactOutput, RetrieveArtifactOutput
from chungoid.registry import register_system_agent
GENERATED_CODE_ARTIFACTS_COLLECTION = "generated_code_artifacts"
PROJECT_DOCUMENTATION_ARTIFACTS_COLLECTION = "project_documentation_artifacts"
LIVE_CODEBASE_COLLECTION = "live_codebase"

logger = logging.getLogger(__name__)

PROMPT_ID = "smart_code_generator_v1"
PROMPT_VERSION = "0.2.0"
PROMPT_SUB_DIR = "autonomous_engine"

# Define compatible schemas that match orchestrator expectations
class SmartCodeGeneratorAgentInput(BaseModel):
    task_id: str = Field(..., description="Unique ID for this code generation task.")
    target_file_path: str = Field(..., description="Intended relative path of the file to be created or modified.")
    code_specification: str = Field(..., description="Detailed specification for the code to be generated.")
    programming_language: Optional[str] = Field(None, description="Target programming language (auto-detected from file extension if not provided).")
    existing_code_context: Optional[str] = Field(None, description="Existing code context if modifying a file.")
    requirements: Optional[List[str]] = Field(None, description="List of requirements for the code generation.")

class SmartCodeGeneratorAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    target_file_path: str = Field(..., description="The intended relative path (mirrors input).")
    status: str = Field(..., description="Status of the code generation attempt.")
    generated_code_content: str = Field(..., description="The generated code content.")
    generated_code_doc_id: Optional[str] = Field(None, description="Document ID for traceability.")
    confidence_score: ConfidenceScore = Field(..., description="Confidence in the generated code.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the generation process.")

@register_system_agent(capabilities=["systematic_implementation", "code_generation", "quality_validation"])
class CoreCodeGeneratorAgent_v1(ProtocolAwareAgent):
    AGENT_ID: ClassVar[str] = "SmartCodeGeneratorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Smart Code Generator Agent"
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    CAPABILITIES: ClassVar[List[str]] = ["systematic_implementation", "code_generation", "quality_validation"]
    DESCRIPTION: ClassVar[str] = "Autonomously generates high-quality code based on specifications using LLM-driven systematic implementation."
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.CODE_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorAgentInput]] = SmartCodeGeneratorAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorAgentOutput]] = SmartCodeGeneratorAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["code_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["tool_validation", "plan_review"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ["agent_communication", "tool_validation", "error_recovery"]

    _llm_provider: LLMProvider
    _prompt_manager: PromptManager
    _logger: logging.Logger

    def __init__(self, 
                 llm_provider: LLMProvider, 
                 prompt_manager: PromptManager, 
                 system_context: Optional[Dict[str, Any]] = None,
                 config: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        if not llm_provider:
            raise ValueError("LLMProvider is required for SmartCodeGeneratorAgent_v1")
        if not prompt_manager:
            raise ValueError("PromptManager is required for SmartCodeGeneratorAgent_v1")

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
        
        self._logger.info(f"{self.AGENT_ID} (v{self.VERSION}) initialized for autonomous code generation.")

    async def _execute_phase_logic(self, phase: ProtocolPhase, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute protocol phase logic for code generation."""
        if phase.name == "analysis" or phase == ProtocolPhase.ANALYSIS:
            # Analyze code requirements and specifications
            specification = context.get("code_specification", "")
            target_file = context.get("target_file_path", "")
            analysis = await self._analyze_code_requirements(specification, target_file)
            return {"analysis": analysis, "phase": "analysis"}
        elif phase.name == "implementation" or phase == ProtocolPhase.IMPLEMENTATION:
            # Generate the actual code
            specification = context.get("code_specification", "")
            target_file = context.get("target_file_path", "")
            code_result = await self._generate_code_implementation(specification, target_file, context)
            return {"code_result": code_result, "phase": "implementation"}
        elif phase.name == "validation" or phase == ProtocolPhase.VALIDATION:
            # Validate the generated code
            code = context.get("generated_code", "")
            validation = await self._validate_generated_code(code, context)
            return {"validation": validation, "phase": "validation"}
        else:
            # Default phase handling
            phase_name = phase.name if hasattr(phase, 'name') else str(phase)
            return {"phase": phase_name, "status": "completed"}

    async def _analyze_code_requirements(self, specification: str, target_file: str) -> Dict[str, Any]:
        """Analyze code requirements and specifications."""
        try:
            language = self._detect_language_from_path(target_file)
            
            analysis = {
                "programming_language": language,
                "target_file": target_file,
                "specification_complexity": "medium",
                "estimated_lines": 50,
                "required_imports": [],
                "key_functions": []
            }
            
            # Basic analysis based on specification
            if "class" in specification.lower():
                analysis["code_type"] = "class_definition"
            elif "function" in specification.lower():
                analysis["code_type"] = "function_definition"
            else:
                analysis["code_type"] = "script"
            
            # Estimate complexity
            if len(specification) > 500:
                analysis["specification_complexity"] = "high"
                analysis["estimated_lines"] = 100
            elif len(specification) < 100:
                analysis["specification_complexity"] = "low"
                analysis["estimated_lines"] = 20
            
            return analysis
            
        except Exception as e:
            self._logger.error(f"Error in code requirements analysis: {e}")
            return {
                "programming_language": "python",
                "target_file": target_file,
                "specification_complexity": "unknown",
                "estimated_lines": 50,
                "error": str(e)
            }

    async def _generate_code_implementation(self, specification: str, target_file: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code implementation based on specification."""
        try:
            generation_context = {
                "code_specification": specification,
                "target_file_path": target_file,
                "programming_language": self._detect_language_from_path(target_file),
                "existing_code_context": context.get("existing_code_context"),
                "requirements": context.get("requirements", [])
            }
            
            # Use the existing LLM generation method
            result = await self._generate_code_with_llm(generation_context)
            
            return {
                "generated_code": result["generated_code"],
                "approach": result.get("approach", "standard"),
                "execution_time": result.get("execution_time", 0),
                "success": True
            }
            
        except Exception as e:
            self._logger.error(f"Error in code implementation: {e}")
            # Generate fallback code
            fallback_code = self._generate_fallback_code_for_spec(specification, target_file)
            return {
                "generated_code": fallback_code,
                "approach": "fallback",
                "success": False,
                "error": str(e)
            }

    def _generate_fallback_code_for_spec(self, specification: str, target_file: str) -> str:
        """Generate basic fallback code based on specification."""
        language = self._detect_language_from_path(target_file)
        
        if language == "python":
            return f'''#!/usr/bin/env python3
"""
{specification}
"""

def main():
    """Main function implementing the specification."""
    print("Hello, World!")
    # TODO: Implement the actual functionality
    pass

if __name__ == "__main__":
    main()
'''
        elif language == "javascript":
            return f'''/**
 * {specification}
 */

function main() {{
    console.log("Hello, World!");
    // TODO: Implement the actual functionality
}}

main();
'''
        else:
            return f'''/*
 * {specification}
 */

// TODO: Implement the actual functionality
'''

    async def _generate_code_with_llm(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate code based on specifications."""
        
        # Create comprehensive code generation prompt
        generation_prompt = f"""
You are an expert software developer. Generate high-quality, production-ready code based on the following specification.

TASK DETAILS:
- Task ID: {context['task_id']}
- Target File: {context['target_file_path']}
- Programming Language: {context['programming_language']}

CODE SPECIFICATION:
{context['code_specification']}

REQUIREMENTS:
{chr(10).join(f"- {req}" for req in context['requirements']) if context['requirements'] else "No specific requirements provided"}

EXISTING CODE CONTEXT:
{context['existing_code_context'] or "No existing code context provided"}

Please generate complete, functional code that:
1. Implements the specified functionality correctly
2. Follows best practices for {context['programming_language']}
3. Includes appropriate error handling
4. Is well-documented with comments
5. Is production-ready and maintainable

Return your response in JSON format:
{{
    "generated_code": "The complete code implementation",
    "approach": "Brief description of the implementation approach",
    "key_features": ["List of key features implemented"],
    "dependencies": ["List of any external dependencies required"],
    "usage_notes": "Brief notes on how to use the generated code",
    "confidence_assessment": "high|medium|low"
}}

Focus on correctness, clarity, and maintainability. Generate complete, executable code.
"""

        try:
            # Call LLM with structured output expectation
            llm_response = await self._llm_provider.generate(
                prompt=generation_prompt,
                temperature=0.2,  # Lower temperature for more consistent code generation
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            # Parse LLM response
            code_data = json.loads(llm_response)
            
            self._logger.debug(f"LLM code generation completed for {context['programming_language']} code")
            
            return {
                "generated_code": code_data.get("generated_code", ""),
                "approach": code_data.get("approach", "standard"),
                "key_features": code_data.get("key_features", []),
                "dependencies": code_data.get("dependencies", []),
                "usage_notes": code_data.get("usage_notes", ""),
                "confidence_assessment": code_data.get("confidence_assessment", "medium"),
                "execution_time": 0  # Could be measured if needed
            }
            
        except Exception as e:
            self._logger.warning(f"LLM code generation failed, using fallback: {e}")
            return self._generate_fallback_code_dict(context)

    async def _validate_generated_code(self, code: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated code for basic correctness."""
        validation_results = {
            "passed": True,
            "issues": [],
            "quality_score": 1.0,
            "metrics": {}
        }
        
        try:
            # Basic validation checks
            if not code or len(code.strip()) == 0:
                validation_results["passed"] = False
                validation_results["issues"].append("Generated code is empty")
                validation_results["quality_score"] = 0.0
                return validation_results
            
            # Language-specific validation
            language = context.get("programming_language", "").lower()
            
            if language == "python":
                validation_results.update(self._validate_python_code(code))
            elif language in ["javascript", "js"]:
                validation_results.update(self._validate_javascript_code(code))
            else:
                # Generic validation for other languages
                validation_results.update(self._validate_generic_code(code))
            
            # Calculate overall metrics
            validation_results["metrics"] = {
                "lines_of_code": len(code.split('\n')),
                "non_empty_lines": len([line for line in code.split('\n') if line.strip()]),
                "comment_lines": len([line for line in code.split('\n') if line.strip().startswith('#') or line.strip().startswith('//')]),
                "estimated_complexity": "medium"  # Could be enhanced with actual complexity analysis
            }
            
            self._logger.debug(f"Code validation completed: {validation_results['passed']}, quality: {validation_results['quality_score']}")
            
        except Exception as e:
            self._logger.warning(f"Code validation failed: {e}")
            validation_results["passed"] = False
            validation_results["issues"].append(f"Validation error: {e}")
            validation_results["quality_score"] = 0.5
        
        return validation_results

    def _validate_python_code(self, code: str) -> Dict[str, Any]:
        """Validate Python-specific code."""
        issues = []
        quality_score = 1.0
        
        try:
            # Try to compile the code
            compile(code, '<string>', 'exec')
        except SyntaxError as e:
            issues.append(f"Python syntax error: {e}")
            quality_score -= 0.5
        except Exception as e:
            issues.append(f"Python compilation error: {e}")
            quality_score -= 0.3
        
        # Check for basic Python best practices
        if 'import' not in code and 'def' in code:
            # Might need imports
            quality_score -= 0.1
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0.0, quality_score)
        }

    def _validate_javascript_code(self, code: str) -> Dict[str, Any]:
        """Validate JavaScript-specific code."""
        issues = []
        quality_score = 1.0
        
        # Basic JavaScript validation (could be enhanced with actual JS parser)
        if code.count('{') != code.count('}'):
            issues.append("Mismatched curly braces")
            quality_score -= 0.3
        
        if code.count('(') != code.count(')'):
            issues.append("Mismatched parentheses")
            quality_score -= 0.3
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0.0, quality_score)
        }

    def _validate_generic_code(self, code: str) -> Dict[str, Any]:
        """Generic code validation for unknown languages."""
        issues = []
        quality_score = 0.8  # Lower baseline for unknown languages
        
        # Basic structural checks
        if len(code.strip()) < 10:
            issues.append("Code appears too short to be meaningful")
            quality_score -= 0.3
        
        return {
            "passed": len(issues) == 0,
            "issues": issues,
            "quality_score": max(0.0, quality_score)
        }

    def _calculate_confidence_score(self, generation_result: Dict[str, Any], 
                                  validation_result: Dict[str, Any], 
                                  context: Dict[str, Any]) -> ConfidenceScore:
        """Calculate confidence score based on generation and validation results."""
        
        base_score = 0.5
        
        # Factor in LLM confidence assessment
        llm_confidence = generation_result.get("confidence_assessment", "medium")
        if llm_confidence == "high":
            base_score += 0.3
        elif llm_confidence == "medium":
            base_score += 0.1
        # low confidence adds nothing
        
        # Factor in validation results
        if validation_result["passed"]:
            base_score += 0.2
        
        base_score += validation_result["quality_score"] * 0.2
        
        # Factor in code completeness
        code_length = len(generation_result.get("generated_code", ""))
        if code_length > 100:  # Reasonable code length
            base_score += 0.1
        
        # Ensure score is within bounds
        final_score = max(0.0, min(1.0, base_score))
        
        # Determine reasoning
        reasoning_parts = []
        if validation_result["passed"]:
            reasoning_parts.append("code passes validation")
        if llm_confidence == "high":
            reasoning_parts.append("high LLM confidence")
        if validation_result["quality_score"] > 0.8:
            reasoning_parts.append("good code quality metrics")
        
        reasoning = "Code generated with " + ", ".join(reasoning_parts) if reasoning_parts else "Basic code generation completed"
        
        return ConfidenceScore(
            value=final_score,
            explanation=reasoning
        )

    def _detect_language_from_path(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        path = Path(file_path)
        extension = path.suffix.lower()
        
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        
        return language_map.get(extension, 'unknown')

    def _generate_fallback_code_dict(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate fallback code structure when LLM fails."""
        language = context.get("programming_language", "python")
        
        if language.lower() == "python":
            fallback_code = f'''#!/usr/bin/env python3
"""
{context.get("code_specification", "Generated code")}

This is a basic template generated as fallback.
Please implement the actual functionality.
"""

def main():
    """Main function - implement your logic here."""
    print("Hello, World!")
    # TODO: Implement {context.get("code_specification", "the required functionality")}

if __name__ == "__main__":
    main()
'''
        else:
            fallback_code = f'''// {context.get("code_specification", "Generated code")}
// This is a basic template generated as fallback.
// Please implement the actual functionality.

console.log("Hello, World!");
// TODO: Implement {context.get("code_specification", "the required functionality")}
'''
        
        return {
            "generated_code": fallback_code,
            "approach": "fallback_template",
            "key_features": ["basic_structure", "placeholder_implementation"],
            "dependencies": [],
            "usage_notes": "This is a fallback template that needs implementation",
            "confidence_assessment": "low"
        }

    def _generate_fallback_code(self, task_input: SmartCodeGeneratorAgentInput) -> str:
        """Generate basic fallback code when everything fails."""
        context = {
            "code_specification": task_input.code_specification,
            "programming_language": task_input.programming_language or self._detect_language_from_path(task_input.target_file_path)
        }
        
        return self._generate_fallback_code_dict(context)["generated_code"]

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = SmartCodeGeneratorAgentInput.model_json_schema()
        output_schema = SmartCodeGeneratorAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=CoreCodeGeneratorAgent_v1.AGENT_ID,
            name=CoreCodeGeneratorAgent_v1.AGENT_NAME,
            version=CoreCodeGeneratorAgent_v1.VERSION,
            description=CoreCodeGeneratorAgent_v1.DESCRIPTION,
            categories=[CoreCodeGeneratorAgent_v1.CATEGORY.value],
            visibility=CoreCodeGeneratorAgent_v1.VISIBILITY.value,
            input_schema=input_schema,
            output_schema=output_schema,
            capability_profile={
                "generates_code": True,
                "validates_code": True,
                "multi_language_support": True,
                "autonomous_operation": True,
                "llm_driven_generation": True,
                "quality_assessment": True
            },
            metadata={
                "autonomous": True,
                "requires_llm": True,
                "supported_languages": ["python", "javascript", "typescript", "java", "cpp", "c", "csharp", "go", "rust"],
                "output_format": "structured_code_with_metadata"
            }
        )

async def main_test_smart_code_gen():
    logging.basicConfig(level=logging.DEBUG)
    print("Test stub for SmartCodeGeneratorAgent_v1 needs full environment setup.")

if __name__ == "__main__":
    import asyncio
    print("To test SmartCodeGeneratorAgent_v1, please run through an integration test or a dedicated test script with mocked/real dependencies.") 