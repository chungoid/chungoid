"""
SmartCodeGeneratorAgent_v1: Intelligent code generation with context awareness.

This agent generates source code files based on project specifications, blueprints,
and requirements using LLM processing abilities. It follows the unified smart agent
architecture with intelligent context support.

Key Features:
- Intelligent context processing from orchestrator
- Multi-language code generation support
- Project structure awareness
- Quality-driven iterative generation
- MCP tool integration for file operations
"""

from __future__ import annotations

import logging
import datetime
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.utils.chromadb_migration_utils import migrate_store_artifact
from chungoid.registry import register_autonomous_engine_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    AgentExecutionResult,
    ExecutionMetadata,
    ExecutionMode,
    CompletionReason,
    IterationResult,
    StageInfo,
)

logger = logging.getLogger(__name__)

class SmartCodeGeneratorInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this code generation task.")
    project_id: str = Field(..., description="Project ID for context.")
    
    # Traditional fields - optional when using intelligent context
    task_description: Optional[str] = Field(None, description="Core description of the code to be generated.")
    target_file_path: Optional[str] = Field(None, description="Intended relative path of the file to be created.")
    programming_language: Optional[str] = Field(None, description="Target programming language.")
    
    # Intelligent context fields
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")
    target_languages: Optional[List[str]] = Field(None, description="Target programming languages")
    technologies: Optional[List[str]] = Field(None, description="Project technologies")
    
    @model_validator(mode='after')
    def check_intelligent_context_requirements(self) -> 'SmartCodeGeneratorInput':
        """Validate requirements based on execution mode (intelligent context vs traditional)."""
        
        if self.intelligent_context:
            # Intelligent context mode - requires project specifications and user goal
            if not self.project_specifications:
                raise ValueError("project_specifications is required when intelligent_context=True")
            if not self.user_goal:
                raise ValueError("user_goal is required when intelligent_context=True")
        else:
            # Traditional mode - requires task description and target file path
            if not self.task_description:
                raise ValueError("task_description is required when intelligent_context=False")
            if not self.target_file_path:
                raise ValueError("target_file_path is required when intelligent_context=False")
            if not self.programming_language:
                raise ValueError("programming_language is required when intelligent_context=False")
        
        return self

class SmartCodeGeneratorOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    status: str = Field(..., description="Status of code generation (SUCCESS, FAILURE, etc.).")
    generated_files: List[Dict[str, Any]] = Field(default_factory=list, description="List of generated code files with metadata.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the generated code.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")

@register_autonomous_engine_agent(capabilities=["code_generation", "systematic_implementation", "quality_validation"])
class SmartCodeGeneratorAgent_v1(UnifiedAgent):
    """
    Intelligent code generation agent with context awareness.
    
    Generates source code files based on project specifications, blueprints,
    and requirements using LLM processing abilities. Follows the unified smart agent
    architecture with intelligent context support.
    
    ✨ PURE UAEI ARCHITECTURE - Clean execution paths with unified interface.
    ✨ INTELLIGENT CONTEXT SUPPORT - Uses orchestrator's intelligent project analysis.
    """
    
    AGENT_ID: ClassVar[str] = "SmartCodeGeneratorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Smart Code Generator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Generates source code files with intelligent context awareness and quality validation."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "smart_code_generator_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "1.0.0"
    CAPABILITIES: ClassVar[List[str]] = ["code_generation", "systematic_implementation", "quality_validation", "complex_analysis"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.AUTONOMOUS_COORDINATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorInput]] = SmartCodeGeneratorInput
    OUTPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorOutput]] = SmartCodeGeneratorOutput

    # Protocol definitions following smart agent best practices
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["code_generation", "systematic_implementation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["quality_validation", "artifact_management"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'error_recovery']

    def __init__(
        self,
        llm_provider: LLMProvider,
        prompt_manager: PromptManager,
        **kwargs
    ):
        # Enable refinement capabilities for intelligent code generation
        super().__init__(
            llm_provider=llm_provider, 
            prompt_manager=prompt_manager,
            enable_refinement=True,  # Enable intelligent refinement
            **kwargs
        )

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Phase 3 UAEI implementation for smart code generation.
        Enhanced with Phase 4 refinement capabilities.
        
        Runs comprehensive code generation workflow: analysis → planning → generation → validation
        With refinement: previous work analysis → current state analysis → intelligent refinement
        """
        try:
            # Convert inputs to expected format
            if isinstance(context.inputs, dict):
                task_input = SmartCodeGeneratorInput(**context.inputs)
            elif hasattr(context.inputs, 'dict'):
                inputs = context.inputs.dict()
                task_input = SmartCodeGeneratorInput(**inputs)
            else:
                task_input = context.inputs

            # Phase 4: Check for refinement context
            refinement_context = context.shared_context.get("refinement_context")
            if self.enable_refinement and refinement_context:
                self.logger.info(f"[Refinement] Using refinement context with {len(refinement_context.get('previous_outputs', []))} previous outputs")
                # Use refinement-aware analysis that considers previous work
                analysis_result = await self._analyze_with_refinement_context(
                    task_input, context.shared_context, refinement_context
                )
            elif task_input.intelligent_context and task_input.project_specifications:
                self.logger.info("Using intelligent project specifications from orchestrator")
                analysis_result = await self._extract_analysis_from_intelligent_specs(
                    task_input.project_specifications, 
                    task_input.user_goal
                )
            else:
                self.logger.info("Using traditional project analysis")
                analysis_result = await self._analyze_project_requirements(task_input, context.shared_context)
            
            # Phase 2: Planning - Plan code structure and files based on LLM analysis
            planning_result = await self._plan_code_structure(analysis_result, task_input, context.shared_context)
            
            # Phase 3: Generation - Generate code files
            generation_result = await self._generate_code_files(planning_result, task_input, context.shared_context)
            
            # Phase 4: Validation - Validate generated code quality
            validation_result = await self._validate_generated_code(generation_result, task_input, context.shared_context)
            
            # Inject refinement context for iteration-aware quality scoring
            if self.enable_refinement and refinement_context:
                self._current_refinement_context = refinement_context
            
            # Calculate quality score based on validation results
            quality_score = self._calculate_quality_score(validation_result)
            
            # Create output
            output = SmartCodeGeneratorOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                generated_files=generation_result.get("generated_files", []),
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="comprehensive_validation",
                    explanation="Based on code quality analysis and validation checks"
                )
            )
            
            tools_used = ["project_analysis", "code_planning", "code_generation", "quality_validation"]
            
            return IterationResult(
                output=output,
                quality_score=quality_score,
                tools_used=tools_used,
                protocol_used="smart_code_generation_protocol"
            )
            
        except Exception as e:
            self.logger.error(f"Code generation iteration failed: {e}")
            
            # Create error output
            error_output = SmartCodeGeneratorOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())),
                project_id=getattr(task_input, 'project_id', None) or 'intelligent_project',
                status="ERROR",
                generated_files=[],
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="smart_code_generation_protocol"
            )

    async def _extract_analysis_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract analysis from intelligent project specifications using LLM processing."""
        
        try:
            if self.llm_provider:
                # Use LLM to intelligently analyze the project specifications and plan code generation strategy
                prompt = f"""
                You are a smart code generator agent. Analyze the following project specifications and user goal to create an intelligent code generation strategy.
                
                User Goal: {user_goal}
                
                Project Specifications:
                - Project Type: {project_specs.get('project_type', 'unknown')}
                - Primary Language: {project_specs.get('primary_language', 'unknown')}
                - Target Languages: {project_specs.get('target_languages', [])}
                - Target Platforms: {project_specs.get('target_platforms', [])}
                - Technologies: {project_specs.get('technologies', [])}
                - Required Dependencies: {project_specs.get('required_dependencies', [])}
                - Optional Dependencies: {project_specs.get('optional_dependencies', [])}
                
                Provide a detailed JSON analysis with the following structure:
                {{
                    "project_type": "...",
                    "primary_language": "...",
                    "target_languages": [...],
                    "technologies": [...],
                    "required_dependencies": [...],
                    "optional_dependencies": [...],
                    "code_generation_strategy": {{
                        "approach": "...",
                        "file_structure": [...],
                        "implementation_priorities": [...],
                        "architectural_patterns": [...]
                    }},
                    "complexity_assessment": {{
                        "level": "low|medium|high",
                        "factors": [...],
                        "estimated_files": 0,
                        "estimated_lines": 0
                    }},
                    "quality_requirements": {{
                        "testing_strategy": "...",
                        "documentation_level": "...",
                        "code_standards": [...]
                    }},
                    "implementation_considerations": [...],
                    "potential_challenges": [...],
                    "confidence_score": 0.0-1.0,
                    "reasoning": "..."
                }}
                """
                
                response = await self.llm_provider.generate(prompt)
                
                if response:
                    try:
                        # Extract JSON from markdown code blocks if present
                        json_content = self._extract_json_from_response(response)
                        parsed_result = json.loads(json_content)
                        
                        # Validate that we got a dictionary as expected
                        if not isinstance(parsed_result, dict):
                            self.logger.warning(f"Expected dict from code generation analysis, got {type(parsed_result)}. Using fallback.")
                            return self._generate_fallback_code_analysis(project_specs, user_goal)
                        
                        # Add metadata about the intelligent analysis
                        parsed_result["intelligent_analysis"] = True
                        parsed_result["project_specifications"] = project_specs
                        parsed_result["analysis_method"] = "llm_intelligent_processing"
                        parsed_result["code_generation_needed"] = True
                        return parsed_result
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse LLM response as JSON: {e}")
            
            # Fallback to basic extraction if LLM fails
            self.logger.info("Using fallback analysis due to LLM unavailability")
            return self._generate_fallback_code_analysis(project_specs, user_goal)
            
        except Exception as e:
            self.logger.error(f"Error in intelligent specs analysis: {e}")
            return self._generate_fallback_code_analysis(project_specs, user_goal)

    def _generate_fallback_code_analysis(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Generate fallback code analysis when LLM is unavailable."""
        
        analysis = {
            "project_type": project_specs.get("project_type", "cli_tool"),
            "primary_language": project_specs.get("primary_language", "python"),
            "target_languages": project_specs.get("target_languages", ["python"]),
            "technologies": project_specs.get("technologies", []),
            "required_dependencies": project_specs.get("required_dependencies", []),
            "optional_dependencies": project_specs.get("optional_dependencies", []),
            "code_generation_strategy": {
                "approach": "template_based",
                "file_structure": ["main module", "dependencies", "documentation"],
                "implementation_priorities": ["core functionality", "error handling", "documentation"],
                "architectural_patterns": ["modular design"]
            },
            "complexity_assessment": {
                "level": "medium",
                "factors": ["project type", "technology stack"],
                "estimated_files": 3,
                "estimated_lines": 200
            },
            "quality_requirements": {
                "testing_strategy": "basic",
                "documentation_level": "standard",
                "code_standards": ["PEP8"] if project_specs.get("primary_language") == "python" else ["standard"]
            },
            "implementation_considerations": ["dependency management", "error handling"],
            "potential_challenges": ["integration complexity", "performance optimization"],
            "intelligent_analysis": True,
            "analysis_method": "fallback_extraction",
            "code_generation_needed": True
        }
        
        return analysis

    async def _analyze_with_refinement_context(
        self, 
        task_input: SmartCodeGeneratorInput, 
        shared_context: Dict[str, Any], 
        refinement_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze project requirements with refinement context from previous iterations."""
        
        try:
            previous_outputs = refinement_context.get("previous_outputs", [])
            current_state = refinement_context.get("current_state", {})
            iteration = refinement_context.get("iteration", 0)
            previous_quality = refinement_context.get("previous_quality_score", 0.0)
            
            # Build refinement-aware prompt using the base class method
            refinement_prompt = self._build_refinement_prompt(task_input, refinement_context)
            
            if self.llm_provider:
                # Enhanced prompt for code generation refinement
                enhanced_prompt = f"""
                {refinement_prompt}
                
                SPECIFIC CODE GENERATION REFINEMENT INSTRUCTIONS:
                
                Previous Code Generation Analysis:
                """
                
                # Add analysis of previous code generation attempts
                for output in previous_outputs[-2:]:  # Last 2 outputs
                    enhanced_prompt += f"""
                    
                Iteration {output['iteration']}:
                - Quality Score: {output['quality_score']:.2f}
                - Generated Files: {len(output.get('metadata', {}).get('generated_files', []))} files
                - Status: {output.get('metadata', {}).get('status', 'unknown')}
                """
                
                # Add current project state analysis
                if current_state.get('code_analysis'):
                    enhanced_prompt += f"""
                    
                Current Code State:
                - Existing code analysis: {current_state['code_analysis']}
                """
                
                if current_state.get('file_structure'):
                    enhanced_prompt += f"""
                - File structure: {current_state['file_structure']}
                """
                
                enhanced_prompt += f"""
                
                REFINEMENT GOALS:
                1. Improve upon previous code generation quality (current: {previous_quality:.2f})
                2. Address any issues identified in previous iterations
                3. Ensure better integration with existing project structure
                4. Generate more robust and maintainable code
                5. Improve error handling and documentation
                
                Provide a refined analysis in the same JSON format as before, but with improvements based on the refinement context.
                """
                
                response = await self.llm_provider.generate(enhanced_prompt)
                
                if response:
                    try:
                        json_content = self._extract_json_from_response(response)
                        parsed_result = json.loads(json_content)
                        
                        # Validate that we got a dictionary as expected
                        if not isinstance(parsed_result, dict):
                            self.logger.warning(f"Expected dict from refinement analysis, got {type(parsed_result)}. Using fallback.")
                            return self._generate_fallback_refinement_analysis(task_input, previous_outputs, current_state)
                        
                        # Add refinement metadata
                        parsed_result["refinement_analysis"] = True
                        parsed_result["refinement_iteration"] = iteration
                        parsed_result["previous_quality_score"] = previous_quality
                        parsed_result["refinement_improvements"] = [
                            "Enhanced based on previous iterations",
                            "Improved integration with existing code",
                            "Better error handling and validation"
                        ]
                        parsed_result["analysis_method"] = "llm_refinement_processing"
                        
                        return parsed_result
                        
                    except json.JSONDecodeError as e:
                        self.logger.warning(f"Failed to parse refinement LLM response as JSON: {e}")
            
            # Fallback: enhance previous analysis with refinement insights
            self.logger.info("Using fallback refinement analysis")
            return self._generate_fallback_refinement_analysis(task_input, previous_outputs, current_state)
            
        except Exception as e:
            self.logger.error(f"Error in refinement analysis: {e}")
            # Fall back to standard analysis
            if task_input.intelligent_context and task_input.project_specifications:
                return await self._extract_analysis_from_intelligent_specs(
                    task_input.project_specifications, task_input.user_goal
                )
            else:
                return await self._analyze_project_requirements(task_input, shared_context)
    
    def _generate_fallback_refinement_analysis(
        self, 
        task_input: SmartCodeGeneratorInput, 
        previous_outputs: List[Dict[str, Any]], 
        current_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback refinement analysis when LLM is unavailable."""
        
        # Start with base analysis
        if task_input.intelligent_context and task_input.project_specifications:
            base_analysis = {
                "project_type": task_input.project_specifications.get("project_type", "cli_tool"),
                "primary_language": task_input.project_specifications.get("primary_language", "python"),
                "target_languages": task_input.project_specifications.get("target_languages", ["python"]),
                "technologies": task_input.project_specifications.get("technologies", []),
                "required_dependencies": task_input.project_specifications.get("required_dependencies", []),
                "optional_dependencies": task_input.project_specifications.get("optional_dependencies", [])
            }
        else:
            base_analysis = {
                "project_type": "cli_tool",
                "primary_language": "python",
                "target_languages": ["python"],
                "technologies": [],
                "required_dependencies": [],
                "optional_dependencies": []
            }
        
        # Enhance with refinement insights
        refinement_improvements = []
        if previous_outputs:
            avg_quality = sum(output.get('quality_score', 0.0) for output in previous_outputs) / len(previous_outputs)
            if avg_quality < 0.7:
                refinement_improvements.extend([
                    "Focus on code quality improvements",
                    "Add comprehensive error handling",
                    "Improve documentation and comments"
                ])
            elif avg_quality < 0.9:
                refinement_improvements.extend([
                    "Fine-tune implementation details",
                    "Optimize performance and structure",
                    "Enhance testing coverage"
                ])
        
        # Add current state insights
        if current_state.get('file_structure'):
            refinement_improvements.append("Integrate with existing file structure")
        if current_state.get('code_analysis'):
            refinement_improvements.append("Build upon existing code patterns")
        
        analysis = {
            **base_analysis,
            "code_generation_strategy": {
                "approach": "refinement_based",
                "file_structure": ["improved main module", "enhanced dependencies", "comprehensive documentation"],
                "implementation_priorities": ["quality improvement", "integration", "error handling", "documentation"],
                "architectural_patterns": ["modular design", "error resilience", "maintainability"]
            },
            "complexity_assessment": {
                "level": "medium",
                "factors": ["refinement requirements", "integration complexity"],
                "estimated_files": max(3, len(previous_outputs) + 1),
                "estimated_lines": 250  # Slightly more for refinement
            },
            "quality_requirements": {
                "testing_strategy": "comprehensive",
                "documentation_level": "detailed",
                "code_standards": ["PEP8", "type hints"] if base_analysis.get("primary_language") == "python" else ["standard", "documentation"]
            },
            "implementation_considerations": refinement_improvements or ["quality improvement", "integration"],
            "potential_challenges": ["maintaining backward compatibility", "improving quality metrics"],
            "refinement_analysis": True,
            "analysis_method": "fallback_refinement",
            "code_generation_needed": True,
            "refinement_improvements": refinement_improvements
        }
        
        return analysis

    def _extract_json_from_response(self, response: str) -> str:
        """Extract JSON content from LLM response, handling markdown code blocks."""
        response = response.strip()
        
        # Check if response is wrapped in markdown code blocks
        if response.startswith('```json'):
            # Find the end of the code block
            lines = response.split('\n')
            json_lines = []
            in_json_block = False
            
            for line in lines:
                if line.strip() == '```json':
                    in_json_block = True
                    continue
                elif line.strip() == '```' and in_json_block:
                    break
                elif in_json_block:
                    json_lines.append(line)
            
            return '\n'.join(json_lines)
        elif response.startswith('```'):
            # Handle generic code blocks
            lines = response.split('\n')
            json_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```') and not in_code_block:
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    break
                elif in_code_block:
                    json_lines.append(line)
            
            return '\n'.join(json_lines)
        else:
            # Response is already clean JSON
            return response

    async def _analyze_project_requirements(self, task_input: SmartCodeGeneratorInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Analysis - Analyze project requirements and structure."""
        self.logger.info("Starting project requirements analysis")
        
        analysis = {
            "project_type": "cli_tool",
            "primary_language": task_input.programming_language or "python",
            "target_languages": task_input.target_languages or ["python"],
            "technologies": task_input.technologies or [],
            "complexity_level": "medium",
            "code_generation_needed": True,
            "analysis_timestamp": datetime.datetime.now().isoformat()
        }
        
        return analysis

    async def _plan_code_structure(self, analysis_result: Dict[str, Any], task_input: SmartCodeGeneratorInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Planning - Plan code structure and files based on LLM analysis."""
        self.logger.info("Starting LLM-driven code structure planning")
        
        # Extract file structure from LLM analysis instead of hardcoded templates
        code_generation_strategy = analysis_result.get("code_generation_strategy", {})
        file_structure = code_generation_strategy.get("file_structure", [])
        complexity_assessment = analysis_result.get("complexity_assessment", {})
        estimated_files = complexity_assessment.get("estimated_files", 3)
        
        planned_files = []
        
        # If LLM provided specific file structure, use it
        if file_structure:
            for i, file_desc in enumerate(file_structure):
                # Extract meaningful file names from LLM descriptions
                file_path = self._extract_file_path_from_description(
                    file_desc, 
                    analysis_result.get("primary_language", "python"),
                    i
                )
                file_type = self._determine_file_type_from_description(file_desc)
                
                planned_files.append({
                    "file_path": file_path,
                    "file_type": file_type,
                    "description": file_desc,
                    "priority": i + 1,
                    "llm_specified": True
                })
        else:
            # Fallback: Create generic structure based on project type and language
            primary_language = analysis_result.get("primary_language", "python")
            project_type = analysis_result.get("project_type", "application")
            
            # Generate main module
            main_file_path = self._generate_main_file_path(project_type, primary_language, task_input)
            planned_files.append({
                "file_path": main_file_path,
                "file_type": "main_module", 
                "description": f"Main {project_type} module",
                "priority": 1,
                "llm_specified": False
            })
            
            # Add dependency file if needed
            if self._needs_dependency_file(analysis_result):
                dep_file = self._get_dependency_file_name(primary_language)
                planned_files.append({
                    "file_path": dep_file,
                    "file_type": "dependency_file",
                    "description": f"{primary_language.title()} dependencies",
                    "priority": 2,
                    "llm_specified": False
                })
            
            # Add documentation if specified
            if self._needs_documentation(analysis_result):
                planned_files.append({
                    "file_path": "README.md",
                    "file_type": "documentation",
                    "description": "Project documentation",
                    "priority": 3,
                    "llm_specified": False
                })
        
        planning = {
            "planned_files": planned_files,
            "total_files": len(planned_files),
            "structure_complexity": complexity_assessment.get("level", "medium"),
            "estimated_lines": complexity_assessment.get("estimated_lines", 200),
            "llm_driven": bool(file_structure),
            "analysis_method": analysis_result.get("analysis_method", "unknown"),
            "planning_timestamp": datetime.datetime.now().isoformat()
        }
        
        self.logger.info(f"Planned {len(planned_files)} files using {'LLM-driven' if file_structure else 'fallback'} planning")
        return planning

    def _extract_file_path_from_description(self, file_desc: str, language: str, index: int) -> str:
        """Extract meaningful file path from LLM file description."""
        file_desc_lower = file_desc.lower()
        
        # Look for explicit file names in the description
        if ".py" in file_desc_lower and language == "python":
            # Extract .py filename if mentioned
            import re
            py_files = re.findall(r'(\w+\.py)', file_desc_lower)
            if py_files:
                return py_files[0]
        
        if ".js" in file_desc_lower and language == "javascript":
            # Extract .js filename if mentioned
            import re
            js_files = re.findall(r'(\w+\.js)', file_desc_lower)
            if js_files:
                return js_files[0]
        
        # Generate meaningful names based on description content
        extension = self._get_file_extension(language)
        
        if "main" in file_desc_lower or "entry" in file_desc_lower:
            return f"main.{extension}"
        elif "wifi" in file_desc_lower or "wireless" in file_desc_lower:
            return f"wifi_scanner.{extension}"
        elif "network" in file_desc_lower and "scanner" in file_desc_lower:
            return f"network_scanner.{extension}"
        elif "scanner" in file_desc_lower:
            return f"scanner.{extension}"
        elif "config" in file_desc_lower or "setting" in file_desc_lower:
            return f"config.{extension}"
        elif "util" in file_desc_lower or "helper" in file_desc_lower:
            return f"utils.{extension}"
        elif "cli" in file_desc_lower or "command" in file_desc_lower:
            return f"cli.{extension}"
        elif "api" in file_desc_lower:
            return f"api.{extension}"
        elif "server" in file_desc_lower:
            return f"server.{extension}"
        elif "client" in file_desc_lower:
            return f"client.{extension}"
        elif "database" in file_desc_lower or "db" in file_desc_lower:
            return f"database.{extension}"
        elif "test" in file_desc_lower:
            return f"test_{index}.{extension}"
        else:
            # Generic module name based on index
            return f"module_{index + 1}.{extension}"

    def _determine_file_type_from_description(self, file_desc: str) -> str:
        """Determine file type from LLM description."""
        file_desc_lower = file_desc.lower()
        
        if "main" in file_desc_lower or "entry" in file_desc_lower:
            return "main_module"
        elif "test" in file_desc_lower:
            return "test_module"
        elif "config" in file_desc_lower:
            return "config_module"
        elif "util" in file_desc_lower or "helper" in file_desc_lower:
            return "utility_module"
        elif "api" in file_desc_lower:
            return "api_module"
        elif "cli" in file_desc_lower:
            return "cli_module"
        elif "database" in file_desc_lower or "db" in file_desc_lower:
            return "database_module"
        else:
            return "code_module"

    def _generate_main_file_path(self, project_type: str, language: str, task_input: SmartCodeGeneratorInput) -> str:
        """Generate appropriate main file path based on project type and language."""
        extension = self._get_file_extension(language)
        
        # Use goal content to determine appropriate name
        if task_input.user_goal:
            goal_lower = task_input.user_goal.lower()
            if "scanner" in goal_lower:
                return f"scanner.{extension}"
            elif "api" in goal_lower:
                return f"api.{extension}"
            elif "server" in goal_lower:
                return f"server.{extension}"
            elif "cli" in goal_lower or "command" in goal_lower:
                return f"cli.{extension}"
            elif "app" in goal_lower:
                return f"app.{extension}"
        
        # Default based on project type
        if project_type == "cli_tool":
            return f"cli.{extension}"
        elif project_type == "web_app":
            return f"app.{extension}"
        elif project_type == "api":
            return f"api.{extension}"
        elif project_type == "library":
            return f"lib.{extension}"
        else:
            return f"main.{extension}"

    def _needs_dependency_file(self, analysis_result: Dict[str, Any]) -> bool:
        """Determine if a dependency file is needed."""
        required_deps = analysis_result.get("required_dependencies", [])
        optional_deps = analysis_result.get("optional_dependencies", [])
        return len(required_deps) > 0 or len(optional_deps) > 0

    def _get_dependency_file_name(self, language: str) -> str:
        """Get appropriate dependency file name for language."""
        if language == "python":
            return "requirements.txt"
        elif language == "javascript" or language == "typescript":
            return "package.json"
        elif language == "java":
            return "pom.xml"
        elif language == "csharp":
            return "packages.config"
        elif language == "go":
            return "go.mod"
        elif language == "rust":
            return "Cargo.toml"
        else:
            return "dependencies.txt"

    def _needs_documentation(self, analysis_result: Dict[str, Any]) -> bool:
        """Determine if documentation should be generated."""
        quality_reqs = analysis_result.get("quality_requirements", {})
        doc_level = quality_reqs.get("documentation_level", "standard")
        return doc_level in ["standard", "detailed", "comprehensive"]

    async def _generate_code_files(self, planning_result: Dict[str, Any], task_input: SmartCodeGeneratorInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Generation - Generate LLM-driven code files."""
        self.logger.info("Starting LLM-driven code file generation")
        
        planned_files = planning_result.get("planned_files", [])
        generated_files = []
        project_path = Path(task_input.project_path or ".")
        
        # Get refinement context for content generation
        refinement_context = shared_context.get("refinement_context")
        
        for file_plan in planned_files:
            file_path = file_plan["file_path"]
            file_type = file_plan["file_type"]
            description = file_plan["description"]
            llm_specified = file_plan.get("llm_specified", False)
            
            try:
                # Generate content using LLM instead of hardcoded templates
                if llm_specified or self.llm_provider:
                    content = await self._generate_llm_code_content(
                        file_plan, task_input, shared_context, refinement_context
                    )
                else:
                    # Only use minimal fallback if LLM is completely unavailable
                    content = self._generate_minimal_fallback_content(file_plan, task_input)
                
                # Write file to project directory
                full_path = project_path / file_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(content, encoding='utf-8')
                
                generated_files.append({
                    "file_path": file_path,
                    "full_path": str(full_path),
                    "file_type": file_type,
                    "content_length": len(content),
                    "status": "success",
                    "generation_method": "llm_driven" if llm_specified or self.llm_provider else "fallback",
                    "description": description
                })
                
                self.logger.info(f"Generated file: {file_path} ({len(content)} characters)")
                
            except Exception as e:
                self.logger.error(f"Failed to generate file {file_path}: {e}")
                generated_files.append({
                    "file_path": file_path,
                    "full_path": str(project_path / file_path),
                    "file_type": file_type,
                    "status": "error",
                    "error": str(e),
                    "description": description
                })
        
        generation = {
            "generated_files": generated_files,
            "total_generated": len([f for f in generated_files if f["status"] == "success"]),
            "total_failed": len([f for f in generated_files if f["status"] == "error"]),
            "llm_driven_files": len([f for f in generated_files if f.get("generation_method") == "llm_driven"]),
            "generation_timestamp": datetime.datetime.now().isoformat()
        }
        
        return generation

    async def _generate_llm_code_content(
        self, 
        file_plan: Dict[str, Any], 
        task_input: SmartCodeGeneratorInput, 
        shared_context: Dict[str, Any],
        refinement_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code content for a specific file using LLM."""
        
        file_path = file_plan["file_path"]
        file_type = file_plan["file_type"]
        description = file_plan["description"]
        
        # Build comprehensive prompt for code generation
        prompt_parts = []
        
        # Base generation request
        prompt_parts.append(f"""Generate complete, production-ready code for the file: {file_path}

File Description: {description}
File Type: {file_type}
User Goal: {task_input.user_goal}""")

        # Add project specifications if available
        if task_input.project_specifications:
            specs = task_input.project_specifications
            prompt_parts.append(f"""
Project Specifications:
- Project Type: {specs.get('project_type', 'unknown')}
- Primary Language: {specs.get('primary_language', 'unknown')}
- Technologies: {specs.get('technologies', [])}
- Required Dependencies: {specs.get('required_dependencies', [])}""")

        # Add refinement context if this is a refinement iteration
        if refinement_context:
            previous_outputs = refinement_context.get("previous_outputs", [])
            if previous_outputs:
                latest_output = previous_outputs[-1]
                prompt_parts.append(f"""
REFINEMENT CONTEXT:
This is iteration {refinement_context.get('iteration', 1)} of refinement.
Previous quality score: {latest_output.get('quality_score', 'unknown')}

Previous implementation issues to address:
{refinement_context.get('refinement_needs', 'General quality improvement needed')}

Please generate IMPROVED code that addresses these specific issues.""")

        # Add specific requirements based on file type
        if file_type == "main_module":
            prompt_parts.append("""
Requirements for main module:
- Include proper shebang line if executable
- Implement complete functionality as described in the user goal
- Include proper error handling and logging
- Follow best practices for the target language
- Include comprehensive docstrings/comments""")
        elif file_type in ["code_module", "api_module", "cli_module", "utility_module"]:
            prompt_parts.append(f"""
Requirements for {file_type}:
- Implement the specific functionality described in the file description
- Include proper class/function definitions
- Add comprehensive error handling
- Include type hints if using Python
- Follow modular design principles""")

        # Final instructions
        prompt_parts.append("""
CRITICAL REQUIREMENTS:
1. Generate COMPLETE, working code - not pseudocode or templates
2. Code must be syntactically correct and runnable
3. Include all necessary imports and dependencies
4. Follow language-specific best practices
5. Add comprehensive documentation
6. Ensure code aligns with the user goal

Return ONLY the complete source code for this file, no explanations or markdown formatting.""")

        prompt = "\n".join(prompt_parts)
        
        try:
            if self.llm_provider:
                response = await self.llm_provider.generate(
                    prompt=prompt,
                    max_tokens=4000,  # Allow for larger code files
                    temperature=0.1   # Lower temperature for more consistent code
                )
                
                if response and response.strip():
                    # Clean up any markdown formatting if present
                    cleaned_content = self._clean_code_response(response)
                    return cleaned_content
                else:
                    self.logger.warning(f"Empty LLM response for {file_path}, using fallback")
                    return self._generate_minimal_fallback_content(file_plan, task_input)
            else:
                self.logger.warning(f"No LLM provider available for {file_path}, using fallback")
                return self._generate_minimal_fallback_content(file_plan, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM code generation failed for {file_path}: {e}")
            return self._generate_minimal_fallback_content(file_plan, task_input)

    def _clean_code_response(self, response: str) -> str:
        """Clean LLM response to extract pure code content."""
        response = response.strip()
        
        # Remove markdown code blocks if present
        if response.startswith('```'):
            lines = response.split('\n')
            code_lines = []
            in_code_block = False
            
            for line in lines:
                if line.strip().startswith('```') and not in_code_block:
                    in_code_block = True
                    continue
                elif line.strip() == '```' and in_code_block:
                    break
                elif in_code_block:
                    code_lines.append(line)
            
            if code_lines:
                return '\n'.join(code_lines)
        
        # Return original if no code blocks found
        return response

    def _generate_minimal_fallback_content(self, file_plan: Dict[str, Any], task_input: SmartCodeGeneratorInput) -> str:
        """Generate minimal fallback content when LLM is unavailable."""
        file_path = file_plan["file_path"]
        file_type = file_plan["file_type"]
        description = file_plan["description"]
        
        # Determine language from file extension
        if file_path.endswith('.py'):
            return self._generate_python_fallback(file_plan, task_input)
        elif file_path.endswith('.js'):
            return self._generate_javascript_fallback(file_plan, task_input)
        elif file_path.endswith('.md'):
            return self._generate_markdown_fallback(file_plan, task_input)
        elif file_path.endswith('.txt'):
            return self._generate_text_fallback(file_plan, task_input)
        else:
            return f"""# {description}
# TODO: Implement {file_path}
# Generated as fallback when LLM was unavailable

# File type: {file_type}
# User goal: {task_input.user_goal or 'Not specified'}
"""

    def _generate_python_fallback(self, file_plan: Dict[str, Any], task_input: SmartCodeGeneratorInput) -> str:
        """Generate minimal Python fallback."""
        file_path = file_plan["file_path"]
        description = file_plan["description"]
        
        is_executable = file_plan.get("file_type") == "main_module"
        shebang = "#!/usr/bin/env python3\n" if is_executable else ""
        
        return f"""{shebang}\"\"\"
{description}

Generated as minimal fallback implementation.
TODO: Implement full functionality for: {task_input.user_goal or 'user requirements'}
\"\"\"

import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    \"\"\"Main function - implement core functionality here.\"\"\"
    logger.info(f"Starting {file_path}")
    
    # TODO: Implement functionality for: {task_input.user_goal or 'user requirements'}
    print(f"This is a placeholder implementation of {file_path}")
    print(f"Goal: {task_input.user_goal or 'Not specified'}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

    def _generate_javascript_fallback(self, file_plan: Dict[str, Any], task_input: SmartCodeGeneratorInput) -> str:
        """Generate minimal JavaScript fallback."""
        description = file_plan["description"]
        
        return f"""/**
 * {description}
 * 
 * Generated as minimal fallback implementation.
 * TODO: Implement full functionality for: {task_input.user_goal or 'user requirements'}
 */

console.log('Starting application');

// TODO: Implement functionality for: {task_input.user_goal or 'user requirements'}

function main() {{
    console.log('This is a placeholder implementation');
    console.log('Goal: {task_input.user_goal or 'Not specified'}');
}}

// Run main function
main();
"""

    def _generate_markdown_fallback(self, file_plan: Dict[str, Any], task_input: SmartCodeGeneratorInput) -> str:
        """Generate minimal Markdown fallback."""
        project_type = "Project"
        if task_input.project_specifications:
            project_type = task_input.project_specifications.get("project_type", "Project").title()
        
        return f"""# {project_type}

{task_input.user_goal or 'Project description not specified'}

## Overview

This is a minimal documentation template generated as fallback.

## Installation

```bash
# Add installation instructions here
```

## Usage

```bash
# Add usage examples here
```

## Features

- TODO: Add project features
- TODO: Document functionality

## Requirements

- TODO: List requirements and dependencies

## License

TODO: Add license information
"""

    def _generate_text_fallback(self, file_plan: Dict[str, Any], task_input: SmartCodeGeneratorInput) -> str:
        """Generate minimal text file fallback."""
        if "requirements" in file_plan["file_path"].lower():
            # Basic requirements.txt
            if task_input.project_specifications:
                deps = task_input.project_specifications.get("required_dependencies", [])
                if deps:
                    return "\n".join(deps) + "\n"
            return "# Add project dependencies here\n"
        else:
            return f"# {file_plan['description']}\n# TODO: Add content for {file_plan['file_path']}\n"

    def _get_file_extension(self, language: str) -> str:
        """Get file extension for programming language."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "java": "java",
            "csharp": "cs",
            "cpp": "cpp",
            "c": "c",
            "go": "go",
            "rust": "rs"
        }
        return extensions.get(language.lower(), "txt")

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = SmartCodeGeneratorInput.model_json_schema()
        output_schema = SmartCodeGeneratorOutput.model_json_schema()
        
        return AgentCard(
            agent_id=SmartCodeGeneratorAgent_v1.AGENT_ID,
            name=SmartCodeGeneratorAgent_v1.AGENT_NAME,
            description=SmartCodeGeneratorAgent_v1.AGENT_DESCRIPTION,
            version=SmartCodeGeneratorAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[SmartCodeGeneratorAgent_v1.CATEGORY.value],
            visibility=SmartCodeGeneratorAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_code": True,
                "supports_multiple_languages": True,
                "intelligent_context_aware": True
            },
            metadata={
                "callable_fn_path": f"{SmartCodeGeneratorAgent_v1.__module__}.{SmartCodeGeneratorAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[SmartCodeGeneratorInput]:
        return SmartCodeGeneratorInput

    def get_output_schema(self) -> Type[SmartCodeGeneratorOutput]:
        return SmartCodeGeneratorOutput 