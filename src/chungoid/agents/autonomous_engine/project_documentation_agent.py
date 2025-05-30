"""
ProjectDocumentationAgent_v1: Clean, unified LLM-powered documentation generation.

This agent generates comprehensive project documentation by:
1. Using unified discovery to understand project structure and technology stack
2. Using YAML prompt template with rich discovery data
3. Letting the LLM make intelligent documentation decisions with maximum intelligence

No legacy patterns, no hardcoded logic, no complex phases.
Pure unified approach for maximum agentic documentation intelligence.
"""

from __future__ import annotations

import logging
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type

from pydantic import BaseModel, Field, model_validator

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.schemas.common import ConfidenceScore
from chungoid.utils.agent_registry import AgentCard
from chungoid.utils.agent_registry_meta import AgentCategory, AgentVisibility
from chungoid.registry import register_autonomous_engine_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
)

logger = logging.getLogger(__name__)


class DocumentationAgentInput(BaseModel):
    """Clean input schema focused on core documentation needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique task identifier")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project identifier")
    
    # Core requirements
    user_goal: str = Field(..., description="What the user wants documented")
    project_path: str = Field(default=".", description="Project directory path")
    
    # Documentation options
    include_api_docs: bool = Field(default=True, description="Whether to generate API documentation")
    include_user_guide: bool = Field(default=True, description="Whether to generate user guide")
    include_dependency_audit: bool = Field(default=True, description="Whether to generate dependency audit")
    
    # Intelligent context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications")
    intelligent_context: bool = Field(default=False, description="Whether intelligent specifications provided")
    
    @model_validator(mode='after')
    def validate_requirements(self) -> 'DocumentationAgentInput':
        """Ensure we have minimum requirements for documentation."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for documentation generation")
        return self


class DocumentationAgentOutput(BaseModel):
    """Clean output schema focused on documentation deliverables."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core documentation deliverables
    documentation_files: Dict[str, str] = Field(default_factory=dict, description="Generated documentation files {file_path: content}")
    documentation_summary: str = Field(..., description="Summary of generated documentation")
    documentation_recommendations: List[str] = Field(default_factory=list, description="Recommendations for documentation maintenance")
    
    # Quality insights
    files_created: List[str] = Field(default_factory=list, description="List of documentation files created")
    project_analysis: Dict[str, Any] = Field(default_factory=dict, description="Analysis of project for documentation purposes")
    
    # Metadata
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in documentation quality")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["documentation_generation", "project_analysis", "comprehensive_reporting"])
class ProjectDocumentationAgent_v1(UnifiedAgent):
    """
    Clean, unified project documentation agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence for documentation.
    No legacy patterns, no hardcoded logic, no complex phases.
    """
    
    AGENT_ID: ClassVar[str] = "ProjectDocumentationAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Documentation Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered project documentation generation"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "project_documentation_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "4.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["documentation_generation", "project_analysis", "comprehensive_reporting"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DOCUMENTATION_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[DocumentationAgentInput]] = DocumentationAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[DocumentationAgentOutput]] = DocumentationAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_documentation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified documentation")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Clean execution: Unified discovery + YAML template + LLM documentation intelligence.
        Single iteration, maximum intelligence, no hardcoded phases.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"Generating documentation: {task_input.user_goal}")

            # Generate documentation using unified approach
            documentation_result = await self._generate_documentation(task_input)
            
            # Create clean output
            output = DocumentationAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                documentation_files=documentation_result["documentation_files"],
                documentation_summary=documentation_result["documentation_summary"],
                documentation_recommendations=documentation_result["documentation_recommendations"],
                files_created=list(documentation_result["documentation_files"].keys()),
                project_analysis=documentation_result["project_analysis"],
                confidence_score=documentation_result["confidence_score"],
                message=f"Generated documentation for: {task_input.user_goal}"
            )
            
            return IterationResult(
                output=output,
                quality_score=documentation_result["confidence_score"].value,
                tools_used=["unified_discovery", "yaml_template", "llm_documentation"],
                protocol_used="unified_documentation"
            )
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            
            # Clean error handling
            error_output = DocumentationAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                documentation_files={},
                documentation_summary="Documentation generation failed",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Documentation generation failed"
                ),
                message="Documentation generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="unified_documentation"
            )

    def _parse_inputs(self, inputs: Any) -> DocumentationAgentInput:
        """Parse inputs cleanly into DocumentationAgentInput with detailed validation."""
        try:
            if isinstance(inputs, DocumentationAgentInput):
                # Validate existing input object
                if not inputs.user_goal or not inputs.user_goal.strip():
                    raise ValueError("DocumentationAgentInput has empty or whitespace user_goal")
                return inputs
            elif isinstance(inputs, dict):
                # Validate required fields before creation
                if 'user_goal' not in inputs:
                    raise ValueError("Missing required field 'user_goal' in input dictionary")
                if not inputs['user_goal'] or not inputs['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return DocumentationAgentInput(**inputs)
            elif hasattr(inputs, 'dict'):
                input_dict = inputs.dict()
                if 'user_goal' not in input_dict:
                    raise ValueError("Missing required field 'user_goal' in input object")
                if not input_dict['user_goal'] or not input_dict['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return DocumentationAgentInput(**input_dict)
            else:
                raise ValueError(f"Invalid input type: {type(inputs)}. Expected DocumentationAgentInput, dict, or object with dict() method. Received: {inputs}")
        except Exception as e:
            raise ValueError(f"Input parsing failed for ProjectDocumentationAgent: {e}. Input received: {inputs}")

    async def _generate_documentation(self, task_input: DocumentationAgentInput) -> Dict[str, Any]:
        """
        Generate documentation using unified discovery + YAML template with detailed validation.
        Pure unified approach - no hardcoded phases or documentation logic.
        """
        try:
            # Validate prompt template access
            prompt_template = self.prompt_manager.get_prompt_definition(
                "project_documentation_agent_v1_prompt",
                "1.0.0",
                sub_path="autonomous_engine"
            )
            
            if not prompt_template:
                raise ValueError("Failed to load documentation prompt template - returned None/empty")
            
            if not hasattr(prompt_template, 'user_prompt_template'):
                raise ValueError(f"Prompt template missing 'user_prompt_template' attribute. Available attributes: {dir(prompt_template)}")
            
            # Unified discovery for intelligent documentation context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "artifacts", "code_analysis"]
            )
            
            if not discovery_results:
                raise ValueError("Universal discovery returned None/empty results")
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            if not technology_context:
                raise ValueError("Technology discovery returned None/empty results")
            
            # Build template variables for maximum LLM documentation intelligence
            template_vars = {
                # Original template variables (maintaining compatibility)
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "project_id": task_input.project_id,
                "include_api_docs": task_input.include_api_docs,
                "include_user_guide": task_input.include_user_guide,
                "include_dependency_audit": task_input.include_dependency_audit,
                "available_tools": self._format_available_tools(),
                
                # Enhanced unified discovery variables for maximum intelligence
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Additional context for intelligent documentation
                "project_specifications": task_input.project_specifications or {},
                "intelligent_context": task_input.intelligent_context
            }
            
            # Render template with validation
            try:
                formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                    prompt_template.user_prompt_template,
                    template_vars
                )
            except Exception as e:
                raise ValueError(f"Failed to render documentation prompt template: {e}. Template variables: {list(template_vars.keys())}")
            
            # Get system prompt if available
            system_prompt = getattr(prompt_template, 'system_prompt', None)
            
            # Call LLM with maximum documentation intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Detailed response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for documentation. Prompt length: {len(formatted_prompt)} chars. User goal: {task_input.user_goal}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for documentation. Response: '{response}'. User goal: {task_input.user_goal}")
            
            if len(response.strip()) < 50:
                raise ValueError(f"LLM documentation response too short ({len(response)} chars). Expected substantial documentation content. Response: '{response}'. User goal: {task_input.user_goal}")
            
            # Parse LLM response (expecting JSON from template) - NO FALLBACKS
            try:
                json_str = self._extract_json_from_response(response)
                if not json_str:
                    raise ValueError(f"No JSON found in documentation response. Response: '{response}'")
                
                result = json.loads(json_str)
                if not isinstance(result, dict):
                    raise ValueError(f"Parsed JSON is not a dictionary: {type(result)}. JSON: '{json_str}'")
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSON from documentation response: {e}. JSON string: '{json_str}'. Full response: '{response}'")
            except Exception as e:
                raise ValueError(f"Failed to extract/parse JSON from documentation response: {e}. Response: '{response}'")
            
            # Transform template output to our clean schema
            documentation_files = {}
            doc_files = result.get("documentation_files", [])
            
            if not doc_files:
                raise ValueError(f"No documentation_files in parsed response. Result keys: {list(result.keys())}. Response: '{response}'")
            
            for doc in doc_files:
                if not isinstance(doc, dict):
                    raise ValueError(f"Documentation file entry is not a dict: {type(doc)}. Entry: {doc}")
                
                file_path = doc.get("path", "README.md")
                content = doc.get("content", "")
                
                if not content or not content.strip():
                    raise ValueError(f"Documentation file '{file_path}' has empty content. Doc entry: {doc}")
                
                documentation_files[file_path] = content
            
            if not documentation_files:
                raise ValueError(f"No valid documentation files extracted. Result: {result}")
            
            return {
                "documentation_files": documentation_files,
                "documentation_summary": self._create_documentation_summary(result, discovery_results),
                "documentation_recommendations": result.get("recommendations", []),
                "project_analysis": {
                    "project_type": technology_context.get("primary_language", "unknown"),
                    "complexity": discovery_results.get("complexity_assessment", "moderate"),
                    "documentation_scope": len(documentation_files),
                    "discovered_patterns": discovery_results.get("patterns", {})
                },
                "confidence_score": ConfidenceScore(
                    value=result.get("confidence", 0.85),
                    method="llm_documentation_assessment",
                    explanation=result.get("reasoning", "Documentation generated successfully with unified discovery")
                )
            }
            
        except Exception as e:
            error_msg = f"""ProjectDocumentationAgent documentation generation failed:

ERROR: {e}

INPUT CONTEXT:
- User Goal: {task_input.user_goal}
- Project Path: {task_input.project_path}
- Project ID: {task_input.project_id}
- Include API Docs: {task_input.include_api_docs}
- Include User Guide: {task_input.include_user_guide}
- Include Dependency Audit: {task_input.include_dependency_audit}

DOCUMENTATION CONTEXT:
- Project Specifications: {task_input.project_specifications}
- Intelligent Context: {task_input.intelligent_context}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _create_documentation_summary(self, llm_result: Dict[str, Any], discovery_results: Dict[str, Any]) -> str:
        """Create a comprehensive documentation summary from LLM result and discovery."""
        doc_files = llm_result.get("documentation_files", [])
        files_count = len(doc_files)
        
        project_info = discovery_results.get("structure", {})
        tech_stack = discovery_results.get("technology", {})
        
        summary_parts = [
            f"Generated {files_count} documentation files",
            f"Project type: {tech_stack.get('primary_language', 'detected from codebase')}",
            f"Documentation scope: {llm_result.get('reasoning', 'comprehensive project documentation')}"
        ]
        
        if project_info.get("has_tests"):
            summary_parts.append("Included testing documentation")
        
        if tech_stack.get("framework"):
            summary_parts.append(f"Framework-specific documentation for {tech_stack['framework']}")
        
        return ". ".join(summary_parts) + "."

    def _format_available_tools(self) -> str:
        """Format available tools for template (simplified unified approach)."""
        tools = [
            "filesystem_read_file - Read and analyze project files",
            "filesystem_list_directory - Explore project structure", 
            "filesystem_write_file - Create documentation files",
            "web_search - Research documentation best practices",
            "unified_discovery - Comprehensive project analysis"
        ]
        
        return "Available tools:\n" + "\n".join(f"- {tool}" for tool in tools)

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = DocumentationAgentInput.model_json_schema()
        output_schema = DocumentationAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=ProjectDocumentationAgent_v1.AGENT_ID,
            name=ProjectDocumentationAgent_v1.AGENT_NAME,
            description=ProjectDocumentationAgent_v1.AGENT_DESCRIPTION,
            version=ProjectDocumentationAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[ProjectDocumentationAgent_v1.CATEGORY.value],
            visibility=ProjectDocumentationAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_documentation": True,
                "clean_documentation": True,
                "no_hardcoded_logic": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{ProjectDocumentationAgent_v1.__module__}.{ProjectDocumentationAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[DocumentationAgentInput]:
        return DocumentationAgentInput

    def get_output_schema(self) -> Type[DocumentationAgentOutput]:
        return DocumentationAgentOutput 