"""
ProjectDocumentationAgent_v1: Dead simple, LLM-powered documentation generation.

This agent generates project documentation by:
1. Using MCP tools to understand the project structure and code
2. Using main prompt template from YAML  
3. Letting the LLM run the show to create comprehensive documentation

No complex phases, no brittle parsing, just LLM + MCP tools.
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
from chungoid.registry import register_autonomous_engine_agent

from ...schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
)

logger = logging.getLogger(__name__)

class ProjectDocumentationInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this documentation task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core fields - simplified
    user_goal: Optional[str] = Field(None, description="What the user built")
    project_path: Optional[str] = Field(None, description="Where the project is located")
    
    # Traditional fields for backward compatibility
    project_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of project blueprint if available")
    generated_code_root_path: Optional[str] = Field(None, description="Path to generated codebase")
    
    # Context fields from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    
    # Documentation options
    include_api_docs: bool = Field(default=True, description="Whether to generate API documentation")
    include_user_guide: bool = Field(default=True, description="Whether to generate user guide")
    include_dependency_audit: bool = Field(default=True, description="Whether to generate dependency audit")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'ProjectDocumentationInput':
        """Ensure we have minimum info to generate documentation."""
        
        # Need either user_goal or project_path
        if not self.user_goal and not self.project_path:
            raise ValueError("Either user_goal or project_path is required")
        
        # Default project_path if not provided
        if not self.project_path:
            self.project_path = "."
        
        return self

class ProjectDocumentationOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    status: str = Field(..., description="Status of documentation generation (SUCCESS, FAILURE, etc.).")
    
    # Generated documentation files
    documentation_files: Dict[str, str] = Field(default_factory=dict, description="Generated documentation files {file_path: content}")
    
    # Results
    generation_summary: str = Field(..., description="Summary of documentation generation")
    files_created: List[str] = Field(default_factory=list, description="List of documentation files created")
    
    # Status and metadata
    message: str = Field(..., description="Message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the documentation quality.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the generation process.")

@register_autonomous_engine_agent(capabilities=["documentation_generation", "project_analysis", "comprehensive_reporting"])
class ProjectDocumentationAgent_v1(UnifiedAgent):
    """
    Dead simple documentation generation agent.
    
    1. MCP tools for project understanding and file operations
    2. Main prompt from YAML template  
    3. LLM runs the show - creates comprehensive documentation without complex phases
    """
    
    AGENT_ID: ClassVar[str] = "ProjectDocumentationAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Documentation Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Simple, LLM-powered documentation generation with MCP tools."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "project_documentation_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "3.0.0"  # Major version bump for simplification
    CAPABILITIES: ClassVar[List[str]] = ["documentation_generation", "project_analysis", "comprehensive_reporting"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.DOCUMENTATION_GENERATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ProjectDocumentationInput]] = ProjectDocumentationInput
    OUTPUT_SCHEMA: ClassVar[Type[ProjectDocumentationOutput]] = ProjectDocumentationOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["simple_documentation_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["mcp_file_operations"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized as simple documentation agent.")

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Recursively convert Pydantic objects to dictionaries."""
        if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            return obj.dict()
        elif isinstance(obj, dict):
            return {key: self._convert_pydantic_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_pydantic_to_dict(item) for item in obj]
        else:
            return obj

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Dead simple execution: Let the LLM understand the project and generate documentation.
        No complex phases, just LLM + MCP tools.
        """
        try:
            # Convert inputs - handle both dict and Pydantic objects
            if isinstance(context.inputs, dict):
                converted_inputs = self._convert_pydantic_to_dict(context.inputs)
                task_input = ProjectDocumentationInput(**converted_inputs)
            elif hasattr(context.inputs, 'dict'):
                task_input = ProjectDocumentationInput(**context.inputs.dict())
            else:
                task_input = context.inputs

            self.logger.info(f"Starting simple documentation generation for: {task_input.user_goal or 'project'}")

            # 1. Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            # 2. Let LLM generate documentation using main prompt template
            documentation_result = await self._generate_documentation_with_llm(task_input, project_context)
            
            # 3. Create simple output
            output = ProjectDocumentationOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS" if documentation_result["success"] else "FAILURE",
                documentation_files=documentation_result.get("documentation_files", {}),
                generation_summary=documentation_result.get("generation_summary", "Documentation generation completed"),
                files_created=list(documentation_result.get("documentation_files", {}).keys()),
                message=documentation_result.get("message", "Documentation generation completed"),
                confidence_score=ConfidenceScore(
                    value=documentation_result.get("confidence", 0.8),
                    method="llm_self_assessment",
                    explanation="LLM assessed its own documentation generation quality"
                ),
                error_message=documentation_result.get("error"),
                usage_metadata={
                    "iteration": iteration + 1,
                    "files_generated": len(documentation_result.get("documentation_files", {})),
                    "documentation_created": documentation_result.get("success", False)
                }
            )
            
            return IterationResult(
                output=output,
                quality_score=documentation_result.get("confidence", 0.8),
                tools_used=["llm_analysis", "mcp_file_operations"],
                protocol_used="simple_documentation_generation"
            )
            
        except Exception as e:
            self.logger.error(f"Documentation generation failed: {e}")
            
            # Safe error output
            task_id = getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4())
            project_id = getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown'
            
            error_output = ProjectDocumentationOutput(
                task_id=task_id,
                project_id=project_id,
                status="ERROR",
                documentation_files={},
                generation_summary="Documentation generation failed",
                files_created=[],
                message="Documentation generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="simple_documentation_generation"
            )

    async def _gather_project_context(self, task_input: ProjectDocumentationInput) -> str:
        """
        Simple method: Gather all relevant project context for the LLM.
        "Learn about the project" - scan files, read code, understand structure.
        """
        self.logger.info(f"Gathering project context from: {task_input.project_path}")
        
        context_parts = []
        
        # Add user goal
        if task_input.user_goal:
            context_parts.append(f"User Goal: {task_input.user_goal}")
        
        # Add project specifications if available
        if task_input.project_specifications:
            context_parts.append(f"Project Specifications: {json.dumps(task_input.project_specifications, indent=2)}")
        
        # Try to read existing blueprint if available
        if task_input.project_blueprint_doc_id:
            try:
                blueprint_result = await self._call_mcp_tool("chromadb_retrieve_document", {
                    "collection_name": "blueprint_artifacts_collection",
                    "document_id": task_input.project_blueprint_doc_id
                })
                if blueprint_result.get("success"):
                    blueprint_content = blueprint_result.get("content", "")
                    context_parts.append(f"Project Blueprint: {blueprint_content}")
                    self.logger.info("Successfully retrieved project blueprint")
            except Exception as e:
                self.logger.warning(f"Could not retrieve blueprint: {e}")
        
        # Scan project directory with MCP tools
        try:
            list_result = await self._call_mcp_tool("filesystem_list_directory", {
                "directory_path": task_input.project_path
            })
            
            if list_result.get("success"):
                files = list_result.get("files", [])
                if files:
                    context_parts.append(f"Project files in {task_input.project_path}: {files}")
                    
                    # Read key project files for context
                    important_files = []
                    for file_info in files[:15]:  # Limit to first 15 files
                        filename = file_info.get("name", "") if isinstance(file_info, dict) else str(file_info)
                        
                        # Prioritize important project files
                        if any(pattern in filename.lower() for pattern in [
                            "readme", "setup.py", "pyproject.toml", "package.json", "main.", "app.", "index.", 
                            "requirements", "__init__.py", "config", "settings"
                        ]):
                            important_files.append(filename)
                    
                    # Read important files for context
                    for filename in important_files[:8]:  # Limit to 8 important files
                        try:
                            file_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": f"{task_input.project_path}/{filename}"
                            })
                            if file_result.get("success"):
                                content = file_result.get("content", "")
                                # Truncate very long files
                                if len(content) > 2000:
                                    content = content[:2000] + "... [truncated]"
                                context_parts.append(f"--- {filename} ---\n{content}")
                        except Exception as e:
                            self.logger.warning(f"Could not read {filename}: {e}")
                    
                    # Get directory structure for better understanding
                    try:
                        structure_result = await self._call_mcp_tool("filesystem_list_directory", {
                            "directory_path": task_input.project_path,
                            "recursive": True
                        })
                        if structure_result.get("success"):
                            all_files = structure_result.get("files", [])
                            context_parts.append(f"Project structure: {all_files[:30]}")  # Limit structure info
                    except Exception as e:
                        self.logger.warning(f"Could not get project structure: {e}")
                else:
                    context_parts.append(f"Project directory {task_input.project_path} is empty")
            else:
                context_parts.append(f"Could not scan project directory: {list_result.get('error', 'unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"Error scanning project directory: {e}")
            context_parts.append(f"Error scanning project directory: {str(e)}")
        
        return "\n\n".join(context_parts)

    async def _generate_documentation_with_llm(self, task_input: ProjectDocumentationInput, project_context: str) -> Dict[str, Any]:
        """
        Main method: Use YAML prompt template + project context to let LLM generate documentation.
        LLM returns structured documentation data, we execute file creation with MCP tools.
        """
        try:
            # Get main prompt from YAML template (or fallback)
            main_prompt = await self._get_main_prompt(task_input, project_context)
            
            # Let LLM run the show
            self.logger.info("Generating documentation with LLM...")
            response = await self.llm_provider.generate(
                prompt=main_prompt,
                max_tokens=8000,
                temperature=0.1
            )
            
            if not response:
                return {"success": False, "error": "No response from LLM"}
            
            # Try to parse structured response (LLM should return JSON with documentation data)
            try:
                documentation_data = json.loads(response)
                self.logger.info(f"LLM provided documentation with {len(documentation_data.get('documents', []))} documents")
                
                # Create documentation files using MCP tools
                documentation_files = await self._create_documentation_files(documentation_data, task_input)
                
                return {
                    "success": True,
                    "documentation_data": documentation_data,
                    "documentation_files": documentation_files,
                    "confidence": documentation_data.get("confidence", 0.8),
                    "generation_summary": f"Generated {len(documentation_files)} documentation files",
                    "message": f"Generated documentation with {len(documentation_files)} files",
                    "llm_response": response
                }
                
            except json.JSONDecodeError:
                # Fallback: treat response as markdown documentation
                self.logger.info("LLM response not JSON, treating as markdown documentation")
                return await self._fallback_markdown_documentation(response, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM documentation generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_main_prompt(self, task_input: ProjectDocumentationInput, project_context: str) -> str:
        """
        Get the main prompt template from YAML and inject project context.
        Falls back to built-in prompt if YAML not available.
        """
        try:
            # Try to get prompt from YAML template
            prompt_template = self.prompt_manager.get_prompt_definition(
                "project_documentation_agent_v1_prompt",  # prompt_name from YAML id field
                "1.0.0",  # prompt_version from YAML version field
                sub_path="autonomous_engine"  # subdirectory where prompt is located
            )
            
            # Use LLMProvider to render the prompt with variables
            rendered_prompt = await self.llm_provider.generate_response_with_prompt_id(
                prompt_template.id,
                {
                    "user_goal": task_input.user_goal or "Generate project documentation",
                    "project_path": task_input.project_path,
                    "project_context": project_context,
                    "project_id": task_input.project_id,
                    "include_api_docs": task_input.include_api_docs,
                    "include_user_guide": task_input.include_user_guide,
                    "include_dependency_audit": task_input.include_dependency_audit
                }
            )
            self.logger.info("Using YAML prompt template")
            return rendered_prompt
            
        except Exception as e:
            self.logger.warning(f"Could not load YAML prompt template: {e}, using built-in fallback")
            
            # Built-in fallback prompt
            return f"""You are a project documentation generation agent. Create comprehensive documentation for this project.

PROJECT CONTEXT:
{project_context}

INSTRUCTIONS:
1. Analyze the project structure, code, and purpose
2. Generate appropriate documentation for the project type
3. Create clear, professional documentation that helps users understand and use the project
4. Return a JSON response with this exact structure:

{{
    "project_title": "descriptive project title",
    "project_description": "clear description of what this project does",
    "documents": [
        {{
            "filename": "README.md",
            "content": "comprehensive README content with setup, usage, etc.",
            "type": "readme"
        }},
        {{
            "filename": "API_DOCS.md", 
            "content": "API documentation if applicable",
            "type": "api_docs"
        }},
        {{
            "filename": "USER_GUIDE.md",
            "content": "user guide with examples and tutorials",
            "type": "user_guide"
        }},
        {{
            "filename": "DEPENDENCIES.md",
            "content": "dependency audit and security information",
            "type": "dependency_audit"
        }}
    ],
    "confidence": 0.85,
    "reasoning": "explanation of documentation choices"
}}

REQUIREMENTS:
- Create documentation appropriate for the project type and complexity
- Include clear setup and usage instructions
- Document APIs, functions, and key features
- Provide examples where helpful
- Include dependency information and security considerations
- Use professional, clear writing style
- Format with proper Markdown syntax

USER GOAL: {task_input.user_goal or "Generate project documentation"}
PROJECT: {task_input.project_id}

Return ONLY the JSON response, no additional text."""

    async def _create_documentation_files(self, documentation_data: Dict[str, Any], task_input: ProjectDocumentationInput) -> Dict[str, str]:
        """
        Create documentation files using MCP tools.
        """
        self.logger.info("Creating documentation files")
        
        created_files = {}
        project_path = task_input.project_path
        
        try:
            documents = documentation_data.get("documents", [])
            
            # Create docs directory if it doesn't exist
            docs_dir_result = await self._call_mcp_tool("filesystem_create_directory", {
                "directory_path": f"{project_path}/docs",
                "create_parents": True
            })
            
            for doc in documents:
                filename = doc.get("filename", "document.md")
                content = doc.get("content", "")
                doc_type = doc.get("type", "document")
                
                # Determine file path based on document type
                if doc_type == "readme":
                    file_path = f"{project_path}/{filename}"
                else:
                    file_path = f"{project_path}/docs/{filename}"
                
                # Write file using MCP tools
                write_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": file_path,
                    "content": content
                })
                
                if write_result.get("success"):
                    relative_path = filename if doc_type == "readme" else f"docs/{filename}"
                    created_files[relative_path] = content
                    self.logger.info(f"✓ Created {relative_path}")
                else:
                    self.logger.error(f"Failed to create {filename}: {write_result.get('error', 'unknown error')}")
            
            self.logger.info(f"Successfully created {len(created_files)} documentation files")
            
        except Exception as e:
            self.logger.error(f"Failed to create documentation files: {e}")
            created_files["error"] = f"Documentation file creation failed: {str(e)}"
        
        return created_files

    async def _fallback_markdown_documentation(self, response: str, task_input: ProjectDocumentationInput) -> Dict[str, Any]:
        """
        Fallback when LLM doesn't return structured JSON - treat response as README content.
        """
        try:
            # Clean up response
            content = response.strip()
            
            # Use a proper project name based on user goal, not UUID
            if task_input.user_goal:
                # Convert user goal to a safe filename
                project_name = "".join(c for c in task_input.user_goal.lower().replace(" ", "_") if c.isalnum() or c == "_")[:30]
                if not project_name:
                    project_name = "project"
            else:
                project_name = "project"
                
            filename = "README.md"
            
            full_path = Path(task_input.project_path) / filename
            
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": str(full_path),
                "content": content
            })
            
            if write_result.get("success"):
                self.logger.info(f"✓ Generated fallback documentation: {filename}")
                return {
                    "success": True,
                    "documentation_files": {filename: content},
                    "confidence": 0.7,
                    "generation_summary": "Generated documentation (fallback format)",
                    "message": "Generated documentation (fallback format)"
                }
            else:
                return {"success": False, "error": "Failed to write fallback documentation"}
                
        except Exception as e:
            return {"success": False, "error": f"Fallback documentation generation failed: {str(e)}"}

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = ProjectDocumentationInput.model_json_schema()
        output_schema = ProjectDocumentationOutput.model_json_schema()
        
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
                "generates_documentation": True,
                "simple_documentation": True,
                "llm_powered": True,
                "mcp_tools_integrated": True,
                "creates_multiple_docs": True
            },
            metadata={
                "callable_fn_path": f"{ProjectDocumentationAgent_v1.__module__}.{ProjectDocumentationAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[ProjectDocumentationInput]:
        return ProjectDocumentationInput

    def get_output_schema(self) -> Type[ProjectDocumentationOutput]:
        return ProjectDocumentationOutput 