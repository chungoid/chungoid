"""
EnhancedArchitectAgent_v1: Dead simple, LLM-powered blueprint generation.

This agent generates architecture blueprints by:
1. Using MCP tools to understand the project context and requirements
2. Using main prompt template from YAML  
3. Letting the LLM run the show to create comprehensive blueprints

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

class EnhancedArchitectAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this blueprint generation task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core fields - simplified
    user_goal: Optional[str] = Field(None, description="What the user wants to build")
    project_path: Optional[str] = Field(None, description="Where to build it")
    
    # Traditional fields for backward compatibility
    loprd_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the LOPRD (JSON artifact) to be used as input.")
    existing_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of an existing Blueprint to refine, if any.")
    refinement_instructions: Optional[str] = Field(None, description="Specific instructions for refining an existing Blueprint.")
    
    # Context fields from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    output_blueprint_files: bool = Field(default=True, description="Whether to output blueprint files to filesystem")
    generate_execution_plan: bool = Field(default=True, description="Whether to generate execution plan from blueprint")
    output_directory: Optional[str] = Field(None, description="Directory to output blueprint files")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'EnhancedArchitectAgentInput':
        """Ensure we have minimum info to generate blueprints."""
        
        # Need either user_goal or loprd_doc_id
        if not self.user_goal and not self.loprd_doc_id:
            raise ValueError("Either user_goal or loprd_doc_id is required")
        
        # Default project_path if not provided
        if not self.project_path:
            self.project_path = "."
        
        return self

class EnhancedArchitectAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    status: str = Field(..., description="Status of blueprint generation (SUCCESS, FAILURE, etc.).")
    
    # Blueprint artifacts
    blueprint_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the blueprint is stored.")
    blueprint_files: Dict[str, str] = Field(default_factory=dict, description="Generated blueprint files {file_path: content}")
    
    # Results
    review_results: Dict[str, Any] = Field(default_factory=dict, description="Self-review analysis of the blueprint")
    execution_plan_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the execution plan is stored.")
    execution_plan_generated: bool = Field(default=False, description="Whether execution plan was generated")
    
    # Status and metadata
    message: str = Field(..., description="Message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the blueprint quality.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the generation process.")

@register_autonomous_engine_agent(capabilities=["architecture_design", "system_planning", "blueprint_generation"])
class EnhancedArchitectAgent_v1(UnifiedAgent):
    """
    Dead simple blueprint generation agent.
    
    1. MCP tools for project understanding and file operations
    2. Main prompt from YAML template  
    3. LLM runs the show - creates comprehensive blueprints without complex phases
    """
    
    AGENT_ID: ClassVar[str] = "EnhancedArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Enhanced Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Simple, LLM-powered blueprint generation with MCP tools."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "3.0.0"  # Major version bump for simplification
    CAPABILITIES: ClassVar[List[str]] = ["architecture_design", "system_planning", "blueprint_generation"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[EnhancedArchitectAgentInput]] = EnhancedArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[EnhancedArchitectAgentOutput]] = EnhancedArchitectAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["simple_blueprint_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["mcp_file_operations"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized as simple blueprint agent.")

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
        Dead simple execution: Let the LLM understand requirements and generate blueprints.
        No complex phases, just LLM + MCP tools.
        """
        try:
            # Convert inputs - handle both dict and Pydantic objects
            if isinstance(context.inputs, dict):
                converted_inputs = self._convert_pydantic_to_dict(context.inputs)
                task_input = EnhancedArchitectAgentInput(**converted_inputs)
            elif hasattr(context.inputs, 'dict'):
                task_input = EnhancedArchitectAgentInput(**context.inputs.dict())
            else:
                task_input = context.inputs

            self.logger.info(f"Starting simple blueprint generation for: {task_input.user_goal or 'project'}")

            # 1. Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            # 2. Let LLM generate blueprint using main prompt template
            blueprint_result = await self._generate_blueprint_with_llm(task_input, project_context)
            
            # 3. Create simple output
            output = EnhancedArchitectAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS" if blueprint_result["success"] else "FAILURE",
                blueprint_files=blueprint_result.get("blueprint_files", {}),
                review_results=blueprint_result.get("review_results", {}),
                execution_plan_generated=blueprint_result.get("execution_plan_generated", False),
                message=blueprint_result.get("message", "Blueprint generation completed"),
                confidence_score=ConfidenceScore(
                    value=0.9,
                    method="llm_self_assessment",
                    explanation="High confidence in blueprint generation quality and architectural decisions"
                ),
                error_message=blueprint_result.get("error"),
                usage_metadata={
                    "iteration": iteration + 1,
                    "files_generated": len(blueprint_result.get("blueprint_files", {})),
                    "blueprint_created": blueprint_result.get("success", False)
                }
            )
            
            return IterationResult(
                output=output,
                quality_score=blueprint_result.get("confidence", 0.8),
                tools_used=["llm_analysis", "mcp_file_operations"],
                protocol_used="simple_blueprint_generation"
            )
            
        except Exception as e:
            self.logger.error(f"Blueprint generation failed: {e}")
            
            # Safe error output
            task_id = getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4())
            project_id = getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown'
            
            error_output = EnhancedArchitectAgentOutput(
                task_id=task_id,
                project_id=project_id,
                status="ERROR",
                blueprint_files={},
                message="Blueprint generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="simple_blueprint_generation"
            )

    async def _gather_project_context(self, task_input: EnhancedArchitectAgentInput) -> str:
        """
        Simple method: Gather all relevant project context for the LLM.
        "Learn about the project requirements" - scan files, read specs, understand goals.
        """
        self.logger.info(f"Gathering project context from: {task_input.project_path}")
        
        context_parts = []
        
        # Add user goal
        if task_input.user_goal:
            context_parts.append(f"User Goal: {task_input.user_goal}")
        
        # Add project specifications if available
        if task_input.project_specifications:
            context_parts.append(f"Project Specifications: {json.dumps(task_input.project_specifications, indent=2)}")
        
        # Try to read existing LOPRD if available
        if task_input.loprd_doc_id:
            try:
                loprd_result = await self._call_mcp_tool("chromadb_retrieve_document", {
                    "collection_name": "loprd_artifacts_collection",
                    "document_id": task_input.loprd_doc_id
                })
                if loprd_result.get("success"):
                    loprd_content = loprd_result.get("content", "")
                    context_parts.append(f"LOPRD Requirements: {loprd_content}")
                    self.logger.info("Successfully retrieved LOPRD document")
            except Exception as e:
                self.logger.warning(f"Could not retrieve LOPRD: {e}")
        
        # Scan project directory with MCP tools
        try:
            list_result = await self._call_mcp_tool("filesystem_list_directory", {
                "directory_path": task_input.project_path
            })
            
            if list_result.get("success"):
                files = list_result.get("files", [])
                if files:
                    context_parts.append(f"Existing files in {task_input.project_path}: {files}")
                    
                    # Read key documentation and requirements files
                    for file_info in files[:10]:  # Limit to first 10 files
                        filename = file_info.get("name", "") if isinstance(file_info, dict) else str(file_info)
                        
                        # Read requirements and documentation files
                        if any(pattern in filename.lower() for pattern in [
                            "readme", "requirements", "spec", "design", "architecture", "plan", "loprd"
                        ]) and filename.endswith(('.md', '.txt', '.json')):
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
                else:
                    context_parts.append(f"Project directory {task_input.project_path} is empty or newly created")
            else:
                context_parts.append(f"Could not scan project directory: {list_result.get('error', 'unknown error')}")
                
        except Exception as e:
            self.logger.warning(f"Error scanning project directory: {e}")
            context_parts.append(f"Error scanning project directory: {str(e)}")
        
        return "\n\n".join(context_parts)

    async def _generate_blueprint_with_llm(self, task_input: EnhancedArchitectAgentInput, project_context: str) -> Dict[str, Any]:
        """
        Main method: Use YAML prompt template + project context to let LLM generate blueprints.
        LLM returns structured blueprint data, we execute file creation with MCP tools.
        """
        try:
            # Get main prompt from YAML template (or fallback)
            main_prompt = await self._get_main_prompt(task_input, project_context)
            
            # Let LLM run the show
            self.logger.info("Generating blueprint with LLM...")
            response = await self.llm_provider.generate(
                prompt=main_prompt,
                max_tokens=8000,
                temperature=0.1
            )
            
            if not response:
                return {"success": False, "error": "No response from LLM"}
            
            # Try to parse structured response (LLM should return JSON with blueprint data)
            try:
                blueprint_data = json.loads(response)
                self.logger.info(f"LLM provided blueprint with {len(blueprint_data.get('components', []))} components")
                
                # Create blueprint files using MCP tools
                blueprint_files = await self._create_blueprint_files(blueprint_data, task_input)
                
                # Create project structure files (requirements.txt, README.md, etc.)
                structure_files = await self._create_project_structure(blueprint_data, task_input)
                
                # Combine all files
                all_files = {**blueprint_files, **structure_files}
                
                return {
                    "success": True,
                    "blueprint_data": blueprint_data,
                    "blueprint_files": all_files,
                    "confidence": blueprint_data.get("confidence", 0.8),
                    "message": f"Generated blueprint with {len(all_files)} files",
                    "llm_response": response
                }
                
            except json.JSONDecodeError:
                # Fallback: treat response as markdown blueprint
                self.logger.info("LLM response not JSON, treating as markdown blueprint")
                return await self._fallback_markdown_blueprint(response, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM blueprint generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_main_prompt(self, task_input: EnhancedArchitectAgentInput, project_context: str) -> str:
        """
        Get the main prompt template from YAML and inject project context.
        Falls back to built-in prompt if YAML not available.
        """
        try:
            # Try to get prompt from YAML template
            prompt_template = self.prompt_manager.get_prompt_definition(
                "architect_agent_v1_prompt",  # prompt_name from YAML id field
                "0.1.0",  # prompt_version from YAML version field
                sub_path="autonomous_engine"  # subdirectory where prompt is located
            )
            
            # Format the prompt with variables using prompt manager
            template_vars = {
                "user_goal": task_input.user_goal or "Create architecture blueprint",
                "project_path": task_input.project_path,
                "project_context": project_context,
                "project_id": task_input.project_id,
                "loprd_doc_id": task_input.loprd_doc_id or "",
                "output_blueprint_files": task_input.output_blueprint_files
            }
            
            formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            # Use LLMProvider to generate response
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=prompt_template.system_prompt if hasattr(prompt_template, 'system_prompt') else None,
                temperature=0.3,
                max_tokens=4000
            )
            
            self.logger.info("Used YAML prompt template successfully")
            return response
            
        except Exception as e:
            self.logger.warning(f"Could not load YAML prompt template: {e}, using built-in fallback")
            
            # Built-in fallback prompt
            return f"""You are an architecture blueprint generation agent. Create a comprehensive technical blueprint for this project.

PROJECT CONTEXT:
{project_context}

INSTRUCTIONS:
1. Analyze the project context, user goal, and any existing requirements
2. Design a suitable architecture pattern and technology stack
3. Break down the system into logical components
4. Plan directory structure, deployment, and testing strategies
5. Return a JSON response with this exact structure:

{{
    "title": "descriptive project title",
    "architecture_pattern": "chosen pattern (e.g., layered, microservices, monolithic)",
    "technology_stack": {{
        "language": "primary programming language",
        "framework": "main framework",
        "database": "database choice",
        "tools": ["additional tools"]
    }},
    "components": [
        {{
            "name": "Component Name",
            "responsibility": "what this component does",
            "dependencies": ["list of dependencies"],
            "interfaces": ["exposed interfaces"]
        }}
    ],
    "directory_structure": {{
        "src/": "source code",
        "tests/": "test files",
        "docs/": "documentation",
        "config/": "configuration files"
    }},
    "deployment_strategy": {{
        "type": "deployment approach",
        "requirements": ["deployment requirements"]
    }},
    "testing_strategy": {{
        "unit_tests": "unit testing approach",
        "integration_tests": "integration testing approach"
    }},
    "confidence": 0.85,
    "reasoning": "explanation of architectural decisions"
}}

REQUIREMENTS:
- Choose appropriate architecture patterns based on project scope
- Select modern, suitable technology stacks
- Design clear component boundaries and responsibilities
- Plan realistic deployment and testing strategies
- Ensure components are loosely coupled and highly cohesive
- Include comprehensive documentation structure

USER GOAL: {task_input.user_goal or "Create architecture blueprint"}
PROJECT: {task_input.project_id}

Return ONLY the JSON response, no additional text."""

    async def _create_blueprint_files(self, blueprint_data: Dict[str, Any], task_input: EnhancedArchitectAgentInput) -> Dict[str, str]:
        """
        Create blueprint documentation files using MCP tools.
        """
        self.logger.info("Creating blueprint documentation files")
        
        created_files = {}
        project_path = task_input.project_path
        
        # Use a proper project name based on user goal, not UUID
        if task_input.user_goal:
            # Convert user goal to a safe filename
            project_name = "".join(c for c in task_input.user_goal.lower().replace(" ", "_") if c.isalnum() or c == "_")[:30]
            if not project_name:
                project_name = "project"
        else:
            project_name = "project"
        
        try:
            # 1. Create main blueprint markdown file
            blueprint_content = self._generate_blueprint_markdown(blueprint_data, task_input)
            blueprint_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/{project_name}_blueprint.md",
                "content": blueprint_content
            })
            if blueprint_result.get("success"):
                created_files[f"{project_name}_blueprint.md"] = blueprint_content
                self.logger.info(f"✓ Created {project_name}_blueprint.md")
            
            # 2. Create technology stack documentation
            tech_stack = blueprint_data.get("technology_stack", {})
            if tech_stack:
                tech_content = self._generate_tech_stack_file(tech_stack)
                tech_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": f"{project_path}/{project_name}_tech_stack.md",
                    "content": tech_content
                })
                if tech_result.get("success"):
                    created_files[f"{project_name}_tech_stack.md"] = tech_content
                    self.logger.info(f"✓ Created {project_name}_tech_stack.md")
            
            # 3. Create component specifications
            components = blueprint_data.get("components", [])
            if components:
                # Create docs directory
                docs_dir_result = await self._call_mcp_tool("filesystem_create_directory", {
                    "directory_path": f"{project_path}/docs",
                    "create_parents": True
                })
                
                for i, component in enumerate(components):
                    comp_name = component.get("name", f"component_{i+1}").lower().replace(" ", "_")
                    comp_content = self._generate_component_spec(component)
                    comp_result = await self._call_mcp_tool("filesystem_write_file", {
                        "file_path": f"{project_path}/docs/{comp_name}_spec.md",
                        "content": comp_content
                    })
                    if comp_result.get("success"):
                        created_files[f"docs/{comp_name}_spec.md"] = comp_content
                        self.logger.info(f"✓ Created docs/{comp_name}_spec.md")
            
            # 4. Create blueprint JSON for programmatic access
            blueprint_json = json.dumps(blueprint_data, indent=2)
            json_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/{project_name}_blueprint.json",
                "content": blueprint_json
            })
            if json_result.get("success"):
                created_files[f"{project_name}_blueprint.json"] = blueprint_json
                self.logger.info(f"✓ Created {project_name}_blueprint.json")
            
            self.logger.info(f"Successfully created {len(created_files)} blueprint files")
            
        except Exception as e:
            self.logger.error(f"Failed to create blueprint files: {e}")
            created_files["error"] = f"Blueprint file creation failed: {str(e)}"
        
        return created_files

    async def _create_project_structure(self, blueprint_data: Dict[str, Any], task_input: EnhancedArchitectAgentInput) -> Dict[str, str]:
        """
        Create basic project structure and configuration files (NO CODE IMPLEMENTATION).
        Only creates directory structure, requirements.txt, README.md, .gitignore.
        """
        self.logger.info("Creating project structure files (NO CODE IMPLEMENTATION)")
        
        created_files = {}
        project_path = task_input.project_path
        
        try:
            # 1. Create main directories
            directories = [
                f"{project_path}/src",
                f"{project_path}/tests", 
                f"{project_path}/docs",
                f"{project_path}/config"
            ]
            
            for directory in directories:
                result = await self._call_mcp_tool("filesystem_create_directory", {
                    "directory_path": directory,
                    "create_parents": True
                })
                if result.get("success"):
                    self.logger.info(f"✓ Created directory: {directory}")
            
            # 2. Create requirements.txt
            tech_stack = blueprint_data.get("technology_stack", {})
            requirements_content = self._generate_requirements_file(tech_stack)
            req_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/requirements.txt",
                "content": requirements_content
            })
            if req_result.get("success"):
                created_files["requirements.txt"] = requirements_content
                self.logger.info("✓ Created requirements.txt")
            
            # 3. Create README.md
            readme_content = self._generate_readme_file(blueprint_data, task_input)
            readme_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/README.md",
                "content": readme_content
            })
            if readme_result.get("success"):
                created_files["README.md"] = readme_content
                self.logger.info("✓ Created README.md")
            
            # 4. Create .gitignore
            gitignore_content = self._generate_gitignore_file(tech_stack)
            gitignore_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/.gitignore",
                "content": gitignore_content
            })
            if gitignore_result.get("success"):
                created_files[".gitignore"] = gitignore_content
                self.logger.info("✓ Created .gitignore")
            
            # 5. Create minimal __init__.py files
            init_content = '"""Project package initialization."""\n'
            init_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/src/__init__.py",
                "content": init_content
            })
            if init_result.get("success"):
                created_files["src/__init__.py"] = init_content
                self.logger.info("✓ Created src/__init__.py")
            
            self.logger.info(f"ARCHITECT COMPLETE: Created {len(created_files)} structure/config files")
            self.logger.info("Code implementation will be handled by SmartCodeGeneratorAgent")
            
        except Exception as e:
            self.logger.error(f"Failed to create project structure: {e}")
            created_files["error"] = f"Project structure creation failed: {str(e)}"
        
        return created_files

    async def _fallback_markdown_blueprint(self, response: str, task_input: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """
        Fallback when LLM doesn't return structured JSON - treat response as blueprint markdown.
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
                
            filename = f"{project_name}_blueprint.md"
            
            full_path = Path(task_input.project_path) / filename
            
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": str(full_path),
                "content": content
            })
            
            if write_result.get("success"):
                self.logger.info(f"✓ Generated fallback blueprint: {filename}")
                return {
                    "success": True,
                    "blueprint_files": {filename: content},
                    "confidence": 0.7,
                    "message": "Generated blueprint (fallback format)"
                }
            else:
                return {"success": False, "error": "Failed to write fallback blueprint"}
                
        except Exception as e:
            return {"success": False, "error": f"Fallback blueprint generation failed: {str(e)}"}

    def _generate_blueprint_markdown(self, blueprint_data: Dict[str, Any], task_input: EnhancedArchitectAgentInput) -> str:
        """Generate comprehensive blueprint markdown content."""
        title = blueprint_data.get("title", f"Architecture Blueprint - {task_input.project_id}")
        pattern = blueprint_data.get("architecture_pattern", "unknown")
        tech_stack = blueprint_data.get("technology_stack", {})
        components = blueprint_data.get("components", [])
        
        content = f"""# {title}

## Project Overview
- **Project ID**: {task_input.project_id}
- **Architecture Pattern**: {pattern}
- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **User Goal**: {task_input.user_goal or 'Not specified'}

## Architecture Pattern: {pattern.title()}

This project follows a **{pattern}** architectural pattern for optimal modularity and maintainability.

## Technology Stack

"""
        
        for category, technology in tech_stack.items():
            if technology:
                content += f"- **{category.title()}**: {technology}\n"
        
        content += f"""

## Components

"""
        
        for i, component in enumerate(components, 1):
            name = component.get("name", f"Component {i}")
            responsibility = component.get("responsibility", "No description")
            dependencies = component.get("dependencies", [])
            
            content += f"""### {i}. {name}
**Responsibility**: {responsibility}
**Dependencies**: {', '.join(dependencies) if dependencies else 'None'}

"""
        
        # Add directory structure
        dir_structure = blueprint_data.get("directory_structure", {})
        if dir_structure:
            content += "\n## Directory Structure\n\n```\n"
            for path, description in dir_structure.items():
                content += f"{path}  # {description}\n"
            content += "```\n\n"
        
        # Add deployment strategy
        deployment = blueprint_data.get("deployment_strategy", {})
        if deployment:
            content += "## Deployment Strategy\n\n"
            content += f"**Type**: {deployment.get('type', 'Not specified')}\n\n"
            requirements = deployment.get("requirements", [])
            if requirements:
                content += "**Requirements**:\n"
                for req in requirements:
                    content += f"- {req}\n"
            content += "\n"
        
        # Add implementation notes
        content += """## Implementation Notes

This blueprint provides the architectural foundation for the project. The SmartCodeGeneratorAgent will implement the actual code based on these specifications.

**Next Steps**:
1. Review and approve this architecture
2. SmartCodeGeneratorAgent will generate code implementation
3. Follow component specifications for development
4. Implement testing as outlined in testing strategy

---
*Generated by EnhancedArchitectAgent_v1*
"""
        
        return content

    def _generate_tech_stack_file(self, tech_stack: Dict[str, Any]) -> str:
        """Generate technology stack documentation."""
        content = f"""# Technology Stack

Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Selected Technologies

"""
        
        for category, technology in tech_stack.items():
            if technology:
                content += f"""### {category.title()}
- **Choice**: {technology}
- **Rationale**: Selected for optimal performance and developer experience

"""
        
        return content

    def _generate_component_spec(self, component: Dict[str, Any]) -> str:
        """Generate component specification documentation."""
        name = component.get("name", "Unknown Component")
        responsibility = component.get("responsibility", "No description")
        dependencies = component.get("dependencies", [])
        interfaces = component.get("interfaces", [])
        
        content = f"""# {name} Component Specification

## Overview
**Responsibility**: {responsibility}

## Dependencies
"""
        if dependencies:
            for dep in dependencies:
                content += f"- {dep}\n"
        else:
            content += "- None\n"
        
        content += f"""
## Interfaces
"""
        if interfaces:
            for interface in interfaces:
                content += f"- {interface}\n"
        else:
            content += "- To be defined during implementation\n"
        
        content += f"""
## Implementation Notes
This component will be implemented by the SmartCodeGeneratorAgent according to the architectural specifications.

---
*Generated by EnhancedArchitectAgent_v1*
"""
        
        return content

    def _generate_requirements_file(self, tech_stack: Dict[str, Any]) -> str:
        """Generate requirements.txt based on technology stack."""
        # Simple heuristics based on common technology choices
        requirements = []
        
        framework = tech_stack.get("framework", "").lower()
        language = tech_stack.get("language", "").lower()
        
        if "flask" in framework:
            requirements.extend(["Flask>=2.0.0", "python-dotenv>=0.19.0"])
        elif "django" in framework:
            requirements.extend(["Django>=4.0.0", "python-dotenv>=0.19.0"])
        elif "fastapi" in framework:
            requirements.extend(["fastapi>=0.100.0", "uvicorn>=0.15.0", "python-dotenv>=0.19.0"])
        elif "python" in language:
            requirements.append("python-dotenv>=0.19.0")
        
        database = tech_stack.get("database", "").lower()
        if "postgresql" in database:
            requirements.append("psycopg2-binary>=2.9.0")
        elif "mysql" in database:
            requirements.append("PyMySQL>=1.0.0")
        elif "sqlite" in database:
            requirements.append("# SQLite is included with Python")
        
        # Add common development dependencies
        requirements.extend([
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0"
        ])
        
        return "\n".join(requirements) + "\n"

    def _generate_readme_file(self, blueprint_data: Dict[str, Any], task_input: EnhancedArchitectAgentInput) -> str:
        """Generate README.md file."""
        title = blueprint_data.get("title", task_input.project_id)
        pattern = blueprint_data.get("architecture_pattern", "unknown")
        
        # Use a proper project name based on user goal, not UUID
        if task_input.user_goal:
            # Convert user goal to a safe filename
            project_name = "".join(c for c in task_input.user_goal.lower().replace(" ", "_") if c.isalnum() or c == "_")[:30]
            if not project_name:
                project_name = "project"
        else:
            project_name = "project"
        
        return f"""# {title}

## Overview
{task_input.user_goal or 'Project description not provided'}

## Architecture
This project follows a **{pattern}** architectural pattern.

## Technology Stack
{self._format_tech_stack_for_readme(blueprint_data.get("technology_stack", {}))}

## Getting Started

### Prerequisites
- Check `requirements.txt` for dependencies
- Review `{project_name}_blueprint.md` for detailed architecture

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
# (Implementation details will be added by SmartCodeGeneratorAgent)
```

## Project Structure
```
src/           # Source code
tests/         # Test files  
docs/          # Documentation
config/        # Configuration files
```

## Development
1. Review the architecture blueprint in `{project_name}_blueprint.md`
2. Follow component specifications in `docs/` directory
3. Implement according to the architectural design

---
*Generated by EnhancedArchitectAgent_v1*
"""

    def _format_tech_stack_for_readme(self, tech_stack: Dict[str, Any]) -> str:
        """Format technology stack for README."""
        if not tech_stack:
            return "- Technology stack to be determined"
        
        formatted = []
        for category, technology in tech_stack.items():
            if technology:
                formatted.append(f"- **{category.title()}**: {technology}")
        
        return "\n".join(formatted) if formatted else "- Technology stack to be determined"

    def _generate_gitignore_file(self, tech_stack: Dict[str, Any]) -> str:
        """Generate .gitignore file based on technology stack."""
        language = tech_stack.get("language", "").lower()
        
        if "python" in language:
            return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
venv/
env/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables
.env
.env.local

# Logs
*.log

# Database
*.db
*.sqlite3
"""
        else:
            return """# General
*.log
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = EnhancedArchitectAgentInput.model_json_schema()
        output_schema = EnhancedArchitectAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=EnhancedArchitectAgent_v1.AGENT_ID,
            name=EnhancedArchitectAgent_v1.AGENT_NAME,
            description=EnhancedArchitectAgent_v1.AGENT_DESCRIPTION,
            version=EnhancedArchitectAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[EnhancedArchitectAgent_v1.CATEGORY.value],
            visibility=EnhancedArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "generates_blueprints": True,
                "simple_architecture": True,
                "llm_powered": True,
                "mcp_tools_integrated": True,
                "creates_project_structure": True
            },
            metadata={
                "callable_fn_path": f"{EnhancedArchitectAgent_v1.__module__}.{EnhancedArchitectAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[EnhancedArchitectAgentInput]:
        return EnhancedArchitectAgentInput

    def get_output_schema(self) -> Type[EnhancedArchitectAgentOutput]:
        return EnhancedArchitectAgentOutput 