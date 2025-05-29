"""
ProjectSetupAgent_v1: Meta/Setup work consolidation.

This agent handles ALL project meta and setup tasks:
- Documentation generation  
- Dependency management
- Environment setup

Just different prompt templates for each capability. Same MCP + LLM pattern.
No more 3 agents doing the same thing with different prompts!
"""

from __future__ import annotations

import logging
import datetime
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type, Literal
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.agent_registry import AgentCard, AgentCategory, AgentVisibility
from chungoid.schemas.common import ConfidenceScore
from chungoid.schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
    AgentOutput
)
from chungoid.registry import register_system_agent

logger = logging.getLogger(__name__)

class ProjectSetupInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this setup task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core capability selection
    capability: Literal["documentation", "dependencies", "environment"] = Field(
        ..., description="Which setup capability to execute"
    )
    
    # Core fields - simplified
    user_goal: Optional[str] = Field(None, description="What the user wants to build")
    project_path: Optional[str] = Field(None, description="Where to build it")
    
    # Traditional fields for backward compatibility
    task_description: Optional[str] = Field(None, description="Specific task description")
    
    # Context fields from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    
    # Capability-specific options
    include_api_docs: bool = Field(default=True, description="For documentation: include API docs")
    include_user_guide: bool = Field(default=True, description="For documentation: include user guide")
    include_dependency_audit: bool = Field(default=True, description="For documentation: include dependency audit")
    install_dependencies: bool = Field(default=True, description="For environment/dependencies: install packages")
    force_recreate: bool = Field(default=False, description="For environment: force recreation")
    auto_detect_dependencies: bool = Field(default=True, description="For dependencies: auto-detect packages")
    update_existing: bool = Field(default=False, description="For dependencies: update existing")
    include_dev_dependencies: bool = Field(default=True, description="For dependencies: include dev deps")
    resolve_conflicts: bool = Field(default=True, description="For dependencies: resolve conflicts")

    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'ProjectSetupInput':
        if not self.intelligent_context:
            if not self.user_goal and not self.task_description:
                raise ValueError("Either user_goal or task_description must be provided when not using intelligent context")
            if not self.project_path:
                raise ValueError("project_path must be provided when not using intelligent context")
        return self

class ProjectSetupOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    capability: str = Field(..., description="Capability that was executed.")
    status: str = Field(..., description="Status of setup task (SUCCESS, FAILURE, etc.).")
    
    # Universal outputs
    files_created: Dict[str, str] = Field(default_factory=dict, description="Files created {file_path: content}")
    commands_executed: List[str] = Field(default_factory=list, description="Commands executed")
    
    # Results
    setup_summary: str = Field(..., description="Summary of what was accomplished")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for next steps")
    
    # Status and metadata
    message: str = Field(..., description="Message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the results.")
    error_message: Optional[str] = Field(None, description="Error message if task failed.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Metadata about the setup process.")

@register_system_agent(capabilities=["documentation_generation", "dependency_management", "environment_setup"])
class ProjectSetupAgent_v1(UnifiedAgent):
    """
    PROJECT SETUP AGENT - META/SETUP WORK!
    
    Consolidated agent that handles ALL project meta and setup work:
    - Documentation generation (replaces ProjectDocumentationAgent)
    - Dependency management (replaces DependencyManagementAgent)
    - Environment setup (replaces EnvironmentBootstrapAgent)
    
    Same MCP + LLM pattern, just different prompt templates per capability.
    Clean separation: SETUP → DESIGN → EXECUTION.
    """
    
    AGENT_ID: ClassVar[str] = "ProjectSetupAgent_v1"
    AGENT_NAME: ClassVar[str] = "Project Setup Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Consolidated agent for all project setup and meta tasks with MCP tools."
    AGENT_VERSION: ClassVar[str] = "1.0.0"  # New consolidated version
    CAPABILITIES: ClassVar[List[str]] = ["documentation_generation", "dependency_management", "environment_setup"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ProjectSetupInput]] = ProjectSetupInput
    OUTPUT_SCHEMA: ClassVar[Type[ProjectSetupOutput]] = ProjectSetupOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_project_setup"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["mcp_file_operations"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']

    # Capability-specific prompt templates
    PROMPT_TEMPLATES: ClassVar[Dict[str, str]] = {
        "documentation": "project_setup_documentation_prompt",
        "dependencies": "project_setup_dependencies_prompt",
        "environment": "project_setup_environment_prompt"
    }

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Convert Pydantic objects to dictionaries for JSON serialization."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self._convert_pydantic_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_pydantic_to_dict(v) for v in obj]
        else:
            return obj

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """Execute one iteration of project setup."""
        try:
            # Convert context inputs to our input model (like other agents do)
            context_dict = self._convert_pydantic_to_dict(context.inputs)
            task_input = ProjectSetupInput(**context_dict)
            
            self.logger.info(f"ProjectSetupAgent executing {task_input.capability} capability...")

            # Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            # Execute the specific capability with LLM
            result = await self._execute_capability_with_llm(task_input, project_context)
            
            # Execute the setup plan 
            execution_result = await self._execute_setup_plan(result, task_input)
            
            # Create the output
            output = ProjectSetupOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                capability=task_input.capability,
                status="SUCCESS",
                files_created=execution_result.get('files_created', {}),
                commands_executed=execution_result.get('commands_executed', []),
                setup_summary=execution_result.get('setup_summary', f"{task_input.capability.title()} setup completed successfully"),
                recommendations=execution_result.get('recommendations', []),
                message=f"{task_input.capability.title()} setup completed successfully",
                confidence_score=ConfidenceScore(
                    value=0.9,
                    method="agent_assessment",
                    explanation="High confidence in setup completion based on successful task execution"
                ),
                usage_metadata=execution_result.get('usage_metadata', {})
            )

            return IterationResult(
                output=output,
                quality_score=execution_result.get('confidence', 0.85),
                tools_used=execution_result.get('tools_used', []),
                protocol_used="unified_project_setup",
                iteration_metadata={
                    "capability": task_input.capability,
                    "files_created": execution_result.get('files_created', {}),
                    "commands_executed": execution_result.get('commands_executed', [])
                }
            )

        except Exception as e:
            self.logger.error(f"Error in ProjectSetupAgent execution: {str(e)}")
            error_output = ProjectSetupOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())),
                project_id=getattr(task_input, 'project_id', str(uuid.uuid4())),
                capability=getattr(task_input, 'capability', 'unknown'),
                status="FAILURE",
                setup_summary=f"Failed to complete {getattr(task_input, 'capability', 'setup')} task",
                message=f"Setup failed: {str(e)}",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.1,
                tools_used=[],
                protocol_used="unified_project_setup",
                iteration_metadata={
                    "capability": getattr(task_input, 'capability', 'unknown'),
                    "error": str(e)
                }
            )

    async def _gather_project_context(self, task_input: ProjectSetupInput) -> str:
        """Gather comprehensive project context using MCP tools."""
        try:
            self.logger.info("Gathering project context...")
            context_parts = []
            
            # Add user goal and basic info
            if task_input.user_goal:
                context_parts.append(f"User Goal: {task_input.user_goal}")
            if task_input.task_description:
                context_parts.append(f"Task Description: {task_input.task_description}")
            if task_input.project_path:
                context_parts.append(f"Project Path: {task_input.project_path}")

            # Try to read project structure if path exists
            if task_input.project_path:
                try:
                    # Get directory listing using _call_mcp_tool
                    dir_result = await self._call_mcp_tool("filesystem_list_directory", {
                        "directory_path": task_input.project_path
                    })
                    if dir_result and dir_result.get('success'):
                        context_parts.append(f"Project Structure:\n{dir_result.get('result', '')}")
                    
                    # Try to read key files (README, requirements, etc.)
                    key_files = ['README.md', 'README.txt', 'requirements.txt', 'package.json', 'pyproject.toml']
                    for filename in key_files:
                        try:
                            file_path = f"{task_input.project_path}/{filename}"
                            file_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": file_path
                            })
                            if file_result and file_result.get('success'):
                                content = file_result.get('result', '')
                                if isinstance(content, dict) and 'content' in content:
                                    content = content['content']
                                context_parts.append(f"{filename}:\n{str(content)[:1000]}...")  # Limit size
                        except Exception:
                            continue  # File doesn't exist, skip
                            
                except Exception as e:
                    self.logger.warning(f"Could not read project structure: {e}")
                    context_parts.append("Project structure: Unable to read (may be new project)")

            # Add intelligent context if available
            if task_input.intelligent_context and task_input.project_specifications:
                context_parts.append(f"Project Specifications: {json.dumps(task_input.project_specifications, indent=2)}")

            return "\n\n".join(context_parts)

        except Exception as e:
            self.logger.warning(f"Error gathering project context: {e}")
            return f"Basic context: {task_input.user_goal or 'Setup task'}"

    async def _execute_capability_with_llm(self, task_input: ProjectSetupInput, project_context: str) -> Dict[str, Any]:
        """Execute the specific capability using LLM with appropriate prompt."""
        try:
            # Get capability-specific prompt
            prompt = await self._get_capability_prompt(task_input, project_context)
            
            self.logger.info(f"Executing {task_input.capability} with LLM...")
            
            # Call LLM using the correct method name
            response = await self.llm_provider.generate(prompt)
            
            # Parse JSON response
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
                
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                return json.loads(json_str)
            else:
                # Fallback execution if no valid JSON
                return await self._fallback_execution(response_text, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM execution failed: {e}")
            return await self._fallback_execution("LLM failed", task_input)

    async def _get_capability_prompt(self, task_input: ProjectSetupInput, project_context: str) -> str:
        """Get the capability-specific prompt template."""
        try:
            # Get the prompt template name for this capability
            template_name = self.PROMPT_TEMPLATES.get(task_input.capability)
            if not template_name:
                return self._get_universal_fallback_prompt(task_input, project_context)
            
            # Try to load from prompt manager
            try:
                prompt_template = self.prompt_manager.get_prompt_definition(
                    template_name, 
                    "1.0.0",  # version
                    sub_path="autonomous_engine"  # subdirectory
                )
                
                # Prepare context variables for template rendering
                template_vars = {
                    "user_goal": task_input.user_goal or "Project setup",
                    "project_path": task_input.project_path or "/unknown",
                    "project_id": task_input.project_id,
                    "project_context": project_context,
                    "include_api_docs": task_input.include_api_docs,
                    "include_user_guide": task_input.include_user_guide,
                    "include_dependency_audit": task_input.include_dependency_audit,
                    "install_dependencies": task_input.install_dependencies,
                    "force_recreate": task_input.force_recreate,
                    "auto_detect_dependencies": task_input.auto_detect_dependencies,
                    "update_existing": task_input.update_existing,
                    "include_dev_dependencies": task_input.include_dev_dependencies,
                    "resolve_conflicts": task_input.resolve_conflicts
                }
                
                # Render the user prompt template with variables
                rendered_prompt = self.prompt_manager.get_rendered_prompt_template(
                    prompt_template.user_prompt_template, 
                    template_vars
                )
                
                self.logger.info(f"Using YAML prompt template: {template_name}")
                return rendered_prompt
            except Exception:
                return self._get_universal_fallback_prompt(task_input, project_context)
                
        except Exception as e:
            self.logger.warning(f"Could not load prompt template {template_name}: {e}")
            return self._get_universal_fallback_prompt(task_input, project_context)

    def _get_universal_fallback_prompt(self, task_input: ProjectSetupInput, project_context: str) -> str:
        """Universal fallback prompt for any capability."""
        return f"""You are a project setup specialist. Execute the {task_input.capability} capability for this project.

USER GOAL: {task_input.user_goal or 'Project setup'}
PROJECT PATH: {task_input.project_path or '/unknown'}
CAPABILITY: {task_input.capability}

PROJECT CONTEXT:
{project_context}

CAPABILITY-SPECIFIC OPTIONS:
{json.dumps({
    'include_api_docs': task_input.include_api_docs,
    'include_user_guide': task_input.include_user_guide,
    'include_dependency_audit': task_input.include_dependency_audit,
    'install_dependencies': task_input.install_dependencies,
    'force_recreate': task_input.force_recreate,
    'auto_detect_dependencies': task_input.auto_detect_dependencies,
    'update_existing': task_input.update_existing,
    'include_dev_dependencies': task_input.include_dev_dependencies,
    'resolve_conflicts': task_input.resolve_conflicts
}, indent=2)}

INSTRUCTIONS:
1. Analyze the project context and user goal
2. Create a {task_input.capability} plan appropriate for this project
3. Generate necessary files and commands
4. Provide recommendations

RESPONSE FORMAT:
Return a JSON response with this structure:
{{
    "capability": "{task_input.capability}",
    "plan_summary": "Brief description of the plan",
    "files_to_create": [
        {{
            "path": "file/path",
            "content": "file content",
            "description": "what this file does"
        }}
    ],
    "commands_to_execute": [
        {{
            "command": "command to run",
            "description": "what this command does",
            "working_directory": "{task_input.project_path or '/unknown'}"
        }}
    ],
    "recommendations": ["recommendation 1", "recommendation 2"],
    "confidence": 0.85,
    "reasoning": "explanation of decisions made"
}}

Generate a comprehensive {task_input.capability} plan for this project.
Return ONLY the JSON response, no additional text."""

    async def _execute_setup_plan(self, plan: Dict[str, Any], task_input: ProjectSetupInput) -> Dict[str, Any]:
        """Execute the setup plan by creating files and running commands."""
        try:
            execution_result = {
                'files_created': {},
                'commands_executed': [],
                'setup_summary': plan.get('plan_summary', f'{task_input.capability} setup completed'),
                'recommendations': plan.get('recommendations', []),
                'confidence': plan.get('confidence', 0.85),
                'usage_metadata': {}
            }
            
            # Create files
            files_to_create = plan.get('files_to_create', [])
            for file_info in files_to_create:
                if isinstance(file_info, dict) and 'path' in file_info and 'content' in file_info:
                    try:
                        file_path = file_info['path']
                        content = file_info['content']
                        
                        # Use MCP to create file using _call_mcp_tool
                        result = await self._call_mcp_tool("filesystem_write_file", {
                            "file_path": file_path,
                            "content": content
                        })
                        
                        execution_result['files_created'][file_path] = content
                        self.logger.info(f"Created file: {file_path}")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not create file {file_info.get('path', 'unknown')}: {e}")
            
            # Execute commands  
            commands_to_execute = plan.get('commands_to_execute', [])
            for cmd_info in commands_to_execute:
                if isinstance(cmd_info, dict) and 'command' in cmd_info:
                    try:
                        command = cmd_info['command']
                        working_dir = cmd_info.get('working_directory', task_input.project_path)
                        
                        # Use MCP to execute command using _call_mcp_tool 
                        result = await self._call_mcp_tool("terminal_execute_command", {
                            "command": command,
                            "working_directory": working_dir
                        })
                        
                        execution_result['commands_executed'].append(command)
                        self.logger.info(f"Executed command: {command}")
                        
                    except Exception as e:
                        self.logger.warning(f"Could not execute command {cmd_info.get('command', 'unknown')}: {e}")
            
            return execution_result
            
        except Exception as e:
            self.logger.error(f"Plan execution failed: {e}")
            return await self._fallback_execution_result(str(e), task_input)

    async def _fallback_execution(self, response: str, task_input: ProjectSetupInput) -> Dict[str, Any]:
        """Fallback execution when LLM response parsing fails."""
        return {
            "capability": task_input.capability,
            "plan_summary": f"Fallback {task_input.capability} setup",
            "files_to_create": [],
            "commands_to_execute": [],
            "recommendations": [f"Review {task_input.capability} setup manually"],
            "confidence": 0.5,
            "reasoning": f"Fallback execution due to parsing issues: {response[:100]}..."
        }

    async def _fallback_execution_result(self, error: str, task_input: ProjectSetupInput) -> Dict[str, Any]:
        """Fallback execution result when plan execution fails."""
        return {
            'files_created': {},
            'commands_executed': [],
            'setup_summary': f'Failed to complete {task_input.capability} setup: {error}',
            'recommendations': [f'Manually review {task_input.capability} setup'],
            'confidence': 0.3,
            'usage_metadata': {'error': error}
        }

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Return the agent card for this agent."""
        return AgentCard(
            agent_id=ProjectSetupAgent_v1.AGENT_ID,
            name=ProjectSetupAgent_v1.AGENT_NAME,
            description=ProjectSetupAgent_v1.AGENT_DESCRIPTION,
            version=ProjectSetupAgent_v1.AGENT_VERSION,
            category=ProjectSetupAgent_v1.CATEGORY,
            visibility=ProjectSetupAgent_v1.VISIBILITY,
            capabilities=ProjectSetupAgent_v1.CAPABILITIES,
            input_schema=ProjectSetupAgent_v1.INPUT_SCHEMA.model_json_schema(),
            output_schema=ProjectSetupAgent_v1.OUTPUT_SCHEMA.model_json_schema(),
            protocols=ProjectSetupAgent_v1.PRIMARY_PROTOCOLS + ProjectSetupAgent_v1.SECONDARY_PROTOCOLS
        )

    def get_input_schema(self) -> Type[ProjectSetupInput]:
        return ProjectSetupInput

    def get_output_schema(self) -> Type[ProjectSetupOutput]:
        return ProjectSetupOutput 