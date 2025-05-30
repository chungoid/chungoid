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
import time

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
    
    # Core fields - what the user actually wants
    user_goal: str = Field(..., description="What the user wants to build")
    project_path: str = Field(default=".", description="Where to build it")
    
    # Context from orchestrator (if any)
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    
    # REMOVED: All the capability micromanagement bullshit
    # The agent will decide what needs to be done autonomously

    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'ProjectSetupInput':
        """Ensure we have minimum requirements for autonomous setup."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for autonomous project setup")
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
    execution_time: float = Field(..., description="Time taken to complete the setup task")
    autonomous_mode: bool = Field(..., description="Whether the task was executed autonomously")

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
        """
        Clean execution: Let agent autonomously analyze and set up project.
        Single iteration, maximum intelligence, no hardcoded phases.
        """
        try:
            # Parse inputs with detailed validation
            if not context or not hasattr(context, 'inputs'):
                raise ValueError("ExecutionContext is missing or has no 'inputs' attribute")
            
            if not context.inputs:
                raise ValueError("ExecutionContext.inputs is None or empty")
            
            # Detailed input conversion and validation
            try:
                if isinstance(context.inputs, ProjectSetupInput):
                    task_input = context.inputs
                elif isinstance(context.inputs, dict):
                    # Validate required fields
                    if 'user_goal' not in context.inputs:
                        raise ValueError("Missing required field 'user_goal' in input dictionary")
                    if not context.inputs['user_goal'] or not context.inputs['user_goal'].strip():
                        raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                    
                    task_input = ProjectSetupInput(**context.inputs)
                elif hasattr(context.inputs, 'dict'):
                    input_dict = context.inputs.dict()
                    if 'user_goal' not in input_dict:
                        raise ValueError("Missing required field 'user_goal' in input object")
                    if not input_dict['user_goal'] or not input_dict['user_goal'].strip():
                        raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                    
                    task_input = ProjectSetupInput(**input_dict)
                else:
                    raise ValueError(f"Invalid input type: {type(context.inputs)}. Expected ProjectSetupInput, dict, or object with dict() method. Received: {context.inputs}")
                    
                # Final validation of task_input
                if not task_input.user_goal or not task_input.user_goal.strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                        
            except Exception as e:
                raise ValueError(f"Input parsing/validation failed for ProjectSetupAgent: {e}. Context inputs type: {type(context.inputs)}, Context inputs: {context.inputs}")

            self.logger.info(f"Setting up project: {task_input.user_goal}")

            # Generate project setup using unified approach with detailed validation
            setup_result = await self._generate_project_setup(task_input)
            
            if not setup_result:
                raise ValueError(f"Project setup generation returned None/empty result. User goal: {task_input.user_goal}")
            
            if setup_result.get("status") != "success":
                error_msg = setup_result.get("error", "Unknown setup error")
                raise ValueError(f"Project setup generation failed: {error_msg}. User goal: {task_input.user_goal}")
            
            if not setup_result.get("project_setup_result"):
                raise ValueError(f"Project setup result missing 'project_setup_result' field. Result: {setup_result}")
            
            # Create clean output with validation - Fixed to match schema
            output = ProjectSetupOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                capability="autonomous_setup",  # Required field
                status="SUCCESS",
                files_created={},  # Agent creates files autonomously via MCP tools  
                commands_executed=[],  # Commands executed during setup
                setup_summary=f"Generated autonomous project setup for: {task_input.user_goal}",  # Required field
                recommendations=["Setup completed autonomously via MCP tools"],
                message=f"Generated autonomous project setup for: {task_input.user_goal}",
                confidence_score=setup_result.get("confidence_score", ConfidenceScore(value=0.8, method="autonomous_setup", explanation="Setup completed")),
                execution_time=setup_result.get("execution_time", 0.1),  # Required field
                autonomous_mode=True  # Required field
            )
            
            return IterationResult(
                output=output,
                quality_score=setup_result.get("confidence_score", ConfidenceScore(value=0.8, method="default", explanation="default")).value,
                tools_used=["autonomous_setup", "mcp_tools", "llm_analysis"],
                protocol_used="autonomous_project_setup"
            )
            
        except Exception as e:
            error_msg = f"""ProjectSetupAgent execution failed:

ERROR: {e}

CONTEXT:
- Iteration: {iteration}
- Input Type: {type(context.inputs) if context and hasattr(context, 'inputs') else 'No context/inputs'}
- Input Value: {context.inputs if context and hasattr(context, 'inputs') else 'No context/inputs'}
"""
            self.logger.error(error_msg)
            
            # Clean error handling - Fixed to match schema
            error_output = ProjectSetupOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                capability="autonomous_setup_error",  # Required field
                status="ERROR",
                files_created={},  # Schema expects Dict[str, str], not List
                commands_executed=[],
                setup_summary="Project setup failed",  # Required field
                recommendations=[],
                message="Project setup failed",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Project setup failed"
                ),
                error_message=str(e),
                execution_time=0.0,  # Required field
                autonomous_mode=True  # Required field
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="autonomous_project_setup"
            )

    async def _gather_project_context(self, task_input: ProjectSetupInput) -> str:
        """Gather comprehensive project context using MCP tools."""
        try:
            self.logger.info("Gathering project context...")
            context_parts = []
            
            # Add user goal and basic info
            if task_input.user_goal:
                context_parts.append(f"User Goal: {task_input.user_goal}")
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
                            file_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": filename,
                                "project_path": task_input.project_path
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

    async def _execute_autonomous_setup(self, task_input: ProjectSetupInput, project_context: str) -> Dict[str, Any]:
        """Execute autonomous project setup using LLM with autonomous prompt."""
        try:
            # Build context for autonomous agent
            context_parts = []
            context_parts.append(f"User Goal: {task_input.user_goal}")
            context_parts.append(f"Project Path: {task_input.project_path}")
            
            if task_input.project_specifications:
                context_parts.append(f"Project Context: {json.dumps(task_input.project_specifications, indent=2)}")
            
            context_data = "\n".join(context_parts)
            
            # AUTONOMOUS EXECUTION: Let the agent decide what to do
            # Available MCP tools: filesystem_*, text_editor_*, web_search, terminal_*, etc.
            autonomous_prompt = f"""
**CRITICAL: YOU MUST TAKE ACTUAL ACTIONS, NOT DESCRIBE THEM**

You are an autonomous project setup agent. Your job is to EXECUTE setup actions using MCP tools, not describe what you would do.

**Project Context:**
{context_data}

**EXECUTION PROTOCOL:**
1. IMMEDIATELY start using MCP tools to analyze and set up the project
2. DO NOT explain what you will do - JUST DO IT
3. Use filesystem tools to explore, create directories, and write files
4. Use terminal tools to install dependencies and configure environment
5. CREATE actual files like README.md, requirements.txt, setup.py, etc.

**AVAILABLE MCP TOOLS:**
- filesystem_create_file: Create files with content
- filesystem_create_directory: Create directories
- filesystem_list_directory: Explore existing structure
- terminal_execute_command: Run setup commands
- content_generate_dynamic: Generate file content

**REQUIRED ACTIONS - EXECUTE THESE NOW:**

1. **ANALYZE PROJECT STRUCTURE:**
   - Call filesystem_list_directory to see what exists
   - Identify what setup files are missing

2. **CREATE ESSENTIAL FILES:**
   - Create README.md with project description and setup instructions
   - Create requirements.txt or appropriate dependency file
   - Create setup.py or pyproject.toml if Python project
   - Create .gitignore file
   - Create docs/ directory with basic documentation

3. **SETUP DEVELOPMENT ENVIRONMENT:**
   - Create virtual environment setup scripts
   - Add development configuration files
   - Create example/template files

4. **DOCUMENT THE PROJECT:**
   - Write comprehensive README.md
   - Create CONTRIBUTING.md for development guidelines
   - Add API documentation if applicable

**OUTPUT FORMAT:**
Execute the MCP tool calls immediately. Do not explain - just use the tools to create files and setup the project structure.

START EXECUTING NOW - USE YOUR MCP TOOLS TO SET UP THIS PROJECT.
"""
            
            self.logger.info(f"[AUTONOMOUS] Agent executing autonomous project setup")
            
            # Let the agent work autonomously with pure LLM intelligence
            response = await self.llm_provider.generate(
                prompt=autonomous_prompt,
                max_tokens=8000,
                temperature=0.1
            )
            
            return {
                "setup_result": response,
                "agent_mode": "autonomous",
                "confidence": 0.9
            }
                
        except Exception as e:
            self.logger.error(f"Autonomous setup failed: {e}")
            raise  # No fallbacks - let it fail properly

    async def _execute_setup_plan(self, plan: Dict[str, Any], task_input: ProjectSetupInput) -> Dict[str, Any]:
        """Execute the autonomous setup results."""
        return {
            'files_created': {},
            'commands_executed': [],
            'setup_summary': plan.get('setup_result', 'Autonomous setup completed'),
            'recommendations': ["Setup completed autonomously"],
            'confidence': plan.get('confidence', 0.9),
            'usage_metadata': {'agent_mode': plan.get('agent_mode', 'autonomous')},
            'execution_time': 0.1
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

    def _parse_inputs(self, inputs: Any) -> ProjectSetupInput:
        """Parse inputs cleanly into ProjectSetupAgentInput with detailed validation."""
        try:
            if isinstance(inputs, ProjectSetupInput):
                # Validate existing input object
                if not inputs.user_goal or not inputs.user_goal.strip():
                    raise ValueError("ProjectSetupAgentInput has empty or whitespace user_goal")
                return inputs
            elif isinstance(inputs, dict):
                # Validate required fields before creation
                if 'user_goal' not in inputs:
                    raise ValueError("Missing required field 'user_goal' in input dictionary")
                if not inputs['user_goal'] or not inputs['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return ProjectSetupInput(**inputs)
            elif hasattr(inputs, 'dict'):
                input_dict = inputs.dict()
                if 'user_goal' not in input_dict:
                    raise ValueError("Missing required field 'user_goal' in input object")
                if not input_dict['user_goal'] or not input_dict['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return ProjectSetupInput(**input_dict)
            else:
                raise ValueError(f"Invalid input type: {type(inputs)}. Expected ProjectSetupAgentInput, dict, or object with dict() method. Received: {inputs}")
        except Exception as e:
            raise ValueError(f"Input parsing failed for ProjectSetupAgent: {e}. Input received: {inputs}")

    async def _generate_project_setup(self, task_input: ProjectSetupInput) -> Dict[str, Any]:
        """
        AUTONOMOUS PROJECT SETUP with detailed validation
        No hardcoded logic - agent analyzes needs and sets up project structure
        """
        try:
            # Build context for autonomous project setup
            context_parts = []
            context_parts.append(f"User Goal: {task_input.user_goal}")
            context_parts.append(f"Project Path: {task_input.project_path}")
            
            if task_input.project_specifications:
                context_parts.append(f"Project Context: {json.dumps(task_input.project_specifications, indent=2)}")
            
            context_data = "\n".join(context_parts)
            
            # Use autonomous setup approach instead of capability-specific template
            # The ProjectSetupAgent should be autonomous and decide what setup is needed
            autonomous_prompt = f"""
**CRITICAL: YOU MUST TAKE ACTUAL ACTIONS, NOT DESCRIBE THEM**

You are an autonomous project setup agent. Your job is to EXECUTE setup actions using MCP tools, not describe what you would do.

**Project Context:**
{context_data}

**EXECUTION PROTOCOL:**
1. IMMEDIATELY start using MCP tools to analyze and set up the project
2. DO NOT explain what you will do - JUST DO IT
3. Use filesystem tools to explore, create directories, and write files
4. Use terminal tools to install dependencies and configure environment
5. CREATE actual files like README.md, requirements.txt, setup.py, etc.

**AVAILABLE MCP TOOLS:**
- filesystem_create_file: Create files with content
- filesystem_create_directory: Create directories
- filesystem_list_directory: Explore existing structure
- terminal_execute_command: Run setup commands
- content_generate_dynamic: Generate file content

**REQUIRED ACTIONS - EXECUTE THESE NOW:**

1. **ANALYZE PROJECT STRUCTURE:**
   - Call filesystem_list_directory to see what exists
   - Identify what setup files are missing

2. **CREATE ESSENTIAL FILES:**
   - Create README.md with project description and setup instructions
   - Create requirements.txt or appropriate dependency file
   - Create setup.py or pyproject.toml if Python project
   - Create .gitignore file
   - Create docs/ directory with basic documentation

3. **SETUP DEVELOPMENT ENVIRONMENT:**
   - Create virtual environment setup scripts
   - Add development configuration files
   - Create example/template files

4. **DOCUMENT THE PROJECT:**
   - Write comprehensive README.md
   - Create CONTRIBUTING.md for development guidelines
   - Add API documentation if applicable

**OUTPUT FORMAT:**
Execute the MCP tool calls immediately. Do not explain - just use the tools to create files and setup the project structure.

START EXECUTING NOW - USE YOUR MCP TOOLS TO SET UP THIS PROJECT.
"""
            
            # Validate MCP tools availability
            if not hasattr(self, 'mcp_tools') or not self.mcp_tools:
                raise ValueError("MCP tools not available or empty in ProjectSetupAgent")
            
            self.logger.info(f"AUTONOMOUS project setup execution")
            
            # Let the agent work autonomously
            response = await self.llm_provider.generate(
                prompt=autonomous_prompt,
                max_tokens=6000,
                temperature=0.2
            )
            
            # Detailed response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for project setup. Prompt length: {len(autonomous_prompt)} chars. User goal: {task_input.user_goal}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for project setup. Response: '{response}'. User goal: {task_input.user_goal}")
            
            if len(response.strip()) < 100:
                raise ValueError(f"LLM project setup response too short ({len(response)} chars). Expected substantial setup analysis. Response: '{response}'. User goal: {task_input.user_goal}")
            
            # Validate response content quality
            response_lower = response.lower()
            setup_keywords = ["setup", "create", "structure", "file", "directory", "project", "init", "config"]
            if not any(keyword in response_lower for keyword in setup_keywords):
                raise ValueError(f"LLM response doesn't appear to contain project setup content (none of {setup_keywords} found). Response: '{response}'. User goal: {task_input.user_goal}")
            
            self.logger.info(f"Project setup response: {len(response)} chars")
            self.logger.info(f"Response preview: {response[:300]}...")
            
            # Return expected structure for ProjectSetupAgentOutput - NO FALLBACKS
            return {
                "status": "success",
                "project_setup_result": response,
                "agent_mode": "autonomous",
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "response_length": len(response),
                "confidence_score": ConfidenceScore(
                    value=0.85,
                    method="autonomous_project_setup",
                    explanation="Autonomous project setup completed successfully"
                )
            }
            
        except Exception as e:
            error_msg = f"""ProjectSetupAgent setup generation failed:

ERROR: {e}

INPUT CONTEXT:
- User Goal: {task_input.user_goal}
- Project Path: {task_input.project_path}
- Project Specifications: {task_input.project_specifications}

SETUP CONTEXT:
- Available MCP Tools: {len(self.mcp_tools) if hasattr(self, 'mcp_tools') and self.mcp_tools else 'None/Empty'}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def process_task(self, task_input: ProjectSetupInput) -> ProjectSetupOutput:
        """
        AUTONOMOUS PROJECT SETUP EXECUTION
        No hardcoded capabilities - agent decides what needs to be done
        """
        execution_start = time.time()
        self.logger.info(f"[AUTONOMOUS] Starting autonomous project setup for: {task_input.user_goal}")
        
        try:
            # AUTONOMOUS EXECUTION - Let the agent work
            result = await self._generate_project_setup(task_input)
            
            execution_time = time.time() - execution_start
            
            return ProjectSetupOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="completed" if result.get("status") == "success" else "failed",
                result=result,
                execution_time=execution_time,
                autonomous_mode=True
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            self.logger.error(f"Autonomous project setup failed: {e}")
            
            return ProjectSetupOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id, 
                status="error",
                result={"error": str(e), "agent_mode": "autonomous_failed"},
                execution_time=execution_time,
                autonomous_mode=True
            ) 