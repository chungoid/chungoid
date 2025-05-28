"""
InteractiveRequirementsAgent: Dead simple, LLM-powered requirements gathering.

This agent conducts conversations with users by:
1. Using MCP tools to understand existing project state
2. Using main prompt template from YAML  
3. Letting the LLM run the show to conduct intelligent conversations
4. Generating comprehensive project specifications

No complex phases, no brittle conversation management, just LLM + MCP tools.
"""

from __future__ import annotations

import logging
import datetime
import uuid
import json
import os
from typing import Any, Dict, Optional, List, ClassVar, Type
from pathlib import Path

from pydantic import BaseModel, Field

from chungoid.agents.unified_agent import UnifiedAgent
from chungoid.utils.llm_provider import LLMProvider
from chungoid.utils.prompt_manager import PromptManager
from chungoid.utils.agent_registry import AgentCard, AgentCategory, AgentVisibility
from chungoid.schemas.unified_execution_schemas import (
    ExecutionContext as UEContext,
    IterationResult,
    ExecutionMode,
    CompletionReason,
    ExecutionMetadata,
    AgentOutput
)
from chungoid.registry import register_agent

logger = logging.getLogger(__name__)


class InteractiveRequirementsInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this requirements gathering task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core fields - simplified
    project_path: str = Field(..., description="Path to project directory")
    goal_file_path: Optional[str] = Field(None, description="Path to goal file to enhance")
    user_goal: Optional[str] = Field(None, description="What the user wants to build")
    
    # Options
    interactive_mode: bool = Field(default=True, description="Whether to conduct interactive conversation")
    auto_generate: bool = Field(default=False, description="Whether to auto-generate without conversation")
    max_conversation_turns: int = Field(default=10, description="Maximum number of conversation turns")
    
    # Context from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")


class InteractiveRequirementsOutput(AgentOutput):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    status: str = Field(..., description="Status of requirements gathering (SUCCESS, FAILURE, etc.).")
    
    # Gathering results
    enhanced_goal_file: Optional[str] = Field(None, description="Path to enhanced goal file")
    conversation_turns: int = Field(default=0, description="Number of conversation turns conducted")
    requirements_gathered: Dict[str, Any] = Field(default_factory=dict, description="Requirements extracted")
    
    # Results
    requirements_summary: str = Field(..., description="Summary of requirements gathering")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations for next steps")
    confidence_level: float = Field(default=0.0, description="Confidence in gathered requirements")
    
    # Status and metadata
    message: str = Field(..., description="Message detailing the outcome.")
    error_message: Optional[str] = Field(None, description="Error message if gathering failed.")
    total_gathering_time: Optional[float] = Field(None, description="Total time taken for requirements gathering")


@register_agent()
class InteractiveRequirementsAgent(UnifiedAgent):
    """
    Dead simple requirements gathering agent.
    
    1. MCP tools for project understanding and file operations
    2. Main prompt from YAML template  
    3. LLM runs the show - conducts conversations and generates specifications
    """
    
    AGENT_ID: ClassVar[str] = "interactive_requirements_agent"
    AGENT_NAME: ClassVar[str] = "Interactive Requirements Agent"
    AGENT_DESCRIPTION: ClassVar[str] = "Simple, LLM-powered requirements gathering with MCP tools."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "interactive_requirements_agent_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "3.0.0"  # Major version bump for simplification
    CAPABILITIES: ClassVar[List[str]] = ["conversation", "requirements_analysis", "file_operations", "project_analysis"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.SYSTEM_ORCHESTRATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[InteractiveRequirementsInput]] = InteractiveRequirementsInput
    OUTPUT_SCHEMA: ClassVar[Type[InteractiveRequirementsOutput]] = InteractiveRequirementsOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["simple_requirements_gathering"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["mcp_conversation"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Convert Pydantic objects to dictionaries recursively for compatibility."""
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        elif hasattr(obj, 'dict'):
            return obj.dict()
        elif isinstance(obj, dict):
            return {k: self._convert_pydantic_to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_pydantic_to_dict(item) for item in obj]
        else:
            return obj

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Simple requirements gathering iteration: gather project context + let LLM gather requirements.
        """
        start_time = datetime.datetime.now()
        
        try:
            self.logger.info(f"Starting simple requirements gathering for: {context.inputs.get('user_goal', 'Unknown project')}")
            
            # Convert context inputs to our input model
            context_dict = self._convert_pydantic_to_dict(context.inputs)
            task_input = InteractiveRequirementsInput(**context_dict)
            
            # 1. Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            # 2. Let LLM gather requirements (conversation or auto-generate)
            gathering_result = await self._gather_requirements_with_llm(task_input, project_context)
            
            # 3. Build output
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if gathering_result["success"]:
                output = InteractiveRequirementsOutput(
                    task_id=task_input.task_id,
                    project_id=task_input.project_id,
                    status="SUCCESS",
                    enhanced_goal_file=gathering_result.get("enhanced_goal_file"),
                    conversation_turns=gathering_result.get("conversation_turns", 0),
                    requirements_gathered=gathering_result.get("requirements_gathered", {}),
                    requirements_summary=gathering_result.get("requirements_summary", "Requirements gathering completed"),
                    recommendations=gathering_result.get("recommendations", []),
                    confidence_level=gathering_result.get("confidence_level", 0.8),
                    message=f"Requirements gathering completed successfully in {execution_time:.1f}s",
                    total_gathering_time=execution_time
                )
                
                self.logger.info("Requirements gathering completed successfully")
                return IterationResult(
                    iteration_number=iteration,
                    completion_reason=CompletionReason.SUCCESS,
                    agent_output=output,
                    execution_metadata=ExecutionMetadata(
                        mode=ExecutionMode.SINGLE_PASS,
                        protocol_used="simple_requirements_gathering",
                        execution_time=execution_time,
                        iterations_planned=1,
                        tools_utilized=[]
                    )
                )
            else:
                error_msg = gathering_result.get("error", "Requirements gathering failed")
                output = InteractiveRequirementsOutput(
                    task_id=task_input.task_id,
                    project_id=task_input.project_id,
                    status="FAILURE",
                    requirements_summary="Requirements gathering failed",
                    message=f"Requirements gathering failed: {error_msg}",
                    error_message=error_msg
                )
                
                self.logger.error(f"Requirements gathering failed: {error_msg}")
                return IterationResult(
                    iteration_number=iteration,
                    completion_reason=CompletionReason.ERROR,
                    agent_output=output,
                    execution_metadata=ExecutionMetadata(
                        mode=ExecutionMode.SINGLE_PASS,
                        protocol_used="simple_requirements_gathering",
                        execution_time=execution_time,
                        iterations_planned=1,
                        tools_utilized=[]
                    )
                )
                
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            self.logger.error(f"Requirements gathering iteration failed: {e}", exc_info=True)
            
            output = InteractiveRequirementsOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())),
                project_id=getattr(task_input, 'project_id', str(uuid.uuid4())),
                status="ERROR",
                requirements_summary="Requirements gathering encountered an error",
                message=f"Requirements gathering error: {str(e)}",
                error_message=str(e)
            )
            
            return IterationResult(
                iteration_number=iteration,
                completion_reason=CompletionReason.ERROR,
                agent_output=output,
                execution_metadata=ExecutionMetadata(
                    mode=ExecutionMode.SINGLE_PASS,
                    protocol_used="simple_requirements_gathering",
                    execution_time=execution_time,
                    iterations_planned=1,
                    tools_utilized=[]
                )
            )

    async def _gather_project_context(self, task_input: InteractiveRequirementsInput) -> str:
        """
        Gather comprehensive project context using MCP tools.
        """
        self.logger.info(f"Gathering project context from: {task_input.project_path}")
        
        context_parts = []
        
        try:
            # 1. List project directory structure
            if os.path.exists(task_input.project_path):
                project_files = await self._call_mcp_tool("filesystem_list_directory", {
                    "directory_path": task_input.project_path,
                    "recursive": True,
                    "max_depth": 3
                })
                
                if project_files.get("success"):
                    context_parts.append(f"=== PROJECT STRUCTURE ===\n{project_files.get('directory_tree', 'No structure available')}")
            
            # 2. Check for existing goal file
            goal_file_path = task_input.goal_file_path or os.path.join(task_input.project_path, "goal.txt")
            if os.path.exists(goal_file_path):
                try:
                    goal_content = await self._call_mcp_tool("filesystem_read_file", {
                        "file_path": goal_file_path
                    })
                    
                    if goal_content.get("success"):
                        content = goal_content.get("content", "")
                        context_parts.append(f"=== EXISTING GOAL FILE ===\n{content}")
                except:
                    pass  # Goal file doesn't exist or can't be read
            
            # 3. Look for configuration files that give context
            config_files = [
                "package.json", "requirements.txt", "pyproject.toml", "pom.xml", 
                "build.gradle", "Cargo.toml", "go.mod", "composer.json", 
                "README.md", "README.txt", ".gitignore"
            ]
            
            for config_file in config_files:
                try:
                    file_path = os.path.join(task_input.project_path, config_file)
                    if os.path.exists(file_path):
                        file_content = await self._call_mcp_tool("filesystem_read_file", {
                            "file_path": file_path
                        })
                        
                        if file_content.get("success"):
                            content = file_content.get("content", "")[:1500]  # Limit to 1500 chars
                            context_parts.append(f"=== {config_file.upper()} ===\n{content}")
                except:
                    pass  # File doesn't exist or can't be read
            
            # 4. Sample some code files to understand the project
            await self._sample_code_files(task_input.project_path, context_parts)
            
            # 5. Add user goal and project specifications
            if task_input.user_goal:
                context_parts.append(f"=== USER GOAL ===\n{task_input.user_goal}")
            
            if task_input.project_specifications:
                spec_summary = str(task_input.project_specifications)[:2000]  # Limit size
                context_parts.append(f"=== PROJECT SPECIFICATIONS ===\n{spec_summary}")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            self.logger.warning(f"Error gathering project context: {e}")
            return f"Project path: {task_input.project_path}\nUser goal: {task_input.user_goal or 'Not specified'}"

    async def _sample_code_files(self, project_path: str, context_parts: List[str]):
        """
        Sample some code files to understand project language and structure.
        """
        try:
            # Common code file extensions
            code_extensions = ['.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.php', '.rb']
            
            for root, dirs, files in os.walk(project_path):
                # Skip hidden directories and common build/cache directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'target', 'build']]
                
                for file in files[:5]:  # Limit to first 5 files per directory
                    if any(file.endswith(ext) for ext in code_extensions):
                        file_path = os.path.join(root, file)
                        try:
                            file_content = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": file_path
                            })
                            
                            if file_content.get("success"):
                                content = file_content.get("content", "")[:800]  # First 800 chars
                                relative_path = os.path.relpath(file_path, project_path)
                                context_parts.append(f"=== CODE SAMPLE: {relative_path} ===\n{content}")
                                
                                # Limit total code samples
                                if len([p for p in context_parts if "CODE SAMPLE:" in p]) >= 3:
                                    return
                        except:
                            pass  # Can't read file, continue
                break  # Only sample from top-level directory
        except Exception as e:
            self.logger.debug(f"Error sampling code files: {e}")

    async def _gather_requirements_with_llm(self, task_input: InteractiveRequirementsInput, project_context: str) -> Dict[str, Any]:
        """
        Main method: Use YAML prompt template + project context to let LLM gather requirements.
        Either through conversation or auto-generation.
        """
        try:
            # Get main prompt from YAML template (or fallback)
            main_prompt = await self._get_main_prompt(task_input, project_context)
            
            # Let LLM run the show
            self.logger.info("Gathering requirements with LLM...")
            response = await self.llm_provider.generate(
                prompt=main_prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            if not response:
                return {"success": False, "error": "No response from LLM"}
            
            # Try to parse structured response 
            try:
                requirements_result = json.loads(response)
                self.logger.info(f"LLM provided structured requirements")
                
                # Execute any file operations (like writing enhanced goal file)
                execution_results = await self._execute_requirements_plan(requirements_result, task_input)
                
                return {
                    "success": True,
                    **execution_results,
                    "llm_response": response
                }
                
            except json.JSONDecodeError:
                # Fallback: treat response as requirements text
                self.logger.info("LLM response not JSON, treating as requirements text")
                return await self._fallback_requirements_processing(response, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM requirements gathering failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_main_prompt(self, task_input: InteractiveRequirementsInput, project_context: str) -> str:
        """
        Get the main prompt template from YAML and inject project context.
        Falls back to built-in prompt if YAML not available.
        """
        try:
            # Try to get prompt from YAML template
            prompt_template = await self.prompt_manager.get_prompt(
                self.PROMPT_TEMPLATE_NAME,
                {
                    "user_goal": task_input.user_goal or "Gather project requirements",
                    "project_path": task_input.project_path,
                    "project_context": project_context,
                    "interactive_mode": task_input.interactive_mode,
                    "auto_generate": task_input.auto_generate,
                    "max_conversation_turns": task_input.max_conversation_turns
                }
            )
            self.logger.info("Using YAML prompt template")
            return prompt_template
            
        except Exception as e:
            self.logger.warning(f"Could not load YAML prompt template: {e}, using built-in fallback")
            
            # Built-in fallback prompt
            mode_instruction = "conduct an interactive conversation to gather requirements" if task_input.interactive_mode else "automatically generate comprehensive requirements"
            
            return f"""You are an expert requirements gathering agent. Analyze the project and {mode_instruction}.

PROJECT CONTEXT:
{project_context}

INSTRUCTIONS:
1. Analyze the existing project structure, goal files, and code samples
2. Understand what the user wants to build and why
3. {"Conduct a natural conversation to gather detailed requirements" if task_input.interactive_mode else "Generate comprehensive project requirements automatically"}
4. Create or enhance the project goal file with structured specifications
5. Return a JSON response with this exact structure:

{{
    "requirements_gathered": {{
        "project_purpose": "What problem this project solves",
        "target_audience": "Who will use this software",
        "main_features": ["List of core features"],
        "technical_requirements": {{
            "primary_language": "Main programming language",
            "frameworks": ["Required frameworks"],
            "dependencies": ["Key dependencies"],
            "platforms": ["Target platforms"]
        }},
        "success_criteria": ["How success will be measured"]
    }},
    "conversation_approach": "{('interactive' if task_input.interactive_mode else 'auto_generate')}",
    "conversation_turns": {task_input.max_conversation_turns if task_input.interactive_mode else 0},
    "goal_file_content": "Complete YAML content for enhanced goal file",
    "requirements_summary": "Brief summary of what was gathered",
    "recommendations": ["Next steps for the project"],
    "confidence_level": 0.85
}}

{"CONVERSATION MODE: Ask thoughtful questions to understand the project better. Be conversational and helpful." if task_input.interactive_mode else "AUTO-GENERATION MODE: Create comprehensive requirements based on the project context."}

USER GOAL: {task_input.user_goal or "Gather comprehensive project requirements"}
PROJECT PATH: {task_input.project_path}
INTERACTIVE MODE: {task_input.interactive_mode}

Return ONLY the JSON response, no additional text."""

    async def _execute_requirements_plan(self, requirements_result: Dict[str, Any], task_input: InteractiveRequirementsInput) -> Dict[str, Any]:
        """
        Execute the LLM-generated requirements plan (like writing goal files).
        """
        results = {
            "requirements_gathered": requirements_result.get("requirements_gathered", {}),
            "conversation_turns": requirements_result.get("conversation_turns", 0),
            "requirements_summary": requirements_result.get("requirements_summary", "Requirements gathered"),
            "recommendations": requirements_result.get("recommendations", []),
            "confidence_level": requirements_result.get("confidence_level", 0.8),
            "enhanced_goal_file": None
        }
        
        try:
            # Write enhanced goal file if provided
            goal_file_content = requirements_result.get("goal_file_content")
            if goal_file_content:
                goal_file_path = task_input.goal_file_path or os.path.join(task_input.project_path, "goal.txt")
                
                write_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": goal_file_path,
                    "content": goal_file_content
                })
                
                if write_result.get("success"):
                    results["enhanced_goal_file"] = goal_file_path
                    self.logger.info(f"Enhanced goal file written to: {goal_file_path}")
                else:
                    self.logger.warning(f"Failed to write goal file: {write_result.get('error', 'Unknown error')}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Requirements plan execution failed: {e}")
            return results

    async def _fallback_requirements_processing(self, response: str, task_input: InteractiveRequirementsInput) -> Dict[str, Any]:
        """
        Fallback when LLM doesn't return structured JSON - extract requirements from text.
        """
        try:
            # Simple fallback - create basic goal file from response
            goal_file_path = task_input.goal_file_path or os.path.join(task_input.project_path, "goal.txt")
            
            # Create simple goal file content
            goal_content = f"""# Project Requirements

## Generated by Interactive Requirements Agent

{response}

## Additional Information
- Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Project path: {task_input.project_path}
- User goal: {task_input.user_goal or 'Not specified'}
"""
            
            # Write goal file
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": goal_file_path,
                "content": goal_content
            })
            
            enhanced_goal_file = goal_file_path if write_result.get("success") else None
            
            return {
                "success": True,
                "requirements_gathered": {"fallback_requirements": response},
                "conversation_turns": 1,
                "requirements_summary": "Requirements processed with fallback method",
                "recommendations": ["Review and refine the generated requirements"],
                "confidence_level": 0.6,
                "enhanced_goal_file": enhanced_goal_file
            }
            
        except Exception as e:
            self.logger.error(f"Fallback requirements processing failed: {e}")
            return {"success": False, "error": f"Fallback processing failed: {str(e)}"}

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = InteractiveRequirementsInput.model_json_schema()
        output_schema = InteractiveRequirementsOutput.model_json_schema()
        
        return AgentCard(
            agent_id=InteractiveRequirementsAgent.AGENT_ID,
            name=InteractiveRequirementsAgent.AGENT_NAME,
            description=InteractiveRequirementsAgent.AGENT_DESCRIPTION,
            version=InteractiveRequirementsAgent.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[InteractiveRequirementsAgent.CATEGORY.value],
            visibility=InteractiveRequirementsAgent.VISIBILITY.value,
            capability_profile={
                "gathers_requirements": True,
                "simple_conversation": True,
                "llm_powered": True,
                "mcp_tools_integrated": True,
                "supports_any_project": True
            },
            metadata={
                "callable_fn_path": f"{InteractiveRequirementsAgent.__module__}.{InteractiveRequirementsAgent.__name__}"
            }
        )

    def get_input_schema(self) -> Type[InteractiveRequirementsInput]:
        return InteractiveRequirementsInput

    def get_output_schema(self) -> Type[InteractiveRequirementsOutput]:
        return InteractiveRequirementsOutput