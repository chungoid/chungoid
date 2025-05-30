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
        Simple requirements gathering iteration with detailed validation.
        """
        start_time = datetime.datetime.now()
        
        try:
            # Validate execution context
            if not context:
                raise ValueError("ExecutionContext is None - cannot execute InteractiveRequirementsAgent")
            
            if not hasattr(context, 'inputs'):
                raise ValueError("ExecutionContext missing 'inputs' attribute - cannot execute InteractiveRequirementsAgent")
            
            if not context.inputs:
                raise ValueError("ExecutionContext.inputs is None or empty - cannot execute InteractiveRequirementsAgent")
            
            self.logger.info(f"Starting simple requirements gathering for iteration {iteration}")
            
            # Convert context inputs to our input model with detailed validation
            try:
                if isinstance(context.inputs, InteractiveRequirementsInput):
                    task_input = context.inputs
                elif isinstance(context.inputs, dict):
                    # Validate required fields
                    if 'project_path' not in context.inputs:
                        raise ValueError("Missing required field 'project_path' in input dictionary")
                    if not context.inputs['project_path'] or not context.inputs['project_path'].strip():
                        raise ValueError("Field 'project_path' cannot be empty or whitespace")
                    
                    context_dict = self._convert_pydantic_to_dict(context.inputs)
                    task_input = InteractiveRequirementsInput(**context_dict)
                elif hasattr(context.inputs, 'dict'):
                    input_dict = context.inputs.dict()
                    if 'project_path' not in input_dict:
                        raise ValueError("Missing required field 'project_path' in input object")
                    if not input_dict['project_path'] or not input_dict['project_path'].strip():
                        raise ValueError("Field 'project_path' cannot be empty or whitespace")
                    
                    task_input = InteractiveRequirementsInput(**input_dict)
                else:
                    raise ValueError(f"Invalid input type: {type(context.inputs)}. Expected InteractiveRequirementsInput, dict, or object with dict() method. Received: {context.inputs}")
                    
                # Final validation of task_input
                if not task_input.project_path or not task_input.project_path.strip():
                    raise ValueError("Field 'project_path' cannot be empty or whitespace")
                        
            except Exception as e:
                raise ValueError(f"Input parsing/validation failed for InteractiveRequirementsAgent: {e}. Context inputs type: {type(context.inputs)}, Context inputs: {context.inputs}")
            
            # 1. Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            if not project_context or not project_context.strip():
                raise ValueError(f"Failed to gather project context or context is empty. Project path: {task_input.project_path}")
            
            # 2. Let LLM gather requirements (conversation or auto-generate)
            gathering_result = await self._gather_requirements_with_llm(task_input, project_context)
            
            if not gathering_result:
                raise ValueError(f"Requirements gathering returned None/empty result. Project path: {task_input.project_path}")
            
            if not gathering_result.get("success"):
                error_msg = gathering_result.get("error", "Unknown requirements gathering error")
                raise ValueError(f"Requirements gathering failed: {error_msg}. Project path: {task_input.project_path}")
            
            # 3. Build output with validation
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            
            if execution_time < 0:
                raise ValueError(f"Invalid execution time: {execution_time}. Time calculation error.")
            
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
                
        except Exception as e:
            execution_time = (datetime.datetime.now() - start_time).total_seconds()
            error_msg = f"""InteractiveRequirementsAgent execution failed:

ERROR: {e}

CONTEXT:
- Iteration: {iteration}
- Execution Time: {execution_time:.3f}s
- Input Type: {type(context.inputs) if context and hasattr(context, 'inputs') else 'No context/inputs'}
- Input Value: {context.inputs if context and hasattr(context, 'inputs') else 'No context/inputs'}

REQUIREMENTS GATHERING DETAILS:
- Project Path: {getattr(task_input, 'project_path', 'Unknown') if 'task_input' in locals() else 'Not parsed'}
- User Goal: {getattr(task_input, 'user_goal', 'Unknown') if 'task_input' in locals() else 'Not parsed'}
- Interactive Mode: {getattr(task_input, 'interactive_mode', 'Unknown') if 'task_input' in locals() else 'Not parsed'}
"""
            self.logger.error(error_msg)
            
            output = InteractiveRequirementsOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
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
        Main method: Use YAML prompt template + project context to let LLM gather requirements with detailed validation.
        Either through conversation or auto-generation.
        """
        try:
            # Validate inputs
            if not project_context or not project_context.strip():
                raise ValueError(f"Project context is empty or whitespace. Cannot gather requirements without context. Project path: {task_input.project_path}")
            
            # Get main prompt from YAML template - NO FALLBACKS
            main_prompt = await self._get_main_prompt(task_input, project_context)
            
            if not main_prompt or not main_prompt.strip():
                raise ValueError(f"Failed to get main prompt from YAML template or prompt is empty. Project path: {task_input.project_path}")
            
            # Let LLM run the show
            self.logger.info("Gathering requirements with LLM...")
            response = await self.llm_provider.generate(
                prompt=main_prompt,
                max_tokens=4000,
                temperature=0.7
            )
            
            # Detailed LLM response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for requirements gathering. Prompt length: {len(main_prompt)} chars. Project path: {task_input.project_path}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for requirements gathering. Response: '{response}'. Project path: {task_input.project_path}")
            
            if len(response.strip()) < 50:
                raise ValueError(f"LLM requirements response too short ({len(response)} chars). Expected substantial requirements analysis. Response: '{response}'. Project path: {task_input.project_path}")
            
            # Validate response content quality
            response_lower = response.lower()
            requirements_keywords = ["requirement", "specification", "feature", "functionality", "goal", "objective"]
            if not any(keyword in response_lower for keyword in requirements_keywords):
                raise ValueError(f"LLM response doesn't appear to contain requirements content (none of {requirements_keywords} found). Response: '{response}'. Project path: {task_input.project_path}")
            
            # Parse structured response - NO FALLBACKS, FAIL LOUDLY
            try:
                requirements_result = json.loads(response)
                if not isinstance(requirements_result, dict):
                    raise ValueError(f"LLM response is not a valid JSON dictionary: {type(requirements_result)}. Response: '{response}'")
                    
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse LLM response as JSON for requirements gathering: {e}. Response: '{response}'. Project path: {task_input.project_path}")
            except Exception as e:
                raise ValueError(f"Failed to process LLM response for requirements gathering: {e}. Response: '{response}'. Project path: {task_input.project_path}")
                
            self.logger.info(f"LLM provided structured requirements")
            
            # Execute any file operations (like writing enhanced goal file)
            execution_results = await self._execute_requirements_plan(requirements_result, task_input)
            
            return {
                "success": True,
                **execution_results,
                "llm_response": response
            }
            
        except Exception as e:
            error_msg = f"""InteractiveRequirementsAgent LLM requirements gathering failed:

ERROR: {e}

INPUT CONTEXT:
- Project Path: {task_input.project_path}
- User Goal: {task_input.user_goal}
- Interactive Mode: {task_input.interactive_mode}
- Auto Generate: {task_input.auto_generate}

LLM CONTEXT:
- Project Context Length: {len(project_context) if project_context else 0} chars
- Project Context Preview: {project_context[:300] if project_context else 'None'}...
"""
            self.logger.error(error_msg)
            return {
                "success": False,
                "error": str(e)
            }

    async def _get_main_prompt(self, task_input: InteractiveRequirementsInput, project_context: str) -> str:
        """
        Get the main prompt template from YAML and inject project context.
        NO FALLBACKS - fail if YAML not available.
        """
        # Get prompt from YAML template - NO FALLBACKS
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