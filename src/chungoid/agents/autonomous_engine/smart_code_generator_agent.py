"""
SmartCodeGeneratorAgent_v1: Dead simple, LLM-powered code generation.

This agent generates source code files by:
1. Using MCP tools to understand the project context
2. Using main prompt template from YAML  
3. Letting the LLM run the show with direct file operations

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

class SmartCodeGeneratorInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this code generation task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core fields - simplified
    user_goal: Optional[str] = Field(None, description="What the user wants to build")
    project_path: Optional[str] = Field(None, description="Where to build it")
    
    # Traditional fields for backward compatibility
    task_description: Optional[str] = Field(None, description="Core description of the code to be generated.")
    target_file_path: Optional[str] = Field(None, description="Intended relative path of the file to be created.")
    programming_language: Optional[str] = Field(None, description="Target programming language.")
    
    # Context fields from other agents
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Project specs from orchestrator")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    target_languages: Optional[List[str]] = Field(None, description="Target programming languages")
    technologies: Optional[List[str]] = Field(None, description="Project technologies")
    requirements_context: Optional[Dict[str, Any]] = Field(None, description="Requirements from previous stage")
    architecture_context: Optional[Dict[str, Any]] = Field(None, description="Architecture from previous stage")
    risk_context: Optional[Dict[str, Any]] = Field(None, description="Risk assessment from previous stage")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'SmartCodeGeneratorInput':
        """Ensure we have minimum info to generate code."""
        
        # Need either user_goal or task_description
        if not self.user_goal and not self.task_description:
            raise ValueError("Either user_goal or task_description is required")
        
        # Default project_path if not provided
        if not self.project_path:
            self.project_path = "."
        
        return self

class SmartCodeGeneratorOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    status: str = Field(..., description="Status of code generation (SUCCESS, FAILURE, etc.).")
    generated_files: List[Dict[str, Any]] = Field(default_factory=list, description="List of generated code files with metadata.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the generated code.")
    error_message: Optional[str] = Field(None, description="Error message if generation failed.")

@register_autonomous_engine_agent(capabilities=["code_generation", "systematic_implementation"])
class SmartCodeGeneratorAgent_v1(UnifiedAgent):
    """
    Dead simple code generation agent.
    
    1. MCP tools for file operations and project understanding
    2. Main prompt from YAML template  
    3. LLM runs the show - no complex phases or state management
    """
    
    AGENT_ID: ClassVar[str] = "SmartCodeGeneratorAgent_v1"
    AGENT_NAME: ClassVar[str] = "Smart Code Generator Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Simple, LLM-powered code generation with MCP tools."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "smart_code_generator_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "2.0.0"  # Major version bump for simplification
    CAPABILITIES: ClassVar[List[str]] = ["code_generation", "systematic_implementation"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.AUTONOMOUS_COORDINATION
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorInput]] = SmartCodeGeneratorInput
    OUTPUT_SCHEMA: ClassVar[Type[SmartCodeGeneratorOutput]] = SmartCodeGeneratorOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["simple_code_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["mcp_file_operations"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing']

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)

    def _convert_pydantic_to_dict(self, obj: Any) -> Any:
        """Recursively convert Pydantic objects to dictionaries."""
        if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
            # This is a Pydantic object
            return obj.dict()
        elif isinstance(obj, dict):
            # Recursively process dictionary values
            return {key: self._convert_pydantic_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            # Recursively process list/tuple items
            return [self._convert_pydantic_to_dict(item) for item in obj]
        else:
            # Return as-is for primitive types
            return obj

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        Dead simple execution: Let the LLM understand the project and generate code.
        No complex phases, just LLM + MCP tools.
        """
        try:
            # Convert inputs - handle both dict and Pydantic objects
            if isinstance(context.inputs, dict):
                # Convert any nested Pydantic objects to dicts
                converted_inputs = self._convert_pydantic_to_dict(context.inputs)
                task_input = SmartCodeGeneratorInput(**converted_inputs)
            elif hasattr(context.inputs, 'dict'):
                # Direct Pydantic object
                task_input = SmartCodeGeneratorInput(**context.inputs.dict())
            else:
                task_input = context.inputs

            self.logger.info(f"Starting simple code generation for: {task_input.user_goal or task_input.task_description}")

            # 1. Gather project context using MCP tools
            project_context = await self._gather_project_context(task_input)
            
            # 2. Let LLM generate code using main prompt template
            generation_result = await self._generate_code_with_llm(task_input, project_context)
            
            # 3. Create simple output
            output = SmartCodeGeneratorOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS" if generation_result["success"] else "FAILURE",
                generated_files=generation_result.get("files", []),
                confidence_score=ConfidenceScore(
                    value=0.9,
                    method="llm_self_assessment",
                    explanation="High confidence in code generation quality and completeness"
                ),
                error_message=generation_result.get("error")
            )
            
            return IterationResult(
                output=output,
                quality_score=generation_result.get("confidence", 0.8),
                tools_used=["llm_analysis", "mcp_file_operations"],
                protocol_used="simple_code_generation"
            )
            
        except Exception as e:
            self.logger.error(f"Code generation failed: {e}")
            
            # Safe error output
            task_id = getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4())
            project_id = getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown'
            
            error_output = SmartCodeGeneratorOutput(
                task_id=task_id,
                project_id=project_id,
                status="ERROR",
                generated_files=[],
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="simple_code_generation"
            )

    async def _gather_project_context(self, task_input: SmartCodeGeneratorInput) -> str:
        """
        Simple method: Gather all relevant project context for the LLM.
        "Learn about the fuckin' project" - scan files, read docs, understand context.
        """
        self.logger.info(f"Gathering project context from: {task_input.project_path}")
        
        context_parts = []
        
        # Add user goal/task description
        if task_input.user_goal:
            context_parts.append(f"User Goal: {task_input.user_goal}")
        elif task_input.task_description:
            context_parts.append(f"Task: {task_input.task_description}")
        
        # Add project specifications if available
        if task_input.project_specifications:
            context_parts.append(f"Project Specifications: {json.dumps(task_input.project_specifications, indent=2)}")
        
        # Add context from previous stages
        if task_input.requirements_context:
            context_parts.append(f"Requirements Context: {json.dumps(task_input.requirements_context, indent=2)}")
        if task_input.architecture_context:
            context_parts.append(f"Architecture Context: {json.dumps(task_input.architecture_context, indent=2)}")
        if task_input.risk_context:
            context_parts.append(f"Risk Context: {json.dumps(task_input.risk_context, indent=2)}")
        
        # Scan project directory with MCP tools
        try:
            list_result = await self._call_mcp_tool("filesystem_list_directory", {
                "directory_path": task_input.project_path
            })
            
            if list_result.get("success"):
                files = list_result.get("files", [])
                if files:
                    context_parts.append(f"Existing files in {task_input.project_path}: {files}")
                    
                    # Read key documentation files
                    for file_info in files[:10]:  # Limit to first 10 files
                        filename = file_info.get("name", "") if isinstance(file_info, dict) else str(file_info)
                        
                        # Read documentation files
                        if any(pattern in filename.lower() for pattern in [
                            "readme", "blueprint", "spec", "requirements", "architecture", "design", "plan"
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

    async def _generate_code_with_llm(self, task_input: SmartCodeGeneratorInput, project_context: str) -> Dict[str, Any]:
        """
        Main method: Use YAML prompt template + project context to let LLM generate code.
        LLM returns structured instructions for files to create, we execute with MCP tools.
        """
        try:
            # Get main prompt from YAML template (or fallback)
            main_prompt = await self._get_main_prompt(task_input, project_context)
            
            # Let LLM run the show
            self.logger.info("Generating code with LLM...")
            response = await self.llm_provider.generate(
                prompt=main_prompt,
                max_tokens=6000,
                temperature=0.1
            )
            
            if not response:
                return {"success": False, "error": "No response from LLM"}
            
            # Try to parse structured response (LLM should return JSON with file instructions)
            try:
                # LLM should return JSON with files to create
                llm_instructions = json.loads(response)
                self.logger.info(f"LLM provided structured instructions for {len(llm_instructions.get('files', []))} files")
                
                # Execute LLM's file creation instructions using MCP tools
                generated_files = await self._execute_file_instructions(
                    llm_instructions, 
                    task_input.project_path
                )
                
                return {
                    "success": True,
                    "files": generated_files,
                    "confidence": llm_instructions.get("confidence", 0.8),
                    "llm_response": response
                }
                
            except json.JSONDecodeError:
                # Fallback: treat response as single file content
                self.logger.info("LLM response not JSON, treating as single file content")
                return await self._fallback_single_file(response, task_input)
                
        except Exception as e:
            self.logger.error(f"LLM code generation failed: {e}")
            return {"success": False, "error": str(e)}

    async def _get_main_prompt(self, task_input: SmartCodeGeneratorInput, project_context: str) -> str:
        """
        Get the main prompt template from YAML and inject project context.
        Falls back to built-in prompt if YAML not available.
        """
        try:
            # Try to get prompt from YAML template
            prompt_template = self.prompt_manager.get_prompt_definition(
                "smart_code_generator_agent_v1_prompt",  # prompt_name from YAML id field
                "0.2.0",  # prompt_version from YAML version field
                sub_path="autonomous_engine"  # subdirectory where prompt is located
            )
            
            # Gather unified discovery results
            try:
                discovery_results = await self._universal_discovery(
                    task_input.project_path or ".", 
                    ["environment", "dependencies", "structure", "code_patterns"]
                )
                
                technology_context = await self._universal_technology_discovery(
                    task_input.project_path or "."
                )
            except Exception:
                self.logger.warning("Could not use unified discovery, using fallback values")
                discovery_results = "{}"
                technology_context = "{}"
            
            # Prepare template variables for rendering (unified approach)
            template_vars = {
                "user_goal": task_input.user_goal or task_input.task_description,
                "project_path": task_input.project_path,
                "project_context": project_context,
                "project_id": task_input.project_id,
                "target_file_path": task_input.target_file_path or "",
                "programming_language": task_input.programming_language or "python",
                "discovery_results": discovery_results,
                "technology_context": technology_context
            }
            
            # Render the user prompt template with variables
            rendered_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            self.logger.info("Using YAML prompt template")
            return rendered_prompt
            
        except Exception as e:
            self.logger.warning(f"Could not load YAML prompt template: {e}, using built-in fallback")
            
            # Built-in fallback prompt
            return f"""You are a code generation agent. Generate complete, working code for this project.

PROJECT CONTEXT:
{project_context}

INSTRUCTIONS:
1. Analyze the project context, user goal, and any existing documentation
2. Determine what files need to be created or modified
3. Generate complete, production-ready code for each file
4. Return a JSON response with this exact structure:

{{
    "files": [
        {{
            "path": "relative/file/path.py",
            "content": "complete file content here",
            "description": "what this file does"
        }}
    ],
    "confidence": 0.85,
    "reasoning": "explanation of your approach and decisions"
}}

REQUIREMENTS:
- Generate COMPLETE, working code (not pseudocode or templates)
- Include proper imports, error handling, and documentation
- Follow best practices for the target language
- Code must be syntactically correct and runnable
- If architect documentation exists, follow their specifications
- Include comprehensive docstrings and comments
- Implement proper logging and error handling

TARGET: {task_input.target_file_path or "Create appropriate files"}
LANGUAGE: {task_input.programming_language or "python"}

Return ONLY the JSON response, no additional text."""

    async def _execute_file_instructions(self, instructions: Dict[str, Any], project_path: str) -> List[Dict[str, Any]]:
        """
        Execute the LLM's file creation instructions using MCP tools.
        """
        generated_files = []
        files_to_create = instructions.get("files", [])
        
        self.logger.info(f"Executing file creation instructions for {len(files_to_create)} files")
        
        for file_instruction in files_to_create:
            file_path = file_instruction.get("path", "")
            content = file_instruction.get("content", "")
            description = file_instruction.get("description", "")
            
            if not file_path or not content:
                self.logger.warning(f"Skipping incomplete file instruction: {file_instruction}")
                continue
                
            try:
                # Ensure path is relative and safe
                safe_path = str(Path(file_path)).replace("..", "")
                full_path = Path(project_path) / safe_path
                
                # Create directory if needed
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file using MCP tools
                write_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": str(full_path),
                    "content": content
                })
                
                if write_result.get("success"):
                    generated_files.append({
                        "file_path": safe_path,
                        "full_path": str(full_path),
                        "content_length": len(content),
                        "description": description,
                        "status": "success"
                    })
                    self.logger.info(f"✓ Generated: {safe_path} ({len(content)} chars)")
                else:
                    generated_files.append({
                        "file_path": safe_path,
                        "status": "error",
                        "error": write_result.get("error", "Write failed")
                    })
                    self.logger.error(f"✗ Failed to write: {safe_path}")
                    
            except Exception as e:
                self.logger.error(f"Error creating file {file_path}: {e}")
                generated_files.append({
                    "file_path": file_path,
                    "status": "error", 
                    "error": str(e)
                })
        
        return generated_files

    async def _fallback_single_file(self, response: str, task_input: SmartCodeGeneratorInput) -> Dict[str, Any]:
        """
        Fallback when LLM doesn't return structured JSON - treat response as single file content.
        """
        try:
            # Determine filename from context
            if task_input.target_file_path:
                filename = task_input.target_file_path
            elif task_input.user_goal:
                # Simple heuristic based on goal
                goal_lower = task_input.user_goal.lower()
                if "cli" in goal_lower:
                    filename = "main.py"
                elif "web" in goal_lower or "api" in goal_lower:
                    filename = "app.py"
                elif "script" in goal_lower:
                    filename = "script.py"
                else:
                    filename = "main.py"
            else:
                filename = "generated_code.py"
            
            # Clean up response (remove markdown formatting if present)
            content = self._clean_code_response(response)
            
            full_path = Path(task_input.project_path) / filename
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": str(full_path),
                "content": content
            })
            
            if write_result.get("success"):
                self.logger.info(f"✓ Generated fallback file: {filename}")
                return {
                    "success": True,
                    "files": [{
                        "file_path": filename,
                        "full_path": str(full_path),
                        "content_length": len(content),
                        "description": "Generated code file (fallback)",
                        "status": "success"
                    }],
                    "confidence": 0.7
                }
            else:
                return {"success": False, "error": "Failed to write fallback file"}
                
        except Exception as e:
            return {"success": False, "error": f"Fallback generation failed: {str(e)}"}

    def _clean_code_response(self, response: str) -> str:
        """Remove markdown formatting from LLM response to get clean code."""
        response = response.strip()
        
        # Remove markdown code blocks
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
        
        return response

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
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
                "simple_architecture": True,
                "llm_powered": True,
                "mcp_tools_integrated": True
            },
            metadata={
                "callable_fn_path": f"{SmartCodeGeneratorAgent_v1.__module__}.{SmartCodeGeneratorAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[SmartCodeGeneratorInput]:
        return SmartCodeGeneratorInput

    def get_output_schema(self) -> Type[SmartCodeGeneratorOutput]:
        return SmartCodeGeneratorOutput 