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
    
    # Core autonomous inputs
    user_goal: str = Field(..., description="What the user wants to build")
    project_path: str = Field(default=".", description="Project directory to generate code in")
    
    # Optional context (no micromanagement)
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Optional project context")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'SmartCodeGeneratorInput':
        """Ensure we have minimum requirements for autonomous code generation."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for autonomous code generation")
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
        """Recursively convert Pydantic objects to dictionaries with validation."""
        try:
            if hasattr(obj, 'dict') and callable(getattr(obj, 'dict')):
                # This is a Pydantic object
                result = obj.dict()
                if not isinstance(result, dict):
                    raise ValueError(f"Pydantic object.dict() returned non-dict: {type(result)}")
                return result
            elif isinstance(obj, dict):
                # Recursively process dictionary values
                return {key: self._convert_pydantic_to_dict(value) for key, value in obj.items()}
            elif isinstance(obj, (list, tuple)):
                # Recursively process list/tuple items
                return [self._convert_pydantic_to_dict(item) for item in obj]
            else:
                # Return as-is for primitive types
                return obj
        except Exception as e:
            raise ValueError(f"Failed to convert Pydantic object to dict: {e}. Object type: {type(obj)}, Object: {obj}")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        AUTONOMOUS CODE GENERATION: Actually use MCP tools to explore and generate code.
        """
        try:
            # Convert inputs with detailed validation
            if not context or not hasattr(context, 'inputs'):
                raise ValueError("ExecutionContext is missing or has no 'inputs' attribute")
            
            if not context.inputs:
                raise ValueError("ExecutionContext.inputs is None or empty")
            
            # Detailed input conversion and validation
            try:
                if isinstance(context.inputs, dict):
                    # Validate required fields
                    if 'user_goal' not in context.inputs:
                        raise ValueError("Missing required field 'user_goal' in input dictionary")
                    if not context.inputs['user_goal'] or not context.inputs['user_goal'].strip():
                        raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                    
                    converted_inputs = self._convert_pydantic_to_dict(context.inputs)
                    task_input = SmartCodeGeneratorInput(**converted_inputs)
                elif hasattr(context.inputs, 'dict'):
                    input_dict = context.inputs.dict()
                    if 'user_goal' not in input_dict:
                        raise ValueError("Missing required field 'user_goal' in input object")
                    if not input_dict['user_goal'] or not input_dict['user_goal'].strip():
                        raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                    
                    task_input = SmartCodeGeneratorInput(**input_dict)
                else:
                    # Fallback assignment
                    if not hasattr(context.inputs, 'user_goal'):
                        raise ValueError(f"Input object missing 'user_goal' attribute. Type: {type(context.inputs)}")
                    
                    task_input = context.inputs
                    
                    # Validate the task_input
                    if not task_input.user_goal or not task_input.user_goal.strip():
                        raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                        
            except Exception as e:
                raise ValueError(f"Input parsing/validation failed for SmartCodeGeneratorAgent: {e}. Context inputs type: {type(context.inputs)}, Context inputs: {context.inputs}")

            self.logger.info(f"🚀 AUTONOMOUS CODE GENERATION: {task_input.user_goal}")

            # STEP 1: Actually explore the project using MCP tools
            project_info = await self._autonomous_project_exploration(task_input)
            
            # STEP 2: Generate code content using LLM
            code_content = await self._autonomous_code_generation(task_input, project_info)
            
            # STEP 3: Actually create files using MCP tools
            created_files = await self._autonomous_file_creation(task_input, code_content)
            
            # STEP 4: Create concrete output with validation
            if not created_files:
                raise ValueError(f"No files were created. User goal: {task_input.user_goal}")
            
            output = SmartCodeGeneratorOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                generated_files=created_files,
                confidence_score=ConfidenceScore(
                    value=0.9,
                    method="autonomous_mcp_execution",
                    explanation=f"Successfully generated {len(created_files)} files using MCP tools"
                )
            )
            
            self.logger.info(f"✅ AUTONOMOUS SUCCESS: Created {len(created_files)} files")
            
            return IterationResult(
                output=output,
                quality_score=0.9,
                tools_used=["mcp_filesystem", "llm_generation", "autonomous_execution"],
                protocol_used="autonomous_code_generation"
            )
            
        except Exception as e:
            error_msg = f"""SmartCodeGeneratorAgent execution failed:

ERROR: {e}

CONTEXT:
- Iteration: {iteration}
- Input Type: {type(context.inputs) if context and hasattr(context, 'inputs') else 'No context/inputs'}
- Input Value: {context.inputs if context and hasattr(context, 'inputs') else 'No context/inputs'}
"""
            self.logger.error(error_msg)
            
            error_output = SmartCodeGeneratorOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                generated_files=[],
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="autonomous_code_generation"
            )

    async def _autonomous_project_exploration(self, task_input: SmartCodeGeneratorInput) -> Dict[str, Any]:
        """Use MCP tools to actually explore the project."""
        try:
            self.logger.info(f"🔍 Exploring project: {task_input.project_path}")
            
            # Use MCP tools to list directory contents
            list_result = await self._call_mcp_tool("filesystem_list_directory", {
                "directory_path": task_input.project_path,
                "project_path": task_input.project_path
            })
            
            project_info = {
                "project_path": task_input.project_path,
                "user_goal": task_input.user_goal,
                "existing_files": [],
                "project_type": "unknown",
                "documentation": []
            }
            
            if list_result.get("success"):
                files = list_result.get("files", [])
                project_info["existing_files"] = [f.get("name", str(f)) if isinstance(f, dict) else str(f) for f in files]
                
                # Read key documentation files
                for file_info in files[:5]:  # Limit to avoid overwhelming
                    filename = file_info.get("name", "") if isinstance(file_info, dict) else str(file_info)
                    
                    if any(doc in filename.lower() for doc in ["readme", "goal", "spec", "requirements"]):
                        try:
                            read_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": filename,
                                "project_path": task_input.project_path
                            })
                            if read_result.get("success"):
                                content = read_result.get("content", "")[:1000]  # Truncate
                                project_info["documentation"].append({
                                    "file": filename,
                                    "content": content
                                })
                                self.logger.info(f"📄 Read documentation: {filename}")
                        except Exception as e:
                            self.logger.warning(f"Could not read {filename}: {e}")
            
            self.logger.info(f"📊 Project exploration complete: {len(project_info['existing_files'])} files found")
            return project_info
            
        except Exception as e:
            self.logger.error(f"Project exploration failed: {e}")
            return {
                "project_path": task_input.project_path,
                "user_goal": task_input.user_goal,
                "existing_files": [],
                "error": str(e)
            }

    async def _autonomous_code_generation(self, task_input: SmartCodeGeneratorInput, project_info: Dict[str, Any]) -> str:
        """Use LLM to generate actual code content with detailed validation."""
        # Build comprehensive prompt
        prompt = f"""Generate complete, working Python code for this project.

USER GOAL: {task_input.user_goal}

PROJECT CONTEXT:
- Project Path: {project_info.get('project_path', '.')}
- Existing Files: {', '.join(project_info.get('existing_files', []))}

DOCUMENTATION FOUND:
{chr(10).join([f"- {doc['file']}: {doc['content'][:200]}..." for doc in project_info.get('documentation', [])])}

REQUIREMENTS:
1. Generate COMPLETE, working Python code (not pseudocode)
2. Include all necessary imports and dependencies
3. Add proper error handling and logging
4. Include command-line argument parsing if needed
5. Make code production-ready and executable
6. Follow Python best practices (PEP 8, type hints, docstrings)

Return ONLY the Python code, no markdown formatting or explanations."""

        self.logger.info("🧠 Generating code with LLM...")
        
        try:
            response = await self.llm_provider.generate(
                prompt=prompt,
                max_tokens=4000,
                temperature=0.1
            )
            
            # Detailed LLM response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for code generation. Prompt length: {len(prompt)} chars. User goal: {task_input.user_goal}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for code generation. Response: '{response}'. User goal: {task_input.user_goal}")
            
            if len(response.strip()) < 50:
                raise ValueError(f"LLM code response too short for meaningful code ({len(response)} chars). Expected substantial Python code. Response: '{response}'. User goal: {task_input.user_goal}")
            
            # Validate it looks like code
            response_lower = response.lower()
            if "import " not in response_lower and "def " not in response_lower and "class " not in response_lower:
                raise ValueError(f"LLM response doesn't appear to contain Python code (no 'import', 'def', or 'class' found). Response: '{response}'. User goal: {task_input.user_goal}")
            
            self.logger.info(f"📝 Code generated: {len(response)} characters")
            self.logger.info(f"📝 Code preview: {response[:200]}...")
            
            return response.strip()
            
        except Exception as e:
            error_msg = f"""SmartCodeGeneratorAgent code generation failed:

ERROR: {e}

PROMPT USED ({len(prompt)} chars):
{prompt}

INPUT CONTEXT:
- User Goal: {task_input.user_goal}
- Project Path: {task_input.project_path}
- Project Info: {project_info}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def _autonomous_file_creation(self, task_input: SmartCodeGeneratorInput, code_content: str) -> List[Dict[str, Any]]:
        """Actually create files using MCP tools with detailed validation."""
        if not code_content or not code_content.strip():
            raise ValueError(f"Cannot create file with empty code content. User goal: {task_input.user_goal}")
        
        created_files = []
        
        # Determine appropriate filename
        filename = self._determine_filename(task_input)
        if not filename or not filename.strip():
            raise ValueError(f"Failed to determine filename for code. User goal: {task_input.user_goal}")
        
        # Clean up code content
        clean_content = self._clean_code_response(code_content)
        
        if not clean_content or len(clean_content.strip()) < 20:
            raise ValueError(f"Code content too short after cleaning ({len(clean_content)} chars). Original: '{code_content}'. Cleaned: '{clean_content}'. User goal: {task_input.user_goal}")
        
        self.logger.info(f"💾 Creating file: {filename}")
        
        try:
            # Use MCP tools to write the file
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": filename,
                "content": clean_content,
                "project_path": task_input.project_path
            })
            
            if not write_result:
                raise ValueError(f"MCP tool returned None/empty result for file creation. Filename: {filename}")
            
            if not write_result.get("success"):
                error_msg = write_result.get("error", "Unknown MCP tool error")
                raise ValueError(f"MCP tool failed to create {filename}: {error_msg}. Write result: {write_result}")
            
            created_files.append({
                "file_path": filename,
                "full_path": f"{task_input.project_path}/{filename}",
                "content_length": len(clean_content),
                "description": f"Generated {filename} for: {task_input.user_goal}",
                "status": "success"
            })
            self.logger.info(f"✅ Successfully created: {filename} ({len(clean_content)} chars)")
            
            # Also create a basic requirements.txt if needed
            if "import " in clean_content and not any("requirements" in f.get("file_path", "") for f in created_files):
                req_file = await self._create_requirements_file(task_input, clean_content)
                if req_file:
                    created_files.append(req_file)
            
            if not created_files:
                raise ValueError(f"No files were created successfully. User goal: {task_input.user_goal}")
            
            return created_files
            
        except Exception as e:
            error_msg = f"""SmartCodeGeneratorAgent file creation failed:

ERROR: {e}

CONTEXT:
- Filename: {filename}
- Content length: {len(clean_content)} chars
- User goal: {task_input.user_goal}
- Project path: {task_input.project_path}

CONTENT PREVIEW:
{clean_content[:500]}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    async def _create_requirements_file(self, task_input: SmartCodeGeneratorInput, code_content: str):
        """Create a basic requirements.txt file."""
        # Basic requirements based on common imports
        requirements = []
        if "requests" in code_content:
            requirements.append("requests")
        if "numpy" in code_content:
            requirements.append("numpy")
        if "pandas" in code_content:
            requirements.append("pandas")
        if "click" in code_content:
            requirements.append("click")
        if "rich" in code_content:
            requirements.append("rich")
        
        if requirements:
            req_content = "\n".join(requirements) + "\n"
        else:
            req_content = "# Add your project dependencies here\n"
        
        write_result = await self._call_mcp_tool("filesystem_write_file", {
            "file_path": "requirements.txt",
            "content": req_content,
            "project_path": task_input.project_path
        })
        
        if write_result.get("success"):
            self.logger.info("📦 Created requirements.txt")
            return {
                "file_path": "requirements.txt",
                "full_path": f"{task_input.project_path}/requirements.txt",
                "description": "Basic requirements file",
                "status": "success"
            }
        else:
            error_msg = write_result.get("error", "Unknown error")
            self.logger.error(f"Failed to create requirements.txt: {error_msg}")
            return None

    def _determine_filename(self, task_input: SmartCodeGeneratorInput) -> str:
        """
        Determine the filename for the generated code based on context.
        """        
        # Infer filename from user goal and language
        language = "python"  # Default for autonomous mode
        
        if task_input.user_goal:
            goal_lower = task_input.user_goal.lower()
            
            # Language-specific filename patterns
            if language.lower() == "python":
                if "cli" in goal_lower or "command" in goal_lower or "scanner" in goal_lower:
                    return "main.py"
                elif "web" in goal_lower or "api" in goal_lower or "server" in goal_lower:
                    return "app.py"
                elif "script" in goal_lower:
                    return "script.py"
                else:
                    return "main.py"
            elif language.lower() == "javascript":
                if "web" in goal_lower or "frontend" in goal_lower:
                    return "app.js"
                elif "server" in goal_lower or "backend" in goal_lower:
                    return "server.js"
                else:
                    return "index.js"
            elif language.lower() == "java":
                return "Main.java"
            elif language.lower() == "cpp" or language.lower() == "c++":
                return "main.cpp"
            elif language.lower() == "c":
                return "main.c"
            else:
                return f"main.{language.lower()}"
        
        # Fallback based on language
        if language.lower() == "python":
            return "main.py"
        elif language.lower() == "javascript":
            return "index.js"
        elif language.lower() == "java":
            return "Main.java"
        else:
            return f"main.{language.lower()}"

    def _clean_code_response(self, response: str) -> str:
        """Remove markdown formatting and extract code from LLM response."""
        response = response.strip()
        
        # First, try to extract from JSON if it looks like JSON
        if response.startswith('{') and response.endswith('}'):
            try:
                json_data = json.loads(response)
                
                # Look for the generated_code field (YAML template format)
                if "generated_code" in json_data:
                    code_content = json_data["generated_code"]
                    self.logger.info("Extracted code from 'generated_code' field in JSON response")
                    return code_content
                
                # Look for other common code fields
                code_fields = ["code", "content", "source", "implementation", "program", "script"]
                for field in code_fields:
                    if field in json_data and isinstance(json_data[field], str):
                        self.logger.info(f"Extracted code from '{field}' field in JSON response")
                        return json_data[field]
                
                # Look for any string field that contains code-like patterns
                for key, value in json_data.items():
                    if isinstance(value, str) and len(value) > 50:
                        # Check for code-like patterns
                        if any(pattern in value for pattern in ["import ", "def ", "class ", "function", "var ", "const ", "#include", "public class"]):
                            self.logger.info(f"Found code-like content in '{key}' field")
                            return value
                            
            except json.JSONDecodeError:
                self.logger.warning("Response looks like JSON but failed to parse, trying markdown extraction")
        
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

    async def _generate_smart_code(self, task_input: SmartCodeGeneratorAgentInput) -> Dict[str, Any]:
        """
        AUTONOMOUS CODE GENERATION
        No hardcoded logic - agent analyzes project and decides what code to generate
        """
        try:
            # Get autonomous code generation prompt
            prompt_template = self.prompt_manager.get_prompt_definition(
                "smart_code_generator_agent_v1_prompt",
                "1.0.0",
                sub_path="autonomous_engine"
            )
            
            # Build context for autonomous code generation
            context_parts = []
            context_parts.append(f"User Goal: {task_input.user_goal}")
            context_parts.append(f"Project Path: {task_input.project_path}")
            
            if task_input.project_specifications:
                context_parts.append(f"Project Specifications: {json.dumps(task_input.project_specifications, indent=2)}")
            
            context_data = "\n".join(context_parts)
            
            # AUTONOMOUS EXECUTION: Let the agent decide what code to generate
            # Available MCP tools: filesystem_*, text_editor_*, web_search, etc.
            formatted_prompt = prompt_template["content"].format(
                context_data=context_data,
                available_mcp_tools=", ".join(self.mcp_tools.keys())
            )
            
            self.logger.info(f"[AUTONOMOUS] Agent generating code autonomously")
            
            # Let the agent work autonomously
            response = await self._call_llm_with_retry(
                messages=[{"role": "user", "content": formatted_prompt}],
                max_tokens=16000,  # Larger for code generation
                temperature=0.1
            )
            
            return {
                "status": "success",
                "code_generation_result": response,
                "agent_mode": "autonomous",
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path
            }
            
        except Exception as e:
            self.logger.error(f"Autonomous code generation failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "agent_mode": "autonomous_failed"
            } 