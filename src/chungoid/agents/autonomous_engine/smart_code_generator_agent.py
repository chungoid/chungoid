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

            self.logger.info(f"AUTONOMOUS CODE GENERATION: {task_input.user_goal}")

            # STEP 1: Actually explore the project using MCP tools
            project_info = await self._autonomous_project_exploration(task_input)
            
            # STEP 2: Generate code content using LLM
            code_content = await self._autonomous_code_generation(task_input, project_info)
            
            # STEP 3: Actually create files using MCP tools
            created_files = await self._autonomous_file_creation(task_input, code_content, project_info)
            
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
            
            self.logger.info(f"AUTONOMOUS SUCCESS: Created {len(created_files)} files")
            
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
        """Use MCP tools to analyze the ACTUAL IMPLEMENTATION for intelligent code generation."""
        try:
            self.logger.info(f"EXPLORATION START: Analyzing project at {task_input.project_path}")
            exploration_start_time = datetime.datetime.now()
            
            project_info = {
                "project_path": task_input.project_path,
                "user_goal": task_input.user_goal,
                "existing_files": [],
                "implementation_files": [],
                "project_type": "unknown",
                "dependencies": [],
                "entry_points": [],
                "code_structure": {},
                "existing_docs": [],
                "needs_requirements": True
            }

            # STEP 1: Universal File Discovery
            list_result = await self._call_mcp_tool("filesystem_list_directory", {
                "directory_path": task_input.project_path,
                "recursive": True,
                "bypass_cache": getattr(task_input, 'cache_bypassed', False)
            })

            if list_result.get("success"):
                all_items = list_result.get("items", [])
                file_names = [item["path"] for item in all_items if item["type"] == "file"]
                
                project_info["existing_files"] = file_names
                self.logger.info(f"FILESYSTEM SCAN: Found {len(file_names)} files")
                
                # STEP 2: Universal Project Type Detection and Implementation Analysis
                implementation_patterns = {
                    "python": [".py", ".pyx", ".pyi"],
                    "javascript": [".js", ".mjs", ".jsx"],
                    "typescript": [".ts", ".tsx"],
                    "rust": [".rs"],
                    "go": [".go"],
                    "cpp": [".cpp", ".cc", ".cxx", ".c", ".h", ".hpp"],
                    "java": [".java"],
                    "csharp": [".cs"],
                    "php": [".php"],
                    "ruby": [".rb"],
                    "swift": [".swift"],
                    "kotlin": [".kt"],
                    "dart": [".dart"],
                    "scala": [".scala"],
                    "clojure": [".clj", ".cljs"],
                    "elixir": [".ex", ".exs"],
                    "haskell": [".hs"],
                    "lua": [".lua"],
                    "perl": [".pl", ".pm"],
                    "r": [".r", ".R"],
                    "matlab": [".m"],
                    "shell": [".sh", ".bash", ".zsh", ".fish"]
                }
                
                config_patterns = [
                    "package.json", "requirements.txt", "Cargo.toml", "go.mod", "pom.xml", 
                    "build.gradle", "composer.json", "Gemfile", "Package.swift", "pubspec.yaml",
                    "project.clj", "mix.exs", "stack.yaml", "Makefile", "CMakeLists.txt",
                    "Dockerfile", "docker-compose.yml", ".env", "config.yaml", "config.json"
                ]
                
                # Detect project type and find implementation files
                project_types_found = {}
                implementation_files = []
                config_files = []
                
                for file_path in file_names:
                    file_lower = file_path.lower()
                    file_name = file_path.split('/')[-1].lower()
                    
                    # Check config files
                    if any(pattern in file_name for pattern in config_patterns):
                        config_files.append(file_path)
                        if "requirements.txt" in file_name:
                            project_info["needs_requirements"] = False
                    
                    # Check implementation files
                    for proj_type, extensions in implementation_patterns.items():
                        if any(file_lower.endswith(ext) for ext in extensions):
                            if proj_type not in project_types_found:
                                project_types_found[proj_type] = []
                            project_types_found[proj_type].append(file_path)
                            implementation_files.append(file_path)
                
                # Determine primary project type
                if project_types_found:
                    primary_type = max(project_types_found.keys(), key=lambda x: len(project_types_found[x]))
                    project_info["project_type"] = primary_type
                    self.logger.info(f"PROJECT TYPE: Detected {primary_type} with {len(project_types_found[primary_type])} files")
                else:
                    # Fallback to python for new projects
                    project_info["project_type"] = "python"
                    self.logger.info("PROJECT TYPE: Defaulting to python for new project")
                
                project_info["implementation_files"] = implementation_files
                self.logger.info(f"IMPLEMENTATION ANALYSIS: Found {len(implementation_files)} source code files")
                
                # STEP 3: Read Key Implementation Files for Context
                files_to_analyze = []
                
                # Prioritize existing implementation files for context
                if implementation_files:
                    # Read existing implementation to understand patterns and extend
                    priority_files = []
                    for file_path in implementation_files + config_files:
                        file_name = file_path.split('/')[-1].lower()
                        if any(pattern in file_name for pattern in ['main', 'index', 'app', 'server', 'cli', '__init__', 'package.json', 'requirements.txt', 'cargo.toml']):
                            priority_files.append(file_path)
                    
                    # Add some top-level implementation files
                    top_level_impl = [f for f in implementation_files if '/' not in f.strip('./')]
                    
                    # Combine and dedupe, limit to reasonable number
                    files_to_analyze = list(dict.fromkeys(priority_files + top_level_impl + config_files))[:6]
                    
                    self.logger.info(f"IMPLEMENTATION READING: Analyzing {len(files_to_analyze)} existing files for context")
                    
                    # Read and analyze implementation files
                    for i, filename in enumerate(files_to_analyze):
                        try:
                            self.logger.info(f"READING IMPLEMENTATION {i+1}/{len(files_to_analyze)}: {filename}")
                            read_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": filename,
                                "max_bytes": 4000  # Read substantial content for implementation analysis
                            })
                            
                            if read_result.get("success"):
                                content = read_result.get("content", "")
                                if content and len(content.strip()) > 20:
                                    project_info["existing_docs"].append({
                                        "file": filename,
                                        "content": content,
                                        "type": "implementation",
                                        "size": len(content)
                                    })
                                    self.logger.info(f"READ SUCCESS: {filename} ({len(content)} chars)")
                            else:
                                self.logger.warning(f"READ FAILED: {filename} - {read_result.get('error', 'Unknown error')}")
                        except Exception as e:
                            self.logger.warning(f"READ EXCEPTION: {filename} - {e}")
                
                # STEP 4: Read Documentation for Additional Context (but don't prioritize)
                doc_file_patterns = ["readme", "goal", "spec", "requirements", "architecture", "design"]
                potential_docs = [f for f in file_names if any(pattern in f.lower() for pattern in doc_file_patterns)]
                
                if potential_docs:
                    self.logger.info(f"DOCUMENTATION CONTEXT: Found {len(potential_docs)} documentation files")
                    
                    # Read a few docs for context (limit to 2 to focus on implementation)
                    for i, filename in enumerate(potential_docs[:2]):
                        try:
                            self.logger.info(f"READING DOC CONTEXT {i+1}/2: {filename}")
                            read_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": filename,
                                "max_bytes": 1500  # Less content for docs
                            })
                            
                            if read_result.get("success"):
                                content = read_result.get("content", "")
                                if content and len(content.strip()) > 20:
                                    project_info["existing_docs"].append({
                                        "file": filename,
                                        "content": content,
                                        "type": "documentation", 
                                        "size": len(content)
                                    })
                                    self.logger.info(f"READ SUCCESS: {filename} ({len(content)} chars)")
                        except Exception as e:
                            self.logger.warning(f"READ EXCEPTION: {filename} - {e}")
            else:
                self.logger.error(f"FILESYSTEM SCAN FAILED: {list_result.get('error', 'Unknown error')}")
            
            exploration_time = (datetime.datetime.now() - exploration_start_time).total_seconds()
            
            # COMPREHENSIVE STATE LOGGING
            self.logger.info(f"EXPLORATION COMPLETE ({exploration_time:.2f}s):")
            self.logger.info(f"  PROJECT STATE SUMMARY:")
            self.logger.info(f"    • Total files found: {len(project_info['existing_files'])}")
            self.logger.info(f"    • Project type: {project_info['project_type']}")
            self.logger.info(f"    • Implementation files: {len(project_info['implementation_files'])}")
            self.logger.info(f"    • Files analyzed: {len(project_info['existing_docs'])}")
            self.logger.info(f"    • Needs requirements: {project_info['needs_requirements']}")
            self.logger.info(f"    • Project path: {project_info['project_path']}")
            
            return project_info
            
        except Exception as e:
            self.logger.error(f"EXPLORATION FAILED: {e}")
            return {
                "project_path": task_input.project_path,
                "user_goal": task_input.user_goal,
                "existing_files": [],
                "implementation_files": [],
                "project_type": "python",
                "existing_docs": [],
                "error": str(e)
            }

    async def _autonomous_code_generation(self, task_input: SmartCodeGeneratorInput, project_info: Dict[str, Any]) -> str:
        """Use LLM to generate code content based on implementation analysis."""
        
        # Separate implementation and documentation context
        implementation_context = []
        documentation_context = []
        
        for doc in project_info.get('existing_docs', []):
            if doc.get('type') == 'implementation':
                implementation_context.append(doc)
            else:
                documentation_context.append(doc)
        
        # Build implementation analysis section
        implementation_analysis = ""
        if implementation_context:
            implementation_analysis = "EXISTING IMPLEMENTATION ANALYSIS:\n"
            for doc in implementation_context:
                implementation_analysis += f"\n=== {doc['file']} ===\n{doc['content']}\n"
        
        # Build documentation context (if any)
        existing_docs_context = ""
        if documentation_context:
            existing_docs_context = "\nDOCUMENTATION CONTEXT:\n"
            for doc in documentation_context:
                existing_docs_context += f"\n--- {doc['file']} ---\n{doc['content'][:300]}...\n"
        
        # Determine what kind of code to generate based on project analysis
        project_type = project_info.get('project_type', 'python')
        has_implementation = len(implementation_context) > 0
        
        if has_implementation:
            task_type = "EXTEND_EXISTING"
            generation_instruction = "EXTEND and IMPROVE the existing implementation"
        else:
            task_type = "CREATE_NEW"
            generation_instruction = "CREATE a new implementation"
        
        # Build comprehensive prompt based on implementation analysis
        prompt = f"""Generate complete, production-ready {project_type} code for this project.

USER GOAL: {task_input.user_goal}

PROJECT CONTEXT:
- Project Type: {project_info.get('project_type', 'unknown')}
- Total Files: {len(project_info.get('existing_files', []))}
- Implementation Files: {len(project_info.get('implementation_files', []))}
- Task Type: {task_type}
- Project Path: {project_info.get('project_path', '.')}

{implementation_analysis}

{existing_docs_context}

TASK: {generation_instruction} based on the analysis above.

REQUIREMENTS:
1. **{task_type} APPROACH**: {'Analyze existing code patterns and extend/improve them' if has_implementation else 'Create new implementation following best practices'}
2. **COMPLETE IMPLEMENTATION**: Generate working, executable code (not pseudocode)
3. **TECHNOLOGY CONSISTENCY**: Use {project_type} and follow its conventions
4. **PRODUCTION READY**: Include proper error handling, logging, and documentation
5. **INTEGRATION**: {'Maintain compatibility with existing code structure' if has_implementation else 'Create well-structured, modular code'}
6. **BEST PRACTICES**: Follow {project_type} coding standards and patterns

{'EXTEND THE EXISTING CODEBASE by:' if has_implementation else 'CREATE NEW CODEBASE with:'}
- {'Adding new functionality to existing modules' if has_implementation else 'Proper module structure and organization'}
- {'Improving existing implementations' if has_implementation else 'Clean, readable code architecture'}
- {'Maintaining existing code style and patterns' if has_implementation else 'Comprehensive error handling and logging'}
- {'Adding missing features or fixing issues' if has_implementation else 'Command-line interface (if applicable)'}

Return ONLY the {project_type} code, no markdown formatting or explanations."""

        self.logger.info("CODE GENERATION: Generating implementation-focused code with LLM...")
        
        try:
            response = await self.llm_provider.generate_response(
                prompt=prompt,
                agent_id=self.AGENT_ID,
                iteration_context={"task": "implementation_based_code_generation"}
            )
            
            # Enhanced LLM response validation
            if not response:
                raise ValueError(f"LLM provider returned None/empty response for code generation. Prompt length: {len(prompt)} chars. User goal: {task_input.user_goal}")
            
            if not response.strip():
                raise ValueError(f"LLM provider returned whitespace-only response for code generation. Response: '{response}'. User goal: {task_input.user_goal}")
            
            if len(response.strip()) < 50:
                raise ValueError(f"LLM code response too short for meaningful code ({len(response)} chars). Expected substantial {project_type} code. Response: '{response}'. User goal: {task_input.user_goal}")
            
            # Validate it looks like code based on project type
            response_lower = response.lower()
            validation_keywords = {
                "python": ["import ", "def ", "class ", "if __name__"],
                "javascript": ["function", "const ", "let ", "import", "require"],
                "typescript": ["function", "const ", "let ", "import", "interface", "type"],
                "rust": ["fn ", "use ", "struct", "impl"],
                "go": ["func ", "package ", "import"],
                "java": ["class ", "public ", "import"],
                "cpp": ["#include", "int main", "class ", "namespace"],
                "csharp": ["class ", "using ", "namespace"],
                "php": ["<?php", "function", "class"],
                "ruby": ["def ", "class ", "require"],
                "shell": ["#!/bin/", "function", "if ["]
            }
            
            expected_keywords = validation_keywords.get(project_type, ["import ", "def ", "class "])
            if not any(keyword in response_lower for keyword in expected_keywords):
                raise ValueError(f"LLM response doesn't appear to contain {project_type} code (no expected keywords found). Expected: {expected_keywords}. Response: '{response}'. User goal: {task_input.user_goal}")
            
            self.logger.info(f"CODE GENERATED: {len(response)} characters of {project_type} code")
            self.logger.info(f"CODE PREVIEW: {response[:200]}...")
            
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

    async def _autonomous_file_creation(self, task_input: SmartCodeGeneratorInput, code_content: str, project_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Actually create files using MCP tools with implementation-focused intelligence."""
        if not code_content or not code_content.strip():
            raise ValueError(f"Cannot create file with empty code content. User goal: {task_input.user_goal}")
        
        created_files = []
        
        # Determine appropriate filename based on project analysis
        filename = self._determine_filename_smart(task_input, project_info)
        if not filename or not filename.strip():
            raise ValueError(f"Failed to determine filename for code. User goal: {task_input.user_goal}")
        
        # Clean up code content
        clean_content = self._clean_code_response(code_content)
        
        if not clean_content or len(clean_content.strip()) < 20:
            raise ValueError(f"Code content too short after cleaning ({len(clean_content)} chars). Original: '{code_content}'. Cleaned: '{clean_content}'. User goal: {task_input.user_goal}")
        
        self.logger.info(f"FILE CREATION: Creating {filename} ({len(clean_content)} chars)")
        
        try:
            # Use MCP tools to write the file
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": filename,
                "content": clean_content
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
                "status": "success",
                "project_type": project_info.get("project_type", "unknown")
            })
            self.logger.info(f"FILE CREATED: {filename} ({len(clean_content)} chars)")
            
            # Create additional files based on project type and needs
            additional_files = await self._create_additional_files(task_input, project_info, clean_content)
            created_files.extend(additional_files)
            
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
- Project type: {project_info.get('project_type', 'unknown')}

CONTENT PREVIEW:
{clean_content[:500]}
"""
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _determine_filename_smart(self, task_input: SmartCodeGeneratorInput, project_info: Dict[str, Any]) -> str:
        """Determine filename based on project analysis and type detection."""
        project_type = project_info.get("project_type", "python")
        existing_files = project_info.get("existing_files", [])
        
        # Check if we're extending existing files or creating new ones
        has_implementation = len(project_info.get("implementation_files", [])) > 0
        
        if has_implementation:
            # Find the main implementation file to extend or create a complementary file
            impl_files = project_info.get("implementation_files", [])
            main_files = [f for f in impl_files if any(pattern in f.lower() for pattern in ['main', 'app', 'index', 'server', 'cli'])]
            
            if main_files:
                # Create a complementary file or module
                base_name = main_files[0].split('.')[0]
                if project_type == "python":
                    return f"{base_name}_enhanced.py"
                elif project_type == "javascript":
                    return f"{base_name}_enhanced.js"
                else:
                    return f"{base_name}_enhanced.{self._get_extension(project_type)}"
        
        # Create new main file based on project type and user goal
        goal_lower = task_input.user_goal.lower() if task_input.user_goal else ""
        
        if project_type == "python":
            if "cli" in goal_lower or "command" in goal_lower or "scanner" in goal_lower:
                return "main.py"
            elif "web" in goal_lower or "api" in goal_lower or "server" in goal_lower:
                return "app.py"
            elif "test" in goal_lower:
                return "test_main.py"
            else:
                return "main.py"
        elif project_type == "javascript":
            if "web" in goal_lower or "frontend" in goal_lower:
                return "app.js"
            elif "server" in goal_lower or "backend" in goal_lower:
                return "server.js"
            else:
                return "index.js"
        elif project_type == "typescript":
            if "web" in goal_lower or "frontend" in goal_lower:
                return "app.ts"
            elif "server" in goal_lower or "backend" in goal_lower:
                return "server.ts"
            else:
                return "index.ts"
        else:
            return f"main.{self._get_extension(project_type)}"

    def _get_extension(self, project_type: str) -> str:
        """Get file extension for project type."""
        extensions = {
            "python": "py",
            "javascript": "js",
            "typescript": "ts",
            "rust": "rs",
            "go": "go",
            "java": "java",
            "cpp": "cpp",
            "csharp": "cs",
            "php": "php",
            "ruby": "rb",
            "swift": "swift",
            "kotlin": "kt",
            "dart": "dart",
            "scala": "scala",
            "shell": "sh"
        }
        return extensions.get(project_type, "txt")

    async def _create_additional_files(self, task_input: SmartCodeGeneratorInput, project_info: Dict[str, Any], main_content: str) -> List[Dict[str, Any]]:
        """Create additional files based on project needs."""
        additional_files = []
        project_type = project_info.get("project_type", "python")
        
        # Create requirements file if needed and it's a Python project
        if project_type == "python" and project_info.get("needs_requirements", True):
            req_file = await self._create_requirements_file_smart(task_input, main_content)
            if req_file:
                additional_files.append(req_file)
        
        # Create package.json if needed and it's a JavaScript/TypeScript project
        elif project_type in ["javascript", "typescript"] and not any("package.json" in f for f in project_info.get("existing_files", [])):
            package_file = await self._create_package_json(task_input, project_info)
            if package_file:
                additional_files.append(package_file)
        
        return additional_files

    async def _create_requirements_file_smart(self, task_input: SmartCodeGeneratorInput, code_content: str):
        """Create an intelligent requirements.txt file based on actual imports."""
        # Analyze imports in the code
        requirements = set()
        
        lines = code_content.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('import ') or line.startswith('from '):
                # Extract package names from imports
                if 'import ' in line:
                    parts = line.split('import ')
                    if len(parts) > 1:
                        package = parts[1].split()[0].split('.')[0]
                        # Map common packages to their pip names
                        pip_mappings = {
                            'requests': 'requests',
                            'numpy': 'numpy',
                            'pandas': 'pandas',
                            'flask': 'Flask',
                            'django': 'Django',
                            'rich': 'rich',
                            'click': 'click',
                            'asyncio': '',  # Built-in
                            'json': '',     # Built-in
                            'os': '',       # Built-in
                            'sys': '',      # Built-in
                            'datetime': '', # Built-in
                            'typing': '',   # Built-in
                            'nmap': 'python-nmap',
                            'subprocess': '' # Built-in
                        }
                        
                        if package in pip_mappings and pip_mappings[package]:
                            requirements.add(pip_mappings[package])
        
        # Add common requirements based on user goal
        goal_lower = task_input.user_goal.lower() if task_input.user_goal else ""
        if "cli" in goal_lower or "command" in goal_lower:
            requirements.add("click>=8.0.0")
        if "web" in goal_lower or "api" in goal_lower:
            requirements.add("flask>=2.0.0")
        if "scanner" in goal_lower or "network" in goal_lower:
            requirements.add("python-nmap>=0.7.1")
            requirements.add("requests>=2.28.0")
        
        if not requirements:
            return None  # Don't create empty requirements file
        
        requirements_content = '\n'.join(sorted(requirements)) + '\n'
        
        try:
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": "requirements.txt",
                "content": requirements_content
            })
            
            if write_result.get("success"):
                self.logger.info("CREATED requirements.txt with intelligent dependencies")
                return {
                    "file_path": "requirements.txt",
                    "full_path": f"{task_input.project_path}/requirements.txt",
                    "content_length": len(requirements_content),
                    "description": "Intelligent requirements file based on code analysis",
                    "status": "success"
                }
            else:
                error_msg = write_result.get("error", "Unknown error")
                self.logger.error(f"Failed to create requirements.txt: {error_msg}")
                return None
        except Exception as e:
            self.logger.error(f"Exception creating requirements.txt: {e}")
            return None

    async def _create_package_json(self, task_input: SmartCodeGeneratorInput, project_info: Dict[str, Any]):
        """Create a basic package.json for JavaScript/TypeScript projects."""
        project_name = task_input.project_path.split('/')[-1] or "my-project"
        
        package_content = {
            "name": project_name,
            "version": "1.0.0",
            "description": task_input.user_goal or "Generated project",
            "main": "index.js",
            "scripts": {
                "start": "node index.js",
                "dev": "nodemon index.js",
                "test": "jest"
            },
            "dependencies": {},
            "devDependencies": {
                "nodemon": "^2.0.0",
                "jest": "^28.0.0"
            }
        }
        
        # Add dependencies based on user goal
        goal_lower = task_input.user_goal.lower() if task_input.user_goal else ""
        if "web" in goal_lower or "api" in goal_lower:
            package_content["dependencies"]["express"] = "^4.18.0"
        if "database" in goal_lower or "mongo" in goal_lower:
            package_content["dependencies"]["mongoose"] = "^6.0.0"
        
        package_json_content = json.dumps(package_content, indent=2)
        
        try:
            write_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": "package.json",
                "content": package_json_content
            })
            
            if write_result.get("success"):
                self.logger.info("CREATED package.json with intelligent dependencies")
                return {
                    "file_path": "package.json",
                    "full_path": f"{task_input.project_path}/package.json",
                    "content_length": len(package_json_content),
                    "description": "Package configuration for JavaScript/TypeScript project",
                    "status": "success"
                }
        except Exception as e:
            self.logger.error(f"Exception creating package.json: {e}")
            return None

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

    async def _generate_smart_code(self, task_input: SmartCodeGeneratorInput) -> Dict[str, Any]:
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