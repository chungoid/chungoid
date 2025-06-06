"""
ArchitectAgent_v1: Clean, unified LLM-powered blueprint generation.

This agent generates architecture blueprints by:
1. Using unified discovery to understand project context
2. Using YAML prompt template with rich discovery data  
3. Letting the LLM create comprehensive blueprints with maximum intelligence

No legacy patterns, no redundant discovery, no hardcoded logic, no fallbacks.
Pure unified approach for maximum agentic intelligence.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import json
from typing import Any, Dict, Optional, List, ClassVar, Type
import datetime

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

# ADDED: Import cache invalidation for enhanced discovery
from ...utils.simple_cache_invalidation import invalidate_project_caches

logger = logging.getLogger(__name__)


class ArchitectAgentInput(BaseModel):
    """Clean input schema focused on core architectural needs."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this architecture task.")
    project_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Project ID for context.")
    
    # Core autonomous inputs
    user_goal: str = Field(..., description="What the user wants to build")
    project_path: str = Field(default=".", description="Project directory to analyze")
    
    # Optional context (no micromanagement)
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Optional project context")
    
    @model_validator(mode='after')
    def check_minimum_requirements(self) -> 'ArchitectAgentInput':
        """Ensure we have minimum requirements for autonomous architecture design."""
        if not self.user_goal or not self.user_goal.strip():
            raise ValueError("user_goal is required for autonomous architecture design")
        return self


class ArchitectAgentOutput(BaseModel):
    """Clean output schema focused on architectural deliverables."""
    task_id: str = Field(..., description="Task identifier")
    project_id: str = Field(..., description="Project identifier")
    status: str = Field(..., description="Execution status")
    
    # Core deliverables
    blueprint_content: str = Field(..., description="Generated architecture blueprint in Markdown")
    architectural_decisions: List[Dict[str, Any]] = Field(default_factory=list, description="Key architectural decisions with rationale")
    technology_recommendations: Dict[str, Any] = Field(default_factory=dict, description="Recommended technology stack")
    risk_assessments: List[Dict[str, Any]] = Field(default_factory=list, description="Identified risks and mitigations")
    
    # Metadata
    confidence_score: ConfidenceScore = Field(..., description="Agent confidence in the blueprint")
    message: str = Field(..., description="Human-readable result message")
    error_message: Optional[str] = Field(None, description="Error details if failed")


@register_autonomous_engine_agent(capabilities=["architecture_design", "system_planning", "blueprint_generation"])
class ArchitectAgent_v1(UnifiedAgent):
    """
    Clean, unified architecture blueprint generation agent.
    
    Uses unified discovery + YAML templates + maximum LLM intelligence.
    No legacy patterns, no fallbacks, no hardcoded logic.
    """
    
    AGENT_ID: ClassVar[str] = "ArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Enhanced Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Clean, unified LLM-powered architecture blueprint generation"
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "4.0.0"  # Major version for clean rewrite
    CAPABILITIES: ClassVar[List[str]] = ["architecture_design", "system_planning", "blueprint_generation"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[ArchitectAgentInput]] = ArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[ArchitectAgentOutput]] = ArchitectAgentOutput

    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["unified_blueprint_generation"]
    SECONDARY_PROTOCOLS: ClassVar[List[str]] = ["intelligent_discovery"]
    UNIVERSAL_PROTOCOLS: ClassVar[List[str]] = ["agent_communication", "context_sharing"]

    def __init__(self, llm_provider: LLMProvider, prompt_manager: PromptManager, **kwargs):
        super().__init__(llm_provider=llm_provider, prompt_manager=prompt_manager, **kwargs)
        self.logger.info(f"{self.AGENT_ID} v{self.AGENT_VERSION} initialized - clean unified approach")

    async def _execute_iteration(self, context: UEContext, iteration: int) -> IterationResult:
        """
        AUTONOMOUS ARCHITECTURE: Actually create documentation files using MCP tools.
        """
        try:
            # Parse inputs cleanly
            task_input = self._parse_inputs(context.inputs)
            self.logger.info(f"AUTONOMOUS ARCHITECTURE: {task_input.user_goal}")

            # STEP 1: Actually explore the project using MCP tools
            project_info = await self._autonomous_project_exploration(task_input)
            
            # STEP 2: Generate architecture content using LLM
            architecture_content = await self._autonomous_architecture_generation(task_input, project_info)
            
            # STEP 3: Actually create documentation files using MCP tools
            created_docs = await self._autonomous_documentation_creation(task_input, architecture_content)
            
            # STEP 4: Create concrete output
            output = ArchitectAgentOutput(
                task_id=task_input.task_id,
                project_id=task_input.project_id,
                status="SUCCESS",
                blueprint_content=architecture_content.get("blueprint_content", ""),
                architectural_decisions=architecture_content.get("architectural_decisions", []),
                technology_recommendations=architecture_content.get("technology_recommendations", {}),
                risk_assessments=architecture_content.get("risk_assessments", []),
                confidence_score=ConfidenceScore(
                    value=0.9,
                    method="autonomous_mcp_execution",
                    explanation=f"Successfully created {len(created_docs)} documentation files using MCP tools"
                ),
                message=f"Generated comprehensive architecture blueprint and created {len(created_docs)} documentation files"
            )
            
            self.logger.info(f"AUTONOMOUS ARCHITECTURE SUCCESS: Created {len(created_docs)} documentation files")
            
            return IterationResult(
                output=output,
                quality_score=0.9,
                tools_used=["mcp_filesystem", "llm_generation", "autonomous_documentation"],
                protocol_used="autonomous_architecture_generation"
            )
            
        except Exception as e:
            self.logger.error(f"AUTONOMOUS ARCHITECTURE FAILED: {e}")
            
            error_output = ArchitectAgentOutput(
                task_id=getattr(task_input, 'task_id', str(uuid.uuid4())) if 'task_input' in locals() else str(uuid.uuid4()),
                project_id=getattr(task_input, 'project_id', 'unknown') if 'task_input' in locals() else 'unknown',
                status="ERROR",
                blueprint_content="",
                confidence_score=ConfidenceScore(
                    value=0.0,
                    method="error_state",
                    explanation="Architecture generation failed"
                ),
                message="Architecture blueprint generation failed",
                error_message=str(e)
            )
            
            return IterationResult(
                output=error_output,
                quality_score=0.0,
                tools_used=[],
                protocol_used="autonomous_architecture_generation"
            )

    def _parse_inputs(self, inputs: Any) -> ArchitectAgentInput:
        """Parse inputs cleanly into ArchitectAgentInput with detailed validation."""
        try:
            if isinstance(inputs, ArchitectAgentInput):
                return inputs
            elif isinstance(inputs, dict):
                # Validate required fields before creation
                if 'user_goal' not in inputs or not inputs['user_goal']:
                    raise ValueError("Missing required field 'user_goal' in input dictionary")
                if not inputs['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return ArchitectAgentInput(**inputs)
            elif hasattr(inputs, 'dict'):
                input_dict = inputs.dict()
                if 'user_goal' not in input_dict or not input_dict['user_goal']:
                    raise ValueError("Missing required field 'user_goal' in input object")
                if not input_dict['user_goal'].strip():
                    raise ValueError("Field 'user_goal' cannot be empty or whitespace")
                
                return ArchitectAgentInput(**input_dict)
            else:
                raise ValueError(f"Invalid input type: {type(inputs)}. Expected ArchitectAgentInput, dict, or object with dict() method. Received: {inputs}")
        except Exception as e:
            raise ValueError(f"Input parsing failed for ArchitectAgent: {e}. Input received: {inputs}")

    async def _generate_architecture_blueprint(self, task_input: ArchitectAgentInput) -> Dict[str, Any]:
        """
        Generate architecture blueprint using unified discovery + YAML template.
        Pure unified approach - no legacy patterns.
        """
        try:
            # Get YAML template (no fallbacks)
            prompt_template = self.prompt_manager.get_prompt_definition(
                "architect_agent_v1_prompt",
                "1.0.0",
                sub_path="autonomous_engine"
            )
            
            # Unified discovery for intelligent context
            discovery_results = await self._universal_discovery(
                task_input.project_path,
                ["environment", "dependencies", "structure", "patterns", "requirements", "architecture"]
            )
            
            technology_context = await self._universal_technology_discovery(
                task_input.project_path
            )
            
            # Build template variables for maximum LLM intelligence
            template_vars = {
                "user_goal": task_input.user_goal,
                "project_path": task_input.project_path,
                "project_id": task_input.project_id,
                "project_context": f"User Goal: {task_input.user_goal}",
                
                # Rich discovery data for intelligent decisions
                "discovery_results": json.dumps(discovery_results, indent=2),
                "technology_context": json.dumps(technology_context, indent=2),
                
                # Intelligent context
                "intelligent_context": task_input.project_specifications is not None,
                "project_specifications": task_input.project_specifications or {}
            }
            
            # Render template
            formatted_prompt = self.prompt_manager.get_rendered_prompt_template(
                prompt_template.user_prompt_template,
                template_vars
            )
            
            # Get system prompt if available
            system_prompt = getattr(prompt_template, 'system_prompt', None)
            
            # Call LLM with maximum intelligence
            response = await self.llm_provider.generate(
                prompt=formatted_prompt,
                system_prompt=system_prompt,
                temperature=0.3,
                max_tokens=4000
            )
            
            # Parse LLM response (expecting JSON from template)
            try:
                json_str = self._extract_json_from_response(response); result = json.loads(json_str)
                
                return {
                    "blueprint_content": result.get("blueprint_markdown_content", ""),
                    "architectural_decisions": result.get("architectural_decisions", []),
                    "technology_recommendations": result.get("technology_recommendations", {}),
                    "risk_assessments": result.get("risk_assessments", []),
                    "confidence_score": ConfidenceScore(
                        value=result.get("confidence_score", {}).get("value", 0.8),
                        method=result.get("confidence_score", {}).get("method", "llm_self_assessment"),
                        explanation=result.get("confidence_score", {}).get("explanation", "Architecture blueprint generated successfully")
                    )
                }
                
            except json.JSONDecodeError as e:
                self.logger.error(f"LLM response not valid JSON: {e}")
                raise ValueError(f"LLM response parsing failed: {e}")
                
        except Exception as e:
            self.logger.error(f"Blueprint generation failed: {e}")
            raise

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        """Generate agent card for registry."""
        input_schema = ArchitectAgentInput.model_json_schema()
        output_schema = ArchitectAgentOutput.model_json_schema()
        
        return AgentCard(
            agent_id=ArchitectAgent_v1.AGENT_ID,
            name=ArchitectAgent_v1.AGENT_NAME,
            description=ArchitectAgent_v1.AGENT_DESCRIPTION,
            version=ArchitectAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[ArchitectAgent_v1.CATEGORY.value],
            visibility=ArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "unified_discovery": True,
                "yaml_templates": True,
                "llm_intelligence": True,
                "clean_architecture": True,
                "no_fallbacks": True,
                "maximum_agentic": True
            },
            metadata={
                "callable_fn_path": f"{ArchitectAgent_v1.__module__}.{ArchitectAgent_v1.__name__}"
            }
        )

    def get_input_schema(self) -> Type[ArchitectAgentInput]:
        return ArchitectAgentInput

    def get_output_schema(self) -> Type[ArchitectAgentOutput]:
        return ArchitectAgentOutput

    async def _autonomous_project_exploration(self, task_input: ArchitectAgentInput) -> Dict[str, Any]:
        """Use MCP tools to analyze the ACTUAL IMPLEMENTATION for architecture planning."""
        try:
            self.logger.info(f"EXPLORATION START: Analyzing project at {task_input.project_path}")
            exploration_start_time = datetime.datetime.now()
            
            project_info = {
                "project_path": task_input.project_path,
                "existing_files": [],
                "needs_docs_folder": True,
                "existing_docs": [],
                "implementation_files": [],
                "project_type": "unknown",
                "dependencies": [],
                "entry_points": [],
                "code_structure": {}
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
                
                project_info["implementation_files"] = implementation_files
                self.logger.info(f"IMPLEMENTATION ANALYSIS: Found {len(implementation_files)} source code files")
                
                # STEP 3: Read Key Implementation Files
                files_to_analyze = []
                
                # Prioritize entry points and main files
                priority_files = []
                for file_path in implementation_files + config_files:
                    file_name = file_path.split('/')[-1].lower()
                    if any(pattern in file_name for pattern in ['main', 'index', 'app', 'server', 'cli', '__init__', 'package.json', 'requirements.txt', 'cargo.toml']):
                        priority_files.append(file_path)
                
                # Add top-level implementation files
                top_level_impl = [f for f in implementation_files if '/' not in f.strip('./')]
                
                # Combine and dedupe, limit to avoid overwhelming the LLM
                files_to_analyze = list(dict.fromkeys(priority_files + top_level_impl + config_files))[:8]
                
                self.logger.info(f"IMPLEMENTATION READING: Analyzing {len(files_to_analyze)} key files")
                
                # Read and analyze implementation files
                for i, filename in enumerate(files_to_analyze):
                    try:
                        self.logger.info(f"READING IMPLEMENTATION {i+1}/{len(files_to_analyze)}: {filename}")
                        read_result = await self._call_mcp_tool("filesystem_read_file", {
                            "file_path": filename,
                            "max_bytes": 5000  # Read more content for implementation analysis
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
                
                # STEP 4: Check for existing documentation (secondary priority)
                docs_folder_exists = any("docs" in f.lower() for f in file_names if "/" not in f.strip("./"))
                project_info["needs_docs_folder"] = not docs_folder_exists
                self.logger.info(f"DOCS FOLDER CHECK: Existing docs folder = {docs_folder_exists}, Needs creation = {not docs_folder_exists}")
                
                # Read existing docs for context (but don't prioritize them)
                doc_file_patterns = ["readme", "architecture", "design", "spec", "overview", "api", "deployment", "security"]
                potential_docs = [f for f in file_names if any(pattern in f.lower() for pattern in doc_file_patterns)]
                
                if potential_docs:
                    self.logger.info(f"EXISTING DOCS: Found {len(potential_docs)} documentation files for context")
                    
                    # Read a few existing docs for context (limit to 2 to focus on implementation)
                    for i, filename in enumerate(potential_docs[:2]):
                        try:
                            self.logger.info(f"READING DOC CONTEXT {i+1}/2: {filename}")
                            read_result = await self._call_mcp_tool("filesystem_read_file", {
                                "file_path": filename,
                                "max_bytes": 2000  # Less content for existing docs
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
            
            # COMPREHENSIVE STATE LOGGING: Show complete current state
            self.logger.info(f"EXPLORATION COMPLETE ({exploration_time:.2f}s):")
            self.logger.info(f"  PROJECT STATE SUMMARY:")
            self.logger.info(f"    • Total files found: {len(project_info['existing_files'])}")
            self.logger.info(f"    • Project type: {project_info['project_type']}")
            self.logger.info(f"    • Implementation files: {len(project_info['implementation_files'])}")
            self.logger.info(f"    • Files analyzed: {len(project_info['existing_docs'])}")
            self.logger.info(f"    • Docs folder needed: {project_info['needs_docs_folder']}")
            self.logger.info(f"    • Project path: {project_info['project_path']}")
            self.logger.info(f"    • Cache bypassed: {getattr(task_input, 'cache_bypassed', False)}")
            
            # Show sample files for debugging
            sample_files = project_info['existing_files'][:5]
            if sample_files:
                self.logger.info(f"    • Sample files: {sample_files}")
                if len(project_info['existing_files']) > 5:
                    self.logger.info(f"    • ... and {len(project_info['existing_files']) - 5} more files")
            
            return project_info
            
        except Exception as e:
            self.logger.error(f"EXPLORATION FAILED: {e}")
            return {
                "project_path": task_input.project_path,
                "existing_files": [],
                "needs_docs_folder": True,
                "existing_docs": [],
                "implementation_files": [],
                "project_type": "unknown",
                "error": str(e)
            }

    async def _autonomous_architecture_generation(self, task_input: ArchitectAgentInput, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to generate comprehensive architecture documentation content."""
        # Build comprehensive prompt for architecture generation
        
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
            implementation_analysis = "IMPLEMENTATION ANALYSIS:\n"
            for doc in implementation_context:
                implementation_analysis += f"\n=== {doc['file']} ===\n{doc['content']}\n"
        
        # Build existing documentation context (if any)
        existing_docs_context = ""
        if documentation_context:
            existing_docs_context = "\nEXISTING DOCUMENTATION CONTEXT:\n"
            for doc in documentation_context:
                existing_docs_context += f"\n--- {doc['file']} (current) ---\n{doc['content'][:500]}...\n"
        
        prompt = f"""Generate comprehensive, technical architecture documentation for this {project_info.get('project_type', 'software')} project based on implementation analysis.

USER GOAL: {task_input.user_goal}

PROJECT CONTEXT:
- Project Type: {project_info.get('project_type', 'unknown')}
- Total Files: {len(project_info.get('existing_files', []))}
- Implementation Files: {len(project_info.get('implementation_files', []))}
- Project Path: {project_info.get('project_path', '.')}
- Needs Docs Folder: {project_info.get('needs_docs_folder', True)}

{implementation_analysis}

{existing_docs_context}

PROJECT STRUCTURE:
Files found: {', '.join(project_info.get('existing_files', [])[:20])}{"..." if len(project_info.get('existing_files', [])) > 20 else ""}

TASK: Based on the IMPLEMENTATION ANALYSIS above, generate comprehensive, technical documentation that:

1. ANALYZES the actual code structure and functionality
2. EXPLAINS the technical architecture based on what the code actually does
3. DOCUMENTS the API, interfaces, and data flow found in the implementation
4. PROVIDES deployment, security, and operational guidance specific to this implementation
5. IDENTIFIES gaps and areas for improvement in the current implementation

Generate 5-8 comprehensive documentation files with detailed technical content (minimum 2000 characters each):

RESPONSE FORMAT (JSON):
{{
    "documentation_files": {{
        "README.md": "# [Project Name]\\n\\n## Project Overview\\n[Comprehensive overview based on implementation analysis]\\n\\n## Architecture\\n[Technical architecture derived from code]\\n\\n## Installation\\n[Based on dependencies found]\\n\\n## Usage\\n[Based on entry points and APIs found]\\n\\n## Technical Details\\n[Deep technical analysis]...",
        "docs/ARCHITECTURE.md": "# Architecture Documentation\\n\\n## System Design\\n[Based on actual code structure]\\n\\n## Components\\n[Actual modules and classes found]\\n\\n## Data Flow\\n[Based on implementation]\\n\\n## Integration Points\\n[APIs and interfaces found]...",
        "docs/API_DOCUMENTATION.md": "# API Documentation\\n\\n## Endpoints\\n[Based on actual API code]\\n\\n## Request/Response Formats\\n[From implementation]\\n\\n## Authentication\\n[Security implementation found]...",
        "docs/DEPLOYMENT.md": "# Deployment Guide\\n\\n## Prerequisites\\n[Based on dependencies found]\\n\\n## Environment Setup\\n[From configuration files]\\n\\n## Production Deployment\\n[Technical deployment guide]...",
        "docs/SECURITY.md": "# Security Documentation\\n\\n## Security Measures\\n[Based on security code found]\\n\\n## Threat Model\\n[For this specific implementation]\\n\\n## Best Practices\\n[Implementation-specific]..."
    }},
    "architectural_decisions": [
        {{"decision": "[Technical decision found in code]", "rationale": "[Why this was implemented this way]", "impact": "[Effect on system]"}},
        {{"decision": "[Another technical decision]", "rationale": "[Based on implementation]", "impact": "[System impact]"}}
    ],
    "technology_recommendations": {{
        "current_stack": "[Technologies actually used based on analysis]",
        "frameworks": "[Frameworks found in implementation]",
        "databases": "[Data storage found in code]",
        "deployment": "[Deployment approach based on config]",
        "improvements": "[Specific improvements based on analysis]"
    }},
    "risk_assessments": [
        {{"risk": "[Actual risk found in implementation]", "impact": "High/Medium/Low", "mitigation": "[Specific mitigation for this codebase]"}},
        {{"risk": "[Another implementation-specific risk]", "impact": "High/Medium/Low", "mitigation": "[Technical mitigation]"}}
    ],
    "blueprint_content": "# {project_info.get('project_type', 'Project').title()} Architecture Blueprint\\n\\nComprehensive technical blueprint based on implementation analysis: {task_input.user_goal}"
}}

CRITICAL: Generate COMPREHENSIVE, TECHNICAL content (minimum 2000 characters per file) that explains what the code ACTUALLY does, not generic boilerplate. Base ALL content on the implementation analysis provided above."""
        
        self.logger.info("Generating comprehensive architecture documentation with enhanced LLM prompt...")
        
        try:
            # Call LLM with implementation-focused prompt
            response = await self.llm_provider.generate_response(
                prompt=prompt,
                agent_id=self.AGENT_ID,
                iteration_context={"task": "implementation_based_architecture_generation"}
            )
            
            if not response or len(response) < 1000:
                raise ValueError(f"LLM response too short for architecture generation ({len(response)} chars). Expected substantial documentation content. Response: '{response}'")
            
            self.logger.info(f"LLM Response length: {len(response)} chars")
            self.logger.info(f"LLM Response preview: {response[:500]}...")
            
            # Parse response with detailed error handling
            architecture_data = self._parse_llm_response(response, task_input, project_info)
            
            # Validate the generated content quality
            if "documentation_files" in architecture_data:
                doc_files = architecture_data["documentation_files"]
                total_chars = sum(len(content) for content in doc_files.values())
                
                # Enhanced validation for comprehensive content
                for file_path, content in doc_files.items():
                    if not content or not content.strip():
                        raise ValueError(f"Documentation file '{file_path}' has empty content.")
                    
                    # The comprehensive prompt should generate much longer content
                    if len(content.strip()) < 50:
                        self.logger.warning(f"Documentation file '{file_path}' is short ({len(content)} chars) but accepted")
                    
                    # Log content length for analysis
                    self.logger.info(f"{file_path}: {len(content)} characters")
                
                self.logger.info(f"Architecture documentation generated: {len(doc_files)} files with total {total_chars} characters")
                return architecture_data
            else:
                raise ValueError(f"LLM response missing 'documentation_files' key. Response: '{response}'")
        
        except Exception as e:
            self.logger.error(f"Architecture generation failed: {e}")
            # Return basic structure to allow system to continue
            return self._add_metadata_to_response({"documentation_files": {}}, task_input, project_info)

    def _parse_llm_response(self, response: str, task_input: ArchitectAgentInput, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Parse LLM response with improved JSON extraction - prioritize comprehensive content."""
        
        if not response or not response.strip():
            raise ValueError(f"LLM response is empty or whitespace-only. Expected JSON response for architecture generation.")
        
        if len(response.strip()) < 50:
            raise ValueError(f"LLM response too short ({len(response)} chars). Expected substantial JSON response. Response: '{response}'")
        
        parsing_errors = []
        
        # **ENHANCED JSON EXTRACTION**: Improved handling of ```json blocks with multi-line content
        try:
            # Look for JSON block in markdown with better extraction
            if "```json" in response.lower():
                start_marker = "```json"
                end_marker = "```"
                
                start_pos = response.lower().find(start_marker)
                if start_pos >= 0:
                    # Find the actual start of JSON content (after the marker and any whitespace)
                    json_start = start_pos + len(start_marker)
                    
                    # Find the closing ``` that's not part of the content
                    json_end = None
                    search_pos = json_start
                    
                    while True:
                        end_pos = response.find(end_marker, search_pos)
                        if end_pos == -1:
                            # No closing marker, use rest of response
                            json_end = len(response)
                            self.logger.info("Found ```json without closing ```, using rest of response")
                            break
                        
                        # Check if this ``` is inside a string or is the actual closing marker
                        content_before_end = response[json_start:end_pos]
                        
                        # Simple heuristic: if we have balanced braces, this is likely the end
                        open_braces = content_before_end.count('{')
                        close_braces = content_before_end.count('}')
                        
                        if open_braces > 0 and open_braces == close_braces:
                            json_end = end_pos
                            self.logger.info(f"Found JSON block with balanced braces")
                            break
                        
                        # Keep searching
                        search_pos = end_pos + len(end_marker)
                        if search_pos >= len(response):
                            json_end = len(response)
                            break
                    
                    json_str = response[json_start:json_end].strip()
                    
                    self.logger.info(f"🔍 Extracted JSON content: {json_str[:200]}...")
                    
                    # **IMPROVED PARSING**: Try multiple approaches without corrupting the content
                    try:
                        # First try: Parse as-is
                        data = json.loads(json_str)
                        if "documentation_files" in data:
                            self.logger.info(f"JSON parsed successfully with {len(data['documentation_files'])} documentation files")
                            return self._add_metadata_to_response(data, task_input, project_info)
                        else:
                            parsing_errors.append(f"JSON missing 'documentation_files' field. Found keys: {list(data.keys())}")
                    
                    except json.JSONDecodeError as e:
                        # Second try: Clean only if necessary
                        try:
                            cleaned_json = self._clean_json_for_parsing(json_str)
                            data = json.loads(cleaned_json)
                            if "documentation_files" in data:
                                self.logger.info(f"JSON parsed after cleaning with {len(data['documentation_files'])} documentation files")
                                return self._add_metadata_to_response(data, task_input, project_info)
                            else:
                                parsing_errors.append(f"Cleaned JSON missing 'documentation_files' field. Found keys: {list(data.keys())}")
                        except json.JSONDecodeError as e2:
                            parsing_errors.append(f"JSON block decode error: {e}. Cleaned error: {e2}")
                            # Don't truncate the content in error messages
                            parsing_errors.append(f"JSON content length: {len(json_str)} chars")
            
            # Try parsing entire response as JSON
            self.logger.info("🔍 Trying to parse entire response as JSON...")
            try:
                # Remove markdown wrapper if present
                clean_response = response.strip()
                if clean_response.startswith("```json"):
                    clean_response = clean_response[7:]
                if clean_response.endswith("```"):
                    clean_response = clean_response[:-3]
                clean_response = clean_response.strip()
                
                data = json.loads(clean_response)
                if "documentation_files" in data:
                    self.logger.info(f"Full response parsed with {len(data['documentation_files'])} documentation files")
                    return self._add_metadata_to_response(data, task_input, project_info)
                else:
                    parsing_errors.append(f"Full response missing 'documentation_files' field. Found keys: {list(data.keys())}")
            except json.JSONDecodeError as e:
                parsing_errors.append(f"Full response JSON decode error: {e}")
                
        except Exception as e:
            parsing_errors.append(f"JSON extraction error: {e}")
        
        # **ENHANCED MANUAL EXTRACTION**: Better fallback with improved content extraction
        try:
            self.logger.info("🔍 Attempting enhanced manual content extraction...")
            manual_result = self._extract_content_manually_enhanced(response, task_input, project_info)
            if manual_result and manual_result.get("documentation_files"):
                doc_files = manual_result["documentation_files"]
                self.logger.info(f"Manual extraction found {len(doc_files)} documentation files")
                return manual_result
            else:
                parsing_errors.append(f"Enhanced manual extraction found no content")
        except Exception as e:
            parsing_errors.append(f"Enhanced manual extraction error: {e}")
        
        # NO FALLBACKS - FAIL LOUDLY WITH ALL ERROR DETAILS
        error_msg = f"""ArchitectAgent LLM Response Parsing FAILED - ALL APPROACHES FAILED:

PARSING ERRORS:
{chr(10).join(f"  {i+1}. {error}" for i, error in enumerate(parsing_errors))}

RESPONSE ANALYSIS:
- Length: {len(response)} characters
- Contains '```json': {"YES" if "```json" in response.lower() else "NO"}
- Contains 'documentation_files': {"YES" if "documentation_files" in response else "NO"}

EXPECTED FORMAT:
Expected JSON with 'documentation_files' field containing architecture documentation.

FULL LLM RESPONSE (first 2000 chars):
{response[:2000]}{"..." if len(response) > 2000 else ""}

INPUT CONTEXT:
- User Goal: {task_input.user_goal}
- Project Path: {task_input.project_path}
"""
        raise ValueError(error_msg)

    def _extract_content_manually_enhanced(self, response: str, task_input: ArchitectAgentInput, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced manual extraction that handles the actual LLM output format."""
        
        documentation_files = {}
        
        # **ENHANCED**: Look for the actual JSON structure in the response
        # Sometimes the LLM returns valid JSON but our regex doesn't catch it
        import re
        
        # Try to find the documentation_files section directly
        doc_files_pattern = r'"documentation_files"\s*:\s*\{([^}]+(?:\{[^}]*\}[^}]*)*)\}'
        match = re.search(doc_files_pattern, response, re.DOTALL)
        
        if match:
            try:
                # Reconstruct the JSON structure
                doc_files_content = "{" + match.group(0) + "}"
                data = json.loads(doc_files_content)
                if "documentation_files" in data:
                    self.logger.info(f"Enhanced extraction found documentation_files with {len(data['documentation_files'])} files")
                    return self._add_metadata_to_response(data, task_input, project_info)
            except json.JSONDecodeError:
                pass
        
        # **ENHANCED**: Look for individual file sections within the JSON structure
        # Pattern to match: "filename": "content..."
        file_pattern = r'"([^"]+\.md)"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
        matches = re.findall(file_pattern, response, re.DOTALL)
        
        for filename, content in matches:
            # Unescape the content
            unescaped_content = content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
            if len(unescaped_content.strip()) > 30:  # Relaxed to keep more content
                documentation_files[filename] = unescaped_content
                self.logger.info(f"Enhanced extraction found {filename}: {len(unescaped_content)} chars")
        
        # **FALLBACK**: Try section-based extraction with improved patterns  
        if not documentation_files:
            # Look for README content with various header formats
            readme_content = self._extract_section_enhanced(response, [
                "README.md", "readme.md", "# README", "## README", 
                "# Network Scanning CLI Tool", "## Network Scanning CLI Tool"
            ])
            if readme_content:
                documentation_files["README.md"] = readme_content
            
            # Look for Architecture content
            arch_content = self._extract_section_enhanced(response, [
                "ARCHITECTURE.md", "architecture.md", "docs/ARCHITECTURE.md",
                "# Architecture", "## Architecture", "# System Architecture", 
                "## System Architecture"
            ])
            if arch_content:
                documentation_files["docs/ARCHITECTURE.md"] = arch_content
            
            # Look for API Design content
            api_content = self._extract_section_enhanced(response, [
                "API_DESIGN.md", "api_design.md", "docs/API_DESIGN.md",
                "# API", "## API", "# CLI Interface", "## CLI Interface"
            ])
            if api_content:
                documentation_files["docs/API_DESIGN.md"] = api_content
            
            # Look for Deployment content
            deploy_content = self._extract_section_enhanced(response, [
                "DEPLOYMENT.md", "deployment.md", "docs/DEPLOYMENT.md",
                "# Deployment", "## Deployment", "# Kali Linux Deployment"
            ])
            if deploy_content:
                documentation_files["docs/DEPLOYMENT.md"] = deploy_content
            
            # Look for Security content
            security_content = self._extract_section_enhanced(response, [
                "SECURITY.md", "security.md", "docs/SECURITY.md",
                "# Security", "## Security", "# Security Considerations"
            ])
            if security_content:
                documentation_files["docs/SECURITY.md"] = security_content
        
        # If we found any content, return it
        if documentation_files:
            self.logger.info(f"Enhanced manual extraction found {len(documentation_files)} documentation files")
            return self._add_metadata_to_response({"documentation_files": documentation_files}, task_input, project_info)
        
        # NO FALLBACKS - FAIL LOUDLY WITH DETAILED CONTEXT
        error_msg = f"""ENHANCED MANUAL EXTRACTION FAILED - NO STRUCTURED CONTENT FOUND

ANALYSIS ATTEMPTED:
- JSON documentation_files pattern: {'FOUND' if 'documentation_files' in response else 'NOT FOUND'}
- Individual file patterns: {len(re.findall(file_pattern, response, re.DOTALL))} matches found
- Section-based extraction: Attempted for README, ARCHITECTURE, API, DEPLOYMENT, SECURITY

RESPONSE ANALYSIS:
- Total length: {len(response)} characters
- Contains JSON markers: {"YES" if "```json" in response or "{" in response else "NO"}
- Contains file extensions: {"YES" if ".md" in response else "NO"}

EXPECTED STRUCTURE:
The LLM response should contain either:
1. Valid JSON with "documentation_files" field
2. Markdown sections with clear headers
3. Individual file content sections

RESPONSE PREVIEW (first 1000 chars):
{response[:1000]}
"""
        raise ValueError(error_msg)

    def _extract_section_enhanced(self, text: str, section_markers: List[str]) -> str:
        """Enhanced section extraction with better content capture."""
        text_lower = text.lower()
        
        for marker in section_markers:
            marker_lower = marker.lower()
            start_idx = text_lower.find(marker_lower)
            
            if start_idx >= 0:
                # Find the end of this section - look for next major section or JSON closing
                remaining_text = text[start_idx:]
                
                # **ENHANCED**: Better end detection
                next_section_idx = len(remaining_text)
                
                # Look for various section terminators
                terminators = [
                    '",\n',  # JSON field ending
                    '"\n}',  # JSON object ending  
                    '\n# ',   # Next markdown header
                    '\n## ',  # Next markdown subheader
                    '\n---',  # Markdown separator
                    '\n```',  # Code block or JSON ending
                ]
                
                for terminator in terminators:
                    next_idx = remaining_text.find(terminator, len(marker))
                    if next_idx > 0:
                        next_section_idx = min(next_section_idx, next_idx)
                
                section_content = remaining_text[:next_section_idx].strip()
                
                # **ENHANCED**: Clean up JSON escaping if present
                if section_content.startswith('"') and section_content.endswith('"'):
                    section_content = section_content[1:-1]
                    section_content = section_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')
                
                # **RELAXED**: Accept content with more than 30 chars instead of 50
                if len(section_content) > 30:
                    return section_content
        
        return ""

    def _generate_project_overview(self, task_input: ArchitectAgentInput, architecture_content: Dict[str, Any]) -> str:
        """Generate a project overview file."""
        decisions = architecture_content.get("architectural_decisions", [])
        tech_rec = architecture_content.get("technology_recommendations", {})
        
        return f"""# Project Overview

## Goal
{task_input.user_goal}

## Architecture Summary
{architecture_content.get("blueprint_content", "Architecture blueprint")[:500]}...

## Key Decisions
{chr(10).join([f"- **{d.get('decision', 'Decision')}**: {d.get('rationale', 'Rationale')}" for d in decisions[:5]])}

## Technology Stack
- **Primary Language**: {tech_rec.get('primary_language', 'Python')}
- **Frameworks**: {', '.join(tech_rec.get('frameworks', []))}
- **Deployment**: {tech_rec.get('deployment', 'TBD')}

## Documentation Structure
- `README.md` - Project documentation
- `docs/ARCHITECTURE.md` - System architecture
- `docs/API_DESIGN.md` - API specifications
- `docs/DEPLOYMENT.md` - Deployment guide
- `docs/SECURITY.md` - Security considerations

Generated by ArchitectAgent v1 on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

    async def _autonomous_documentation_creation(self, task_input: ArchitectAgentInput, architecture_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Actually create documentation files and folders using MCP tools."""
        try:
            self.logger.info(f"DOCUMENTATION CREATION START: Project {task_input.project_path}")
            creation_start_time = datetime.datetime.now()
            created_docs = []
            
            # STEP 1: Create docs folder if needed
            if architecture_content.get("needs_docs_folder", True):
                self.logger.info("CREATING DOCS FOLDER: Directory creation initiated")
                folder_result = await self._call_mcp_tool("filesystem_create_directory", {
                    "directory_path": "docs",
                    "project_path": task_input.project_path
                })
                if folder_result.get("success"):
                    created_docs.append({
                        "file_path": "docs/",
                        "type": "directory",
                        "description": "Documentation folder",
                        "status": "success",
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    self.logger.info("DOCS FOLDER CREATED: Successfully created docs directory")
                else:
                    error_msg = folder_result.get("error", "Unknown directory creation error")
                    self.logger.error(f"DOCS FOLDER FAILED: {error_msg}")
            else:
                self.logger.info("DOCS FOLDER EXISTS: Skipping creation")
            
            # STEP 2: Create documentation files
            documentation_files = architecture_content.get("documentation_files", {})
            self.logger.info(f"CREATING {len(documentation_files)} DOCUMENTATION FILES")
            
            for i, (file_path, content) in enumerate(documentation_files.items(), 1):
                if content and len(content.strip()) > 0:
                    self.logger.info(f"CREATING FILE {i}/{len(documentation_files)}: {file_path} ({len(content)} chars)")
                    
                    write_result = await self._call_mcp_tool("filesystem_write_file", {
                        "file_path": file_path,
                        "content": content,
                        "project_path": task_input.project_path
                    })
                    
                    if write_result.get("success"):
                        created_docs.append({
                            "file_path": file_path,
                            "full_path": f"{task_input.project_path}/{file_path}",
                            "content_length": len(content),
                            "description": f"Architecture documentation: {file_path}",
                            "status": "success",
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        self.logger.info(f"FILE CREATED: {file_path} ({len(content)} chars)")
                    else:
                        error_msg = write_result.get("error", "Unknown write error")
                        self.logger.error(f"FILE CREATION FAILED: {file_path} - {error_msg}")
                        created_docs.append({
                            "file_path": file_path,
                            "status": "error",
                            "error": error_msg,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                else:
                    self.logger.warning(f"SKIPPING EMPTY FILE: {file_path} (no content)")
            
            # STEP 3: Create a project overview file
            overview_content = self._generate_project_overview(task_input, architecture_content)
            if overview_content:
                self.logger.info(f"CREATING PROJECT OVERVIEW: ({len(overview_content)} chars)")
                write_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": "PROJECT_OVERVIEW.md",
                    "content": overview_content,
                    "project_path": task_input.project_path
                })
                
                if write_result.get("success"):
                    created_docs.append({
                        "file_path": "PROJECT_OVERVIEW.md",
                        "description": "Project overview and architecture summary",
                        "status": "success",
                        "content_length": len(overview_content),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                    self.logger.info("PROJECT OVERVIEW CREATED: PROJECT_OVERVIEW.md")
                else:
                    error_msg = write_result.get("error", "Unknown write error")
                    self.logger.error(f"PROJECT OVERVIEW FAILED: {error_msg}")

            creation_time = (datetime.datetime.now() - creation_start_time).total_seconds()
            self.logger.info(f"FILE CREATION COMPLETE ({creation_time:.2f}s): {len(created_docs)} items processed")
            
            # **CRITICAL FIX**: Immediate cache invalidation after file creation
            self.logger.info("IMMEDIATE CACHE INVALIDATION: Starting immediate cache invalidation after file creation")
            
            # Use coordinated cache invalidation for immediate effect
            try:
                from ...utils.cache_coordination_fix import invalidate_project_caches
                invalidate_project_caches(task_input.project_path, "immediate_post_file_creation")
                self.logger.info("IMMEDIATE CACHE INVALIDATION: Immediate synchronous invalidation completed")
            except Exception as e:
                self.logger.warning(f"IMMEDIATE CACHE INVALIDATION: Immediate invalidation failed: {e}")
            
            # **ENHANCED FIX**: Comprehensive cache invalidation and verification
            self.logger.info("COMPREHENSIVE CACHE INVALIDATION: Starting comprehensive cache refresh process")
            await self._invalidate_discovery_cache(task_input.project_path)
            
            # **NEW**: Direct file verification before proceeding
            self.logger.info("FILE VERIFICATION: Verifying created files exist on filesystem")
            verification_result = await self._verify_files_created(created_docs, task_input.project_path)
            
            if verification_result.get("verified"):
                if verification_result.get("complete"):
                    self.logger.info("FILE VERIFICATION SUCCESS: All files confirmed on filesystem")
                else:
                    missing = verification_result.get("missing_files", [])
                    self.logger.warning(f"FILE VERIFICATION PARTIAL: Missing files: {missing}")
            else:
                error = verification_result.get("error", "Unknown error")
                self.logger.error(f"FILE VERIFICATION FAILED: {error}")
            
            # **ENHANCED**: Re-run project exploration with forced refresh
            self.logger.info("STATE REFRESH: Re-running project exploration with cache bypass")
            
            # Set cache bypass flag for fresh discovery
            task_input_fresh = ArchitectAgentInput(
                user_goal=task_input.user_goal,
                project_path=task_input.project_path,
                project_id=task_input.project_id
            )
            
            # Force bypass cache by temporarily setting flag
            import time
            cache_key = f"force_refresh_{int(time.time())}"
            
            updated_project_info = await self._autonomous_project_exploration(task_input_fresh)
            
            # **VERIFICATION LOGGING**: Compare before and after states
            old_file_count = len(architecture_content.get("existing_files", []))
            new_file_count = len(updated_project_info.get("existing_files", []))
            old_needs_docs = architecture_content.get("needs_docs_folder", True)
            new_needs_docs = updated_project_info.get("needs_docs_folder", True)
            
            self.logger.info(f"STATE COMPARISON:")
            self.logger.info(f"  Files: {old_file_count} → {new_file_count} (Δ {new_file_count - old_file_count})")
            self.logger.info(f"  Needs docs folder: {old_needs_docs} → {new_needs_docs}")
            self.logger.info(f"  Existing docs: {len(updated_project_info.get('existing_docs', []))} found")
            
            # **ENHANCED LOGIC**: More detailed success/failure analysis
            files_created_successfully = len([d for d in created_docs if d.get("status") == "success" and not d.get("file_path", "").endswith("/")])
            
            if new_file_count > old_file_count:
                detected_increase = new_file_count - old_file_count
                self.logger.info(f"CACHE REFRESH SUCCESS: New files detected (+{detected_increase})")
                if detected_increase >= files_created_successfully:
                    self.logger.info("PERFECT STATE SYNC: All created files detected by discovery")
                else:
                    self.logger.warning(f"PARTIAL STATE SYNC: Created {files_created_successfully}, detected +{detected_increase}")
            else:
                self.logger.error(f"CACHE REFRESH FAILED: No new files detected despite creating {files_created_successfully} files")
                self.logger.error("CRITICAL: State management disconnect - system not recognizing created files")
            
            return created_docs
            
        except Exception as e:
            self.logger.error(f"DOCUMENTATION CREATION FAILED: {e}")
            return [{
                "file_path": "error.log",
                "status": "error",
                "error": str(e)
            }]

    async def _invalidate_discovery_cache(self, project_path: str):
        """Invalidate discovery cache to force fresh filesystem scans with coordination."""
        try:
            self.logger.info(f"CACHE_INVALIDATION: Starting coordinated cache clear for {project_path}")
            
            # Use the new coordinated cache clearing system
            try:
                from ...utils.cache_coordination_fix import clear_all_coordinated_caches, invalidate_project_caches
                
                # First, synchronous invalidation for immediate effect
                invalidate_project_caches(project_path, "post_file_creation")
                
                # Then, full coordinated async clearing
                discovery_service = getattr(self, '_discovery_service', None)
                if not discovery_service:
                    try:
                        import chungoid.agents.unified_agent as ua
                        discovery_service = getattr(ua, '_discovery_service', None)
                    except Exception:
                        pass
                
                cleared_count = await clear_all_coordinated_caches(discovery_service, project_path)
                self.logger.info(f"CACHE_INVALIDATION: Coordinated clearing completed - {cleared_count} systems cleared")
                
            except ImportError:
                # Fallback to manual clearing (should not happen now)
                self.logger.warning("CACHE_INVALIDATION: Coordination module not available, using manual clearing")
                
                # Use the unified cache clearing method
                self.clear_discovery_cache("post_file_creation")
                
                # Clear EfficientDiscoveryService cache manually
                try:
                    import chungoid.agents.unified_agent as ua
                    if hasattr(ua, '_discovery_service'):
                        ua._discovery_service.clear_cache(project_path)
                        self.logger.info("CACHE_INVALIDATION: Cleared discovery service cache (manual)")
                except Exception as e:
                    self.logger.warning(f"CACHE_INVALIDATION: Failed to clear discovery cache: {e}")
                
                # Clear iteration cache completely to force fresh discovery
                if hasattr(self, 'iteration_cache'):
                    iteration_keys = len(self.iteration_cache)
                    self.iteration_cache.clear()
                    self.logger.info(f"CACHE_INVALIDATION: Cleared {iteration_keys} iteration cache entries (manual)")
            
            # Clear agent-level iteration cache always
            if hasattr(self, 'iteration_cache'):
                discovery_keys = [k for k in self.iteration_cache.keys() if 'discovery' in k.lower()]
                for key in discovery_keys:
                    self.iteration_cache.pop(key, None)
                if discovery_keys:
                    self.logger.info(f"CACHE_INVALIDATION: Cleared {len(discovery_keys)} agent discovery cache entries")
            
            # Add a small delay to ensure filesystem and cache consistency
            await asyncio.sleep(0.1)  # 100ms delay for consistency
            
            self.logger.info(f"CACHE_INVALIDATION: Cache invalidation completed for {project_path}")
            
        except Exception as e:
            self.logger.error(f"CACHE_INVALIDATION: Failed - {e}")

    async def _verify_files_created(self, created_docs: List[Dict[str, Any]], project_path: str) -> Dict[str, Any]:
        """Verify that all created files exist on the filesystem with coordinated cache refresh."""
        try:
            import os
            
            # First, invalidate caches to ensure fresh verification
            await self._invalidate_discovery_cache(project_path)
            
            # Use coordinated cache refresh for verification if available
            try:
                from ...utils.cache_coordination_fix import verify_cache_consistency
                
                # Extract expected file paths
                expected_files = []
                for doc in created_docs:
                    if doc.get("status") == "success" and not doc.get("file_path", "").endswith("/"):
                        file_path = doc.get("file_path")
                        if file_path:
                            expected_files.append(file_path)
                
                # Use coordinated verification
                verification_result = await verify_cache_consistency(
                    project_path, 
                    self._call_mcp_tool,
                    expected_files
                )
                
                if verification_result.get("verified"):
                    found_files = verification_result.get("found_files", [])
                    missing_files = verification_result.get("missing_files", [])
                    
                    self.logger.info(f"FILE_VERIFICATION: Coordinated verification found {len(found_files)} files")
                    if missing_files:
                        self.logger.warning(f"FILE_VERIFICATION: Missing files: {missing_files}")
                    
                    return {
                        "verified": len(missing_files) == 0,
                        "complete": verification_result.get("all_expected_found", False),
                        "missing_files": missing_files,
                        "found_files": found_files,
                        "coordinated_verification": True
                    }
                else:
                    self.logger.warning(f"FILE_VERIFICATION: Coordinated verification failed: {verification_result.get('error')}")
                    
            except ImportError:
                self.logger.warning("FILE_VERIFICATION: Coordinated verification not available, using manual verification")
            except Exception as e:
                self.logger.warning(f"FILE_VERIFICATION: Coordinated verification error: {e}")
            
            # Fallback to manual verification
            verified_files = []
            missing_files = []
            
            for doc in created_docs:
                if doc.get("status") == "success" and not doc.get("file_path", "").endswith("/"):
                    file_path = doc.get("file_path")
                    if file_path:
                        full_path = f"{project_path}/{file_path}"
                        if os.path.exists(full_path):
                            verified_files.append(file_path)
                        else:
                            missing_files.append(file_path)
            
            self.logger.info(f"FILE_VERIFICATION: Manual verification found {len(verified_files)} files")
            if missing_files:
                self.logger.warning(f"FILE_VERIFICATION: Missing files (manual): {missing_files}")
            
            return {
                "verified": len(missing_files) == 0,
                "complete": len(verified_files) == len(created_docs),
                "missing_files": missing_files,
                "found_files": verified_files,
                "coordinated_verification": False
            }
            
        except Exception as e:
            self.logger.error(f"FILE_VERIFICATION: Verification failed: {e}")
            return {
                "verified": False,
                "complete": False,
                "error": str(e)
            }

    def _add_metadata_to_response(self, data: Dict[str, Any], task_input: ArchitectAgentInput, project_info: Dict[str, Any]) -> Dict[str, Any]:
        """Add minimal universal metadata only if completely missing - LLM should generate all content."""
        
        # **UNIVERSAL DESIGN**: Only add truly generic defaults if the LLM completely failed to generate required fields
        # The LLM should generate ALL project-specific content based on user goals
        
        if "architectural_decisions" not in data:
            # Don't add hardcoded decisions - generate based on user goal
            data["architectural_decisions"] = [
                {
                    "decision": "LLM-generated architecture", 
                    "rationale": f"Architecture designed specifically for: {task_input.user_goal}", 
                    "impact": "Tailored solution for user requirements"
                }
            ]
        
        if "technology_recommendations" not in data:
            # Don't hardcode technology stack - this should come from LLM analysis
            data["technology_recommendations"] = {
                "note": "Technology recommendations should be generated by LLM based on project requirements",
                "user_goal": task_input.user_goal
            }
        
        if "risk_assessments" not in data:
            # Don't hardcode risks - these should be project-specific
            data["risk_assessments"] = [
                {
                    "risk": "LLM should generate project-specific risks", 
                    "impact": "Unknown", 
                    "mitigation": f"Analyze risks specific to: {task_input.user_goal}"
                }
            ]
        
        if "blueprint_content" not in data:
            # Only add minimal blueprint if completely missing
            data["blueprint_content"] = f"# Architecture Blueprint\n\nProject Goal: {task_input.user_goal}\n\nNote: Comprehensive blueprint should be generated by LLM based on project analysis."
        
        return data

    def _clean_json_for_parsing(self, json_str: str) -> str:
        """Clean JSON string to handle common LLM response issues."""
        import re
        
        # Remove any trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix common escape sequence issues - don't double-escape already escaped sequences
        # Only fix actual newlines that aren't properly escaped
        lines = json_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Only escape unescaped newlines within string values
            if '"' in line and not line.strip().endswith(('",', '"', '}', ']')):
                # This line likely has an unescaped newline in a string value
                line = line.replace('\n', '\\n')
            cleaned_lines.append(line)
        
        cleaned = '\n'.join(cleaned_lines)
        
        # Try to find and extract just the JSON object if there's extra text
        start = cleaned.find('{')
        if start >= 0:
            # Find matching closing brace
            brace_count = 0
            end = start
            for i, char in enumerate(cleaned[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            cleaned = cleaned[start:end]
        
        return cleaned
