from __future__ import annotations

import logging
import datetime # For potential timestamping
import uuid
import json # For LOPRD content if it's retrieved as JSON string
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, ClassVar
import time

from pydantic import BaseModel, Field, validator

from chungoid.agents.unified_agent import UnifiedAgent
from ...utils.llm_provider import LLMProvider
from ...utils.prompt_manager import PromptManager, PromptRenderError
from ...schemas.common import ConfidenceScore
from ...utils.agent_registry import AgentCard # For AgentCard
from ...utils.agent_registry_meta import AgentCategory, AgentVisibility # For AgentCard
from chungoid.registry import register_autonomous_engine_agent
from ...schemas.unified_execution_schemas import (
     ExecutionContext as UEContext,
     AgentExecutionResult,
     ExecutionMetadata,
     ExecutionMode,
     CompletionReason,
     IterationResult,
     StageInfo,
 )
from ...utils.chromadb_migration_utils import migrate_store_artifact, migrate_retrieve_artifact

logger = logging.getLogger(__name__)

# MIGRATED: Collection constants moved here from PCMA
LOPRD_ARTIFACTS_COLLECTION = "loprd_artifacts_collection"
BLUEPRINT_ARTIFACTS_COLLECTION = "blueprint_artifacts_collection"
EXECUTION_PLAN_ARTIFACTS_COLLECTION = "execution_plan_artifacts_collection"
ARTIFACT_TYPE_PROJECT_BLUEPRINT_MD = "ProjectBlueprint_MD"
ARTIFACT_TYPE_LOPRD_JSON = "LOPRD_JSON"
ARTIFACT_TYPE_EXECUTION_PLAN_JSON = "ExecutionPlan_JSON"

ARCHITECT_AGENT_PROMPT_NAME = "architect_agent_v1_prompt.yaml" # In server_prompts/autonomous_engine/

class ProtocolExecutionError(Exception):
    """Raised when protocol execution fails."""
    pass

# --- Input and Output Schemas for the Enhanced Agent --- #

class EnhancedArchitectAgentInput(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this Blueprint generation task.")
    project_id: Optional[str] = Field(None, description="Identifier for the current project.")
    
    # Traditional fields - optional when using intelligent context
    loprd_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the LOPRD (JSON artifact) to be used as input.")
    existing_blueprint_doc_id: Optional[str] = Field(None, description="ChromaDB ID of an existing Blueprint to refine, if any.")
    refinement_instructions: Optional[str] = Field(None, description="Specific instructions for refining an existing Blueprint.")
    cycle_id: Optional[str] = Field(None, description="The ID of the current refinement cycle, passed by ARCA for lineage tracking.")
    
    # NEW: Enhanced workflow control
    output_blueprint_files: bool = Field(default=True, description="Whether to output blueprint files to filesystem like code generator")
    generate_execution_plan: bool = Field(default=True, description="Whether to generate execution plan from blueprint")
    output_directory: Optional[str] = Field(None, description="Directory to output blueprint files (defaults to ./blueprints/)")
    
    # ADDED: Intelligent project analysis from orchestrator
    project_specifications: Optional[Dict[str, Any]] = Field(None, description="Intelligent project specifications from orchestrator analysis")
    intelligent_context: bool = Field(default=False, description="Whether intelligent project specifications are provided")
    user_goal: Optional[str] = Field(None, description="Original user goal for context")
    project_path: Optional[str] = Field(None, description="Project path for context")
    
    @validator('loprd_doc_id')
    def validate_traditional_or_intelligent_context(cls, v, values):
        """Ensure either traditional fields or intelligent context is provided."""
        intelligent_context = values.get('intelligent_context', False)
        user_goal = values.get('user_goal')
        project_specifications = values.get('project_specifications')
        
        if intelligent_context:
            # When using intelligent context, user_goal and project_specifications are required
            if not user_goal:
                raise ValueError("user_goal is required when intelligent_context=True")
            if not project_specifications:
                raise ValueError("project_specifications is required when intelligent_context=True")
            # loprd_doc_id is optional in this case
            return v
        else:
            # When not using intelligent context, loprd_doc_id is required
            if not v:
                raise ValueError("loprd_doc_id is required when intelligent_context=False")
            return v

class EnhancedArchitectAgentOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id from input.")
    project_id: str = Field(..., description="Echoed project_id from input.")
    
    # Blueprint artifacts
    blueprint_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated/updated Project Blueprint (Markdown) is stored.")
    blueprint_files: Dict[str, str] = Field(default_factory=dict, description="Generated blueprint files saved to filesystem {file_path: content}")
    
    # Review results
    review_results: Dict[str, Any] = Field(default_factory=dict, description="Self-review analysis of the generated blueprint")
    
    # Execution plan
    execution_plan_document_id: Optional[str] = Field(None, description="ChromaDB document ID where the generated execution plan (JSON) is stored.")
    execution_plan_generated: bool = Field(default=False, description="Whether execution plan was successfully generated")
    
    # Status and metadata
    status: str = Field(..., description="Status of the Blueprint generation (e.g., SUCCESS, FAILURE_LLM, FAILURE_INPUT_RETRIEVAL).")
    message: str = Field(..., description="A message detailing the outcome.")
    confidence_score: Optional[ConfidenceScore] = Field(None, description="Agent's confidence in the quality and completeness of the Blueprint.")
    error_message: Optional[str] = Field(None, description="Error message if status is not SUCCESS.")
    llm_full_response: Optional[str] = Field(None, description="Full raw response from the LLM for debugging.")
    usage_metadata: Optional[Dict[str, Any]] = Field(None, description="Token usage or other metadata from the LLM call.")

@register_autonomous_engine_agent(capabilities=["architecture_design", "system_planning", "blueprint_generation", "blueprint_review", "execution_planning", "file_generation"])
class EnhancedArchitectAgent_v1(UnifiedAgent):
    """
    ENHANCED ARCHITECT AGENT - Complete Blueprint Lifecycle Management
    
    Consolidates the capabilities of:
    - ArchitectAgent_v1: Blueprint generation from LOPRD  
    - BlueprintReviewerAgent_v1: Blueprint quality review and optimization
    - BlueprintToFlowAgent_v1: Execution plan generation from blueprints
    
    Generates technical blueprints, performs self-review, outputs files like code generator,
    and converts blueprints to execution plans - eliminating agent redundancy.
    
    PURE UAEI ARCHITECTURE - Unified Agent Execution Interface only.
    MCP TOOL INTEGRATION - Uses ChromaDB MCP tools for artifact management.
    """
    
    AGENT_ID: ClassVar[str] = "EnhancedArchitectAgent_v1"
    AGENT_NAME: ClassVar[str] = "Enhanced Architect Agent v1"
    AGENT_DESCRIPTION: ClassVar[str] = "Complete blueprint lifecycle: generation, review, file output, and execution planning."
    PROMPT_TEMPLATE_NAME: ClassVar[str] = "architect_agent_v1_prompt.yaml"
    AGENT_VERSION: ClassVar[str] = "2.0.0"
    CAPABILITIES: ClassVar[List[str]] = ["architecture_design", "system_planning", "blueprint_generation", "blueprint_review", "execution_planning", "file_generation", "complex_analysis"]
    CATEGORY: ClassVar[AgentCategory] = AgentCategory.PLANNING_AND_DESIGN 
    VISIBILITY: ClassVar[AgentVisibility] = AgentVisibility.PUBLIC
    INPUT_SCHEMA: ClassVar[Type[EnhancedArchitectAgentInput]] = EnhancedArchitectAgentInput
    OUTPUT_SCHEMA: ClassVar[Type[EnhancedArchitectAgentOutput]] = EnhancedArchitectAgentOutput
    
    PRIMARY_PROTOCOLS: ClassVar[List[str]] = ["enhanced_architecture_planning", "blueprint_lifecycle_management"]
    SECONDARY_PROTOCOLS: ClassVar[list[str]] = ["quality_validation", "execution_planning"]
    UNIVERSAL_PROTOCOLS: ClassVar[list[str]] = ['agent_communication', 'context_sharing', 'tool_validation']

    def __init__(self, 
                 llm_provider: LLMProvider,
                 prompt_manager: PromptManager,
                 system_context: Optional[Dict[str, Any]] = None,
                 agent_id: Optional[str] = None,
                 **kwargs):
        # Enable refinement capabilities for intelligent architecture design
        super().__init__(
            llm_provider=llm_provider,
            prompt_manager=prompt_manager,
            system_context=system_context,
            agent_id=agent_id or self.AGENT_ID,
            enable_refinement=True,  # Enable intelligent refinement
            **kwargs
        )
        
        self._llm_provider = llm_provider
        self._prompt_manager = prompt_manager
        
        self.logger.info(f"{self.AGENT_ID} (v{self.AGENT_VERSION}) initialized as Enhanced UAEI agent with full blueprint lifecycle.")

    async def _execute_iteration(
        self, 
        context: UEContext, 
        iteration: int
    ) -> IterationResult:
        """
        Enhanced UAEI implementation - Complete blueprint lifecycle workflow.
        
        Runs the complete enhanced architecture design workflow:
        1. Discovery: Retrieve LOPRD and analyze requirements
        2. Analysis: Analyze system requirements and constraints
        3. Planning: Plan architecture approach and components
        4. Design: Create detailed blueprint structure
        5. Review: Self-review blueprint quality and completeness (NEW)
        6. Output: Save blueprint files to filesystem (NEW)
        7. Flow Generation: Convert blueprint to execution plan (NEW)
        8. Validation: Final validation of all outputs
        """
        self.logger.info(f"[EnhancedArchitect] Starting iteration {iteration + 1} - Full blueprint lifecycle")
        
        # Convert inputs to proper type
        if isinstance(context.inputs, dict):
            inputs = EnhancedArchitectAgentInput(**context.inputs)
        elif isinstance(context.inputs, EnhancedArchitectAgentInput):
            inputs = context.inputs
        else:
            # Fallback for other types - use dict conversion to preserve all defaults
            input_dict = context.inputs.dict() if hasattr(context.inputs, 'dict') else {}
            # Ensure minimum required fields
            if "project_id" not in input_dict:
                input_dict["project_id"] = "default"
            if "loprd_doc_id" not in input_dict and not input_dict.get("intelligent_context", False):
                input_dict["loprd_doc_id"] = ""
            
            inputs = EnhancedArchitectAgentInput(**input_dict)
        
        try:
            # Phase 1: Discovery - Retrieve LOPRD
            self.logger.info("Phase 1: LOPRD discovery")
            
            # Check if we have intelligent project specifications from orchestrator
            if inputs.project_specifications and inputs.intelligent_context:
                self.logger.info("Using intelligent project specifications from orchestrator")
                loprd_data = self._extract_loprd_from_intelligent_specs(inputs.project_specifications, inputs.user_goal)
            else:
                self.logger.info("Using traditional LOPRD retrieval")
                loprd_data = await self._discover_loprd(inputs)
            
            # Phase 2: Analysis - Analyze requirements
            self.logger.info("Phase 2: Requirements analysis")
            requirements = await self._analyze_requirements(loprd_data, inputs)
            
            # Phase 3: Planning - Plan architecture
            self.logger.info("Phase 3: Architecture planning") 
            architecture_plan = await self._plan_architecture(requirements, inputs)
            
            # Phase 4: Design - Create blueprint
            self.logger.info("Phase 4: Blueprint design")
            blueprint = await self._design_blueprint(architecture_plan, inputs)
            
            # Phase 5: Review - Self-review blueprint (NEW)
            self.logger.info("Phase 5: Blueprint self-review")
            review_results = await self._review_blueprint(blueprint, requirements, inputs)
            
            # Phase 6: Output - Save blueprint files (NEW)
            self.logger.info("Phase 6: Blueprint file output")
            blueprint_files = {}
            if inputs.output_blueprint_files:
                blueprint_files = await self._create_project_structure(blueprint, inputs)
                
                # Also create the actual blueprint documentation files
                blueprint_doc_files = await self._create_blueprint_files(blueprint, inputs)
                blueprint_files.update(blueprint_doc_files)
            
            # Phase 7: Flow Generation - Convert to execution plan (NEW)
            self.logger.info("Phase 7: Execution plan generation")
            execution_plan_result = {}
            if inputs.generate_execution_plan:
                execution_plan_result = await self._generate_execution_plan(blueprint, review_results, inputs)
            
            # Store blueprint artifact in ChromaDB
            blueprint_doc_id = f"blueprint_{inputs.task_id}_{uuid.uuid4().hex[:8]}"
            try:
                await migrate_store_artifact(
                    collection_name=BLUEPRINT_ARTIFACTS_COLLECTION,
                    document_id=blueprint_doc_id,
                    content=blueprint,
                    metadata={
                        "agent_id": self.AGENT_ID,
                        "artifact_type": ARTIFACT_TYPE_PROJECT_BLUEPRINT_MD,
                        "project_id": inputs.project_id or "unknown",
                        "loprd_source": inputs.loprd_doc_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "enhanced_workflow": True,
                        "review_completed": True,
                        "files_generated": len(blueprint_files),
                        "execution_plan_generated": execution_plan_result.get("success", False)
                    }
                )
                self.logger.info(f"Stored enhanced blueprint artifact with ID: {blueprint_doc_id}")
            except Exception as e:
                self.logger.error(f"Failed to store blueprint artifact: {e}")
                blueprint_doc_id = f"failed_storage_{uuid.uuid4().hex[:8]}"
            
            # Phase 8: Validation - Final validation
            self.logger.info("Phase 8: Final validation")
            final_validation = await self._validate_enhanced_output(blueprint, review_results, blueprint_files, execution_plan_result, inputs)
            
            # Calculate quality score
            quality_score = self._calculate_enhanced_quality_score(final_validation, blueprint, review_results, execution_plan_result)
            
            # Create enhanced output
            output = EnhancedArchitectAgentOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id or "unknown",
                blueprint_document_id=blueprint_doc_id,
                blueprint_files=blueprint_files,
                review_results=review_results,
                execution_plan_document_id=execution_plan_result.get("execution_plan_doc_id"),
                execution_plan_generated=execution_plan_result.get("success", False),
                status="SUCCESS",
                message="Enhanced architecture blueprint lifecycle completed successfully",
                confidence_score=ConfidenceScore(
                    value=quality_score,
                    method="enhanced_lifecycle_validation",
                    explanation=f"Quality based on blueprint design, self-review, file generation, and execution planning"
                ),
                usage_metadata={
                    "iteration": iteration + 1,
                    "phases_executed": ["discovery", "analysis", "planning", "design", "review", "output", "flow_generation", "validation"],
                    "components_designed": len(blueprint.get("components", [])),
                    "files_generated": len(blueprint_files),
                    "review_issues_found": len(review_results.get("issues_found", [])),
                    "execution_plan_stages": execution_plan_result.get("stages_count", 0)
                }
            )
            
            tools_used = ["architecture_discovery", "requirements_analysis", "blueprint_design", "self_review", "file_generation", "execution_planning", "enhanced_validation"]
            
        except Exception as e:
            self.logger.error(f"EnhancedArchitectAgent iteration failed: {e}")
            
            # Create error output
            output = EnhancedArchitectAgentOutput(
                task_id=inputs.task_id,
                project_id=inputs.project_id or "unknown",
                blueprint_document_id=None,
                blueprint_files={},
                review_results={"error": str(e)},
                execution_plan_document_id=None,
                execution_plan_generated=False,
                status="FAILURE",
                message=f"Enhanced architecture design failed: {str(e)}",
                error_message=str(e),
                confidence_score=ConfidenceScore(
                    value=0.1,
                    method="error_fallback",
                    explanation=f"Execution failed: {str(e)}"
                )
            )
            
            quality_score = 0.1
            tools_used = []
        
        # Return iteration result for Phase 3 multi-iteration support
        return IterationResult(
            output=output,
            quality_score=quality_score,
            tools_used=tools_used,
            protocol_used=self.PRIMARY_PROTOCOLS[0] if self.PRIMARY_PROTOCOLS else "enhanced_architecture_planning"
        )

    def _extract_loprd_from_intelligent_specs(self, project_specs: Dict[str, Any], user_goal: str) -> Dict[str, Any]:
        """Extract LOPRD-like data from intelligent project specifications."""
        
        # Convert intelligent project specifications to LOPRD format
        loprd_content = {
            "project_overview": {
                "name": project_specs.get("project_type", "Unknown Project"),
                "description": user_goal[:200] + "..." if len(user_goal) > 200 else user_goal,
                "type": project_specs.get("project_type", "unknown"),
                "target_audience": ["End users", "Developers"]
            },
            "functional_requirements": [
                f"Implement {tech} functionality" for tech in project_specs.get("technologies", [])[:5]
            ],
            "non_functional_requirements": [
                {"category": "Performance", "requirement": "System should be responsive"},
                {"category": "Security", "requirement": "Secure data handling"},
                {"category": "Scalability", "requirement": f"Support {project_specs.get('target_platforms', ['multiple platforms'])}"}
            ],
            "user_stories": [
                {
                    "id": "US001",
                    "as_a": "user",
                    "i_want": f"to use a {project_specs.get('project_type', 'tool')}",
                    "so_that": "I can accomplish my goals efficiently"
                }
            ],
            "technical_specifications": {
                "architecture": "modular",
                "technologies": project_specs.get("technologies", []),
                "dependencies": project_specs.get("required_dependencies", []),
                "deployment": f"Compatible with {', '.join(project_specs.get('target_platforms', ['Linux']))}"
            }
        }
        
        return {
            "status": "SUCCESS",
            "content": loprd_content,
            "source": "intelligent_specifications"
        }

    async def _discover_loprd(self, inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Discover and retrieve LOPRD artifact."""
        
        # ENHANCED: Use universal MCP tool access for intelligent LOPRD discovery
        if self.enable_refinement:
            self.logger.info("[MCP] Using universal MCP tool access for intelligent LOPRD discovery")
            
            # Get ALL available tools (no filtering)
            tool_discovery = await self._get_all_available_mcp_tools()
            
            if tool_discovery["discovery_successful"]:
                all_tools = tool_discovery["tools"]
                
                # Use ChromaDB tools for LOPRD retrieval with enhanced context
                loprd_result = {}
                if "chromadb_query_documents" in all_tools:
                    self.logger.info("[MCP] Using ChromaDB for enhanced LOPRD retrieval")
                    loprd_result = await self._call_mcp_tool(
                        "chromadb_query_documents",
                        {
                            "query": f"document_id:{inputs.loprd_doc_id} project_id:{inputs.project_id}",
                            "collection": LOPRD_ARTIFACTS_COLLECTION,
                            "limit": 1
                        }
                    )
                
                # Use content tools for deeper analysis
                structure_analysis = {}
                if "web_content_extract" in all_tools and loprd_result.get("success"):
                    self.logger.info("[MCP] Using content extraction for project structure analysis")
                    structure_analysis = await self._call_mcp_tool(
                        "web_content_extract",
                        {
                            "content": str(loprd_result.get("structure", {})),
                            "extraction_type": "text"
                        }
                    )
                
                # Use intelligence tools for architecture strategy
                intelligence_analysis = {}
                if "adaptive_learning_analyze" in all_tools:
                    self.logger.info("[MCP] Using adaptive_learning_analyze for architecture strategy")
                    intelligence_analysis = await self._call_mcp_tool(
                        "adaptive_learning_analyze",
                        {
                            "context": {
                                "loprd_data": loprd_result,
                                "structure_analysis": structure_analysis,
                                "project_id": inputs.project_id
                            }, 
                            "domain": "architecture_design"
                        }
                    )
                
                # Use filesystem tools for project context
                project_context = {}
                if "filesystem_project_scan" in all_tools:
                    self.logger.info("[MCP] Using filesystem_project_scan for project context")
                    project_context = await self._call_mcp_tool(
                        "filesystem_project_scan",
                        {"scan_path": f"./projects/{inputs.project_id}"}
                    )
                
                # Combine MCP tool results for enhanced LOPRD discovery
                if any([loprd_result.get("success"), structure_analysis.get("success"), intelligence_analysis.get("success")]):
                    self.logger.info("[MCP] Successfully enhanced LOPRD discovery with MCP tools")
                    return {
                        "status": "SUCCESS",
                        "content": loprd_result.get("result", {}),
                        "enhanced_analysis": {
                            "structure_analysis": structure_analysis,
                            "intelligence_analysis": intelligence_analysis,
                            "project_context": project_context
                        },
                        "source": "mcp_enhanced_discovery"
                    }
        
        try:
            loprd_result = await migrate_retrieve_artifact(
                collection_name=LOPRD_ARTIFACTS_COLLECTION,
                document_id=inputs.loprd_doc_id,
                project_id=inputs.project_id or "unknown"
            )
            
            if loprd_result["status"] != "SUCCESS":
                raise Exception(f"Failed to retrieve LOPRD: {loprd_result.get('error')}")
            
            return loprd_result
            
        except Exception as e:
            self.logger.error(f"LOPRD discovery failed: {e}")
            # Return minimal fallback data
            return {
                "status": "FALLBACK",
                "content": {
                    "project_overview": "Basic project requirements",
                    "functional_requirements": ["Core functionality"],
                    "non_functional_requirements": ["Performance", "Security"]
                }
            }

    async def _enhanced_discovery_with_universal_tools(self, inputs: EnhancedArchitectAgentInput, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Universal tool access pattern for ArchitectAgent."""
        
        # 1. Get ALL available tools (no filtering)
        tool_discovery = await self._get_all_available_mcp_tools()
        
        if not tool_discovery["discovery_successful"]:
            self.logger.error("[MCP] Tool discovery failed - falling back to limited functionality")
            return {"error": "Tool discovery failed", "limited_functionality": True}
        
        all_tools = tool_discovery["tools"]
        
        # 2. Intelligent tool selection based on context
        selected_tools = self._intelligently_select_tools(all_tools, inputs, shared_context)
        
        # 3. Use ChromaDB tools for LOPRD and historical architecture patterns
        loprd_analysis = {}
        if "chromadb_query_documents" in selected_tools:
            loprd_analysis = await self._call_mcp_tool(
                "chromadb_query_documents",
                {"query": f"document_id:{inputs.loprd_doc_id} project_id:{inputs.project_id}", "limit": 1}
            )
        
        # 4. Use intelligence tools for architecture strategy
        intelligence_analysis = {}
        if "adaptive_learning_analyze" in selected_tools:
            intelligence_analysis = await self._call_mcp_tool(
                "adaptive_learning_analyze",
                {"context": loprd_analysis, "domain": self.AGENT_ID}
            )
        
        # 5. Use content tools for deeper LOPRD analysis
        content_analysis = {}
        if "web_content_extract" in selected_tools and loprd_analysis.get("success"):
            content_analysis = await self._call_mcp_tool(
                "web_content_extract",
                {
                    "content": str(loprd_analysis.get("structure", {})),
                    "extraction_type": "text"
                }
            )
        
        # 6. Use filesystem tools for project structure analysis
        project_structure = {}
        if "filesystem_project_scan" in selected_tools:
            project_structure = await self._call_mcp_tool(
                "filesystem_project_scan",
                {"scan_path": shared_context.get("project_root_path", ".")}
            )
        
        # 7. Use terminal tools for environment validation
        environment_info = {}
        if "terminal_get_environment" in selected_tools:
            environment_info = await self._call_mcp_tool(
                "terminal_get_environment",
                {}
            )
        
        # 8. Use tool discovery for architecture recommendations
        tool_recommendations = {}
        if "get_tool_composition_recommendations" in selected_tools:
            tool_recommendations = await self._call_mcp_tool(
                "get_tool_composition_recommendations",
                {"context": {"agent_id": self.AGENT_ID, "task_type": "architecture_design"}}
            )
        
        # 9. Combine all analyses
        return {
            "universal_tool_access": True,
            "tools_available": len(all_tools),
            "tools_selected": len(selected_tools),
            "tool_categories": tool_discovery["categories"],
            "loprd_analysis": loprd_analysis,
            "intelligence_analysis": intelligence_analysis,
            "content_analysis": content_analysis,
            "project_structure": project_structure,
            "environment_info": environment_info,
            "tool_recommendations": tool_recommendations,
            "agent_domain": self.AGENT_ID,
            "analysis_timestamp": time.time()
        }

    def _intelligently_select_tools(self, all_tools: Dict[str, Any], inputs: Any, shared_context: Dict[str, Any]) -> Dict[str, Any]:
        """Intelligent tool selection - agents choose which tools to use."""
        
        # Start with core tools every agent should consider
        core_tools = [
            "filesystem_project_scan",
            "chromadb_query_documents", 
            "terminal_get_environment"
        ]
        
        # Add architecture-specific tools
        architecture_tools = [
            "web_content_extract",
            "get_tool_composition_recommendations",
            "chromadb_query_collection"
        ]
        core_tools.extend(architecture_tools)
        
        # Add intelligence tools for all agents
        intelligence_tools = [
            "adaptive_learning_analyze",
            "get_real_time_performance_analysis",
            "generate_performance_recommendations"
        ]
        core_tools.extend(intelligence_tools)
        
        # Select available tools
        selected = {}
        for tool_name in core_tools:
            if tool_name in all_tools:
                selected[tool_name] = all_tools[tool_name]
        
        self.logger.info(f"[MCP] Selected {len(selected)} tools for {getattr(self, 'AGENT_ID', 'unknown_agent')}")
        return selected

    async def _analyze_requirements(self, loprd_data: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Analyze requirements from LOPRD data."""
        loprd_content = loprd_data.get("content", {})
        
        # Extract requirements from LOPRD
        functional_reqs = loprd_content.get("functional_requirements", [])
        non_functional_reqs = loprd_content.get("non_functional_requirements", [])
        
        # Analyze complexity and constraints
        complexity_score = len(functional_reqs) + len(non_functional_reqs) * 0.5
        
        return {
            "functional_requirements": functional_reqs,
            "non_functional_requirements": non_functional_reqs,
            "complexity_score": complexity_score,
            "stakeholder_needs": loprd_content.get("stakeholder_requirements", []),
            "constraints": loprd_content.get("constraints", []),
            "analysis_confidence": 0.85
        }
    
    async def _plan_architecture(self, requirements: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Use LLM brain to intelligently plan architecture based on requirements."""
        
        architecture_prompt = f"""Analyze these requirements and intelligently design architecture.

Requirements Analysis:
{json.dumps(requirements, indent=2)}

Context:
- User Goal: {inputs.user_goal or 'Not specified'}
- Project Type: {inputs.project_specifications.get('project_type') if inputs.project_specifications else 'Unknown'}

Task: Design intelligent architecture that matches the actual requirements. Consider:
1. What architecture pattern truly fits these requirements?
2. What technology stack makes sense for this specific project?
3. What components logically emerge from the functional requirements?
4. What deployment approach suits the project scope?

Return JSON matching this schema:
{{
  "architecture_pattern": "intelligent pattern choice with reasoning",
  "technology_stack": {{
    "backend": "appropriate choice",
    "database": "fits requirements", 
    "cache": "if needed",
    "frontend": "if applicable",
    "deployment": "makes sense"
  }},
  "component_breakdown": [
    {{
      "name": "meaningful component name",
      "responsibility": "clear responsibility from requirements",
      "interfaces": ["appropriate interfaces"],
      "dependencies": ["logical dependencies"]
    }}
  ],
  "design_decisions": ["technical decisions with reasoning"]
}}

Be intelligent and context-aware, not formulaic."""

        try:
            response = await self.llm_provider.generate(architecture_prompt)
            json_content = self._extract_json_from_response(response)
            return json.loads(json_content) if json_content else {}
        except Exception as e:
            self.logger.error(f"LLM architecture planning failed: {e}")
            raise ProtocolExecutionError(f"Architecture planning failed: {e}")
    
    def _break_down_components(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """This method is obsolete - LLM handles component breakdown in _plan_architecture."""
        return []
    
    async def _design_blueprint(self, architecture_plan: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Use LLM brain to create detailed blueprint structure."""
        
        blueprint_prompt = f"""Create detailed technical blueprint from this architecture plan.

Architecture Plan:
{json.dumps(architecture_plan, indent=2)}

Context:
- User Goal: {inputs.user_goal or 'Not specified'}
- Project Path: {inputs.project_path or 'Not specified'}

Task: Design a comprehensive blueprint with a detailed directory structure. Consider:
1. What directory structure fits the chosen architecture?
2. What deployment strategy suits the technology choices?
3. What testing approach matches the architecture complexity?
4. What documentation is needed?

Return JSON with this structure:
{{
  "title": "descriptive project title",
  "architecture_pattern": "{architecture_plan.get('architecture_pattern')}",
  "technology_stack": {architecture_plan.get('technology_stack')},
  "components": {architecture_plan.get('component_breakdown')},
  "directory_structure": {{"intelligent": "directory layout"}},
  "deployment_strategy": {{"appropriate": "deployment approach"}},
  "testing_strategy": {{"suitable": "testing framework choices"}},
  "documentation": {{"needed": "documentation types"}}
}}

Design intelligently based on actual project needs."""

        try:
            response = await self.llm_provider.generate(blueprint_prompt)
            json_content = self._extract_json_from_response(response)
            return json.loads(json_content) if json_content else {}
        except Exception as e:
            self.logger.error(f"LLM blueprint design failed: {e}")
            raise ProtocolExecutionError(f"Blueprint design failed: {e}")
    
    def _plan_deployment_strategy(self, pattern: str) -> Dict[str, Any]:
        """This method is obsolete - LLM handles deployment planning in _design_blueprint."""
        return {}
    
    def _plan_testing_strategy(self, pattern: str) -> Dict[str, Any]:
        """This method is obsolete - LLM handles testing strategy in _design_blueprint."""
        return {}
    
    async def _validate_design(self, blueprint: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Validate the architecture design quality."""
        validation_issues = []
        completeness_score = 1.0
        
        # Check required sections
        required_sections = ["architecture_pattern", "technology_stack", "components", "directory_structure"]
        for section in required_sections:
            if section not in blueprint or not blueprint[section]:
                validation_issues.append(f"Missing {section}")
                completeness_score -= 0.2
        
        # Check component count
        components = blueprint.get("components", [])
        if len(components) == 0:
            validation_issues.append("No components defined")
            completeness_score -= 0.3
        
        # Check technology stack completeness  
        tech_stack = blueprint.get("technology_stack", {})
        if not tech_stack.get("backend"):
            validation_issues.append("Backend technology not specified")
            completeness_score -= 0.2
        
        return {
            "is_valid": len(validation_issues) == 0,
            "validation_issues": validation_issues,
            "completeness_score": max(0.1, completeness_score),
            "quality_metrics": {
                "component_count": len(components),
                "technology_choices": len(tech_stack),
                "documentation_coverage": len(blueprint.get("documentation", {}))
            }
        }
    
    def _calculate_quality_score(self, validation_result: Dict[str, Any], blueprint: Dict[str, Any], requirements: Dict[str, Any]) -> float:
        """Calculate quality score based on validation and blueprint completeness."""
        base_score = 1.0
        
        # Deduct for validation issues
        if not validation_result.get("is_valid", False):
            issues_count = len(validation_result.get("validation_issues", []))
            base_score -= 0.2 * min(issues_count, 3)  # Max 0.6 deduction
        
        # Apply completeness score
        completeness_score = validation_result.get("completeness_score", 1.0)
        base_score *= completeness_score
        
        # Bonus for comprehensive design
        components_count = len(blueprint.get("components", []))
        if components_count >= 3:
            base_score += 0.1
            
        # Bonus for good technology choices
        tech_stack = blueprint.get("technology_stack", {})
        if len(tech_stack) >= 3:
            base_score += 0.1
            
        return max(0.1, min(base_score, 1.0))

    async def _review_blueprint(self, blueprint: Dict[str, Any], requirements: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """NEW: Self-review blueprint quality and completeness."""
        self.logger.info("Performing blueprint self-review")
        
        review_results = {
            "review_completed": True,
            "timestamp": datetime.datetime.now().isoformat(),
            "issues_found": [],
            "strengths_identified": [],
            "optimization_suggestions": [],
            "completeness_scores": {},
            "overall_review_score": 0.0
        }
        
        try:
            # Check completeness
            required_sections = ["title", "architecture_pattern", "technology_stack", "components", "directory_structure"]
            completeness_scores = {}
            
            for section in required_sections:
                if section in blueprint and blueprint[section]:
                    if isinstance(blueprint[section], (list, dict)):
                        completeness_scores[section] = 1.0 if blueprint[section] else 0.0
                    else:
                        completeness_scores[section] = 1.0 if str(blueprint[section]).strip() else 0.0
                else:
                    completeness_scores[section] = 0.0
                    review_results["issues_found"].append(f"Missing or empty section: {section}")
            
            review_results["completeness_scores"] = completeness_scores
            
            # Check component quality
            components = blueprint.get("components", [])
            if len(components) == 0:
                review_results["issues_found"].append("No components defined in architecture")
            elif len(components) < 2:
                review_results["optimization_suggestions"].append("Consider breaking down functionality into more components")
            else:
                review_results["strengths_identified"].append(f"Well-structured with {len(components)} components")
            
            # Check technology choices
            tech_stack = blueprint.get("technology_stack", {})
            if not tech_stack.get("backend"):
                review_results["issues_found"].append("Backend technology not specified")
            if not tech_stack.get("database"):
                review_results["optimization_suggestions"].append("Consider specifying database technology")
            
            # Check directory structure
            dir_structure = blueprint.get("directory_structure", {})
            if not dir_structure:
                review_results["issues_found"].append("No directory structure provided")
            elif len(dir_structure) < 3:
                review_results["optimization_suggestions"].append("Directory structure could be more detailed")
            else:
                review_results["strengths_identified"].append("Comprehensive directory structure provided")
            
            # Calculate overall review score
            avg_completeness = sum(completeness_scores.values()) / len(completeness_scores) if completeness_scores else 0.0
            issues_penalty = len(review_results["issues_found"]) * 0.1
            review_results["overall_review_score"] = max(0.0, avg_completeness - issues_penalty)
            
            self.logger.info(f"Blueprint self-review completed. Score: {review_results['overall_review_score']:.2f}")
            
        except Exception as e:
            self.logger.error(f"Blueprint self-review failed: {e}")
            review_results["issues_found"].append(f"Review process error: {str(e)}")
            review_results["overall_review_score"] = 0.5
        
        return review_results

    async def _create_project_structure(self, blueprint: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, str]:
        """Create actual project structure using MCP tools based on architecture design."""
        self.logger.info("Creating actual project structure using MCP filesystem tools")
        
        created_files = {}
        project_path = inputs.project_path or "."
        
        try:
            # 1. Create main project directories
            directories = [
                f"{project_path}/src",
                f"{project_path}/tests", 
                f"{project_path}/docs",
                f"{project_path}/config"
            ]
            
            for directory in directories:
                self.logger.info(f"[MCP] Creating directory: {directory}")
                result = await self._call_mcp_tool("filesystem_create_directory", {
                    "directory_path": directory,
                    "create_parents": True
                })
                if result.get("success"):
                    self.logger.info(f"[MCP] Created directory: {directory}")
            
            # 2. Create requirements.txt based on technology stack
            tech_stack = blueprint.get("technology_stack", {})
            
            # Use LLM to generate requirements.txt content intelligently
            requirements_prompt = f"""Generate a requirements.txt file for a {blueprint.get('architecture_pattern', 'modern')} Python project.

Technology Stack:
{json.dumps(tech_stack, indent=2)}

Project Context:
- Architecture: {blueprint.get('architecture_pattern', 'layered')}
- Components: {len(blueprint.get('components', []))}
- Deployment: {blueprint.get('deployment_strategy', {}).get('type', 'standard')}

Generate ONLY the requirements.txt content with appropriate versions. Include:
- Core dependencies for the specified technology stack
- Development dependencies for testing and linting
- Production-ready versions (not 'latest')
- Common Python project dependencies

Return just the file content, no markdown formatting."""

            try:
                requirements_response = await self.llm_provider.generate(requirements_prompt)
                requirements_content = requirements_response.strip()
            except Exception as e:
                self.logger.error(f"LLM generation failed for requirements.txt: {e}")
                raise ProtocolExecutionError(f"Requirements.txt generation failed: {e}")
            
            req_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/requirements.txt",
                "content": requirements_content
            })
            if req_result.get("success"):
                created_files["requirements.txt"] = requirements_content
                self.logger.info("[MCP] Created requirements.txt")
            
            # 3. Create README.md
            # Use LLM to generate README.md content intelligently
            readme_prompt = f"""Generate a comprehensive README.md for this project.

Project Details:
- Title: {blueprint.get('title', 'Project')}
- Architecture: {blueprint.get('architecture_pattern', 'layered')}
- Technology Stack: {json.dumps(tech_stack, indent=2)}
- Components: {[comp.get('name', 'Component') for comp in blueprint.get('components', [])]}

Include these sections:
1. Project title and description
2. Features/capabilities
3. Installation instructions
4. Usage examples
5. Architecture overview
6. Contributing guidelines
7. License information

Make it professional and comprehensive. Return just the markdown content."""

            try:
                readme_response = await self.llm_provider.generate(readme_prompt)
                readme_content = readme_response.strip()
            except Exception as e:
                self.logger.error(f"LLM generation failed for README.md: {e}")
                raise ProtocolExecutionError(f"README.md generation failed: {e}")
            
            readme_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/README.md", 
                "content": readme_content
            })
            if readme_result.get("success"):
                created_files["README.md"] = readme_content
                self.logger.info("[MCP] Created README.md")
            
            # 4. Create main entry point skeleton
            # Use LLM to generate main.py entry point intelligently
            main_prompt = f"""Generate a main.py entry point for this project.

Project Context:
- Architecture: {blueprint.get('architecture_pattern', 'layered')}
- Technology Stack: {json.dumps(tech_stack, indent=2)}
- Components: {[comp.get('name', 'Component') for comp in blueprint.get('components', [])]}

Requirements:
- Professional Python code structure
- Proper imports and error handling
- Configuration management
- Entry point that can be run with `python src/main.py`
- Include docstrings and type hints
- Follow Python best practices

Return just the Python code, no markdown formatting."""

            try:
                main_response = await self.llm_provider.generate(main_prompt)
                main_content = main_response.strip()
            except Exception as e:
                self.logger.error(f"LLM generation failed for main.py: {e}")
                raise ProtocolExecutionError(f"Main.py generation failed: {e}")
            
            main_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/src/main.py",
                "content": main_content
            })
            if main_result.get("success"):
                created_files["src/main.py"] = main_content
                self.logger.info("[MCP] Created src/main.py")
            
            # 5. Create __init__.py files
            init_content = "# Package initialization\n"
            
            init_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/src/__init__.py",
                "content": init_content
            })
            if init_result.get("success"):
                created_files["src/__init__.py"] = init_content
            
            # 6. Create component skeleton files based on architecture
            components = blueprint.get("components", [])
            for component in components:
                comp_name = component.get("name", "component").lower().replace(" ", "_")
                
                # Use LLM to generate component skeleton intelligently
                component_prompt = f"""Generate a Python module for this component.

Component Details:
- Name: {component.get('name', 'Component')}
- Responsibility: {component.get('responsibility', 'Component functionality')}
- Interfaces: {component.get('interfaces', [])}
- Dependencies: {component.get('dependencies', [])}

Project Context:
- Architecture: {blueprint.get('architecture_pattern', 'layered')}
- Technology Stack: {json.dumps(tech_stack, indent=2)}

Requirements:
- Professional Python class structure
- Proper docstrings and type hints
- Error handling and logging
- Follow the specified responsibility
- Include necessary imports
- Implement the specified interfaces
- Follow Python best practices

Return just the Python code, no markdown formatting."""

                try:
                    component_response = await self.llm_provider.generate(component_prompt)
                    comp_content = component_response.strip()
                except Exception as e:
                    self.logger.error(f"LLM generation failed for {comp_name}.py: {e}")
                    raise ProtocolExecutionError(f"Component {comp_name}.py generation failed: {e}")
                
                comp_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": f"{project_path}/src/{comp_name}.py",
                    "content": comp_content
                })
                if comp_result.get("success"):
                    created_files[f"src/{comp_name}.py"] = comp_content
                    self.logger.info(f"[MCP] Created src/{comp_name}.py")
            
            # 7. Create .gitignore
            # Use LLM to generate .gitignore content intelligently
            gitignore_prompt = f"""Generate a comprehensive .gitignore file for this project.

Technology Stack:
{json.dumps(tech_stack, indent=2)}

Project Context:
- Architecture: {blueprint.get('architecture_pattern', 'layered')}
- Deployment: {blueprint.get('deployment_strategy', {}).get('type', 'standard')}

Include ignore patterns for:
- Python specific files (__pycache__, *.pyc, etc.)
- Virtual environments
- IDE/editor files
- OS-specific files
- Log files and temporary files
- Configuration files with secrets
- Build artifacts
- Testing artifacts
- Any technology-specific patterns

Return just the gitignore content, no markdown formatting."""

            try:
                gitignore_response = await self.llm_provider.generate(gitignore_prompt)
                gitignore_content = gitignore_response.strip()
            except Exception as e:
                self.logger.error(f"LLM generation failed for .gitignore: {e}")
                raise ProtocolExecutionError(f"Gitignore generation failed: {e}")
            
            git_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/.gitignore",
                "content": gitignore_content
            })
            if git_result.get("success"):
                created_files[".gitignore"] = gitignore_content
                self.logger.info("[MCP] Created .gitignore")
            
            self.logger.info(f"Successfully created {len(created_files)} project files using MCP tools")
            
        except Exception as e:
            self.logger.error(f"Failed to create project structure via MCP: {e}")
            created_files["error"] = f"Project structure creation failed: {str(e)}"
        
        return created_files

    async def _create_blueprint_files(self, blueprint: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, str]:
        """Create blueprint documentation files using MCP tools."""
        self.logger.info("Creating blueprint documentation files using MCP filesystem tools")
        
        created_files = {}
        project_path = inputs.project_path or "."  # Same as code generator - use project root directly
        project_name = inputs.project_id or "project"
        
        try:
            # No special blueprints directory - write directly to project root like code generator does
            self.logger.info(f"[MCP] Writing blueprint files to project root: {project_path}")
            
            # 1. Create main blueprint markdown file
            blueprint_content = self._generate_blueprint_markdown(blueprint, inputs)
            blueprint_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/{project_name}_blueprint.md",
                "content": blueprint_content
            })
            if blueprint_result.get("success"):
                created_files[f"{project_name}_blueprint.md"] = blueprint_content
                self.logger.info(f"[MCP] Created {project_name}_blueprint.md")
            
            # 2. Create technology stack file
            tech_stack = blueprint.get("technology_stack", {})
            if tech_stack:
                tech_content = self._generate_tech_stack_file(tech_stack)
                tech_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": f"{project_path}/{project_name}_tech_stack.md",
                    "content": tech_content
                })
                if tech_result.get("success"):
                    created_files[f"{project_name}_tech_stack.md"] = tech_content
                    self.logger.info(f"[MCP] Created {project_name}_tech_stack.md")
            
            # 3. Create component specifications (in docs subdirectory)
            components = blueprint.get("components", [])
            if components:
                # Create docs directory for component specs
                docs_dir_result = await self._call_mcp_tool("filesystem_create_directory", {
                    "directory_path": f"{project_path}/docs",
                    "create_parents": True
                })
                
                for i, component in enumerate(components):
                    comp_name = component.get("name", f"component_{i+1}").lower().replace(" ", "_")
                    comp_content = self._generate_component_spec(component, i)
                    comp_result = await self._call_mcp_tool("filesystem_write_file", {
                        "file_path": f"{project_path}/docs/{comp_name}_spec.md",
                        "content": comp_content
                    })
                    if comp_result.get("success"):
                        created_files[f"docs/{comp_name}_spec.md"] = comp_content
                        self.logger.info(f"[MCP] Created docs/{comp_name}_spec.md")
            
            # 4. Create directory structure documentation
            dir_structure = blueprint.get("directory_structure", {})
            if dir_structure:
                structure_content = self._generate_directory_structure_file(dir_structure)
                structure_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": f"{project_path}/{project_name}_directory_structure.md",
                    "content": structure_content
                })
                if structure_result.get("success"):
                    created_files[f"{project_name}_directory_structure.md"] = structure_content
                    self.logger.info(f"[MCP] Created {project_name}_directory_structure.md")
            
            # 5. Create deployment guide
            deployment = blueprint.get("deployment_strategy", {})
            if deployment:
                deploy_content = self._generate_deployment_guide(deployment)
                deploy_result = await self._call_mcp_tool("filesystem_write_file", {
                    "file_path": f"{project_path}/{project_name}_deployment_guide.md",
                    "content": deploy_content
                })
                if deploy_result.get("success"):
                    created_files[f"{project_name}_deployment_guide.md"] = deploy_content
                    self.logger.info(f"[MCP] Created {project_name}_deployment_guide.md")
            
            # 6. Create blueprint JSON file for programmatic access
            blueprint_json = json.dumps(blueprint, indent=2)
            json_result = await self._call_mcp_tool("filesystem_write_file", {
                "file_path": f"{project_path}/{project_name}_blueprint.json",
                "content": blueprint_json
            })
            if json_result.get("success"):
                created_files[f"{project_name}_blueprint.json"] = blueprint_json
                self.logger.info(f"[MCP] Created {project_name}_blueprint.json")
            
            self.logger.info(f"Successfully created {len(created_files)} blueprint documentation files in project root")
            
        except Exception as e:
            self.logger.error(f"Failed to create blueprint files via MCP: {e}")
            created_files["error"] = f"Blueprint file creation failed: {str(e)}"
        
        return created_files

    async def _generate_execution_plan(self, blueprint: Dict[str, Any], review_results: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """NEW: Generate execution plan from blueprint."""
        self.logger.info("Generating execution plan from blueprint")
        
        try:
            # Create execution plan structure
            execution_plan = {
                "id": str(uuid.uuid4()),
                "name": f"Implementation Plan for {inputs.project_id or 'Project'}",
                "description": f"Execution plan generated from architecture blueprint. Confidence: {review_results.get('overall_review_score', 0.8)}",
                "project_id": inputs.project_id,
                "version": "1.0",
                "created_at": datetime.datetime.now().isoformat(),
                "updated_at": datetime.datetime.now().isoformat(),
                "global_config": {
                    "project_root": inputs.project_path or "./",
                    "blueprint_id": inputs.task_id,
                    "source_dir": "src",
                    "test_dir": "tests"
                },
                "stages": {},
                "initial_stage": "stage_1_environment_setup"
            }
            
            # Generate stages based on blueprint components
            stages = self._generate_execution_stages(blueprint, inputs)
            execution_plan["stages"] = stages
            
            # Store execution plan in ChromaDB
            plan_doc_id = f"execution_plan_{inputs.task_id}_{uuid.uuid4().hex[:8]}"
            try:
                await migrate_store_artifact(
                    collection_name=EXECUTION_PLAN_ARTIFACTS_COLLECTION,
                    document_id=plan_doc_id,
                    content=json.dumps(execution_plan, indent=2),
                    metadata={
                        "agent_id": self.AGENT_ID,
                        "artifact_type": ARTIFACT_TYPE_EXECUTION_PLAN_JSON,
                        "project_id": inputs.project_id or "unknown",
                        "blueprint_source": inputs.task_id,
                        "timestamp": datetime.datetime.now().isoformat(),
                        "stages_count": len(stages)
                    }
                )
                self.logger.info(f"Stored execution plan with ID: {plan_doc_id}")
                
                return {
                    "success": True,
                    "execution_plan_doc_id": plan_doc_id,
                    "execution_plan": execution_plan,
                    "stages_count": len(stages)
                }
                
            except Exception as e:
                self.logger.error(f"Failed to store execution plan: {e}")
                return {
                    "success": False,
                    "error": f"Storage failed: {str(e)}",
                    "execution_plan": execution_plan,
                    "stages_count": len(stages)
                }
                
        except Exception as e:
            self.logger.error(f"Execution plan generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "stages_count": 0
            }

    def _generate_execution_stages(self, blueprint: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Generate execution stages for the execution plan."""
        stages = {}
        
        # Stage 1: Environment Setup
        stages["stage_1_environment_setup"] = {
            "name": "Environment Setup",
            "agent_id": "SystemFileSystemAgent_v1",
            "description": "Set up project directory structure and initial configuration",
            "number": 1.0,
            "inputs": {
                "project_id": f"{{{{context.project_id}}}}",
                "directory_structure": blueprint.get("directory_structure", {}),
                "operation": "create_project_structure"
            },
            "output_context_path": "stage_outputs.environment_setup",
            "success_criteria": ["context.stage_outputs.environment_setup.status == 'SUCCESS'"],
            "next_stage": "stage_2_dependencies_setup",
            "task_context_dependencies": [
                {"source_document_type": "blueprint", "section_id": "directory_structure"},
                {"source_document_type": "blueprint", "section_id": "project_setup"}
            ]
        }
        
        # Stage 2: Dependencies Setup  
        stages["stage_2_dependencies_setup"] = {
            "name": "Dependencies and Configuration",
            "agent_id": "SmartCodeGeneratorAgent_v1",
            "description": "Generate dependency files and configuration based on technology stack",
            "number": 2.0,
            "inputs": {
                "project_id": f"{{{{context.project_id}}}}",
                "task_description": f"Generate dependency files for {blueprint.get('technology_stack', {})}",
                "technologies": blueprint.get("technology_stack", {}),
                "target_file_path": "requirements.txt"
            },
            "output_context_path": "stage_outputs.dependencies_setup",
            "success_criteria": ["context.stage_outputs.dependencies_setup.status == 'SUCCESS'"],
            "next_stage": "stage_3_core_implementation",
            "depends_on": ["stage_1_environment_setup"],
            "task_context_dependencies": [
                {"source_document_type": "blueprint", "section_id": "technology_stack"},
                {"source_document_type": "blueprint", "section_id": "dependencies"}
            ]
        }
        
        # Stage 3+: Component Implementation
        components = blueprint.get("components", [])
        if components:
            for i, component in enumerate(components):
                stage_num = 3 + i
                stage_name = f"stage_{stage_num}_component_{component.get('name', f'comp_{i+1}').lower().replace(' ', '_')}"
                
                stages[stage_name] = {
                    "name": f"Implement {component.get('name', f'Component {i+1}')}",
                    "agent_id": "SmartCodeGeneratorAgent_v1",
                    "description": f"Generate code for {component.get('responsibility', 'component functionality')}",
                    "number": float(stage_num),
                    "inputs": {
                        "project_id": f"{{{{context.project_id}}}}",
                        "task_description": component.get("responsibility", "Implement component"),
                        "component_spec": component,
                        "architecture_pattern": blueprint.get("architecture_pattern", "layered")
                    },
                    "output_context_path": f"stage_outputs.component_{i+1}",
                    "success_criteria": [f"context.stage_outputs.component_{i+1}.status == 'SUCCESS'"],
                    "next_stage": f"stage_{stage_num+1}_testing" if i == len(components) - 1 else f"stage_{stage_num+1}_component_{components[i+1].get('name', f'comp_{i+2}').lower().replace(' ', '_')}",
                    "depends_on": ["stage_2_dependencies_setup"],
                    "task_context_dependencies": [
                        {"source_document_type": "blueprint", "section_id": f"components.{component.get('name', f'component_{i+1}')}"},
                        {"source_document_type": "blueprint", "section_id": "architecture_pattern"}
                    ]
                }
        
        # Final stage: Testing
        final_stage_num = 3 + len(components)
        stages[f"stage_{final_stage_num}_testing"] = {
            "name": "Generate Tests",
            "agent_id": "CoreTestGeneratorAgent_v1",
            "description": "Generate comprehensive tests for implemented components",
            "number": float(final_stage_num),
            "inputs": {
                "project_id": f"{{{{context.project_id}}}}",
                "testing_strategy": blueprint.get("testing_strategy", {}),
                "components": components
            },
            "output_context_path": "stage_outputs.testing",
            "success_criteria": ["context.stage_outputs.testing.status == 'SUCCESS'"],
            "next_stage": "FINAL_STEP",
            "depends_on": [f"stage_{final_stage_num-1}_component_{components[-1].get('name', f'comp_{len(components)}').lower().replace(' ', '_')}"] if components else ["stage_2_dependencies_setup"],
            "task_context_dependencies": [
                {"source_document_type": "blueprint", "section_id": "testing_strategy"},
                {"source_document_type": "blueprint", "section_id": "components"}
            ]
        }
        
        return stages

    async def _validate_enhanced_output(self, blueprint: Dict[str, Any], review_results: Dict[str, Any], 
                                       blueprint_files: Dict[str, str], execution_plan_result: Dict[str, Any], 
                                       inputs: EnhancedArchitectAgentInput) -> Dict[str, Any]:
        """Final validation of all enhanced outputs."""
        validation = {
            "validation_completed": True,
            "blueprint_valid": True,
            "review_valid": True,
            "files_valid": True,
            "execution_plan_valid": True,
            "overall_valid": True,
            "validation_issues": []
        }
        
        # Validate blueprint
        if not blueprint or not blueprint.get("title"):
            validation["blueprint_valid"] = False
            validation["validation_issues"].append("Blueprint missing or incomplete")
        
        # Validate review
        if not review_results.get("review_completed"):
            validation["review_valid"] = False
            validation["validation_issues"].append("Blueprint review not completed")
        
        # Validate files (if requested)
        if inputs.output_blueprint_files:
            if not blueprint_files or "error" in blueprint_files:
                validation["files_valid"] = False
                validation["validation_issues"].append("Blueprint files generation failed")
        
        # Validate execution plan (if requested)
        if inputs.generate_execution_plan:
            if not execution_plan_result.get("success"):
                validation["execution_plan_valid"] = False
                validation["validation_issues"].append("Execution plan generation failed")
        
        # Overall validation
        validation["overall_valid"] = all([
            validation["blueprint_valid"],
            validation["review_valid"],
            validation["files_valid"] or not inputs.output_blueprint_files,
            validation["execution_plan_valid"] or not inputs.generate_execution_plan
        ])
        
        return validation

    def _calculate_enhanced_quality_score(self, validation_result: Dict[str, Any], blueprint: Dict[str, Any], 
                                        review_results: Dict[str, Any], execution_plan_result: Dict[str, Any]) -> float:
        """Calculate enhanced quality score based on all outputs."""
        base_score = 1.0
        
        # Blueprint quality (40%)
        blueprint_score = self._calculate_quality_score(validation_result, blueprint, {})
        
        # Review quality (25%)
        review_score = review_results.get("overall_review_score", 0.5)
        
        # File generation quality (15%)
        files_score = 1.0 if validation_result.get("files_valid", True) else 0.3
        
        # Execution plan quality (20%)
        plan_score = 1.0 if execution_plan_result.get("success", False) else 0.3
        
        # Weighted average
        enhanced_score = (
            blueprint_score * 0.4 +
            review_score * 0.25 +
            files_score * 0.15 +
            plan_score * 0.2
        )
        
        return max(0.1, min(enhanced_score, 1.0))

    def _generate_blueprint_markdown(self, blueprint: Dict[str, Any], inputs: EnhancedArchitectAgentInput) -> str:
        """Generate comprehensive blueprint markdown content."""
        title = blueprint.get("title", f"Architecture Blueprint - {inputs.project_id}")
        pattern = blueprint.get("architecture_pattern", "unknown")
        tech_stack = blueprint.get("technology_stack", {})
        components = blueprint.get("components", [])
        deployment = blueprint.get("deployment_strategy", {})
        testing = blueprint.get("testing_strategy", {})
        
        content = f"""# {title}

## Project Overview
- **Project ID**: {inputs.project_id or 'Unknown'}
- **Architecture Pattern**: {pattern}
- **Generated**: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Architecture Pattern
This project follows a **{pattern}** architectural pattern, chosen for optimal {pattern} characteristics.

## Technology Stack
"""
        
        for category, technology in tech_stack.items():
            if technology:
                content += f"- **{category.title()}**: {technology}\n"
        
        content += f"""
## Components

This architecture consists of {len(components)} main components:

"""
        
        for i, component in enumerate(components, 1):
            name = component.get("name", f"Component {i}")
            responsibility = component.get("responsibility", "No description available")
            interfaces = component.get("interfaces", [])
            
            content += f"""### {i}. {name}
**Responsibility**: {responsibility}

**Interfaces**: {', '.join(interfaces) if interfaces else 'None specified'}

"""
        
        if deployment:
            content += f"""## Deployment Strategy
- **Type**: {deployment.get('type', 'Unknown')}
- **Orchestration**: {deployment.get('orchestration', 'Not specified')}
- **Scaling**: {deployment.get('scaling', 'Not specified')}
- **Monitoring**: {deployment.get('monitoring', 'Not specified')}

"""
        
        if testing:
            content += f"""## Testing Strategy
- **Unit Tests**: {testing.get('unit_tests', 'Not specified')}
- **Integration Tests**: {testing.get('integration_tests', 'Not specified')}
- **E2E Tests**: {testing.get('e2e_tests', 'Not specified')}
- **Coverage Target**: {testing.get('coverage_target', 'Not specified')}

"""
        
        content += f"""## Implementation Notes
This blueprint was generated by the Enhanced Architect Agent v{self.AGENT_VERSION} with full lifecycle management including self-review, file generation, and execution planning.

For detailed component specifications, see the individual component files in the `components/` directory.
"""
        
        return content

    def _generate_tech_stack_file(self, tech_stack: Dict[str, Any]) -> str:
        """Generate technology stack documentation."""
        content = """# Technology Stack

## Overview
This document details the technology choices for this project.

"""
        
        for category, technology in tech_stack.items():
            if technology:
                content += f"""## {category.title()}
- **Choice**: {technology}
- **Rationale**: Selected for {category} requirements

"""
        
        return content

    def _generate_component_spec(self, component: Dict[str, Any], index: int) -> str:
        """Generate individual component specification."""
        name = component.get("name", f"Component {index + 1}")
        responsibility = component.get("responsibility", "No description available")
        interfaces = component.get("interfaces", [])
        dependencies = component.get("dependencies", [])
        
        content = f"""# {name} Specification

## Responsibility
{responsibility}

## Interfaces
"""
        
        if interfaces:
            for interface in interfaces:
                content += f"- {interface}\n"
        else:
            content += "- None specified\n"
        
        content += f"""
## Dependencies
"""
        
        if dependencies:
            for dep in dependencies:
                content += f"- {dep}\n"
        else:
            content += "- None specified\n"
        
        content += f"""
## Implementation Notes
This component should be implemented following the project's architecture pattern and coding standards.
"""
        
        return content

    def _generate_directory_structure_file(self, dir_structure: Dict[str, Any]) -> str:
        """Generate directory structure documentation."""
        content = """# Directory Structure

## Project Layout
This document describes the recommended directory structure for the project.

```
"""
        
        def format_structure(structure, indent=0):
            result = ""
            for key, value in structure.items():
                result += "  " * indent + key + "\n"
                if isinstance(value, dict):
                    result += format_structure(value, indent + 1)
            return result
        
        content += format_structure(dir_structure)
        content += """```

## Directory Descriptions
Each directory serves a specific purpose in the project organization:

"""
        
        # Add descriptions for common directories
        common_descriptions = {
            "src/": "Source code directory",
            "tests/": "Test files and test utilities", 
            "docs/": "Documentation files",
            "config/": "Configuration files",
            "requirements.txt": "Python dependencies",
            "README.md": "Project overview and setup instructions"
        }
        
        for path, desc in common_descriptions.items():
            if any(path in str(dir_structure).lower() for path in [path.lower()]):
                content += f"- **{path}**: {desc}\n"
        
        return content

    def _generate_deployment_guide(self, deployment: Dict[str, Any]) -> str:
        """Generate deployment guide."""
        content = f"""# Deployment Guide

## Overview
This project uses a **{deployment.get('type', 'standard')}** deployment strategy.

## Configuration
- **Type**: {deployment.get('type', 'Not specified')}
- **Orchestration**: {deployment.get('orchestration', 'Not specified')}
- **Scaling**: {deployment.get('scaling', 'Not specified')}
- **Monitoring**: {deployment.get('monitoring', 'Not specified')}

## Deployment Steps
1. Prepare the deployment environment
2. Configure the application settings
3. Deploy the application
4. Verify the deployment
5. Set up monitoring and logging

## Monitoring
Monitor the application health and performance using the configured monitoring solution.
"""
        
        return content

    @staticmethod
    def get_agent_card_static() -> AgentCard:
        input_schema = EnhancedArchitectAgentInput.model_json_schema()
        output_schema = EnhancedArchitectAgentOutput.model_json_schema()
        module_path = EnhancedArchitectAgent_v1.__module__
        class_name = EnhancedArchitectAgent_v1.__name__

        return AgentCard(
            agent_id=EnhancedArchitectAgent_v1.AGENT_ID,
            name=EnhancedArchitectAgent_v1.AGENT_NAME,
            description=EnhancedArchitectAgent_v1.AGENT_DESCRIPTION,
            version=EnhancedArchitectAgent_v1.AGENT_VERSION,
            input_schema=input_schema,
            output_schema=output_schema,
            categories=[cat.value for cat in [EnhancedArchitectAgent_v1.CATEGORY, AgentCategory.AUTONOMOUS_PROJECT_ENGINE]],
            visibility=EnhancedArchitectAgent_v1.VISIBILITY.value,
            capability_profile={
                "consumes_artifacts": ["LOPRD_JSON"],
                "generates_blueprints": ["ProjectBlueprint_Markdown"],
                "generates_files": ["blueprint_md", "component_specs", "tech_stack", "deployment_guide"],
                "generates_execution_plans": ["MasterExecutionPlan_JSON"],
                "architecture_patterns": ["monolithic", "layered", "microservices"],
                "workflow_phases": ["discovery", "analysis", "planning", "design", "review", "output", "flow_generation", "validation"],
                "primary_function": "Complete Blueprint Lifecycle Management"
            },
            metadata={
                "callable_fn_path": f"{module_path}.{class_name}",
                "enhanced_version": True,
                "consolidates_agents": ["ArchitectAgent_v1", "BlueprintReviewerAgent_v1", "BlueprintToFlowAgent_v1"]
            }
        ) 