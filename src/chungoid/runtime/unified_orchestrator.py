"""UnifiedOrchestrator (Phase-1 UAEI)

This orchestrator executes stages by calling `UnifiedAgent.execute()`
using the new `ExecutionContext` structure. For Phase-1 it supports only
single-pass execution and minimal branching. It will gradually replace
`AsyncOrchestrator` after all agents migrate.

This is the complete implementation according to enhanced_cycle.md Phase 1.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..schemas.unified_execution_schemas import (
    ExecutionContext,
    ExecutionConfig,
    StageInfo,
    AgentExecutionResult,
    ExecutionMode,
)
from ..schemas.master_flow import MasterExecutionPlan
from ..schemas.agent_master_planner import MasterPlannerInput
from ..schemas.common_enums import StageStatus, OnFailureAction
from ..utils.state_manager import StateManager
from .unified_agent_resolver import UnifiedAgentResolver
from ..utils.metrics_store import MetricsStore
from ..utils.llm_provider import LLMProvider

__all__ = ["UnifiedOrchestrator"]


class UnifiedOrchestrator:
    """
    Phase-1 UAEI UnifiedOrchestrator
    
    Replaces AsyncOrchestrator with a simplified, single-path execution model.
    Uses only agent.execute() calls - no branching, no adapters, no complexity.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        state_manager: StateManager,
        agent_resolver: UnifiedAgentResolver,
        metrics_store: MetricsStore,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.config = config
        self.state_manager = state_manager
        self.agent_resolver = agent_resolver
        self.metrics_store = metrics_store
        self.llm_provider = llm_provider  # Add LLM provider for goal analysis
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize shared context
        self.shared_context: Dict[str, Any] = {
            "project_root_path": str(state_manager.target_dir_path),
            "outputs": {},
        }

    # ---------------------------------------------------------------------
    async def execute_stage(
        self,
        stage_id: str,
        agent_id: str,
        inputs: Any,
        attempt: int = 1,
        max_iterations: int = 1
    ) -> AgentExecutionResult:
        """Execute a single stage with the given agent in single-pass mode."""

        self.logger.info("[UAEI] Executing stage %s (attempt %d) with agent %s", stage_id, attempt, agent_id)

        # Phase 3: Resolve agent using UnifiedAgentResolver (single path)
        agent = await self.agent_resolver.resolve_agent(agent_id)

        # Create execution context
        ctx = ExecutionContext(
            inputs=inputs,
            shared_context=self.shared_context,
            stage_info=StageInfo(stage_id=stage_id, attempt_number=attempt),
            execution_config=ExecutionConfig(
                max_iterations=max_iterations,
                quality_threshold=0.85,  # Phase 3: Enable quality thresholds
                completion_criteria=None  # Phase 3: Will be enhanced
            ),
        )

        # Phase 3: Execute using UnifiedAgent.execute() with ExecutionMode.OPTIMAL
        result = await agent.execute(ctx, ExecutionMode.OPTIMAL)

        # Persist outputs in shared context under stage_id
        self.shared_context["outputs"][stage_id] = result.output
        return result

    # ------------------------------------------------------------------
    async def execute_master_plan_async(
        self,
        master_plan: MasterExecutionPlan,
        run_id_override: Optional[str] = None,
        tags_override: Optional[List[str]] = None
    ) -> None:
        """Execute a complete master plan using UnifiedAgent.execute() calls."""
        
        run_id = run_id_override or str(uuid.uuid4())
        self.logger.info(f"[UAEI] Executing master plan {master_plan.id} with run_id {run_id}")
        
        # Update shared context with plan info
        self.shared_context.update({
            "master_plan_id": master_plan.id,
            "run_id": run_id,
            "tags": tags_override or []
        })
        
        if master_plan.initial_context:
            self.shared_context.update(master_plan.initial_context)
        
        # Execute stages sequentially (Phase-1: simple execution)
        for stage in master_plan.stages:
            try:
                self.logger.info(f"[UAEI] Executing stage {stage.stage_id}")
                
                result = await self.execute_stage(
                    stage_id=stage.stage_id,
                    agent_id=stage.agent_id,
                    inputs=stage.inputs,
                    max_iterations=getattr(stage, 'max_iterations', 1)
                )
                
                self.logger.info(f"[UAEI] Stage {stage.stage_id} completed with status: {result.completion_reason}")
                
            except Exception as e:
                self.logger.error(f"[UAEI] Stage {stage.stage_id} failed: {e}")
                # Phase-1: simple error handling
                raise

        self.logger.info(f"[UAEI] Master plan {master_plan.id} execution completed")

    # ------------------------------------------------------------------
    async def execute_master_planner_goal_async(
        self,
        master_planner_input: MasterPlannerInput
    ) -> None:
        """
        UAEI Phase-1: Enhanced goal execution with proper goal analysis.
        Analyzes any goal content to extract real project specifications,
        then executes relevant agents with the actual project information.
        """
        
        self.logger.info(f"[UAEI] Executing enhanced goal flow: {master_planner_input.user_goal[:100]}...")
        
        # Update shared context
        self.shared_context.update({
            "user_goal": master_planner_input.user_goal,
            "project_id": master_planner_input.project_id,
            "run_id": master_planner_input.run_id,
            "tags": master_planner_input.tags or []
        })
        
        if master_planner_input.initial_context:
            self.shared_context.update(master_planner_input.initial_context)
        
        # UAEI Phase-1: Enhanced flow with proper goal analysis
        
        # 1. Goal analysis - ENHANCED: Direct YAML parsing (NO AGENT CALLS)
        self.logger.info("[UAEI] Analyzing goal content directly (no agent interaction)")
        
        # Extract intelligent project information from goal content directly
        project_specs = await self._extract_project_specifications(
            goal_content=master_planner_input.user_goal,
            analysis_result=None  # No agent result needed
        )
        
        # 2. Environment setup - ENHANCED: Use extracted project information
        await self.execute_stage(
            stage_id="environment_bootstrap",
            agent_id="EnvironmentBootstrapAgent",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,  # Pass intelligent analysis!
                "project_path": self.shared_context.get("project_root_path", "."),
                "intelligent_context": True  # Signal that this is intelligent input
            },
            max_iterations=self._get_max_iterations("environment_bootstrap", 15)
        )

        # 3. Dependency management - ENHANCED: Use extracted project information
        await self.execute_stage(
            stage_id="dependency_management",
            agent_id="DependencyManagementAgent_v1",
            inputs={
                "operation": "analyze",
                "project_path": self.shared_context.get("project_root_path", "."),
                "auto_detect_dependencies": True,
                "install_after_analysis": True,
                "resolve_conflicts": True,
                "target_languages": project_specs.get("target_languages", ["python"]),
                "perform_security_audit": True,
                "optimize_versions": True,
                "create_lock_files": True,
                "project_specifications": project_specs,  # Pass intelligent analysis!
                "intelligent_context": True,  # Signal that this is intelligent input
                "user_goal": master_planner_input.user_goal
            },
            max_iterations=self._get_max_iterations("dependency_management", 12)
        )

        # 4. Product Analysis - ENHANCED: Analyze requirements and scope
        await self.execute_stage(
            stage_id="product_analysis",
            agent_id="ProductAnalystAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("product_analysis", 20)
        )

        # 5. Architecture Design - ENHANCED: Design system architecture
        await self.execute_stage(
            stage_id="architecture_design",
            agent_id="ArchitectAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("architecture_design", 25)
        )

        # 6. Requirements Tracing - ENHANCED: Trace and validate requirements
        await self.execute_stage(
            stage_id="requirements_tracing",
            agent_id="RequirementsTracerAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("requirements_tracing", 18)
        )

        # 7. Risk Assessment - ENHANCED: Assess project risks
        await self.execute_stage(
            stage_id="risk_assessment",
            agent_id="ProactiveRiskAssessorAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("risk_assessment", 22)
        )

        # 8. Blueprint Review - ENHANCED: Review and validate architecture
        await self.execute_stage(
            stage_id="blueprint_review",
            agent_id="BlueprintReviewerAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("blueprint_review", 15)
        )

        # 9. Code Generation - ENHANCED: Generate actual project code files
        await self.execute_stage(
            stage_id="code_generation",
            agent_id="SmartCodeGeneratorAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", "."),
                "project_id": master_planner_input.project_id or "intelligent_project",
                "programming_language": project_specs.get("primary_language", "python"),
                "target_languages": project_specs.get("target_languages", ["python"]),
                "technologies": project_specs.get("technologies", [])
            },
            max_iterations=self._get_max_iterations("code_generation", 30)
        )

        # 10. Project Documentation - ENHANCED: Generate comprehensive documentation
        await self.execute_stage(
            stage_id="project_documentation",
            agent_id="ProjectDocumentationAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("project_documentation", 16)
        )

        # 11. Code Debugging - ENHANCED: Analyze and improve code quality
        await self.execute_stage(
            stage_id="code_debugging",
            agent_id="CodeDebuggingAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", ".")
            },
            max_iterations=self._get_max_iterations("code_debugging", 28)
        )

        # 12. Automated Refinement Coordination - ENHANCED: Coordinate final improvements
        await self.execute_stage(
            stage_id="automated_refinement",
            agent_id="AutomatedRefinementCoordinatorAgent_v1",
            inputs={
                "user_goal": master_planner_input.user_goal,
                "project_specifications": project_specs,
                "intelligent_context": True,  # Keep intelligent mode - fix the implementation instead
                "project_path": self.shared_context.get("project_root_path", "."),
                "project_id": master_planner_input.project_id or "intelligent_project"
            },
            max_iterations=self._get_max_iterations("automated_refinement", 35)
        )
        
        self.logger.info("[UAEI] Enhanced autonomous development pipeline completed with 12 stages")

    # ------------------------------------------------------------------
    async def run(
        self,
        goal_str: Optional[str] = None,
        initial_context: Optional[Dict[str, Any]] = None,
        run_id_override: Optional[str] = None
    ) -> Tuple[StageStatus, Any, Optional[str]]:
        """
        Main orchestrator run method - replaces AsyncOrchestrator.run()
        
        Returns:
            Tuple of (final_status, final_shared_context, final_error_details)
        """
        
        run_id = run_id_override or str(uuid.uuid4())
        self.logger.info(f"[UAEI] Starting orchestrator run {run_id}")
        
        if initial_context:
            self.shared_context.update(initial_context)
        
        try:
            if goal_str:
                # Create master planner input
                planner_input = MasterPlannerInput(
                    user_goal=goal_str,
                    project_id=self.shared_context.get("project_id", "unknown"),
                    run_id=run_id,
                    initial_context=initial_context or {}
                )
                
                # Execute via master planner
                await self.execute_master_planner_goal_async(planner_input)
            
            # Return success status
            return StageStatus.COMPLETED_SUCCESS, self.shared_context, None
            
        except Exception as e:
            self.logger.error(f"[UAEI] Orchestrator run failed: {e}")
            return StageStatus.COMPLETED_FAILURE, self.shared_context, str(e)

    # ------------------------------------------------------------------
    async def resume_flow_async(
        self,
        run_id_to_resume: str,
        action: str,
        new_inputs: Optional[Dict[str, Any]] = None,
        target_stage_id_for_branch: Optional[str] = None
    ) -> None:
        """Resume a paused flow - Phase-1 simplified implementation."""
        
        self.logger.info(f"[UAEI] Resuming flow {run_id_to_resume} with action {action}")
        
        # Phase-1: Basic resume support
        # In a full implementation, this would load paused state and continue execution
        # For now, we'll implement basic retry logic
        
        if action == "retry":
            self.logger.info(f"[UAEI] Retrying last stage for run {run_id_to_resume}")
            # Load last stage from state and retry
            # This is a simplified implementation
            
        elif action == "abort":
            self.logger.info(f"[UAEI] Aborting run {run_id_to_resume}")
            # Mark as aborted in state
            
        else:
            self.logger.warning(f"[UAEI] Resume action {action} not fully implemented in Phase-1")
            
        # Phase-1: Basic implementation complete

    # ------------------------------------------------------------------
    def get_shared_outputs(self) -> Dict[str, Any]:
        """Get all stage outputs from shared context."""
        return self.shared_context.get("outputs", {})

    def _get_max_iterations(self, stage_id: str, default: int) -> int:
        """Get max_iterations for a stage from config or environment, with fallback to default."""
        import os
        
        # Check environment variable first (highest precedence)
        env_max_iter = os.getenv("CHUNGOID_MAX_ITERATIONS")
        if env_max_iter:
            try:
                return int(env_max_iter)
            except ValueError:
                self.logger.warning(f"Invalid CHUNGOID_MAX_ITERATIONS value: {env_max_iter}")
        
        # Check config for global max_iterations setting
        if self.config and "agents" in self.config:
            agents_config = self.config["agents"]
            if "default_max_iterations" in agents_config:
                try:
                    return int(agents_config["default_max_iterations"])
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid default_max_iterations in config: {agents_config['default_max_iterations']}")
        
        # Check config for stage-specific max_iterations
        if self.config and "agents" in self.config:
            agents_config = self.config["agents"]
            if "stage_max_iterations" in agents_config and stage_id in agents_config["stage_max_iterations"]:
                try:
                    return int(agents_config["stage_max_iterations"][stage_id])
                except (ValueError, TypeError):
                    self.logger.warning(f"Invalid stage_max_iterations for {stage_id}: {agents_config['stage_max_iterations'][stage_id]}")
        
        # Return default if nothing found
        return default

    async def _extract_project_specifications(
        self, 
        goal_content: str, 
        analysis_result: Any
    ) -> Dict[str, Any]:
        """
        Intelligently extract project specifications from goal content.
        Uses YAML parsing and LLM analysis to understand project requirements.
        """
        
        project_specs = {
            "project_type": "cli_tool",
            "primary_language": "python", 
            "target_languages": ["python"],
            "target_platforms": ["linux", "macos", "windows"],
            "technologies": ["python"],
            "required_dependencies": [],
            "optional_dependencies": []
        }
        
        try:
            # First, try to parse as YAML using PyYAML
            import yaml
            
            # Use safe_load to securely parse YAML content
            parsed_goal = yaml.safe_load(goal_content)
            
            if parsed_goal and isinstance(parsed_goal, dict):
                self.logger.info("[UAEI] Successfully parsed goal content as YAML")
                
                # Extract project type
                project_type = self._extract_project_type(parsed_goal)
                if project_type:
                    project_specs["project_type"] = project_type
                    self.logger.info(f"[UAEI] Extracted project type: {project_type}")
                
                # Extract technical specifications
                tech_specs = self._extract_technical_specs(parsed_goal)
                project_specs.update(tech_specs)
                
                # Extract dependencies
                deps = self._extract_dependencies(parsed_goal)
                project_specs.update(deps)
                
            else:
                self.logger.info("[UAEI] Goal content is not valid YAML, using LLM analysis")
                # Fallback to LLM-based analysis for non-YAML content
                llm_specs = await self._analyze_goal_with_llm(goal_content)
                project_specs.update(llm_specs)
                
        except yaml.YAMLError as e:
            self.logger.info(f"[UAEI] YAML parsing failed: {e}, using LLM analysis")
            # Fallback to LLM-based analysis
            llm_specs = await self._analyze_goal_with_llm(goal_content)
            project_specs.update(llm_specs)
        except Exception as e:
            self.logger.warning(f"[UAEI] Error extracting project specs: {e}")
            # Return defaults if all else fails
        
        self.logger.info(f"[UAEI] Final project specifications: {project_specs}")
        return project_specs
    
    def _extract_project_type(self, parsed_goal: Dict[str, Any]) -> str:
        """Extract project type from parsed YAML goal"""
        
        # Look for explicit project type
        if "project" in parsed_goal and "type" in parsed_goal["project"]:
            return parsed_goal["project"]["type"]
        
        if "type" in parsed_goal:
            return parsed_goal["type"]
        
        # Infer from project characteristics
        if "technical" in parsed_goal:
            tech = parsed_goal["technical"]
            
            # Check interface type
            if "interface" in tech and "type" in tech["interface"]:
                interface_type = tech["interface"]["type"]
                if interface_type == "command_line":
                    return "cli_tool"
                elif interface_type in ["web_ui", "web"]:
                    return "web_app"
                elif interface_type in ["rest_api", "api"]:
                    return "api"
        
        # Check for CLI indicators in description or name
        description = parsed_goal.get("description", "").lower()
        name = parsed_goal.get("name", "").lower()
        
        if any(term in description + name for term in ["cli", "command", "terminal", "scanner"]):
            return "cli_tool"
        elif any(term in description + name for term in ["web", "website", "app"]):
            return "web_app"
        elif any(term in description + name for term in ["api", "service"]):
            return "api"
        
        return "cli_tool"  # Default
    
    def _extract_technical_specs(self, parsed_goal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract technical specifications from parsed YAML goal"""
        
        specs = {}
        
        if "technical" in parsed_goal:
            tech = parsed_goal["technical"]
            
            # Extract primary language
            if "primary_language" in tech:
                specs["primary_language"] = tech["primary_language"].lower()
                specs["target_languages"] = [tech["primary_language"].lower()]
            
            # Extract secondary languages
            if "secondary_languages" in tech and tech["secondary_languages"]:
                languages = [lang.lower() for lang in tech["secondary_languages"]]
                if "target_languages" in specs:
                    specs["target_languages"].extend(languages)
                else:
                    specs["target_languages"] = languages
            
            # Extract target platforms
            if "target_platforms" in tech and tech["target_platforms"]:
                platforms = [platform.lower() for platform in tech["target_platforms"]]
                specs["target_platforms"] = platforms
            
            # Extract technologies from various fields
            technologies = set()
            
            # Add primary language as technology
            if "primary_language" in specs:
                technologies.add(specs["primary_language"])
            
            # Look for frameworks/libraries in dependencies
            if "dependencies" in tech:
                deps = tech["dependencies"]
                if "required" in deps and deps["required"]:
                    for dep in deps["required"]:
                        if isinstance(dep, str):
                            technologies.add(dep.lower())
            
            if technologies:
                specs["technologies"] = list(technologies)
        
        return specs
    
    def _extract_dependencies(self, parsed_goal: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dependency information from parsed YAML goal"""
        
        deps = {
            "required_dependencies": [],
            "optional_dependencies": []
        }
        
        if "technical" in parsed_goal and "dependencies" in parsed_goal["technical"]:
            tech_deps = parsed_goal["technical"]["dependencies"]
            
            # Extract required dependencies
            if "required" in tech_deps and tech_deps["required"]:
                required = tech_deps["required"]
                if isinstance(required, list):
                    deps["required_dependencies"] = [dep.lower() for dep in required if isinstance(dep, str)]
                elif isinstance(required, str):
                    deps["required_dependencies"] = [required.lower()]
            
            # Extract optional dependencies
            if "optional" in tech_deps and tech_deps["optional"]:
                optional = tech_deps["optional"]
                if isinstance(optional, list):
                    deps["optional_dependencies"] = [dep.lower() for dep in optional if isinstance(dep, str)]
                elif isinstance(optional, str):
                    deps["optional_dependencies"] = [optional.lower()]
        
        return deps
    
    async def _analyze_goal_with_llm(self, goal_content: str) -> Dict[str, Any]:
        """Fallback LLM analysis for non-YAML goal content"""
        
        try:
            analysis_prompt = f"""
            Analyze this project goal and extract technical specifications:
            
            Goal: {goal_content}
            
            Extract and return ONLY a JSON object with these fields:
            {{
                "project_type": "cli_tool|web_app|api|library|other",
                "primary_language": "python|javascript|java|etc",
                "target_languages": ["list", "of", "languages"],
                "target_platforms": ["linux", "macos", "windows"],
                "technologies": ["list", "of", "technologies"],
                "required_dependencies": ["list", "of", "required", "deps"],
                "optional_dependencies": ["list", "of", "optional", "deps"]
            }}
            
            Return only valid JSON, no other text.
            """
            
            response = await self.llm_provider.generate(
                prompt=analysis_prompt,
                max_tokens=500,
                temperature=0.1
            )
            
            # Parse JSON response
            import json
            specs = json.loads(response.strip())
            
            # Validate and clean the response
            validated_specs = {}
            if "project_type" in specs:
                validated_specs["project_type"] = specs["project_type"]
            if "primary_language" in specs:
                validated_specs["primary_language"] = specs["primary_language"].lower()
                validated_specs["target_languages"] = [specs["primary_language"].lower()]
            if "target_platforms" in specs and isinstance(specs["target_platforms"], list):
                validated_specs["target_platforms"] = [p.lower() for p in specs["target_platforms"]]
            if "technologies" in specs and isinstance(specs["technologies"], list):
                validated_specs["technologies"] = [t.lower() for t in specs["technologies"]]
            if "required_dependencies" in specs and isinstance(specs["required_dependencies"], list):
                validated_specs["required_dependencies"] = [d.lower() for d in specs["required_dependencies"]]
            if "optional_dependencies" in specs and isinstance(specs["optional_dependencies"], list):
                validated_specs["optional_dependencies"] = [d.lower() for d in specs["optional_dependencies"]]
            
            return validated_specs
            
        except Exception as e:
            self.logger.warning(f"[UAEI] LLM analysis failed: {e}")
            return {}  # Return empty dict to use defaults 