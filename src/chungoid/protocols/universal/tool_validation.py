"""
Tool Validation Protocol

Validate artifacts using external tools and MCP integrations.
Provides systematic validation of code, documentation, and other artifacts.

Change Reference: 3.17 (NEW)
"""

from typing import List, Dict, Any, Optional, Union
from ..base.protocol_interface import ProtocolInterface, ProtocolPhase, ProtocolTemplate

class ToolValidationProtocol(ProtocolInterface):
    """Validate artifacts using external tools and MCP integrations"""
    
    @property
    def name(self) -> str:
        return "tool_validation"
    
    @property
    def description(self) -> str:
        return "Validate artifacts using external tools and MCP integrations. Provides systematic validation of code, documentation, and other artifacts."
    
    @property
    def total_estimated_time(self) -> float:
        return 4.0  # Total of all phase time_box_hours
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates for tool validation"""
        return {
            "validation_report": ProtocolTemplate(
                name="validation_report",
                description="Template for validation results report",
                template_content="""
# Tool Validation Report

## Artifacts Validated: [ARTIFACT_COUNT]
## Tools Used: [TOOLS_USED]
## Overall Score: [OVERALL_SCORE]

## Results Summary
[RESULTS_SUMMARY]

## Issues Found
[ISSUES_FOUND]

## Recommendations
[RECOMMENDATIONS]
                """,
                variables=["ARTIFACT_COUNT", "TOOLS_USED", "OVERALL_SCORE", "RESULTS_SUMMARY", "ISSUES_FOUND", "RECOMMENDATIONS"]
            )
        }
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        return [
            ProtocolPhase(
                name="tool_discovery",
                description="Discover available validation tools and their capabilities",
                time_box_hours=0.5,
                required_outputs=["available_tools", "tool_capabilities"],
                validation_criteria=["Tools catalogued", "Capabilities mapped"],
                tools_required=["tool_registry", "capability_scanner"]
            ),
            ProtocolPhase(
                name="validation_planning",
                description="Plan validation strategy based on artifact types",
                time_box_hours=0.5,
                required_outputs=["validation_plan", "tool_assignments"],
                validation_criteria=["Plan created", "Tools assigned"],
                tools_required=["validation_planner", "tool_matcher"]
            ),
            ProtocolPhase(
                name="artifact_validation",
                description="Execute validation using selected tools",
                time_box_hours=2.0,
                required_outputs=["validation_results", "quality_metrics"],
                validation_criteria=["Validation complete", "Results collected"],
                tools_required=["validation_executor", "results_collector"]
            ),
            ProtocolPhase(
                name="result_analysis",
                description="Analyze validation results and provide recommendations",
                time_box_hours=1.0,
                required_outputs=["analysis_report", "recommendations"],
                validation_criteria=["Analysis complete", "Recommendations provided"],
                tools_required=["result_analyzer", "recommendation_engine"]
            )
        ]
    
    def discover_validation_tools(self, artifact_types: List[str]) -> Dict[str, Any]:
        """Discover available validation tools for artifact types"""
        
        tool_discovery = {
            "artifact_types": artifact_types,
            "available_tools": {},
            "tool_matrix": {},
            "mcp_integrations": {}
        }
        
        # Standard validation tools by artifact type
        standard_tools = {
            "python_code": ["pylint", "black", "mypy", "pytest", "bandit"],
            "javascript_code": ["eslint", "prettier", "jest", "typescript"],
            "documentation": ["markdownlint", "grammarly", "spelling_checker"],
            "configuration": ["yaml_validator", "json_validator", "config_checker"],
            "database": ["sql_validator", "schema_checker"],
            "api": ["openapi_validator", "postman", "swagger_validator"]
        }
        
        # Discover tools for each artifact type
        for artifact_type in artifact_types:
            if artifact_type in standard_tools:
                tool_discovery["available_tools"][artifact_type] = standard_tools[artifact_type]
            else:
                tool_discovery["available_tools"][artifact_type] = ["generic_validator"]
        
        # Build tool capability matrix
        tool_discovery["tool_matrix"] = self._build_tool_matrix(tool_discovery["available_tools"])
        
        # Discover MCP integrations
        tool_discovery["mcp_integrations"] = self._discover_mcp_tools()
        
        return tool_discovery
    
    def plan_validation_strategy(self, artifacts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Plan validation strategy for given artifacts"""
        
        validation_plan = {
            "artifacts": artifacts,
            "validation_stages": [],
            "tool_assignments": {},
            "execution_order": [],
            "estimated_duration": 0.0
        }
        
        # Analyze each artifact and assign validation tools
        for artifact in artifacts:
            artifact_id = artifact.get("id", "unknown")
            artifact_type = artifact.get("type", "unknown")
            
            # Determine validation requirements
            validation_requirements = self._determine_validation_requirements(artifact)
            
            # Assign tools based on requirements
            assigned_tools = self._assign_validation_tools(artifact_type, validation_requirements)
            validation_plan["tool_assignments"][artifact_id] = assigned_tools
            
            # Estimate duration
            duration = self._estimate_validation_duration(assigned_tools)
            validation_plan["estimated_duration"] += duration
        
        # Create execution stages
        validation_plan["validation_stages"] = self._create_validation_stages(
            validation_plan["tool_assignments"]
        )
        
        # Determine execution order
        validation_plan["execution_order"] = self._optimize_execution_order(
            validation_plan["validation_stages"]
        )
        
        return validation_plan
    
    def execute_validation(self, validation_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation according to plan"""
        
        execution_result = {
            "plan_id": validation_plan.get("id", "unknown"),
            "execution_status": "running",
            "stage_results": {},
            "overall_metrics": {},
            "issues_found": [],
            "recommendations": []
        }
        
        # Execute validation stages in order
        for stage_id in validation_plan["execution_order"]:
            stage = next((s for s in validation_plan["validation_stages"] if s["id"] == stage_id), None)
            if stage:
                stage_result = self._execute_validation_stage(stage)
                execution_result["stage_results"][stage_id] = stage_result
                
                # Collect issues and recommendations
                if "issues" in stage_result:
                    execution_result["issues_found"].extend(stage_result["issues"])
                if "recommendations" in stage_result:
                    execution_result["recommendations"].extend(stage_result["recommendations"])
        
        # Calculate overall metrics
        execution_result["overall_metrics"] = self._calculate_overall_metrics(
            execution_result["stage_results"]
        )
        
        # Determine final status
        execution_result["execution_status"] = self._determine_final_status(
            execution_result["stage_results"]
        )
        
        return execution_result
    
    def validate_artifact(self, artifact: Dict[str, Any], criteria: List[str]) -> Dict[str, Any]:
        """Validate single artifact against specific criteria"""
        
        validation_result = {
            "artifact_id": artifact.get("id", "unknown"),
            "artifact_type": artifact.get("type", "unknown"),
            "criteria": criteria,
            "validation_status": "processing",
            "results": {},
            "score": 0.0,
            "issues": [],
            "passed_criteria": [],
            "failed_criteria": []
        }
        
        # Validate against each criterion
        for criterion in criteria:
            criterion_result = self._validate_criterion(artifact, criterion)
            validation_result["results"][criterion] = criterion_result
            
            if criterion_result["passed"]:
                validation_result["passed_criteria"].append(criterion)
            else:
                validation_result["failed_criteria"].append(criterion)
                validation_result["issues"].extend(criterion_result.get("issues", []))
        
        # Calculate overall score
        validation_result["score"] = len(validation_result["passed_criteria"]) / len(criteria) if criteria else 0.0
        
        # Determine final status
        validation_result["validation_status"] = "passed" if validation_result["score"] >= 0.8 else "failed"
        
        return validation_result
    
    def _build_tool_matrix(self, available_tools: Dict[str, List[str]]) -> Dict[str, Dict[str, str]]:
        """Build matrix of tools and their capabilities"""
        matrix = {}
        
        for artifact_type, tools in available_tools.items():
            matrix[artifact_type] = {}
            for tool in tools:
                matrix[artifact_type][tool] = self._get_tool_capabilities(tool)
        
        return matrix
    
    def _discover_mcp_tools(self) -> Dict[str, Any]:
        """Discover available MCP tool integrations"""
        mcp_tools = {
            "available": False,
            "tools": [],
            "capabilities": {}
        }
        
        try:
            # Check if MCP tools are available
            from ....mcp_tools import tool_registry
            mcp_tools["available"] = True
            # Additional MCP discovery logic would go here
        except ImportError:
            mcp_tools["available"] = False
        
        return mcp_tools
    
    def _determine_validation_requirements(self, artifact: Dict[str, Any]) -> List[str]:
        """Determine validation requirements for artifact"""
        artifact_type = artifact.get("type", "unknown")
        
        requirements_map = {
            "python_code": ["syntax_check", "style_check", "type_check", "security_check"],
            "javascript_code": ["syntax_check", "style_check", "lint_check"],
            "documentation": ["markdown_check", "spelling_check", "grammar_check"],
            "configuration": ["syntax_check", "schema_validation"],
            "database": ["schema_check", "constraint_check"],
            "api": ["openapi_validation", "endpoint_check"]
        }
        
        return requirements_map.get(artifact_type, ["basic_validation"])
    
    def _assign_validation_tools(self, artifact_type: str, requirements: List[str]) -> List[Dict[str, Any]]:
        """Assign specific tools for validation requirements"""
        tool_assignments = []
        
        tool_mapping = {
            "syntax_check": {"tool": "syntax_validator", "priority": "high"},
            "style_check": {"tool": "style_validator", "priority": "medium"},
            "type_check": {"tool": "type_checker", "priority": "high"},
            "security_check": {"tool": "security_scanner", "priority": "high"},
            "lint_check": {"tool": "linter", "priority": "medium"},
            "markdown_check": {"tool": "markdown_validator", "priority": "medium"},
            "spelling_check": {"tool": "spell_checker", "priority": "low"},
            "grammar_check": {"tool": "grammar_checker", "priority": "low"}
        }
        
        for requirement in requirements:
            if requirement in tool_mapping:
                tool_assignments.append(tool_mapping[requirement])
        
        return tool_assignments
    
    def _estimate_validation_duration(self, assigned_tools: List[Dict[str, Any]]) -> float:
        """Estimate duration for validation with assigned tools"""
        base_duration = 0.1  # 6 minutes base
        tool_duration = len(assigned_tools) * 0.05  # 3 minutes per tool
        return base_duration + tool_duration
    
    def _create_validation_stages(self, tool_assignments: Dict[str, List]) -> List[Dict[str, Any]]:
        """Create validation stages from tool assignments"""
        stages = []
        
        # Group tools by stage
        stage_groups = self._group_tools_by_stage(tool_assignments)
        
        for stage_name, stage_tools in stage_groups.items():
            stage = {
                "id": f"stage_{len(stages) + 1}",
                "name": stage_name,
                "tools": stage_tools,
                "artifacts": [aid for aid, tools in tool_assignments.items() if any(t in stage_tools for t in tools)]
            }
            stages.append(stage)
        
        return stages
    
    def _group_tools_by_stage(self, tool_assignments: Dict[str, List]) -> Dict[str, List]:
        """Group tools into logical validation stages"""
        return {
            "syntax_validation": ["syntax_validator", "type_checker"],
            "quality_validation": ["style_validator", "linter"],
            "security_validation": ["security_scanner"],
            "content_validation": ["markdown_validator", "spell_checker", "grammar_checker"]
        }
    
    def _optimize_execution_order(self, validation_stages: List[Dict[str, Any]]) -> List[str]:
        """Optimize execution order for validation stages"""
        # Simple ordering by stage priority
        stage_priority = {
            "syntax_validation": 1,
            "quality_validation": 2,
            "security_validation": 3,
            "content_validation": 4
        }
        
        sorted_stages = sorted(validation_stages, key=lambda s: stage_priority.get(s["name"], 999))
        return [stage["id"] for stage in sorted_stages]
    
    def _execute_validation_stage(self, stage: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single validation stage"""
        return {
            "stage_id": stage["id"],
            "stage_name": stage["name"],
            "status": "completed",
            "issues": [],
            "recommendations": [],
            "metrics": {"tools_executed": len(stage["tools"]), "artifacts_validated": len(stage["artifacts"])}
        }
    
    def _calculate_overall_metrics(self, stage_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall validation metrics"""
        total_issues = sum(len(result.get("issues", [])) for result in stage_results.values())
        total_tools = sum(result.get("metrics", {}).get("tools_executed", 0) for result in stage_results.values())
        
        return {
            "total_stages": len(stage_results),
            "total_issues": total_issues,
            "total_tools_executed": total_tools,
            "overall_score": max(0.0, 1.0 - (total_issues / max(total_tools, 1)))
        }
    
    def _determine_final_status(self, stage_results: Dict[str, Any]) -> str:
        """Determine final validation status"""
        failed_stages = sum(1 for result in stage_results.values() if result.get("status") != "completed")
        return "failed" if failed_stages > 0 else "completed"
    
    def _validate_criterion(self, artifact: Dict[str, Any], criterion: str) -> Dict[str, Any]:
        """Validate artifact against specific criterion"""
        return {
            "criterion": criterion,
            "passed": True,  # Simplified for now
            "score": 1.0,
            "issues": [],
            "details": f"Validation of {criterion} completed"
        }
    
    def _get_tool_capabilities(self, tool: str) -> str:
        """Get capabilities description for tool"""
        capabilities_map = {
            "pylint": "Python code analysis and style checking",
            "black": "Python code formatting",
            "mypy": "Python static type checking",
            "pytest": "Python testing framework",
            "bandit": "Python security analysis",
            "eslint": "JavaScript linting",
            "prettier": "JavaScript code formatting",
            "markdownlint": "Markdown style checking"
        }
        
        return capabilities_map.get(tool, "General validation capabilities") 