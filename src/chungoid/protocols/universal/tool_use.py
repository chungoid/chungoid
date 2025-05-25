"""
Tool Use Protocol for Autonomous Execution

Implements the tool use pattern for intelligent tool selection and coordination
based on real tool capabilities and task requirements.

This protocol enables agents to:
- Analyze available tool capabilities and constraints
- Select optimal tools based on task requirements and context
- Coordinate tool usage with dependency analysis and sequencing
- Monitor and optimize tool performance for efficiency

Week 2 Implementation: Modern Agentic Patterns with Real Tool Integration
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from ..base.protocol_interface import ProtocolInterface, ProtocolPhase, ProtocolTemplate, PhaseStatus
import logging
from dataclasses import dataclass
from enum import Enum


class ToolCapabilityType(Enum):
    """Types of tool capabilities."""
    FILESYSTEM = "filesystem"
    DATABASE = "database"
    TERMINAL = "terminal"
    CONTENT = "content"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    COMMUNICATION = "communication"


@dataclass
class ToolCapability:
    """Represents a tool's capability."""
    name: str
    capability_type: ToolCapabilityType
    input_types: List[str]
    output_types: List[str]
    dependencies: List[str]
    performance_metrics: Dict[str, float]
    constraints: Dict[str, Any]
    reliability_score: float = 0.9


@dataclass
class ToolSelectionCriteria:
    """Criteria for tool selection."""
    required_capabilities: List[ToolCapabilityType]
    preferred_tools: List[str]
    performance_requirements: Dict[str, float]
    constraint_requirements: Dict[str, Any]
    optimization_goals: List[str]


@dataclass
class ToolCoordinationPlan:
    """Plan for coordinating multiple tools."""
    tool_sequence: List[str]
    parallel_groups: List[List[str]]
    dependency_graph: Dict[str, List[str]]
    resource_allocation: Dict[str, float]
    validation_checkpoints: List[str]


class ToolUseProtocol(ProtocolInterface):
    """
    Protocol for intelligent tool selection and coordination
    using real tool capability analysis and optimization.
    """
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.tool_capabilities: Dict[str, ToolCapability] = {}
        self.tool_performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.coordination_patterns: Dict[str, ToolCoordinationPlan] = {}
        
    @property
    def name(self) -> str:
        return "tool_use"
    
    @property
    def description(self) -> str:
        return "Intelligent tool selection and coordination based on real tool capabilities"
    
    @property
    def total_estimated_time(self) -> float:
        return 1.5  # 1.5 hours for complete tool use optimization
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        """Initialize tool use protocol phases."""
        return [
            ProtocolPhase(
                name="capability_analysis",
                description="Analyze available tool capabilities and constraints",
                time_box_hours=0.3,
                required_outputs=[
                    "tool_capability_map",
                    "performance_baselines",
                    "constraint_analysis",
                    "capability_gaps"
                ],
                validation_criteria=[
                    "all_tools_analyzed",
                    "capabilities_mapped_accurately",
                    "performance_baselines_established"
                ],
                tools_required=[
                    "filesystem_project_scan",
                    "terminal_get_environment",
                    "chroma_list_collections",
                    "content_validate"
                ]
            ),
            
            ProtocolPhase(
                name="intelligent_selection",
                description="Select optimal tools based on task requirements",
                time_box_hours=0.4,
                required_outputs=[
                    "tool_selection_strategy",
                    "optimization_criteria",
                    "selection_rationale",
                    "alternative_options"
                ],
                validation_criteria=[
                    "optimal_tools_selected",
                    "selection_criteria_met",
                    "alternatives_considered"
                ],
                tools_required=[
                    "content_generate",
                    "chroma_query_documents",
                    "filesystem_read_file"
                ],
                dependencies=["capability_analysis"]
            ),
            
            ProtocolPhase(
                name="coordination_planning",
                description="Plan tool coordination with dependency analysis",
                time_box_hours=0.4,
                required_outputs=[
                    "coordination_plan",
                    "dependency_graph",
                    "execution_sequence",
                    "resource_allocation"
                ],
                validation_criteria=[
                    "coordination_plan_created",
                    "dependencies_resolved",
                    "execution_sequence_optimized"
                ],
                tools_required=[
                    "content_generate",
                    "filesystem_write_file",
                    "chroma_store_document"
                ],
                dependencies=["intelligent_selection"]
            ),
            
            ProtocolPhase(
                name="performance_optimization",
                description="Monitor and optimize tool performance",
                time_box_hours=0.3,
                required_outputs=[
                    "performance_metrics",
                    "optimization_recommendations",
                    "efficiency_improvements",
                    "monitoring_setup"
                ],
                validation_criteria=[
                    "performance_monitored",
                    "optimizations_identified",
                    "monitoring_established"
                ],
                tools_required=[
                    "terminal_execute_command",
                    "chromadb_batch_operations",
                    "filesystem_project_scan"
                ],
                dependencies=["coordination_planning"]
            ),
            
            ProtocolPhase(
                name="adaptive_learning",
                description="Learn from tool usage patterns for future optimization",
                time_box_hours=0.1,
                required_outputs=[
                    "usage_patterns",
                    "learning_insights",
                    "adaptation_strategies",
                    "future_recommendations"
                ],
                validation_criteria=[
                    "patterns_identified",
                    "insights_captured",
                    "adaptations_planned"
                ],
                tools_required=[
                    "chroma_store_document",
                    "chromadb_reflection_query",
                    "content_generate"
                ],
                dependencies=["performance_optimization"]
            )
        ]
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize tool use protocol templates."""
        return {
            "capability_analysis_template": ProtocolTemplate(
                name="capability_analysis_template",
                description="Template for analyzing tool capabilities",
                template_content="""
# Tool Capability Analysis Report

## Available Tools Overview
**Total Tools Analyzed**: [total_tools]
**Tool Categories**: [tool_categories]
**Analysis Date**: [analysis_date]

## Tool Capabilities by Category

### Filesystem Tools
**Tools**: [filesystem_tools]
**Capabilities**: [filesystem_capabilities]
**Performance**: [filesystem_performance]
**Constraints**: [filesystem_constraints]

### Database Tools
**Tools**: [database_tools]
**Capabilities**: [database_capabilities]
**Performance**: [database_performance]
**Constraints**: [database_constraints]

### Terminal Tools
**Tools**: [terminal_tools]
**Capabilities**: [terminal_capabilities]
**Performance**: [terminal_performance]
**Constraints**: [terminal_constraints]

### Content Tools
**Tools**: [content_tools]
**Capabilities**: [content_capabilities]
**Performance**: [content_performance]
**Constraints**: [content_constraints]

## Performance Baselines
**Average Response Time**: [avg_response_time]ms
**Success Rate**: [success_rate]%
**Resource Utilization**: [resource_utilization]%
**Reliability Score**: [reliability_score]/10

## Capability Gaps
**Missing Capabilities**: [missing_capabilities]
**Improvement Areas**: [improvement_areas]
**Recommended Additions**: [recommended_additions]

## Tool Interaction Matrix
[tool_interaction_matrix]
""",
                variables=["total_tools", "tool_categories", "analysis_date",
                          "filesystem_tools", "filesystem_capabilities", "filesystem_performance", "filesystem_constraints",
                          "database_tools", "database_capabilities", "database_performance", "database_constraints",
                          "terminal_tools", "terminal_capabilities", "terminal_performance", "terminal_constraints",
                          "content_tools", "content_capabilities", "content_performance", "content_constraints",
                          "avg_response_time", "success_rate", "resource_utilization", "reliability_score",
                          "missing_capabilities", "improvement_areas", "recommended_additions", "tool_interaction_matrix"]
            ),
            
            "tool_selection_template": ProtocolTemplate(
                name="tool_selection_template",
                description="Template for intelligent tool selection",
                template_content="""
# Intelligent Tool Selection Strategy

## Task Requirements Analysis
**Primary Objective**: [primary_objective]
**Required Capabilities**: [required_capabilities]
**Performance Requirements**: [performance_requirements]
**Constraint Requirements**: [constraint_requirements]

## Selected Tools

### Primary Tools
1. **[primary_tool_1]**
   - **Capability**: [primary_tool_1_capability]
   - **Selection Reason**: [primary_tool_1_reason]
   - **Expected Performance**: [primary_tool_1_performance]

2. **[primary_tool_2]**
   - **Capability**: [primary_tool_2_capability]
   - **Selection Reason**: [primary_tool_2_reason]
   - **Expected Performance**: [primary_tool_2_performance]

3. **[primary_tool_3]**
   - **Capability**: [primary_tool_3_capability]
   - **Selection Reason**: [primary_tool_3_reason]
   - **Expected Performance**: [primary_tool_3_performance]

### Supporting Tools
**Validation Tools**: [validation_tools]
**Monitoring Tools**: [monitoring_tools]
**Backup Tools**: [backup_tools]

## Selection Criteria Met
**Capability Coverage**: [capability_coverage]%
**Performance Score**: [performance_score]/10
**Reliability Score**: [reliability_score]/10
**Efficiency Score**: [efficiency_score]/10

## Alternative Options
**Alternative 1**: [alternative_1] - [alternative_1_reason]
**Alternative 2**: [alternative_2] - [alternative_2_reason]
**Alternative 3**: [alternative_3] - [alternative_3_reason]

## Optimization Goals
**Primary Goal**: [primary_optimization_goal]
**Secondary Goals**: [secondary_optimization_goals]
**Success Metrics**: [success_metrics]
""",
                variables=["primary_objective", "required_capabilities", "performance_requirements", "constraint_requirements",
                          "primary_tool_1", "primary_tool_1_capability", "primary_tool_1_reason", "primary_tool_1_performance",
                          "primary_tool_2", "primary_tool_2_capability", "primary_tool_2_reason", "primary_tool_2_performance",
                          "primary_tool_3", "primary_tool_3_capability", "primary_tool_3_reason", "primary_tool_3_performance",
                          "validation_tools", "monitoring_tools", "backup_tools",
                          "capability_coverage", "performance_score", "reliability_score", "efficiency_score",
                          "alternative_1", "alternative_1_reason", "alternative_2", "alternative_2_reason",
                          "alternative_3", "alternative_3_reason",
                          "primary_optimization_goal", "secondary_optimization_goals", "success_metrics"]
            ),
            
            "coordination_plan_template": ProtocolTemplate(
                name="coordination_plan_template",
                description="Template for tool coordination planning",
                template_content="""
# Tool Coordination Plan

## Execution Strategy
**Coordination Approach**: [coordination_approach]
**Total Execution Time**: [total_execution_time]
**Parallel Execution**: [parallel_execution_enabled]
**Resource Requirements**: [resource_requirements]

## Tool Execution Sequence
### Phase 1: [phase_1_name]
**Tools**: [phase_1_tools]
**Duration**: [phase_1_duration]
**Dependencies**: [phase_1_dependencies]
**Outputs**: [phase_1_outputs]

### Phase 2: [phase_2_name]
**Tools**: [phase_2_tools]
**Duration**: [phase_2_duration]
**Dependencies**: [phase_2_dependencies]
**Outputs**: [phase_2_outputs]

### Phase 3: [phase_3_name]
**Tools**: [phase_3_tools]
**Duration**: [phase_3_duration]
**Dependencies**: [phase_3_dependencies]
**Outputs**: [phase_3_outputs]

## Dependency Graph
**Critical Path**: [critical_path]
**Parallel Opportunities**: [parallel_opportunities]
**Bottlenecks**: [bottlenecks]
**Risk Points**: [risk_points]

## Resource Allocation
**CPU Allocation**: [cpu_allocation]
**Memory Allocation**: [memory_allocation]
**I/O Allocation**: [io_allocation]
**Network Allocation**: [network_allocation]

## Validation Checkpoints
**Checkpoint 1**: [checkpoint_1] - [checkpoint_1_criteria]
**Checkpoint 2**: [checkpoint_2] - [checkpoint_2_criteria]
**Checkpoint 3**: [checkpoint_3] - [checkpoint_3_criteria]

## Error Handling
**Failure Recovery**: [failure_recovery_strategy]
**Rollback Plan**: [rollback_plan]
**Alternative Paths**: [alternative_paths]
""",
                variables=["coordination_approach", "total_execution_time", "parallel_execution_enabled", "resource_requirements",
                          "phase_1_name", "phase_1_tools", "phase_1_duration", "phase_1_dependencies", "phase_1_outputs",
                          "phase_2_name", "phase_2_tools", "phase_2_duration", "phase_2_dependencies", "phase_2_outputs",
                          "phase_3_name", "phase_3_tools", "phase_3_duration", "phase_3_dependencies", "phase_3_outputs",
                          "critical_path", "parallel_opportunities", "bottlenecks", "risk_points",
                          "cpu_allocation", "memory_allocation", "io_allocation", "network_allocation",
                          "checkpoint_1", "checkpoint_1_criteria", "checkpoint_2", "checkpoint_2_criteria",
                          "checkpoint_3", "checkpoint_3_criteria",
                          "failure_recovery_strategy", "rollback_plan", "alternative_paths"]
            )
        }
    
    def analyze_tool_capabilities(self, available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze capabilities of available tools."""
        capability_analysis = {
            "tool_capabilities": self._map_tool_capabilities(available_tools),
            "performance_baselines": self._establish_performance_baselines(available_tools),
            "constraint_analysis": self._analyze_constraints(available_tools),
            "capability_gaps": self._identify_capability_gaps(available_tools)
        }
        
        self.logger.info(f"Analyzed capabilities of {len(available_tools)} tools")
        return capability_analysis
    
    def select_optimal_tools(self, task_requirements: Dict[str, Any], 
                           available_tools: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal tools based on task requirements."""
        selection_criteria = self._create_selection_criteria(task_requirements)
        
        tool_selection = {
            "primary_tools": self._select_primary_tools(selection_criteria, available_tools),
            "supporting_tools": self._select_supporting_tools(selection_criteria, available_tools),
            "selection_rationale": self._generate_selection_rationale(selection_criteria),
            "alternatives": self._identify_alternatives(selection_criteria, available_tools)
        }
        
        self.logger.info(f"Selected {len(tool_selection['primary_tools'])} primary tools for task")
        return tool_selection
    
    def plan_tool_coordination(self, selected_tools: List[str], 
                             task_context: Dict[str, Any]) -> ToolCoordinationPlan:
        """Plan coordination of selected tools."""
        coordination_plan = ToolCoordinationPlan(
            tool_sequence=self._determine_execution_sequence(selected_tools, task_context),
            parallel_groups=self._identify_parallel_groups(selected_tools, task_context),
            dependency_graph=self._build_dependency_graph(selected_tools),
            resource_allocation=self._allocate_resources(selected_tools, task_context),
            validation_checkpoints=self._define_validation_checkpoints(selected_tools)
        )
        
        self.coordination_patterns[task_context.get('task_id', 'default')] = coordination_plan
        self.logger.info(f"Created coordination plan for {len(selected_tools)} tools")
        return coordination_plan
    
    def optimize_tool_performance(self, coordination_plan: ToolCoordinationPlan,
                                 performance_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize tool performance based on monitoring data."""
        optimization_results = {
            "performance_metrics": self._calculate_performance_metrics(performance_data),
            "bottleneck_analysis": self._analyze_bottlenecks(coordination_plan, performance_data),
            "optimization_recommendations": self._generate_optimizations(coordination_plan, performance_data),
            "efficiency_improvements": self._identify_efficiency_improvements(performance_data)
        }
        
        self.logger.info("Tool performance optimization completed")
        return optimization_results
    
    def learn_from_usage_patterns(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Learn from tool usage patterns for future optimization."""
        learning_results = {
            "usage_patterns": self._analyze_usage_patterns(usage_history),
            "success_patterns": self._identify_success_patterns(usage_history),
            "failure_patterns": self._identify_failure_patterns(usage_history),
            "adaptation_strategies": self._develop_adaptation_strategies(usage_history)
        }
        
        self.logger.info("Learning from usage patterns completed")
        return learning_results
    
    # Helper methods for tool capability analysis
    
    def _map_tool_capabilities(self, tools: Dict[str, Any]) -> Dict[str, ToolCapability]:
        """Map capabilities of available tools."""
        capabilities = {}
        
        for tool_name, tool_info in tools.items():
            capability_type = self._determine_capability_type(tool_name)
            
            capabilities[tool_name] = ToolCapability(
                name=tool_name,
                capability_type=capability_type,
                input_types=self._analyze_input_types(tool_name, tool_info),
                output_types=self._analyze_output_types(tool_name, tool_info),
                dependencies=self._analyze_dependencies(tool_name, tool_info),
                performance_metrics=self._get_performance_metrics(tool_name),
                constraints=self._analyze_constraints_for_tool(tool_name, tool_info),
                reliability_score=self._calculate_reliability_score(tool_name)
            )
        
        self.tool_capabilities = capabilities
        return capabilities
    
    def _establish_performance_baselines(self, tools: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """Establish performance baselines for tools."""
        baselines = {}
        
        for tool_name in tools.keys():
            baselines[tool_name] = {
                "avg_response_time": self._get_avg_response_time(tool_name),
                "success_rate": self._get_success_rate(tool_name),
                "throughput": self._get_throughput(tool_name),
                "resource_usage": self._get_resource_usage(tool_name)
            }
        
        return baselines
    
    def _analyze_constraints(self, tools: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze constraints for tool usage."""
        constraints = {}
        
        for tool_name in tools.keys():
            tool_constraints = []
            
            # Analyze common constraints
            if "filesystem" in tool_name:
                tool_constraints.extend(["file_permissions", "disk_space", "path_access"])
            elif "terminal" in tool_name:
                tool_constraints.extend(["execution_permissions", "environment_variables", "command_availability"])
            elif "chroma" in tool_name:
                tool_constraints.extend(["database_connection", "collection_access", "memory_limits"])
            elif "content" in tool_name:
                tool_constraints.extend(["network_access", "content_size_limits", "format_support"])
            
            constraints[tool_name] = tool_constraints
        
        return constraints
    
    def _identify_capability_gaps(self, tools: Dict[str, Any]) -> List[str]:
        """Identify gaps in tool capabilities."""
        gaps = []
        
        # Check for missing capability types
        available_types = set()
        for tool_name in tools.keys():
            available_types.add(self._determine_capability_type(tool_name))
        
        all_types = set(ToolCapabilityType)
        missing_types = all_types - available_types
        
        for missing_type in missing_types:
            gaps.append(f"Missing {missing_type.value} capabilities")
        
        # Check for specific capability gaps
        if not any("batch" in tool for tool in tools.keys()):
            gaps.append("Missing batch operation capabilities")
        
        if not any("validate" in tool for tool in tools.keys()):
            gaps.append("Missing validation capabilities")
        
        return gaps
    
    def _determine_capability_type(self, tool_name: str) -> ToolCapabilityType:
        """Determine the capability type of a tool."""
        if "filesystem" in tool_name:
            return ToolCapabilityType.FILESYSTEM
        elif "chroma" in tool_name:
            return ToolCapabilityType.DATABASE
        elif "terminal" in tool_name:
            return ToolCapabilityType.TERMINAL
        elif "content" in tool_name:
            return ToolCapabilityType.CONTENT
        elif "validate" in tool_name:
            return ToolCapabilityType.VALIDATION
        else:
            return ToolCapabilityType.ANALYSIS
    
    def _analyze_input_types(self, tool_name: str, tool_info: Any) -> List[str]:
        """Analyze input types for a tool."""
        input_types = []
        
        # Common input type patterns
        if "filesystem" in tool_name:
            if "read" in tool_name:
                input_types.extend(["file_path", "encoding"])
            elif "write" in tool_name:
                input_types.extend(["file_path", "content", "encoding"])
            elif "list" in tool_name:
                input_types.extend(["directory_path", "recursive"])
        
        elif "chroma" in tool_name:
            if "query" in tool_name:
                input_types.extend(["collection_name", "query_texts", "n_results"])
            elif "add" in tool_name:
                input_types.extend(["collection_name", "documents", "metadata"])
        
        elif "terminal" in tool_name:
            input_types.extend(["command", "working_directory", "environment"])
        
        elif "content" in tool_name:
            input_types.extend(["url", "content_type", "parameters"])
        
        return input_types
    
    def _analyze_output_types(self, tool_name: str, tool_info: Any) -> List[str]:
        """Analyze output types for a tool."""
        output_types = []
        
        # Common output type patterns
        if "filesystem" in tool_name:
            if "read" in tool_name:
                output_types.extend(["content", "metadata", "status"])
            elif "write" in tool_name:
                output_types.extend(["success", "file_info", "status"])
            elif "list" in tool_name:
                output_types.extend(["file_list", "directory_info", "status"])
        
        elif "chroma" in tool_name:
            if "query" in tool_name:
                output_types.extend(["documents", "distances", "metadata"])
            elif "add" in tool_name:
                output_types.extend(["ids", "success", "status"])
        
        elif "terminal" in tool_name:
            output_types.extend(["stdout", "stderr", "return_code", "status"])
        
        elif "content" in tool_name:
            output_types.extend(["content", "metadata", "status"])
        
        return output_types
    
    def _analyze_dependencies(self, tool_name: str, tool_info: Any) -> List[str]:
        """Analyze dependencies for a tool."""
        dependencies = []
        
        # Common dependency patterns
        if "filesystem" in tool_name:
            dependencies.extend(["file_system_access", "path_permissions"])
        
        elif "chroma" in tool_name:
            dependencies.extend(["chromadb_connection", "collection_existence"])
        
        elif "terminal" in tool_name:
            dependencies.extend(["shell_access", "command_availability"])
        
        elif "content" in tool_name:
            dependencies.extend(["network_access", "url_accessibility"])
        
        return dependencies
    
    def _get_performance_metrics(self, tool_name: str) -> Dict[str, float]:
        """Get performance metrics for a tool."""
        # Placeholder implementation - would use real performance data
        return {
            "avg_response_time": 100.0,  # ms
            "success_rate": 0.95,
            "throughput": 10.0,  # operations/second
            "resource_usage": 0.1  # normalized
        }
    
    def _analyze_constraints_for_tool(self, tool_name: str, tool_info: Any) -> Dict[str, Any]:
        """Analyze constraints for a specific tool."""
        constraints = {}
        
        # Common constraint patterns
        if "filesystem" in tool_name:
            constraints.update({
                "max_file_size": "100MB",
                "supported_encodings": ["utf-8", "ascii"],
                "concurrent_operations": 5
            })
        
        elif "chroma" in tool_name:
            constraints.update({
                "max_documents": 10000,
                "max_query_size": 1000,
                "concurrent_queries": 10
            })
        
        elif "terminal" in tool_name:
            constraints.update({
                "max_execution_time": 300,  # seconds
                "allowed_commands": ["safe_commands_only"],
                "concurrent_executions": 3
            })
        
        return constraints
    
    def _calculate_reliability_score(self, tool_name: str) -> float:
        """Calculate reliability score for a tool."""
        # Placeholder implementation - would use historical data
        base_score = 0.9
        
        # Adjust based on tool type
        if "filesystem" in tool_name:
            base_score = 0.95  # Filesystem operations are generally reliable
        elif "terminal" in tool_name:
            base_score = 0.85  # Terminal operations can be less predictable
        elif "chroma" in tool_name:
            base_score = 0.90  # Database operations are moderately reliable
        elif "content" in tool_name:
            base_score = 0.80  # Network operations can be unreliable
        
        return base_score
    
    # Helper methods for tool selection
    
    def _create_selection_criteria(self, task_requirements: Dict[str, Any]) -> ToolSelectionCriteria:
        """Create selection criteria from task requirements."""
        return ToolSelectionCriteria(
            required_capabilities=self._extract_required_capabilities(task_requirements),
            preferred_tools=task_requirements.get('preferred_tools', []),
            performance_requirements=task_requirements.get('performance_requirements', {}),
            constraint_requirements=task_requirements.get('constraint_requirements', {}),
            optimization_goals=task_requirements.get('optimization_goals', ['efficiency', 'reliability'])
        )
    
    def _extract_required_capabilities(self, task_requirements: Dict[str, Any]) -> List[ToolCapabilityType]:
        """Extract required capabilities from task requirements."""
        capabilities = []
        
        task_type = task_requirements.get('task_type', '')
        
        if 'file' in task_type or 'filesystem' in task_type:
            capabilities.append(ToolCapabilityType.FILESYSTEM)
        
        if 'database' in task_type or 'query' in task_type:
            capabilities.append(ToolCapabilityType.DATABASE)
        
        if 'execute' in task_type or 'command' in task_type:
            capabilities.append(ToolCapabilityType.TERMINAL)
        
        if 'content' in task_type or 'web' in task_type:
            capabilities.append(ToolCapabilityType.CONTENT)
        
        if 'validate' in task_type or 'check' in task_type:
            capabilities.append(ToolCapabilityType.VALIDATION)
        
        return capabilities
    
    def _select_primary_tools(self, criteria: ToolSelectionCriteria, 
                            available_tools: Dict[str, Any]) -> List[str]:
        """Select primary tools based on criteria."""
        primary_tools = []
        
        # Score tools based on criteria
        tool_scores = {}
        for tool_name, capability in self.tool_capabilities.items():
            score = self._calculate_tool_score(tool_name, capability, criteria)
            tool_scores[tool_name] = score
        
        # Select top tools for each required capability
        for required_capability in criteria.required_capabilities:
            best_tool = self._find_best_tool_for_capability(required_capability, tool_scores)
            if best_tool and best_tool not in primary_tools:
                primary_tools.append(best_tool)
        
        # Add preferred tools if they meet criteria
        for preferred_tool in criteria.preferred_tools:
            if (preferred_tool in available_tools and 
                preferred_tool not in primary_tools and
                self._meets_criteria(preferred_tool, criteria)):
                primary_tools.append(preferred_tool)
        
        return primary_tools[:5]  # Limit to top 5 primary tools
    
    def _select_supporting_tools(self, criteria: ToolSelectionCriteria,
                               available_tools: Dict[str, Any]) -> List[str]:
        """Select supporting tools for validation and monitoring."""
        supporting_tools = []
        
        # Add validation tools
        for tool_name in available_tools.keys():
            if "validate" in tool_name or "check" in tool_name:
                supporting_tools.append(tool_name)
        
        # Add monitoring tools
        for tool_name in available_tools.keys():
            if "monitor" in tool_name or "status" in tool_name:
                supporting_tools.append(tool_name)
        
        return supporting_tools[:3]  # Limit to top 3 supporting tools
    
    def _calculate_tool_score(self, tool_name: str, capability: ToolCapability,
                            criteria: ToolSelectionCriteria) -> float:
        """Calculate score for a tool based on selection criteria."""
        score = 0.0
        
        # Capability match score
        if capability.capability_type in criteria.required_capabilities:
            score += 0.4
        
        # Performance score
        performance_metrics = capability.performance_metrics
        for req_metric, req_value in criteria.performance_requirements.items():
            if req_metric in performance_metrics:
                if performance_metrics[req_metric] >= req_value:
                    score += 0.2
        
        # Reliability score
        score += capability.reliability_score * 0.3
        
        # Preference bonus
        if tool_name in criteria.preferred_tools:
            score += 0.1
        
        return score
    
    def _find_best_tool_for_capability(self, capability: ToolCapabilityType,
                                     tool_scores: Dict[str, float]) -> Optional[str]:
        """Find the best tool for a specific capability."""
        best_tool = None
        best_score = 0.0
        
        for tool_name, capability_info in self.tool_capabilities.items():
            if capability_info.capability_type == capability:
                score = tool_scores.get(tool_name, 0.0)
                if score > best_score:
                    best_score = score
                    best_tool = tool_name
        
        return best_tool
    
    def _meets_criteria(self, tool_name: str, criteria: ToolSelectionCriteria) -> bool:
        """Check if a tool meets the selection criteria."""
        if tool_name not in self.tool_capabilities:
            return False
        
        capability = self.tool_capabilities[tool_name]
        
        # Check performance requirements
        for req_metric, req_value in criteria.performance_requirements.items():
            if req_metric in capability.performance_metrics:
                if capability.performance_metrics[req_metric] < req_value:
                    return False
        
        # Check constraint requirements
        for req_constraint, req_value in criteria.constraint_requirements.items():
            if req_constraint in capability.constraints:
                if capability.constraints[req_constraint] != req_value:
                    return False
        
        return True
    
    def _generate_selection_rationale(self, criteria: ToolSelectionCriteria) -> Dict[str, str]:
        """Generate rationale for tool selection."""
        return {
            "capability_coverage": f"Selected tools cover {len(criteria.required_capabilities)} required capabilities",
            "performance_optimization": f"Tools optimized for {', '.join(criteria.optimization_goals)}",
            "constraint_compliance": "All selected tools meet specified constraints",
            "reliability_focus": "Prioritized tools with high reliability scores"
        }
    
    def _identify_alternatives(self, criteria: ToolSelectionCriteria,
                             available_tools: Dict[str, Any]) -> List[Dict[str, str]]:
        """Identify alternative tool options."""
        alternatives = []
        
        # Find alternative tools for each capability
        for capability in criteria.required_capabilities:
            alternative_tools = []
            for tool_name, tool_capability in self.tool_capabilities.items():
                if tool_capability.capability_type == capability:
                    alternative_tools.append(tool_name)
            
            if len(alternative_tools) > 1:
                alternatives.append({
                    "capability": capability.value,
                    "alternatives": ", ".join(alternative_tools[:3]),
                    "reason": "Multiple tools available for this capability"
                })
        
        return alternatives
    
    # Helper methods for coordination planning
    
    def _determine_execution_sequence(self, tools: List[str], 
                                    context: Dict[str, Any]) -> List[str]:
        """Determine optimal execution sequence for tools."""
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(tools)
        
        # Perform topological sort to determine sequence
        sequence = self._topological_sort(dependency_graph)
        
        return sequence
    
    def _identify_parallel_groups(self, tools: List[str],
                                context: Dict[str, Any]) -> List[List[str]]:
        """Identify groups of tools that can run in parallel."""
        parallel_groups = []
        
        # Group tools by capability type that don't have dependencies
        capability_groups = {}
        for tool in tools:
            if tool in self.tool_capabilities:
                capability_type = self.tool_capabilities[tool].capability_type
                if capability_type not in capability_groups:
                    capability_groups[capability_type] = []
                capability_groups[capability_type].append(tool)
        
        # Tools of different types can often run in parallel
        for capability_type, tool_group in capability_groups.items():
            if len(tool_group) > 1:
                # Check if tools in the group can run in parallel
                parallel_group = self._filter_parallel_compatible(tool_group)
                if len(parallel_group) > 1:
                    parallel_groups.append(parallel_group)
        
        return parallel_groups
    
    def _build_dependency_graph(self, tools: List[str]) -> Dict[str, List[str]]:
        """Build dependency graph for tools."""
        dependency_graph = {}
        
        for tool in tools:
            dependencies = []
            if tool in self.tool_capabilities:
                tool_dependencies = self.tool_capabilities[tool].dependencies
                
                # Find dependencies that are in our tool list
                for dep in tool_dependencies:
                    for other_tool in tools:
                        if dep in other_tool or other_tool in dep:
                            dependencies.append(other_tool)
            
            dependency_graph[tool] = dependencies
        
        return dependency_graph
    
    def _allocate_resources(self, tools: List[str], 
                          context: Dict[str, Any]) -> Dict[str, float]:
        """Allocate resources for tool execution."""
        total_tools = len(tools)
        base_allocation = 1.0 / total_tools if total_tools > 0 else 1.0
        
        resource_allocation = {}
        
        for tool in tools:
            # Adjust allocation based on tool characteristics
            allocation = base_allocation
            
            if tool in self.tool_capabilities:
                capability = self.tool_capabilities[tool]
                
                # Heavy I/O tools get more resources
                if capability.capability_type in [ToolCapabilityType.FILESYSTEM, ToolCapabilityType.DATABASE]:
                    allocation *= 1.2
                
                # Terminal tools might need more CPU
                elif capability.capability_type == ToolCapabilityType.TERMINAL:
                    allocation *= 1.1
                
                # Content tools might need more network
                elif capability.capability_type == ToolCapabilityType.CONTENT:
                    allocation *= 1.15
            
            resource_allocation[tool] = min(allocation, 0.5)  # Cap at 50% of resources
        
        # Normalize allocations to sum to 1.0
        total_allocation = sum(resource_allocation.values())
        if total_allocation > 0:
            for tool in resource_allocation:
                resource_allocation[tool] /= total_allocation
        
        return resource_allocation
    
    def _define_validation_checkpoints(self, tools: List[str]) -> List[str]:
        """Define validation checkpoints for tool execution."""
        checkpoints = []
        
        # Add checkpoint after each major phase
        tool_phases = self._group_tools_by_phase(tools)
        
        for phase_name, phase_tools in tool_phases.items():
            checkpoint_name = f"validate_{phase_name}_completion"
            checkpoints.append(checkpoint_name)
        
        # Add final validation checkpoint
        checkpoints.append("validate_final_results")
        
        return checkpoints
    
    def _topological_sort(self, dependency_graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph."""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                # Cycle detected, skip
                return
            if node in visited:
                return
            
            temp_visited.add(node)
            for dependency in dependency_graph.get(node, []):
                visit(dependency)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for node in dependency_graph:
            if node not in visited:
                visit(node)
        
        return result
    
    def _filter_parallel_compatible(self, tools: List[str]) -> List[str]:
        """Filter tools that can run in parallel."""
        compatible_tools = []
        
        for tool in tools:
            # Check if tool supports parallel execution
            if tool in self.tool_capabilities:
                constraints = self.tool_capabilities[tool].constraints
                concurrent_ops = constraints.get('concurrent_operations', 1)
                
                if concurrent_ops > 1:
                    compatible_tools.append(tool)
        
        return compatible_tools
    
    def _group_tools_by_phase(self, tools: List[str]) -> Dict[str, List[str]]:
        """Group tools by execution phase."""
        phases = {
            "preparation": [],
            "execution": [],
            "validation": []
        }
        
        for tool in tools:
            if "validate" in tool or "check" in tool:
                phases["validation"].append(tool)
            elif "prepare" in tool or "setup" in tool:
                phases["preparation"].append(tool)
            else:
                phases["execution"].append(tool)
        
        return phases
    
    # Helper methods for performance optimization
    
    def _calculate_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate performance metrics from monitoring data."""
        metrics = {}
        
        if 'response_times' in performance_data:
            response_times = performance_data['response_times']
            metrics['avg_response_time'] = sum(response_times) / len(response_times)
            metrics['max_response_time'] = max(response_times)
            metrics['min_response_time'] = min(response_times)
        
        if 'success_count' in performance_data and 'total_count' in performance_data:
            metrics['success_rate'] = performance_data['success_count'] / performance_data['total_count']
        
        if 'throughput' in performance_data:
            metrics['throughput'] = performance_data['throughput']
        
        return metrics
    
    def _analyze_bottlenecks(self, plan: ToolCoordinationPlan,
                           performance_data: Dict[str, Any]) -> List[str]:
        """Analyze bottlenecks in tool execution."""
        bottlenecks = []
        
        # Analyze critical path for bottlenecks
        for tool in plan.tool_sequence:
            if tool in performance_data:
                tool_data = performance_data[tool]
                
                # Check for slow response times
                if tool_data.get('avg_response_time', 0) > 1000:  # > 1 second
                    bottlenecks.append(f"{tool}: Slow response time")
                
                # Check for low success rates
                if tool_data.get('success_rate', 1.0) < 0.9:  # < 90%
                    bottlenecks.append(f"{tool}: Low success rate")
                
                # Check for resource contention
                if tool_data.get('resource_usage', 0) > 0.8:  # > 80%
                    bottlenecks.append(f"{tool}: High resource usage")
        
        return bottlenecks
    
    def _generate_optimizations(self, plan: ToolCoordinationPlan,
                              performance_data: Dict[str, Any]) -> List[str]:
        """Generate optimization recommendations."""
        optimizations = []
        
        # Analyze parallel execution opportunities
        if len(plan.parallel_groups) < len(plan.tool_sequence) / 2:
            optimizations.append("Increase parallel execution of independent tools")
        
        # Analyze resource allocation
        for tool, allocation in plan.resource_allocation.items():
            if tool in performance_data:
                actual_usage = performance_data[tool].get('resource_usage', 0)
                if actual_usage < allocation * 0.5:
                    optimizations.append(f"Reduce resource allocation for {tool}")
                elif actual_usage > allocation * 1.2:
                    optimizations.append(f"Increase resource allocation for {tool}")
        
        # Analyze tool selection
        for tool in plan.tool_sequence:
            if tool in performance_data:
                success_rate = performance_data[tool].get('success_rate', 1.0)
                if success_rate < 0.8:
                    optimizations.append(f"Consider alternative tool for {tool}")
        
        return optimizations
    
    def _identify_efficiency_improvements(self, performance_data: Dict[str, Any]) -> List[str]:
        """Identify efficiency improvements."""
        improvements = []
        
        # Analyze overall efficiency
        total_time = sum(data.get('execution_time', 0) for data in performance_data.values())
        if total_time > 300:  # > 5 minutes
            improvements.append("Optimize overall execution time")
        
        # Analyze tool-specific improvements
        for tool, data in performance_data.items():
            if data.get('cache_hit_rate', 0) < 0.5:
                improvements.append(f"Improve caching for {tool}")
            
            if data.get('retry_count', 0) > 2:
                improvements.append(f"Reduce retry frequency for {tool}")
        
        return improvements
    
    # Helper methods for learning and adaptation
    
    def _analyze_usage_patterns(self, usage_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in tool usage."""
        patterns = {
            "most_used_tools": self._find_most_used_tools(usage_history),
            "common_sequences": self._find_common_sequences(usage_history),
            "success_correlations": self._find_success_correlations(usage_history),
            "timing_patterns": self._analyze_timing_patterns(usage_history)
        }
        
        return patterns
    
    def _identify_success_patterns(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns that lead to success."""
        success_patterns = []
        
        successful_runs = [run for run in usage_history if run.get('success', False)]
        
        if successful_runs:
            # Analyze common tools in successful runs
            tool_frequency = {}
            for run in successful_runs:
                for tool in run.get('tools_used', []):
                    tool_frequency[tool] = tool_frequency.get(tool, 0) + 1
            
            most_successful_tools = sorted(tool_frequency.items(), key=lambda x: x[1], reverse=True)[:3]
            for tool, frequency in most_successful_tools:
                success_patterns.append(f"Tool {tool} appears in {frequency}/{len(successful_runs)} successful runs")
        
        return success_patterns
    
    def _identify_failure_patterns(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """Identify patterns that lead to failure."""
        failure_patterns = []
        
        failed_runs = [run for run in usage_history if not run.get('success', True)]
        
        if failed_runs:
            # Analyze common failure causes
            failure_causes = {}
            for run in failed_runs:
                cause = run.get('failure_cause', 'unknown')
                failure_causes[cause] = failure_causes.get(cause, 0) + 1
            
            most_common_failures = sorted(failure_causes.items(), key=lambda x: x[1], reverse=True)[:3]
            for cause, frequency in most_common_failures:
                failure_patterns.append(f"Failure cause '{cause}' appears in {frequency}/{len(failed_runs)} failed runs")
        
        return failure_patterns
    
    def _develop_adaptation_strategies(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """Develop strategies for adapting tool usage."""
        strategies = []
        
        # Analyze performance trends
        if len(usage_history) > 10:
            recent_runs = usage_history[-10:]
            older_runs = usage_history[-20:-10] if len(usage_history) > 20 else []
            
            if older_runs:
                recent_success_rate = sum(1 for run in recent_runs if run.get('success', False)) / len(recent_runs)
                older_success_rate = sum(1 for run in older_runs if run.get('success', False)) / len(older_runs)
                
                if recent_success_rate < older_success_rate:
                    strategies.append("Performance declining - review recent tool changes")
                elif recent_success_rate > older_success_rate:
                    strategies.append("Performance improving - continue current approach")
        
        # Analyze tool diversity
        all_tools_used = set()
        for run in usage_history:
            all_tools_used.update(run.get('tools_used', []))
        
        if len(all_tools_used) < len(self.tool_capabilities) * 0.5:
            strategies.append("Increase tool diversity to explore better options")
        
        return strategies
    
    # Additional helper methods
    
    def _find_most_used_tools(self, usage_history: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Find the most frequently used tools."""
        tool_usage = {}
        
        for run in usage_history:
            for tool in run.get('tools_used', []):
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
        
        return sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _find_common_sequences(self, usage_history: List[Dict[str, Any]]) -> List[str]:
        """Find common tool usage sequences."""
        sequences = {}
        
        for run in usage_history:
            tools = run.get('tools_used', [])
            if len(tools) > 1:
                for i in range(len(tools) - 1):
                    sequence = f"{tools[i]} -> {tools[i+1]}"
                    sequences[sequence] = sequences.get(sequence, 0) + 1
        
        common_sequences = sorted(sequences.items(), key=lambda x: x[1], reverse=True)[:3]
        return [seq for seq, count in common_sequences if count > 1]
    
    def _find_success_correlations(self, usage_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Find correlations between tool usage and success."""
        correlations = {}
        
        for tool_name in self.tool_capabilities.keys():
            tool_successes = 0
            tool_uses = 0
            
            for run in usage_history:
                if tool_name in run.get('tools_used', []):
                    tool_uses += 1
                    if run.get('success', False):
                        tool_successes += 1
            
            if tool_uses > 0:
                correlations[tool_name] = tool_successes / tool_uses
        
        return correlations
    
    def _analyze_timing_patterns(self, usage_history: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze timing patterns in tool usage."""
        timing_patterns = {}
        
        for tool_name in self.tool_capabilities.keys():
            execution_times = []
            
            for run in usage_history:
                if tool_name in run.get('tool_timings', {}):
                    execution_times.append(run['tool_timings'][tool_name])
            
            if execution_times:
                timing_patterns[tool_name] = {
                    'avg_time': sum(execution_times) / len(execution_times),
                    'min_time': min(execution_times),
                    'max_time': max(execution_times)
                }
        
        return timing_patterns
    
    # Performance monitoring helper methods
    
    def _get_avg_response_time(self, tool_name: str) -> float:
        """Get average response time for a tool."""
        if tool_name in self.tool_performance_history:
            times = [entry.get('response_time', 100) for entry in self.tool_performance_history[tool_name]]
            return sum(times) / len(times) if times else 100.0
        return 100.0  # Default 100ms
    
    def _get_success_rate(self, tool_name: str) -> float:
        """Get success rate for a tool."""
        if tool_name in self.tool_performance_history:
            successes = [entry.get('success', True) for entry in self.tool_performance_history[tool_name]]
            return sum(successes) / len(successes) if successes else 0.95
        return 0.95  # Default 95%
    
    def _get_throughput(self, tool_name: str) -> float:
        """Get throughput for a tool."""
        if tool_name in self.tool_performance_history:
            throughputs = [entry.get('throughput', 10) for entry in self.tool_performance_history[tool_name]]
            return sum(throughputs) / len(throughputs) if throughputs else 10.0
        return 10.0  # Default 10 ops/sec
    
    def _get_resource_usage(self, tool_name: str) -> float:
        """Get resource usage for a tool."""
        if tool_name in self.tool_performance_history:
            usages = [entry.get('resource_usage', 0.1) for entry in self.tool_performance_history[tool_name]]
            return sum(usages) / len(usages) if usages else 0.1
        return 0.1  # Default 10% resource usage 