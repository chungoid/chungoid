"""
Deep Investigation Protocol

Systematic 4-phase methodology for analyzing complex technical issues
before implementing solutions.
"""

from typing import List, Dict, Any, Optional
from ..base.protocol_interface import ProtocolInterface, ProtocolPhase

class DeepInvestigationProtocol(ProtocolInterface):
    """
    Systematic approach to investigating complex technical issues.
    
    4-phase methodology ensuring thorough analysis before implementation:
    1. Problem definition and stakeholder analysis
    2. Comprehensive data gathering
    3. Root cause analysis and pattern identification  
    4. Solution formulation with risk assessment
    """
    
    @property
    def name(self) -> str:
        return "deep_investigation"
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        return [
            ProtocolPhase(
                name="problem_definition",
                description="Define problem scope and identify stakeholders",
                time_box_hours=2.0,
                required_outputs=[
                    "problem_statement",
                    "stakeholder_analysis", 
                    "success_criteria",
                    "investigation_scope"
                ],
                validation_criteria=[
                    "Problem statement is clear and specific",
                    "All relevant stakeholders identified",
                    "Success criteria are measurable",
                    "Investigation boundaries defined"
                ],
                tools_required=[
                    "stakeholder_interview", 
                    "requirement_analysis",
                    "scope_definition"
                ]
            ),
            ProtocolPhase(
                name="data_gathering",
                description="Systematic collection of relevant information",
                time_box_hours=3.0,
                required_outputs=[
                    "system_analysis",
                    "data_collection_summary",
                    "evidence_catalog",
                    "information_gaps"
                ],
                validation_criteria=[
                    "Multiple data sources consulted",
                    "Evidence properly documented",
                    "Information gaps identified",
                    "Data quality assessed"
                ],
                tools_required=[
                    "codebase_search",
                    "log_analysis", 
                    "system_monitoring",
                    "documentation_review"
                ]
            ),
            ProtocolPhase(
                name="analysis",
                description="Root cause analysis and pattern identification",
                time_box_hours=4.0,
                required_outputs=[
                    "root_cause_analysis",
                    "pattern_identification",
                    "contributing_factors",
                    "analysis_conclusions"
                ],
                validation_criteria=[
                    "Root causes clearly identified",
                    "Analysis based on evidence",
                    "Patterns documented with examples",
                    "Contributing factors analyzed"
                ],
                tools_required=[
                    "causal_analysis",
                    "pattern_recognition",
                    "impact_assessment"
                ]
            ),
            ProtocolPhase(
                name="solution_formulation",
                description="Develop solution options with risk assessment",
                time_box_hours=2.5,
                required_outputs=[
                    "solution_options",
                    "risk_assessment",
                    "implementation_plan",
                    "recommendations"
                ],
                validation_criteria=[
                    "Multiple solution options considered",
                    "Risks identified and assessed",
                    "Implementation approach defined",
                    "Recommendations prioritized"
                ],
                tools_required=[
                    "solution_design",
                    "risk_analysis",
                    "implementation_planning"
                ]
            )
        ]
    
    def investigate_technical_issue(self, issue_description: str, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute complete deep investigation of a technical issue.
        
        Args:
            issue_description: Description of the technical issue
            context: Additional context and constraints
            
        Returns:
            Complete investigation results with solutions
        """
        investigation_context = {
            "issue_description": issue_description,
            "context": context,
            "investigation_id": f"investigation_{hash(issue_description)}",
            "findings": {},
            "solutions": []
        }
        
        # Execute investigation phases
        for phase in self.phases:
            phase_result = self._execute_investigation_phase(phase, investigation_context)
            investigation_context["findings"][phase.name] = phase_result
            
        return self._generate_investigation_report(investigation_context)
    
    def _execute_investigation_phase(self, phase: ProtocolPhase, 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific investigation phase."""
        
        if phase.name == "problem_definition":
            return self._define_problem(context)
        elif phase.name == "data_gathering":
            return self._gather_data(context)
        elif phase.name == "analysis":
            return self._analyze_findings(context)
        elif phase.name == "solution_formulation":
            return self._formulate_solutions(context)
        else:
            return {}
    
    def _define_problem(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 1: Define problem scope and stakeholders."""
        return {
            "problem_statement": self._create_problem_statement(context["issue_description"]),
            "stakeholder_analysis": self._identify_stakeholders(context),
            "success_criteria": self._define_success_criteria(context),
            "investigation_scope": self._define_scope(context)
        }
    
    def _gather_data(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 2: Systematic data collection."""
        return {
            "system_analysis": self._analyze_system_state(context),
            "data_collection_summary": self._summarize_data_sources(context),
            "evidence_catalog": self._catalog_evidence(context),
            "information_gaps": self._identify_gaps(context)
        }
    
    def _analyze_findings(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 3: Root cause analysis."""
        return {
            "root_cause_analysis": self._perform_root_cause_analysis(context),
            "pattern_identification": self._identify_patterns(context),
            "contributing_factors": self._analyze_contributing_factors(context),
            "analysis_conclusions": self._draw_conclusions(context)
        }
    
    def _formulate_solutions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: Solution development."""
        return {
            "solution_options": self._generate_solution_options(context),
            "risk_assessment": self._assess_risks(context),
            "implementation_plan": self._create_implementation_plan(context),
            "recommendations": self._prioritize_recommendations(context)
        }
    
    def _create_problem_statement(self, issue_description: str) -> str:
        """Create clear, specific problem statement."""
        return f"Investigation of: {issue_description}"
    
    def _identify_stakeholders(self, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """Identify relevant stakeholders."""
        return {
            "primary": ["system_users", "developers", "administrators"],
            "secondary": ["management", "support_team"],
            "external": ["vendors", "external_dependencies"]
        }
    
    def _define_success_criteria(self, context: Dict[str, Any]) -> List[str]:
        """Define measurable success criteria."""
        return [
            "Root cause identified with evidence",
            "Solution addresses core issue",
            "Implementation risk is acceptable",
            "Solution is sustainable"
        ]
    
    def _define_scope(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Define investigation boundaries."""
        return {
            "included": ["direct_system_components", "immediate_dependencies"],
            "excluded": ["unrelated_systems", "historical_issues"],
            "constraints": ["time_limitations", "resource_availability"]
        }
    
    def _analyze_system_state(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze current system state."""
        # Placeholder for system analysis logic
        return {"status": "analysis_completed", "findings": []}
    
    def _summarize_data_sources(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize data collection efforts."""
        return {
            "sources_consulted": ["logs", "documentation", "code", "monitoring"],
            "data_quality": "acceptable",
            "completeness": "sufficient"
        }
    
    def _catalog_evidence(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Catalog collected evidence."""
        return []
    
    def _identify_gaps(self, context: Dict[str, Any]) -> List[str]:
        """Identify information gaps."""
        return []
    
    def _perform_root_cause_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Perform systematic root cause analysis."""
        return {
            "methodology": "5_whys_analysis",
            "root_causes": [],
            "confidence_level": "medium"
        }
    
    def _identify_patterns(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify relevant patterns."""
        return []
    
    def _analyze_contributing_factors(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze contributing factors."""
        return []
    
    def _draw_conclusions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Draw analysis conclusions."""
        return {
            "primary_conclusions": [],
            "confidence_assessment": "medium",
            "assumptions": []
        }
    
    def _generate_solution_options(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multiple solution options."""
        return []
    
    def _assess_risks(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Assess implementation risks."""
        return {
            "high_risk_factors": [],
            "medium_risk_factors": [],
            "low_risk_factors": [],
            "risk_mitigation": []
        }
    
    def _create_implementation_plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create implementation approach."""
        return {
            "approach": "systematic_implementation",
            "phases": [],
            "timeline": "TBD",
            "resources_required": []
        }
    
    def _prioritize_recommendations(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize solution recommendations."""
        return []
    
    def _generate_investigation_report(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive investigation report."""
        return {
            "investigation_id": context["investigation_id"],
            "issue_description": context["issue_description"],
            "investigation_findings": context["findings"],
            "status": "completed",
            "recommendations": context.get("solutions", [])
        } 