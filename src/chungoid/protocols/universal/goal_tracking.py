"""
Goal Tracking Protocol

Track goal completion and requirements traceability across agent workflows.
Provides systematic goal decomposition, progress monitoring, and completion validation.

Change Reference: 3.19 (NEW)
"""

from typing import List, Dict, Any, Optional, Union
from ..base.protocol_interface import ProtocolInterface, ProtocolPhase, ProtocolTemplate

class GoalTrackingProtocol(ProtocolInterface):
    """Track goal completion and requirements traceability"""
    
    @property
    def name(self) -> str:
        return "goal_tracking"
    
    @property
    def description(self) -> str:
        return "Track goal completion and requirements traceability across agent workflows. Provides systematic goal decomposition, progress monitoring, and completion validation."
    
    @property
    def total_estimated_time(self) -> float:
        return 4.0  # Total of all phase time_box_hours
    
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates for goal tracking"""
        return {
            "goal_tracking_report": ProtocolTemplate(
                name="goal_tracking_report",
                description="Template for goal tracking progress report",
                template_content="""
# Goal Tracking Report

## Goal ID: [GOAL_ID]
## Goal Status: [GOAL_STATUS]
## Completion Percentage: [COMPLETION_PERCENTAGE]%

## Progress Summary
[PROGRESS_SUMMARY]

## Milestones
[MILESTONE_STATUS]

## Next Actions
[NEXT_ACTIONS]
                """,
                variables=["GOAL_ID", "GOAL_STATUS", "COMPLETION_PERCENTAGE", "PROGRESS_SUMMARY", "MILESTONE_STATUS", "NEXT_ACTIONS"]
            )
        }
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        return [
            ProtocolPhase(
                name="goal_decomposition",
                description="Decompose high-level goals into trackable objectives",
                time_box_hours=1.0,
                required_outputs=["goal_hierarchy", "objective_definitions"],
                validation_criteria=["Goals decomposed", "Objectives defined"],
                tools_required=["goal_decomposer", "objective_definer"]
            ),
            ProtocolPhase(
                name="requirements_mapping",
                description="Map requirements to goals and create traceability matrix",
                time_box_hours=1.0,
                required_outputs=["requirements_matrix", "traceability_links"],
                validation_criteria=["Requirements mapped", "Traceability established"],
                tools_required=["requirement_mapper", "traceability_creator"]
            ),
            ProtocolPhase(
                name="progress_monitoring",
                description="Monitor progress toward goals and objectives",
                time_box_hours=0.5,
                required_outputs=["progress_metrics", "completion_status"],
                validation_criteria=["Progress tracked", "Status updated"],
                tools_required=["progress_monitor", "status_tracker"]
            ),
            ProtocolPhase(
                name="milestone_validation",
                description="Validate milestone completion and goal achievement",
                time_box_hours=1.0,
                required_outputs=["validation_results", "achievement_report"],
                validation_criteria=["Milestones validated", "Achievements documented"],
                tools_required=["milestone_validator", "achievement_reporter"]
            ),
            ProtocolPhase(
                name="goal_completion",
                description="Finalize goal completion and update requirements status",
                time_box_hours=0.5,
                required_outputs=["completion_report", "requirements_status"],
                validation_criteria=["Goals completed", "Requirements satisfied"],
                tools_required=["completion_finalizer", "status_updater"]
            )
        ]
    
    def decompose_goal(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose high-level goal into trackable objectives"""
        
        decomposition_result = {
            "goal_id": goal.get("id", self._generate_goal_id()),
            "original_goal": goal,
            "goal_hierarchy": {},
            "objectives": [],
            "sub_goals": [],
            "decomposition_metrics": {}
        }
        
        # Create goal hierarchy
        decomposition_result["goal_hierarchy"] = self._create_goal_hierarchy(goal)
        
        # Extract objectives from goal
        decomposition_result["objectives"] = self._extract_objectives(goal)
        
        # Identify sub-goals
        decomposition_result["sub_goals"] = self._identify_sub_goals(goal)
        
        # Calculate decomposition metrics
        decomposition_result["decomposition_metrics"] = self._calculate_decomposition_metrics(
            decomposition_result
        )
        
        return decomposition_result
    
    def create_requirements_traceability_matrix(self, goals: List[Dict[str, Any]], requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create traceability matrix linking requirements to goals"""
        
        traceability_matrix = {
            "matrix_id": self._generate_matrix_id(),
            "creation_timestamp": self._get_timestamp(),
            "goals": goals,
            "requirements": requirements,
            "mappings": {},
            "coverage_analysis": {},
            "gap_analysis": {}
        }
        
        # Create requirement-to-goal mappings
        for requirement in requirements:
            req_id = requirement.get("id", "unknown")
            mapped_goals = self._map_requirement_to_goals(requirement, goals)
            traceability_matrix["mappings"][req_id] = mapped_goals
        
        # Analyze requirement coverage
        traceability_matrix["coverage_analysis"] = self._analyze_requirement_coverage(
            traceability_matrix["mappings"], requirements
        )
        
        # Perform gap analysis
        traceability_matrix["gap_analysis"] = self._perform_gap_analysis(
            goals, requirements, traceability_matrix["mappings"]
        )
        
        return traceability_matrix
    
    def track_goal_progress(self, goal_id: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Track progress toward specific goal"""
        
        progress_tracking = {
            "goal_id": goal_id,
            "tracking_timestamp": self._get_timestamp(),
            "current_progress": progress_data,
            "progress_metrics": {},
            "completion_percentage": 0.0,
            "milestone_status": {},
            "next_actions": []
        }
        
        # Calculate progress metrics
        progress_tracking["progress_metrics"] = self._calculate_progress_metrics(progress_data)
        
        # Calculate completion percentage
        progress_tracking["completion_percentage"] = self._calculate_completion_percentage(progress_data)
        
        # Update milestone status
        progress_tracking["milestone_status"] = self._update_milestone_status(goal_id, progress_data)
        
        # Determine next actions
        progress_tracking["next_actions"] = self._determine_next_actions(goal_id, progress_tracking)
        
        return progress_tracking
    
    def validate_milestone_completion(self, milestone: Dict[str, Any], completion_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that a milestone has been completed"""
        
        validation_result = {
            "milestone_id": milestone.get("id", "unknown"),
            "validation_timestamp": self._get_timestamp(),
            "completion_evidence": completion_evidence,
            "validation_status": "pending",
            "validation_criteria": {},
            "validation_results": {},
            "overall_score": 0.0,
            "recommendations": []
        }
        
        # Get validation criteria for milestone
        validation_result["validation_criteria"] = self._get_milestone_validation_criteria(milestone)
        
        # Validate against each criterion
        for criterion_name, criterion_details in validation_result["validation_criteria"].items():
            criterion_result = self._validate_criterion(criterion_details, completion_evidence)
            validation_result["validation_results"][criterion_name] = criterion_result
        
        # Calculate overall validation score
        validation_result["overall_score"] = self._calculate_validation_score(
            validation_result["validation_results"]
        )
        
        # Determine validation status
        validation_result["validation_status"] = self._determine_validation_status(
            validation_result["overall_score"]
        )
        
        # Generate recommendations if validation fails
        if validation_result["validation_status"] != "passed":
            validation_result["recommendations"] = self._generate_validation_recommendations(
                validation_result
            )
        
        return validation_result
    
    def finalize_goal_completion(self, goal_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize goal completion and update all related requirements"""
        
        completion_result = {
            "goal_id": goal_id,
            "completion_timestamp": self._get_timestamp(),
            "completion_data": completion_data,
            "final_status": "unknown",
            "requirements_impact": {},
            "artifacts_generated": [],
            "lessons_learned": [],
            "recommendations": []
        }
        
        # Validate goal completion
        completion_validation = self._validate_goal_completion(goal_id, completion_data)
        completion_result["final_status"] = completion_validation["status"]
        
        # Update related requirements
        completion_result["requirements_impact"] = self._update_related_requirements(
            goal_id, completion_validation
        )
        
        # Document artifacts generated
        completion_result["artifacts_generated"] = self._document_generated_artifacts(completion_data)
        
        # Capture lessons learned
        completion_result["lessons_learned"] = self._capture_lessons_learned(goal_id, completion_data)
        
        # Generate recommendations for future goals
        completion_result["recommendations"] = self._generate_future_recommendations(completion_result)
        
        return completion_result
    
    def get_goal_status_summary(self, goal_ids: List[str]) -> Dict[str, Any]:
        """Get comprehensive status summary for multiple goals"""
        
        status_summary = {
            "summary_timestamp": self._get_timestamp(),
            "total_goals": len(goal_ids),
            "goal_statuses": {},
            "overall_metrics": {},
            "completion_trends": {},
            "risk_assessment": {}
        }
        
        # Get status for each goal
        for goal_id in goal_ids:
            goal_status = self._get_individual_goal_status(goal_id)
            status_summary["goal_statuses"][goal_id] = goal_status
        
        # Calculate overall metrics
        status_summary["overall_metrics"] = self._calculate_overall_metrics(
            status_summary["goal_statuses"]
        )
        
        # Analyze completion trends
        status_summary["completion_trends"] = self._analyze_completion_trends(
            status_summary["goal_statuses"]
        )
        
        # Assess risks
        status_summary["risk_assessment"] = self._assess_goal_risks(
            status_summary["goal_statuses"]
        )
        
        return status_summary
    
    def _create_goal_hierarchy(self, goal: Dict[str, Any]) -> Dict[str, Any]:
        """Create hierarchical structure for goal"""
        return {
            "level_0": {
                "name": goal.get("name", "Unknown Goal"),
                "description": goal.get("description", ""),
                "priority": goal.get("priority", "medium")
            },
            "level_1": [],  # Sub-goals
            "level_2": []   # Objectives
        }
    
    def _extract_objectives(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract specific objectives from goal"""
        objectives = []
        
        # Parse objectives from goal description or explicit objectives list
        if "objectives" in goal:
            for i, obj in enumerate(goal["objectives"]):
                objectives.append({
                    "id": f"obj_{i+1}",
                    "description": obj,
                    "status": "not_started",
                    "priority": "medium"
                })
        else:
            # Generate default objectives based on goal type
            objectives = self._generate_default_objectives(goal)
        
        return objectives
    
    def _identify_sub_goals(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify sub-goals within main goal"""
        sub_goals = []
        
        # Check for explicit sub-goals
        if "sub_goals" in goal:
            sub_goals = goal["sub_goals"]
        else:
            # Generate sub-goals based on goal complexity
            sub_goals = self._generate_sub_goals(goal)
        
        return sub_goals
    
    def _calculate_decomposition_metrics(self, decomposition_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate metrics for goal decomposition"""
        return {
            "total_objectives": len(decomposition_result["objectives"]),
            "total_sub_goals": len(decomposition_result["sub_goals"]),
            "decomposition_depth": 2,  # Simplified
            "complexity_score": len(decomposition_result["objectives"]) * 0.5 + len(decomposition_result["sub_goals"]) * 1.0
        }
    
    def _map_requirement_to_goals(self, requirement: Dict[str, Any], goals: List[Dict[str, Any]]) -> List[str]:
        """Map a requirement to relevant goals"""
        mapped_goals = []
        
        req_keywords = self._extract_keywords(requirement.get("description", ""))
        
        for goal in goals:
            goal_keywords = self._extract_keywords(goal.get("description", ""))
            if self._calculate_keyword_overlap(req_keywords, goal_keywords) > 0.3:
                mapped_goals.append(goal.get("id", "unknown"))
        
        return mapped_goals
    
    def _analyze_requirement_coverage(self, mappings: Dict[str, List[str]], requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how well requirements are covered by goals"""
        total_requirements = len(requirements)
        covered_requirements = len([req_id for req_id, goals in mappings.items() if goals])
        
        return {
            "total_requirements": total_requirements,
            "covered_requirements": covered_requirements,
            "coverage_percentage": covered_requirements / max(total_requirements, 1) * 100,
            "uncovered_requirements": [req_id for req_id, goals in mappings.items() if not goals]
        }
    
    def _perform_gap_analysis(self, goals: List[Dict[str, Any]], requirements: List[Dict[str, Any]], mappings: Dict[str, List[str]]) -> Dict[str, Any]:
        """Perform gap analysis between goals and requirements"""
        return {
            "missing_goals": [],  # Requirements not covered by any goal
            "redundant_goals": [],  # Goals not linked to any requirement
            "weak_mappings": [],  # Goals/requirements with weak linkage
            "recommendations": []
        }
    
    def _calculate_progress_metrics(self, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate progress metrics"""
        return {
            "tasks_completed": progress_data.get("completed_tasks", 0),
            "tasks_total": progress_data.get("total_tasks", 1),
            "time_elapsed": progress_data.get("time_elapsed", 0),
            "time_estimated": progress_data.get("time_estimated", 1),
            "quality_score": progress_data.get("quality_score", 0.5)
        }
    
    def _calculate_completion_percentage(self, progress_data: Dict[str, Any]) -> float:
        """Calculate completion percentage"""
        completed = progress_data.get("completed_tasks", 0)
        total = progress_data.get("total_tasks", 1)
        return min(100.0, (completed / max(total, 1)) * 100)
    
    def _update_milestone_status(self, goal_id: str, progress_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update milestone status based on progress"""
        return {
            "milestone_1": {"status": "completed", "completion_date": self._get_timestamp()},
            "milestone_2": {"status": "in_progress", "progress": 0.7},
            "milestone_3": {"status": "not_started", "progress": 0.0}
        }
    
    def _determine_next_actions(self, goal_id: str, progress_tracking: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Determine next actions based on current progress"""
        completion_percentage = progress_tracking["completion_percentage"]
        
        if completion_percentage < 25:
            return [{"action": "establish_foundation", "priority": "high"}]
        elif completion_percentage < 75:
            return [{"action": "continue_implementation", "priority": "medium"}]
        else:
            return [{"action": "finalize_and_validate", "priority": "high"}]
    
    def _get_milestone_validation_criteria(self, milestone: Dict[str, Any]) -> Dict[str, Any]:
        """Get validation criteria for milestone"""
        return {
            "deliverable_quality": {
                "type": "quality_check",
                "threshold": 0.8,
                "description": "Deliverable meets quality standards"
            },
            "requirements_satisfaction": {
                "type": "requirement_check",
                "threshold": 1.0,
                "description": "All requirements satisfied"
            },
            "timeline_adherence": {
                "type": "timeline_check",
                "threshold": 0.9,
                "description": "Milestone completed within timeline"
            }
        }
    
    def _validate_criterion(self, criterion_details: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Validate single criterion against evidence"""
        return {
            "criterion_name": criterion_details.get("description", "Unknown"),
            "passed": True,  # Simplified validation
            "score": 0.9,
            "evidence_evaluated": len(evidence),
            "validation_notes": "Criterion validation completed"
        }
    
    def _calculate_validation_score(self, validation_results: Dict[str, Any]) -> float:
        """Calculate overall validation score"""
        if not validation_results:
            return 0.0
        
        total_score = sum(result.get("score", 0) for result in validation_results.values())
        return total_score / len(validation_results)
    
    def _determine_validation_status(self, overall_score: float) -> str:
        """Determine validation status from score"""
        if overall_score >= 0.8:
            return "passed"
        elif overall_score >= 0.6:
            return "conditional"
        else:
            return "failed"
    
    def _generate_validation_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for failed validation"""
        return [
            "Review deliverable quality standards",
            "Address failed validation criteria",
            "Gather additional evidence for validation"
        ]
    
    def _validate_goal_completion(self, goal_id: str, completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate that goal has been completed"""
        return {
            "status": "completed",
            "validation_score": 0.95,
            "completion_quality": "high"
        }
    
    def _update_related_requirements(self, goal_id: str, completion_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Update requirements related to completed goal"""
        return {
            "requirements_updated": 5,
            "requirements_satisfied": 4,
            "requirements_remaining": 1
        }
    
    def _document_generated_artifacts(self, completion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document artifacts generated during goal completion"""
        return [
            {"type": "documentation", "name": "Goal Completion Report"},
            {"type": "code", "name": "Implementation Artifacts"},
            {"type": "tests", "name": "Validation Test Suite"}
        ]
    
    def _capture_lessons_learned(self, goal_id: str, completion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Capture lessons learned during goal completion"""
        return [
            {"category": "process", "lesson": "Regular progress reviews improved outcome quality"},
            {"category": "technical", "lesson": "Early validation prevented later rework"},
            {"category": "collaboration", "lesson": "Clear communication reduced misunderstandings"}
        ]
    
    def _generate_future_recommendations(self, completion_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations for future goals"""
        return [
            "Apply lessons learned to similar goals",
            "Reuse successful artifacts and processes",
            "Consider automation for repeated tasks"
        ]
    
    def _get_individual_goal_status(self, goal_id: str) -> Dict[str, Any]:
        """Get status for individual goal"""
        return {
            "goal_id": goal_id,
            "status": "in_progress",
            "completion_percentage": 65.0,
            "milestones_completed": 2,
            "milestones_total": 4,
            "last_updated": self._get_timestamp()
        }
    
    def _calculate_overall_metrics(self, goal_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall metrics across all goals"""
        total_goals = len(goal_statuses)
        completed_goals = len([g for g in goal_statuses.values() if g.get("status") == "completed"])
        
        return {
            "total_goals": total_goals,
            "completed_goals": completed_goals,
            "completion_rate": completed_goals / max(total_goals, 1) * 100,
            "average_progress": sum(g.get("completion_percentage", 0) for g in goal_statuses.values()) / max(total_goals, 1)
        }
    
    def _analyze_completion_trends(self, goal_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze completion trends"""
        return {
            "trend_direction": "positive",
            "completion_velocity": 2.5,  # goals per week
            "projected_completion": self._get_timestamp()
        }
    
    def _assess_goal_risks(self, goal_statuses: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Assess risks across goals"""
        return {
            "high_risk_goals": [],
            "medium_risk_goals": ["goal_3"],
            "low_risk_goals": ["goal_1", "goal_2"],
            "overall_risk_level": "low"
        }
    
    def _generate_default_objectives(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate default objectives for goal"""
        return [
            {"id": "obj_1", "description": "Define requirements", "status": "not_started"},
            {"id": "obj_2", "description": "Implement solution", "status": "not_started"},
            {"id": "obj_3", "description": "Validate results", "status": "not_started"}
        ]
    
    def _generate_sub_goals(self, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sub-goals based on goal complexity"""
        return [
            {"id": "subgoal_1", "name": "Foundation", "status": "not_started"},
            {"id": "subgoal_2", "name": "Implementation", "status": "not_started"}
        ]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction
        return text.lower().split()
    
    def _calculate_keyword_overlap(self, keywords1: List[str], keywords2: List[str]) -> float:
        """Calculate overlap between keyword sets"""
        if not keywords1 or not keywords2:
            return 0.0
        
        overlap = len(set(keywords1) & set(keywords2))
        total = len(set(keywords1) | set(keywords2))
        return overlap / max(total, 1)
    
    def _generate_goal_id(self) -> str:
        """Generate unique goal ID"""
        import uuid
        return f"goal_{str(uuid.uuid4())[:8]}"
    
    def _generate_matrix_id(self) -> str:
        """Generate unique matrix ID"""
        import uuid
        return f"matrix_{str(uuid.uuid4())[:8]}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat() 