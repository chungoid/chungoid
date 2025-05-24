"""
Adaptive Learning System

Provides intelligent strategy learning, pattern recognition, and autonomous
optimization capabilities for the Chungoid autonomous coding system. 

Enables the system to:
- Analyze historical execution data to identify successful strategies
- Automatically update agent behaviors based on empirical success
- Perform A/B testing of strategy variants
- Apply cross-project learning insights

Features:
- Pattern Recognition Engine for identifying successful strategies
- Strategy Evolution Framework for autonomous behavior updates
- Performance Analytics for measuring strategy effectiveness
- Cross-Project Learning for knowledge transfer
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics
import numpy as np
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class StrategyType(str, Enum):
    """Types of strategies that can be learned and optimized."""
    DEPENDENCY_RESOLUTION = "dependency_resolution"
    ERROR_RECOVERY = "error_recovery"
    TOOL_SELECTION = "tool_selection"
    AGENT_COORDINATION = "agent_coordination"
    ENVIRONMENT_SETUP = "environment_setup"
    TESTING_APPROACH = "testing_approach"
    CODE_GENERATION = "code_generation"
    PROJECT_ANALYSIS = "project_analysis"


class LearningOutcome(str, Enum):
    """Possible outcomes of learning experiments."""
    SUCCESS = "success"
    FAILURE = "failure" 
    PARTIAL_SUCCESS = "partial_success"
    INCONCLUSIVE = "inconclusive"


@dataclass
class ExecutionPattern:
    """Represents a pattern identified from execution data."""
    pattern_id: str
    pattern_type: StrategyType
    description: str
    context_features: Dict[str, Any]
    actions_taken: List[str]
    success_indicators: List[str]
    failure_indicators: List[str]
    confidence_score: float
    sample_size: int
    first_observed: str
    last_observed: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['pattern_type'] = self.pattern_type.value
        return data


@dataclass
class StrategyVariant:
    """Represents a variant of a strategy to be tested."""
    variant_id: str
    strategy_type: StrategyType
    description: str
    parameters: Dict[str, Any]
    implementation_details: Dict[str, Any]
    expected_improvement: float
    confidence_level: float
    test_conditions: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['strategy_type'] = self.strategy_type.value
        return data


@dataclass
class LearningExperiment:
    """Represents an A/B testing experiment for strategy variants."""
    experiment_id: str
    strategy_type: StrategyType
    baseline_variant: StrategyVariant
    test_variant: StrategyVariant
    start_time: str
    end_time: Optional[str]
    target_metric: str
    success_criteria: Dict[str, Any]
    results: Optional[Dict[str, Any]]
    outcome: Optional[LearningOutcome]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['strategy_type'] = self.strategy_type.value
        if data['outcome']:
            data['outcome'] = self.outcome.value
        return data


@dataclass
class CrossProjectInsight:
    """Represents insights that can be applied across projects."""
    insight_id: str
    source_projects: List[str]
    target_domains: List[str]
    strategy_type: StrategyType
    insight_description: str
    applicability_conditions: List[str]
    success_probability: float
    confidence_score: float
    generated_at: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['strategy_type'] = self.strategy_type.value
        return data


class PatternRecognitionEngine:
    """Analyzes execution data to identify patterns and strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.min_sample_size = 3
        self.min_confidence_threshold = 0.6
        
    async def analyze_execution_patterns(
        self,
        execution_data: List[Dict[str, Any]],
        time_window_days: int = 30
    ) -> List[ExecutionPattern]:
        """Analyze execution data to identify successful patterns."""
        self.logger.info(f"Analyzing execution patterns from {len(execution_data)} executions")
        
        # Filter data by time window
        cutoff_date = datetime.now() - timedelta(days=time_window_days)
        recent_data = [
            ex for ex in execution_data 
            if datetime.fromisoformat(ex.get('timestamp', '1970-01-01')) > cutoff_date
        ]
        
        patterns = []
        
        # Group executions by strategy type and context
        grouped_data = self._group_executions_by_context(recent_data)
        
        for context_key, executions in grouped_data.items():
            if len(executions) >= self.min_sample_size:
                pattern = await self._extract_pattern_from_group(context_key, executions)
                if pattern and pattern.confidence_score >= self.min_confidence_threshold:
                    patterns.append(pattern)
        
        self.logger.info(f"Identified {len(patterns)} significant patterns")
        return patterns
    
    def _group_executions_by_context(
        self, 
        execution_data: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group executions by similar context features."""
        grouped = defaultdict(list)
        
        for execution in execution_data:
            # Create context key based on relevant features
            context_features = [
                execution.get('project_type', 'unknown'),
                execution.get('strategy_type', 'unknown'),
                execution.get('agent_type', 'unknown'),
                execution.get('complexity_level', 'unknown')
            ]
            context_key = "|".join(str(f) for f in context_features)
            grouped[context_key].append(execution)
        
        return dict(grouped)
    
    async def _extract_pattern_from_group(
        self,
        context_key: str,
        executions: List[Dict[str, Any]]
    ) -> Optional[ExecutionPattern]:
        """Extract a pattern from a group of similar executions."""
        if not executions:
            return None
        
        # Calculate success rate
        successful = [ex for ex in executions if ex.get('success', False)]
        success_rate = len(successful) / len(executions)
        
        if success_rate < 0.7:  # Only identify patterns with high success rate
            return None
        
        # Extract common actions and indicators
        common_actions = self._find_common_elements([
            ex.get('actions_taken', []) for ex in successful
        ])
        
        success_indicators = self._find_common_elements([
            ex.get('success_indicators', []) for ex in successful
        ])
        
        # Build context features
        context_features = {}
        for key in ['project_type', 'strategy_type', 'complexity_level']:
            values = [ex.get(key) for ex in executions if ex.get(key)]
            if values:
                context_features[key] = Counter(values).most_common(1)[0][0]
        
        pattern_id = f"pattern_{hash(context_key)}_{int(datetime.now().timestamp())}"
        
        return ExecutionPattern(
            pattern_id=pattern_id,
            pattern_type=StrategyType(context_features.get('strategy_type', 'TOOL_SELECTION')),
            description=f"Successful pattern for {context_key} context",
            context_features=context_features,
            actions_taken=common_actions,
            success_indicators=success_indicators,
            failure_indicators=[],
            confidence_score=success_rate,
            sample_size=len(executions),
            first_observed=min(ex.get('timestamp', '') for ex in executions),
            last_observed=max(ex.get('timestamp', '') for ex in executions)
        )
    
    def _find_common_elements(self, lists: List[List[str]]) -> List[str]:
        """Find elements that appear in most of the lists."""
        if not lists:
            return []
        
        element_counts = Counter()
        for lst in lists:
            element_counts.update(set(lst))
        
        # Return elements that appear in at least 60% of lists
        threshold = len(lists) * 0.6
        return [element for element, count in element_counts.items() if count >= threshold]


class StrategyEvolutionFramework:
    """Manages automatic strategy updates based on learned patterns."""
    
    def __init__(self, chroma_manager=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.chroma_manager = chroma_manager
        self.evolution_history = []
        
    async def propose_strategy_updates(
        self,
        patterns: List[ExecutionPattern],
        current_strategies: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Propose strategy updates based on identified patterns."""
        self.logger.info(f"Proposing strategy updates from {len(patterns)} patterns")
        
        proposals = []
        
        for pattern in patterns:
            if pattern.confidence_score >= 0.8:  # High confidence patterns only
                proposal = await self._create_strategy_proposal(pattern, current_strategies)
                if proposal:
                    proposals.append(proposal)
        
        self.logger.info(f"Generated {len(proposals)} strategy update proposals")
        return proposals
    
    async def _create_strategy_proposal(
        self,
        pattern: ExecutionPattern,
        current_strategies: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a specific strategy update proposal from a pattern."""
        strategy_key = pattern.pattern_type.value
        current_strategy = current_strategies.get(strategy_key, {})
        
        # Analyze what changes would incorporate this pattern
        proposed_changes = {
            'strategy_type': pattern.pattern_type.value,
            'pattern_source': pattern.pattern_id,
            'confidence': pattern.confidence_score,
            'changes': {
                'preferred_actions': pattern.actions_taken,
                'success_indicators': pattern.success_indicators,
                'context_conditions': pattern.context_features
            },
            'expected_improvement': self._estimate_improvement(pattern),
            'implementation_priority': self._calculate_priority(pattern)
        }
        
        return proposed_changes
    
    def _estimate_improvement(self, pattern: ExecutionPattern) -> float:
        """Estimate potential improvement from adopting a pattern."""
        # Simple heuristic based on confidence and sample size
        base_improvement = pattern.confidence_score - 0.5  # Improvement over random
        sample_factor = min(pattern.sample_size / 10, 1.0)  # Cap at 10 samples
        return base_improvement * sample_factor
    
    def _calculate_priority(self, pattern: ExecutionPattern) -> float:
        """Calculate implementation priority for a pattern."""
        # Priority based on confidence, sample size, and recency
        confidence_weight = pattern.confidence_score * 0.4
        sample_weight = min(pattern.sample_size / 20, 0.3)
        
        # Recency weight (patterns observed recently get higher priority)
        try:
            days_since_last = (datetime.now() - datetime.fromisoformat(pattern.last_observed)).days
            recency_weight = max(0, 0.3 - (days_since_last / 30) * 0.3)
        except:
            recency_weight = 0.1
        
        return confidence_weight + sample_weight + recency_weight
    
    async def apply_strategy_update(
        self,
        proposal: Dict[str, Any],
        agent_configs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a strategy update to agent configurations."""
        strategy_type = proposal['strategy_type']
        changes = proposal['changes']
        
        self.logger.info(f"Applying strategy update for {strategy_type}")
        
        # Update configurations based on strategy type
        updated_configs = agent_configs.copy()
        
        if strategy_type in updated_configs:
            config = updated_configs[strategy_type]
            
            # Apply changes
            config.update({
                'preferred_actions': changes.get('preferred_actions', []),
                'success_criteria': changes.get('success_indicators', []),
                'context_awareness': changes.get('context_conditions', {}),
                'last_updated': datetime.now().isoformat(),
                'update_source': proposal['pattern_source'],
                'confidence': proposal['confidence']
            })
        
        # Log the evolution
        evolution_record = {
            'timestamp': datetime.now().isoformat(),
            'strategy_type': strategy_type,
            'proposal': proposal,
            'applied': True
        }
        self.evolution_history.append(evolution_record)
        
        return updated_configs


class ABTestingEngine:
    """Manages A/B testing of strategy variants."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.active_experiments = {}
        self.completed_experiments = []
        
    async def create_experiment(
        self,
        baseline_strategy: Dict[str, Any],
        test_strategy: Dict[str, Any],
        success_metric: str,
        duration_days: int = 7
    ) -> LearningExperiment:
        """Create a new A/B testing experiment."""
        experiment_id = f"experiment_{int(datetime.now().timestamp())}"
        
        baseline_variant = StrategyVariant(
            variant_id=f"{experiment_id}_baseline",
            strategy_type=StrategyType(baseline_strategy.get('type', 'TOOL_SELECTION')),
            description="Current baseline strategy",
            parameters=baseline_strategy,
            implementation_details={},
            expected_improvement=0.0,
            confidence_level=0.8,
            test_conditions=[]
        )
        
        test_variant = StrategyVariant(
            variant_id=f"{experiment_id}_test",
            strategy_type=StrategyType(test_strategy.get('type', 'TOOL_SELECTION')),
            description="Proposed optimized strategy",
            parameters=test_strategy,
            implementation_details={},
            expected_improvement=test_strategy.get('expected_improvement', 0.1),
            confidence_level=0.7,
            test_conditions=[]
        )
        
        experiment = LearningExperiment(
            experiment_id=experiment_id,
            strategy_type=baseline_variant.strategy_type,
            baseline_variant=baseline_variant,
            test_variant=test_variant,
            start_time=datetime.now().isoformat(),
            end_time=(datetime.now() + timedelta(days=duration_days)).isoformat(),
            target_metric=success_metric,
            success_criteria={'improvement_threshold': 0.05},
            results=None,
            outcome=None
        )
        
        self.active_experiments[experiment_id] = experiment
        self.logger.info(f"Created A/B test experiment: {experiment_id}")
        
        return experiment
    
    async def collect_experiment_results(
        self,
        experiment_id: str,
        execution_data: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Collect and analyze results for an A/B test experiment."""
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        
        # Filter data for this experiment
        experiment_data = [
            ex for ex in execution_data 
            if ex.get('experiment_id') == experiment_id
        ]
        
        if len(experiment_data) < 10:  # Need minimum sample size
            return None
        
        # Split data by variant
        baseline_data = [ex for ex in experiment_data if ex.get('variant') == 'baseline']
        test_data = [ex for ex in experiment_data if ex.get('variant') == 'test']
        
        if not baseline_data or not test_data:
            return None
        
        # Calculate metrics
        baseline_success_rate = sum(1 for ex in baseline_data if ex.get('success', False)) / len(baseline_data)
        test_success_rate = sum(1 for ex in test_data if ex.get('success', False)) / len(test_data)
        
        improvement = test_success_rate - baseline_success_rate
        relative_improvement = improvement / baseline_success_rate if baseline_success_rate > 0 else 0
        
        # Determine statistical significance (simplified)
        sample_sizes = [len(baseline_data), len(test_data)]
        min_sample_size = min(sample_sizes)
        confidence = min(0.95, 0.5 + (min_sample_size / 50) * 0.4)  # Rough approximation
        
        results = {
            'baseline_success_rate': baseline_success_rate,
            'test_success_rate': test_success_rate,
            'improvement': improvement,
            'relative_improvement': relative_improvement,
            'sample_sizes': sample_sizes,
            'confidence': confidence,
            'statistically_significant': abs(improvement) > 0.05 and confidence > 0.8
        }
        
        # Determine outcome
        if results['statistically_significant']:
            if improvement > experiment.success_criteria['improvement_threshold']:
                outcome = LearningOutcome.SUCCESS
            elif improvement < -experiment.success_criteria['improvement_threshold']:
                outcome = LearningOutcome.FAILURE
            else:
                outcome = LearningOutcome.INCONCLUSIVE
        else:
            outcome = LearningOutcome.INCONCLUSIVE
        
        # Update experiment
        experiment.results = results
        experiment.outcome = outcome
        experiment.end_time = datetime.now().isoformat()
        
        # Move to completed experiments
        self.completed_experiments.append(experiment)
        del self.active_experiments[experiment_id]
        
        self.logger.info(f"Completed experiment {experiment_id} with outcome: {outcome.value}")
        
        return results


class CrossProjectLearningEngine:
    """Manages learning and insight transfer across projects."""
    
    def __init__(self, chroma_manager=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.chroma_manager = chroma_manager
        self.insights_database = []
        
    async def extract_cross_project_insights(
        self,
        project_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[CrossProjectInsight]:
        """Extract insights that can be applied across multiple projects."""
        self.logger.info(f"Extracting cross-project insights from {len(project_data)} projects")
        
        insights = []
        
        # Find patterns that work across multiple projects
        common_patterns = await self._find_cross_project_patterns(project_data)
        
        for pattern_info in common_patterns:
            insight = await self._create_cross_project_insight(pattern_info)
            if insight:
                insights.append(insight)
        
        self.logger.info(f"Extracted {len(insights)} cross-project insights")
        return insights
    
    async def _find_cross_project_patterns(
        self,
        project_data: Dict[str, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Find patterns that appear successful across multiple projects."""
        pattern_tracker = defaultdict(lambda: {
            'projects': set(),
            'success_rates': [],
            'contexts': [],
            'actions': []
        })
        
        for project_id, executions in project_data.items():
            successful_executions = [ex for ex in executions if ex.get('success', False)]
            
            for execution in successful_executions:
                strategy_type = execution.get('strategy_type', 'unknown')
                actions = execution.get('actions_taken', [])
                context = execution.get('context_features', {})
                
                # Create pattern signature
                action_signature = tuple(sorted(actions))
                pattern_key = f"{strategy_type}:{hash(action_signature)}"
                
                pattern_tracker[pattern_key]['projects'].add(project_id)
                pattern_tracker[pattern_key]['success_rates'].append(1.0)  # Successful execution
                pattern_tracker[pattern_key]['contexts'].append(context)
                pattern_tracker[pattern_key]['actions'].extend(actions)
        
        # Filter patterns that appear in multiple projects
        cross_project_patterns = []
        for pattern_key, info in pattern_tracker.items():
            if len(info['projects']) >= 2:  # At least 2 projects
                avg_success_rate = statistics.mean(info['success_rates'])
                if avg_success_rate >= 0.8:  # High success rate
                    cross_project_patterns.append({
                        'pattern_key': pattern_key,
                        'projects': list(info['projects']),
                        'success_rate': avg_success_rate,
                        'contexts': info['contexts'],
                        'actions': list(set(info['actions']))
                    })
        
        return cross_project_patterns
    
    async def _create_cross_project_insight(
        self,
        pattern_info: Dict[str, Any]
    ) -> Optional[CrossProjectInsight]:
        """Create a cross-project insight from pattern information."""
        strategy_type_str = pattern_info['pattern_key'].split(':')[0]
        
        try:
            strategy_type = StrategyType(strategy_type_str)
        except ValueError:
            return None
        
        # Analyze contexts to determine applicability conditions
        contexts = pattern_info['contexts']
        common_conditions = []
        
        if contexts:
            # Find common context features
            feature_counts = defaultdict(Counter)
            for context in contexts:
                for key, value in context.items():
                    feature_counts[key][value] += 1
            
            # Extract conditions that appear in majority of contexts
            threshold = len(contexts) * 0.6
            for feature, value_counts in feature_counts.items():
                for value, count in value_counts.items():
                    if count >= threshold:
                        common_conditions.append(f"{feature}={value}")
        
        insight_id = f"insight_{hash(pattern_info['pattern_key'])}_{int(datetime.now().timestamp())}"
        
        return CrossProjectInsight(
            insight_id=insight_id,
            source_projects=pattern_info['projects'],
            target_domains=[strategy_type_str],
            strategy_type=strategy_type,
            insight_description=f"Successful pattern for {strategy_type_str} across {len(pattern_info['projects'])} projects",
            applicability_conditions=common_conditions,
            success_probability=pattern_info['success_rate'],
            confidence_score=min(pattern_info['success_rate'], len(pattern_info['projects']) / 5),
            generated_at=datetime.now().isoformat()
        )
    
    async def apply_cross_project_insight(
        self,
        insight: CrossProjectInsight,
        target_project_context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Apply a cross-project insight to a target project context."""
        self.logger.info(f"Applying insight {insight.insight_id} to target project")
        
        # Check if applicability conditions are met
        conditions_met = 0
        for condition in insight.applicability_conditions:
            try:
                key, value = condition.split('=', 1)
                if target_project_context.get(key) == value:
                    conditions_met += 1
            except ValueError:
                continue
        
        # Calculate applicability score
        if insight.applicability_conditions:
            applicability_score = conditions_met / len(insight.applicability_conditions)
        else:
            applicability_score = 0.5  # Default when no specific conditions
        
        if applicability_score < 0.6:  # Not applicable enough
            return None
        
        # Create application recommendation
        recommendation = {
            'insight_id': insight.insight_id,
            'strategy_type': insight.strategy_type.value,
            'description': insight.insight_description,
            'success_probability': insight.success_probability * applicability_score,
            'confidence': insight.confidence_score * applicability_score,
            'applicability_score': applicability_score,
            'recommended_actions': [
                f"Apply {insight.strategy_type.value} strategy",
                "Monitor success metrics",
                "Compare with baseline performance"
            ],
            'source_projects': insight.source_projects,
            'application_context': target_project_context
        }
        
        return recommendation


class AdaptiveLearningSystem:
    """Main orchestrator for adaptive learning capabilities."""
    
    def __init__(self, chroma_manager=None, state_manager=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.chroma_manager = chroma_manager
        self.state_manager = state_manager
        
        # Initialize engines
        self.pattern_engine = PatternRecognitionEngine()
        self.strategy_framework = StrategyEvolutionFramework(chroma_manager)
        self.ab_testing = ABTestingEngine()
        self.cross_project_learning = CrossProjectLearningEngine(chroma_manager)
        
        # Learning configuration
        self.learning_config = {
            'pattern_analysis_interval_days': 7,
            'min_confidence_for_updates': 0.8,
            'max_concurrent_experiments': 3,
            'enable_cross_project_learning': True
        }
    
    async def analyze_and_learn(
        self,
        project_id: Optional[str] = None,
        time_window_days: int = 30
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis and learning cycle."""
        self.logger.info(f"Starting adaptive learning cycle for project: {project_id or 'all'}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'project_id': project_id,
            'patterns_identified': [],
            'strategy_updates_proposed': [],
            'experiments_analyzed': [],
            'cross_project_insights': [],
            'recommendations': []
        }
        
        try:
            # Step 1: Gather execution data
            execution_data = await self._gather_execution_data(project_id, time_window_days)
            
            # Step 2: Identify patterns
            patterns = await self.pattern_engine.analyze_execution_patterns(
                execution_data, time_window_days
            )
            results['patterns_identified'] = [pattern.to_dict() for pattern in patterns]
            
            # Step 3: Propose strategy updates
            current_strategies = await self._get_current_strategies()
            strategy_proposals = await self.strategy_framework.propose_strategy_updates(
                patterns, current_strategies
            )
            results['strategy_updates_proposed'] = strategy_proposals
            
            # Step 4: Analyze ongoing experiments
            experiment_results = await self._analyze_active_experiments(execution_data)
            results['experiments_analyzed'] = experiment_results
            
            # Step 5: Cross-project learning (if enabled)
            if self.learning_config['enable_cross_project_learning']:
                project_data = await self._gather_multi_project_data(time_window_days)
                insights = await self.cross_project_learning.extract_cross_project_insights(project_data)
                results['cross_project_insights'] = [insight.to_dict() for insight in insights]
            
            # Step 6: Generate recommendations
            recommendations = await self._generate_learning_recommendations(results)
            results['recommendations'] = recommendations
            
            self.logger.info(f"Learning cycle completed with {len(patterns)} patterns and {len(recommendations)} recommendations")
            
        except Exception as e:
            self.logger.error(f"Error in adaptive learning cycle: {e}", exc_info=True)
            results['error'] = str(e)
        
        return results
    
    async def _gather_execution_data(
        self,
        project_id: Optional[str],
        time_window_days: int
    ) -> List[Dict[str, Any]]:
        """Gather execution data for analysis."""
        # This would integrate with StateManager and ChromaDB to get actual execution data
        # For now, return a mock structure
        return [
            {
                'timestamp': datetime.now().isoformat(),
                'project_type': 'python',
                'strategy_type': 'dependency_resolution',
                'agent_type': 'DependencyManagementAgent_v1',
                'success': True,
                'actions_taken': ['analyze_dependencies', 'resolve_conflicts', 'install_packages'],
                'success_indicators': ['no_conflicts', 'all_installed', 'tests_pass'],
                'execution_time': 45.2,
                'complexity_level': 'medium'
            }
        ]
    
    async def _get_current_strategies(self) -> Dict[str, Any]:
        """Get current strategy configurations."""
        # This would load current strategy configurations from configuration files
        return {
            'dependency_resolution': {
                'type': 'dependency_resolution',
                'preferred_actions': ['analyze_first', 'resolve_conflicts'],
                'timeout': 300
            },
            'error_recovery': {
                'type': 'error_recovery',
                'max_retries': 3,
                'escalation_threshold': 0.7
            }
        }
    
    async def _analyze_active_experiments(
        self,
        execution_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze results of active A/B testing experiments."""
        experiment_results = []
        
        for experiment_id in list(self.ab_testing.active_experiments.keys()):
            results = await self.ab_testing.collect_experiment_results(
                experiment_id, execution_data
            )
            if results:
                experiment_results.append({
                    'experiment_id': experiment_id,
                    'results': results
                })
        
        return experiment_results
    
    async def _gather_multi_project_data(
        self,
        time_window_days: int
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Gather execution data from multiple projects."""
        # This would query multiple projects' execution data
        return {
            'project_1': [],
            'project_2': []
        }
    
    async def _generate_learning_recommendations(
        self,
        analysis_results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations from learning analysis."""
        recommendations = []
        
        # Recommend high-confidence strategy updates
        for proposal in analysis_results['strategy_updates_proposed']:
            if proposal.get('confidence', 0) >= self.learning_config['min_confidence_for_updates']:
                recommendations.append({
                    'type': 'strategy_update',
                    'priority': 'high',
                    'description': f"Update {proposal['strategy_type']} strategy",
                    'action': 'apply_strategy_update',
                    'details': proposal
                })
        
        # Recommend new experiments for promising patterns
        for pattern_dict in analysis_results['patterns_identified']:
            if pattern_dict.get('confidence_score', 0) >= 0.75:
                recommendations.append({
                    'type': 'experiment',
                    'priority': 'medium',
                    'description': f"A/B test pattern {pattern_dict['pattern_id']}",
                    'action': 'create_ab_test',
                    'details': pattern_dict
                })
        
        # Recommend applying cross-project insights
        for insight_dict in analysis_results['cross_project_insights']:
            if insight_dict.get('confidence_score', 0) >= 0.7:
                recommendations.append({
                    'type': 'cross_project_insight',
                    'priority': 'medium',
                    'description': f"Apply insight {insight_dict['insight_id']}",
                    'action': 'apply_insight',
                    'details': insight_dict
                })
        
        return recommendations


# ============================================================================
# MCP Tool Functions
# ============================================================================

async def adaptive_learning_analyze(
    project_id: Optional[str] = None,
    time_window_days: int = 30,
    enable_cross_project: bool = True
) -> Dict[str, Any]:
    """
    MCP tool function for performing adaptive learning analysis.
    
    Args:
        project_id: Specific project to analyze (None for all projects)
        time_window_days: Number of days of historical data to analyze
        enable_cross_project: Whether to include cross-project learning
        
    Returns:
        Dictionary containing analysis results and recommendations
    """
    try:
        learning_system = AdaptiveLearningSystem()
        learning_system.learning_config['enable_cross_project_learning'] = enable_cross_project
        
        results = await learning_system.analyze_and_learn(
            project_id=project_id,
            time_window_days=time_window_days
        )
        
        return {
            'success': True,
            'results': results,
            'summary': f"Identified {len(results.get('patterns_identified', []))} patterns and {len(results.get('recommendations', []))} recommendations"
        }
        
    except Exception as e:
        logger.error(f"Error in adaptive learning analysis: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'results': None
        }


async def create_strategy_experiment(
    baseline_strategy: Dict[str, Any],
    test_strategy: Dict[str, Any],
    success_metric: str = "success_rate",
    duration_days: int = 7
) -> Dict[str, Any]:
    """
    MCP tool function for creating A/B testing experiments.
    
    Args:
        baseline_strategy: Current baseline strategy configuration
        test_strategy: New strategy variant to test
        success_metric: Metric to optimize for
        duration_days: How long to run the experiment
        
    Returns:
        Dictionary containing experiment details
    """
    try:
        ab_testing = ABTestingEngine()
        
        experiment = await ab_testing.create_experiment(
            baseline_strategy=baseline_strategy,
            test_strategy=test_strategy,
            success_metric=success_metric,
            duration_days=duration_days
        )
        
        return {
            'success': True,
            'experiment': experiment.to_dict(),
            'experiment_id': experiment.experiment_id
        }
        
    except Exception as e:
        logger.error(f"Error creating strategy experiment: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'experiment': None
        }


async def apply_learning_recommendations(
    recommendations: List[Dict[str, Any]],
    auto_apply_threshold: float = 0.9
) -> Dict[str, Any]:
    """
    MCP tool function for applying learning recommendations.
    
    Args:
        recommendations: List of recommendations to process
        auto_apply_threshold: Confidence threshold for automatic application
        
    Returns:
        Dictionary containing application results
    """
    try:
        applied_count = 0
        skipped_count = 0
        results = []
        
        learning_system = AdaptiveLearningSystem()
        
        for recommendation in recommendations:
            confidence = recommendation.get('details', {}).get('confidence', 0)
            
            if confidence >= auto_apply_threshold:
                # Auto-apply high-confidence recommendations
                if recommendation['type'] == 'strategy_update':
                    # Apply strategy update
                    result = {
                        'recommendation_id': recommendation.get('id', 'unknown'),
                        'action': 'applied',
                        'confidence': confidence
                    }
                    applied_count += 1
                else:
                    result = {
                        'recommendation_id': recommendation.get('id', 'unknown'),
                        'action': 'scheduled',
                        'confidence': confidence
                    }
                    applied_count += 1
            else:
                result = {
                    'recommendation_id': recommendation.get('id', 'unknown'),
                    'action': 'skipped_low_confidence',
                    'confidence': confidence
                }
                skipped_count += 1
            
            results.append(result)
        
        return {
            'success': True,
            'applied_count': applied_count,
            'skipped_count': skipped_count,
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error applying learning recommendations: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'applied_count': 0
        } 