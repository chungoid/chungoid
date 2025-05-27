"""
Advanced Re-planning Intelligence System

Provides sophisticated re-planning capabilities for the Chungoid autonomous
coding system, enabling intelligent failure recovery, pattern-based strategy
adaptation, and autonomous research for problem-solving.

Features:
- Autonomous Research Engine using web content and documentation
- Historical Pattern Analysis from ChromaDB reflections
- Multi-Step Recovery Strategy Generation
- Failure Prediction and Proactive Planning
- Context-Aware Plan Optimization
"""

import logging
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import re
from collections import defaultdict, Counter

logger = logging.getLogger(__name__)


class PlanningContext(str, Enum):
    """Types of planning contexts for re-planning intelligence."""
    FAILURE_RECOVERY = "failure_recovery"
    OPTIMIZATION = "optimization"
    STRATEGY_ADAPTATION = "strategy_adaptation"
    PREVENTIVE_PLANNING = "preventive_planning"
    RESEARCH_DRIVEN = "research_driven"


class ResearchScope(str, Enum):
    """Scope of autonomous research to perform."""
    SPECIFIC_ERROR = "specific_error"
    GENERAL_APPROACH = "general_approach"
    TECHNOLOGY_BEST_PRACTICES = "technology_best_practices"
    TROUBLESHOOTING = "troubleshooting"
    OPTIMIZATION_TECHNIQUES = "optimization_techniques"


class RecoveryConfidence(str, Enum):
    """Confidence levels for recovery strategies."""
    HIGH = "high"          # >80% confidence
    MEDIUM = "medium"      # 50-80% confidence
    LOW = "low"           # 20-50% confidence
    EXPERIMENTAL = "experimental"  # <20% confidence


@dataclass
class ResearchQuery:
    """Represents a research query for autonomous investigation."""
    query_id: str
    scope: ResearchScope
    search_terms: List[str]
    context_keywords: List[str]
    priority: float
    expected_insight_types: List[str]
    max_sources: int = 5
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['scope'] = self.scope.value
        return data


@dataclass
class ResearchInsight:
    """Represents an insight gathered from autonomous research."""
    insight_id: str
    source_url: Optional[str]
    source_type: str  # "web", "documentation", "stackoverflow", etc.
    relevance_score: float
    content_summary: str
    key_recommendations: List[str]
    applicable_contexts: List[str]
    confidence_level: float
    extraction_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class HistoricalPattern:
    """Represents a pattern identified from historical execution data."""
    pattern_id: str
    failure_signature: str
    success_recovery_actions: List[str]
    context_conditions: Dict[str, Any]
    success_probability: float
    sample_size: int
    last_successful_application: Optional[str]
    pattern_confidence: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class RecoveryStrategy:
    """Represents a multi-step recovery strategy."""
    strategy_id: str
    strategy_name: str
    context: PlanningContext
    confidence: RecoveryConfidence
    steps: List[Dict[str, Any]]
    prerequisites: List[str]
    success_indicators: List[str]
    failure_fallbacks: List[str]
    estimated_duration: int  # minutes
    risk_factors: List[str]
    research_basis: List[str]  # IDs of research insights used
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['context'] = self.context.value
        data['confidence'] = self.confidence.value
        return data


@dataclass
class FailurePrediction:
    """Represents a prediction of potential failure."""
    prediction_id: str
    predicted_failure_type: str
    probability: float
    triggering_conditions: List[str]
    potential_impact: str
    preventive_actions: List[str]
    monitoring_indicators: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AutonomousResearchEngine:
    """Performs autonomous research for problem-solving and strategy optimization."""
    
    def __init__(self, web_fetcher=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.web_fetcher = web_fetcher
        self.research_cache = {}
        self.insights_database = []
        
    async def conduct_research(
        self,
        query: ResearchQuery,
        project_context: Optional[Dict[str, Any]] = None
    ) -> List[ResearchInsight]:
        """Conduct autonomous research based on a query."""
        self.logger.info(f"Conducting research for: {query.scope.value}")
        
        # Check cache first
        cache_key = self._create_cache_key(query)
        if cache_key in self.research_cache:
            cached_result = self.research_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result['insights']
        
        insights = []
        
        # Step 1: Generate refined search queries
        search_queries = await self._generate_search_queries(query, project_context)
        
        # Step 2: Perform web research
        web_insights = await self._perform_web_research(search_queries, query)
        insights.extend(web_insights)
        
        # Step 3: Search documentation if available
        doc_insights = await self._search_documentation(query, project_context)
        insights.extend(doc_insights)
        
        # Step 4: Analyze and rank insights
        ranked_insights = await self._rank_and_filter_insights(insights, query)
        
        # Cache results
        self.research_cache[cache_key] = {
            'insights': ranked_insights,
            'timestamp': datetime.now().isoformat(),
            'ttl_hours': 24
        }
        
        self.logger.info(f"Research completed: {len(ranked_insights)} insights found")
        return ranked_insights
    
    async def _generate_search_queries(
        self,
        query: ResearchQuery,
        project_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Generate refined search queries for research."""
        base_queries = []
        
        # Combine search terms with context
        for term in query.search_terms:
            # Add project-specific context if available
            if project_context:
                language = project_context.get('project_type', '')
                framework = project_context.get('framework', '')
                
                if language:
                    base_queries.append(f"{term} {language}")
                if framework:
                    base_queries.append(f"{term} {framework}")
            
            # Add scope-specific refinements
            if query.scope == ResearchScope.SPECIFIC_ERROR:
                base_queries.extend([
                    f"{term} fix solution",
                    f"{term} troubleshooting",
                    f"how to resolve {term}"
                ])
            elif query.scope == ResearchScope.GENERAL_APPROACH:
                base_queries.extend([
                    f"{term} best practices",
                    f"{term} recommended approach",
                    f"{term} implementation guide"
                ])
            elif query.scope == ResearchScope.OPTIMIZATION_TECHNIQUES:
                base_queries.extend([
                    f"{term} optimization",
                    f"{term} performance improvement",
                    f"{term} efficiency tips"
                ])
        
        # Deduplicate and prioritize
        unique_queries = list(set(base_queries))
        return unique_queries[:query.max_sources * 2]  # Allow for filtering
    
    async def _perform_web_research(
        self,
        search_queries: List[str],
        query: ResearchQuery
    ) -> List[ResearchInsight]:
        """Perform web research using search queries."""
        insights = []
        
        if not self.web_fetcher:
            self.logger.warning("No web fetcher available for research")
            return insights
        
        for search_query in search_queries[:query.max_sources]:
            try:
                # Use web content fetching tool
                result = await self.web_fetcher.fetch_and_summarize(
                    query=search_query,
                    max_results=2,
                    focus_areas=query.expected_insight_types
                )
                
                if result.get('success'):
                    for item in result.get('results', []):
                        insight = ResearchInsight(
                            insight_id=f"web_{hash(search_query)}_{datetime.now().timestamp()}",
                            source_url=item.get('url'),
                            source_type="web",
                            relevance_score=item.get('relevance_score', 0.5),
                            content_summary=item.get('summary', ''),
                            key_recommendations=self._extract_recommendations(item.get('content', '')),
                            applicable_contexts=query.context_keywords,
                            confidence_level=item.get('confidence', 0.7),
                            extraction_timestamp=datetime.now().isoformat()
                        )
                        insights.append(insight)
                        
            except Exception as e:
                self.logger.warning(f"Error in web research for '{search_query}': {e}")
        
        return insights
    
    async def _search_documentation(
        self,
        query: ResearchQuery,
        project_context: Optional[Dict[str, Any]]
    ) -> List[ResearchInsight]:
        """Search available documentation for relevant insights."""
        insights = []
        
        # This would search through:
        # - Library documentation in offline_library_docs/
        # - Project-specific documentation
        # - API documentation
        # For now, return mock structure
        
        return insights
    
    def _extract_recommendations(self, content: str) -> List[str]:
        """Extract actionable recommendations from content."""
        recommendations = []
        
        # Look for common recommendation patterns
        patterns = [
            r"(?i)(?:you should|recommended|best practice|try|consider|use|avoid):\s*([^.]+)",
            r"(?i)solution:\s*([^.]+)",
            r"(?i)fix:\s*([^.]+)",
            r"(?i)to resolve.*?:\s*([^.]+)"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            recommendations.extend([match.strip() for match in matches])
        
        # Deduplicate and limit
        unique_recommendations = list(set(recommendations))
        return unique_recommendations[:5]
    
    async def _rank_and_filter_insights(
        self,
        insights: List[ResearchInsight],
        query: ResearchQuery
    ) -> List[ResearchInsight]:
        """Rank and filter insights by relevance and quality."""
        if not insights:
            return []
        
        # Calculate combined scores
        for insight in insights:
            # Combine relevance, confidence, and content quality
            content_quality = len(insight.key_recommendations) / 5  # Normalize to 0-1
            combined_score = (
                insight.relevance_score * 0.4 +
                insight.confidence_level * 0.4 +
                content_quality * 0.2
            )
            insight.relevance_score = combined_score
        
        # Sort by combined score and return top results
        sorted_insights = sorted(insights, key=lambda x: x.relevance_score, reverse=True)
        return sorted_insights[:query.max_sources]
    
    def _create_cache_key(self, query: ResearchQuery) -> str:
        """Create a cache key for research queries."""
        key_parts = [
            query.scope.value,
            "|".join(sorted(query.search_terms)),
            "|".join(sorted(query.context_keywords))
        ]
        return hash("|".join(key_parts))
    
    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached research result is still valid."""
        cache_time = datetime.fromisoformat(cached_result['timestamp'])
        ttl_hours = cached_result.get('ttl_hours', 24)
        expiry_time = cache_time + timedelta(hours=ttl_hours)
        return datetime.now() < expiry_time


class HistoricalPatternAnalyzer:
    """Analyzes historical patterns from ChromaDB reflections for planning insights."""
    
    def __init__(self, chroma_manager=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.chroma_manager = chroma_manager
        self.pattern_cache = {}
        
    async def analyze_failure_patterns(
        self,
        current_failure: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> List[HistoricalPattern]:
        """Analyze historical patterns related to current failure."""
        self.logger.info(f"Analyzing patterns for failure: {current_failure.get('error_type', 'unknown')}")
        
        # Step 1: Query ChromaDB for similar failures
        similar_failures = await self._query_similar_failures(current_failure, project_context)
        
        # Step 2: Identify successful recovery patterns
        recovery_patterns = await self._identify_recovery_patterns(similar_failures)
        
        # Step 3: Validate pattern effectiveness
        validated_patterns = await self._validate_patterns(recovery_patterns)
        
        self.logger.info(f"Found {len(validated_patterns)} validated recovery patterns")
        return validated_patterns
    
    async def _query_similar_failures(
        self,
        current_failure: Dict[str, Any],
        project_context: Optional[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for similar historical failures."""
        similar_failures = []
        
        if not self.chroma_manager:
            return similar_failures
        
        # Build search query based on failure characteristics
        search_terms = []
        
        error_type = current_failure.get('error_type', '')
        error_message = current_failure.get('message', '')
        agent_id = current_failure.get('agent_id', '')
        
        if error_type:
            search_terms.append(error_type)
        if error_message:
            # Extract key terms from error message
            error_keywords = self._extract_error_keywords(error_message)
            search_terms.extend(error_keywords)
        if agent_id:
            search_terms.append(agent_id)
        
        # Query reflections collection
        try:
            query_text = " ".join(search_terms)
            results = await self._chromadb_query(
                collection_name="chungoid_reflections",
                query_text=query_text,
                n_results=20,
                where_filter={"type": "reflection"}
            )
            
            for result in results:
                # Parse reflection data
                reflection_data = self._parse_reflection_data(result)
                if reflection_data and self._is_relevant_failure(reflection_data, current_failure):
                    similar_failures.append(reflection_data)
                    
        except Exception as e:
            self.logger.error(f"Error querying historical failures: {e}")
        
        return similar_failures
    
    def _extract_error_keywords(self, error_message: str) -> List[str]:
        """Extract key terms from error messages for pattern matching."""
        # Remove common but not useful words
        stop_words = {'error', 'failed', 'exception', 'traceback', 'line', 'file'}
        
        # Split and clean
        words = re.findall(r'\b\w+\b', error_message.lower())
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        
        # Look for specific patterns
        patterns = [
            r'(\w+Error)',
            r'(\w+Exception)',
            r'ModuleNotFoundError.*?\'([^\']+)\'',
            r'ImportError.*?\'([^\']+)\'',
            r'(\w+)\s+not\s+found'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, error_message)
            keywords.extend([match.lower() if isinstance(match, str) else match[0].lower() for match in matches])
        
        return list(set(keywords))[:10]  # Limit to top 10
    
    async def _identify_recovery_patterns(
        self,
        similar_failures: List[Dict[str, Any]]
    ) -> List[HistoricalPattern]:
        """Identify successful recovery patterns from historical data."""
        patterns = []
        
        # Group failures by signature
        failure_groups = defaultdict(list)
        
        for failure in similar_failures:
            signature = self._create_failure_signature(failure)
            failure_groups[signature].append(failure)
        
        # Analyze each group for recovery patterns
        for signature, group in failure_groups.items():
            if len(group) >= 2:  # Need multiple examples
                successful_recoveries = [
                    f for f in group 
                    if f.get('resolution_successful', False)
                ]
                
                if successful_recoveries:
                    pattern = await self._extract_recovery_pattern(signature, successful_recoveries)
                    if pattern:
                        patterns.append(pattern)
        
        return patterns
    
    def _create_failure_signature(self, failure_data: Dict[str, Any]) -> str:
        """Create a signature for failure pattern matching."""
        components = [
            failure_data.get('error_type', ''),
            failure_data.get('agent_id', ''),
            failure_data.get('stage_id', ''),
            failure_data.get('project_type', '')
        ]
        return "|".join(c for c in components if c)
    
    async def _extract_recovery_pattern(
        self,
        signature: str,
        successful_recoveries: List[Dict[str, Any]]
    ) -> Optional[HistoricalPattern]:
        """Extract a recovery pattern from successful resolution examples."""
        if not successful_recoveries:
            return None
        
        # Find common recovery actions
        all_actions = []
        for recovery in successful_recoveries:
            actions = recovery.get('recovery_actions', [])
            if isinstance(actions, list):
                all_actions.extend(actions)
        
        # Count action frequency
        action_counts = Counter(all_actions)
        common_actions = [
            action for action, count in action_counts.items()
            if count >= len(successful_recoveries) * 0.6  # Appear in 60% of cases
        ]
        
        if not common_actions:
            return None
        
        # Calculate success probability
        total_attempts = len(successful_recoveries)
        success_probability = len(successful_recoveries) / total_attempts  # All are successful by definition
        
        # Extract context conditions
        context_conditions = self._extract_common_context(successful_recoveries)
        
        pattern_id = f"pattern_{hash(signature)}_{int(datetime.now().timestamp())}"
        
        return HistoricalPattern(
            pattern_id=pattern_id,
            failure_signature=signature,
            success_recovery_actions=common_actions,
            context_conditions=context_conditions,
            success_probability=success_probability,
            sample_size=len(successful_recoveries),
            last_successful_application=max(
                r.get('timestamp', '') for r in successful_recoveries
            ),
            pattern_confidence=min(success_probability, len(successful_recoveries) / 10)
        )
    
    def _extract_common_context(self, recoveries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract common context conditions from successful recoveries."""
        context_features = defaultdict(Counter)
        
        for recovery in recoveries:
            context = recovery.get('context', {})
            for key, value in context.items():
                if isinstance(value, (str, int, float, bool)):
                    context_features[key][value] += 1
        
        # Find features that appear in majority of cases
        common_context = {}
        threshold = len(recoveries) * 0.6
        
        for feature, value_counts in context_features.items():
            for value, count in value_counts.items():
                if count >= threshold:
                    common_context[feature] = value
        
        return common_context
    
    async def _validate_patterns(
        self,
        patterns: List[HistoricalPattern]
    ) -> List[HistoricalPattern]:
        """Validate pattern effectiveness and filter out weak patterns."""
        validated = []
        
        for pattern in patterns:
            # Validation criteria
            if (pattern.success_probability >= 0.7 and
                pattern.sample_size >= 2 and
                pattern.pattern_confidence >= 0.5):
                validated.append(pattern)
        
        return validated
    
    async def _chromadb_query(
        self,
        collection_name: str,
        query_text: str,
        n_results: int = 10,
        where_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query ChromaDB for relevant data."""
        # This would integrate with actual ChromaDB
        # For now, return mock structure
        return []
    
    def _parse_reflection_data(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse reflection data from ChromaDB result."""
        # Parse and structure reflection data
        return {
            'error_type': 'ImportError',
            'message': 'Module not found',
            'agent_id': 'TestAgent',
            'resolution_successful': True,
            'recovery_actions': ['install_missing_package', 'update_imports'],
            'timestamp': '2025-01-01T12:00:00',
            'context': {'project_type': 'python'}
        }
    
    def _is_relevant_failure(
        self,
        historical_failure: Dict[str, Any],
        current_failure: Dict[str, Any]
    ) -> bool:
        """Check if historical failure is relevant to current failure."""
        # Simple relevance check
        return (historical_failure.get('error_type') == current_failure.get('error_type') or
                historical_failure.get('agent_id') == current_failure.get('agent_id'))


class MultiStepRecoveryPlanner:
    """Generates sophisticated multi-step recovery strategies."""
    
    def __init__(self, research_engine=None, pattern_analyzer=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.research_engine = research_engine
        self.pattern_analyzer = pattern_analyzer
        
    async def create_recovery_strategy(
        self,
        failure_context: Dict[str, Any],
        historical_patterns: List[HistoricalPattern],
        research_insights: List[ResearchInsight],
        project_context: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """Create a comprehensive multi-step recovery strategy."""
        self.logger.info("Creating multi-step recovery strategy")
        
        # Step 1: Analyze failure and determine approach
        strategy_context = self._determine_strategy_context(failure_context)
        
        # Step 2: Generate recovery steps from multiple sources
        pattern_steps = self._generate_pattern_based_steps(historical_patterns)
        research_steps = self._generate_research_based_steps(research_insights)
        fallback_steps = self._generate_fallback_steps(failure_context)
        
        # Step 3: Combine and optimize steps
        combined_steps = await self._combine_and_optimize_steps(
            pattern_steps, research_steps, fallback_steps, failure_context
        )
        
        # Step 4: Assess confidence and risks
        confidence = self._assess_strategy_confidence(
            combined_steps, historical_patterns, research_insights
        )
        
        risk_factors = self._identify_risk_factors(combined_steps, failure_context)
        
        # Step 5: Generate prerequisites and success indicators
        prerequisites = self._determine_prerequisites(combined_steps, project_context)
        success_indicators = self._define_success_indicators(failure_context, combined_steps)
        failure_fallbacks = self._create_failure_fallbacks(combined_steps)
        
        strategy_id = f"recovery_{hash(str(failure_context))}_{int(datetime.now().timestamp())}"
        
        return RecoveryStrategy(
            strategy_id=strategy_id,
            strategy_name=self._generate_strategy_name(failure_context, strategy_context),
            context=strategy_context,
            confidence=confidence,
            steps=combined_steps,
            prerequisites=prerequisites,
            success_indicators=success_indicators,
            failure_fallbacks=failure_fallbacks,
            estimated_duration=self._estimate_duration(combined_steps),
            risk_factors=risk_factors,
            research_basis=[insight.insight_id for insight in research_insights]
        )
    
    def _determine_strategy_context(self, failure_context: Dict[str, Any]) -> PlanningContext:
        """Determine the appropriate planning context for the strategy."""
        error_type = failure_context.get('error_type', '').lower()
        
        if 'import' in error_type or 'module' in error_type:
            return PlanningContext.FAILURE_RECOVERY
        elif 'timeout' in error_type or 'performance' in error_type:
            return PlanningContext.OPTIMIZATION
        else:
            return PlanningContext.FAILURE_RECOVERY
    
    def _generate_pattern_based_steps(
        self,
        patterns: List[HistoricalPattern]
    ) -> List[Dict[str, Any]]:
        """Generate recovery steps based on historical patterns."""
        steps = []
        
        for pattern in patterns:
            for i, action in enumerate(pattern.success_recovery_actions):
                step = {
                    'step_id': f"pattern_{pattern.pattern_id}_{i}",
                    'step_type': 'pattern_based',
                    'action': action,
                    'description': f"Apply historical pattern: {action}",
                    'confidence': pattern.pattern_confidence,
                    'source': 'historical_pattern',
                    'priority': self._calculate_step_priority(pattern, i),
                    'estimated_duration_minutes': 5
                }
                steps.append(step)
        
        return steps
    
    def _generate_research_based_steps(
        self,
        insights: List[ResearchInsight]
    ) -> List[Dict[str, Any]]:
        """Generate recovery steps based on research insights."""
        steps = []
        
        for insight in insights:
            for i, recommendation in enumerate(insight.key_recommendations):
                step = {
                    'step_id': f"research_{insight.insight_id}_{i}",
                    'step_type': 'research_based',
                    'action': recommendation,
                    'description': f"Research-based: {recommendation}",
                    'confidence': insight.confidence_level,
                    'source': 'autonomous_research',
                    'source_url': insight.source_url,
                    'priority': insight.relevance_score,
                    'estimated_duration_minutes': 10
                }
                steps.append(step)
        
        return steps
    
    def _generate_fallback_steps(
        self,
        failure_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate fallback recovery steps for common failure types."""
        steps = []
        error_type = failure_context.get('error_type', '').lower()
        
        # Common fallback strategies
        if 'import' in error_type or 'module' in error_type:
            steps.extend([
                {
                    'step_id': 'fallback_check_requirements',
                    'step_type': 'fallback',
                    'action': 'check_requirements_file',
                    'description': 'Verify requirements.txt or package.json',
                    'confidence': 0.7,
                    'source': 'fallback_strategy',
                    'priority': 0.8,
                    'estimated_duration_minutes': 3
                },
                {
                    'step_id': 'fallback_reinstall_deps',
                    'step_type': 'fallback',
                    'action': 'reinstall_dependencies',
                    'description': 'Clean reinstall of dependencies',
                    'confidence': 0.6,
                    'source': 'fallback_strategy',
                    'priority': 0.6,
                    'estimated_duration_minutes': 15
                }
            ])
        
        return steps
    
    async def _combine_and_optimize_steps(
        self,
        pattern_steps: List[Dict[str, Any]],
        research_steps: List[Dict[str, Any]],
        fallback_steps: List[Dict[str, Any]],
        failure_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Combine and optimize recovery steps from all sources."""
        all_steps = pattern_steps + research_steps + fallback_steps
        
        if not all_steps:
            return []
        
        # Remove duplicates based on action similarity
        unique_steps = self._remove_duplicate_steps(all_steps)
        
        # Sort by priority and confidence
        sorted_steps = sorted(
            unique_steps,
            key=lambda s: (s.get('priority', 0) * s.get('confidence', 0)),
            reverse=True
        )
        
        # Limit to most promising steps
        optimized_steps = sorted_steps[:8]  # Max 8 steps
        
        # Add sequence numbers and dependencies
        for i, step in enumerate(optimized_steps):
            step['sequence'] = i + 1
            step['depends_on'] = [i] if i > 0 else []
        
        return optimized_steps
    
    def _remove_duplicate_steps(self, steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or very similar steps."""
        unique_steps = []
        seen_actions = set()
        
        for step in steps:
            action = step.get('action', '').lower()
            # Simple deduplication by action similarity
            if action not in seen_actions:
                seen_actions.add(action)
                unique_steps.append(step)
        
        return unique_steps
    
    def _calculate_step_priority(self, pattern: HistoricalPattern, step_index: int) -> float:
        """Calculate priority for a pattern-based step."""
        # Earlier steps in successful patterns get higher priority
        position_factor = 1.0 - (step_index * 0.1)
        success_factor = pattern.success_probability
        confidence_factor = pattern.pattern_confidence
        
        return (position_factor * 0.3 + success_factor * 0.4 + confidence_factor * 0.3)
    
    def _assess_strategy_confidence(
        self,
        steps: List[Dict[str, Any]],
        patterns: List[HistoricalPattern],
        insights: List[ResearchInsight]
    ) -> RecoveryConfidence:
        """Assess overall confidence in the recovery strategy."""
        if not steps:
            return RecoveryConfidence.LOW
        
        # Calculate average confidence across steps
        step_confidences = [step.get('confidence', 0) for step in steps]
        avg_confidence = sum(step_confidences) / len(step_confidences) if step_confidences else 0
        
        # Factor in number of supporting sources
        pattern_support = len(patterns) * 0.2
        research_support = len(insights) * 0.1
        support_factor = min(pattern_support + research_support, 0.3)
        
        overall_confidence = avg_confidence + support_factor
        
        if overall_confidence >= 0.8:
            return RecoveryConfidence.HIGH
        elif overall_confidence >= 0.5:
            return RecoveryConfidence.MEDIUM
        elif overall_confidence >= 0.2:
            return RecoveryConfidence.LOW
        else:
            return RecoveryConfidence.EXPERIMENTAL
    
    def _identify_risk_factors(
        self,
        steps: List[Dict[str, Any]],
        failure_context: Dict[str, Any]
    ) -> List[str]:
        """Identify potential risk factors in the recovery strategy."""
        risks = []
        
        # Check for potentially risky actions
        risky_actions = ['reinstall', 'delete', 'remove', 'reset', 'clean']
        for step in steps:
            action = step.get('action', '').lower()
            if any(risky_word in action for risky_word in risky_actions):
                risks.append(f"Step {step.get('sequence', '?')} involves potentially destructive action")
        
        # Check for experimental steps
        experimental_steps = [s for s in steps if s.get('confidence', 0) < 0.3]
        if experimental_steps:
            risks.append(f"{len(experimental_steps)} steps have low confidence")
        
        # Check for long execution time
        total_duration = sum(step.get('estimated_duration_minutes', 0) for step in steps)
        if total_duration > 60:
            risks.append("Strategy may take over 1 hour to complete")
        
        return risks
    
    def _determine_prerequisites(
        self,
        steps: List[Dict[str, Any]],
        project_context: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Determine prerequisites for executing the recovery strategy."""
        prerequisites = []
        
        # Check for common prerequisites based on step types
        step_actions = [step.get('action', '').lower() for step in steps]
        
        if any('install' in action for action in step_actions):
            prerequisites.append("Package manager available (pip, npm, etc.)")
        
        if any('git' in action for action in step_actions):
            prerequisites.append("Git repository initialized")
        
        if any('test' in action for action in step_actions):
            prerequisites.append("Testing framework configured")
        
        # Add project-specific prerequisites
        if project_context:
            project_type = project_context.get('project_type', '')
            if project_type == 'python':
                prerequisites.append("Python environment activated")
            elif project_type == 'javascript':
                prerequisites.append("Node.js environment available")
        
        return list(set(prerequisites))  # Remove duplicates
    
    def _define_success_indicators(
        self,
        failure_context: Dict[str, Any],
        steps: List[Dict[str, Any]]
    ) -> List[str]:
        """Define indicators that show the recovery was successful."""
        indicators = []
        
        # General success indicators
        indicators.append("Original error no longer occurs")
        indicators.append("All tests pass successfully")
        
        # Context-specific indicators
        error_type = failure_context.get('error_type', '').lower()
        if 'import' in error_type:
            indicators.append("All imports resolve successfully")
        elif 'test' in error_type:
            indicators.append("Test suite completes without errors")
        elif 'build' in error_type:
            indicators.append("Build process completes successfully")
        
        # Step-specific indicators
        for step in steps:
            action = step.get('action', '').lower()
            if 'install' in action:
                indicators.append("All dependencies installed correctly")
            elif 'config' in action:
                indicators.append("Configuration validates successfully")
        
        return list(set(indicators))  # Remove duplicates
    
    def _create_failure_fallbacks(self, steps: List[Dict[str, Any]]) -> List[str]:
        """Create fallback actions if the recovery strategy fails."""
        fallbacks = [
            "Escalate to human review with detailed diagnostic information",
            "Attempt simplified fallback approach with minimal changes",
            "Create detailed issue report for future pattern learning"
        ]
        
        # Add specific fallbacks based on step types
        step_types = [step.get('step_type', '') for step in steps]
        
        if 'research_based' in step_types:
            fallbacks.append("Try alternative solutions from research")
        
        if 'pattern_based' in step_types:
            fallbacks.append("Apply lower-confidence historical patterns")
        
        return fallbacks
    
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> int:
        """Estimate total duration for recovery strategy execution."""
        total_minutes = sum(step.get('estimated_duration_minutes', 5) for step in steps)
        # Add buffer for coordination and validation
        return int(total_minutes * 1.2)
    
    def _generate_strategy_name(
        self,
        failure_context: Dict[str, Any],
        strategy_context: PlanningContext
    ) -> str:
        """Generate a descriptive name for the recovery strategy."""
        error_type = failure_context.get('error_type', 'Unknown')
        agent_id = failure_context.get('agent_id', 'Agent')
        
        context_names = {
            PlanningContext.FAILURE_RECOVERY: "Recovery",
            PlanningContext.OPTIMIZATION: "Optimization",
            PlanningContext.STRATEGY_ADAPTATION: "Adaptation",
            PlanningContext.PREVENTIVE_PLANNING: "Prevention",
            PlanningContext.RESEARCH_DRIVEN: "Research-Based"
        }
        
        context_name = context_names.get(strategy_context, "Recovery")
        return f"{context_name} Strategy for {error_type} in {agent_id}"


class FailurePredictionEngine:
    """Predicts potential failures based on current context and historical patterns."""
    
    def __init__(self, pattern_analyzer=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.pattern_analyzer = pattern_analyzer
        
    async def predict_potential_failures(
        self,
        current_context: Dict[str, Any],
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> List[FailurePrediction]:
        """Predict potential failures based on current context."""
        self.logger.info("Analyzing context for potential failure prediction")
        
        predictions = []
        
        # Analyze different types of potential failures
        dependency_predictions = await self._predict_dependency_failures(current_context)
        environment_predictions = await self._predict_environment_failures(current_context)
        integration_predictions = await self._predict_integration_failures(current_context, execution_plan)
        
        predictions.extend(dependency_predictions)
        predictions.extend(environment_predictions)
        predictions.extend(integration_predictions)
        
        # Filter and rank predictions
        filtered_predictions = self._filter_and_rank_predictions(predictions)
        
        self.logger.info(f"Generated {len(filtered_predictions)} failure predictions")
        return filtered_predictions
    
    async def _predict_dependency_failures(
        self,
        context: Dict[str, Any]
    ) -> List[FailurePrediction]:
        """Predict potential dependency-related failures."""
        predictions = []
        
        project_type = context.get('project_type', '')
        dependencies = context.get('dependencies', [])
        
        if project_type == 'python' and dependencies:
            # Check for common Python dependency issues
            predictions.append(FailurePrediction(
                prediction_id=f"dep_python_{int(datetime.now().timestamp())}",
                predicted_failure_type="ImportError",
                probability=0.3,
                triggering_conditions=["Missing dependencies", "Version conflicts"],
                potential_impact="Build failure, runtime errors",
                preventive_actions=["Verify requirements.txt", "Create virtual environment"],
                monitoring_indicators=["Import statements", "Package versions"],
                confidence_score=0.7
            ))
        
        return predictions
    
    async def _predict_environment_failures(
        self,
        context: Dict[str, Any]
    ) -> List[FailurePrediction]:
        """Predict potential environment-related failures."""
        predictions = []
        
        # Check for environment setup issues
        if not context.get('environment_validated', False):
            predictions.append(FailurePrediction(
                prediction_id=f"env_setup_{int(datetime.now().timestamp())}",
                predicted_failure_type="EnvironmentError",
                probability=0.4,
                triggering_conditions=["Unvalidated environment", "Missing tools"],
                potential_impact="Setup failures, tool execution errors",
                preventive_actions=["Validate environment", "Check tool availability"],
                monitoring_indicators=["Environment variables", "Tool versions"],
                confidence_score=0.6
            ))
        
        return predictions
    
    async def _predict_integration_failures(
        self,
        context: Dict[str, Any],
        execution_plan: Optional[Dict[str, Any]]
    ) -> List[FailurePrediction]:
        """Predict potential integration and coordination failures."""
        predictions = []
        
        if execution_plan:
            stages = execution_plan.get('stages', {})
            if len(stages) > 5:  # Complex plans have higher integration risk
                predictions.append(FailurePrediction(
                    prediction_id=f"integration_{int(datetime.now().timestamp())}",
                    predicted_failure_type="CoordinationError",
                    probability=0.25,
                    triggering_conditions=["Complex execution plan", "Multiple agent interactions"],
                    potential_impact="Stage failures, context loss",
                    preventive_actions=["Validate stage dependencies", "Implement checkpoints"],
                    monitoring_indicators=["Stage success rates", "Context consistency"],
                    confidence_score=0.5
                ))
        
        return predictions
    
    def _filter_and_rank_predictions(
        self,
        predictions: List[FailurePrediction]
    ) -> List[FailurePrediction]:
        """Filter and rank predictions by relevance and confidence."""
        # Filter out low-confidence or low-probability predictions
        filtered = [
            p for p in predictions
            if p.confidence_score >= 0.4 and p.probability >= 0.2
        ]
        
        # Sort by combined score of probability and confidence
        sorted_predictions = sorted(
            filtered,
            key=lambda p: p.probability * p.confidence_score,
            reverse=True
        )
        
        return sorted_predictions[:5]  # Top 5 predictions


class AdvancedReplanningIntelligence:
    """Main orchestrator for advanced re-planning intelligence capabilities."""
    
    def __init__(self, web_fetcher=None, chroma_manager=None):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.research_engine = AutonomousResearchEngine(web_fetcher)
        self.pattern_analyzer = HistoricalPatternAnalyzer(chroma_manager)
        self.recovery_planner = MultiStepRecoveryPlanner(self.research_engine, self.pattern_analyzer)
        self.failure_predictor = FailurePredictionEngine(self.pattern_analyzer)
        
    async def create_intelligent_recovery_plan(
        self,
        failure_context: Dict[str, Any],
        project_context: Optional[Dict[str, Any]] = None
    ) -> RecoveryStrategy:
        """Create a comprehensive intelligent recovery plan."""
        self.logger.info(f"Creating intelligent recovery plan for: {failure_context.get('error_type', 'unknown')}")
        
        # Step 1: Conduct autonomous research
        research_query = self._create_research_query(failure_context, project_context)
        research_insights = await self.research_engine.conduct_research(
            research_query, project_context
        )
        
        # Step 2: Analyze historical patterns
        historical_patterns = await self.pattern_analyzer.analyze_failure_patterns(
            failure_context, project_context
        )
        
        # Step 3: Create recovery strategy
        recovery_strategy = await self.recovery_planner.create_recovery_strategy(
            failure_context, historical_patterns, research_insights, project_context
        )
        
        self.logger.info(f"Recovery plan created with {len(recovery_strategy.steps)} steps")
        return recovery_strategy
    
    async def predict_and_prevent_failures(
        self,
        current_context: Dict[str, Any],
        execution_plan: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Predict potential failures and suggest preventive measures."""
        self.logger.info("Performing failure prediction and prevention analysis")
        
        # Predict potential failures
        predictions = await self.failure_predictor.predict_potential_failures(
            current_context, execution_plan
        )
        
        # Generate preventive strategies for high-risk predictions
        preventive_strategies = []
        for prediction in predictions:
            if prediction.probability >= 0.3:  # High-risk predictions
                strategy = await self._create_preventive_strategy(prediction, current_context)
                if strategy:
                    preventive_strategies.append(strategy)
        
        return {
            'predictions': [p.to_dict() for p in predictions],
            'preventive_strategies': preventive_strategies,
            'overall_risk_score': self._calculate_overall_risk(predictions),
            'recommendations': self._generate_prevention_recommendations(predictions)
        }
    
    def _create_research_query(
        self,
        failure_context: Dict[str, Any],
        project_context: Optional[Dict[str, Any]]
    ) -> ResearchQuery:
        """Create a research query based on failure context."""
        error_type = failure_context.get('error_type', '')
        error_message = failure_context.get('message', '')
        
        # Determine research scope
        if 'import' in error_type.lower() or 'module' in error_type.lower():
            scope = ResearchScope.SPECIFIC_ERROR
        elif 'timeout' in error_type.lower() or 'performance' in error_type.lower():
            scope = ResearchScope.OPTIMIZATION_TECHNIQUES
        else:
            scope = ResearchScope.TROUBLESHOOTING
        
        # Build search terms
        search_terms = [error_type]
        if error_message:
            search_terms.extend(error_message.split()[:5])  # First 5 words
        
        # Add context keywords
        context_keywords = []
        if project_context:
            project_type = project_context.get('project_type', '')
            framework = project_context.get('framework', '')
            if project_type:
                context_keywords.append(project_type)
            if framework:
                context_keywords.append(framework)
        
        query_id = f"research_{hash(str(failure_context))}_{int(datetime.now().timestamp())}"
        
        return ResearchQuery(
            query_id=query_id,
            scope=scope,
            search_terms=search_terms,
            context_keywords=context_keywords,
            priority=1.0,
            expected_insight_types=["solution", "workaround", "best_practice"],
            max_sources=3
        )
    
    async def _create_preventive_strategy(
        self,
        prediction: FailurePrediction,
        context: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Create a preventive strategy for a failure prediction."""
        return {
            'prediction_id': prediction.prediction_id,
            'strategy_type': 'preventive',
            'actions': prediction.preventive_actions,
            'monitoring': prediction.monitoring_indicators,
            'priority': prediction.probability * prediction.confidence_score,
            'estimated_effort': 'low'
        }
    
    def _calculate_overall_risk(self, predictions: List[FailurePrediction]) -> float:
        """Calculate overall risk score based on predictions."""
        if not predictions:
            return 0.0
        
        # Weight by probability and confidence
        weighted_risks = [
            p.probability * p.confidence_score for p in predictions
        ]
        
        # Calculate aggregate risk (not simply additive)
        overall_risk = 1 - (1 - max(weighted_risks)) * 0.8  # Max risk with slight penalty
        return min(overall_risk, 1.0)
    
    def _generate_prevention_recommendations(
        self,
        predictions: List[FailurePrediction]
    ) -> List[str]:
        """Generate high-level prevention recommendations."""
        recommendations = []
        
        if predictions:
            recommendations.append("Implement proactive monitoring for identified risk factors")
            recommendations.append("Consider preventive actions for high-probability failures")
            
            # Add specific recommendations based on prediction types
            failure_types = [p.predicted_failure_type for p in predictions]
            if any('dependency' in ft.lower() for ft in failure_types):
                recommendations.append("Validate and freeze dependency versions")
            if any('environment' in ft.lower() for ft in failure_types):
                recommendations.append("Implement environment validation checks")
        
        return recommendations


# ============================================================================
# MCP Tool Functions
# ============================================================================

async def create_intelligent_recovery_plan(
    failure_context: Dict[str, Any],
    project_context: Optional[Dict[str, Any]] = None,
    enable_research: bool = True
) -> Dict[str, Any]:
    """
    MCP tool function for creating intelligent recovery plans.
    
    Args:
        failure_context: Context information about the failure
        project_context: Current project context
        enable_research: Whether to conduct autonomous research
        
    Returns:
        Dictionary containing the recovery strategy
    """
    try:
        replanning_system = AdvancedReplanningIntelligence()
        
        if not enable_research:
            replanning_system.research_engine = None
        
        recovery_strategy = await replanning_system.create_intelligent_recovery_plan(
            failure_context, project_context
        )
        
        return {
            'success': True,
            'recovery_strategy': recovery_strategy.to_dict(),
            'summary': f"Created {recovery_strategy.confidence.value} confidence recovery plan with {len(recovery_strategy.steps)} steps"
        }
        
    except Exception as e:
        logger.error(f"Error creating recovery plan: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'recovery_strategy': None
        }


async def predict_potential_failures(
    current_context: Dict[str, Any],
    execution_plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    MCP tool function for predicting potential failures.
    
    Args:
        current_context: Current execution context
        execution_plan: Optional execution plan to analyze
        
    Returns:
        Dictionary containing failure predictions and preventive strategies
    """
    try:
        replanning_system = AdvancedReplanningIntelligence()
        
        prediction_results = await replanning_system.predict_and_prevent_failures(
            current_context, execution_plan
        )
        
        return {
            'success': True,
            'prediction_results': prediction_results,
            'summary': f"Identified {len(prediction_results['predictions'])} potential failure scenarios"
        }
        
    except Exception as e:
        logger.error(f"Error predicting failures: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'prediction_results': None
        }


async def analyze_historical_patterns(
    failure_context: Dict[str, Any],
    project_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    MCP tool function for analyzing historical failure patterns.
    
    Args:
        failure_context: Context information about the failure
        project_context: Current project context
        
    Returns:
        Dictionary containing historical patterns and insights
    """
    try:
        pattern_analyzer = HistoricalPatternAnalyzer()
        
        patterns = await pattern_analyzer.analyze_failure_patterns(
            failure_context, project_context
        )
        
        return {
            'success': True,
            'patterns': [pattern.to_dict() for pattern in patterns],
            'summary': f"Found {len(patterns)} relevant historical patterns"
        }
        
    except Exception as e:
        logger.error(f"Error analyzing patterns: {e}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            'patterns': []
        } 