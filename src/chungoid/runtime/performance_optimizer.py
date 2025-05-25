"""
Performance Optimization Module for Enhanced Agent System

This module implements performance optimization strategies for the enhanced autonomous
agent orchestration system, following Systematic Implementation Protocol Phase 4
requirements and incorporating AI agent evaluation best practices.

Key Features:
- Agent resolution query optimization with caching
- Real-time performance monitoring and metrics
- Memory usage optimization
- Capability mapping cache management
- Temporal dynamics tracking for agent state management
"""

import asyncio
import time
import logging
import psutil
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import lru_cache
import weakref
import gc
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Real-time performance metrics for agent system evaluation."""
    
    # Agent Resolution Metrics
    resolution_time_avg: float = 0.0
    resolution_time_max: float = 0.0
    resolution_cache_hit_rate: float = 0.0
    resolution_success_rate: float = 0.0
    
    # Execution Metrics
    execution_time_avg: float = 0.0
    execution_time_max: float = 0.0
    concurrent_executions: int = 0
    
    # Memory Metrics
    memory_usage_mb: float = 0.0
    memory_peak_mb: float = 0.0
    gc_collections: int = 0
    
    # System Health Metrics
    cpu_usage_percent: float = 0.0
    active_agents: int = 0
    failed_resolutions: int = 0
    
    # Temporal Dynamics (following Galileo's recommendations)
    state_coherence_score: float = 1.0
    action_sequence_efficiency: float = 1.0
    
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CacheEntry:
    """Cache entry for agent resolution results."""
    
    result: Any
    timestamp: datetime
    access_count: int = 0
    hit_count: int = 0
    
    def is_expired(self, ttl_seconds: int = 300) -> bool:
        """Check if cache entry has expired (default 5 minutes)."""
        return (datetime.now() - self.timestamp).total_seconds() > ttl_seconds
    
    def access(self):
        """Record cache access."""
        self.access_count += 1
        self.hit_count += 1


class PerformanceMonitor:
    """
    Real-time performance monitoring system for autonomous agents.
    
    Implements continuous evaluation principles from Galileo's AI agent
    evaluation framework, focusing on temporal dynamics and multi-objective
    optimization challenges.
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: deque = deque(maxlen=max_history)
        self.resolution_times: deque = deque(maxlen=100)
        self.execution_times: deque = deque(maxlen=100)
        self.memory_samples: deque = deque(maxlen=50)
        
        # Performance thresholds (configurable)
        self.thresholds = {
            'resolution_time_warning': 0.2,  # 200ms
            'resolution_time_critical': 0.5,  # 500ms
            'memory_usage_warning': 500,  # 500MB
            'memory_usage_critical': 1000,  # 1GB
            'cache_hit_rate_warning': 0.7,  # 70%
            'success_rate_critical': 0.95  # 95%
        }
        
        self._start_time = time.time()
        self._last_gc_count = gc.get_count()
    
    def record_resolution_time(self, duration: float, success: bool = True):
        """Record agent resolution performance."""
        self.resolution_times.append(duration)
        
        # Check thresholds and log warnings
        if duration > self.thresholds['resolution_time_critical']:
            logger.warning(f"Critical resolution time: {duration:.3f}s")
        elif duration > self.thresholds['resolution_time_warning']:
            logger.info(f"Slow resolution time: {duration:.3f}s")
    
    def record_execution_time(self, duration: float):
        """Record agent execution performance."""
        self.execution_times.append(duration)
    
    def record_memory_usage(self):
        """Record current memory usage."""
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.memory_samples.append(memory_mb)
        
        if memory_mb > self.thresholds['memory_usage_critical']:
            logger.error(f"Critical memory usage: {memory_mb:.1f}MB")
            # Trigger garbage collection
            gc.collect()
        elif memory_mb > self.thresholds['memory_usage_warning']:
            logger.warning(f"High memory usage: {memory_mb:.1f}MB")
    
    def get_current_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics snapshot."""
        # Calculate averages
        resolution_avg = sum(self.resolution_times) / len(self.resolution_times) if self.resolution_times else 0.0
        resolution_max = max(self.resolution_times) if self.resolution_times else 0.0
        
        execution_avg = sum(self.execution_times) / len(self.execution_times) if self.execution_times else 0.0
        execution_max = max(self.execution_times) if self.execution_times else 0.0
        
        memory_current = self.memory_samples[-1] if self.memory_samples else 0.0
        memory_peak = max(self.memory_samples) if self.memory_samples else 0.0
        
        # GC metrics
        current_gc = sum(gc.get_count())
        gc_collections = current_gc - sum(self._last_gc_count)
        self._last_gc_count = gc.get_count()
        
        return PerformanceMetrics(
            resolution_time_avg=resolution_avg,
            resolution_time_max=resolution_max,
            execution_time_avg=execution_avg,
            execution_time_max=execution_max,
            memory_usage_mb=memory_current,
            memory_peak_mb=memory_peak,
            gc_collections=gc_collections,
            cpu_usage_percent=psutil.cpu_percent(),
        )
    
    def add_metrics_snapshot(self, metrics: PerformanceMetrics):
        """Add metrics snapshot to history."""
        self.metrics_history.append(metrics)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.metrics_history:
            return {"status": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-10:]  # Last 10 snapshots
        
        return {
            "summary": {
                "uptime_seconds": time.time() - self._start_time,
                "total_snapshots": len(self.metrics_history),
                "recent_snapshots": len(recent_metrics)
            },
            "resolution_performance": {
                "avg_time": sum(m.resolution_time_avg for m in recent_metrics) / len(recent_metrics),
                "max_time": max(m.resolution_time_max for m in recent_metrics),
                "avg_cache_hit_rate": sum(m.resolution_cache_hit_rate for m in recent_metrics) / len(recent_metrics)
            },
            "execution_performance": {
                "avg_time": sum(m.execution_time_avg for m in recent_metrics) / len(recent_metrics),
                "max_time": max(m.execution_time_max for m in recent_metrics),
                "max_concurrent": max(m.concurrent_executions for m in recent_metrics)
            },
            "resource_usage": {
                "current_memory_mb": recent_metrics[-1].memory_usage_mb,
                "peak_memory_mb": max(m.memory_peak_mb for m in recent_metrics),
                "avg_cpu_percent": sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
            },
            "health_indicators": {
                "all_thresholds_ok": self._check_all_thresholds(recent_metrics[-1]),
                "warnings": self._get_current_warnings(recent_metrics[-1])
            }
        }
    
    def _check_all_thresholds(self, metrics: PerformanceMetrics) -> bool:
        """Check if all performance thresholds are within acceptable ranges."""
        return (
            metrics.resolution_time_avg < self.thresholds['resolution_time_warning'] and
            metrics.memory_usage_mb < self.thresholds['memory_usage_warning'] and
            metrics.resolution_cache_hit_rate > self.thresholds['cache_hit_rate_warning']
        )
    
    def _get_current_warnings(self, metrics: PerformanceMetrics) -> List[str]:
        """Get list of current performance warnings."""
        warnings = []
        
        if metrics.resolution_time_avg > self.thresholds['resolution_time_warning']:
            warnings.append(f"Slow agent resolution: {metrics.resolution_time_avg:.3f}s")
        
        if metrics.memory_usage_mb > self.thresholds['memory_usage_warning']:
            warnings.append(f"High memory usage: {metrics.memory_usage_mb:.1f}MB")
        
        if metrics.resolution_cache_hit_rate < self.thresholds['cache_hit_rate_warning']:
            warnings.append(f"Low cache hit rate: {metrics.resolution_cache_hit_rate:.1%}")
        
        return warnings


class AgentResolutionCache:
    """
    High-performance cache for agent resolution results.
    
    Implements caching strategies to optimize agent resolution queries,
    reducing latency and improving system responsiveness.
    """
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 300):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, CacheEntry] = {}
        self._access_order: deque = deque()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'expired': 0
        }
    
    def _make_key(self, task_type: str, capabilities: List[str], prefer_autonomous: bool) -> str:
        """Create cache key from resolution parameters."""
        capabilities_str = ','.join(sorted(capabilities))
        return f"{task_type}:{capabilities_str}:{prefer_autonomous}"
    
    def get(self, task_type: str, capabilities: List[str], prefer_autonomous: bool = True) -> Optional[Any]:
        """Get cached resolution result."""
        key = self._make_key(task_type, capabilities, prefer_autonomous)
        
        if key in self._cache:
            entry = self._cache[key]
            
            # Check if expired
            if entry.is_expired(self.ttl_seconds):
                del self._cache[key]
                self._stats['expired'] += 1
                return None
            
            # Record hit and update access order
            entry.access()
            self._access_order.append(key)
            self._stats['hits'] += 1
            
            return entry.result
        
        self._stats['misses'] += 1
        return None
    
    def put(self, task_type: str, capabilities: List[str], prefer_autonomous: bool, result: Any):
        """Cache resolution result."""
        key = self._make_key(task_type, capabilities, prefer_autonomous)
        
        # Evict if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()
        
        # Store new entry
        self._cache[key] = CacheEntry(
            result=result,
            timestamp=datetime.now()
        )
        self._access_order.append(key)
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        while self._access_order:
            lru_key = self._access_order.popleft()
            if lru_key in self._cache:
                del self._cache[lru_key]
                self._stats['evictions'] += 1
                break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self._stats['hits'] + self._stats['misses']
        hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'size': len(self._cache),
            'max_size': self.max_size,
            'hit_rate': hit_rate,
            'hits': self._stats['hits'],
            'misses': self._stats['misses'],
            'evictions': self._stats['evictions'],
            'expired': self._stats['expired']
        }
    
    def clear(self):
        """Clear all cache entries."""
        self._cache.clear()
        self._access_order.clear()
        self._stats = {k: 0 for k in self._stats}


class CapabilityMappingOptimizer:
    """
    Optimizes capability mapping queries for enhanced agent resolution.
    
    Implements pre-computed mappings and efficient lookup structures
    to minimize resolution time for task-type to agent mappings.
    """
    
    def __init__(self):
        self._task_type_mappings: Dict[str, List[str]] = {}
        self._capability_index: Dict[str, List[str]] = defaultdict(list)
        self._agent_capabilities: Dict[str, List[str]] = {}
        self._last_update = datetime.now()
        self._update_lock = asyncio.Lock()
    
    async def initialize_mappings(self, agent_registry):
        """Initialize optimized capability mappings from agent registry."""
        async with self._update_lock:
            logger.info("Initializing capability mappings for performance optimization")
            
            # Clear existing mappings
            self._task_type_mappings.clear()
            self._capability_index.clear()
            self._agent_capabilities.clear()
            
            # Build optimized lookup structures
            for agent_info in agent_registry.list_all_agents():
                agent_id = agent_info.agent_id
                capabilities = getattr(agent_info, 'capabilities', [])
                
                # Store agent capabilities
                self._agent_capabilities[agent_id] = capabilities
                
                # Build reverse index: capability -> agents
                for capability in capabilities:
                    self._capability_index[capability].append(agent_id)
            
            # Build task type mappings
            self._build_task_type_mappings()
            
            self._last_update = datetime.now()
            logger.info(f"Capability mappings initialized: {len(self._task_type_mappings)} task types, "
                       f"{len(self._capability_index)} capabilities, {len(self._agent_capabilities)} agents")
    
    def _build_task_type_mappings(self):
        """Build optimized task type to capability mappings."""
        # Core task type vocabulary with optimized mappings
        task_mappings = {
            'requirements_analysis': ['requirements_analysis', 'stakeholder_analysis', 'documentation'],
            'architecture_design': ['architecture_design', 'system_planning', 'blueprint_generation'],
            'environment_setup': ['environment_setup', 'dependency_management', 'project_bootstrapping'],
            'dependency_management': ['dependency_analysis', 'package_management', 'conflict_resolution'],
            'code_generation': ['code_generation', 'implementation', 'module_creation'],
            'code_debugging': ['code_debugging', 'error_analysis', 'automated_fixes'],
            'test_generation': ['test_generation', 'validation', 'quality_assurance'],
            'quality_validation': ['review_protocol', 'quality_validation', 'architectural_review'],
            'risk_assessment': ['risk_assessment', 'deep_investigation', 'impact_analysis'],
            'documentation': ['documentation_generation', 'project_analysis', 'comprehensive_reporting'],
            'file_operations': ['file_operations', 'directory_management', 'filesystem_access'],
            'project_coordination': ['autonomous_coordination', 'quality_gates', 'refinement_orchestration']
        }
        
        self._task_type_mappings.update(task_mappings)
    
    @lru_cache(maxsize=256)
    def get_agents_for_capability(self, capability: str) -> Tuple[str, ...]:
        """Get agents that have a specific capability (cached)."""
        return tuple(self._capability_index.get(capability, []))
    
    @lru_cache(maxsize=128)
    def get_capabilities_for_task_type(self, task_type: str) -> Tuple[str, ...]:
        """Get required capabilities for a task type (cached)."""
        return tuple(self._task_type_mappings.get(task_type, []))
    
    def find_matching_agents(self, task_type: str, required_capabilities: List[str]) -> List[str]:
        """Find agents matching task type and capabilities with optimized lookup."""
        # Get base capabilities for task type
        task_capabilities = list(self.get_capabilities_for_task_type(task_type))
        
        # Combine with additional required capabilities
        all_capabilities = set(task_capabilities + required_capabilities)
        
        # Find agents that have all required capabilities
        matching_agents = []
        
        for agent_id, agent_capabilities in self._agent_capabilities.items():
            agent_cap_set = set(agent_capabilities)
            
            # Check if agent has all required capabilities
            if all_capabilities.issubset(agent_cap_set):
                matching_agents.append(agent_id)
        
        return matching_agents
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get capability mapping optimization statistics."""
        return {
            'task_types': len(self._task_type_mappings),
            'capabilities': len(self._capability_index),
            'agents': len(self._agent_capabilities),
            'last_update': self._last_update.isoformat(),
            'cache_info': {
                'capability_cache': self.get_agents_for_capability.cache_info()._asdict(),
                'task_type_cache': self.get_capabilities_for_task_type.cache_info()._asdict()
            }
        }


class EnhancedPerformanceOptimizer:
    """
    Main performance optimization coordinator for the enhanced agent system.
    
    Integrates caching, monitoring, and optimization strategies following
    Systematic Implementation Protocol Phase 4 requirements.
    """
    
    def __init__(self, agent_registry=None):
        self.monitor = PerformanceMonitor()
        self.cache = AgentResolutionCache()
        self.capability_optimizer = CapabilityMappingOptimizer()
        self.agent_registry = agent_registry
        
        # Performance optimization settings
        self.optimization_enabled = True
        self.monitoring_interval = 30  # seconds
        self._monitoring_task = None
        
        logger.info("Enhanced Performance Optimizer initialized")
    
    async def initialize(self, agent_registry=None):
        """Initialize performance optimization system."""
        if agent_registry:
            self.agent_registry = agent_registry
        
        if self.agent_registry:
            await self.capability_optimizer.initialize_mappings(self.agent_registry)
        
        # Start monitoring task
        if self.optimization_enabled:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Performance optimization system initialized and monitoring started")
    
    async def _monitoring_loop(self):
        """Continuous monitoring loop for performance metrics."""
        while self.optimization_enabled:
            try:
                # Record current memory usage
                self.monitor.record_memory_usage()
                
                # Get current metrics and add to history
                metrics = self.monitor.get_current_metrics()
                
                # Add cache statistics to metrics
                cache_stats = self.cache.get_stats()
                metrics.resolution_cache_hit_rate = cache_stats['hit_rate']
                
                self.monitor.add_metrics_snapshot(metrics)
                
                # Log performance warnings if any
                warnings = self.monitor._get_current_warnings(metrics)
                for warning in warnings:
                    logger.warning(f"Performance warning: {warning}")
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in performance monitoring loop: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def optimize_agent_resolution(self, task_type: str, required_capabilities: List[str], 
                                      prefer_autonomous: bool = True) -> Optional[Any]:
        """
        Optimized agent resolution with caching and performance monitoring.
        
        Implements multi-objective optimization balancing accuracy, efficiency,
        and autonomous capability as recommended by Galileo's evaluation framework.
        """
        start_time = time.time()
        
        try:
            # Try cache first
            cached_result = self.cache.get(task_type, required_capabilities, prefer_autonomous)
            if cached_result is not None:
                resolution_time = time.time() - start_time
                self.monitor.record_resolution_time(resolution_time, success=True)
                return cached_result
            
            # Perform optimized resolution
            matching_agents = self.capability_optimizer.find_matching_agents(
                task_type, required_capabilities
            )
            
            # Apply autonomous preference optimization
            if prefer_autonomous and matching_agents:
                # Filter for autonomous-capable agents first
                autonomous_agents = [
                    agent_id for agent_id in matching_agents
                    if self._is_autonomous_capable(agent_id)
                ]
                if autonomous_agents:
                    matching_agents = autonomous_agents
            
            # Select best agent (could implement more sophisticated selection logic)
            result = matching_agents[0] if matching_agents else None
            
            # Cache the result
            if result:
                self.cache.put(task_type, required_capabilities, prefer_autonomous, result)
            
            # Record performance metrics
            resolution_time = time.time() - start_time
            self.monitor.record_resolution_time(resolution_time, success=result is not None)
            
            return result
            
        except Exception as e:
            resolution_time = time.time() - start_time
            self.monitor.record_resolution_time(resolution_time, success=False)
            logger.error(f"Error in optimized agent resolution: {e}")
            return None
    
    def _is_autonomous_capable(self, agent_id: str) -> bool:
        """Check if agent is autonomous-capable (simplified check)."""
        # This would integrate with the actual agent registry
        # For now, assume agents with certain capabilities are autonomous
        agent_capabilities = self.capability_optimizer._agent_capabilities.get(agent_id, [])
        autonomous_indicators = [
            'requirements_analysis', 'architecture_design', 'environment_setup',
            'dependency_management', 'risk_assessment', 'quality_validation'
        ]
        return any(cap in autonomous_indicators for cap in agent_capabilities)
    
    def get_comprehensive_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance optimization report."""
        return {
            "performance_monitoring": self.monitor.get_performance_report(),
            "cache_performance": self.cache.get_stats(),
            "capability_optimization": self.capability_optimizer.get_optimization_stats(),
            "system_status": {
                "optimization_enabled": self.optimization_enabled,
                "monitoring_active": self._monitoring_task is not None and not self._monitoring_task.done(),
                "uptime_seconds": time.time() - self.monitor._start_time
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown performance optimization system."""
        self.optimization_enabled = False
        
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Performance optimization system shutdown complete")


# Global performance optimizer instance
_global_optimizer: Optional[EnhancedPerformanceOptimizer] = None


async def get_performance_optimizer() -> EnhancedPerformanceOptimizer:
    """Get or create global performance optimizer instance."""
    global _global_optimizer
    
    if _global_optimizer is None:
        _global_optimizer = EnhancedPerformanceOptimizer()
        await _global_optimizer.initialize()
    
    return _global_optimizer


async def optimize_agent_resolution(task_type: str, required_capabilities: List[str], 
                                   prefer_autonomous: bool = True) -> Optional[Any]:
    """Convenience function for optimized agent resolution."""
    optimizer = await get_performance_optimizer()
    return await optimizer.optimize_agent_resolution(task_type, required_capabilities, prefer_autonomous)


def get_performance_report() -> Dict[str, Any]:
    """Get current performance optimization report."""
    if _global_optimizer:
        return _global_optimizer.get_comprehensive_performance_report()
    return {"status": "Performance optimizer not initialized"} 