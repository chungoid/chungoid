"""
Observability Protocols Module

Week 5 Implementation: Architecture Visualization & Observability

This module contains protocols for architecture visualization and drift detection:
- AutonomousArchitectureVisualizerProtocol: Living architecture documentation with C4 model generation
- ArchitectureDriftDetectorProtocol: Real-time architecture drift detection and recommendations

These protocols provide comprehensive observability into the autonomous execution system,
enabling real-time monitoring, visualization, and proactive drift management.
"""

from .autonomous_architecture_visualizer import (
    AutonomousArchitectureVisualizerProtocol,
    DiagramLevel,
    DiagramFormat,
    ArchitectureElementType,
    ArchitectureElement,
    C4ModelDiagram,
    ArchitectureSnapshot,
    VisualizationConfig
)

from .architecture_drift_detector import (
    ArchitectureDriftDetectorProtocol,
    DriftSeverity,
    DriftType,
    DriftCause,
    ArchitectureBaseline,
    DriftMetric,
    DriftAnalysis,
    DriftRecommendation,
    DriftAlert
)

__all__ = [
    # Architecture Visualizer Protocol
    "AutonomousArchitectureVisualizerProtocol",
    "DiagramLevel",
    "DiagramFormat", 
    "ArchitectureElementType",
    "ArchitectureElement",
    "C4ModelDiagram",
    "ArchitectureSnapshot",
    "VisualizationConfig",
    
    # Architecture Drift Detector Protocol
    "ArchitectureDriftDetectorProtocol",
    "DriftSeverity",
    "DriftType",
    "DriftCause",
    "ArchitectureBaseline",
    "DriftMetric",
    "DriftAnalysis",
    "DriftRecommendation",
    "DriftAlert"
] 