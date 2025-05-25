"""
Base Protocol Interface

Defines the structure and interface for all protocols in the chungoid system.
Enables agents to follow rigorous, systematic methodologies.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from enum import Enum

class PhaseStatus(Enum):
    """Status of a protocol phase."""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress" 
    VALIDATION_PENDING = "validation_pending"
    COMPLETED = "completed"
    FAILED = "failed"
    REQUIRES_RETRY = "requires_retry"

@dataclass
class ProtocolPhase:
    """Represents a single phase in a protocol."""
    
    name: str
    description: str
    time_box_hours: float
    required_outputs: List[str]
    validation_criteria: List[str]
    tools_required: List[str]
    dependencies: List[str] = None
    
    # Runtime state
    status: PhaseStatus = PhaseStatus.NOT_STARTED
    outputs: Dict[str, Any] = None
    validation_results: Dict[str, bool] = None
    execution_time: float = 0.0
    retry_count: int = 0
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.outputs is None:
            self.outputs = {}
        if self.validation_results is None:
            self.validation_results = {}

@dataclass
class ProtocolTemplate:
    """Template for creating protocol artifacts."""
    
    name: str
    description: str
    template_content: str
    variables: List[str] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = []

class ProtocolInterface(ABC):
    """Base interface for all protocols."""
    
    def __init__(self):
        self.phases: List[ProtocolPhase] = []
        self.templates: Dict[str, ProtocolTemplate] = {}
        self.current_phase_index: int = 0
        self.context: Dict[str, Any] = {}
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Protocol name."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Protocol description."""
        pass
    
    @property
    @abstractmethod
    def total_estimated_time(self) -> float:
        """Total estimated time in hours."""
        pass
    
    @abstractmethod
    def initialize_phases(self) -> List[ProtocolPhase]:
        """Initialize all protocol phases."""
        pass
    
    @abstractmethod
    def initialize_templates(self) -> Dict[str, ProtocolTemplate]:
        """Initialize protocol templates."""
        pass
    
    def setup(self, context: Dict[str, Any] = None):
        """Setup the protocol with initial context."""
        if context:
            self.context.update(context)
        
        self.phases = self.initialize_phases()
        self.templates = self.initialize_templates()
        self.current_phase_index = 0
    
    def get_current_phase(self) -> Optional[ProtocolPhase]:
        """Get the current active phase."""
        if self.current_phase_index >= len(self.phases):
            return None
        return self.phases[self.current_phase_index]
    
    def get_phase_by_name(self, name: str) -> Optional[ProtocolPhase]:
        """Get a phase by name."""
        for phase in self.phases:
            if phase.name == name:
                return phase
        return None
    
    def advance_to_next_phase(self) -> bool:
        """Advance to the next phase."""
        current_phase = self.get_current_phase()
        if current_phase and current_phase.status == PhaseStatus.COMPLETED:
            self.current_phase_index += 1
            return True
        return False
    
    def is_phase_ready(self, phase: ProtocolPhase) -> bool:
        """Check if a phase is ready to execute (dependencies met)."""
        for dep_name in phase.dependencies:
            dep_phase = self.get_phase_by_name(dep_name)
            if not dep_phase or dep_phase.status != PhaseStatus.COMPLETED:
                return False
        return True
    
    def get_template(self, name: str, **variables) -> str:
        """Get a template with variables substituted."""
        if name not in self.templates:
            raise ValueError(f"Template '{name}' not found")
        
        template = self.templates[name]
        content = template.template_content
        
        # Simple variable substitution
        for var_name, var_value in variables.items():
            placeholder = f"[{var_name}]"
            content = content.replace(placeholder, str(var_value))
        
        return content
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get a summary of protocol progress."""
        completed = sum(1 for p in self.phases if p.status == PhaseStatus.COMPLETED)
        failed = sum(1 for p in self.phases if p.status == PhaseStatus.FAILED)
        in_progress = sum(1 for p in self.phases if p.status == PhaseStatus.IN_PROGRESS)
        
        return {
            "total_phases": len(self.phases),
            "completed": completed,
            "failed": failed, 
            "in_progress": in_progress,
            "current_phase": self.get_current_phase().name if self.get_current_phase() else None,
            "completion_percentage": (completed / len(self.phases)) * 100 if self.phases else 0
        }
    
    def reset_protocol(self):
        """Reset protocol to initial state."""
        for phase in self.phases:
            phase.status = PhaseStatus.NOT_STARTED
            phase.outputs = {}
            phase.validation_results = {}
            phase.execution_time = 0.0
            phase.retry_count = 0
        
        self.current_phase_index = 0
        self.context = {} 