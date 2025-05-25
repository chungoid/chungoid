# Chungoid-Core Onboarding Guide

## Welcome to Chungoid-Core: The Protocol-Driven Autonomous Development Engine

Chungoid-core is the foundational autonomous agent system that transforms software development from template generation to systematic engineering through rigorous protocol-driven methodologies.

---

## What is Chungoid-Core?

**Core Mission**: Enable autonomous agents to follow proven methodologies for investigation, implementation, and quality assurance, transforming them from "single-shot executors" to "protocol-driven engineers."

**Key Innovation**: A comprehensive protocol system that ensures agents follow the same rigorous approaches that lead to successful outcomes, backed by systematic validation and iterative improvement.

---

## Architecture Overview

### Core Components

```
chungoid-core/
├── src/chungoid/
│   ├── protocols/              # 🆕 Protocol System (NEW)
│   │   ├── base/              # Protocol framework & interfaces
│   │   ├── investigation/     # Deep Investigation Protocol
│   │   ├── implementation/    # Deep Implementation Protocol  
│   │   ├── quality/          # Quality gates & validation
│   │   └── docs/             # Protocol documentation
│   ├── agents/               # Enhanced Protocol-Aware Agents
│   ├── runtime/              # Execution & orchestration engine
│   ├── schemas/              # Data models & validation
│   ├── utils/                # Utilities & helpers
│   └── mcp_tools/           # Tool integrations
```

### Revolutionary Protocol System

**Before Protocols:**
- Agents: Single LLM call → Store result → Move on
- Output: Placeholder code with TODOs
- Quality: Hope and manual fixes

**After Protocols:**
- Agents: Follow systematic phases → Tool validation → Iterative improvement
- Output: Working, tested, integrated systems  
- Quality: Built-in validation gates and continuous verification

---

## Getting Started

### 1. Environment Setup

```bash
# Navigate to chungoid-core
cd chungoid-core/

# Install dependencies
pip install -e .

# Verify installation
python -c "from chungoid.protocols import get_protocol; print('Protocols ready!')"
```

### 2. Understanding the Protocol System

#### Available Protocols

1. **Deep Investigation Protocol** - Systematic debugging and problem analysis
2. **Deep Implementation Protocol** - Architectural discovery and compatible feature implementation
3. **Quality Gate Protocol** - Validation and verification frameworks

#### Basic Protocol Usage

```python
from chungoid.protocols import get_protocol
from chungoid.agents.protocol_aware_agent import ArchitectureDiscoveryAgent

# Load a protocol
protocol = get_protocol("deep_implementation")

# Execute with an agent
agent = ArchitectureDiscoveryAgent(config)
result = agent.execute_with_protocol(
    task={"feature_name": "New Agent Type"}, 
    protocol_name="deep_implementation"
)

# Monitor progress
status = agent.get_protocol_status()
print(f"Progress: {status['progress_summary']['completion_percentage']}%")
```

### 3. Protocol-Enhanced Development Workflow

#### Research & Architecture Discovery
```python
# Phase 1: Comprehensive architecture understanding
research_agent = ArchitectureDiscoveryAgent(config)
architecture_analysis = research_agent.execute_with_protocol(
    {"feature_name": "Iterative Coding Agent"},
    "deep_implementation"
)

# Results in:
# - Complete component inventory
# - Pattern catalog with examples
# - Integration point mapping
# - Technology stack analysis
```

#### Implementation with Tool Validation  
```python
# Phase 2: Protocol-guided implementation
coding_agent = IterativeCodingAgent(config)

# Follow detailed implementation plan
implementation_plan = architecture_analysis["phases"][3]["outputs"]

# Iterative development with validation
for component in implementation_plan["components"]:
    code = coding_agent.generate_code(component)
    
    # Protocol enforces tool validation
    validation = coding_agent.validate_with_tools(code, component.quality_criteria)
    
    while not validation.passed:
        code = coding_agent.improve_code(code, validation.feedback)
        validation = coding_agent.validate_with_tools(code, component.quality_criteria)
```

---

## Core Concepts

### 1. Protocol-Aware Agents

All agents inherit from `ProtocolAwareAgent` which provides:

- **Systematic Execution**: Follow proven methodologies phase-by-phase
- **Validation Gates**: Quality checkpoints that prevent advancement until criteria are met
- **Tool Integration**: Built-in ability to use validation tools (compilers, testers, linters)
- **Iterative Improvement**: Retry and refine based on tool feedback
- **Template System**: Consistent artifact generation using protocol templates

### 2. Protocol Phases

Each protocol consists of systematic phases:

#### Deep Implementation Protocol Phases:
1. **Architecture Discovery** (3 hours) - Understand existing patterns
2. **Integration Analysis** (2 hours) - Map all touchpoints
3. **Compatibility Design** (2.5 hours) - Design seamless integration
4. **Implementation Planning** (1.5 hours) - Detailed step-by-step plan

### 3. Validation & Quality Gates

Protocols include automatic validation:
- **Required Outputs**: Each phase must produce specific deliverables
- **Validation Criteria**: Quality standards that must be met
- **Tool Requirements**: Specific tools that must be used for verification
- **Dependency Tracking**: Phases can only proceed when dependencies are complete

### 4. Tool-Driven Development

Agents can systematically use tools for:
- **Code Validation**: Compilers, syntax checkers
- **Quality Assurance**: Linters, formatters, complexity analyzers  
- **Testing**: Unit test runners, integration test frameworks
- **Build Systems**: Package managers, build tools, containerization

---

## Advanced Usage

### Creating Custom Protocols

```python
from chungoid.protocols.base.protocol_interface import ProtocolInterface, ProtocolPhase

class CustomProtocol(ProtocolInterface):
    @property
    def name(self) -> str:
        return "custom_protocol"
    
    def initialize_phases(self) -> List[ProtocolPhase]:
        return [
            ProtocolPhase(
                name="analysis",
                description="Custom analysis phase",
                time_box_hours=1.0,
                required_outputs=["analysis_document"],
                validation_criteria=["Analysis complete"],
                tools_required=["codebase_search", "read_file"]
            )
        ]
```

### Enhancing Existing Agents

```python
from chungoid.agents.protocol_aware_agent import ProtocolAwareAgent

class MyCustomAgent(ProtocolAwareAgent):
    def _execute_phase_logic(self, phase: ProtocolPhase) -> Dict[str, Any]:
        if phase.name == "analysis":
            # Use tools systematically
            if "codebase_search" in phase.tools_required:
                search_results = self.use_codebase_search("agent patterns")
            
            # Generate outputs using protocol templates
            analysis_doc = self.use_protocol_template(
                "analysis_template", 
                findings=search_results
            )
            
            return {"analysis_document": analysis_doc}
```

### Integration with Existing Chungoid Systems

The protocol system integrates seamlessly with existing chungoid infrastructure:

- **ChromaDB**: Artifacts automatically stored and retrievable
- **Context Management**: Protocol results flow through shared context
- **Success Criteria**: Enhanced validation beyond simple completion checks
- **Error Handling**: Protocol-aware error recovery and investigation
- **Monitoring**: Built-in progress tracking and metrics

---

## Key Documentation

### Protocol Documentation
- [`DEEP_INVESTIGATION_PROTOCOL.md`](src/chungoid/protocols/docs/DEEP_INVESTIGATION_PROTOCOL.md) - Systematic debugging methodology
- [`DEEP_PLANNING_PROTOCOL.md`](src/chungoid/protocols/docs/DEEP_PLANNING_PROTOCOL.md) - Architecture-compatible feature planning and design
- [`SYSTEMATIC_IMPLEMENTATION_PROTOCOL.md`](src/chungoid/protocols/docs/SYSTEMATIC_IMPLEMENTATION_PROTOCOL.md) - Converting blueprints to working, tested code
- [`PROTOCOL_OVERVIEW.md`](src/chungoid/protocols/docs/PROTOCOL_OVERVIEW.md) - Two-protocol system overview
- [`PROTOCOL_USAGE_GUIDE.md`](src/chungoid/protocols/docs/PROTOCOL_USAGE_GUIDE.md) - Complete usage examples and patterns

### Code References
- [`protocol_interface.py`](src/chungoid/protocols/base/protocol_interface.py) - Base protocol framework
- [`protocol_aware_agent.py`](src/chungoid/agents/protocol_aware_agent.py) - Enhanced agent base class
- [`deep_implementation.py`](src/chungoid/protocols/implementation/deep_implementation.py) - Implementation protocol

---

## Expected Outcomes

### Quality Transformation
- **Code Compilation Rate**: 95%+ (vs previous ~20%)
- **Functional Completeness**: Working end-to-end systems
- **Integration Success**: Seamless component compatibility
- **Goal Coverage**: 100% requirement traceability

### Process Transformation  
- **From Template Generation**: To systematic engineering
- **From Single-Shot**: To iterative improvement
- **From Assumption-Based**: To tool-validated development
- **From Isolated Agents**: To collaborative protocol coordination

### Agent Behavior Transformation
- **Systematic Investigation**: Follow proven debugging methodologies
- **Architectural Consistency**: New features feel native to existing systems
- **Iterative Refinement**: Continuous improvement until quality gates pass
- **Goal Accountability**: Complete traceability from requirements to implementation

---

## Implementation Roadmap

### Phase 1: Protocol Foundation COMPLETE
- [x] Protocol system architecture
- [x] Base protocol interfaces  
- [x] Deep Investigation Protocol
- [x] Deep Implementation Protocol
- [x] Protocol-aware agent base class

### Phase 2: Agent Enhancement (NEXT)
- [ ] Update existing agents to inherit from `ProtocolAwareAgent`
- [ ] Integrate tool validation capabilities
- [ ] Add quality gate enforcement
- [ ] Implement iterative improvement loops

### Phase 3: Tool Integration (NEXT)  
- [ ] Compiler/syntax validation tools
- [ ] Test framework integration  
- [ ] Code quality tool integration
- [ ] Build system automation

### Phase 4: Advanced Protocols
- [ ] Goal Accountability Protocol
- [ ] Integration Testing Protocol
- [ ] Performance Optimization Protocol
- [ ] Security Validation Protocol

---

## Contributing

### Development Guidelines

1. **Follow Existing Patterns**: Use `codebase_search` to understand current patterns before implementing
2. **Protocol-Driven Development**: Use the Deep Implementation Protocol for new features
3. **Quality Gates**: Ensure all code passes validation before submission
4. **Documentation**: Update protocol documentation for any new methodologies

### Code Quality Standards

- **Type Hints**: All functions must have complete type annotations
- **Documentation**: Comprehensive docstrings for all public interfaces
- **Testing**: Protocol-validated test coverage for all new functionality
- **Error Handling**: Consistent error patterns following existing conventions

### Testing with Protocols

```python
# Example: Testing protocol execution
def test_deep_planning_protocol():
    agent = ArchitectureDiscoveryAgent(test_config)
    result = agent.execute_with_protocol(
        {"feature_name": "Test Feature"},
        "deep_planning"
    )
    
    assert result["overall_success"] == True
    assert len(result["phases"]) == 4
    assert all(phase["success"] for phase in result["phases"])
```