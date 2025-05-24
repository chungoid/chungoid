# Architecture Overview: Chungoid Autonomous Agentic System

*Last updated: 2025-05-23 by Claude (Post Phase 0 Analysis)*

## Table of Contents
1. System Architecture Overview
2. Core Components & Services
3. Agent Framework Architecture
4. State Management & Persistence
5. Tool Integration via MCP
6. Communication & Context Flow
7. Deployment & Operations Architecture

---

## 1. System Architecture Overview

### 1.1 High-Level Architecture
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Chungoid Core System                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                Agent Layer                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ MasterPlanner   │  │ CodeGenerator   │  │ TestRunner      │  │ Analysis   │ │
│  │ Agent           │  │ Agent           │  │ Agent           │  │ Agents     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                            Orchestration Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ AsyncOrchestra  │  │ Context         │  │ State           │  │ Error      │ │
│  │ tor             │  │ Resolution      │  │ Manager         │  │ Handling   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                               Service Layer                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ Smart           │  │ Project Type    │  │ State           │  │ Agent      │ │
│  │ Dependency      │  │ Detection       │  │ Persistence     │  │ Health     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                Tool Layer                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ ChromaDB        │  │ File System     │  │ Terminal        │  │ Web        │ │
│  │ MCP Suite       │  │ MCP Suite       │  │ MCP Suite       │  │ Content    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                           Infrastructure Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ ChromaDB        │  │ File System     │  │ Process         │  │ Network    │ │
│  │ Vector Store    │  │ Operations      │  │ Management      │  │ Services   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Architectural Principles
- **Event-Driven**: Asynchronous operations with reactive state management
- **Context-Aware**: Rich context propagation through all system layers
- **Tool-Composable**: MCP-based tool integration enabling complex workflows
- **State-Persistent**: Comprehensive state management with recovery capabilities
- **Learning-Enabled**: Built-in reflection and adaptation mechanisms

---

## 2. Core Components & Services

### 2.1 AsyncOrchestrator
**Purpose**: Central execution engine for coordinating agent workflows
**Key Features**:
- Master execution plan interpretation and execution
- Stage-by-stage workflow coordination with conditional branching
- Error handling and recovery orchestration
- Context management and propagation
- Integration with StateManager for comprehensive tracking

**Architecture**:
```python
class AsyncOrchestrator:
    def __init__(self, agent_provider, state_manager, metrics_store):
        self.context_resolver = ContextResolutionService()
        self.condition_evaluator = ConditionEvaluationService()
        self.success_criteria_evaluator = SuccessCriteriaService()
        self.error_handler_service = OrchestrationErrorHandlerService()
    
    async def run(self, goal_str=None, master_plan_obj=None, ...):
        # Flow start recording
        # Stage-by-stage execution with StateManager integration
        # Comprehensive error handling and recovery
        # Flow end recording with results
```

### 2.2 StateManager
**Purpose**: Comprehensive state persistence and reflection management
**Key Features**:
- ✅ **Active Integration**: Full integration with AsyncOrchestrator
- Flow and stage lifecycle tracking (`record_flow_start`, `record_stage_start`, `record_stage_end`, `record_flow_end`)
- Rich reflection storage in ChromaDB
- Context snapshotting and restoration
- Project state management with atomic updates

**Current Implementation Status**: ✅ FULLY IMPLEMENTED

### 2.3 Agent Provider System
**Purpose**: Dynamic agent instantiation and management
**Key Features**:
- Registry-based agent lookup and instantiation
- Shared context propagation to agent instances
- Agent capability discovery and metadata management
- Support for both sync and async agent patterns

### 2.4 Smart Services (Phase 1 Target)
**Planned Services**:
- **Smart Dependency Analysis**: AST-based import detection + LLM reasoning
- **Project Type Detection**: Multi-signal project analysis with confidence scoring
- **Agent Health Monitoring**: Performance tracking and optimization suggestions
- **Dynamic Error Classification**: Intelligent error categorization and recovery

---

## 3. Agent Framework Architecture

### 3.1 Agent Design Pattern
```python
class BaseAgent:
    """Base class for all autonomous agents"""
    
    async def invoke_async(self, inputs: AgentInputSchema, full_context: SharedContext) -> AgentOutputSchema:
        # 1. Input validation and context extraction
        # 2. Core reasoning and domain-specific logic
        # 3. Tool composition and external integrations
        # 4. Output generation with reflection logging
        # 5. Error handling with structured error details
```

### 3.2 Agent Categories
1. **Planning Agents**: `MasterPlannerAgent`, strategy and workflow planning
2. **Development Agents**: `SmartCodeGeneratorAgent`, code generation and modification
3. **Environment Agents**: `EnvironmentBootstrapAgent`, `DependencyManagementAgent`
4. **Testing Agents**: `SystemTestRunnerAgent`, `TestFailureAnalysisAgent`
5. **Analysis Agents**: Project analysis, code quality, performance monitoring

### 3.3 Agent Communication
- **Shared Context**: Rich context object passed between all agents
- **Output Chaining**: Agent outputs become inputs for subsequent stages
- **Reflection Integration**: All agent reasoning logged for learning and debugging
- **Error Propagation**: Structured error details with recovery suggestions

---

## 4. State Management & Persistence

### 4.1 StateManager Integration Points
```
AsyncOrchestrator Lifecycle:
┌─────────────────┐
│ Flow Start      │──→ record_flow_start(run_id, flow_id, context)
├─────────────────┤
│ Stage Loop:     │
│  ├─ Stage Start │──→ record_stage_start(run_id, flow_id, stage_id, agent_id)
│  ├─ Agent Exec  │──→ [Agent performs work with reflection logging]
│  └─ Stage End   │──→ record_stage_end(run_id, flow_id, stage_id, status, outputs)
├─────────────────┤
│ Flow End        │──→ record_flow_end(run_id, flow_id, final_status, outputs)
└─────────────────┘
```

### 4.2 ChromaDB Integration
**Collections**:
- `chungoid_context`: Project context and artifact storage
- `chungoid_reflections`: Agent reasoning and decision logging
- `dev_meta_chungoid_core_deep_embeddings`: Codebase analysis and semantic search

**Usage Patterns**:
- Semantic search for similar code patterns and solutions
- Historical reflection analysis for learning and optimization
- Context-aware tool and strategy recommendations

### 4.3 Project State Schema
```python
class ProjectStateV2:
    project_id: str
    project_name: Optional[str]
    initial_user_goal_summary: str
    overall_project_status: str
    schema_version: str
    last_updated: datetime
    run_history: Dict[str, Any]  # Detailed execution history
```

---

## 5. Tool Integration via MCP

### 5.1 MCP Tool Suite Architecture
**Design Philosophy**: Each tool suite provides a focused set of capabilities with intelligent composition support.

**Current Tool Suites**:
1. **ChromaDB MCP Suite**: Vector storage, semantic search, reflection querying
2. **File System MCP Suite**: Project-aware file operations with safety checks
3. **Terminal MCP Suite**: Secure command execution with sandboxing
4. **Content MCP Suite**: Dynamic content generation and caching

### 5.2 Tool Composition Patterns
```python
# Example: Multi-step dependency analysis and installation
async def smart_dependency_workflow():
    # 1. Project type detection
    project_info = await project_type_detection_tool(project_root)
    
    # 2. Smart dependency analysis  
    dependencies = await smart_dependency_analysis_tool(project_root, project_info)
    
    # 3. Dependency installation
    install_result = await terminal_tool(f"pip install -r {dependencies.requirements_file}")
    
    # 4. Verification
    verification = await test_runner_tool("pip check")
    
    return DependencyWorkflowResult(project_info, dependencies, install_result, verification)
```

### 5.3 Tool Manifest System
**Dynamic Tool Discovery**:
- Capability descriptions with usage patterns
- Success metrics and performance characteristics
- Tool composition recommendations
- Runtime tool availability checking

---

## 6. Communication & Context Flow

### 6.1 SharedContext Architecture
```python
class SharedContext:
    run_id: str
    flow_id: str
    data: Dict[str, Any]  # Project context, artifacts, intermediate results
    outputs: Dict[str, Any]  # Stage outputs and final results
    current_stage_id: Optional[str]
    current_stage_status: Optional[StageStatus]
    flow_has_warnings: bool = False
    
    def update_current_stage_output(self, output: Any):
        """Update current stage output and maintain history"""
    
    def add_stage_output_to_history(self, stage_name: str, output: Any):
        """Maintain complete stage execution history"""
```

### 6.2 Context Resolution Service
**Purpose**: Resolve context references and variable substitutions
**Key Features**:
- ✅ **Fixed**: SharedContext path resolution (`{context.data.project_id}`)
- Dynamic variable substitution with fallback mechanisms
- Type-safe context access with validation
- Integration with agent input schemas

### 6.3 Error Context Propagation
```python
class AgentErrorDetails:
    error_type: str
    message: str
    stage_id: str
    agent_id: str
    context_snapshot: Optional[Dict[str, Any]]
    suggested_recovery_actions: List[str]
    retry_strategy: Optional[str]
```

---

## 7. Deployment & Operations Architecture

### 7.1 Execution Environment
- **Python 3.13+**: Async/await support with modern language features
- **Virtual Environment**: Isolated Python environment per project
- **Process Sandboxing**: Secure execution of external commands and tools
- **Resource Management**: Memory and CPU monitoring with cleanup

### 7.2 Configuration Management
**Hierarchical Configuration**:
1. Environment variables (secrets, system-specific settings)
2. Project-level YAML configurations
3. Global system defaults
4. Runtime overrides

### 7.3 Monitoring & Observability
- **Execution Metrics**: Stage timing, success rates, resource usage
- **Reflection Analytics**: Agent reasoning quality and pattern analysis
- **Performance Monitoring**: System resource usage and optimization opportunities
- **Error Analytics**: Failure pattern recognition and recovery effectiveness

### 7.4 Data Management
**ChromaDB Storage**:
- Vector embeddings for semantic search and similarity analysis
- Structured metadata for filtering and querying
- Automatic cleanup and retention policies
- Backup and recovery procedures

---

## 8. Security & Safety

### 8.1 Execution Safety
- **Sandboxed Commands**: All terminal operations run in controlled environments
- **Input Validation**: Comprehensive validation of all user inputs and agent outputs
- **Output Sanitization**: Safe handling of generated code and system responses
- **Resource Limits**: CPU, memory, and execution time constraints

### 8.2 Data Security
- **Context Isolation**: Project contexts are isolated and secured
- **Sensitive Data Handling**: Secure management of credentials and secrets
- **Audit Trails**: Complete logging of all system operations and decisions
- **Access Controls**: Appropriate permissions for file and system operations

---

*This architecture overview provides the technical foundation for understanding the Chungoid autonomous agentic system. It serves as a reference for development, integration, and evolution of system components.* 