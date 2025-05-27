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
│  │ Automated       │  │ SmartCode       │  │ CodeDebugging   │  │ Other AE   │ │
│  │ Refinement (ARCA)│  │ Generator Agent │  │ Agent           │  │ Agents     │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                            Orchestration Layer                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ Unified         │  │ UnifiedAgent    │  │ StateManager    │  │ Orchestration│ │
│  │ Orchestrator    │  │ Resolver        │  │                 │  │ ErrorHandler │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                               Service Layer                                    │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ LLMProvider     │  │ PromptManager   │  │ AgentRegistry   │  │ MetricsStore│ │
│  │                 │  │                 │  │                 │  │            │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────────────┘ │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                Tool Layer                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐ │
│  │ ChromaDB        │  │ File System     │  │ Terminal        │  │ Content    │ │
│  │ MCP Suite (4)   │  │ MCP Suite (3)   │  │ MCP Suite (2)   │  │ MCP Suite (2)│ │
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

### 2.1 UnifiedOrchestrator
**Location**: `chungoid.runtime.unified_orchestrator.UnifiedOrchestrator`
**Purpose**: Central execution engine for coordinating agent workflows based on `MasterExecutionPlan`. It has replaced the older `AsyncOrchestrator`.
**Key Features**:
- Interprets and executes `MasterExecutionPlan` stage by stage.
- Utilizes `UnifiedAgentResolver` to get agent instances.
- Interacts with `StateManager` for lifecycle tracking and persistence.
- Manages a `shared_context` dictionary for data flow between stages.
- Collaborates with `OrchestrationErrorHandlerService` for robust error handling.
- Employs `MetricsStore` for performance tracking.

**Architecture Snippet (Conceptual from `UnifiedOrchestrator.__init__`)**:
```python
class UnifiedOrchestrator:
    def __init__(
        self,
        config: Dict[str, Any],
        state_manager: StateManager,
        agent_resolver: UnifiedAgentResolver,
        metrics_store: MetricsStore,
        llm_provider: Optional[LLMProvider] = None
    ):
        self.config = config
        self.state_manager = state_manager
        self.agent_resolver = agent_resolver
        self.metrics_store = metrics_store
        self.llm_provider = llm_provider
        self.shared_context: Dict[str, Any] = {{}} # Initialized
        # ...
    
    async def execute_master_plan_async(self, master_plan: MasterExecutionPlan, ...):
        # Executes stages sequentially
        # Calls agent.execute() for each stage
        pass

    async def run(self, goal_str: Optional[str] = None, ...):
        # Main entry point, potentially invoking master planner
        pass
```

### 2.2 StateManager
**Location**: `chungoid.utils.state_manager.StateManager`
**Purpose**: Comprehensive state persistence, project status tracking, and reflection management.
**Key Features**:
- ✅ **Active Integration**: Fully integrated with `UnifiedOrchestrator`.
- Manages `project_status.json` (defined by `ProjectStateV2` schema) with file locking.
- Tracks flow and stage lifecycle (`record_flow_start`, `record_stage_start`, etc.).
- Persists agent reflections and project context to ChromaDB (collections: `chungoid_context`, `chungoid_reflections`).
- Supports saving/loading of paused flow states (`PausedRunDetails`).

**Current Implementation Status**: ✅ FULLY IMPLEMENTED

### 2.3 UnifiedAgentResolver (Agent Provider System)
**Location**: `chungoid.runtime.unified_agent_resolver.UnifiedAgentResolver`
**Purpose**: Dynamic agent instantiation and management.
**Key Features**:
- Uses `AgentRegistry` for agent class lookup by `agent_id`.
- Instantiates `UnifiedAgent` subclasses.
- Passes shared dependencies like `LLMProvider` and `PromptManager` to agents.
- Handles agent-specific configurations (e.g., refinement capabilities).

### 2.4 OrchestrationErrorHandlerService
**Location**: `chungoid.runtime.services.orchestration_error_handler_service.OrchestrationErrorHandlerService`
**Purpose**: Handles errors encountered during stage execution within the `UnifiedOrchestrator`.
**Key Features**:
- Standardizes errors into `AgentErrorDetails`.
- Manages retry logic based on stage configuration and defaults.
- Can invoke a reviewer agent (e.g., `MasterPlannerReviewerAgent`) for complex failures.
- Interacts with `StateManager` and `MetricsStore`.

### 2.5 Smart Services (Phase 1 Target)
**Planned Services**:
- **Smart Dependency Analysis**: AST-based import detection + LLM reasoning
- **Project Type Detection**: Multi-signal project analysis with confidence scoring
- **Agent Health Monitoring**: Performance tracking and optimization suggestions
- **Dynamic Error Classification**: Intelligent error categorization and recovery

---

## 3. Agent Framework Architecture

### 3.1 UnifiedAgent (Base Class)
**Location**: `chungoid.agents.unified_agent.UnifiedAgent`
**Purpose**: Single, universal base class for all autonomous agents in Chungoid.
**Key Features**:
- Defines a universal `execute(context: ExecutionContext, execution_mode: ExecutionMode)` method.
- Supports single-pass, multi-iteration, and refinement-enhanced execution modes.
- Integrates `LLMProvider` and `PromptManager`.
- Child agents must define `AGENT_ID`, `AGENT_VERSION`, `PRIMARY_PROTOCOLS`, `CAPABILITIES`.
- Can utilize MCP tools and ChromaDB for refinement if `enable_refinement` is true.

**Conceptual Structure**:
```python
class UnifiedAgent(BaseModel, ABC):
    AGENT_ID: ClassVar[str]
    AGENT_VERSION: ClassVar[str]
    # ... other classvars

    llm_provider: LLMProvider
    prompt_manager: PromptManager
    enable_refinement: bool = False
    # ...

    async def execute(
        self, 
        context: ExecutionContext,
        execution_mode: ExecutionMode = ExecutionMode.OPTIMAL
    ) -> AgentExecutionResult:
        # 1. Determine optimal execution strategy (if mode is OPTIMAL)
        # 2. Loop through iterations if multi-iteration mode:
        #    a. Enhance context with refinement data (if enabled and iteration > 0)
        #    b. Call _execute_iteration(current_context, iteration) for core agent logic
        #    c. Store iteration output (if refinement enabled)
        #    d. Assess completion and quality
        #    e. Enhance context for next iteration
        # 3. Return final AgentExecutionResult
        pass

    async def _execute_iteration(
        self, 
        context: ExecutionContext, 
        iteration: int
    ) -> IterationResult:
        # Abstract method: Implemented by specific agent subclasses
        # Contains the core logic for a single pass/iteration of the agent
        raise NotImplementedError
```

### 3.2 Agent Categories & Examples
(Based on `system_overview.md` and actual agents in `chungoid-core/src/chungoid/agents/autonomous_engine/`)

1.  **Orchestration & Coordination**:
    *   `AutomatedRefinementCoordinatorAgent_v1` (ARCA)
2.  **Planning & Design**:
    *   `ArchitectAgent`
    *   `ProductAnalystAgent`
3.  **Development & Implementation**:
    *   `SmartCodeGeneratorAgent`
    *   `EnvironmentBootstrapAgent`
    *   `DependencyManagementAgent`
4.  **Review & Analysis**:
    *   `BlueprintReviewerAgent`
    *   `RequirementsTracerAgent`
    *   `ProactiveRiskAssessorAgent`
5.  **Debugging & Documentation**:
    *   `CodeDebuggingAgent`
    *   `ProjectDocumentationAgent`
6.  **System Agents**:
    *   `NoOpAgent`

### 3.3 Agent Communication
- **Shared Context (`UnifiedOrchestrator.shared_context`)**: A dictionary passed and updated through stages, accessible to agents via `ExecutionContext.shared_context`.

### 4.2 ChromaDB Integration
**Collections managed by `StateManager`**:
- `chungoid_context`: Stores project context artifacts linked to specific stages or runs.
- `chungoid_reflections`: Persists agent reasoning, decision logs, and learning data.

**Collection for Codebase Analysis (managed externally, e.g., by `dev/scripts/embed_chungoid_core_for_meta_analysis.py`)**:
- `dev_meta_chungoid_core_deep_embeddings`: Contains fine-grained, embedded chunks of the `chungoid-core/` source code (modules, classes, functions) with detailed metadata. This collection is primarily for agents performing meta-analysis or seeking to understand the core system's implementation details.

**Usage Patterns**:
- Semantic search for similar code patterns and solutions
- Historical reflection analysis for learning and optimization
- Context-aware tool and strategy recommendations

### 4.3 Project State Schema
The primary state of a project is captured by the `ProjectStateV2` schema, managed by `StateManager` in `project_status.json`.

**`ProjectStateV2` (from `chungoid.schemas.project_state.ProjectStateV2`)**:
```python
class ProjectStateV2(BaseModel):
    project_id: str
    project_name: Optional[str]
    overall_project_status: str # e.g., "initializing", "cycle_in_progress"
    current_cycle_id: Optional[str]
    cycle_history: List[CycleHistoryItem] # Detailed history of each cycle
    run_history: Dict[str, RunRecord]   # Detailed history of each execution run
    master_loprd_id: Optional[str]
    master_blueprint_id: Optional[str]
    master_execution_plan_id: Optional[str]
    link_to_live_codebase_collection_snapshot: Optional[str]
    schema_version: str = Field(default="2.0")
    last_updated: Optional[datetime]

# Supporting schemas like CycleHistoryItem, RunRecord, StageRecord provide further detail.
# class RunRecord(BaseModel):
#     run_id: str
#     flow_id: str
#     start_time: datetime
#     # ... other fields ...
#     stages: List[StageRecord]

# class StageRecord(BaseModel):
#     stage_id: str
#     agent_id: str
#     # ... other fields ...
```

---

## 6. Communication & Context Flow

### 6.1 SharedContext Architecture
The `SharedContext` is a core concept for data exchange between stages in the `UnifiedOrchestrator`. It's not a rigid Pydantic model but rather a flexible dictionary (`Dict[str, Any]`) managed by the `UnifiedOrchestrator`.

**Key characteristics**:
-   **Initialization**: `self.shared_context = {}` in `UnifiedOrchestrator`.
-   **Data Propagation**: Agents receive relevant parts of this shared context via their `ExecutionContext.shared_context`.
-   **Output Storage**: Outputs from each stage are typically stored back into `shared_context['outputs'][stage_id]`.
-   **Dynamic Structure**: Can hold various data types, project artifacts, intermediate results, and references needed across the workflow.

**Conceptual Usage (within `UnifiedOrchestrator`)**:
```python
# In UnifiedOrchestrator:
self.shared_context: Dict[str, Any] = {
    "project_root_path": "...",
    "outputs": {}, # Stores outputs from each stage
    "master_plan_id": "...",
    "run_id": "..."
    # Other dynamic data as needed by the flow
}

# When executing a stage:
ctx = ExecutionContext(
    inputs=stage_inputs,
    shared_context=self.shared_context, # Passed to the agent
    # ...
)
result = await agent.execute(ctx, ...)

# After stage execution:
self.shared_context["outputs"][stage_id] = result.output
```

### 6.2 Context Resolution Service
(Note: The `UnifiedOrchestrator` does not directly use a distinct `ContextResolutionService` class as previously conceptualized. Context creation and input resolution for stages are handled within the orchestrator logic itself, leveraging the `shared_context` dictionary and `MasterExecutionPlan` specifications.)

The primary mechanism for context is the `shared_context` dictionary within the `UnifiedOrchestrator`. Inputs for each stage are typically defined in the `MasterExecutionPlan` and can reference values from this `shared_context` (e.g., outputs of previous stages). The orchestrator resolves these references before invoking an agent for a particular stage.

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