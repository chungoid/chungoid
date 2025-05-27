# Autonomous System Architecture Diagram

This document provides a comprehensive visual overview of Chungoid's current autonomous development system, showing the actual agents, MCP tools, and workflows as implemented in the codebase.

```mermaid
graph TD
    %% Style Definitions
    classDef userInput fill:#e1f5fe,stroke:#0277bd,stroke-width:2px,color:#000;
    classDef cliCommand fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px,color:#000;
    classDef coreInfra fill:#fff3e0,stroke:#f57c00,stroke-width:2px,color:#000;
    classDef orchestrator fill:#e8f5e8,stroke:#388e3c,stroke-width:3px,color:#000;
    classDef agent fill:#e3f2fd,stroke:#1976d2,stroke-width:2px,color:#000;
    classDef mcpTools fill:#fce4ec,stroke:#c2185b,stroke-width:2px,color:#000;
    classDef datastore fill:#f1f8e9,stroke:#689f38,stroke-width:2px,color:#000;
    classDef output fill:#fff8e1,stroke:#ffa000,stroke-width:2px,color:#000;

    %% User Input
    UserGoal["User Goal File\n(goal.txt)"]:::userInput
    CLICommand["chungoid build\n--goal-file goal.txt\n--project-dir ."]:::cliCommand
    
    %% Core Infrastructure
    subgraph CoreInfraGraph ["Chungoid Core Infrastructure"]
        CLI["CLI Handler\n(cli.py)"]:::coreInfra
        Engine["ChungoidEngine (Conceptual)\n(engine.py - interacts with UnifiedOrchestrator)"]:::coreInfra
        StateManager["StateManager\n(.chungoid/project_status.json)"]:::coreInfra
        AgentResolver["UnifiedAgentResolver\n(unified_agent_resolver.py)"]:::coreInfra
        PromptManager["PromptManager"]:::coreInfra
        LLMProvider["LLMProvider"]:::coreInfra
    end

    %% Central Orchestrator
    UnifiedOrch["UnifiedOrchestrator\n(Primary Execution Engine)"]:::orchestrator

    %% ChromaDB Context & Learning
    ChromaDB["ChromaDB Instance\n(Project Context & Reflections)"]:::datastore

    %% Autonomous Agents (UnifiedAgent Subclasses)
    subgraph AutonomousAgents ["Autonomous Agents (UnifiedAgent Subclasses)"]
        ARCA["AutomatedRefinementCoordinatorAgent_v1 (ARCA)"]:::agent
        ArchitectAgent["ArchitectAgent_v1"]:::agent
        SmartCodeAgent["SmartCodeGeneratorAgent_v1"]:::agent
        EnvBootstrapAgent["EnvironmentBootstrapAgent_v1"]:::agent
        DocAgent["ProjectDocumentationAgent_v1"]:::agent
        OtherAgents["Other Specialized Agents..."]:::agent
    end

    %% MCP Tool Ecosystem (11 Tools)
    subgraph MCPToolsGraph ["MCP Tool Ecosystem (11 Tools)"]
        ChromaTools["ChromaDB Suite (4 tools)\n• chroma_list_collections\n• chroma_create_collection\n• chromadb_query_collection\n• chromadb_reflection_query"]:::mcpTools
        
        FSTools["Filesystem Suite (3 tools)\n• filesystem_read_file\n• filesystem_project_scan\n• filesystem_batch_operations"]:::mcpTools
        
        TerminalTools["Terminal Suite (2 tools)\n• tool_run_terminal_command\n• terminal_classify_command"]:::mcpTools
        
        ContentTools["Content Suite (2 tools)\n• tool_fetch_web_content\n• mcptool_get_named_content"]:::mcpTools
    end

    %% Project Output
    CompleteProject["Complete Working Project\n• Source code\n• Tests\n• Documentation\n• Dependencies\n• Deployment configs"]:::output

    %% Main Flow Connections
    UserGoal --> CLICommand
    CLICommand --> CLI
    CLI --> Engine
    Engine --> UnifiedOrch
    
    %% Core Infrastructure Connections
    Engine --> StateManager
    Engine --> AgentResolver
    Engine --> PromptManager
    Engine --> LLMProvider
    
    %% Orchestrator to Agent Communication
    UnifiedOrch --> AgentResolver
    AgentResolver --> AutonomousAgents
    
    %% Agent to Tool Communication
    AutonomousAgents --> MCPToolsGraph
    
    %% ChromaDB Integration
    AutonomousAgents --> ChromaDB % Agents can use ChromaDB directly or via tools
    ChromaTools --> ChromaDB
    StateManager --> ChromaDB % For reflections, context
    
    %% State Management
    UnifiedOrch --> StateManager
    
    %% Final Output
    UnifiedOrch --> CompleteProject
    
    %% Execution Plan Flow (Conceptual - MasterExecutionPlan)
    AutonomousAgents -.->|"May generate or use MasterExecutionPlan"| UnifiedOrch
    
    %% Error Handling Flow (Conceptual - OrchestrationErrorHandlerService)
    UnifiedOrch -.->|"On Error, may use ErrorHandlerService"| UnifiedOrch
```

## System Components Overview

### 🎯 **Primary Workflow**
1. **User Input**: `chungoid build --goal-file goal.txt --project-dir .`
2. **Plan Creation**: An initial plan (e.g., a `MasterExecutionPlan`) is determined or generated based on the goal.
3. **Autonomous Execution**: `UnifiedOrchestrator` coordinates specialized agents (subclasses of `UnifiedAgent`).
4. **Tool Integration**: Agents use **11 MCP tools** for actual work, alongside direct SDK/library usage.
5. **State Persistence**: All progress tracked in ChromaDB (for reflections, context) and `.chungoid/project_status.json`.
6. **Complete Project**: Working, tested, documented software delivered.

### 🤖 **Agent Ecosystem**
Chungoid employs a suite of specialized autonomous agents, all inheriting from `UnifiedAgent`. Key examples include:
- **`AutomatedRefinementCoordinatorAgent_v1` (ARCA)**: Orchestrates iterative refinement of artifacts.
- **`ArchitectAgent_v1`**: Handles system architecture and design.
- **`SmartCodeGeneratorAgent_v1`**: Generates code with quality validation.
- **`EnvironmentBootstrapAgent_v1`**: Manages project setup and environment configuration.
- **`ProjectDocumentationAgent_v1`**: Generates and updates project documentation.
- ... and other specialized agents for tasks like dependency management, debugging, requirements analysis, etc.
(Note: Older agent names like `MasterPlannerAgent`, `CoreCodeGeneratorAgent_v1`, and `ProjectChromaManagerAgent_v1` are deprecated or refactored.)

### 🛠️ **MCP Tool Suites (11 Tools)**

#### **ChromaDB Suite (4 tools)**
- `chroma_list_collections`, `chroma_create_collection`
- `chromadb_query_collection`, `chromadb_reflection_query`

#### **Filesystem Suite (3 tools)**
- `filesystem_read_file`, `filesystem_project_scan`, `filesystem_batch_operations`

#### **Terminal Suite (2 tools)**
- `tool_run_terminal_command`, `terminal_classify_command`

#### **Content Suite (2 tools)**
- `tool_fetch_web_content`, `mcptool_get_named_content`

### 🔄 **Execution Flow**

1. **Goal Processing**: User provides natural language goal.
2. **Dynamic Orchestration**: `UnifiedOrchestrator` executes plan stages using appropriate `UnifiedAgent` subclasses.
3. **Tool Integration**: Agents invoke MCP tools or other libraries for implementation work.
4. **State Management**: Progress tracked in `.chungoid/project_status.json` (via `StateManager`) and ChromaDB.
5. **Error Recovery**: The `UnifiedOrchestrator` and individual agents have error handling; a dedicated `OrchestrationErrorHandlerService` may assist.
6. **Continuous Learning**: ChromaDB stores reflections and context, enabling agents to learn from past executions (a design goal).

### 📊 **Data & State Management**

- **Primary State**: `.chungoid/project_status.json` (project execution state, managed by `StateManager`).
- **Context & Learning**: ChromaDB collections for project context, agent reflections, and historical learning.
- **Tool Discovery**: Dynamic tool manifest system (`tool_manifest.py`, `manifest_initialization.py`) for MCP tools.
- **Execution Plans**: YAML-based `MasterExecutionPlan` for structured workflow definition, interpreted by `UnifiedOrchestrator`.

### 🚀 **Key Capabilities**

- **Autonomous Development**: Aiming for end-to-end project generation with minimal human intervention.
- **Multi-Language Support**: Capable of supporting various programming languages.
- **Intelligent Recovery**: Built-in error handling and retry mechanisms.
- **Context Awareness**: Agents operate with access to shared project context.
- **Tool Composition**: MCP tools can be used by agents in flexible ways.
- **Continuous Learning**: Design goal to improve performance based on historical execution data via ChromaDB reflections.

This autonomous system represents a complete paradigm shift from traditional development tools to AI-driven, context-aware autonomous development that learns and improves with each project. 