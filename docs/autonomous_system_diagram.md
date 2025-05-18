# Autonomous System Architecture Overview

This document provides a visual overview of the autonomous project generation system, as detailed in `dev/planning/lifecycle.md`. It illustrates the major phases, key components, agents, artifacts, and their interactions.

```mermaid
graph TD
    %% Style Definitions
    classDef phase fill:#f9f,stroke:#333,stroke-width:2px,color:#333;
    classDef coreComp fill:#lightgrey,stroke:#333,stroke-width:2px;
    classDef newAgent fill:#ccf,stroke:#333,stroke-width:2px;
    classDef artifact fill:#fff,stroke:#333,stroke-width:1px,color:#333;
    classDef datastore fill:#cfc,stroke:#333,stroke-width:2px;

    %% Central Data Stores & Core Orchestration
    subgraph CoreInfrastructure [Chungoid Core Infrastructure]
        direction LR
        FlowExecutor[Flow Executor]:::coreComp
        AgentRegistry[Agent Registry]:::coreComp
        StateManager["State Manager (.json)"]:::coreComp
        ChromaDB["ChromaDB Project Context Repository"]:::datastore
        PromptManager["Prompt Manager"]:::coreComp
    end

    ChromaDB --> AgentRegistry
    StateManager --> FlowExecutor
    AgentRegistry --> FlowExecutor
    PromptManager --> FlowExecutor


    %% Input
    UserInput["User Goal Request (.txt/.json)"]:::artifact

    %% Phase 1: Goal Understanding & Architectural Design
    subgraph Phase1 ["Phase 1: Goal Understanding & Design"]
        direction TB
        P1_ArchitectAgent["ArchitectAgent (New)"]:::newAgent
        P1_RefinedGoal["refined_user_goal.md"]:::artifact
        P1_Assumptions["assumptions_and_ambiguities.md"]:::artifact
        P1_TechRationale["technology_rationale.md"]:::artifact
        P1_Blueprint["ProjectBlueprint.md"]:::artifact
    end
    class Phase1 phase;
    UserInput --> P1_ArchitectAgent;
    P1_ArchitectAgent --> P1_RefinedGoal;
    P1_ArchitectAgent --> P1_Assumptions;
    P1_ArchitectAgent --> P1_TechRationale;
    P1_ArchitectAgent --> P1_Blueprint;
    P1_RefinedGoal --> ChromaDB;
    P1_Assumptions --> ChromaDB;
    P1_TechRationale --> ChromaDB;
    P1_Blueprint --> ChromaDB;

    %% Phase 2: Detailed Planning & Workflow Generation
    subgraph Phase2 ["Phase 2: Detailed Planning"]
        direction TB
        P2_BlueprintReviewer["BlueprintReviewerAgent (New)"]:::newAgent
        P2_BlueprintToFlow["BlueprintToFlowAgent (New)"]:::newAgent
        P2_MasterPlan["MasterExecutionPlan.yaml"]:::artifact
    end
    class Phase2 phase;
    P1_Blueprint --> P2_BlueprintReviewer;
    %% Refinement loop implied
    P2_BlueprintReviewer --> P1_Blueprint;
    P1_Blueprint --> P2_BlueprintToFlow;
    P2_BlueprintToFlow --> P2_MasterPlan;
    P2_MasterPlan --> StateManager; %% Stored/tracked by StateManager
    P2_MasterPlan --> ChromaDB; %% Also context for other agents

    %% Connection to Flow Executor
    P2_MasterPlan --> FlowExecutor;

    %% Phase 3: Code Generation, Integration & Initial Setup
    subgraph Phase3 ["Phase 3: Code Generation & Setup"]
        direction TB
        P3_FileSystemAgent["SystemFileSystemAgent (New/Core)"]:::newAgent
        P3_SmartCodeGen["SmartCodeGeneratorAgent (New/Core)"]:::newAgent
        P3_SmartCodeInteg["SmartCodeIntegrationAgent (New/Core)"]:::newAgent
        P3_LiveCode["Live Codebase Collection (ChromaDB)"]:::datastore
        P3_ExternalMCP["External MCP Servers Tools Doc (ChromaDB)"]:::datastore
        P3_LibDoc["Library Documentation (ChromaDB)"]:::datastore
    end
    class Phase3 phase;
    FlowExecutor --> P3_FileSystemAgent;
    FlowExecutor --> P3_SmartCodeGen;
    P3_SmartCodeGen --> P3_LiveCode;
    P3_SmartCodeGen --> P3_ExternalMCP;
    P3_SmartCodeGen --> P3_LibDoc;
    P3_SmartCodeGen --> P3_SmartCodeInteg;
    P3_SmartCodeInteg -- "writes code" --> P3_LiveCode;
    ChromaDB -- "provides context" --> P3_SmartCodeGen

    %% Phase 4: Testing & Refinement
    subgraph Phase4 ["Phase 4: Testing & Refinement"]
        direction TB
        P4_SmartTestGen["SmartTestGeneratorAgent (New/Core)"]:::newAgent
        P4_SystemTestRunner["SystemTestRunnerAgent (Core)"]:::coreComp
        P4_BugFixerAgent["BugFixerAgent (New)"]:::newAgent
        P4_TestReports["Test Reports (ChromaDB)"]:::artifact
    end
    class Phase4 phase;
    P3_LiveCode --> P4_SmartTestGen;
    FlowExecutor --> P4_SmartTestGen;
    P4_SmartTestGen -- "writes tests" --> P3_LiveCode; %% Tests are also code
    FlowExecutor --> P4_SystemTestRunner;
    P3_LiveCode --> P4_SystemTestRunner;
    P4_SystemTestRunner --> P4_TestReports;
    P4_TestReports --> ChromaDB;
    P4_TestReports --> P4_BugFixerAgent;
    FlowExecutor --> P4_BugFixerAgent;
    P4_BugFixerAgent -- "fixes code" --> P3_LiveCode;

    %% Phase 5: Documentation & Packaging
    subgraph Phase5 ["Phase 5: Documentation & Packaging"]
        direction TB
        P5_ProjectDocAgent["ProjectDocumentationAgent (New)"]:::newAgent
        P5_SmartCodeGenPack["SmartCodeGeneratorAgent (for Packaging)"]:::newAgent
        P5_Readme["README.md"]:::artifact
        P5_Deps["Dependency Files (e.g., requirements.txt)"]:::artifact
    end
    class Phase5 phase;
    P3_LiveCode --> P5_ProjectDocAgent;
    ChromaDB -- "Uses full project context" --> P5_ProjectDocAgent; %% Uses full project context
    FlowExecutor --> P5_ProjectDocAgent;
    P5_ProjectDocAgent --> P5_Readme;
    P5_Readme --> P3_LiveCode; %% Docs added to codebase
    FlowExecutor --> P5_SmartCodeGenPack;
    P3_LiveCode --> P5_SmartCodeGenPack;
    P5_SmartCodeGenPack --> P5_Deps;
    P5_Deps --> P3_LiveCode; %% Deps added to codebase

    %% Phase 6: Finalization & Release Preparation
    subgraph Phase6 ["Phase 6: Finalization & Release"]
        direction TB
        P6_SystemCommand["SystemCommandAgent (New/Core)"]:::newAgent
        P6_CodeReviewer["CodeReviewerAgent (New, Future)"]:::newAgent
        P6_FinalCode["Final Formatted Code (LiveCode Collection)"]:::datastore
        P6_Release["Project Archive/Release"]:::artifact
    end
    class Phase6 phase;
    P3_LiveCode --> P6_SystemCommand; %% Linting/Formatting
    FlowExecutor --> P6_SystemCommand;
    P3_LiveCode --> P6_CodeReviewer;
    ChromaDB --> P6_CodeReviewer;
    FlowExecutor --> P6_CodeReviewer;
    P6_SystemCommand --> P6_FinalCode;
    P6_CodeReviewer --> P6_FinalCode;
    P6_FinalCode --> P6_Release;

    %% General Linkages
    Phase1 --> Phase2;
    Phase2 --> Phase3;
    Phase3 --> Phase4;
    Phase4 --> Phase5;
    Phase5 --> Phase6;

```

This diagram outlines:
- The progression through the six major phases.
- Key `chungoid-core` infrastructure components like `FlowExecutor`, `AgentRegistry`, `StateManager`, and `ChromaDB` and their central roles.
- The introduction of new conceptual agents (e.g., `ArchitectAgent`, `BlueprintToFlowAgent`, `SmartCodeGeneratorAgent`) at different stages, noting that these would be built upon or are specializations of existing `chungoid-core` agent capabilities.
- Critical artifacts generated and consumed (e.g., `ProjectBlueprint.md`, `MasterExecutionPlan.yaml`, `live_codebase_collection`).
- The central role of ChromaDB as the Project Context Repository.

This Mermaid code can be pasted into any Markdown viewer or editor that supports Mermaid to render the visual diagram. 