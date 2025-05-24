# Autonomous Project Engine: Architecture & Operational Model

## 1. Overview

The Autonomous Project Engine (APE) within Chungoid Core is designed to automate significant portions of the software development lifecycle. It takes a high-level user goal and orchestrates a series of specialized LLM-backed agents to perform requirements analysis, architectural design, planning, code generation, testing, and documentation. The engine operates in iterative, multi-cycle autonomous loops, with human-gated reviews and interventions possible between major cycles (e.g., after LOPRD generation, after Blueprint generation, after code generation).

This document outlines the APE's architecture, key agent roles, its operational model, and the usage of project state information.

## 2. Core Architectural Principles

*   **Agent-Based Specialization:** Each phase of the lifecycle is handled by a dedicated agent with specific skills, prompts, and knowledge (e.g., `ProductAnalystAgent_v1`, `ArchitectAgent_v1`, `SmartCodeGeneratorAgent_v1`).
*   **Artifact-Driven Workflow:** Agents produce and consume well-defined artifacts (LOPRDs, Blueprints, Code, Reports) stored and versioned, primarily managed by the `ProjectChromaManagerAgent_v1`.
*   **Autonomous Refinement Loops:** The `AutomatedRefinementCoordinatorAgent_v1` (ARCA) orchestrates quality gates. It uses confidence scores from generating agents and feedback from quality assurance agents (like `ProactiveRiskAssessorAgent_v1` and `RequirementsTracerAgent_v1`) to decide if an artifact is accepted or needs revision by its original generator.
*   **Centralized Orchestration:** The `AsyncOrchestrator` executes a `MasterExecutionPlan.yaml`, which defines the sequence of agent invocations, context passing, and conditional logic for the overall autonomous flow.
*   **Data Persistence & Context Management:** The `ProjectChromaManagerAgent_v1` (PCMA) is crucial for storing all project-related data (artifacts, reports, logs, agent reflections) in a structured way within project-specific ChromaDB collections. This allows agents to retrieve necessary context and ensures traceability.
*   **Human-in-the-Loop (Gated Cycles):** While individual sub-loops (e.g., LOPRD refinement) aim for full autonomy, the transition between major lifecycle phases (e.g., LOPRD -> Blueprint -> Plan -> Code -> Docs) can be gated by human review or approval, managed through the orchestration plan or external triggers.

## 3. Key Agent Roles & Responsibilities

*(This section will detail each primary agent in the APE. For brevity, summaries are provided here. Refer to individual agent design documents for full details.)*

*   **`ProjectChromaManagerAgent_v1` (PCMA):**
    *   **Responsibilities:** Manages all project-specific data in ChromaDB collections. Handles artifact storage, retrieval, versioning (conceptual), and provides a query interface for other agents.
    *   **Key Collections:** `project_goals`, `planning_artifacts` (LOPRDs, Blueprints, Plans), `risk_assessment_reports`, `traceability_reports`, `optimization_suggestion_reports`, `live_codebase_collection`, `test_reports_collection`, `agent_logs_collection`, etc.

*   **`ProductAnalystAgent_v1`:**
    *   **Responsibilities:** Takes a refined user goal and generates a detailed LLM-Optimized Product Requirements Document (LOPRD) in JSON format. Participates in refinement loops orchestrated by ARCA.

*   **`ArchitectAgent_v1`:**
    *   **Responsibilities:** Transforms an approved LOPRD into a `ProjectBlueprint.md`, outlining the system architecture, components, data models, and technical stack. Participates in refinement loops.

*   **`BlueprintReviewerAgent_v1`:**
    *   **Responsibilities:** Performs an advanced review of a `ProjectBlueprint.md`, suggesting optimizations, architectural alternatives, and identifying potential flaws. Its feedback can be used by ARCA or the `BlueprintToFlowAgent`.

*   **`BlueprintToFlowAgent_v1` (Capability of `SystemMasterPlannerAgent_v1`):**
    *   **Responsibilities:** Takes an approved and possibly reviewed `ProjectBlueprint.md` and generates a `MasterExecutionPlan.yaml` detailing the stages, agents, and tasks required to implement the project.

*   **`ProactiveRiskAssessorAgent_v1` (PRAA):**
    *   **Responsibilities:** Analyzes artifacts (LOPRDs, Blueprints, Plans, Code Modules) for potential risks, issues, and high-ROI optimization opportunities. Produces markdown reports used by ARCA.

*   **`RequirementsTracerAgent_v1` (RTA):**
    *   **Responsibilities:** Verifies traceability between different lifecycle artifacts (e.g., LOPRD to Blueprint, Blueprint to Plan). Produces a markdown report used by ARCA.

*   **`AutomatedRefinementCoordinatorAgent_v1` (ARCA):**
    *   **Responsibilities:** Acts as the quality gate and orchestrator for autonomous refinement sub-cycles. Evaluates artifact confidence, PRAA reports, and RTA reports to decide if an artifact is accepted or needs revision. Triggers the `ProjectDocumentationAgent_v1` after a successful cycle.

*   **`SmartCodeGeneratorAgent_v1` (Enhancement of `CoreCodeGeneratorAgent_v1`):**
    *   **Responsibilities:** Generates code for specific modules/components based on tasks from the `MasterExecutionPlan`, using LOPRD/Blueprint context from PCMA. Outputs code and a confidence score.

*   **`SmartCodeIntegrationAgent_v1` (Enhancement of `CoreCodeIntegrationAgent_v1`):**
    *   **Responsibilities:** Integrates generated code into the project's live codebase (managed via PCMA), handling file operations and versioning (conceptual).

*   **`SmartTestGeneratorAgent_v1` (Enhancement of `CoreTestGeneratorAgent_v1`):**
    *   **Responsibilities:** Generates unit/integration tests based on LOPRD (Acceptance Criteria), Blueprint, and generated code.

*   **`SystemTestRunnerAgent_v1`:**
    *   **Responsibilities:** Executes generated tests and reports results (pass/fail, coverage if available) to PCMA.

*   **`ProjectDocumentationAgent_v1`:**
    *   **Responsibilities:** Generates project documentation (README, API docs, dependency audit, etc.) based on all available project artifacts (LOPRD, Blueprint, Plan, code structure) retrieved via PCMA.

## 4. Operational Model: Multi-Cycle Autonomous Operation

The APE operates through a sequence of major cycles, each potentially containing autonomous sub-cycles for refinement.

1.  **Goal Intake & LOPRD Cycle:**
    *   User provides a high-level goal.
    *   `ProductAnalystAgent_v1` generates an initial LOPRD.
    *   ARCA, PRAA, and `ProductAnalystAgent_v1` engage in an autonomous loop until the LOPRD meets quality criteria or is flagged for human review.

2.  **Architectural Design (Blueprint) Cycle:**
    *   `ArchitectAgent_v1` generates a Blueprint from the approved LOPRD.
    *   (Optional) `BlueprintReviewerAgent_v1` provides feedback.
    *   ARCA, PRAA, RTA, and `ArchitectAgent_v1` engage in an autonomous loop until the Blueprint is accepted or flagged.

3.  **Execution Planning Cycle:**
    *   `BlueprintToFlowAgent_v1` generates a `MasterExecutionPlan.yaml` from the approved Blueprint.
    *   ARCA, PRAA, RTA, and `BlueprintToFlowAgent_v1` engage in an autonomous loop until the Plan is accepted or flagged.

4.  **Code Generation & Testing Cycle (per component/task in Plan):**
    *   For each task in the Plan:
        *   `SmartCodeGeneratorAgent_v1` generates code.
        *   `SmartCodeIntegrationAgent_v1` integrates it.
        *   `SmartTestGeneratorAgent_v1` generates tests.
        *   `SystemTestRunnerAgent_v1` runs tests.
        *   ARCA (potentially with PRAA for code) evaluates code quality and test results, orchestrating debugging/regeneration loops with the code/test generation agents until criteria are met or flagged.

5.  **Documentation Cycle:**
    *   After the final plan/code cycle is accepted by ARCA, ARCA triggers the `ProjectDocumentationAgent_v1`.
    *   `ProjectDocumentationAgent_v1` generates all project documentation.
    *   (Future) ARCA might review documentation confidence and potentially trigger a refinement loop for documentation.

6.  **Cycle Iteration & Human Gating:**
    *   After each major cycle (LOPRD, Blueprint, Plan, full Code/Test, Documentation), the `MasterExecutionPlan` can specify a pause for human review.
    *   The output of a cycle (e.g., a fully documented project, or a project at the Blueprint stage with flagged issues) is presented.
    *   Humans can review, provide feedback, adjust goals, or approve continuation to the next cycle or a new iteration of the current cycle with modifications.

## 5. Project Status (`project_status.json`)

While the APE relies heavily on `ProjectChromaManagerAgent_v1` for detailed artifact and log storage, the `project_status.json` file (managed by `StateManager`) in the `.chungoid` directory of a project serves a critical role for high-level tracking and recovery of `AsyncOrchestrator` runs.

*   **Purpose:**
    *   Tracks the state of individual stages within a `MasterExecutionPlan` execution (run ID, status, inputs, outputs, errors).
    *   Persists paused flow states, allowing `AsyncOrchestrator` to resume interrupted runs.
    *   Stores high-level metadata about each run (start/end times, overall status).

*   **Key Sections & Information:**
    *   `runs`: A list of all execution runs, each with:
        *   `run_id`: Unique identifier for the execution.
        *   `flow_id`: Identifier of the `MasterExecutionPlan` being executed.
        *   `start_time`, `end_time`.
        *   `status`: Overall status of the run (e.g., `COMPLETED_SUCCESS`, `COMPLETED_FAILURE`, `PAUSED_FOR_REVIEW`, `RUNNING`).
        *   `stages`: A dictionary mapping `stage_id` to its execution details (status, attempts, outputs, errors).
        *   `current_stage_id`: The stage currently being processed or last processed.
        *   `full_context_snapshot`: (Potentially large) A snapshot of the orchestrator's context at pause points.
    *   `paused_runs`: A dictionary mapping `run_id` to `PausedRunDetails`, including:
        *   `run_id`, `flow_id`, `paused_at_stage_id`, `status`, `timestamp`.
        *   `error_details`: Information about the error that caused the pause.
        *   `resume_actions_taken`: Log of resume attempts.
        *   `last_full_context`: The execution context at the point of pause.

*   **Interaction with APE:**
    *   The `AsyncOrchestrator` continuously updates `project_status.json` during a flow run.
    *   ARCA's decisions (accept/refine) influence stage progression, which is reflected in `project_status.json`.
    *   When a cycle needs human review (as defined in the `MasterExecutionPlan`), the orchestrator pauses and updates `project_status.json` to `PAUSED_FOR_REVIEW`. The `flow resume` CLI command uses this information.
    *   The APE itself (agents like PCMA) does not directly write to `project_status.json`; this is the orchestrator's and state manager's responsibility. Agents interact with PCMA for their detailed data needs.

## 6. Future Enhancements

*   More sophisticated confidence score mechanisms.
*   Deeper integration of ARCA with code quality tools.
*   Autonomous refinement loops for documentation.
*   Enhanced capabilities for agents to query and learn from past project data in PCMA.
*   Tool usage by agents (e.g., code linters, doc generators like Sphinx).

## 7. Key Areas for Future Refinement (Post-MVP)

While the MVP of the Autonomous Project Engine establishes a foundational end-to-end capability, several key areas are targeted for significant refinement and enhancement in future iterations:

*   **Robust Data Persistence (PCMA Implementation):**
    *   Transition `ProjectChromaManagerAgent_v1` from mocked behavior to full ChromaDB integration.
    *   Implement comprehensive metadata strategies, artifact versioning, and advanced query capabilities within PCMA.

*   **Agent Intelligence and Prompt Engineering:**
    *   Continuously refine LLM prompts for all agents based on empirical results to improve output quality, consistency, and adherence to structured formats (JSON, Markdown).
    *   Enhance agents' ability to utilize broader contextual information retrieved from PCMA.
    *   Develop more sophisticated error handling and fallback strategies within LLM interactions.

*   **ARCA Decision-Making and Coordination:**
    *   Evolve ARCA's decision logic beyond simple confidence scores to incorporate content-aware analysis of PRAA/RTA reports, potentially using LLM-based summarization or issue extraction.
    *   Enable ARCA to provide more targeted and actionable refinement instructions to generating agents.
    *   Implement a full refinement loop for documentation, allowing ARCA to assess and request revisions for generated project docs.

*   **Handling Project Complexity and Types:**
    *   Test and adapt the APE for more complex projects involving multiple interacting components, diverse tech stacks, and larger codebases.
    *   Develop strategies for handling existing codebases (brownfield projects), including code analysis and targeted modifications.

*   **Tool Integration and Action Capabilities:**
    *   Integrate external developer tools (linters, static analyzers, build systems, test runners) by enabling agents (or specialized tool-using agents) to invoke them and parse their outputs.
    *   Allow agents to perform real file system operations and version control actions (e.g., Git) through a secure, audited mechanism (likely via `FileSystemAgent` or similar).

*   **Orchestration Flexibility:**
    *   Explore more advanced conditional logic and scripting capabilities within `MasterExecutionPlan` definitions.
    *   Investigate dynamic plan adjustments by meta-agents based on runtime conditions.

*   **Human-in-the-Loop (HITL) Experience:**
    *   Develop richer interfaces and tools for human reviewers to inspect artifacts, provide feedback, and manage autonomous cycles.
    *   Offer more granular control over HITL intervention points within the `MasterExecutionPlan`.

*   **Learning and Adaptation:**
    *   Lay groundwork for agents to learn from past projects, feedback, and outcomes to improve their performance over time (long-term research area). 