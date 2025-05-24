<!-- 
This file was automatically synchronized from dev/docs/execution_runtime_overview.md
Last sync: 2025-05-23T18:46:08.926591
Transform: adapt
Description: Update execution runtime documentation for users
-->

# Execution Runtime – Architecture Overview

> **Status:** Stable – last updated 2025-05-16.

The **Execution Runtime** is the control-loop that turns a YAML Flow definition (e.g., `ExecutionPlan` or `MasterExecutionPlan`) into real agent calls. It lives inside `chungoid-core` so *all* projects that embed Chungoid can rely on the same semantics and metrics. The runtime is designed to be:

* **Deterministic** – given the same Flow YAML + context, the same stage order will be produced (idempotent side-effects aside).
* **Extensible** – new routing rules, plugin hooks and storage back-ends can be added without changing the core loop.
* **Observable** – every stage execution can emit metrics, and its state can be persisted for inspection and resumption.
* **Intervenable** – supports pausing flows for human review or on error, and resuming them with various actions.

If you are familiar with classic workflow engines, think *mini-Airflow* but embedded, agent-aware, and with human-in-the-loop capabilities.

```mermaid
flowchart LR
    subgraph User Flow YAML
        A[Flow YAML] -- parsed --> B[ExecutionPlan / MasterExecutionPlan]
    end
    B --> C[AsyncOrchestrator]
    C --> D{Stage Loop}
    D -->|invoke agent via AgentProvider| E[Agent]
    C -->|state persistence| G[StateManager]
    D -->|record metrics (conceptual)| F[MetricsStore (Chroma)]
    G --> H[(.chungoid/project_status.json)]
```

## Key Components
| Module | Location | Purpose |
|--------|----------|---------|
| `StageSpec` | `chungoid.runtime.orchestrator` | Pydantic model representing a single stage's definition within a flow. |
| `ExecutionPlan` / `MasterExecutionPlan` | `chungoid.runtime.orchestrator` / `chungoid.schemas.master_flow` | Validated, in-memory representation of the entire flow. `MasterExecutionPlan` is typically used by `AsyncOrchestrator`. |
| `AsyncOrchestrator` | `chungoid.runtime.orchestrator` | The primary orchestrator. Walks the execution graph, executes stages asynchronously, resolves agents (by ID or category), manages context (including structured updates via `output_context_path`), evaluates success criteria, handles errors, checkpoints, and human intervention (pause/resume) via `StateManager`. |
| `SyncOrchestrator` | `chungoid.runtime.orchestrator` | A simpler, synchronous orchestrator, primarily for basic sequential execution or testing. |
| `AgentProvider` | `chungoid.utils.agent_resolver` | Interface for resolving `agent_id` or `agent_category` to an executable agent callable. |
| `StateManager` | `chungoid.utils.state_manager` | Persists and retrieves flow state, including paused run details, artifacts, and status. Crucial for `AsyncOrchestrator`'s pause/resume. |
| `MetricsStore` + `MetricEvent` | `chungoid.utils.metrics_store` | (Conceptual for direct orchestrator use, more integrated at a higher level) Persists per-stage timing & status events. |

## Lifecycle of an Asynchronously Orchestrated Flow

1.  **Parse & Plan Creation**: An `ExecutionPlan` or `MasterExecutionPlan` is created from a YAML definition (e.g., via `ExecutionPlan.from_yaml()` or by loading a `MasterFlowSchema`). This plan is validated against the defined schema.
2.  **Orchestrator Initialization**: An `AsyncOrchestrator` instance is created, supplied with the execution plan, an `AgentProvider` (to find and call agents), and a `StateManager` (to persist and manage state).
3.  **Execution Start (`run` method)**: The `AsyncOrchestrator.run()` method initiates the flow, typically starting from the `start_stage` defined in the plan. This triggers the internal `_execute_loop`.
4.  **Stage Loop (`_execute_loop`)**: For each stage:
    *   **Checkpoint Handling**: If a `checkpoint` is defined for the stage, the orchestrator saves the current flow state via `StateManager` and pauses, awaiting a `confirm_continue` action.
    *   **Agent Resolution**: The `agent_id` or `agent_category` (with optional `agent_selection_preferences`) from the `StageSpec` is resolved to an agent callable using the `AgentProvider`.
    *   **Agent Invocation**: The agent is invoked asynchronously with the appropriate context and inputs.
    *   **Result Processing**: The agent's result is processed. `success_criteria` defined in the `StageSpec` are evaluated against the agent's output. If criteria are not met, the stage is marked as failed. If it's an `AgentErrorDetails` indicating a recoverable error or a need for input, this is also noted.
    *   **Error Handling**: If the agent raises an unhandled exception, returns `AgentErrorDetails`, or fails `success_criteria`, the orchestrator consults the stage's `on_error` configuration. It might save the flow state (via `StateManager`) and pause for intervention, or terminate the flow.
    *   **Context Update**: The flow's main `context` is updated. The agent's raw output is stored in `context.outputs.<stage_name>`. If `output_context_path` is specified in the `StageSpec`, the output is also placed at that structured path (e.g., within `context.intermediate_outputs`).
    *   **State Update**: `StateManager` is used to record the completion status of the stage.
    *   **Next Stage Determination**: Based on the current stage's `next_stage`, `next_stage_true`/`next_stage_false` fields, and the evaluation of any `condition` (using `_parse_condition` and the current context), the orchestrator determines the next stage to execute.
5.  **Flow Termination**: The flow concludes when a stage has no `next_stage` defined (i.e., `next_stage: null`), an explicit terminal stage like `FINAL_STEP` is reached, or if an unhandled error leads to termination, or if an `abort_flow` action is issued. Internal terminal markers like `_END_SUCCESS_` are handled gracefully by the loop.
6.  **Human Intervention (`resume_flow` method)**: If a flow is paused (due to a checkpoint or an error):
    *   A human (or an automated system) can inspect the paused state using `chungoid inspect-intervention <run_id>`.
    *   The `AsyncOrchestrator.resume_flow()` method is called (typically via `chungoid resume-flow <run_id> --action ...`).
    *   Based on the specified action (`retry`, `skip_stage`, `force_branch`, `retry_with_inputs`, `abort_flow`, `confirm_continue`), the `StateManager` loads the paused state, the orchestrator modifies the execution context or target stage as needed, and then re-enters the `_execute_loop` to continue execution.

### Detailed Orchestrator Behaviors

*   **Condition Parsing (`_parse_condition`)**: When a `StageSpec` includes a `condition` field (e.g., `"outputs.stage_a.result == 'go_left'"`), the `AsyncOrchestrator`'s `_parse_condition` method evaluates this string against the current flow `context`. It supports basic comparisons (e.g., `==`, `!=`) and resolves dotted paths (like `outputs.stage_a.result`) to actual values within the context. The outcome (boolean) determines whether `next_stage_true` or `next_stage_false` is chosen.

*   **Success Criteria Evaluation (`_evaluate_criterion`)**: After an agent successfully executes, if `success_criteria` are defined in the `StageSpec`, the `AsyncOrchestrator`'s `_evaluate_criterion` method evaluates each criterion against the agent's direct output. This process is similar to condition parsing, supporting path resolution and various operators (e.g., `EXISTS`, `IS_NOT_EMPTY`, `CONTAINS`, `ENDS_WITH`, `==`, `>`). If any criterion fails, the stage is considered to have failed, triggering error handling logic.

*   **Error Handling (`on_error` field)**: If a stage execution results in an error (agent exception, `AgentErrorDetails`, or failed success criteria), the `AsyncOrchestrator` inspects the `on_error` field of the `StageSpec`. This field can specify:
    *   A specific stage to transition to (e.g., an error handling stage).
    *   A directive to `PAUSE_FOR_INTERVENTION`, which saves the flow state using `StateManager` and halts execution, requiring manual resumption.
    *   A directive to `FAIL_FAST`, which terminates the flow immediately.
    *   (Future) More complex configurations like retry limits with delays.
    If `on_error` is not defined, a default behavior (e.g., pause or fail fast) is applied.

*   **`resume_flow` Actions in Detail**:
    *   `retry`: Re-attempts the paused stage with its original inputs and context.
    *   `retry_with_inputs`: Re-attempts the paused stage, but with new inputs provided via the `--input-json` CLI option. The orchestrator updates the stage's input portion of the context before retrying.
    *   `skip_stage`: Ignores the paused stage and attempts to proceed to the stage that would normally follow it (as defined by `next_stage` or conditional logic from the skipped stage's perspective, using the context *as it was* when the flow paused).
    *   `force_branch <target_stage_id>`: Jumps execution to an arbitrary `<target_stage_id>` within the flow. The context remains as it was at the point of pause.
    *   `abort_flow`: Terminates the flow run permanently. The `StateManager` records this terminal state.
    *   `confirm_continue`: Used specifically for flows paused at a `HUMAN_REVIEW_REQUIRED` checkpoint. This action signals that the review is complete, and the orchestrator proceeds to execute the stage that was checkpointed.

## Extensibility Hooks (Roadmap)
* **AsyncOrchestrator** – True async variant for IO-bound agents.
* **Plugin System** – Insert custom behaviours (retry, circuit-breaker, logging) per stage.
* **OpenTelemetry Exporter** – Bridge `MetricEvent` into OTLP collectors.

---

## Alternative: Simple `FlowExecutor`

For scenarios requiring a lightweight, standalone execution of Stage-Flow YAMLs without the overhead of the full orchestration metrics, state persistence for resume, or advanced error handling features described herein, `chungoid-core` also provides a `FlowExecutor` class in `chungoid.flow_executor`.

This executor:
- Directly parses and validates Stage-Flow YAML against its schema (`stage_flow_schema.json`).
- Iterates through stages, invoking registered agents. It uses an `AgentProvider` (e.g., `DictAgentProvider` or `RegistryAgentProvider`) for resolving `agent_id` to callables, similar to the main `AsyncOrchestrator`.
- Supports basic conditional branching based on agent results.
- Is designed for simplicity and can be useful for testing or embedded use cases where the full runtime is not needed.
- The primary entry point is its `run(yaml_path)` method, which executes the flow and returns an ordered list of completed stage names.
- Includes a basic CLI interface (runnable via `python -m chungoid.flow_executor <path_to_yaml>`) for quick testing with dummy agents if the file is executed directly.

It does not use the `ExecutionPlan` object, `MetricsStore`, or the advanced human intervention capabilities detailed in this document for the main orchestrator.

---

## Flow Examples

Sample flow YAML files demonstrating various features of the Execution DSL, including stage definitions, agent invocation, and conditional logic, can be found in the `dev/examples/` directory within the project. These examples can be run using the `chungoid run` CLI command (which utilizes the main orchestrator) or, for basic flows, potentially adapted for use with the `FlowExecutor`.

---

## Why another runtime?
LLM-driven systems need a thin orchestration layer that can:
1. Respect human-authored flow graphs (not hard-coded DAGs in Python).
2. Branch on **semantic conditions** (e.g. sentiment score) that agents
   compute at run-time.
3. Keep state minimal so any agent can reconstruct context from Chroma.

The Phase-6 runtime is the first iteration that checks all three boxes while
remaining <300 LOC.

---

## Execution Lifecycle (happy-path)
1. **Initialisation** – caller builds `ExecutionPlan` from YAML; schema guard
   rejects invalid keys early.
2. **Run loop** – `current = start_stage` → evaluate → call agent (TBD) →
   decide next edge.
3. **Metrics emission** – in the `finally:` block the orchestrator records a
   `MetricEvent` whether the stage succeeded or threw.
4. **Termination** – flow ends when `next` is `null` / missing, or when
   `max_hops` is exceeded (safe-guard against cycles).

### Error handling
If a stage raises, the orchestrator looks for an `on_error` field.  This uses
*Last updated: 2025-05-10 (skeleton)*

### Human Intervention
*   **`chungoid flow resume <run_id> --action <action_type> [--target-stage <stage_id>] [--inputs <json_string>]`**: Provides CLI commands to resume a paused flow with options to retry, skip, force branch, or abort.

---
*This is a living document.*
*Last updated: 2025-05-16 by Gemini Assistant* 