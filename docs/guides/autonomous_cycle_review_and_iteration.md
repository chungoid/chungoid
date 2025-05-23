# Autonomous Cycle Review and Iteration Guide

This guide outlines the process for human review of outputs from the Autonomous Project Engine (APE) and how to initiate subsequent refinement cycles.

## 1. Overview of Autonomous Cycles

The APE, as detailed in the `autonomous_project_engine_overview.md`, operates in major cycles (LOPRD, Blueprint, Plan, Code, Documentation). While internal sub-cycles (e.g., LOPRD refinement via ARCA) aim for full autonomy, the `MasterExecutionPlan` can be designed to pause at the end of major cycles for human review and explicit approval before proceeding.

Even without explicit pause points, after an end-to-end run (like the `flask_app_full_cycle_v1.yaml` example), a human review of the final output is crucial before considering the project complete or starting a new iteration with significant changes.

## 2. Initiating an Autonomous Run

An autonomous project generation cycle is typically initiated using the `chungoid build` CLI command (primary usage) or `chungoid flow run` for advanced workflows. You provide either a goal file or a pre-defined `MasterExecutionPlan.yaml`.

### Primary Method: `chungoid build`

**Recommended for most users (99% of use cases):**
```bash
# Create your goal file
echo "Build a Python Flask web application with user authentication and SQLite database" > goal.txt

# Initialize the project directory (optional)
chungoid init ./my-flask-app

# Build the complete project
chungoid build --goal-file goal.txt --project-dir ./my-flask-app --tags "flask,webapp,auth"
```

### Advanced Method: `chungoid flow run`

**For complex workflows or predefined execution plans:**
```bash
# Using a predefined master plan
chungoid flow run --flow-yaml dev/examples/master_plans/flask_app_full_cycle_v1.yaml --project-dir ./dummy_project --tags "flask_mvp_test,p3m5_demo"

# Using a goal with specific initial context
chungoid flow run --goal "Build a React dashboard" --project-dir ./dashboard --initial-context '{"framework": "react", "typescript": true}'
```

### Command Options

*   `--goal-file FILE` / `--goal "text"`: The user goal (required for `build`, optional for `flow run`)
*   `--project-dir DIR`: Target project directory where `.chungoid` state will be managed (default: current directory)
*   `--flow-yaml FILE`: Specifies a pre-defined master plan to execute (advanced usage)
*   `--initial-context JSON`: Additional context as JSON string
*   `--tags TAGS`: Comma-separated tags for easier identification of the run in logs/metrics
*   `--run-id ID`: Custom run identifier (auto-generated if not provided)

## 3. Monitoring and Reviewing a Run

During and after the execution, several sources provide insight into the APE's operation:

### 3.1. Console Output

The CLI will stream logs from the orchestrator and agents, providing real-time (or near real-time) status updates, decisions made, and errors encountered.

### 3.2. Project Status (`<project_dir>/.chungoid/project_status.json`)

This JSON file is the primary record of the execution flow:
*   **Overall Run Status:** Indicates if the flow is `RUNNING`, `COMPLETED_SUCCESS`, `COMPLETED_FAILURE`, or `PAUSED_FOR_REVIEW`.
*   **Stage Details:** For each stage in the `MasterExecutionPlan`, it records:
    *   `status` (e.g., `COMPLETED_SUCCESS`, `PENDING`, `ERROR`).
    *   `attempts`.
    *   `inputs` provided to the agent.
    *   `outputs` produced by the agent (including generated artifact document IDs).
    *   `error_details` if any.
*   **Paused Runs:** If a flow pauses (due to error or planned human intervention), details are stored under the `paused_runs` key, including the `run_id`, `paused_at_stage_id`, and context.

**Inspecting `project_status.json` helps in:**
*   Understanding which stage failed and why.
*   Finding the document IDs of generated artifacts (LOPRD, Blueprint, code modules, reports, etc.).
*   Tracing the data flow between stages.

### 3.3. Metrics Store (`<project_dir>/.chungoid/metrics.jsonl`)

This file contains detailed event logs for various actions (flow start/end, stage start/end, agent invocation, LLM calls, errors). Use the `chungoid metrics` CLI commands to inspect these:

*   `chungoid metrics list --project-dir ./dummy_project --run-id <run_id_from_status_json>`: Lists events for a specific run.
*   `chungoid metrics summary --project-dir ./dummy_project --run-id <run_id_from_status_json>`: Provides a high-level summary of a run.

### 3.4. (Conceptual) Artifact Review via `ProjectChromaManagerAgent_v1` (PCMA)

In the MVP, agents generate mock document IDs. In a full implementation with PCMA:
*   The document IDs found in `project_status.json` (outputs of stages) would correspond to actual artifacts stored in the project's ChromaDB instance.
*   You would use PCMA's query interface (or a dedicated UI/CLI tool built on PCMA) to retrieve and inspect the content of LOPRDs, Blueprints, generated code files, PRAA/RTA reports, and final documentation.
*   For the MVP, you review the *mock content* described or generated by the agents if it's part of their logs or `llm_full_response` in `project_status.json` where applicable.

## 4. Handling Paused Flows & Human Intervention

If a `MasterExecutionPlan` includes explicit human review stages, or if an unrecoverable error occurs that ARCA cannot resolve, the flow will pause. The `project_status.json` will indicate this.

1.  **Notification:** The CLI output will typically indicate the pause and the `run_id`.
2.  **Review:** Examine `project_status.json`, logs, and (conceptually) artifacts to understand the reason for the pause and the current state.
3.  **Resume:** Use the `chungoid flow resume` command:
    ```bash
    chungoid flow resume <run_id> --project-dir ./dummy_project --action <action_type> --inputs '{...json...}' 
    ```
    *   `<run_id>`: The ID of the paused flow.
    *   `--action`: The intervention action (e.g., `retry`, `skip_stage`, `provide_clarification`, `abort`).
    *   `--inputs`: JSON string with necessary data for the chosen action (e.g., updated inputs for an agent, clarification text).

    Refer to the CLI help (`chungoid flow resume -h`) for specific actions and their requirements.

## 5. Initiating a Subsequent Refinement Cycle (Major Iteration)

Autonomous refinement loops handled by ARCA are for fine-tuning within a major lifecycle stage (e.g., getting the LOPRD right based on PRAA feedback).

If, after reviewing the output of a major cycle (e.g., the generated Blueprint is fundamentally misaligned with a changed understanding of the user goal), you need to make significant changes that go beyond ARCA's automated capabilities, you would typically initiate a new run with modified inputs:

1.  **Analyze Outputs:** Review the generated artifacts (LOPRD, Blueprint, Plan, Code, Docs) and identify areas for major revision.
2.  **Modify Inputs for the Next Run:**
    *   **Refined User Goal:** If the core requirements changed, update the initial user goal string.
    *   **Specific Agent Feedback:** If a particular agent needs different instructions for a new iteration (e.g., `ProductAnalystAgent_v1` needs to consider new constraints), you might prepare specific refinement instructions.
    *   **MasterExecutionPlan Adjustment:** For very significant changes, you might even adjust the `MasterExecutionPlan.yaml` itself to change the sequence of agents or their high-level inputs.
3.  **Initiate a New Run:**
    ```bash
    chungoid flow run --goal "<new_or_revised_user_goal>" --project-dir ./dummy_project --tags "iteration2_feedback_xyz"
    # OR, if using a plan and providing specific agent inputs through initial_context:
    chungoid flow run --flow-yaml <plan_name.yaml> --project-dir ./dummy_project --initial-context '{"product_analyst_refinement_instructions": "Focus on mobile usability for V2."}' --tags "iteration2_pa_feedback"
    ```
    Using a new `run_id` (automatically generated or specified with `--run-id`) will keep this iteration separate from previous ones, allowing for comparison.

By reviewing artifacts and logs, and by providing targeted inputs for subsequent runs, humans guide the APE's iterations towards the desired outcome, especially when changes are too broad or strategic for the automated refinement loops.

## 6. Example: Reviewing the Flask App MVP Output

After running `flask_app_full_cycle_v1.yaml`:
1.  Check `dummy_project/.chungoid/project_status.json` for the overall status and outputs of each stage (e.g., `loprd_gen_flask.loprd_doc_id`, `blueprint_gen_flask.blueprint_document_id`, `codegen_app_py_flask.generated_code_artifact_doc_id`, `doc_gen_flask.readme_doc_id`).
2.  Use `chungoid metrics summary --project-dir ./dummy_project --run-id <run_id>`.
3.  (Conceptually) Retrieve the artifacts using these IDs from PCMA. For MVP, examine agent logs or `llm_full_response` fields in `project_status.json` if agents stored their mocked full outputs there.
4.  Based on this review:
    *   If the (mocked) Flask app and its documentation are satisfactory for the MVP, the cycle is complete.
    *   If significant changes are needed (e.g., user goal was to include a database, but the MVP plan didn't), you would refine the `user_goal_flask` in the `global_config` of the YAML (or provide a new goal via CLI) and re-run, starting a new iteration. 