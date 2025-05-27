# Chungoid Execution DSL Schema Documentation

This document describes the structure, fields, and usage of the Chungoid Execution DSL schema, as defined in `chungoid-core/schemas/execution_dsl.json`.

For background on JSON Schema, see [JSON Schema: Getting Started](https://json-schema.org/learn/getting-started-step-by-step).

---

## Overview
The Execution DSL schema defines the structure for flow execution plans in Chungoid. It ensures that all flows are valid, extensible, and robust for orchestration.

---

## Top-Level Fields

| Field         | Type     | Required | Description |
|-------------- |----------|----------|-------------|
| `name`        | string   | No       | Human-readable name for the flow. |
| `description` | string   | No       | Description of the flow's purpose. |
| `version`     | string   | No       | Version identifier for the flow. |
| `start_stage` | string   | Yes      | The name of the first stage to execute. |
| `stages`      | object   | Yes      | Mapping of stage names to stage definitions. |

---

## `stages` Object
- Keys: Stage names (must match `^[a-zA-Z_][\w-]*$`)
- Values: Stage definition objects (see below)

### Stage Definition Fields
| Field            | Type     | Required | Description |
|------------------|----------|----------|-------------|
| `agent_id`       | string   | Yes      | The agent to execute at this stage. If `agent_category` is specified, this can be omitted. |
| `agent_category` | string   | No       | Category of agent to invoke if `agent_id` is not specified. Enables dynamic agent selection. |
| `agent_selection_preferences` | object | No       | Preferences for selecting an agent from `agent_category`. Used only if `agent_category` is specified. |
| `inputs`         | object   | No       | Input parameters for the agent. |
| `next`           | string/object/null | No | Next stage to execute, or conditional branching object. |
| `on_error`       | string/object/null | No | Stage to execute on error, or conditional branching object. |
| `parallel_group` | string   | No       | Group name for parallel execution. |
| `plugins`        | array    | No       | List of plugin names to apply at this stage. |
| `success_criteria` | array    | No       | (P2.4) List of strings describing conditions that must be met for the stage to be considered successful. These are evaluated by the orchestrator's `_evaluate_criterion` method against the direct output of the agent. Criteria should be simple and directly evaluatable (e.g., path resolution, string operations, numeric comparisons). Complex boolean logic within a single criterion string (e.g., an OR condition) might require specific orchestrator parsing capabilities. Examples:
|                  |          |          |   `["output_field_name EXISTS", "string_output CONTAINS 'substring'", "numeric_output > 0", "items_list IS_NOT_EMPTY"]` |
| `clarification_checkpoint` | object | No   | (P2.5) Defines a point where the orchestrator may pause to seek user clarification. See `clarification_checkpoint` object details below. |
| `extra`          | object   | No       | Arbitrary extra data for extensibility. |

#### `next` and `on_error` Conditional Objects
- May be a string (stage name), null, or an object:
  - `condition`: string (expression to evaluate)
  - `true`: string (stage if condition is true)
  - `false`: string (stage if condition is false)

#### `clarification_checkpoint` Object (P2.5)
This optional object within a stage definition allows for explicit user interaction points. It should conform to the `ClarificationCheckpointSpec` model.

| Field                     | Type   | Required | Description |
|---------------------------|--------|----------|-------------|
| `prompt_message_for_user` | string | Yes      | The question or information to present to the user when paused for clarification. |
| `target_context_path`     | string | No       | Optional dot-notation path in the context where the user's input (from `chungoid flow resume --action provide_clarification --inputs ...`) should be placed. E.g., `stage_inputs.parameter_name` or `shared_data.user_provided_value`. If omitted, user input is merged into the root of the context. |
| `expected_input_schema`   | object | No       | Optional JSON schema defining the expected structure of the user's input JSON. This is primarily for documentation or future client-side validation. |
| `pause_condition`         | string | No       | An optional expression. If provided, the orchestrator will only pause if this condition evaluates to true at the checkpoint. If omitted, it always pauses if the checkpoint is reached and the stage was otherwise successful. |

#### Agent Category Selection Fields (New)
When `agent_id` is not specified for a stage, the system can dynamically select an agent based on a category and preferences.

*   **`agent_category: Optional[str]`**
    *   Specifies the functional category of the agent to be selected (e.g., "CodeGeneration", "Testing", "FileSystem").
    *   The `AsyncOrchestrator` will use this category to query the `AgentRegistry`.
*   **`agent_selection_preferences: Optional[Dict[str, Any]]`**
    *   A dictionary providing preferences to refine the selection of an agent from the specified `agent_category`.
    *   This is only used if `agent_category` is provided.
    *   The structure of this object can be flexible, but common keys recognized by the `RegistryAgentProvider` include:
        *   `capability_profile_match: Dict[str, Any]`: Requires the selected agent's `capability_profile` in its `AgentCard` to have exact matches for the specified key-value pairs.
            *   Example: `{"language": "python", "framework": "fastapi"}`
        *   `priority_gte: int`: Filters for agents with a `priority` in their `AgentCard` greater than or equal to the specified value.
        *   `version_preference: str`: Specifies a version preference.
            *   Example: `"latest_semver"` (requires agents to have a `version_semver` in their `capability_profile` or `AgentCard.version` for comparison).
    *   **Example Snippet (within a stage definition):**
        ```yaml
        # ...
        agent_category: "CodeGeneration"
        agent_selection_preferences:
          capability_profile_match:
            language: "python"
            feature: "template_based_generation"
          priority_gte: 3
          version_preference: "latest_semver"
        # ...
        ```

---

## Extensibility & Best Practices
- All fields except `extra` are strictly validated (`additionalProperties: false`).
- Use `extra` for future or custom fields.
- Use clear, descriptive names for stages and agents.
- Validate all flows before execution.

---

## Example (YAML)
```yaml
name: Example Flow
version: "1.0"
description: A sample flow using all schema features.
start_stage: s1
stages:
  s1:
    agent_id: agent.alpha
    inputs:
      param1: foo
    next:
      condition: "input > 5"
      true: s2
      false: s3
    plugins: ["log", "audit"]
    success_criteria:
      - "Output value 'result' is positive"
      - "Log file 's1_details.log' contains 'SUCCESS'"
    clarification_checkpoint:
      prompt: "Do you want to proceed with aggressive optimization (s2) or conservative (s3) based on input?"
      pause_condition: "input < 10"
    extra:
      custom_field: 42
  s2:
    agent_id: agent.beta
    next: s4
  s3:
    agent_id: agent.gamma
    on_error:
      condition: "error_code == 404"
      true: s4
      false: s5
  s4:
    agent_id: agent.delta
    parallel_group: group1
  s5:
    agent_id: agent.epsilon
```

---

## Example (JSON)
```json
{
  "name": "Example Flow",
  "version": "1.0",
  "description": "A sample flow using all schema features.",
  "start_stage": "s1",
  "stages": {
    "s1": {
      "agent_id": "agent.alpha",
      "inputs": { "param1": "foo" },
      "next": {
        "condition": "input > 5",
        "true": "s2",
        "false": "s3"
      },
      "plugins": ["log", "audit"],
      "success_criteria": [
        "Output value 'result' is positive",
        "Log file 's1_details.log' contains 'SUCCESS'"
      ],
      "clarification_checkpoint": {
        "prompt_message_for_user": "Do you want to proceed with aggressive optimization (s2) or conservative (s3) based on input?",
        "pause_condition": "input < 10"
      },
      "extra": { "custom_field": 42 }
    },
    "s2": { "agent_id": "agent.beta", "next": "s4" },
    "s3": {
      "agent_id": "agent.gamma",
      "on_error": {
        "condition": "error_code == 404",
        "true": "s4",
        "false": "s5"
      }
    },
    "s4": { "agent_id": "agent.delta", "parallel_group": "group1" },
    "s5": { "agent_id": "agent.epsilon" }
  }
}
```

---

## Further Reading
- [JSON Schema: Getting Started](https://json-schema.org/learn/getting-started-step-by-step)
- [Chungoid Execution DSL Condition Syntax](./execution_dsl_conditions.md)

---

*This is a living document.*
*Last updated: 2025-05-16 by Gemini Assistant* 