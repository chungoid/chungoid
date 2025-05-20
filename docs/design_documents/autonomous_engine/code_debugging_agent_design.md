# Agent Design Document: CodeDebuggingAgent_v1

---
*   **Agent Name:** `CodeDebuggingAgent_v1` (Alias: `AutomatedDebuggingAgent_v1`)
*   **Version:** 1.0
*   **Owner:** Meta Engineering Team
*   **Status:** Proposed
*   **Category (Proposed):** `CODE_REMEDIATION`
*   **Blueprint References:**
    *   [Autonomous Project Engine Blueprint](../../../planning/blueprint_autonomous_project_engine.md) (see P3.M4.4.2)
    *   [Phase 3.1 Detailed Blueprint & Roadmap](../../../planning/phase_3_1_detailed_blueprint_and_roadmap.md) (see P3.1.4.2)
*   **Related Documents:**
    *   [EXECUTION_PLAN.md](../../../planning/EXECUTION_PLAN.md) (see P3.1.4.3)
---

## 1. High-Level Goal

The `CodeDebuggingAgent_v1` is a specialized agent responsible for analyzing faulty code in conjunction with failed test reports and relevant contextual information (LOPRD requirements, Blueprint sections, etc.). Its primary goal is to identify the likely cause of software bugs and propose specific, targeted code modifications (patches or corrected snippets) to fix them. This agent operates under the direction and orchestration of the `AutomatedRefinementCoordinatorAgent` (ARCA) as part of an autonomous refinement loop.

## 2. Detailed Responsibilities

*   Ingest details of faulty code, associated failing test reports, and contextual project artifacts.
*   Analyze stack traces, error messages, and code logic to hypothesize bug origins.
*   Leverage LOPRD requirements and Blueprint design specifications to understand the intended behavior of the code.
*   Formulate minimal, targeted code changes to address identified bugs.
*   Provide explanations for the diagnosed bug and the proposed solution.
*   Estimate the confidence in the proposed fix.
*   Report back to ARCA with the proposed fix, explanation, confidence, and status.
*   Indicate if more context is needed or if a bug is deemed unfixable within its current capabilities.

## 3. Input Schema (`DebuggingTaskInput`)

```json
{
  "type": "object",
  "properties": {
    "faulty_code_path": {
      "type": "string",
      "description": "Path to the code file needing debugging (retrieved by ProjectChromaManagerAgent)."
    },
    "faulty_code_snippet": {
      "type": "string",
      "description": "(Optional) The specific code snippet if already localized by ARCA or a previous process."
    },
    "failed_test_reports": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "test_name": {"type": "string"},
          "error_message": {"type": "string"},
          "stack_trace": {"type": "string"},
          "expected_behavior_summary": {"type": "string", "description": "(Optional) Summary of what the test expected."}
        },
        "required": ["test_name", "error_message", "stack_trace"]
      },
      "description": "List of structured test failure objects (from ProjectChromaManagerAgent, originally from SystemTestRunnerAgent)."
    },
    "relevant_loprd_requirements_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of LOPRD requirement IDs (e.g., FRs, ACs) that the faulty code was intended to satisfy (from ProjectChromaManagerAgent)."
    },
    "relevant_blueprint_section_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "List of Blueprint section IDs relevant to the code's design (from ProjectChromaManagerAgent)."
    },
    "previous_debugging_attempts": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "attempted_fix_summary": {"type": "string"},
          "outcome": {"type": "string", "description": "e.g., 'tests_still_failed', 'new_errors_introduced'"}
        }
      },
      "description": "(Optional) List of previous fixes attempted for this issue in the current cycle, to avoid loops and provide history to the LLM."
    },
    "max_iterations_for_this_call": {
      "type": "integer",
      "description": "(Optional) A limit set by ARCA for this specific debugging invocation's internal reasoning if applicable."
    }
  },
  "required": [
    "faulty_code_path",
    "failed_test_reports",
    "relevant_loprd_requirements_ids"
  ]
}
```

## 4. Output Schema (`DebuggingTaskOutput`)

```json
{
  "type": "object",
  "properties": {
    "proposed_solution_type": {
      "type": "string",
      "enum": ["CODE_PATCH", "MODIFIED_SNIPPET", "NO_FIX_IDENTIFIED", "NEEDS_MORE_CONTEXT"],
      "description": "Type of solution being proposed."
    },
    "proposed_code_changes": {
      "type": "string",
      "description": "The actual patch (e.g., diff format using `diff -u`) or the full modified code snippet. Null if no fix is identified."
    },
    "explanation_of_fix": {
      "type": "string",
      "description": "LLM-generated explanation of the diagnosed bug and the proposed fix. Null if no fix is identified."
    },
    "confidence_score": {
      "type": "number",
      "minimum": 0.0,
      "maximum": 1.0,
      "description": "Likelihood the proposed fix resolves the issue. If NO_FIX_IDENTIFIED, this might reflect confidence in that assessment."
    },
    "areas_of_uncertainty": {
      "type": "array",
      "items": {"type": "string"},
      "description": "(Optional) Any parts of the code, problem, or context the agent is unsure about."
    },
    "suggestions_for_ARCA": {
      "type": "string",
      "description": "(Optional) E.g., 'Consider broader refactoring if this pattern repeats,' or 'Unable to fix without X specific context from Y module.'"
    },
    "status": {
      "type": "string",
      "enum": ["SUCCESS_FIX_PROPOSED", "FAILURE_NO_FIX_IDENTIFIED", "FAILURE_NEEDS_CLARIFICATION", "ERROR_INTERNAL"],
      "description": "Overall status of the debugging attempt."
    }
  },
  "required": [
    "proposed_solution_type",
    "confidence_score",
    "status"
  ]
}
```

## 5. Core Logic & LLM Interaction

The agent's core logic is LLM-driven, guided by a carefully constructed prompt.

1.  **Contextual Understanding:** The LLM parses all inputs: faulty code, test failures (paying close attention to stack traces and specific error messages), LOPRD requirements, and relevant blueprint design notes. The objective is to build a comprehensive understanding of the code's intended functionality, its designed architecture, and the precise nature of the failure.
2.  **Hypothesis Generation:** Based on the error messages, stack traces, and code structure, the LLM hypothesizes potential root causes for the bug(s).
3.  **Solution Formulation:** For the most probable hypothesis, the LLM formulates a targeted code change. This may involve modifying existing lines, adding new ones, or deleting incorrect lines. The emphasis is on making the change as minimal as possible while effectively addressing the identified root cause.
4.  **Self-Correction/Refinement (Internal):** The prompt will encourage the LLM to internally review its proposed fix against the test failures and requirements. This step aims to ensure the proposed solution is logical and does not inadvertently violate other constraints or introduce new, obvious errors.
5.  **Confidence Assessment:** The LLM is prompted to provide a numerical confidence score (0.0-1.0) reflecting its belief that the proposed fix will resolve the issue(s).
6.  **Output Generation:** The LLM formats its findings according to the `DebuggingTaskOutput` schema.

**Prompt Strategy Highlights:**
*   Clearly define the agent's role as a debugger.
*   Emphasize understanding the *intent* behind the code by cross-referencing LOPRD requirements and Blueprint designs.
*   Instruct the LLM to meticulously analyze stack traces and error messages.
*   Guide towards proposing minimal, targeted, and precise changes. Avoid broad refactoring unless specifically requested or deemed essential with strong justification.
*   Require clear, concise explanations for both the diagnosed bug and the proposed fix.
*   Mandate the provision of a confidence score.
*   Provide explicit instructions for the output format (e.g., for a patch in `diff -u` format or a JSON object containing the modified code snippet within the larger `DebuggingTaskOutput` JSON).
*   Include examples of good vs. bad fixes if possible.
*   Instruct on how to request more context if necessary.

## 6. Interactions

### 6.1. Interaction with ARCA (`AutomatedRefinementCoordinatorAgent`)

*   **Invocation:** ARCA invokes `CodeDebuggingAgent_v1` when test failures are detected and a debugging attempt is warranted. ARCA prepares the `DebuggingTaskInput`, gathering all necessary file paths, test reports, and contextual artifact IDs from `ProjectChromaManagerAgent`.
*   **Response Handling:** ARCA receives the `DebuggingTaskOutput`.
    *   If `status` is `SUCCESS_FIX_PROPOSED` and `confidence_score` meets ARCA's threshold, ARCA instructs `SmartCodeIntegrationAgent` (or a similar agent/process) to apply the proposed code change. ARCA then triggers re-testing.
    *   If `status` is `FAILURE_NO_FIX_IDENTIFIED` or confidence is low, ARCA may try re-invoking the debugger with more context (if available and ARCA's retry limits for the stage permit) or escalate the issue for human review.
    *   If `status` is `FAILURE_NEEDS_CLARIFICATION`, ARCA attempts to retrieve the requested context (e.g., by querying `ProjectChromaManagerAgent`) and re-invokes the debugger if successful and retries allow. Otherwise, it escalates.
*   **Loop Management:** ARCA manages the overall retry limit for debugging a specific issue within a cycle to prevent infinite loops.

### 6.2. Interaction with `ProjectChromaManagerAgent`

*   The `CodeDebuggingAgent_v1` itself does not directly interact with `ProjectChromaManagerAgent`.
*   ARCA is responsible for querying `ProjectChromaManagerAgent` to gather all contextual information (LOPRD details, Blueprint sections, code files, test reports) needed for the `DebuggingTaskInput`.
*   ARCA also uses `ProjectChromaManagerAgent` to log the history and outcomes of debugging attempts.

## 7. Error Handling within the Agent

*   If the agent encounters an internal error (e.g., LLM API failure, inability to parse inputs), it should output a `DebuggingTaskOutput` with `status: ERROR_INTERNAL` and provide details if possible.
*   If the LLM indicates it cannot identify a fix, the agent sets `status: FAILURE_NO_FIX_IDENTIFIED`.
*   If the LLM indicates insufficient context, the agent sets `status: FAILURE_NEEDS_CLARIFICATION` and ideally populates `suggestions_for_ARCA` with details about the missing information.

## 8. Dependencies

*   **Base LLM:** For the core analysis, hypothesis, and code modification capabilities.
*   **`AutomatedRefinementCoordinatorAgent` (ARCA):** For invocation, context provision, and result processing.
*   **`ProjectChromaManagerAgent`:** Accessed by ARCA to provide necessary contextual data.
*   **`AgentRegistry`:** For agent registration and discovery by ARCA.
*   **Schema Definitions:** Pydantic models for `DebuggingTaskInput` and `DebuggingTaskOutput`.

## 9. Testing Strategy

*   **Unit Tests:**
    *   Validate `DebuggingTaskInput` and `DebuggingTaskOutput` schema handling.
    *   Test the agent's internal logic for preparing LLM prompts (mocking LLM calls).
    *   Test parsing of (mocked) LLM responses and correct population of `DebuggingTaskOutput`.
*   **Integration Tests:**
    *   Crucial for this agent.
    *   Set up small, self-contained test projects with a variety of known, common bugs and corresponding failing unit tests.
    *   ARCA invokes `CodeDebuggingAgent_v1` within a simulated workflow.
    *   Verify:
        *   Correct identification and proposed fixes for different bug types (e.g., off-by-one, incorrect variable, simple logic errors, type mismatches).
        *   Application of the fix leads to tests passing.
        *   Handling of unfixable bugs (graceful failure/escalation).
        *   Correct requests for more context.
        *   Accuracy of confidence scoring.
        *   ARCA's logging of debugging attempts and outcomes via `ProjectChromaManagerAgent`.
*   **Qualitative/Manual Testing:**
    *   Review LLM-generated explanations for clarity, accuracy, and insightfulness.
    *   Assess the quality, minimality, and correctness of proposed code changes.
    *   Evaluate the agent's performance on more complex bugs.

## 10. Future Enhancements

*   Ability to learn from past successful/failed debugging attempts (potentially by storing and retrieving solution patterns from ChromaDB).
*   More sophisticated static analysis capabilities (e.g., basic linting or type checking) before LLM invocation.
*   Interactive debugging mode where ARCA can provide clarifications mid-process.
*   Support for analyzing more complex error types (e.g., concurrency issues, memory leaks - significantly harder).
*   Automated generation of new test cases to confirm a fix and check for regressions.

## 11. Open Questions/Design Decisions

*   Exact threshold for ARCA to accept a fix based on confidence score.
*   Detailed strategy for ARCA to gather "more context" when requested – how specific can the debugger be, and how effectively can ARCA fulfill these requests?
*   Format of the code patch: `diff -u` is standard, but requires careful application. Full snippet replacement might be simpler initially but less targeted. The `proposed_solution_type` allows for flexibility.

---
_This is a living document. Initial draft: {{YYYY-MM-DD}} by Gemini Assistant._ 