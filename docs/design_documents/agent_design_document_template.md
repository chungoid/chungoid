# Agent Design Document: [AgentName_vX]

**Version:** X.Y
**Status:** Draft | Review | Approved
**Owner:** [Your Name/Team]
**Created:** YYYY-MM-DD
**Last Updated:** YYYY-MM-DD
**Related Blueprint:** [Link to Blueprint if applicable]
**Related LOPRD:** [Link to LOPRD if applicable]

---

## 1. Purpose & Scope

*   **Purpose:** Clearly define the agent's primary objective(s). What problem does it solve? What value does it provide?
*   **Scope:** Detail the boundaries of the agent's responsibilities. What is it explicitly responsible for, and what is considered out of scope?

---

## 2. High-Level Architecture

*   Illustrate how this agent fits within the broader autonomous engine or system.
*   Include a simple diagram if it helps clarify interactions with other major components or agents.
*   Describe key dependencies on other agents or services.

---

## 3. Agent Responsibilities & Capabilities

*   List the specific tasks and functions the agent is designed to perform.
*   Detail its core capabilities and any specialized skills.
*   Example:
    *   Responsibility 1: Analyze input X to produce output Y.
    *   Capability 1.1: Parse format A.
    *   Capability 1.2: Apply algorithm B.

---

## 4. Input/Output Schemas

*   Define all Pydantic models for data structures used as inputs and outputs.
*   Include examples where helpful.

### 4.1 Input Schema(s)

```python
# from pydantic import BaseModel, Field
# from typing import List, Optional

# class AgentInputModel(BaseModel):
#     parameter_one: str = Field(description="Description of parameter one.")
#     parameter_two: Optional[int] = Field(None, description="Description of optional parameter two.")
#     # ... other input fields
```

### 4.2 Output Schema(s)

```python
# class AgentOutputModel(BaseModel):
#     result_data: dict = Field(description="The primary output data.")
#     status_message: str = Field(description="A message indicating the outcome.")
#     # ... other output fields
```

---

## 5. API Contracts

*   **If this agent provides MCP (Meta-Chungoid Protocol) tools:**
    *   Provide OpenAPI specifications for each tool.
    *   Alternatively, provide clear Python function signatures and docstrings.
*   **If this agent consumes MCP tools:**
    *   List the external tools it relies on and their expected API.
*   **Internal Interface (if called directly by other Python components):**
    *   Define the Python class and method signatures.

---

## 6. Key Algorithms & Logic Flows

*   Describe the core algorithms or decision-making processes the agent uses.
*   Use pseudo-code, flowcharts, or numbered steps to explain complex logic.
*   For interactions with other agents or services, sequence diagrams can be very helpful.

**Example Logic Flow:**
1.  Receive input data (conforming to Input Schema).
2.  Validate input data.
3.  If validation fails, return error (conforming to Output Schema).
4.  Perform core processing step A (using Algorithm X).
5.  Perform core processing step B.
6.  Format results (conforming to Output Schema).
7.  Return results.

---

## 7. Prompting Strategy & Templates (If LLM-based)

*   **System Prompt:**
    ```text
    # (Full system prompt here)
    ```
*   **User Prompt Template(s) / Examples:**
    ```text
    # (Example user prompt, showing placeholders for dynamic content)
    # User Goal: {{user_goal_description}}
    # Context Documents:
    # {{#each context_documents}}
    # - ID: {{this.id}}, Summary: {{this.summary}}
    # {{/each}}
    ```
*   **Strategy for Prompt Versioning & Management:** (e.g., file naming conventions, storage location, use of a prompt registry)
*   **Plan for Prompt Evaluation & Refinement:** (e.g., metrics, A/B testing, golden datasets)
*   **LOPRD Interaction Prompts (if applicable):** Specific prompts related to generating or consuming LOPRDs.
    *   Example: "Based on User Story US-001 from the LOPRD (Document ID: {{loprd_id}}), identify the key acceptance criteria..."

---

## 8. Interaction with `ProjectChromaManagerAgent` (or other data stores)

*   **ChromaDB Collections Accessed:** List the names of ChromaDB collections this agent queries or writes to.
*   **Data Retrieved:**
    *   For each collection: What type of data is queried?
    *   Example queries (conceptual or actual ChromaQL/client library calls).
    *   Schema of retrieved data (if not already covered in Input/Output Schemas).
*   **Data Stored:**
    *   For each collection: What type of data is written?
    *   Schema of stored data (if not already covered in Input/Output Schemas).
    *   Example: "Stores `loprd_analysis_report.json` in the `project_planning_artifacts` collection."

---

## 9. Confidence Score Generation & Interpretation

*   **Generation (if this agent produces artifacts requiring confidence):**
    *   Method: (e.g., verbalized confidence from LLM, token probabilities, model self-assessment, perplexity-based, custom heuristics).
    *   Schema of the confidence score output (e.g., `{ "value": 0.95, "method": "LLM self-assessment", "explanation": "High confidence due to clear input requirements." }`).
    *   How is this score integrated into the agent's primary output?
*   **Interpretation (if this agent consumes artifacts with confidence scores):**
    *   How does this agent use confidence scores from upstream agents/processes?
    *   Are there specific thresholds that trigger different behaviors?

---

## 10. Error Handling, Resilience, and Retry Mechanisms

*   **Known Failure Modes:** List potential errors or failure scenarios.
*   **Error Detection:** How does the agent detect these failures?
*   **Handling Strategy:**
    *   What specific actions are taken for different errors (e.g., log and continue, retry, escalate, return specific error output)?
    *   Are there any retry mechanisms? If so, describe the backoff strategy, number of retries, etc.
*   **Resilience:** How does the agent maintain state or recover gracefully from failures?

---

## 11. Testing Strategy & Metrics

*   **Unit Tests:**
    *   Key components/functions to be unit tested.
    *   Mocking strategy for dependencies (LLMs, ChromaDB, other agents).
*   **Integration Tests:**
    *   How will interactions with other agents or services be tested?
    *   Scenarios for testing end-to-end flows involving this agent.
*   **Performance Tests (if applicable):**
    *   Key performance indicators (KPIs) (e.g., latency, throughput).
    *   Test scenarios.
*   **Key Metrics for Success:**
    *   How will the agent's effectiveness and quality be measured? (e.g., accuracy of output, task completion rate, reduction in human effort).

---

## 12. Alternatives Considered & Rationale for Chosen Design

*   Briefly describe other significant design approaches or algorithms that were considered.
*   Explain why the current design was chosen over the alternatives (e.g., performance, simplicity, scalability, alignment with requirements).

---

## 13. Open Issues & Future Work

*   List any known limitations, open questions, or areas that require further investigation.
*   Outline potential future enhancements or capabilities for this agent.

---
_This document adheres to the template: `chungoid-core/docs/templates/agent_design_document_template.md`_ 