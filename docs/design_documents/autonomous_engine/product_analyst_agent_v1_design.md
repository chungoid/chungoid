---
title: "Agent Design: ProductAnalystAgent_v1"
category: design_document
owner: "Meta Engineering Team"
status: "Draft"
created: YYYY-MM-DD # To be filled
updated: YYYY-MM-DD # To be filled
version: "0.1.0"
related_blueprint: "blueprint_autonomous_project_engine.md"
related_agents: ["ProjectChromaManagerAgent", "AutomatedRefinementCoordinatorAgent"]
---

# Agent Design: ProductAnalystAgent_v1

## 0. Document History
| Version | Date       | Author             | Changes                                      |
|---------|------------|--------------------|----------------------------------------------|
| 0.1.0   | YYYY-MM-DD | Gemini Assistant   | Initial Draft                                |

## 1. Purpose & Scope

### 1.1. Purpose
The `ProductAnalystAgent_v1` is responsible for transforming a high-level, refined user goal and associated assumptions/ambiguities into a detailed, structured LLM-Optimized Product Requirements Document (LOPRD) in JSON format. This LOPRD serves as the foundational specification for subsequent project phases like architectural design, planning, and development within the Autonomous Project Engine.

### 1.2. Scope
#### 1.2.1. In Scope
*   Consuming a `refined_user_goal.md`, `assumptions_and_ambiguities.md`, and an `loprd_schema.json` as primary inputs.
*   Generating a comprehensive LOPRD in JSON format that strictly adheres to the `loprd_schema.json`.
*   Identifying and articulating:
    *   User Stories
    *   Functional Requirements (FRs)
    *   Acceptance Criteria (ACs) for FRs/User Stories
    *   Non-Functional Requirements (NFRs)
    *   Project Scope (In-Scope, Out-of-Scope)
    *   Assumptions and Constraints relevant to the LOPRD.
    *   Optionally, a Data Dictionary/Glossary.
*   Ensuring all key elements within the LOPRD (User Stories, FRs, ACs, NFRs) have unique identifiers.
*   Providing a self-assessed confidence score for the generated LOPRD.
*   Interacting with `ProjectChromaManagerAgent` to retrieve inputs and store the output LOPRD and confidence score.
*   Responding to refinement requests from `AutomatedRefinementCoordinatorAgent` (ARCA) by generating revised LOPRDs.

#### 1.2.2. Out of Scope
*   Generating the initial `refined_user_goal.md` or `assumptions_and_ambiguities.md` (these are inputs).
*   Performing architectural design (handled by `ArchitectAgent`).
*   Creating project plans or task breakdowns (handled by `BlueprintToFlowAgent`).
*   Directly generating code or test cases.
*   Validating the *feasibility* of requirements beyond initial LLM-based assessment (deeper feasibility is assessed by downstream agents and processes).
*   Making final decisions on conflicting requirements without escalation or guidance (complex conflicts might be flagged for ARCA or human review post-cycle).

## 2. High-Level Architecture

```mermaid
graph TD
    A[Refined User Goal Doc ID] --> PAA{ProductAnalystAgent_v1};
    B[Assumptions Doc ID] --> PAA;
    C[LOPRD Schema Doc ID] --> PAA;
    PAA -.-> PCMA(ProjectChromaManagerAgent): Retrieve Inputs;
    PCMA --> PAA: Input Documents;
    PAA -> LLM([LLM Model]): Generate LOPRD Content;
    LLM --> PAA: LOPRD JSON + Confidence Hints;
    PAA -.-> PCMA: Store LOPRD.json & Confidence Score;
    ARCA(AutomatedRefinementCoordinatorAgent) -- Refinement Feedback --> PAA;
```
*The `ProductAnalystAgent_v1` receives pointers to input documents (refined goal, assumptions, LOPRD schema) from an orchestrator. It uses `ProjectChromaManagerAgent` to fetch these documents. It then interacts with an LLM, guided by its prompt, to generate the LOPRD JSON. The generated LOPRD and a confidence score are stored back via `ProjectChromaManagerAgent`. It can be re-invoked by ARCA with feedback to refine the LOPRD.*

## 3. Agent Responsibilities & Capabilities

### 3.1. Core Responsibilities
*   Accurately interpret and decompose refined user goals.
*   Systematically structure requirements information according to the LOPRD schema.
*   Ensure clarity, testability, and reasonable completeness of the generated LOPRD.
*   Generate a self-assessment of confidence in the output.

### 3.2. Key Capabilities
*   **Requirement Elicitation (LLM-driven):** Extract and infer detailed requirements from potentially high-level goals.
*   **Structured Data Generation:** Produce valid JSON output conforming to a complex schema.
*   **User Story Formulation:** Create well-formed user stories.
*   **FR & AC Definition:** Define specific functional requirements and their corresponding acceptance criteria.
*   **NFR Identification:** Identify relevant non-functional requirements.
*   **Schema Adherence:** Strictly follow the provided LOPRD JSON schema.
*   **Confidence Assessment:** Provide a qualitative or quantitative measure of confidence in the generated LOPRD.

## 4. Input/Output Schemas

### 4.1. Input Schema(s)
*Primary Input (e.g., from Orchestrator/ARCA):*
```python
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class ProductAnalystTaskInput(BaseModel):
    task_id: str = Field(..., description="Unique identifier for this LOPRD generation task.")
    refined_user_goal_doc_id: str = Field(..., description="ChromaDB ID of the refined_user_goal.md document.")
    assumptions_doc_id: str = Field(..., description="ChromaDB ID of the assumptions_and_ambiguities.md document.")
    loprd_schema_doc_id: str = Field(..., description="ChromaDB ID of the loprd_schema.json document.")
    previous_loprd_doc_id: Optional[str] = Field(None, description="ChromaDB ID of a previous LOPRD version, if this is a refinement task.")
    refinement_feedback: Optional[Dict[str, Any]] = Field(None, description="Structured feedback from ARCA or other reviewers for LOPRD refinement.")
    project_name: str = Field(..., description="Name of the project for LOPRD metadata.")
    # Config options for LLM, e.g., model choice, temperature, can be passed here or be part of agent's own config
    agent_config: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        extra = 'forbid'
```

### 4.2. Output Schema(s)
*Primary Output (to Orchestrator/ARCA, with LOPRD stored via ProjectChromaManagerAgent):*
```python
class ProductAnalystTaskOutput(BaseModel):
    task_id: str = Field(..., description="Echoed task_id.")
    status: str = Field(..., description="SUCCESS or FAILURE.")
    generated_loprd_doc_id: Optional[str] = Field(None, description="ChromaDB ID of the generated llm_optimized_prd.json.")
    confidence_score: Optional[float] = Field(None, description="Agent's confidence in the generated LOPRD (0.0-1.0).")
    confidence_assessment_notes: Optional[str] = Field(None, description="Brief explanation accompanying the confidence score.")
    error_message: Optional[str] = Field(None, description="Error message if status is FAILURE.")

    class Config:
        extra = 'forbid'
```
*(The LOPRD JSON itself is stored in ChromaDB by ProjectChromaManagerAgent, not directly in this output schema).*

## 5. API Contracts

`ProductAnalystAgent_v1` is primarily orchestrated and does not expose its own MCP tools. It consumes services from `ProjectChromaManagerAgent` and interacts with an LLM provider.

### 5.1. Provided MCP Tools (If Any)
*   None.

### 5.2. Consumed MCP Tools (If Any)
*   Relies on an LLM provider (e.g., OpenAI via `LLMProvider` utility from `chungoid-core`) which is not strictly an MCP tool but a service.
*   Indirectly relies on tools used by `ProjectChromaManagerAgent` for ChromaDB interaction.

### 5.3. Python Interface (If applicable for direct orchestration)
```python
from schemas.loprd_schema import LOPRD # Assuming loprd_schema.json is translated to a Pydantic model

class IProductAnalystAgent_v1:
    async def generate_loprd(self, input_data: ProductAnalystTaskInput) -> ProductAnalystTaskOutput:
        """
        Generates or refines an LLM-Optimized Product Requirements Document.
        """
        raise NotImplementedError
```

## 6. Key Algorithms & Logic Flows

### 6.1. Main Processing Loop
1.  Receive `ProductAnalystTaskInput`.
2.  Validate input.
3.  Use `ProjectChromaManagerAgent` to retrieve content of `refined_user_goal.md`, `assumptions_and_ambiguities.md`, and `loprd_schema.json` (and `previous_loprd.json` + `refinement_feedback` if applicable) based on provided Doc IDs.
4.  Construct a detailed prompt for the LLM, incorporating the retrieved content, task instructions from its own prompt definition (`product_analyst_agent_v1.yaml`), and any refinement feedback.
5.  Invoke the LLM to generate the LOPRD JSON content and a verbal confidence assessment.
6.  Parse the LLM response:
    *   Extract the LOPRD JSON string.
    *   Validate the JSON string against the retrieved `loprd_schema.json`.
        *   If invalid, attempt self-correction (e.g., re-prompt LLM with error) for a limited number of tries or log error.
    *   Extract/derive a numerical confidence score (e.g., 0.0-1.0) from the verbal assessment and potentially other heuristics.
7.  If successful and LOPRD is valid:
    *   Generate `loprd_metadata` (document_id, version, timestamps, etc.).
    *   Instruct `ProjectChromaManagerAgent` to store the complete LOPRD JSON (including generated metadata) in the `project_planning_artifacts` collection.
    *   Receive the new `generated_loprd_doc_id` from `ProjectChromaManagerAgent`.
8.  Return `ProductAnalystTaskOutput` with status, `generated_loprd_doc_id`, confidence score, and assessment notes.

### 6.2. Refinement Loop (Interaction with ARCA)
1.  If `ProductAnalystTaskInput` includes `previous_loprd_doc_id` and `refinement_feedback`:
2.  The agent retrieves the previous LOPRD and the feedback.
3.  The prompt to the LLM is augmented to specifically address the feedback and revise the previous LOPRD, rather than starting from scratch (unless feedback dictates a full rewrite).
4.  The version in `loprd_metadata` should be incremented.

## 7. Prompting Strategy & Templates

### 7.1. Core System Prompt
*   Defined in `chungoid-core/server_prompts/autonomous_engine/product_analyst_agent_v1.yaml`.
*   Focuses on persona, primary responsibility (LOPRD generation from goal), adherence to schema, and generation of key LOPRD components (user stories, FR/AC, NFRs).

### 7.2. Task-Specific Prompt Templates (Examples)
*   The main prompt structure is within `chungoid-core/server_prompts/autonomous_engine/product_analyst_agent_v1.yaml` under `prompt_details` and `user_prompt`.
*   Key dynamic elements injected into the prompt will be:
    *   Content of `refined_user_goal.md`.
    *   Content of `assumptions_and_ambiguities.md`.
    *   The `loprd_schema.json` (or a summary/key aspects if too large for direct injection).
    *   If it's a refinement task: content of the previous LOPRD and specific feedback from ARCA.

### 7.3. Prompt Versioning & Management
*   Prompts are versioned within the YAML file itself (e.g., `# Prompt Version: 0.1.0`).
*   Managed in `chungoid-core/server_prompts/autonomous_engine/product_analyst_agent_v1.yaml`.

### 7.4. LOPRD Interaction Prompts
*   The agent *generates* the LOPRD. Its prompts are designed to guide the LLM in structuring the output to match the `loprd_schema.json`. It explicitly instructs the LLM on generating User Stories, FRs, ACs, NFRs, unique IDs, etc.

### 7.5. Prompt Evaluation & Refinement Plan
*   Evaluation based on:
    *   Schema validity of the generated LOPRD JSON.
    *   Completeness against the input `refined_user_goal.md`.
    *   Clarity and testability of requirements (can be partially assessed by PRAA/RTA later).
    *   Agent's self-assessed confidence score and its correlation with actual quality.
    *   Feedback from downstream agents (e.g., `ArchitectAgent` struggling to use the LOPRD).
    *   Number of refinement cycles initiated by ARCA for LOPRDs generated by this agent.
*   Refinement will involve updating the YAML prompt definition and its version.

## 8. Interaction with `ProjectChromaManagerAgent`

### 8.1. Data Read from ChromaDB
*   **Collection:** `project_goals`
    *   **Data:** Content of `refined_user_goal.md` (identified by `refined_user_goal_doc_id`).
    *   **Purpose:** Primary input for LOPRD generation.
*   **Collection:** `project_planning_artifacts` (or a dedicated schema store if schemas are versioned in ChromaDB)
    *   **Data:** Content of `loprd_schema.json` (identified by `loprd_schema_doc_id`).
    *   **Data:** Content of `assumptions_and_ambiguities.md` (identified by `assumptions_doc_id`).
    *   **Data (for refinement):** Content of a previous `llm_optimized_prd.json` (identified by `previous_loprd_doc_id`).
    *   **Purpose:** To guide LOPRD structure and content.

### 8.2. Data Written to ChromaDB
*   **Collection:** `project_planning_artifacts`
    *   **Data:** The generated `llm_optimized_prd.json` (full JSON content).
    *   **Metadata stored alongside (or within the LOPRD JSON):** LOPRD metadata (doc_id, version, timestamps, project_name, authors, source_goal_id), agent's confidence score, confidence assessment notes.
    *   **Purpose:** To persist the primary output of the agent for use by downstream agents.
*   **Collection:** `agent_reflections_and_logs`
    *   **Data:** Log of LOPRD generation task (input IDs, output LOPRD ID, confidence, errors if any).
    *   **Purpose:** Auditing and debugging.

## 9. Confidence Score Generation & Interpretation

### 9.1. Generation (If this agent generates artifacts)
*   The agent's prompt explicitly asks the LLM to provide a verbal confidence assessment (e.g., "Confidence: High/Medium/Low because...").
*   The agent's Python wrapper code will parse this verbal statement (and potentially other signals from the LLM response, if available) to derive a numerical score (e.g., High=0.9, Medium=0.7, Low=0.5) and capture the reasoning as `confidence_assessment_notes`.
*   The schema for stored confidence data will be part of the `ProductAnalystTaskOutput` and also potentially stored with the LOPRD artifact's metadata in ChromaDB: `{ value: float (0-1), explanation: Optional[str] }`.

### 9.2. Interpretation (If this agent consumes artifacts with confidence scores)
*   Not applicable for its primary generation task. However, if refining a previous LOPRD, it might implicitly consider the (low) confidence that triggered the refinement.

## 10. Error Handling, Resilience, and Retry Mechanisms

### 10.1. Error Detection & Reporting
*   **Schema Validation Failure:** If the LLM output for LOPRD does not conform to `loprd_schema.json` after parsing.
*   **LLM API Errors:** Timeouts, rate limits, content filtering issues from the LLM provider.
*   **ChromaDB Errors:** Failures during retrieval of inputs or storage of outputs via `ProjectChromaManagerAgent`.
*   Errors will be reported in the `error_message` field of `ProductAnalystTaskOutput` and logged to `agent_reflections_and_logs`.

### 10.2. Retry Strategies
*   **LOPRD Schema Validation Failure:** The agent may attempt self-correction by re-prompting the LLM with the validation errors for a limited number of tries (e.g., 1-2 retries).
*   **LLM API Errors (Transient):** Implement exponential backoff and retry for a configurable number of attempts (e.g., 3 retries for timeouts).

### 10.3. Failure Escalation
*   If retries fail, or for non-transient errors, the agent will report status as `FAILURE` in `ProductAnalystTaskOutput` with a detailed `error_message`.
*   The `AutomatedRefinementCoordinatorAgent` (ARCA) will be responsible for handling persistent failures from this agent (e.g., flagging for human review post-cycle).

### 10.4. Resilience to Input Issues
*   **Missing/Invalid Doc IDs:** The agent will fail fast if required document IDs in `ProductAnalystTaskInput` are missing or if `ProjectChromaManagerAgent` cannot retrieve them, reporting an error.
*   **Ambiguous Goal (after initial refinement):** The agent will attempt to generate the LOPRD to the best of its ability based on the provided `refined_user_goal.md` and `assumptions_and_ambiguities.md`. Any significant ambiguities encountered during LOPRD generation that it cannot resolve will be noted in its confidence assessment (lowering confidence) and potentially flagged in the LOPRD's 'assumptions' section if appropriate.

## 11. Testing Strategy & Metrics

### 11.1. Unit Tests
*   Test the parsing of `ProductAnalystTaskInput`.
*   Test the construction of prompts to the LLM based on different inputs (initial generation vs. refinement).
*   Test the parsing of LLM responses (extracting JSON, verbal confidence).
*   Test the logic for deriving numerical confidence from verbal assessment.
*   Test interaction points with a mocked `ProjectChromaManagerAgent` (verifying correct calls for data retrieval and storage).
*   Test schema validation logic for LOPRD JSON.
*   Test error handling and reporting mechanisms.

### 11.2. Integration Tests
*   Test with a live (or well-mocked) LLM provider to ensure valid LOPRD JSON is generated for sample goals.
*   Test with a live `ProjectChromaManagerAgent` and ChromaDB instance to ensure correct data flow for inputs and outputs.
*   Test the refinement loop: agent receives feedback from a (mocked) ARCA and produces a revised LOPRD.
*   Test end-to-end flow: Orchestrator invokes agent -> agent retrieves inputs -> generates LOPRD -> stores LOPRD -> returns output.

### 11.3. Performance Tests (If Applicable)
*   Not initially critical, but track average LOPRD generation time for different goal complexities.

### 11.4. Key Metrics for Success/Evaluation
*   **LOPRD Schema Compliance:** Percentage of generated LOPRDs that validate against the schema without errors.
*   **LOPRD Quality (Manual/Automated Review):** Score based on clarity, completeness, testability of requirements (may involve PRAA/RTA outputs or human spot-checks initially).
*   **Confidence Score Accuracy:** Correlation between agent's confidence and actual LOPRD quality.
*   **Refinement Cycles:** Number of ARCA-initiated refinement cycles needed for LOPRDs from this agent. Lower is better.
*   **Downstream Usability:** Feedback from `ArchitectAgent` on the usability and completeness of the LOPRD.

## 12. Alternatives Considered & Rationale for Chosen Design

*   **Alternative 1: Multi-Agent LOPRD Generation:** Considered having separate agents for User Stories, FRs, NFRs.
    *   **Pros:** Potentially deeper focus for each agent.
    *   **Cons:** Increased orchestration complexity, potential for inconsistencies between LOPRD sections, higher overhead.
    *   **Reason for not choosing:** A single, powerful `ProductAnalystAgent_v1` with a comprehensive prompt is expected to be more coherent and efficient for initial LOPRD generation. Specialization can be introduced later if needed for refinement.
*   **Alternative 2: Direct Pydantic Model Output from LLM:** Instead of JSON string + validation, try to get the LLM to directly output data that can be parsed into a Pydantic model for LOPRD.
    *   **Pros:** Could simplify validation if LLM is highly reliable with complex structured output.
    *   **Cons:** LLMs are often less reliable with very complex nested structures and strict enum adherence directly; might require more sophisticated prompting or output parsing/correction logic. JSON is a more universal and robust intermediate format.
    *   **Reason for not choosing (initially):** Starting with JSON string output and separate validation is a more robust initial approach. Exploring direct Pydantic output with tools like `Instructor` could be a future enhancement.

## 13. Open Issues & Future Work

*   **Issue 1:** Scalability of LOPRD schema: If the `loprd_schema.json` becomes extremely large, directly injecting it into the LLM prompt might hit token limits. May need strategies for schema summarization or providing only relevant schema sections.
*   **Issue 2:** Handling highly ambiguous or contradictory `refined_user_goal.md` inputs effectively.
*   **Future Enhancement 1:** Incorporate more sophisticated confidence scoring mechanisms (e.g., analyzing LLM token probabilities if accessible).
*   **Future Enhancement 2:** Enable the agent to proactively ask clarifying questions if major ambiguities are detected, rather than just noting them (would require an interaction mechanism with an orchestrator or user).
*   **Future Enhancement 3:** Versioning of LOPRDs within ChromaDB and ability to diff versions.

---
*This is a living document.*
*Last updated: YYYY-MM-DD by Gemini Assistant* 