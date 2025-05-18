---
title: "Refined User Goal Specification (refined_user_goal.md)"
category: schema_specification
owner: "Meta Engineering Team"
status: "Draft"
created: YYYY-MM-DD # To be filled by automated update
updated: YYYY-MM-DD # To be filled by automated update
version: "0.1.0"
related_artifacts: ["ProductAnalystAgent_v1", "loprd_schema.json"]
---

# Refined User Goal Specification (`refined_user_goal.md`)

**Version:** 0.1.0

## 1. Introduction

The `refined_user_goal.md` file is a critical input artifact for the `ProductAnalystAgent_v1` within the Autonomous Project Engine. It represents a structured, human-readable, and LLM-parsable definition of a project goal that has undergone initial refinement from a raw user request or internal trigger.

Its purpose is to provide a clear, concise, yet sufficiently detailed starting point for the generation of a comprehensive LLM-Optimized Product Requirements Document (LOPRD).

## 2. File Format

The file MUST be in Markdown format (`.md`) and include a YAML frontmatter section.

## 3. YAML Frontmatter Schema

The YAML frontmatter provides essential metadata about the goal.

```yaml
---
goal_id: string # (Required) Auto-generated or user-provided unique identifier (e.g., UUID).
project_name_suggestion: string # (Required) A concise, descriptive name suggested for the project stemming from this goal.
created_at: string # (Required) Timestamp of creation in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
last_updated_at: string # (Required) Timestamp of the last update in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
source: string # (Optional) Origin of the goal (e.g., "User Input", "Internal System Trigger", "Phase 2 MVP Demo"). Default: "User Input".
version: string # (Required) Semantic version of this goal document itself (e.g., "1.0", "1.1").
---
```

### Field Descriptions:
*   `goal_id` (string, required): A universally unique identifier for this specific goal document.
*   `project_name_suggestion` (string, required): A human-friendly suggested name for the project. This aids in identifying the project through its lifecycle.
*   `created_at` (string, required): The date and time when this goal document was initially created.
*   `last_updated_at` (string, required): The date and time when this goal document was last modified.
*   `source` (string, optional): Describes where the goal originated from. Helpful for tracking and context.
*   `version` (string, required): The version of this `refined_user_goal.md` document, to track its own evolution if it undergoes revisions before LOPRD generation.

## 4. Markdown Content Structure

The body of the Markdown document should follow a structured format using standard Markdown headings.

### 4.1. Title
```markdown
# Refined Project Goal: [Clear Title for the Goal]
```
*   **(Required)** A main H1 title that clearly and concisely states the project goal.

### 4.2. Section 1: Executive Summary
```markdown
## 1. Executive Summary
*(A brief, 1-3 sentence overview of what the project aims to achieve.)*
```
*   **(Required)** A concise summary providing a high-level understanding of the project's primary objective.

### 4.3. Section 2: Detailed Goal Description
```markdown
## 2. Detailed Goal Description
*(A more comprehensive explanation of the project's objectives. This section should elaborate on the problem being solved, the desired outcomes, and the core functionalities envisioned. Use clear, unambiguous language. Bullet points can be used for clarity.)*
```
*   **(Required)** An in-depth explanation of the goal. This section should provide enough detail for the `ProductAnalystAgent_v1` to understand the nuances of what needs to be built.

### 4.4. Section 3: Key Features / Major Components
```markdown
## 3. Key Features / Major Components
*(A list of the primary features or major components that define the scope of this goal. This helps in breaking down the project later.)*
*   Feature/Component 1: Brief description.
*   Feature/Component 2: Brief description.
*   ...
```
*   **(Required)** A bulleted list of the most important features or distinct parts of the intended system or outcome. This helps in scoping and initial breakdown.

### 4.5. Section 4: Target Audience / Users
```markdown
## 4. Target Audience / Users
*(Who is this project for? Describe the primary users and their needs if known.)*
```
*   **(Optional, but Highly Recommended)** Description of the intended users or beneficiaries of the project. Understanding the audience helps in shaping requirements appropriately.

### 4.6. Section 5: Success Criteria (High-Level)
```markdown
## 5. Success Criteria (High-Level)
*(What are the top-level indicators that this project goal has been successfully achieved? These are not detailed acceptance criteria but rather high-level success markers.)*
*   Success Marker 1.
*   Success Marker 2.
*   ...
```
*   **(Optional, but Highly Recommended)** A list of broad conditions or outcomes that would indicate the project successfully met its goal. These are not specific test cases but general achievements.

### 4.7. Section 6: Known Constraints & Limitations
```markdown
## 6. Known Constraints & Limitations
*(List any known constraints (e.g., specific technologies to use/avoid, budget, timeline if applicable from the user's initial request, existing systems to integrate with) or limitations.)*
```
*   **(Optional)** Any restrictions or boundary conditions that are known at the time of goal definition and must be considered during development (e.g., technology stack, budget, deadlines, regulatory compliance).

### 4.8. Section 7: Out of Scope (Initial thoughts)
```markdown
## 7. Out of Scope (Initial thoughts)
*(Optional: Initial thoughts on what is explicitly NOT part of this goal. This will be refined into the LOPRD's scope section.)*
```
*   **(Optional)** A preliminary list of items that are explicitly not intended to be covered by this project goal. This helps in managing expectations and defining boundaries early.

### 4.9. Section 8: (Optional) Key Technologies/Stack Preferences
```markdown
## 8. (Optional) Key Technologies/Stack Preferences
*(If the user or context provides specific technology preferences or requirements, list them here. E.g., "Python backend," "React frontend," "AWS deployment.")*
```
*   **(Optional)** If there are any predefined choices or preferences for technologies, platforms, or tools to be used.

### 4.10. Section 9: (Optional) Initial Assumptions & Ambiguities
```markdown
## 9. (Optional) Initial Assumptions & Ambiguities
*(This section is for capturing initial thoughts during goal refinement. The content here should ideally be separated into `assumptions_and_ambiguities.md` before being fed to the ProductAnalystAgent, but can be jotted down here initially.)*
*   Assumption 1: ...
*   Ambiguity 1 (Needs Clarification): ...
```
*   **(Optional)** A temporary holding place for assumptions made or ambiguities identified during the process of refining the user goal. It is strongly recommended that these points are formalized and moved to a separate `assumptions_and_ambiguities.md` document before the `ProductAnalystAgent_v1` processes the goal, as per the blueprint.

## 5. Example `refined_user_goal.md`

```markdown
---
goal_id: "a1b2c3d4-e5f6-7890-1234-567890abcdef"
project_name_suggestion: "Simple Bitcoin Price Viewer"
created_at: "2024-05-20T10:00:00Z"
last_updated_at: "2024-05-20T10:30:00Z"
source: "User Request via CLI"
version: "1.0"
---

# Refined Project Goal: Create a Simple Web Application to View Bitcoin Price

## 1. Executive Summary
Develop a minimalist web application that displays the current price of Bitcoin in USD, automatically refreshing every 60 seconds. The application should also show a historical price chart for the last 7 days.

## 2. Detailed Goal Description
The primary objective is to provide users with a quick and easy way to check the current Bitcoin price and observe its recent trend. The application will fetch real-time price data from a public cryptocurrency API (e.g., CoinGecko, CoinCap). It will feature a clean, user-friendly interface displaying the current price prominently and a simple line chart for the 7-day historical data.

## 3. Key Features / Major Components
*   Display current Bitcoin (BTC) price in USD.
*   Automatic price refresh every 60 seconds.
*   Display a 7-day historical price chart for BTC/USD.
*   Simple, clean, and responsive user interface.
*   Backend service to fetch data from a public crypto API.

## 4. Target Audience / Users
*   Casual cryptocurrency observers.
*   Individuals wanting a quick check on Bitcoin's price without logging into an exchange.

## 5. Success Criteria (High-Level)
*   The application accurately displays the current Bitcoin price.
*   The price updates automatically as specified.
*   The 7-day historical chart is correctly rendered and informative.
*   The application is accessible and usable on common web browsers (desktop and mobile).

## 6. Known Constraints & Limitations
*   Must use a free, publicly accessible API for price data.
*   The application should be deployable as a simple standalone web service.
*   Focus on simplicity; no user accounts or complex financial analysis tools.

## 7. Out of Scope (Initial thoughts)
*   Tracking prices of other cryptocurrencies.
*   Portfolio management features.
*   Trading capabilities.
*   Advanced charting tools or technical indicators.

## 8. (Optional) Key Technologies/Stack Preferences
*   Backend: Python (Flask or FastAPI preferred).
*   Frontend: HTML, CSS, JavaScript (a simple JS charting library like Chart.js is acceptable).

## 9. (Optional) Initial Assumptions & Ambiguities
*   Assumption: A reliable public API for Bitcoin price data is readily available.
*   Ambiguity: Specific error handling for API downtime needs further definition.
```

## 6. Evolution & Versioning

This specification may evolve. Changes will be tracked via the `version` field in the frontmatter and documented in a changelog if significant structural modifications occur. 