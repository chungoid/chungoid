# Chungoid — Model-Context-Protocol (MCP) Server

<!--[BADGES]-->
![Tests](https://github.com/your-org/your-repo/actions/workflows/test.yml/badge.svg)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)

> The **Chungoid MCP server** orchestrates an AI-driven, stage-based workflow that turns an idea into a production-ready codebase.  This README tells you *exactly* how to get a project bootstrapped and running in minutes.

---

## TL;DR — Bootstrap a New Project in 30 Seconds

```bash
# 0.  Install dependencies in a fresh venv
pip install -e .[dev]  # from repo root

# 1.  Start the embedded ChromaDB (optional – script handles it automatically)
make chroma-dev-server &

# 2.  Launch the server (stdio mode)
python chungoid-core/chungoidmcp.py &  # or `python -m chungoidmcp`

# 3.  In your chat client / CLI, initialise a directory:
@chungoid initialize_project target_directory="$(pwd)"

# 4.  Ask for the first stage prompt:
@chungoid prepare_next_stage
```

*That's it.*  Your working directory now holds a `.chungoid/` folder with `project_status.json`, and Stage 0 is ready to walk you through discovery and design.

---

## What it Does

The server manages a multi-stage process where an AI agent (like you!) interacts via defined tools to perform tasks like design, planning, implementation, validation, and release preparation for a target software project.

## Getting Started: Initializing a Project

To begin a new project managed by Chungoid:

1.  **Navigate:** Open your terminal *in the directory* where you want your new project to reside.
2.  **Initialize:** Use the Chungoid client (e.g., Cursor chat) connected to this server and run the initialization:
    ```
    @chungoid initialize_project target_directory="."
    ```
    *   (Replace `.` with the full path if you aren't running the client from the target directory itself).
    *   This creates a `.chungoid/` directory containing essential state files (`project_status.json`).
    *   It also automatically sets the project context for your current session.

## The Workflow: Stages 0-5

Chungoid uses a sequential, stage-based workflow. Each stage focuses on a specific part of development and has defined goals and expected outputs (artifacts).

1.  **Stage 0: Discovery & Design:** Understand the goal, research, create `requirements.md` and `blueprint.md`.
2.  **Stage 1: Design Validation:** Review Stage 0 artifacts, ensure feasibility/clarity, produce `validation_report.json`.
3.  **Stage 2: Implementation Planning:** Create `implementation_plan.md` and `detailed_interfaces.md` based on the validated design.
4.  **Stage 3: Implementation & Unit Testing:** Write code incrementally according to the plan, add unit tests, run static analysis. Produce code artifacts and reports (`static_analysis_report.json`, `unit_test_report.json`).
5.  **Stage 4: Validation & QA:** Perform integration testing, security checks, etc., on the implemented code. Produce reports (`integration_report.json`, etc.).
6.  **Stage 5: Release Preparation:** Finalize `README.md`, `docs/`, packaging files, and `release_notes.md`.

## Interaction: Using the Tools

You interact with the server using specific tools via your client (e.g., `@chungoid tool_name ...`). Key tools include:

*   **`initialize_project`**: (Human driven) As shown above, Sets up a new project directory.
*   **`set_project_context`**: (Human Driven)Tells the server which project directory subsequent commands should apply to for your session. Useful if managing multiple projects or if context is lost.
*   **`get_project_status`**: (Human Driven) Retrieves the current status, including completed stages and runs.
*   **`load_reflections`**: (Self/Agent/Engine Driven) Loads reflections/notes stored from previous stages.
*   **`retrieve_reflections`**:(Self/Agent/Engine Driven) Searches stored reflections for specific information.
*   **`prepare_next_stage`**: (Self/Agent/Engine Driven) Determines the next stage based on the project status and provides you with the prompt (role, goals, tasks) for that stage.
*   **`get_file relative_path`**: (Self/Agent/Engine Driven) Reads the content of a file within the *currently set project context*.
*   **`set_pending_reflection`**: (Self/Agent/Engine Driven) *Required before submitting.* Stages your reflection text temporarily.
*   **`submit_stage_artifacts`**: (Self/Agent/Engine Driven) Submits the results of a stage. This updates the project status and stores artifact/reflection context. *Note: The `reflection_text` is picked up automatically from the previous `set_pending_reflection` call.*

**Typical Flow:**

1.  Write a brief summary of the goal you wish to acheive with your software project in goal.txt
2.  Send agent request to `set_project_context` or `initialize_project`
3.  Send agent request to `prepare_next_stage`
4.  Refine your goal.txt by discussing with the agent how to optimize it for success.
5.  Send agent request to `execute_next_stage`
6.  Follow the stages workflow and let the agent guide you through the phases & use its chungoid tools to 
store state artifacts, research artifacts, documentation artifacts, etc. 
7.  Use `get_project_status` at any point in time to reflect on current state and next steps if you need guidance.

## Key Concepts

*   **Project Status (`.chungoid/project_status.json`):** Tracks the history of stage runs and their outcomes (PASS/FAIL).
*   **Artifacts:** Files generated or modified during a stage (code, documents, reports).
*   **Reflections:** Your thoughts, analysis, or rationale recorded during a stage, stored for context.
*   **Context:** Information (status, artifacts, reflections) gathered and potentially passed into stage prompts.

## Development

This project structure contains the server implementation. Development *of* the server (meta-development) occurs in the parent `chungoid-mcp` (currently private) repository structure. This is autonomously built from within an abstraction layer which uses a modified version of the chungoid workflow, and soon.. A2A protocol.

## Origin Story

In the early days, ambitious AI projects were often chaotic. Developers faced a swirling vortex of high-level goals, vague requirements, shifting dependencies, and endless potential paths. Progress stalled in "analysis paralysis," the sheer complexity overwhelming any attempt at structured development. They needed a new paradigm.

The breakthrough came from a simple observation: even the most complex system could be broken down. Like eating an elephant one bite at a time, the team realized they needed to isolate manageable "chunks" of the problem – a specific feature, a single module, a defined stage of development. This wasn't just task breakdown; it was about creating self-contained units of work with clear inputs and outputs.

Just chunking wasn't enough. They needed a *process* to handle each chunk consistently: define it (Stage 0), validate it (Stage 1), plan its implementation (Stage 2), build and test it (Stage 3), validate the build (Stage 4), and prepare it for integration (Stage 5). Crucially, the system needed to *learn* from each chunk – reflections stored in a persistent memory (like ChromaDB) to inform the next. This meta-cognitive loop was vital.

They weren't just creating *chunks*; they were designing a system *that operated on chunks*. This system had its own lifecycle, its own internal state, its own memory, and distinct operational phases (the stages). It felt less like a static plan and more like an autonomous entity designed specifically to *process* these chunks. The suffix "-oid" came to mind – signifying something "like" or "resembling" a self-contained, purposeful entity. It wasn't just *a* chunk; it was the **Chunk-Processor**, the **Chunk-Handler**, the **Chunk-oid**.

During a late-night whiteboard session, mapping out the flow between stages, agents, and the reflection database, someone drew a box around the entire process – the State Manager, the Prompt Manager, the Stage Executor, the Memory. "This whole thing," they declared, gesturing at the diagram, "it's the... the *Chungoid*. It takes the big messy goal, breaks it into chunks, digests each one through the stages, learns, and moves on."

The name stuck. "Chungoid" came to represent not just the act of chunking, but the entire **meta-cognitive, agent-driven framework** designed to systematically consume complexity through sequential, reflective stages. It embodied the structured approach, the learning capability, and the staged progression – the intelligent system that brings order to the chaos of creation.

## Logging Configuration (Environment Overrides)

Chungoid-core uses a centralised logging helper (`utils.logger_setup`) that
reads settings from **config.yaml** but **can be overridden via environment
variables** at runtime.  This is handy when you need verbose logs for
troubleshooting or JSON logs in CI without modifying repo-tracked files.

| Variable | Purpose | Example |
|----------|---------|---------|
| `CHUNGOID_LOGGING_LEVEL` | Override `logging.level` in config. Accepts standard Python levels (`DEBUG`, `INFO`, etc.). | `export CHUNGOID_LOGGING_LEVEL=DEBUG` |
| `CHUNGOID_LOGGING_FORMAT` | Set formatter: `text` (default) or `json`. | `export CHUNGOID_LOGGING_FORMAT=json` |
| `CHUNGOID_LOGGING_FILE` | Path for rotating file handler. Leave blank to disable file logging. | `export CHUNGOID_LOGGING_FILE=/tmp/chungoid.log` |
| `CHUNGOID_LOGGING_MAX_BYTES` | Max size (bytes) before rotation. | `export CHUNGOID_LOGGING_MAX_BYTES=1048576` |
| `CHUNGOID_LOGGING_BACKUP_COUNT` | How many rotated files to keep. | `export CHUNGOID_LOGGING_BACKUP_COUNT=3` |

When any of these variables are present, `utils.config_loader` injects the
value into the runtime config before `utils.logger_setup.setup_logging()` is
called. 