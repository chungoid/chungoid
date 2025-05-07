# Chungoid — Model-Context-Protocol (MCP) Server

<!--[BADGES]-->
<!-- CI badge: replace <owner>/<repo> with your GitHub slug if the fork is elsewhere -->
[![Tests](https://github.com/chungoid/chungoid/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/chungoid/chungoid/actions/workflows/test.yml)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)

> The **Chungoid MCP server** orchestrates an AI-driven, stage-based workflow that turns an idea into a production-ready codebase.  This README tells you *exactly* how to get a project bootstrapped and running in minutes.

---

## TL;DR — Bootstrap a New Project with Chungoid

1.  **Install `pipx`** (if you haven't already):
    *   Linux (Debian/Ubuntu): `sudo apt update && sudo apt install pipx`
    *   macOS (using Homebrew): `brew install pipx`
    *   Other systems: See [pipx installation guide](https://pypa.github.io/pipx/installation/).
    Then run `pipx ensurepath` to add its binary location to your `PATH` (you might need to open a new terminal).

2.  **Install the `chungoid` CLI tools**:
    ```bash
    # Clone the chungoid repository
    git clone https://github.com/chungoid/chungoid.git
    cd chungoid

    # Install using pipx (from the root of the 'chungoid' repository clone)
    # This makes `chungoid-server` globally available.
    pipx install .
    ```

3.  **Start your new project**:
    ```bash
    # Create and navigate to your new project directory (OUTSIDE the chungoid repo clone)
    # Example: if your `chungoid` repo clone is at `~/dev/chungoid`,
    # you might create your new project at `~/projects/my_new_chungoid_project`.
    mkdir -p ~/my_new_chungoid_project 
    cd ~/my_new_chungoid_project

    # Configure your MCP Client (e.g., Cursor) to use `chungoid-server`.
    # See "Getting Started" section below for an example `mcp.json` snippet for Cursor

    # Interact with Chungoid via your MCP Client (e.g., Cursor):
    # - The server should start based on your client's MCP configuration when you open the project.
    # - Use `@chungoid set_project_context` to set projects current working directory.
    # - Use tools like `@chungoid initialize_project` to set up the .chungoid directory.
    # - Ask your cursor agent what steps you should take next if you're in doubt!
    # - Then, begin the workflow with `@chungoid prepare_next_stage`.
    ```
Your new project directory (`~/my_new_chungoid_project`) will now have a `.chungoid/` folder, ready for Stage –1.

---

## What it Does

The server manages a multi-stage process where an AI agent (like you!) interacts via defined tools to perform tasks like design, planning, implementation, validation, and release preparation for a target software project.

## Getting Started: Initializing a Project

*   **Installation:** First, install the `chungoid-server` and its tools using `pipx` by cloning the `chungoid` repository and running `pipx install .` from its root, as described in the TL;DR section. This makes `chungoid-server` globally available.
*   **MCP Client Configuration:** Configure your MCP client (e.g., Cursor, an IDE plugin) to use `chungoid-server`.
    *   The client needs to know how to start `chungoid-server`.
    *   It should pass the path to your target project directory when starting the server (often as `${workspaceFolder}`).
    *   Example `mcp.json` entry for Cursor:
        ```json
        "chungoid": {
            "command": "chungoid-server",
            "transportType": "stdio",
            "args": [],
            "env": {
                "CHUNGOID_PROJECT_DIR": "${workspaceFolder}",
                "CHROMA_CLIENT_TYPE": "persistent",
                "CHUNGOID_LOGGING_LEVEL": "DEBUG"
                }
            }
            // "CHROMA_MODE": "http", // Example: if using a remote Chroma server
            // "CHROMA_HOST": "localhost",
            // "CHROMA_PORT": "8000"
        ```
*   **Project Initialization via MCP Client:**
    1.  Open your chosen project directory in your MCP client (e.g., open the folder in Cursor).
    2.  If your client starts `chungoid-server` with the correct `--project-dir ${workspaceFolder}`, the context is usually set but run @chungoid `@chungoid set_project_context` in cursor chat to be certain.
    3.  Use the `initialize_project` tool via your client:
        ```
        @chungoid initialize_project
        ```
        This command will instruct the `chungoid-server` (already context-aware of your project directory) to create the `.chungoid/` subdirectory and `project_status.json` if they don't exist.
    4.  Follow your agent's instructions and interact to clarify a well-refined project goal & advance through the stages, starting with `@chungoid prepare_next_stage`.
*   **Cursor Rule (Recommended):** For consistent agent behavior, especially with Cursor, ensure the `chungoid_bootstrap.mdc` rule is in your project's `.cursor/rules/` directory. You can copy it using the `chungoid-export-rule .` command from your project root, or Stage -1 might do it for you.

## The Workflow: Stage –1 → Stage 5 *(optional Stage&nbsp;6)*

Chungoid uses a sequential, stage-based workflow. Each stage focuses on a specific part of development and has defined goals and expected outputs (artifacts).

0.  **Stage –1: Goal Draft & Scope Clarification:** Elicit a clear, bounded project goal, confirm KPIs, and brainstorm candidate libraries.  Output: `goal_draft.md` (+ optional `goal_questions.json`).
1.  **Stage 0: Discovery & Design:** Understand the refined goal, research, create `requirements.md` and `blueprint.md`.
2.  **Stage 1: Design Validation:** Review Stage 0 artifacts, ensure feasibility/clarity, produce `validation_report.json`.
3.  **Stage 2: Implementation Planning:** Create `implementation_plan.md` and `detailed_interfaces.md` based on the validated design.
4.  **Stage 3: Implementation & Unit Testing:** Write code incrementally according to the plan, add unit tests, run static analysis. Produce code artifacts and reports (`static_analysis_report.json`, `unit_test_report.json`).
5.  **Stage 4: Validation & QA:** Perform integration testing, security checks, etc., on the implemented code. Produce reports (`integration_report.json`, etc.).
6.  **Stage 5: Release Preparation:** Finalize `README.md`, `docs/`, packaging files, and `release_notes.md`.
7.  **Stage 6 (Post-Release Retrospective, *optional*):** Run additional CI tests, gather metrics, and document lessons learned in `retrospective.md`.  Helps feed improvements back into Stage –1 for the next project cycle.

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
0.1.  Optional: Use chungoid mcp_chungoid_export_cursor_rule in cursor chat.
0.2.  Optional: Select add context (ctrl+alt+p in cursor chat) and select add new rule, name it: chungoid
0.3.  Optional: From the add context menu select chungoid.mdc to apply it as a new rule to better follow the system.
1.  Write a brief summary of the goal you wish to acheive with your software project in goal.txt
2.  Send agent request to `set_project_context` or `initialize_project`
3.  Send agent request to `prepare_next_stage`
4.  Refine your goal.txt by discussing with the agent how to optimize it for success.
5.  Send agent request to `execute_next_stage`
6.  Follow the stages workflow and let the agent guide you through the phases & use its chungoid tools to 
store state artifacts, research artifacts, documentation artifacts, etc. 
7.  Use `get_project_status` at any point in time to reflect on current state and next steps if you need guidance.
8. Use `set_pending_reflections` `load_reflections` and `store_reflections` to store context in the projects `.chungoid/chroma_db` subdirectory.

## Key Concepts

*   **Project Status (`.chungoid/project_status.json`):** Tracks the history of stage runs and their outcomes (PASS/FAIL).
*   **Artifacts:** Files generated or modified during a stage (code, documents, reports).
*   **Reflections:** Your thoughts, analysis, or rationale recorded during a stage, stored for context.
*   **Context:** Information (status, artifacts, reflections) gathered and potentially passed into stage prompts.

## Compute Stacking — Optional Chungoid Power-Ups

> *Everything below is optional.*  Chungoid-core runs out-of-the-box with plain files on disk.  But if you enable these extra components the agent can "stack" new kinds of computation and context, giving it **exponential leverage** during the stage workflow.

| Add-on | What it brings | Install | Docs / Repo |
|--------|---------------|---------|-------------|
| **ChromaDB** | Embedded or HTTP vector store for long-term reflections, planning docs, and fetched library docs.  Enables fast semantic search so later stages recall past decisions instead of repeating analysis. | Built-in (install via `pip install chromadb` — already in requirements.txt) | https://docs.trychroma.com |
| **MCP Sequential Thinking** | Tool that forces the agent to reason step-by-step, self-critique, and verify outputs before showing them to you.  Early stages (-1, 0, 1) use it to spot gaps; later stages (2-5) use it for code/test validation. | `instructions at ->` | https://github.com/modelcontextprotocol/servers/ |
| **Context7 Library Docs** | On-demand retrieval of third-party API documentation (`resolve-library-id` → `get-library-docs`).  Reduces hallucinations and saves you from hunting docs manually. | `instructions at ->` | https://github.com/upstash/context7 |

> These packages are **maintained by their own teams**.  Chungoid merely detects and uses them if they're present; see each repository for license and security details.

Chungoid calls this synergy **compute stacking**: each layer (vector memory, disciplined reasoning, live docs) augments the next, letting the agent solve harder problems with fewer tokens and less user micro-management. Enable as many layers as your environment allows; the workflow adapts automatically.

---

## Development

This `chungoid` repository contains the Python package for the `chungoid-server` and its core logic. For development *of* this `chungoid` Python package:

1.  Clone this `chungoid` repository from GitHub.
2.  Navigate to the root of your cloned `chungoid` repository.
3.  Create and activate a Python virtual environment here:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
4.  Install in editable mode with development dependencies (from the root of the `chungoid` repository):
    ```bash
    pip install -e \".[dev,test]\"
    ```
5.  Set up pre-commit hooks:
    ```bash
    pre-commit install
    ```
Now you can make changes to the code (e.g., within `src/chungoid/`). Run tests with `pytest` (also from the root of the `chungoid` repository).

Development of the overarching meta-level Chungoid MCP (the system that *uses* the `chungoid` Python package for its own development) may occur in a separate, broader project structure (e.g., the `chungoid-mcp` project if that's the name, or the root of the `chungoid` repository if it also serves as the meta-project).

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

## Bringing the Chungoid Cursor Rule into **your** project  

Chungoid-core ships a master Cursor rule file  
`chungoid_bootstrap.mdc` (see `chungoid_core/cursor_rules/`).  
It keeps the agent behaviour consistent across every workspace.  
Below are two equally simple ways to copy the rule into a new project, depending on **how you run Chungoid**.

### If you are a *CLI-only* user  
(You launch the server from the shell and talk to it via an MCP client on the command-line.)

```bash
# One-liner from project root – copies rule into .cursor/rules/
chungoid-export-rule              # installed automatically with the package

# Want a custom location?
chungoid-export-rule ./some/other/path
```
The helper is installed as a console script when you `pip install chungoid-core`.
It will create the target directory if it doesn't exist and prints the path of the copied file.

### If you are an IDE / Cursor user  
(Your IDE starts the MCP server via `launch_server.sh`.)

Nothing to install manually:  **Stage –1** checks for the rule and auto-copies it if missing by calling the built-in tool handler:

```tool_code
print(default_api.mcp_chungoid_export_cursor_rule(dest_path=".cursor/rules"))
```

You'll see a confirmation in your chat pane:
```
✓ Copied chungoid_bootstrap.mdc → .cursor/rules/
```
If you ever delete the file, just re-run Stage –1 or call the tool handler yourself.

> **Why is this needed?**  The rule embeds Golden Principles (stage fidelity, reflection requirements, doc-flow, etc.) that keep the agent on track. Storing it inside the project means you can tweak it locally without touching the global package, while still starting from the canonical version.

--- 

## Enhancements (May 2025)

| Area | What changed | Where/Artifacts |
|------|--------------|-----------------|
| **New Micro-Stage** | Added **Stage –1: Goal Draft & Scope Clarification** with sequential-thinking prompt. | `server_prompts/stages/stage_minus1_goal_draft.yaml` |
| **Prompt Refinement** | Injected explicit `mcp_sequentialthinking_sequentialthinking` reflection contracts into **Stage 0** & **Stage 1** prompts. | `stage0.yaml`, `stage1.yaml` |
| **Library-Docs Flow** | Integrated Context7 retrieval + fallback `doc_requests.yaml` (with JSON schema + validator script). | `dev/schemas/doc_requests_schema.yaml`, `dev/scripts/validate_doc_requests.py` |
| **Cursor Rule** | Packaged `chungoid_bootstrap.mdc` + helper `chungoid-export-rule` for one-liner installation. | `.cursor/rules/`, console entry-point |
| **Tests & CI** | Added pytest cases ensuring every stage references the sequential-thinking tool and doc-request schema passes.  Updated GitHub Actions matrix for Py 3.11/3.12. | `tests/unit/`, `.github/workflows/python-tests.yml` |
| **Compute Stacking Docs** | Added "🚀 Compute Stacking — Optional Power-Ups" section (ChromaDB, MCP Sequential Thinking, Context7). | README section above |

> Use these notes to track breaking changes when upgrading Chungoid-core. 

## License

Chungoid-core is released under the **GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)**.

This strongest copyleft license ensures:

* Any distributed or network-hosted derivative must remain open-source under the same terms.
* Contributors and original authors receive credit via preserved copyright headers.

Refer to the [`LICENSE`](../LICENSE) file for the full legal text and how to apply headers in source files.

--- 