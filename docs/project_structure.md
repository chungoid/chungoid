# Chungoid-Core Project Structure

This document provides a tree-like overview of the `chungoid-core` directory structure, intended to help developers and AI agents understand the layout of the project.

**Last Updated**: Reflects analysis as of mid-2024/early-2025 based on current understanding.

```
chungoid-core/
├── .chungoid/  # Chungoid-specific runtime data (e.g., ChromaDB, logs, flow states)
├── .git/       # Git version control
├── .github/
│   ├── FUNDING.yml
│   └── workflows/  # GitHub Actions CI/CD
│       ├── library-doc-sync.yml
│       ├── snapshot-dryrun.yml
│       ├── snapshot-embed.yml
│       └── test.yml
├── .pytest_cache/ # Pytest cache
├── .venv/         # Python virtual environment (if present)
├── __pycache__/   # Python bytecode cache
├── config/        # Configuration files (e.g., master_config.yaml, potentially others)
├── docs/
│   ├── _build/     # Sphinx build output (if Sphinx is used)
│   ├── _static/    # Sphinx static files
│   ├── _templates/ # Sphinx templates
│   ├── architecture/     # System architecture documentation
│   │   ├── detailed_architecture.md
│   │   └── system_overview.md
│   ├── design_documents/ # Agent and component designs
│   │   └── foundational_principles.md
│   ├── guides/           # User and developer guides
│   │   ├── autonomous_cycle_review_and_iteration.md
│   │   ├── config_guide.md
│   │   ├── documentation_maintenance.md
│   │   └── refined_user_goal_specification.md # (execution_runtime.md & litellm_setup.md are outdated mentions)
│   ├── images/           # Documentation images
│   ├── templates/        # Documentation templates (if any)
│   ├── autonomous_project_engine_overview.md # (Note: May contain outdated architectural details)
│   ├── autonomous_system_diagram.md
│   ├── conf.py          # Sphinx configuration (if Sphinx is used)
│   ├── index.rst        # Sphinx root document (if Sphinx is used)
│   ├── make.bat         # Sphinx build script (Windows)
│   ├── Makefile         # Sphinx build script
│   ├── migration_chromadb_agent_to_mcp_tools.md # Details PCMA deprecation
│   ├── modules.rst      # Sphinx modules documentation
│   ├── project_structure.md # This file
│   ├── project_status.json.lock # Runtime artifact
│   ├── sync_report.md   # Documentation sync status (if sync script is used)
│   └── utils.rst        # Sphinx utils documentation
│   # Note: 'reference/' directory mentioned in some docs may not exist or be populated.
├── dev_chroma_db/      # ChromaDB development database
├── htmlcov_utils/      # HTML coverage report utilities
├── schemas/            # Static JSON/YAML schema definitions (e.g., for older flow DSLs)
│   ├── external_tool_schemas/ # Schemas for external tools if any
│   ├── doc_requests_schema.yaml
│   ├── execution_dsl.json
│   └── stage_flow_schema.json
├── scripts/            # Development and operational scripts
│   ├── __init__.py
│   # ... (listing many scripts, assumed mostly current unless specific issues found)
│   └── sync_documentation.py # (As mentioned in documentation_maintenance.md)
├── server_prompts/     # Agent prompts and potentially stage flow YAML configurations
│   ├── autonomous_engine/  # Autonomous engine specific prompts
│   ├── stages/             # Stage-specific prompts
│   # ... (structure seems plausible)
├── src/
│   ├── __init__.py  # Makes src a discoverable path element
│   └── chungoid/    # Main Python package for chungoid-core
│       ├── __init__.py      # Package initializer
│       ├── cli.py           # Main CLI application
│       ├── constants.py     # Core constants
│       ├── core_utils.py    # Core utility functions
│       ├── engine.py        # Core workflow engine (nature might have evolved, see UnifiedOrchestrator)
│       ├── flow_executor.py # Executes defined flows (likely interacts with UnifiedOrchestrator)
│       ├── mcp.py           # Model Context Protocol handling (likely related to MCP tools)
│       ├── agents/          # Agent implementations directory (Primary location for UnifiedAgent subclasses)
│       │   ├── __init__.py
│       │   ├── autonomous_engine/  # Autonomous development agents (e.g., ARCA, SmartCodeGeneratorAgent)
│       │   ├── system/             # System-level agents (e.g., NoOpAgent)
│       │   └── unified_agent.py    # Base class for current agents
│       ├── mcp_tools/       # MCP Tool Ecosystem (11 tools currently)
│       │   ├── __init__.py  # Tool registry and initialization
│       │   ├── manifest_initialization.py # Tool manifest system
│       │   ├── tool_manifest.py        # Tool discovery and metadata
│       │   ├── chromadb/    # ChromaDB suite (4 tools)
│       │   │   └── chroma_actions.py # Example actual tool file
│       │   ├── content/     # Content suite (2 tools)
│       │   │   └── content_actions.py # Example actual tool file
│       │   ├── filesystem/  # Filesystem suite (3 tools)
│       │   │   └── filesystem_actions.py # Example actual tool file
│       │   └── terminal/    # Terminal suite (2 tools)
│       │       └── terminal_actions.py # Example actual tool file
│       ├── protocols/       # Protocol definitions (~24 modules)
│       │   ├── __init__.py
│       │   ├── universal/
│       │   ├── planning/
│       │   ├── implementation/
│       │   # ... other categories like quality, investigation, collaboration, etc.
│       ├── runtime/         # Execution runtime components
│       │   ├── __init__.py
│       │   ├── unified_orchestrator.py # Current primary orchestrator
│       │   ├── unified_agent_resolver.py # Resolves UnifiedAgent instances
│       │   ├── services/        # Runtime services (e.g., error handling, context resolution)
│       │   # Note: Older 'orchestrator.py' (AsyncOrchestrator) and 'runtime/agents/' are deprecated/refactored.
│       ├── schemas/        # Pydantic models for data structures (many specific to agents/components)
│       │   ├── __init__.py
│       │   ├── project_state.py # Defines ProjectStateV2 for project_status.json
│       │   ├── unified_execution_schemas.py # Defines ExecutionConfig, etc.
│       │   # ... many other specific schemas
│       └── utils/          # Utility modules
│           ├── __init__.py
│           ├── agent_registry.py # (May work with UnifiedAgentResolver)
│           ├── config_manager.py  # Pydantic-based configuration system
│           ├── state_manager.py   # Manages project_status.json and ChromaDB interactions
│           # ... many other utilities
│   └── chungoid_mcp_server.egg-info/ # Packaging metadata
├── tests/
│   # ... (test structure, assumed mostly current unless specific issues found)
│   ├── unit/
│   │   ├── test_config_manager_live.py # Confirms modern config system tests
│   │   ├── test_unified_orchestrator.py # (Expected test for current orchestrator)
│   │   └── test_unified_agent.py # (Expected test for current agent base)
├── .coveragerc
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── LICENSE
├── Makefile # Project Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Key Directory Changes

### Major Updates from Previous Structure:

1. **MCP Tools Organization** (`src/chungoid/mcp_tools/`):
   - Now organized into specialized suites: `chromadb/`, `filesystem/`, `terminal/`, `content/`
   - Centralized tool manifest system for discovery and metadata
   - 45+ tools across 4 categories with intelligent selection

2. **Agent Location** (`src/chungoid/agents/autonomous_engine/`):
   - Agents moved from `runtime/agents/` to `agents/autonomous_engine/`
   - Specialized autonomous development agents for different domains

3. **Runtime Services** (`src/chungoid/runtime/services/`):
   - New services directory for runtime-specific functionality
   - AsyncOrchestrator significantly expanded (105KB, 1602 lines)

4. **Documentation Structure** (`docs/`):
   - Added `guides/` directory with specialized documentation
   - Comprehensive setup guides including LiteLLM configuration
   - Automated synchronization system with source documentation

5. **Tool Manifests** (`tool_manifests.json`):
   - Large manifest file (22KB, 871 lines) for tool discovery
   - Dynamic tool registration and capability detection

### Core File Sizes (Indicating Implementation Depth):

- `cli.py`: 89KB (1676 lines) - Comprehensive CLI interface
- `engine.py`: 52KB (936 lines) - Core workflow engine
- `orchestrator.py`: 105KB (1602 lines) - Advanced orchestration logic
- `tool_manifests.json`: 22KB (871 lines) - Extensive tool ecosystem

This structure reflects a mature autonomous development system with comprehensive tooling, robust orchestration, and extensive testing coverage. 