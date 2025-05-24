# Chungoid-Core Project Structure

This document provides a tree-like overview of the `chungoid-core` directory structure, intended to help developers and AI agents understand the layout of the project.

**Last Updated**: 2025 (Current Implementation)

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
├── config/        # Configuration files
├── docs/
│   ├── _build/     # Sphinx build output
│   ├── _static/    # Sphinx static files
│   ├── _templates/ # Sphinx templates
│   ├── architecture/     # System architecture documentation
│   ├── design_documents/ # Agent and component designs
│   ├── guides/           # User and developer guides
│   │   ├── autonomous_cycle_review_and_iteration.md
│   │   ├── documentation_maintenance.md
│   │   ├── execution_runtime.md
│   │   └── litellm_setup.md
│   ├── images/           # Documentation images
│   ├── reference/        # Reference documentation
│   ├── templates/        # Documentation templates
│   ├── autonomous_project_engine_overview.md
│   ├── autonomous_system_diagram.md
│   ├── conf.py          # Sphinx configuration
│   ├── index.rst        # Sphinx root document
│   ├── make.bat         # Sphinx build script (Windows)
│   ├── Makefile         # Sphinx build script
│   ├── modules.rst      # Sphinx modules documentation
│   ├── project_structure.md # This file
│   ├── project_status.json.lock # Runtime artifact
│   ├── sync_report.md   # Documentation sync status
│   └── utils.rst
├── dev_chroma_db/      # ChromaDB development database
├── htmlcov_utils/      # HTML coverage report utilities
├── schemas/            # Static JSON/YAML schema definitions
│   ├── doc_requests_schema.yaml
│   ├── execution_dsl.json
│   └── stage_flow_schema.json
├── scripts/            # Development and operational scripts
│   ├── __init__.py
│   ├── add_docs.py
│   ├── agent_registry.py
│   ├── check_chroma.py
│   ├── check_metrics_store.py
│   ├── core_mcp_client.py
│   ├── coverage_audit.py
│   ├── doc_helpers.py
│   ├── embed_changed_files.py
│   ├── embed_core_snapshot.py
│   ├── embed_meta.py
│   ├── export_jsonschemas.py
│   ├── extract_tool_spec.py
│   ├── flow_registry_cli.py
│   ├── flow_run.py
│   ├── ingest_llms_files.py
│   ├── list_library_embeddings.py
│   ├── log_process_feedback.py
│   ├── metrics_cli.py
│   ├── migrate_stage_flows.py
│   ├── prompt_linter.py
│   ├── prompt_renderer.py
│   ├── reregister_core_test_agent.py
│   ├── seed_flow_registry.py
│   ├── snapshot_core_tarball.py
│   ├── sync_agent_registry.py
│   ├── sync_library_docs.py
│   ├── test_chroma_persist.py
│   └── test_tool_runner.py
├── server_prompts/     # Agent prompts and stage flow YAML configurations
│   ├── autonomous_engine/  # Autonomous engine specific prompts
│   ├── stages/             # Stage-specific prompts
│   │   ├── stage_minus1_goal_draft.yaml
│   │   ├── stage0.yaml
│   │   ├── stage1.yaml
│   │   ├── stage2.yaml
│   │   ├── stage3.yaml
│   │   ├── stage4.yaml
│   │   └── stage5.yaml
│   ├── common_template.yaml
│   ├── common.yaml
│   └── initial_status.json
├── src/
│   ├── __init__.py  # Makes src a discoverable path element
│   └── chungoid/    # Main Python package for chungoid-core
│       ├── __init__.py      # Package initializer
│       ├── cli.py           # Main CLI application (89KB, 1676 lines)
│       ├── constants.py     # Core constants
│       ├── core_utils.py    # Core utility functions
│       ├── engine.py        # Core workflow engine (52KB, 936 lines)
│       ├── flow_executor.py # Executes defined flows
│       ├── mcp.py           # Model Context Protocol handling (13KB, 273 lines)
│       ├── agents/          # Agent implementations directory
│       │   └── autonomous_engine/  # Autonomous development agents
│       ├── mcp_tools/       # MCP Tool Ecosystem (45+ tools)
│       │   ├── __init__.py  # Tool registry and initialization
│       │   ├── manifest_initialization.py # Tool manifest system
│       │   ├── tool_manifest.py        # Tool discovery and metadata
│       │   ├── chromadb/    # ChromaDB suite (17 tools)
│       │   │   ├── collection_management_tools.py
│       │   │   ├── document_operations_tools.py
│       │   │   ├── project_integration_tools.py
│       │   │   └── reflection_tools.py
│       │   ├── content/     # Content suite (8 tools)
│       │   │   ├── content_generation_tools.py
│       │   │   ├── web_content_tools.py
│       │   │   └── management_tools.py
│       │   ├── filesystem/  # Filesystem suite (12 tools)
│       │   │   ├── file_operations_tools.py
│       │   │   ├── project_tools.py
│       │   │   └── advanced_operations_tools.py
│       │   └── terminal/    # Terminal suite (8 tools)
│       │       ├── command_execution_tools.py
│       │       ├── environment_tools.py
│       │       └── security_tools.py
│       ├── runtime/         # Execution runtime components
│       │   ├── __init__.py
│       │   ├── orchestrator.py  # AsyncOrchestrator (105KB, 1602 lines)
│       │   ├── agents/          # Runtime agent base classes
│       │   │   ├── __init__.py
│       │   │   ├── agent_base.py
│       │   │   ├── core_code_generator_agent.py
│       │   │   ├── core_test_generator_agent.py
│       │   │   ├── master_planner_agent.py
│       │   │   ├── system_file_system_agent.py
│       │   │   ├── system_master_planner_agent.py
│       │   │   ├── system_master_planner_reviewer_agent.py
│       │   │   ├── system_test_runner_agent.py
│       │   │   └── mocks/        # Mock agents for testing
│       │   │       ├── mock_code_generator_agent.py
│       │   │       ├── mock_human_input_agent.py
│       │   │       ├── mock_system_requirements_gathering_agent.py
│       │   │       ├── mock_test_generation_agent.py
│       │   │       ├── mock_test_generator_agent.py
│       │   │       └── testing_mock_agents.py
│       │   └── services/        # Runtime services
│       ├── schemas/        # Pydantic models for data structures
│       │   ├── __init__.py
│       │   ├── a2a_schemas.py
│       │   ├── agent_code_generator.py
│       │   ├── agent_code_integration.py
│       │   ├── agent_master_planner.py
│       │   ├── agent_master_planner_reviewer.py
│       │   ├── agent_mock_code_generator.py
│       │   ├── agent_mock_human_input.py
│       │   ├── agent_mock_system_requirements_gathering.py
│       │   ├── agent_mock_test_generator.py
│       │   ├── agent_system_test_runner.py
│       │   ├── agent_test_generation.py
│       │   ├── agent_test_generator.py
│       │   ├── common.py
│       │   ├── common_enums.py
│       │   ├── errors.py
│       │   ├── flows.py
│       │   ├── master_flow.py # Defines MasterExecutionPlan
│       │   ├── metrics.py
│       │   └── user_goal_schemas.py
│       └── utils/          # Utility modules
│           ├── __init__.py
│           ├── a2a_utils.py
│           ├── agent_registry.py
│           ├── agent_registry_meta.py
│           ├── agent_resolver.py
│           ├── analysis_utils.py
│           ├── chroma_client_factory.py
│           ├── chroma_utils.py
│           ├── config_manager.py  # Modern Pydantic-based configuration system
│           ├── core_snapshot_utils.py
│           ├── exceptions.py
│           ├── feedback_store.py
│           ├── flow_api.py
│           ├── flow_registry.py
│           ├── flow_registry_singleton.py
│           ├── goal.py
│           ├── llm_provider.py
│           ├── logger_setup.py
│           ├── master_flow_registry.py
│           ├── mcp_server.py
│           ├── memory_vector.py
│           ├── metrics_store.py
│           ├── prompt_manager.py
│           ├── reflection_store.py
│           ├── security.py
│           ├── state_manager.py
│           ├── template_helpers.py
│           └── tool_adapters.py
├── tests/
│   ├── .chungoid/          # Test-specific chungoid data
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures and hooks
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_cli_flow_resume.py
│   ├── legacy/
│   │   └── __init__.py     # Legacy tests
│   ├── test_projects/      # Test project configurations/setups
│   │   ├── __init__.py
│   │   └── test_integration_project/
│   ├── unit/               # Unit tests for individual modules/classes
│   │   ├── .chungoid/      # Unit-test specific chungoid data
│   │   ├── __init__.py
│   │   ├── runtime/        # Runtime component tests
│   │   ├── test_a2a_dev_cli.py
│   │   ├── test_a2a_utils.py
│   │   ├── test_agent_registry.py
│   │   ├── test_agent_resolver.py
│   │   ├── test_analysis_utils.py
│   │   ├── test_chroma_modes.py
│   │   ├── test_chroma_utils.py
│   │   ├── test_chroma_utils_live.py
│   │   ├── test_cli_utils.py
│   │   ├── test_config_manager_live.py  # Tests for new configuration system
│   │   ├── test_core_code_generator_agent.py
│   │   ├── test_core_mcp_client.py
│   │   ├── test_core_snapshot_dryrun.py
│   │   ├── test_core_stage_executor_agent.py
│   │   ├── test_core_test_generator_agent.py
│   │   ├── test_doc_requests_schema.py
│   │   ├── test_execution_dsl_validator.py
│   │   ├── test_execution_runtime.py
│   │   ├── test_feedback_store.py
│   │   ├── test_flow_api_endpoint.py
│   │   ├── test_flow_registry.py
│   │   ├── test_flow_run_cli.py
│   │   ├── test_logger_setup_live.py
│   │   ├── test_master_flow_registry.py
│   │   ├── test_mcp_run_endpoint.py
│   │   ├── test_metrics_store.py
│   │   ├── test_orchestrator.py
│   │   ├── test_orchestrator_reviewer_integration.py
│   │   ├── test_prompt_manager.py
│   │   ├── test_prompt_manager_core.py
│   │   ├── test_prompt_manager_live.py
│   │   ├── test_reflection_api.py
│   │   ├── test_reflection_store.py
│   │   ├── test_registry_dispatch_live.py
│   │   ├── test_security.py
│   │   ├── test_security_live.py
│   │   ├── test_snapshot_tarball.py
│   │   ├── test_smoke_coverage.py
│   │   ├── test_stage_flow_schema.py
│   │   ├── test_stage_minus1_prompt.py
│   │   ├── test_stage_minus1_prompt_schema.py
│   │   ├── test_stage_prompts_sequential.py
│   │   ├── test_state_manager.py
│   │   ├── test_state_manager_core.py
│   │   ├── test_state_manager_live.py
│   │   ├── test_system_master_planner_agent.py
│   │   ├── test_template_helpers_live.py
│   │   ├── test_tool_adapters.py
│   │   └── test_validate_planning.py
│   ├── test_doc_manifest.py
│   ├── test_integration.py
│   ├── test_mcp_invoke.py
│   ├── test_mcp_metadata.py
│   └── test_no_rogue_modules.py
├── .coverage           # Coverage data file
├── .gitignore         # Git ignore rules
├── chungoidmcp.py     # Legacy MCP server entry point (deprecated)
├── config.yaml        # Main configuration
├── CONTRIBUTING.md    # Contribution guidelines
├── launch_server.sh   # Script to launch the MCP server
├── LICENSE            # MIT License
├── Makefile           # Build automation
├── pyproject.toml     # Project build/dependency configuration
├── pytest.ini        # Pytest configuration
├── README.md          # Main documentation
├── requirements.txt   # Pinned dependencies
└── tool_manifests.json # MCP tool discovery manifest (22KB, 871 lines)
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