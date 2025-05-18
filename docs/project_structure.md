# Chungoid-Core Project Structure

This document provides a tree-like overview of the `chungoid-core` directory structure, intended to help developers and AI agents understand the layout of the project.

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
├── .venv/         # Python virtual environment
├── docs/
│   ├── _static/   # Sphinx static files
│   ├── _templates/ # Sphinx templates
│   ├── autonomous_system_diagram.md # Mermaid diagram of system
│   ├── conf.py     # Sphinx configuration
│   ├── images/     # Documentation images
│   ├── index.rst   # Sphinx root document
│   ├── make.bat    # Sphinx build script (Windows)
│   ├── Makefile    # Sphinx build script
│   ├── modules.rst # Sphinx modules documentation
│   ├── project_status.json.lock # Lockfile, likely runtime artifact
│   ├── project_structure.md # This file
│   ├── utils.rst
│   └── WORKFLOW_OVERVIEW.md
├── htmlcov_utils/ # HTML coverage report utilities
├── schemas/       # Static JSON/YAML schema definitions
│   ├── doc_requests_schema.yaml
│   ├── execution_dsl.json
│   └── stage_flow_schema.json
├── scripts/       # Core-specific operational and development scripts
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
├── server_prompts/ # Agent prompts and stage flow YAML configurations
│   ├── common_template.yaml
│   ├── common.yaml
│   ├── initial_status.json
│   ├── master_planner/
│   │   ├── agent_selection_prompt.txt
│   │   ├── decomposition_prompt.txt
│   │   └── sequencing_prompt.txt
│   └── stages/
│       ├── stage_minus1_goal_draft.yaml
│       ├── stage0.yaml
│       ├── stage1.yaml
│       ├── stage2.yaml
│       ├── stage3.yaml
│       ├── stage4.yaml
│       └── stage5.yaml
├── src/
│   ├── __init__.py  # Makes src a discoverable path element for some setups
│   └── chungoid/    # Main Python package for chungoid-core
│       ├── __init__.py # Package initializer
│       ├── cli.py      # Main CLI application
│       ├── constants.py
│       ├── core_utils.py
│       ├── engine.py   # Core workflow engine
│       ├── flow_executor.py # Executes defined flows
│       ├── mcp.py      # Model Context Protocol handling
│       ├── runtime/
│       │   ├── __init__.py
│       │   ├── orchestrator.py
│       │   └── agents/   # Agent implementations
│       │       ├── __init__.py
│       │       ├── agent_base.py # Base class for agents (Moved from models/)
│       │       ├── core_code_generator_agent.py
│       │       ├── core_code_integration_agent.py
│       │       ├── core_stage_executor.py
│       │       ├── core_test_generator_agent.py
│       │       ├── master_planner_agent.py
│       │       ├── mocks/    # Mock agents for testing
│       │       │   ├── mock_code_generator_agent.py
│       │       │   ├── mock_human_input_agent.py
│       │       │   ├── mock_system_requirements_gathering_agent.py
│       │       │   ├── mock_test_generation_agent.py
│       │       │   ├── mock_test_generator_agent.py # (Note similar name to above)
│       │       │   └── testing_mock_agents.py
│       │       ├── system_file_system_agent.py
│       │       ├── system_master_planner_agent.py
│       │       ├── system_master_planner_reviewer_agent.py
│       │       └── system_test_runner_agent.py
│       ├── schemas/    # Pydantic models for data structures
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
│       │   ├── agent_test_generation.py # (Note similar name to agent_test_generator.py below. Review for clarity/redundancy.)
│       │   ├── agent_test_generator.py # (Note similar name to agent_test_generation.py above. Review for clarity/redundancy.)
│       │   ├── common.py
│       │   ├── common_enums.py
│       │   ├── errors.py
│       │   ├── flows.py
│       │   ├── master_flow.py # Defines MasterExecutionPlan etc.
│       │   ├── metrics.py
│       │   └── user_goal_schemas.py
│       └── utils/
│           ├── __init__.py
│           ├── a2a_utils.py
│           ├── agent_registry.py
│           ├── agent_registry_meta.py
│           ├── agent_resolver.py
│           ├── analysis_utils.py
│           ├── chroma_client_factory.py
│           ├── chroma_utils.py
│           ├── config_loader.py
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
│   ├── .chungoid/ # Test-specific chungoid data
│   ├── __init__.py
│   ├── conftest.py # Pytest fixtures and hooks
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_cli_flow_resume.py
│   ├── legacy/
│   │   └── __init__.py # (Likely empty or for old tests)
│   ├── test_doc_manifest.py
│   ├── test_integration.py # (Older top-level integration tests?)
│   ├── test_mcp_invoke.py
│   ├── test_mcp_metadata.py
│   ├── test_no_rogue_modules.py
│   ├── test_projects/ # Test project configurations/setups
│   │   ├── __init__.py
│   │   └── test_integration_project/ # Example project for integration tests
│   └── unit/       # Unit tests for individual modules/classes
│       ├── .chungoid/ # Unit-test specific chungoid data
│       ├── __init__.py
│       ├── runtime/ # Unit tests for runtime components (placeholder, needs population or this dir is for test runtime setup)
│       ├── test_a2a_dev_cli.py
│       ├── test_a2a_utils.py
│       ├── test_agent_registry.py
│       ├── test_agent_resolver.py
│       ├── test_analysis_utils.py
│       ├── test_chroma_modes.py
│       ├── test_chroma_utils.py
│       ├── test_chroma_utils_live.py
│       ├── test_cli_utils.py
│       ├── test_cli_utils.py.bak # (DELETED)
│       ├── test_config_loader_live.py
│       ├── test_core_code_generator_agent.py
│       ├── test_core_mcp_client.py
│       ├── test_core_snapshot_dryrun.py
│       ├── test_core_stage_executor_agent.py
│       ├── test_core_test_generator_agent.py
│       ├── test_doc_requests_schema.py
│       ├── test_execution_dsl_validator.py
│       ├── test_execution_runtime.py
│       ├── test_feedback_store.py
│       ├── test_flow_api_endpoint.py
│       ├── test_flow_registry.py
│       ├── test_flow_run_cli.py
│       ├── test_logger_setup_live.py
│       ├── test_master_flow_registry.py
│       ├── test_mcp_run_endpoint.py
│       ├── test_metrics_store.py
│       ├── test_orchestrator.py
│       ├── test_orchestrator_reviewer_integration.py
│       ├── test_prompt_manager.py
│       ├── test_prompt_manager_core.py
│       ├── test_prompt_manager_live.py
│       ├── test_reflection_api.py
│       ├── test_reflection_store.py
│       ├── test_registry_dispatch_live.py
│       ├── test_security.py
│       ├── test_security_live.py
│       ├── test_snapshot_tarball.py
│       ├── test_smoke_coverage.py
│       ├── test_stage_flow_schema.py
│       ├── test_stage_minus1_prompt.py
│       ├── test_stage_minus1_prompt_schema.py
│       ├── test_stage_prompts_sequential.py
│       ├── test_state_manager.py
│       ├── test_state_manager_core.py
│       ├── test_state_manager_live.py
│       ├── test_system_master_planner_agent.py
│       ├── test_template_helpers_live.py
│       ├── test_tool_adapters.py
│       └── test_validate_planning.py
├── .coverage      # Coverage data file
├── .gitignore
├── CHANGELOG.md   # (Assumed to exist)
├── CONTRIBUTING.md
├── LICENSE
├── Makefile
├── README.md
├── chungoidmcp.py # (MCP server entry point?)
├── config.yaml    # Main configuration
├── launch_server.sh # Script to launch the server
├── pyproject.toml # Project build/dependency configuration
└── requirements.txt # Pinned dependencies
``` 