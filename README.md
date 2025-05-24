# Chungoid

**An autonomous AI development toolkit that orchestrates intelligent workflows through specialized agents and sophisticated CLI commands.**

[![Tests](https://github.com/chungoid/metachungoid/actions/workflows/test.yml/badge.svg?branch=master)](https://github.com/chungoid/metachungoid/actions/workflows/test.yml)
![Coverage](https://img.shields.io/badge/coverage-80%25%2B-brightgreen)
![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

## What is Chungoid?

Chungoid is a sophisticated **autonomous development toolkit** that transforms high-level project goals into working software through intelligent orchestration of specialized AI agents. It operates primarily through a powerful CLI operations.

**Core Components:**
- **CLI**: Comprehensive command-line tools for project management and workflow execution
- **AI Agents**: Specialized agents for planning, coding, testing, and architecture tasks
- **Workflow Orchestration**: Master flow system for complex multi-stage development processes
- **ChromaDB Integration**: Persistent memory and learning from past executions
- **MCP Tools**: Optional Model Context Protocol integration for IDE usage

---

## Installation

### Prerequisites

- **Python 3.11+**
- **Git** for cloning the repository

### Install via pipx (Recommended)

```bash
# Clone the repository
git clone https://github.com/chungoid/chungoid.git
cd chungoid/

# Install globally with pipx
pipx install .

# Verify installation
chungoid --help
chungoid-server --version
```

### Install via pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\\Scripts\\activate  # Windows

# Install package
pip install -e .
```

---

## Quick Start

### 1. Initialize a Project

```bash
# Create and initialize a new project
mkdir my-project
cd my-project
chungoid init .

# Or initialize an existing directory
chungoid init /path/to/existing/project
```

### 2. Check Project Status

```bash
# Basic status
chungoid status

# JSON output for scripting
chungoid status --json

# Check specific project directory
chungoid status /path/to/project --json
```

### 3. Build from a Goal

```bash
# Create a goal file
echo \"Build a REST API for task management with authentication\" > goal.txt

# Build the project
chungoid build --goal-file <filename> --project-dir <project-path>

# Build with additional context
chungoid build --goal-file goal.txt --project-dir . --initial-context '{\"language\": \"python\", \"framework\": \"fastapi\"}'
```

### 4. Run Master Flows

```bash
# Run from a goal description
chungoid flow run --goal \"Create a web scraping tool\" --project-dir .

# Run a specific master flow
chungoid flow run --master-flow-id my-flow-123 --project-dir .

# Run from YAML file with custom settings
chungoid flow run --flow-yaml ./workflows/api-dev.yaml --project-dir . --llm-provider anthropic --llm-model claude-3-5-sonnet-20241022
```

### 5. Resume Interrupted Flows

```bash
# Resume a paused flow
chungoid flow resume abc-123-def --action retry --project-dir .

# Resume with additional inputs
chungoid flow resume abc-123-def --action retry_with_inputs --inputs '{\"new_requirement\": \"add logging\"}' --project-dir .

# Skip problematic stage
chungoid flow resume abc-123-def --action skip_stage --project-dir .
```

---

## Command Reference

#### Build Command

```bash
chungoid build --goal-file <path> [OPTIONS]

Options:
  --goal-file PATH        Path to file containing the user goal (required)
  --project-dir PATH      Target project directory (default: current directory)
  --run-id TEXT          Custom run ID for this execution
  --initial-context TEXT JSON string with initial context variables
  --tags TEXT            Comma-separated tags (e.g., 'dev,release')
```

#### Flow Commands

```bash
# Run flows
chungoid flow run [OPTIONS]

Options:
  --master-flow-id TEXT   ID of master flow to run
  --flow-yaml PATH        Path to specific master flow YAML file
  --goal TEXT             High-level user goal (generates new flow)
  --project-dir PATH      Project directory (default: current directory)
  --initial-context TEXT JSON string with initial context variables
  --run-id TEXT          Custom run ID
  --tags TEXT            Comma-separated tags
  --llm-provider TEXT    Override LLM provider
  --llm-model TEXT       Override LLM model
  --llm-api-key TEXT     Override LLM API key
  --llm-base-url TEXT    Override LLM base URL

# Resume flows
chungoid flow resume <run_id> --action <action> [OPTIONS]

Actions: retry, retry_with_inputs, skip_stage, force_branch, abort, provide_clarification

Options:
  --project-dir PATH      Project directory (default: current directory)
  --inputs TEXT          JSON string with inputs to merge
  --target-stage TEXT    Stage ID to jump to (for force_branch)
  --llm-provider TEXT    Override LLM provider
  --llm-model TEXT       Override LLM model
  --llm-api-key TEXT     Override LLM API key
  --llm-base-url TEXT    Override LLM base URL
```

#### Project Review

```bash
chungoid project review --cycle-id <id> --reviewer-id <id> --comments <text> --decision <decision> [OPTIONS]

Options:
  --project-dir PATH              Project directory (default: current directory)
  --cycle-id TEXT                Cycle ID being reviewed (required)
  --reviewer-id TEXT             Reviewer identifier (required)
  --comments TEXT                Review comments (required)
  --decision [approved|rejected] Review decision (required)
  --next-objective TEXT          Next cycle objective
  --linked-feedback-doc-id TEXT  ChromaDB document ID for detailed feedback
```

#### Utility Commands

```bash
# Show configuration
chungoid utils show-config [--project-dir PATH] [--raw]

# Show available modules
chungoid utils show-modules
```

### Global Options

```bash
chungoid [--log-level LEVEL] <command>

Log Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
```

---

## Configuration

### Environment Variables

```bash
# Core configuration
export CHUNGOID_PROJECT_DIR=\"/path/to/project\"
export CHUNGOID_LOG_LEVEL=\"INFO\"

# LLM Provider configuration
export ANTHROPIC_API_KEY=\"your-api-key\"
export OPENAI_API_KEY=\"your-api-key\"

# ChromaDB configuration  
export CHROMA_CLIENT_TYPE=\"persistent\"
export CHROMA_DB_PATH=\"./.chungoid/chroma_db\"
```

### Project Configuration

Create `.chungoid/chungoid_config.yaml` in your project:

```yaml
# Project identification
project_id: \"unique-project-id\"

# LLM settings
project_settings:
  llm_config:
    provider: \"anthropic\"
    model: \"claude-3-5-sonnet-20241022\"
    api_key: \"${ANTHROPIC_API_KEY}\"

# Agent settings
agents:
  default_timeout: 300
  max_retries: 3
  
# Workflow settings
workflow:
  auto_stage_progression: false
  enable_reflection: true
```

### Master Flow Configuration

Master flows define complex multi-stage workflows:

```yaml
# Example: .chungoid/master_flows/api-development.yaml
name: \"REST API Development Flow\"
description: \"Complete API development with testing\"
start_stage: \"planning\"

stages:
  planning:
    agent_id: \"master_planner_agent\"
    inputs:
      project_type: \"api\"
      requirements: \"${user_goal}\"
    next: \"architecture\"
    
  architecture:
    agent_id: \"architect_agent_v1\"
    inputs:
      design_requirements: \"${planning.output}\"
    next: \"implementation\"
    
  implementation:
    agent_id: \"code_generator_agent\"
    dependencies: [\"planning\", \"architecture\"]
    next: \"testing\"
    
  testing:
    agent_id: \"test_generator_agent\"
    inputs:
      code_base: \"${implementation.output}\"
    next: null
```

---

## Architecture

### System Components

```
┌─────────────────┐    ┌──────────────┐    ┌─────────────────┐
│   CLI Commands  │────│ AsyncOrchest │────│   AI Agents     │
│                 │    │    rator     │    │                 │
│• chungoid build │    │              │    │• MasterPlanner  │
│• chungoid flow  │    │              │    │• CodeGenerator  │
│• chungoid init  │    │              │    │• TestGenerator  │
└─────────────────┘    └──────────────┘    │• ArchitectAgent │
                                           └─────────────────┘
                              │                       │
                    ┌─────────▼─────────┐   ┌─────────▼─────────┐
                    │  State Manager   │   │   Tool Manifest   │
                    │                   │   │                   │
                    │• Project Status   │   │• 45+ MCP Tools    │
                    │• Workflow State   │   │• Performance      │
                    │• Reflection Data  │   │  Analytics        │
                    └───────────────────┘   └───────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │    ChromaDB       │
                    │                   │
                    │• Execution History│
                    │• Code Context     │
                    │• Learning Data    │
                    └───────────────────┘
```

### Agent System

**Production Agents:**
- **MasterPlannerAgent** - High-level project planning and workflow orchestration
- **CodeGeneratorAgent** - Context-aware code generation with best practices
- **TestGeneratorAgent** - Comprehensive test suite generation
- **ArchitectAgent** - System architecture design and validation
- **ProjectChromaManagerAgent** - Knowledge management and retrieval

**Autonomous Engine Agents:**
- **EnvironmentBootstrapAgent** - Multi-language environment setup
- **DependencyManagementAgent** - Smart dependency analysis and management
- **TestFailureAnalysisAgent** - Intelligent test failure diagnosis and resolution

### Tool Ecosystem

**45+ MCP Tools across categories:**
- **ChromaDB Tools (17)**: Vector operations, semantic search, knowledge storage
- **File System Tools (12)**: Project-aware file operations, template expansion
- **Terminal Tools (8)**: Secure command execution, environment management
- **Content Tools (8)**: Dynamic generation, web fetching, caching

---

## Advanced Usage

### Custom Goal Files

```json
{
  \"goal\": \"Build a microservice for user authentication\",
  \"requirements\": [
    \"JWT token-based authentication\",
    \"PostgreSQL database integration\",
    \"RESTful API endpoints\",
    \"Comprehensive test coverage\"
  ],
  \"constraints\": {
    \"language\": \"python\",
    \"framework\": \"fastapi\",
    \"database\": \"postgresql\"
  },
  \"success_criteria\": [
    \"All tests pass\",
    \"API documentation generated\",
    \"Security best practices implemented\"
  ]
}
```

### Workflow Automation

```bash
#!/bin/bash
# Automated development pipeline

PROJECT_DIR=\"./my-microservice\"
GOAL_FILE=\"./goals/auth-service.json\"

# Initialize project
chungoid init \"$PROJECT_DIR\"

# Run development flow
chungoid build --goal-file \"$GOAL_FILE\" --project-dir \"$PROJECT_DIR\" --tags \"automated,production\"

# Check results
chungoid status \"$PROJECT_DIR\" --json | jq '.current_stage'
```

### Integration with CI/CD

```yaml
# .github/workflows/chungoid-build.yml
name: Chungoid Autonomous Build
on: 
  push:
    paths: ['goals/*.json']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      
      - name: Install Chungoid
        run: |
          pip install -e ./chungoid-core
          
      - name: Run Build
        env:
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
        run: |
          chungoid build --goal-file goals/feature.json --project-dir ./output
          
      - name: Upload Results
        uses: actions/upload-artifact@v3
        with:
          name: generated-code
          path: ./output
```

---

## Optional: MCP Server Mode

While Chungoid is primarily a CLI tool, it can also operate as an MCP server for IDE integration:

### MCP Integration (Optional)

Add to your Cursor `mcp.json` settings:

```json
{
  \"mcpServers\": {
    \"chungoid\": {
      \"command\": \"chungoid-server\",
      \"transportType\": \"stdio\",
      \"args\": [],
      \"env\": {
        \"CHUNGOID_PROJECT_DIR\": \"${workspaceFolder}\",
        \"CHUNGOID_LOG_LEVEL\": \"INFO\"
      }
    }
  }
}
```

**MCP Commands:**
```
@chungoid initialize_project
@chungoid get_project_status
@chungoid prepare_next_stage
@chungoid submit_stage_artifacts
```

---

## Troubleshooting

### Common Issues

**Command not found: chungoid**
```bash
# Ensure installation completed
pipx list | grep chungoid

# Reinstall if necessary
pipx reinstall chungoid-mcp-server
```

**Project initialization fails**
```bash
# Check permissions
ls -la .chungoid/

# Reinitialize with debug logging
chungoid --log-level DEBUG init .
```

**Flow execution errors**
```bash
# Check project status
chungoid status --json

# Resume with different action
chungoid flow resume <run-id> --action skip_stage --project-dir .
```

**ChromaDB connection issues**
```bash
# Clear ChromaDB cache
rm -rf .chungoid/chroma_db

# Reinitialize project
chungoid init .
```

### Debug Commands

```bash
# Show detailed configuration
chungoid utils show-config --raw

# Check available modules
chungoid utils show-modules

# Enable debug logging
export CHUNGOID_LOG_LEVEL=DEBUG
chungoid flow run --goal \"test project\" --project-dir .
```

---

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/your-org/chungoid-mcp.git
cd chungoid-mcp/chungoid-core

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with development dependencies
pip install -e \".[dev,test]\"

# Run tests
pytest
```

### Project Structure

```
chungoid-core/
├── src/chungoid/
│   ├── agents/                 # AI agent implementations
│   ├── mcp_tools/             # MCP tool suite (45+ tools)
│   ├── runtime/               # Workflow orchestration
│   ├── schemas/               # Data models and validation
│   ├── utils/                 # Core utilities and services
│   ├── cli.py                 # Main CLI interface
│   ├── engine.py              # Core ChungoidEngine
│   └── mcp.py                 # MCP server entry point
├── tests/                     # Test suites
├── docs/                      # Documentation
└── pyproject.toml            # Package configuration
```

---

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/your-username/chungoid-mcp.git
cd chungoid-mcp/chungoid-core

# Set up development environment
python -m venv .venv
source .venv/bin/activate
pip install -e \".[dev,test]\"

# Run tests
pytest

# Submit changes
git checkout -b feature/amazing-feature
# ... make changes ...
git commit -m \"Add amazing feature\"
git push origin feature/amazing-feature
# Create Pull Request
```

---

## License

Licensed under the **GNU Affero General Public License v3.0** (AGPL-3.0).

See [LICENSE](LICENSE) for full details.

---

## Acknowledgments

Built with:
- **[ChromaDB](https://www.trychroma.com/)** - Vector database for semantic memory
- **[Click](https://click.palletsprojects.com/)** - Command-line interface framework
- **[FastMCP](https://github.com/jlowin/fastmcp)** - MCP server framework
- **[Pydantic](https://pydantic.dev/)** - Data validation and serialization

---

<div align=\"center\">

**Transform your development workflow with autonomous AI-powered tools**

[Get Started](#installation) • [Documentation](docs/) • [GitHub](https://github.com/your-org/chungoid-mcp)

</div>