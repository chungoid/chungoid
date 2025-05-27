# Chungoid

**An autonomous AI development system that uses specialized agents and MCP tools to build complete software projects.**

Chungoid is a sophisticated autonomous development platform that transforms natural language goals into production-ready software through coordinated AI agents. The system uses the Model Context Protocol (MCP) to enable agents to work with over 45 specialized tools, from code generation to testing and documentation.

## How It Works

Chungoid operates through an orchestrated workflow of specialized agents:

1. **Goal Analysis** - Takes your natural language goal and creates detailed requirements
2. **Architecture Design** - Generates project blueprints and execution plans  
3. **Environment Setup** - Bootstraps development environments and dependencies
4. **Code Generation** - Creates production-ready code using context-aware agents
5. **Testing & Validation** - Runs comprehensive testing and quality assurance
6. **Documentation** - Generates complete project documentation

## Quick Start

### Prerequisites

- Python 3.11+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/chungoid/chungoid.git
cd chungoid

# Install dependencies
pip install -r requirements.txt
```

```bash
### Basic Usage

# Initialize configuration
mkdir -p yourproject
cd yourproject
chungoid init .

# Edit configs (see: docs/source/examples/config.yaml & docs/source/reference/config.md)
mv config.yaml yourproject/.chungoid/

# Write a simple goal
echo "i want to use flask/django to build a website" >> goal.txt

# Talk to Chungoid for goal refinement
chungoid discuss .

# Begin the build
chungoid build --goal-file goal.txt --project-dir .
```

## Core Components

### Autonomous Engine Agents

Specialized agents that handle different aspects of autonomous development:

- **ProductAnalystAgent** - Analyzes goals and generates detailed requirements (LOPRD)
- **ArchitectAgent** - Creates system architecture and technical blueprints
- **EnvironmentBootstrapAgent** - Sets up development environments and dependencies
- **DependencyManagementAgent** - Manages package dependencies and conflict resolution
- **SmartCodeGeneratorAgent** - Generates production-ready code with context awareness
- **ProjectDocumentationAgent** - Creates comprehensive project documentation
- **ProactiveRiskAssessorAgent** - Identifies and assesses project risks
- **BlueprintReviewerAgent** - Reviews and validates architectural designs
- **AutomatedRefinementCoordinatorAgent** - Orchestrates quality assurance and refinement
- **RequirementsTracerAgent** - Ensures requirements traceability across artifacts
- **CodeDebuggingAgent** - Analyzes and fixes code issues

### MCP Tool Ecosystem

Over 65+ specialized tools organized into categories:
- **File System Tools** - Project scanning, file operations, code analysis
- **ChromaDB Tools** - Vector storage, semantic search, artifact management
- **Analysis Tools** - Dependency analysis, code quality, security scanning
- **Testing Tools** - Test execution, coverage analysis, validation

### Orchestration Engine

- **UnifiedOrchestrator** - Modern orchestrator using single-path execution model
- **StateManager** - Manages execution state, persistence, and pause/resume capabilities
- **UnifiedAgentResolver** - Handles agent resolution and lifecycle management
- **MetricsStore** - Tracks performance metrics and execution events

## Project Structure

```
chungoid/
├── src/chungoid/           # Core Python package
│   ├── agents/             # Autonomous agent implementations
│   ├── mcp_tools/          # MCP tool suite (45+ tools)
│   ├── runtime/            # Orchestration and execution engine
│   ├── schemas/            # Data models and validation
│   └── utils/              # Utility modules and helpers
├── docs/                   # Documentation (this documentation)
├── tests/                  # Comprehensive test suite
└── scripts/                # Development and deployment scripts
```

## Key Features

### Autonomous Operation
- Agents work independently until tasks are complete
- Self-correction and iterative refinement
- Minimal human intervention required

### Protocol-Driven Architecture
- 17+ specialized protocols guide agent behavior
- Consistent communication patterns
- Extensible and maintainable design

### Tool Composition
- MCP-powered tool ecosystem
- Context-aware tool selection
- Seamless integration across workflows

### Learning & Adaptation
- ChromaDB-powered knowledge management
- Historical execution analysis
- Continuous improvement from past projects

## Configuration

Chungoid uses a YAML-based configuration system located in `.chungoid/config.yaml`:

```yaml
llm:
  provider: "openai"  # or anthropic, ollama, etc.
  default_model: "gpt-4o-mini-2024-07-18"

chroma:
  persist_directory: ".chungoid/chroma_db"

agents:
  timeout: 300
  max_retries: 3
```

## Advanced Usage

### Custom Workflows

Create custom execution flows using YAML:

```yaml
name: "Custom Development Flow"
version: "1.0"
start_stage: "analyze"

stages:
  analyze:
    agent_id: "ProductAnalystAgent_v1"
    next: "design"
  
  design:
    agent_id: "BlueprintGeneratorAgent_v1"
    next: "implement"
  
  implement:
    agent_id: "SmartCodeGeneratorAgent_v1"
    success_criteria: "tests_pass"
```

### Human-in-the-Loop

Pause execution for human review:

```bash
# Review outputs at major milestones
chungoid build --goal-file goal.txt --pause-after blueprint

# Resume execution after review
chungoid resume --project-dir ./my-project
```

## Documentation

- **[Getting Started](getting-started.md)** - Detailed setup and first project
- **[User Guide](user-guide.md)** - Comprehensive usage instructions
- **[Design Documentation](design/)** - System architecture and design decisions
- **[Examples](examples/)** - Sample projects and workflows
- **[API Reference](reference/)** - Complete API documentation

### Contributing
See our [contribution guidelines](CONTRIBUTING.md) for details on development setup and submission process.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/chungoid/chungoid/issues)
- **Discussions**: [GitHub Discussions](https://github.com/chungoid/chungoid/discussions)